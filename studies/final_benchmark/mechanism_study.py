"""Mechanism-driven IA-OSR synthetic experiments for NOMS, v1.

Reproducible paired cost comparisons, parameter-defined structural families,
exact small-N Bellman, PSA1 rollout and baseline policies.
This uses synthetic, model-consistent data, never Alibaba traces.
"""
from __future__ import annotations
import argparse,csv,json,math,random,statistics,time
from functools import lru_cache
from pathlib import Path
import numpy as np
from ia_osr_experiment import Params,Solver,make_random
from ia_osr_psa_audit import PSASolver,clear_all

FAMILIES=('iid','p4_probe','p3_screen','deferred','skip_boundary')


def param_record(p):
    return dict(p=list(p.p),q=list(p.q),r=list(p.r),alpha=list(p.alpha),beta=list(p.beta),eps=p.eps)


def iid(n,rng):
    return make_random(n,rng)


def gen_p4(rng):
    """2-domain pure ordering-information probe, constructed with both strict P4 margins."""
    pi=rng.uniform(.25,.7); pj=rng.uniform(.25,.75)
    ai=rng.uniform(.76,.94);bi=rng.uniform(.77,.97)
    qi=10**rng.uniform(.35,1.05)
    gi=pi*ai+(1-pi)*(1-bi);hi=1-gi
    Ji_plus=qi*gi/(pi*ai);Ji_minus=qi*hi/(pi*(1-ai))
    frac=rng.uniform(.30,.70)
    Jj=Ji_plus**(1-frac)*Ji_minus**frac
    qj=pj*Jj
    m1=pj*hi*qi-pi*(1-ai)*qj
    m2=pi*ai*qj-pj*gi*qi
    assert m1>0 and m2>0
    ri=min(m1,m2)*rng.uniform(.08,.75)
    aj=rng.uniform(.75,.96);bj=rng.uniform(.75,.96)
    # Rj cost > Qj, so j's own refresh is never an attractive one-step savings action.
    rj=2.0*qj
    e=.005
    assert pi*(1-ai)/hi>e
    return Params((pi,pj),(qi,qj),(ri,rj),(ai,aj),(bi,bj),e)


def gen_p3(rng):
    """2-domain screening-favorable first-module region, with admissible negative skip."""
    for _ in range(10000):
        pi=rng.uniform(.14,.38);pj=rng.uniform(.25,.68)
        ai=rng.uniform(.93,.99);bi=rng.uniform(.88,.98)
        qi=10**rng.uniform(.4,1.25); qj=10**rng.uniform(.35,1.05)
        gi=pi*ai+(1-pi)*(1-bi);hi=1-gi
        pim=pi*(1-ai)/hi
        eps=rng.uniform(max(.025,pim+.007),.09)
        if pim>=eps or pi<=eps:continue
        # construct q_i in the algebraic admissible interval if possible
        ri=rng.uniform(.025,.20)*qj
        L=(ri+pi*(1-ai)*qj)/hi
        U=(pi*ai*qj/pj-ri)/gi
        if U<=L*1.15:continue
        qi=math.sqrt(L*U)
        # Only i refresh is useful in this intended 2-domain fixed-module diagnostic
        rj=2*qj
        return Params((pi,pj),(qi,qj),(ri,rj),(ai,.85),(bi,.9),eps)
    raise RuntimeError('P3 generator rejection limit')


def gen_deferred(rng):
    """2 domains; inexpensive j queried first, costly i screened after j failure."""
    pi=rng.uniform(.18,.45); pj=rng.uniform(.5,.8)
    ai=rng.uniform(.95,.995); bi=rng.uniform(.90,.98)
    qi=rng.uniform(14,27);qj=rng.uniform(2.5,6.0)
    ri=rng.uniform(.15,.6)
    rj=2*qj
    eps=rng.uniform(.04,.095)
    # i negative is admissible, i positive etc; j is query-only.
    hi=1-(pi*ai+(1-pi)*(1-bi))
    if pi*(1-ai)/hi>eps:eps=.12
    return Params((pi,pj),(qi,qj),(ri,rj),(ai,.85),(bi,.85),eps)


def gen_skip_boundary(rng):
    """A paired epsilon experiment: two direct SKIPs are just infeasible/feasible."""
    target=rng.uniform(.06,.19)
    p0=rng.uniform(.015,.035)
    p1=(target-p0)/(1-p0)
    p2=rng.uniform(.25,.65);p3=rng.uniform(.22,.6)
    ps=(p0,p1,p2,p3)
    qs=(rng.uniform(13,25),rng.uniform(12,24),rng.uniform(2,6),rng.uniform(3,8))
    rs=tuple(q*rng.uniform(.045,.17) for q in qs)
    aa=tuple(rng.uniform(.8,.97) for _ in ps)
    bb=tuple(rng.uniform(.82,.96) for _ in ps)
    delta=min(.003,target*.015)
    low=Params(ps,qs,rs,aa,bb,target-delta)
    high=Params(ps,qs,rs,aa,bb,target+delta)
    return low,high,target


def get_policy_cost(sol,policy):
    @lru_cache(None)
    def recur(s):
        if sol.stop(s):return 0.
        a=policy(s)
        c,br=sol.successors(s,a)
        return c+sum(pr*recur(ns) for pr,ns in br if pr>0)
    return recur(sol.initial)


def greedy_skip(sol):
    def act(s):
        skips=[a for a in sol.actions(s) if a[0]=='SKIP']
        if skips:
            return max(skips,key=lambda a:(sol.a.q[a[1]]/max(sol.belief(a[1],s[a[1]]),1e-10),-a[1]))
        arr=sol.unfinished(s)
        i=min(arr,key=lambda j:(sol.a.q[j]/max(sol.belief(j,s[j]),1e-10),j))
        return ('Q',i)
    return act


def optimal_no_refresh(sol):
    """Exact best QUERY+SKIP policy under the same epsilon, with REFRESH disabled.

    This is a stronger risk-aware baseline than greedy SKIP and isolates
    attainable information-acquisition value in the small-N model.
    """
    @lru_cache(None)
    def rec(s):
        if sol.stop(s):return 0.
        values=[]
        for act in sol.actions(s):
            if act[0]=='R':continue
            c,branches=sol.successors(s,act)
            values.append(c+sum(pr*rec(ns) for pr,ns in branches if pr>0))
        return min(values)
    return rec(sol.initial)


def cost_actions(sol,which='PSA1'):
    if which=='PSA1':policy=sol.psa_fast_action
    elif which=='Exact':policy=lambda s:min(sol.actions(s),key=lambda a:(round(sol.q_exact(s,a),12),{'STOP':0,'SKIP':1,'Q':2,'R':3}[a[0]],a[1]))
    elif which=='GreedySkip':policy=greedy_skip(sol)
    else:raise ValueError(which)
    @lru_cache(None)
    def rec(s):
        if sol.stop(s):return (0.,0.,0.,0.)
        act=policy(s)
        c,branches=sol.successors(s,act)
        out=[c,float(act[0]=='Q'),float(act[0]=='R'),float(act[0]=='SKIP')]
        for pr,ns in branches:
            if pr<=0:continue
            val=rec(ns)
            for k in range(4):out[k]+=pr*val[k]
        return tuple(out)
    return rec(sol.initial),policy(sol.initial)


def exact_true_miss(sol,ptrue):
    """Evaluate PSA1 policy under a shifted independent-domain true prior.

    Policy and stopping use nominal p_i; Q and R outcome branches use true priors,
    true Bayes posterior after +/- ; 'false miss' at failure terminal means an
    undiscovered feasible domain remains, averaged conditional on the observation history.
    """
    a=sol.a
    gt=tuple(ptrue[i]*a.alpha[i]+(1-ptrue[i])*(1-a.beta[i]) for i in range(sol.n))
    ht=tuple(1-x for x in gt)
    ptplus=tuple(ptrue[i]*a.alpha[i]/gt[i] for i in range(sol.n))
    ptminus=tuple(ptrue[i]*(1-a.alpha[i])/ht[i] for i in range(sol.n))
    def tb(i,k):
        return (ptrue[i] if k in (0,4) else ptplus[i] if k in (1,5) else ptminus[i] if k in (2,6) else 0.)
    @lru_cache(None)
    def rec(s):
        if sol.stop(s):
            missing=1-math.prod(1-tb(i,k) for i,k in enumerate(s) if k!=3)
            return 0.,1.,missing
        act=sol.psa_fast_action(s)
        t,i=act
        if t=='SKIP':
            return rec(sol.set(s,i,s[i]+4))
        if t=='Q':
            child=rec(sol.set(s,i,3));fail=1-tb(i,s[i])
            return a.q[i]+fail*child[0],fail*child[1],fail*child[2]
        if t=='R':
            yp=rec(sol.set(s,i,1));ym=rec(sol.set(s,i,2))
            return (a.r[i]+gt[i]*yp[0]+ht[i]*ym[0],
                    gt[i]*yp[1]+ht[i]*ym[1],
                    gt[i]*yp[2]+ht[i]*ym[2])
        raise ValueError(act)
    c,prob_failure,prob_false_miss=rec(sol.initial)
    return dict(cost_true=c,prob_failure=prob_failure,
                prob_false_miss=prob_false_miss,
                risk_at_failure=prob_false_miss/prob_failure if prob_failure>1e-14 else math.nan)


def logit_shift(p,delta):
    v=math.log(p/(1-p))+delta
    return 1/(1+math.exp(-v))


def p4_diagnostic(par):
    pi,pj=par.p; qi,qj=par.q;ri=par.r[0];ai=par.alpha[0];bi=par.beta[0]
    gi=pi*ai+(1-pi)*(1-bi);hi=1-gi
    pim=pi*(1-ai)/hi;pip=pi*ai/gi
    cprobe=ri+(1-pj*hi)*qi+(1-pi*ai)*qj
    cqi=qi+(1-pi)*qj
    cqj=qj+(1-pj)*qi
    return dict(p4_cross=(qi/pip<qj/pj<qi/pim),
                p4_marg_qiqj=cqi-cprobe,p4_marg_qjqi=cqj-cprobe,
                p4_fixed_saving=min(cqi,cqj)-cprobe,
                p4_all_both_negative_unskippable=(pim>par.eps))


def p3_diagnostic(par):
    pi,pj=par.p;qi,qj=par.q;ri=par.r[0];ai=par.alpha[0];bi=par.beta[0]
    gi=pi*ai+(1-pi)*(1-bi);hi=1-gi
    L=(ri+pi*(1-ai)*qj)/hi;U=(pi*ai*qj/pj-ri)/gi
    screen=ri+gi*qi+(1-pi*ai)*qj
    qi_first=qi+(1-pi)*qj
    qj_first=qj+(1-pj)*qi
    return dict(p3_L=L,p3_U=U,p3_in_interval=L<qi<U,
                p3_saving=min(qi_first,qj_first)-screen,
                p3_negative_skippable=(pi*(1-ai)/hi<=par.eps))


def measure_case(par,family,seed,index,kind='',extra=None,compute_sc=True):
    sol=PSASolver(par);s=sol.initial
    opt=sol.exact(s)
    h1=sol.rollout_cost(s,'H1')
    psa=sol.psa_fast_cost(s)
    sc=sol.rollout_cost(s,'H1SC') if compute_sc else float('nan')
    base=sol.HPSA1fast(s)
    qonly=sol.HQ(s)
    greedy=get_policy_cost(sol,greedy_skip(sol))
    no_refresh=optimal_no_refresh(sol)
    (psac,psanq,psanr,psans),pa=cost_actions(sol,'PSA1')
    assert abs(psac-psa)<1e-9
    (excost,exnq,exnr,exns),oa=cost_actions(sol,'Exact')
    assert abs(excost-opt)<1e-9
    assert opt<=psa+1e-8 and psa<=base+1e-8 and opt<=sc+1e-8
    row=dict(family=family,seed=seed,index=index,kind=kind,N=par.n(),eps=par.eps,
             optimal=opt,query_only=qonly,greedy_skip=greedy,no_refresh_opt=no_refresh,ia_osr_h1=h1,psa1=psa,
             skip_closed=sc,psa_base=base,
             gain_psa_vs_query=100*(qonly-psa)/qonly,
             gain_psa_vs_greedy=100*(greedy-psa)/greedy if greedy>0 else float('nan'),
             gap_psa_pct=100*(psa/opt-1),gap_h1_pct=100*(h1/opt-1),
             first_psa=pa[0],first_exact=oa[0],
             first_psa_domain=pa[1],first_exact_domain=oa[1],
             expected_q_psa=psanq,expected_r_psa=psanr,expected_skip_psa=psans,
             expected_q_exact=exnq,expected_r_exact=exnr,expected_skip_exact=exns,
             parameters=json.dumps(param_record(par),separators=(',',':')))
    if extra:row.update(extra)
    if family=='p4_probe':row.update(p4_diagnostic(par))
    if family=='p3_screen':row.update(p3_diagnostic(par))
    clear_all()
    return row


def seeded_study(outdir,nseeds=5,count=120,bootstrap=1000):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for family in FAMILIES:
        t=time.perf_counter()
        for k in range(nseeds):
            seed=1009+1000*k
            rng=random.Random(seed+7919*FAMILIES.index(family))
            for i in range(count):
                if family=='iid':par=iid(4 if i%2==0 else 5,rng)
                elif family=='p4_probe':par=gen_p4(rng)
                elif family=='p3_screen':par=gen_p3(rng)
                elif family=='deferred':par=gen_deferred(rng)
                else:
                    low,high,target=gen_skip_boundary(rng)
                    for par,kind in ((low,'below'),(high,'above')):
                        rows.append(measure_case(par,family,seed,i,kind,extra={'risk_pair':target}))
                    continue
                rows.append(measure_case(par,family,seed,i))
        print(f'family {family}, generated {sum(x["family"]==family for x in rows)}, elapsed {time.perf_counter()-t:.2f}s',flush=True)
    writecsv(outdir/'cases.csv',rows)
    summ=family_summary(rows,bootstrap)
    writecsv(outdir/'family_summary.csv',summ)
    paired=pair_summary(rows,bootstrap)
    writecsv(outdir/'boundary_pair_summary.csv',paired)
    stats=mechanism_checks(rows)
    (outdir/'mechanism_checks.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
    print(json.dumps({'summary':summ,'paired':paired,'mechanisms':stats},indent=2),flush=True)
    return rows,summ,paired,stats


def writecsv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r.keys()))
    with open(path,'w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)


def bootstrap_ci(arr,fn,bootstrap=1500,seed=837):
    arr=np.asarray(arr,dtype=float);n=len(arr)
    rng=np.random.default_rng(seed); vals=np.zeros(bootstrap)
    for k in range(bootstrap):vals[k]=fn(arr[rng.integers(0,n,n)])
    return float(np.percentile(vals,2.5)),float(np.percentile(vals,97.5))


def family_summary(rows,b):
    out=[]
    for family in FAMILIES:
        groups=['all'] if family!='skip_boundary' else ['below','above']
        for kind in groups:
            sub=[x for x in rows if x['family']==family and (kind=='all' or x['kind']==kind)]
            A=np.array([[x['query_only'],x['psa1'],x['optimal'],x['greedy_skip'],x['ia_osr_h1'],x['skip_closed'],x['gap_psa_pct'],x['no_refresh_opt']] for x in sub])
            def benefit(z):return 100*(z[:,0].mean()-z[:,1].mean())/z[:,0].mean()
            ci=bootstrap_ci(A,benefit,b)
            gci=bootstrap_ci(A,lambda z:z[:,6].mean(),b)
            out.append(dict(family=family,kind=kind,N=len(sub),n_seeds=len(set(x['seed'] for x in sub)),
                            mean_query=float(A[:,0].mean()),mean_psa=float(A[:,1].mean()),mean_opt=float(A[:,2].mean()),
                            mean_greedy=float(A[:,3].mean()),mean_h1=float(A[:,4].mean()),mean_sc=float(A[:,5].mean()),
                            mean_no_refresh_opt=float(A[:,7].mean()),
                            saving_vs_no_refresh_opt_pct=float(100*(A[:,7].mean()-A[:,1].mean())/A[:,7].mean()),
                            saving_pct=benefit(A),saving_low=ci[0],saving_high=ci[1],
                            avg_psa_opt_gap_pct=float(A[:,6].mean()),gap_low=gci[0],gap_high=gci[1],
                            frac_psa_exact=float(np.mean(np.isclose(A[:,1],A[:,2],atol=1e-8))),
                            frac_first_refresh=float(np.mean([x['first_psa']=='R' for x in sub])),
                            frac_first_skip=float(np.mean([x['first_psa']=='SKIP' for x in sub]))))
    return out


def pair_summary(rows,b):
    groups={}
    for x in rows:
        if x['family']=='skip_boundary':groups.setdefault((x['seed'],x['index']),{})[x['kind']]=x
    pairs=[(a['below'],a['above']) for a in groups.values() if 'below' in a and 'above' in a]
    diffs=np.array([[lo['psa1'],hi['psa1'],lo['optimal'],hi['optimal'],lo['greedy_skip'],hi['greedy_skip']] for lo,hi in pairs])
    ci=bootstrap_ci(diffs,lambda z:np.mean(z[:,0]-z[:,1]),b)
    return [dict(n_pairs=len(pairs),mean_delta_psa_cost=float(np.mean(diffs[:,0]-diffs[:,1])),
                 delta_psa_ci_low=ci[0],delta_psa_ci_high=ci[1],
                 fraction_psa_change=float(np.mean(np.abs(diffs[:,0]-diffs[:,1])>1e-9)),
                 mean_delta_opt_cost=float(np.mean(diffs[:,2]-diffs[:,3])),
                 frac_boundary_skip_act=float(np.mean([hi['first_psa']=='SKIP' for lo,hi in pairs])))]


def mechanism_checks(rows):
    out={}
    for family in ['p4_probe','p3_screen','deferred']:
        rr=[x for x in rows if x['family']==family]
        out[family]={'count':len(rr),'PSA_refresh_first':sum(x['first_psa']=='R' for x in rr),
                     'DP_refresh_first':sum(x['first_exact']=='R' for x in rr),
                     'PSA_exact':sum(x['gap_psa_pct']<1e-7 for x in rr),
                     'PSA_cost_mean':statistics.mean(x['psa1'] for x in rr)}
    rr=[x for x in rows if x['family']=='p4_probe']
    out['p4_probe'].update({'crossing_verified':sum(x['p4_cross'] for x in rr),
                            'probe_better_both_fixed':sum(x['p4_fixed_saving']>1e-10 for x in rr),
                            'mean_fixed_saving':statistics.mean(x['p4_fixed_saving'] for x in rr)})
    rr=[x for x in rows if x['family']=='p3_screen']
    out['p3_screen'].update({'strict_interval_verified':sum(x['p3_in_interval'] for x in rr),
                             'screen_better_both_fixed':sum(x['p3_saving']>1e-10 for x in rr),
                             'mean_fixed_saving':statistics.mean(x['p3_saving'] for x in rr)})
    return out


def run_robust(outdir,nseeds=5,count=75,b=1000):
    """Perturbed true priors; decisions use beliefs from independent nominal model."""
    rows=[]
    for seed_i in range(nseeds):
        seed=3001+997*seed_i;rng=random.Random(seed)
        for idx in range(count):
            pa,_,_=gen_skip_boundary(rng)
            for shift in (0.,.5,1.,1.5):
                sol=PSASolver(pa)
                ptrue=tuple(logit_shift(p,shift) for p in pa.p)
                ev=exact_true_miss(sol,ptrue)
                row=dict(seed=seed,index=idx,delta_logodds=shift,eps=pa.eps,
                         **ev,model_psa_cost=sol.psa_fast_cost(sol.initial),
                         pnominal_mean=statistics.mean(pa.p),ptrue_mean=statistics.mean(ptrue))
                rows.append(row);clear_all()
    writecsv(Path(outdir)/'calibration_drift_cases.csv',rows)
    summ=[]
    for shift in (0.,.5,1.,1.5):
        s=[x for x in rows if x['delta_logodds']==shift]
        A=np.array([[x['prob_failure'],x['prob_false_miss'],x['cost_true'],x['model_psa_cost'],x['eps']] for x in s])
        def weight_risk(z):return z[:,1].sum()/z[:,0].sum()
        ci=bootstrap_ci(A,weight_risk,b)
        summ.append(dict(delta_logodds=shift,N=len(s),weighted_risk_given_failure=weight_risk(A),
                         risk_ci_low=ci[0],risk_ci_high=ci[1],
                         mean_eps=float(A[:,4].mean()),mean_actual_cost=float(A[:,2].mean()),
                         mean_nominal_cost=float(A[:,3].mean()),
                         frac_instance_risk_above_eps=float(np.mean([x['risk_at_failure']>x['eps']+1e-9 for x in s]))))
    writecsv(Path(outdir)/'calibration_drift_summary.csv',summ)
    print('calibration',json.dumps(summ,indent=2),flush=True)
    return rows,summ


def run_scale(outdir,ns=(10,20,40,60,80),reps=6):
    rows=[]
    for n in ns:
        for rep in range(reps):
            rng=random.Random(15000+100*n+rep)
            # Hard condition: low p / many SKIP choices and R alternatives
            ps=tuple(.018+rng.uniform(0,.008) for _ in range(n))
            qs=tuple(rng.uniform(4,12) for _ in range(n))
            rs=tuple(.2*q for q in qs)
            aa=(.95,)*n;bb=(.95,)*n
            sol=PSASolver(Params(ps,qs,rs,aa,bb,.1))
            t=time.perf_counter()
            a=sol.psa_fast_action(sol.initial)
            duration=time.perf_counter()-t
            rows.append(dict(N=n,rep=rep,first_action=a[0],first_action_time_s=duration))
            clear_all()
        print('scale',n,'median sec',statistics.median(x['first_action_time_s'] for x in rows if x['N']==n),flush=True)
    writecsv(Path(outdir)/'runtime_scaling.csv',rows)
    sm=[]
    for n in ns:
        times=[x['first_action_time_s'] for x in rows if x['N']==n]
        sm.append(dict(N=n,reps=len(times),median_s=statistics.median(times),min_s=min(times),max_s=max(times)))
    writecsv(Path(outdir)/'runtime_scaling_summary.csv',sm)
    return rows,sm

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',default='results')
    ap.add_argument('--nseeds',type=int,default=5)
    ap.add_argument('--count',type=int,default=120)
    ap.add_argument('--robust-count',type=int,default=75)
    ap.add_argument('--bootstrap',type=int,default=1200)
    ap.add_argument('--no-scale',action='store_true')
    args=ap.parse_args()
    o=Path(args.out);o.mkdir(parents=True,exist_ok=True)
    seeded_study(o,args.nseeds,args.count,args.bootstrap)
    run_robust(o,args.nseeds,args.robust_count,args.bootstrap)
    if not args.no_scale:run_scale(o)


def run_factorial(outdir,seeds=(1201,2201,3201,4201),count_per_seed=45,bootstrap=1000):
    """Controlled matched-parameter grid: identical p,q across epsilon/sensor/cost cells.
    Changing alpha/beta and refresh costs changes only information acquisition,
    not the underlying resource availability or exact query costs.
    """
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in seeds:
        rng=random.Random(seed)
        for idx in range(count_per_seed):
            # nontrivial mixed availability; near-low probabilities permit risk SKIPs.
            ps=(rng.uniform(.012,.065),rng.uniform(.035,.18),rng.uniform(.18,.56),rng.uniform(.26,.75))
            qs=tuple(10**rng.uniform(.45,1.25) for _ in range(4))
            no_r={e:optimal_no_refresh(PSASolver(Params(ps,qs,qs,(.85,)*4,(.85,)*4,e))) for e in (.03,.10,.20)}
            clear_all()
            for eps in (.03,.10,.20):
                for quality in (.70,.85,.97):
                    for ratio in (.05,.20,.50):
                        par=Params(ps,qs,tuple(q*ratio for q in qs),(quality,)*4,(quality,)*4,eps)
                        sol=PSASolver(par)
                        sp=sol.initial
                        cost=sol.psa_fast_cost(sp)
                        opt=sol.exact(sp)
                        qonly=sol.HQ(sp)
                        (cost2,nq,nr,ns),first=cost_actions(sol,'PSA1')
                        assert abs(cost-cost2)<1e-8 and opt<=cost+1e-8
                        risk=exact_true_miss(sol,ps)
                        skip_greedy=get_policy_cost(sol,greedy_skip(sol))
                        no_refresh=no_r[eps]
                        h1=sol.rollout_cost(sp,'H1')
                        assert abs(risk['cost_true']-cost)<1e-8
                        # Each reached failure STOP with calibrated beliefs has risk <= epsilon.
                        assert risk['risk_at_failure']<=eps+1e-9
                        rows.append(dict(seed=seed,index=idx,eps=eps,accuracy=quality,r_over_q=ratio,
                                         psa1=cost,optimal=opt,query_only=qonly,
                                         greedy_skip=skip_greedy,ia_osr_h1=h1,no_refresh_opt=no_refresh,
                                         first_action=first[0],expected_refresh=nr,
                                         expected_query=nq,expected_skip=ns,
                                         risk_given_failure=risk['risk_at_failure'],
                                         prob_failure=risk['prob_failure'],
                                         prob_false_miss=risk['prob_false_miss']))
                        clear_all()
    writecsv(outdir/'sensitivity_cases.csv',rows)
    summ=[]
    for eps in (.03,.10,.20):
        for quality in (.70,.85,.97):
            for ratio in (.05,.20,.50):
                ss=[x for x in rows if x['eps']==eps and x['accuracy']==quality and x['r_over_q']==ratio]
                A=np.array([[x['query_only'],x['psa1'],x['optimal'],x['expected_refresh'],x['prob_failure'],x['prob_false_miss'],x['greedy_skip'],x['ia_osr_h1'],x['no_refresh_opt']] for x in ss])
                benefit=lambda z:100*(z[:,0].mean()-z[:,1].mean())/z[:,0].mean()
                ci=bootstrap_ci(A,benefit,bootstrap)
                summ.append(dict(eps=eps,accuracy=quality,r_over_q=ratio,N=len(ss),
                                 saving_pct=benefit(A),saving_low=ci[0],saving_high=ci[1],
                                 mean_psa_cost=float(A[:,1].mean()),
                                 mean_greedy_cost=float(A[:,6].mean()),
                                 saving_vs_greedy_pct=float(100*(A[:,6].mean()-A[:,1].mean())/A[:,6].mean()),
                                 mean_no_refresh_cost=float(A[:,8].mean()),
                                 saving_vs_no_refresh_pct=float(100*(A[:,8].mean()-A[:,1].mean())/A[:,8].mean()),
                                 exact_value_of_refresh_pct=float(100*(A[:,8].mean()-A[:,2].mean())/A[:,8].mean()),
                                 mean_h1_cost=float(A[:,7].mean()),
                                 saving_vs_h1_pct=float(100*(A[:,7].mean()-A[:,1].mean())/A[:,7].mean()),
                                 mean_opt_gap_pct=float(np.mean(100*(A[:,1]/A[:,2]-1))),
                                 mean_expected_refresh=float(A[:,3].mean()),
                                 fraction_first_refresh=float(np.mean([x['first_action']=='R' for x in ss])),
                                 weighted_conditional_miss_risk=float(A[:,5].sum()/A[:,4].sum())))
    writecsv(outdir/'sensitivity_summary.csv',summ)
    print('factorial:',len(rows),'cases,',len(summ),'cells',flush=True)
    return rows,summ
