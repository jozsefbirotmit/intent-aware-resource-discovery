"""Publication-focused summaries of the unchanged final-policy replays."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'report';P=ROOT/'results/philly';A=ROOT/'results/alibaba2018'
C=['#146b80','#c77c2c','#796395']
plt.rcParams.update({'font.size':10,'axes.spines.top':True,'axes.spines.right':True,'axes.linewidth':.8,'lines.linewidth':1.15,'lines.markersize':5})

def style(ax):ax.grid(True,linewidth=.45,alpha=.45);ax.set_axisbelow(True)
def save(fig,name):
    fig.savefig(R/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(R/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def md(df,columns,formats=None):
    formats=formats or {};s=['| '+' | '.join(columns.values())+' |','| '+' | '.join('---' for _ in columns)+' |']
    for _,r in df.iterrows():
        vals=[]
        for k in columns:
            v=r[k]
            if pd.isna(v):z='n. a.'
            elif k in formats:z=formats[k].format(v)
            elif isinstance(v,(float,np.floating)):z=f'{v:.4f}'
            else:z=str(v)
            vals.append(z)
        s.append('| '+' | '.join(vals)+' |')
    return '\n'.join(s)

def comparisons(df,key):
    cost=df.pivot(index=key,columns='method',values='mean_cost');risk=df.pivot(index=key,columns='method',values='conditional_miss')
    delta=risk.psa1_s-risk.legacy_exact;changed=delta.abs()>1e-12
    return dict(configurations=len(cost),S_DP_cost_max_abs_difference=float(np.max(np.abs(cost.psa1_s-cost.legacy_exact))),
       S_DP_risk_changed=int(changed.sum()),S_DP_risk_lower=int((delta<-1e-12).sum()),S_DP_risk_higher=int((delta>1e-12).sum()),
       S_DP_risk_max_abs_percentage_points=float(100*delta.abs().max()),
       S_cheaper_than_PSA1=int((cost.psa1_s<cost.psa1-1e-9).sum()),S_more_expensive_than_PSA1=int((cost.psa1_s>cost.psa1+1e-9).sum()))

def main():
    R.mkdir(exist_ok=True)
    p=pd.read_csv(P/'policy_summary.csv');aa=pd.read_csv(A/'policy_summary.csv');a=aa[aa.scope=='per_lag_known']
    ps=p[(p.method=='psa1_s')&(p.epsilon==.1)];als=a[(a.method=='psa1_s')&(a.protocol=='nominal_10pct')]
    pw=ps.loc[ps.conditional_miss.idxmax()];aw=als.loc[als.conditional_miss.idxmax()]
    pm=json.loads((P/'replay_metadata.json').read_text());am=json.loads((A/'replay_metadata.json').read_text())
    summary=dict(philly=comparisons(p[p.epsilon==.1],'scenario'),alibaba2018=comparisons(a[a.protocol=='nominal_10pct'],'profile'))
    for name,frame,row in [('philly',ps,pw),('alibaba2018',als,aw)]:
        summary[name].update(S_positive_budget_profiles_over_10pct=int((frame.conditional_miss>.1).sum()),
            S_worst_risk=float(row.conditional_miss),S_worst_misses=int(row.miss_n),S_worst_empty_stops=int(row.empty_stop_n),
            S_worst_risk_CI=[float(row.risk_block95_low),float(row.risk_block95_high)])
    phead=p[(p.calibration=='rolling14d')&(p.lag_min==1)&(p.costs=='heterogeneous')&(p.refresh_ratio==.05)&(p.epsilon==0)&p.method.isin(['query_only','no_refresh_opt','psa1','psa1_s'])]
    ahead=a[(a.lag_s==60)&(a.costs=='heterogeneous')&(a.protocol=='zero_risk_control')&(a.intent==2)&a.method.isin(['query_only','no_refresh_opt','psa1','psa1_s'])]
    phead.to_csv(P/'headline_zero_budget.csv',index=False);ahead.to_csv(A/'headline_zero_budget_C.csv',index=False)
    ptable=phead[phead.method=='psa1_s'].copy();ptable['layout']=ptable.replicate+1
    atable=ahead[ahead.method=='psa1_s'].copy();atable['layout']=atable.layout+1
    summary['primary_zero_budget_savings']=dict(philly_2gpu=ptable[ptable.gpus_requested==2].saving_vs_no_refresh_percent.tolist(),
       philly_4gpu=ptable[ptable.gpus_requested==4].saving_vs_no_refresh_percent.tolist(),alibaba_C=atable.saving_vs_no_refresh_percent.tolist())
    original_al=a[(a.method=='psa1')&(a.protocol=='nominal_10pct')]
    summary['alibaba2018']['old_PSA1_max_risk']=float(original_al.conditional_miss.max())
    choices=json.loads((A/'calibration_choices_frozen_before_test.json').read_text())['choices']
    summary['alibaba2018']['S_tightened_calibration_budgets']=sum(r['selected_epsilon']['psa1_s']<.1 for r in choices)
    summary['alibaba2018']['S_vs_legacy_DP_calibration_budget_differences']=sum(r['selected_epsilon']['psa1_s']!=r['selected_epsilon']['legacy_exact'] for r in choices)
    (ROOT/'results/headline_summary.json').write_text(json.dumps(summary,indent=2)+'\n')

    # Zero-budget paired cost results: final S and original PSA1.
    fig,axs=plt.subplots(1,3,figsize=(11.5,3.5),layout='constrained')
    for ax,g,title in [(axs[0],phead[phead.gpus_requested==2],'Philly: 2-GPU predicate'),(axs[1],phead[phead.gpus_requested==4],'Philly: 4-GPU predicate'),(axs[2],ahead,'Alibaba: CPU≤80%, memory≤90%')]:
        layout='replicate' if 'replicate' in g.columns else 'layout'
        for name,label,color,ls in [('psa1','PSA1',C[1],'--'),('psa1_s','PSA1-S',C[0],'-')]:
            v=g[g.method==name].sort_values(layout)
            ax.plot(v[layout]+1,v.saving_vs_no_refresh_percent,marker='o',markerfacecolor='none',color=color,ls=ls,label=label)
        ax.set_xticks([1,2,3]);ax.set_xlabel('Layout');ax.set_title(title);ax.set_ylim(bottom=0);style(ax)
    axs[0].set_ylabel('Saving vs optimal no-refresh (%)');axs[1].legend(frameon=False)
    fig.suptitle('ε=0; 1-minute signal; Q=(4,2,1,3), R=0.05Q')
    save(fig,'fig1_zero_budget_final_policy')

    fig,axs=plt.subplots(1,2,figsize=(10.5,3.7),layout='constrained')
    for ax,g,title in [(axs[0],ps,'Philly: 112 frozen configurations'),(axs[1],als,'Alibaba: 54 frozen profiles')]:
        good=g.saving_vs_no_refresh_percent.notna()
        ax.scatter(g.loc[good,'saving_vs_no_refresh_percent'],100*g.loc[good,'conditional_miss'],s=25,facecolors='none',edgecolors=C[0],linewidths=.8)
        ax.axhline(10,color='#b74249',ls='--',lw=1.1,label='Model budget ε=10%')
        ax.axvline(0,color='grey',lw=.6);ax.set_xlabel('Saving vs same-budget optimal no-refresh (%)');ax.set_title(title);style(ax)
        if (~good).any():ax.text(.03,.94,f'{int((~good).sum())} profiles: both costs zero',transform=ax.transAxes,va='top',fontsize=9)
    axs[0].set_ylabel('Misses / no-result stops (%)');axs[0].legend(frameon=False,loc='upper left')
    save(fig,'fig2_cost_risk_points')

    v=p[p.epsilon==.1].pivot(index='scenario',columns='method',values='conditional_miss');different=(v.psa1_s-v.legacy_exact).abs()>1e-12
    fig,ax=plt.subplots(figsize=(5.4,4.2),layout='constrained')
    ax.plot([0,21],[0,21],color='grey',lw=.8,label='Identical risk')
    ax.scatter(100*v.loc[~different,'legacy_exact'],100*v.loc[~different,'psa1_s'],facecolors='none',edgecolors=C[0],s=24,linewidths=.8,label='Unchanged')
    ax.scatter(100*v.loc[different,'legacy_exact'],100*v.loc[different,'psa1_s'],facecolors='none',edgecolors=C[1],s=32,linewidths=.9,label='Different optimal policy')
    ax.axhline(10,color='#b74249',ls='--',lw=.8);ax.axvline(10,color='#b74249',ls='--',lw=.8)
    ax.set_xlabel('Legacy exact DP empirical risk (%)');ax.set_ylabel('Final PSA1-S empirical risk (%)')
    ax.set_title('Philly: identical cost, potentially different misses');style(ax);ax.legend(frameon=False,fontsize=8)
    save(fig,'fig3_optimal_policy_ties')

    pcols={'layout':'Elrendezés','gpus_requested':'GPU-feltétel','n':'Tesztpont','no_refresh_same_budget_mean_cost':'Opt. no-refresh',
           'mean_cost':'PSA1-S','saving_vs_no_refresh_percent':'Megtakarítás (%)','miss_n':'Elmulasztás'}
    fmt={'layout':'{:.0f}','gpus_requested':'{:.0f}','n':'{:.0f}','no_refresh_same_budget_mean_cost':'{:.4f}','mean_cost':'{:.4f}',
         'saving_vs_no_refresh_percent':'{:.2f}','miss_n':'{:.0f}'}
    t1=md(ptable,pcols,fmt);t2=md(atable,{k:v for k,v in pcols.items() if k!='gpus_requested'},fmt)
    pe=p[(p.replicate==0)&(p.gpus_requested==2)&(p.lag_min==1)&(p.calibration=='rolling14d')&(p.costs=='heterogeneous')&(p.refresh_ratio==.05)&(p.epsilon==.1)]
    t3=md(pe,{'method':'Módszer','mean_cost':'Költség','saving_vs_no_refresh_percent':'No-refresh megtakarítás (%)','miss_n':'Elmulasztás',
               'empty_stop_n':'Eredmény nélküli lezárás','conditional_miss':'Feltételes arány'},
               {'mean_cost':'{:.4f}','saving_vs_no_refresh_percent':'{:.2f}','miss_n':'{:.0f}','empty_stop_n':'{:.0f}','conditional_miss':'{:.2%}'})
    audit=pd.DataFrame([dict(dataset='Philly',**summary['philly']),dict(dataset='Alibaba-2018',**summary['alibaba2018'])])
    t4=md(audit,{'dataset':'Adat','configurations':'ε=0,10 profil','S_DP_risk_changed':'Eltérő kockázatú profil',
       'S_DP_risk_lower':'S kisebb arány','S_DP_risk_higher':'S nagyobb arány','S_DP_risk_max_abs_percentage_points':'Max. absz. különbség (százalékpont)'},
       {'configurations':'{:.0f}','S_DP_risk_changed':'{:.0f}','S_DP_risk_lower':'{:.0f}','S_DP_risk_higher':'{:.0f}','S_DP_risk_max_abs_percentage_points':'{:.4f}'})
    riskrows=[]
    for name,g,w in [('Philly',ps,pw),('Alibaba-2018',als,aw)]:riskrows.append(dict(data=name,miss=int(w.miss_n),stop=int(w.empty_stop_n),risk=w.conditional_miss,
       lo=w.risk_block95_low,hi=w.risk_block95_high,over=int((g.conditional_miss>.1).sum()),total=len(g)))
    t5=md(pd.DataFrame(riskrows),{'data':'Adat','miss':'Elmulasztás','stop':'Eredmény nélküli lezárás','risk':'Legnagyobb arány','lo':'95% intervallum alja','hi':'95% intervallum teteje','over':'10% feletti profil','total':'Összes profil'},
       {'miss':'{:.0f}','stop':'{:.0f}','risk':'{:.2%}','lo':'{:.2%}','hi':'{:.2%}','over':'{:.0f}','total':'{:.0f}'})
    report=f'''# Végleges PSA1-S: célzott Philly- és Alibaba-2018-replay

2026. november 17. — NOMS 2027, v1

**A két célzott visszajátszás elkészült a feltöltött, végleges algoritmuscsalád változatlan kódjával.** A nyers trace-eket nem dolgoztam újra, a korábbi p/α/β modelleket nem illesztettem újra, és nem hangoltam a teszteredmények alapján. Philly: 224 korábbi elsődleges konfiguráció, 896 mentett időszakmodell. Alibaba-2018: 54 korábbi költségprofil, 27 valószínűségi modell és hat megvizsgált keret, azaz 324 döntési modellváltozat.

A fő eredmény: a végleges PSA1-S költsége ezekben a kis feladatokban egyezik az egzakt DP-ével, **az elmulasztási arány azonban nem mindenhol azonos**. A friss információ költségelőnye a nulla keretű kontrollban is megmarad. A becsült modellbeli kockázati korlát a valós idősoron továbbra sem automatikus garancia.

## 1. Pontosan mi futott?

A felhasznált végleges implementáció a feltöltött `NOMS2027_Final_Algorithm_Benchmark_v1(1).zip` fájlból származik; SHA-256: `9701e7c4c035681cd6e72643b686a8b8d3e3bfc8858c716327a55e8cde85e789`. A `Selective(p,8,True)` osztályt és a benchmark PSA1-konfigurációját változtatás nélkül hívtam meg, csak a világállapotokat és megfigyeléseket adtam át a mért idősorokból.

Az itt vizsgált mindkét adatforrás négy-domaines feladatot ad. Ezért N=4≤K=8: a PSA1-S azonnali admisszibilis STOP-ot választ, vagy eleve egzakt tailből dönt. **A szelektív mélyítés nem aktiválódott**, amit minden modellben ellenőriztem. Ezek a kísérletek a végleges keretrendszer kis problémákon való, mérési adatokkal összekapcsolt működését és a kockázatátvitelt vizsgálják; a szelektív mechanizmus többletét a külön 10–12-domaines szintetikus benchmark támasztja alá.

Öt módszer szerepel:

- `query_only`: a korábbi, kockázati korai lezárás nélküli pontos keresés.
- `no_refresh_opt`: optimális, REFRESH nélküli stratégia, **ugyanazzal az ε-kerettel**; a benchmark 2^N-es rekurzióját akciótáblával egészítettem ki, rootértékét a változatlan benchmark-függvénnyel is ellenőriztem.
- `psa1`: a végleges csomag eredeti PSA1-konfigurációja.
- `psa1_s`: a végleges, K=8-as szelektív konfiguráció.
- `legacy_exact`: a korábbi, explicit SKIP-et is engedő egzakt DP-policy, összevetési kontrollként.

Minden modellnél mind a 256 igazság–REFRESH kimeneten tényleges policy-replay készült. A modell szerinti költséget független kimeneti valószínűségekkel integráltam, és összevetettem a rekurzív solverértékekkel. Az offline táblák építésekor tiszta állapotfüggvény-cache használható; **ez nem online időmérés**. A fő ág már elkészült runtime- és szintetikus benchmarkját nem futtattam újra.

## 2. A korábbi vizsgálatok változatlan részei

**Philly:** az elsődleges, teljes ismert GPU-terhelési/hostmemória-minta; ugyanazok a két-/négy-GPU-s predikátumok, három elrendezés, 1/5/15/60 perces jel, statikus vagy múltbeli 14 napból hetente frissülő modell. A tesztpontok, a hét naptári napos bootstrap és a régi modellek időszakai megmaradtak. A forrásból mentett 256 kimeneti gyakoriság minden időszakban pontosan reprodukálódott. A foglalási naplóval párosított másodlagos mintát most nem futtattam újra.

**Alibaba-2018:** ugyanazok a négy-domaines elrendezések, CPU–memória-feltételek, 1/10/60 perces jelek, költségprofilok és egymástól embargóval elválasztott tanítási–kalibrációs–tesztidőszakok. A modellparaméterek végig a korábbi tanításból rögzített értékek. A kalibrációs kapu változatlan: a {{0;0,01;0,025;0,05;0,075;0,10}} rács legnagyobb kerete, amely legalább 200 eredmény nélküli lezárásból maximum 5% feltételes elmulasztást adott; ennek hiányában nulla. A PSA1-S-re és az új no-refresh referenciára is ezt az eljárást alkalmaztam. A választások a tesztpolicyk kiértékelése előtt fájlba kerültek.

A kalibráción választott ε módszerenként eltérhet. Ilyenkor a CSV minden sorhoz külön megadja az **adott sor saját ε-jével** futó no-refresh referenciát is. A külön kalibrált módszersorok nem értelmezhetők automatikusan azonos valós kockázatú dominancia-összehasonlításként. A fix ε=0 és ε=0,10 protokollok valóban közös modellbeli keretet használnak.

A régi query-only, PSA1 és legacy-exact sorok költségei numerikus tolerancián belül, találati és elmulasztási számai pontosan reprodukálódtak. Mindkét adaton a végleges S modellköltsége egyezett az egzakt referenciával; a legnagyobb eltérés a tárolt ellenőrzés szerint 0. Az inputok és a végleges vendor-kód ellenőrzőösszege megmaradt.

## 3. Miért kellett a döntéseket ténylegesen visszajátszani?

Az egzakt költségminimumhoz több policy is tartozhat. A régi DP explicit SKIP-et és kerekített akcióértékekre épülő holtversenyoldást használ; az új tail négystátuszú, explicit SKIP nélküli akciótáblát ad. A két megoldás azonos várható modellköltség mellett másik megengedett optimális döntést választhat.

A jelen modellrácson minden 256-as kimeneti táblán a költségek egyeztek a végleges S és a régi egzakt policy között. **Az elmulasztások mégis eltérhetnek.** A fix 10%-os keretű tesztsorok összevetése:

{t4}

A Philly legnagyobb abszolút eltérése 6,18 százalékpont; van kedvező és kedvezőtlen irányú változás is. Az Alibaba egyetlen megváltozott sorában a különbség kicsi, körülbelül 0,0175 százalékpont. A fizetett QUERY/REFRESH akciósorokat külön tárolt audit hasonlítja össze; a pusztán nulla költségű SKIP elhagyását nem számítjuk fizetett akcióeltérésnek.

![Azonos költség és eltérő empirikus kockázat a Philly-n](fig3_optimal_policy_ties.png)

Ez a kockázat nem az optimalizáció célfüggvénye: a solver a költséget minimalizálja a megengedett posterior-korláton belül. Nem hangoltam át a holtversenyoldást a teszt alapján. **A cikk végleges S-kockázati számai ebből a visszajátszásból származzanak; a régi DP-sorok egyszerű átnevezése nem lett volna elég.**

## 4. Költségelőny nulla elmulasztási keret mellett

Az előre használt fő beállítások megmaradtak: egyperces jel, heterogén Q=(4,2,1,3), R=0,05Q. Philly-n a gördülő modell; Alibaba-n a rögzített, változatosabb C feltétel (CPU≤80%, memória≤90%). Az ε=0 no-refresh referencia költsége ezekben a mintákban azonos a szigorú query-only költségével.

### Philly

{t1}

A két-GPU-s esetekben a végleges S megtakarítása **14,38–21,15%**, a négy-GPU-s esetekben **6,85–18,73%**, megfigyelt elmulasztás nélkül. Ez a mentett, ismert állapotpredikátumokra és pontos offline QUERY-ra vonatkozik.

### Alibaba-2018, C feltétel

{t2}

A megtakarítás **2,54–3,52%**, szintén nulla elmulasztással. A harmadik elrendezésben a végleges S 3,04%-ot ad a korábbi PSA1 2,94%-ával szemben; a tartomány szélei változatlanok. A teljes paraméterrács kedvezőtlen eseteit is megőriztem: a modellre optimális S realizált költsége rosszabb is lehet a no-refresh referenciánál. Ez nem az egzakt DP hibája, hanem a becsült modell és a teszteloszlás különbsége mellett lehetséges viselkedés.

![Nulla keretű költségkontroll a végleges kóddal](fig1_zero_budget_final_policy.png)

## 5. Az optimális REFRESH nélküli referencia szerepe

A szigorú query-only kontroll és a kockázatkorlátos no-refresh stratégia különböző feladatot oldhat meg, ha ε>0. Az információszerzés vizsgálatához ezért az azonos ε-t használó optimális no-refresh referencia a fő összevetés; a szigorú kontroll a korábbi eredményekhez való kapcsolódást és a nulla keretű vizsgálatot segíti.

Egy már korábban kiemelt Philly-beállítás: első elrendezés, két-GPU-s feltétel, egyperces jel, gördülő modell, heterogén költség, ε=0,10:

{t3}

Ezek költség–kockázat pontok. A közös modellbeli ε nem jelenti, hogy az algoritmusok empirikus elmulasztási aránya azonos. Pozitív keretnél a különböző kockázatfelhasználást is látni kell a költségelőny mellett.

Az Alibaba szigorú A feltételének 18 profiljában a no-refresh és az S is költség nélkül STOP-olhat a kezdeti állapotban. A két nulla költség közötti relatív megtakarítás **nem definiált**; ezt nem írjuk 100%-os információs nyereségnek. A CSV külön megőrzi a szigorú query-only-hoz képesti, ilyen esetben 100%-os értéket, de ez nem az optimális no-refresh stratégia fölötti előny.

![Költség és empirikus kockázat azonos modellbeli keretnél](fig2_cost_risk_points.png)

Az ábra minden fix 10%-os profil pontbecslését mutatja, ahol a no-refresh referencia költsége pozitív. Az intervallumok és a nulla/nevező nélküli esetek az eredménytáblákban szerepelnek. A profilok egymással átfedő időpontokat használnak; nem független mérések.

## 6. A végleges PSA1-S kockázata

Fix ε=0,10 mellett a legnagyobb feltételes elmulasztási arányok:

{t5}

**Philly:** 1996/10692=18,67%, hét naptári napos, leíró mozgóblokk-bootstrap intervallummal 13,41–25,06%. A 112 pozitív keretű S-konfigurációból 34 pontbecslés meghaladta a 10%-ot. A régi PSA1 legrosszabb 18,69%-os értéke tehát nagyon hasonló, de a végleges számok az új S-policyhoz tartoznak.

**Alibaba:** 2412/23425=10,30%, egyórás blokkintervallummal 4,26–16,75%. A korábban kiemelt **7,45% a PSA1 legrosszabb aránya volt**, nem a végleges S-é. Itt az 54 fix keretű S-profilból egy pontbecslés lépte át a 10%-ot. A kalibrációs kapu ezt a profilt nem szigorította, így a kalibráción választott protokollban is megmarad az eltérés.

Az Alibaba S-kerete öt profilban szigorodott; a megválasztott keretek a régi DP kereteivel mind az 54 profilban egyeztek. Az optimalizálási költség pontos számítása és az empirikus kockázatkalibráció külön kérdések: **az egzakt tail a becsült modellt oldja meg, nem teszi automatikusan helyessé a beliefeket**. A no-refresh referencia sem ad önmagában mérésbeli garanciát; az Alibaba fix keretű no-refresh sorai között is volt 10% fölötti pontbecslés.

Minden enumerált és visszajátszott STOP teljesítette saját modelljének posterior-korlátját. A mérésbeli eltérés nem ennek a tételnek az ellenpéldája: a becslési hiba, időbeli változás és domainfüggőség megmaradó korlát. Ezeket itt nem választottam szét okságilag.

Az arány nevezője az eredmény nélküli lezárások száma. Nulla nevezőnél nem definiált, nulla eseménynél az empirikus bootstrap-intervallum elfajulhat. A leíró intervallumok nem adnak eloszlásváltozással szembeni megbízhatósági bizonyítványt. A nulla keretű kontrollokban minden vizsgált kimeneten nulla elmulasztás volt, az offline predikátum pontos QUERY-jának megfelelően.

## 7. A cikkben vállalható szerep

1. A **10–12-domaines független szintetikus benchmark** adja a szelektív előretekintés és a számítási kompromisszum algoritmikus bizonyítékát. A 3,17%-os átlagos gap és az online döntési idők onnan származnak; ezeket nem minősítjük telemetriás értéknek.
2. A **Philly és Alibaba célzott replay** azt mutatja, hogyan kapcsolható a végleges keretrendszer valós mérésekből képzett állapotokhoz. Kis N mellett az S egzakt módja működik; a nulla keretű költségelőny a friss információ értékét támasztja alá.
3. A kockázati számok a **modellből a mérési idősorra való átvitel korlátját** mutatják. A helyes modellre vonatkozó posterior assurance, az empirikus kalibráció és a tényleges scheduler-elhelyezhetőség külön állítás.

Nem állítunk valódi federált hálózaton mért üzenetköltséget vagy online erőforrásfoglalási garanciát. A Philly domainfelosztása modellezett, az Alibaba forráscímkéihez kötött domainjei két-két gépet kínáló részhalmazok. Az eredeti mérési hiányok, normalizált memória és a GPU-VRAM hiánya nem változott. Eagle újrafuttatás és új 12-domaines telemetria-kísérlet nem készült ebben a lezáró munkafázisban.

## 8. Ellenőrzés és reprodukció

A csomag öt célzott unit tesztet, az összes modell és kimenet értékellenőrzését, az eredeti sorok reprodukciós ellenőrzését, mentett inputokat, döntési és holtverseny-auditot, napi/tízperces naptári statisztikákat, ábrákat és a változatlan végleges solverforrásokat tartalmazza. A `README_HU.md` adja a parancsokat. Nincs szükség a nagy eredeti nyersarchívumok új letöltésére.

A két futtatás összesen **1120 Philly- és 1620 Alibaba-eredménysort** adott. Ez módszer-, keret- és mintaváltozatok száma, nem független kísérleteké. A jelen környezet Python 3.12, NumPy 2.3.5, Numba 0.65.1; a fő benchmark Python 3.13-as környezetétől eltérő interpreter miatt új runtime-összehasonlítást nem közlünk.
'''
    (R/'NOMS2027_PSA1S_Frozen_Telemetry_Replay_v1_hu.md').write_text(report)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
