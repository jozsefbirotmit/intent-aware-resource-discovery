"""Build the Hungarian report, exact numeric summaries, and scientific figures."""
import json,hashlib,platform
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'results/alibaba2018';R=ROOT/'report'
COLORS=['#16677d','#ca792c','#776296'];NAMES=['A','B','C']
plt.rcParams.update({'font.size':10,'axes.spines.top':True,'axes.spines.right':True,
 'axes.linewidth':.8,'lines.linewidth':1.15,'lines.markersize':5,'savefig.dpi':180})

def style(ax):
    ax.grid(True,linewidth=.45,alpha=.45);ax.set_axisbelow(True)

def save(fig,name):
    fig.savefig(R/(name+'.png'),bbox_inches='tight');fig.savefig(R/(name+'.pdf'),bbox_inches='tight');plt.close(fig)

def table(df,columns,formats=None):
    formats=formats or {}
    lines=['| '+' | '.join(columns.values())+' |','| '+' | '.join('---' for _ in columns)+' |']
    for _,row in df.iterrows():
        vals=[]
        for c in columns:
            v=row[c]
            if pd.isna(v):s='n. a.'
            elif c in formats:s=formats[c].format(v)
            elif isinstance(v,(float,np.floating)):s=f'{v:.4f}'
            else:s=str(v)
            vals.append(s)
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)

def main():
    R.mkdir(exist_ok=True)
    a=json.loads((D/'audit_metadata.json').read_text());e=json.loads((D/'experiment_metadata.json').read_text())
    c1=json.loads((ROOT/'results/counterexample/joint_subset_diagnosis.json').read_text())
    c2=json.loads((ROOT/'results/counterexample/full_telemetry_diagnosis.json').read_text())
    allrows=pd.read_csv(D/'policy_summary.csv');s=allrows[allrows.scope=='per_lag_known'].copy()
    b=pd.read_csv(D/'brier_scores.csv');cal=pd.read_csv(D/'calibration_budget_grid.csv')
    primary=s[(s.method=='psa1')&(s.costs=='heterogeneous')&(s.lag_s==60)].copy()
    primary['layout_label']=primary.layout.map(lambda k:str(k+1));primary['intent_label']=primary.intent.map(lambda k:NAMES[k])
    primary.to_csv(D/'headline_primary_psa1.csv',index=False)
    nominal=s[(s.method=='psa1')&(s.protocol=='nominal_10pct')].set_index('profile')
    selected=s[(s.method=='psa1')&(s.protocol=='calibration_selected')].set_index('profile')
    changes=selected[selected.epsilon!=.1]
    paired=pd.DataFrame({'layout':selected.layout+1,'intent':selected.intent.map(lambda k:NAMES[k]),'lag_s':selected.lag_s,'costs':selected.costs,
       'selected_epsilon':selected.epsilon,'nominal_cost':nominal.mean_cost,'selected_cost':selected.mean_cost,
       'nominal_miss_n':nominal.miss_n,'selected_miss_n':selected.miss_n,'nominal_empty_stop_n':nominal.empty_stop_n,
       'selected_empty_stop_n':selected.empty_stop_n,'nominal_risk':nominal.conditional_miss,'selected_risk':selected.conditional_miss})
    paired.to_csv(D/'budget_calibration_paired.csv',index=True)
    method_comparisons={}
    for proto in s.protocol.unique():
        w=s[s.protocol==proto].pivot(index='profile',columns='method',values='mean_cost')
        method_comparisons[proto]={'PSA1_cheaper_than_H1':int((w.psa1<w.h1-1e-10).sum()),'PSA1_more_expensive_than_H1':int((w.psa1>w.h1+1e-10).sum()),
           'PSA1_equals_DP':int(np.isclose(w.psa1,w.exact,atol=1e-10,rtol=0).sum()),'PSA2_cheaper_than_PSA1':int((w.psa2<w.psa1-1e-10).sum()),
           'PSA2_more_expensive_than_PSA1':int((w.psa2>w.psa1+1e-10).sum())}
    zero=s[(s.method=='psa1')&(s.protocol=='zero_risk_control')]
    worst=selected.loc[selected.conditional_miss.idxmax()]
    summary=dict(counterexample=dict(exact_cost=c1['exact_cost'],PSA1_cost=c1['PSA1_cost'],depth2_cost=c1['depth_comparison'][1]['policy_cost'],
       gap_percent=c1['gap_percent'],optimal_three_negative_risk=c1['optimal_all_negative_path'][-1]['risk'],
       premature_skip_all_other_negative_risk=c1['all_other_negative_with_unrefreshed_committed_skip_risk']['0']),
       audit=dict(rows=a['rows'],metadata_machines=a['metadata_machines'],usage_machines=a['usage_known_machines'],duplicate_rows=a['window_duplicate_rows'],
        invalid_cpu_rows=a['cpu_invalid_rows'],time_reversals=a['file_order_time_reversals'],
        missing_fraction_among_metadata_machines=1-a['window_unique_machine_time_points']/(a['metadata_machines']*69120)),
       zero_budget=dict(min_saving_percent=float(zero.saving_percent.min()),max_saving_percent=float(zero.saving_percent.max()),misses=int(zero.miss_n.sum())),
       method_comparisons=method_comparisons,budget_calibration=dict(PSA1_profiles_changed=len(changes),profiles=len(selected),
        nominal_profiles_risk_over_10pct=int((nominal.conditional_miss>.1).sum()),selected_profiles_risk_over_10pct=int((selected.conditional_miss>.1).sum()),
        selected_profiles_CI_upper_over_10pct=int((selected.conditional_miss_hi>.1).sum()),worst_selected_profile=int(selected.conditional_miss.idxmax()),
        worst_selected_risk=float(worst.conditional_miss),worst_selected_misses=int(worst.miss_n),worst_selected_stops=int(worst.empty_stop_n),
        worst_selected_CI=[float(worst.conditional_miss_lo),float(worst.conditional_miss_hi)]),result_rows=e['policy_result_rows'])
    (D/'headline_summary.json').write_text(json.dumps(summary,indent=2)+'\n')

    # Counterexample: compare risk thresholds and exactly integrated model costs.
    fig,ax=plt.subplots(1,2,figsize=(10.5,3.6),layout='constrained')
    risks=[c1['optimal_all_negative_path'][-1]['risk'],c1['all_other_negative_with_unrefreshed_committed_skip_risk']['0']]
    ax[0].bar([0,1],np.array(risks)*100,color=[COLORS[0],COLORS[1]],width=.6)
    ax[0].axhline(10,color='#b74148',ls='--',lw=1.1,label='Budget 10%')
    ax[0].set_xticks([0,1],['Three negative refreshes\n(skip D4)','Commit skip D1\n+ all other signals negative'])
    ax[0].set_ylabel('Posterior residual risk (%)');ax[0].legend(frameon=False)
    for i,v in enumerate(risks):
        ax[0].text(i,v*100-.65 if v<.1 else v*100+.3,f'{v*100:.2f}%',ha='center',color='white' if v<.1 else 'black')
    cc=[c1['PSA1_cost'],c1['depth_comparison'][1]['policy_cost'],c1['exact_cost']]
    ax[1].bar(range(3),cc,color=[COLORS[1],COLORS[0],COLORS[2]],width=.6)
    ax[1].set_xticks(range(3),['PSA1','Depth 2','Exact DP']);ax[1].set_ylabel('Model-expected cost')
    for i,v in enumerate(cc):ax[1].text(i,v+.025,f'{v:.4f}',ha='center')
    ax[0].set_ylim(0,13);ax[1].set_ylim(0,1.22)
    for x in ax:style(x)
    save(fig,'fig1_counterexample_mechanism')

    data=np.load(D/'alibaba2018_measured_subset.npz');ts=data['time_s'];measure=np.isfinite(data['cpu'])&np.isfinite(data['mem'])
    matrix=measure.reshape(24,8,8640).mean(axis=2)
    fig,ax=plt.subplots(figsize=(9.4,5.2),layout='constrained')
    im=ax.imshow(matrix,aspect='auto',vmin=0,vmax=1,cmap='viridis',extent=[0,8,24,0])
    ax.axvline(3,color='white',lw=1);ax.axvline(5,color='white',lw=1)
    ax.set_xticks(np.arange(.5,8,1),[str(i) for i in range(1,9)]);ax.set_xlabel('Relative day')
    ax.set_yticks(np.arange(24)+.5,data['machine_ids']);ax.set_ylabel('Selected measured host')
    fig.colorbar(im,ax=ax,label='Fraction of observed CPU + memory samples')
    ax.set_title('Training: days 1–3; calibration: days 4–5; test: days 6–8 (1h embargoes)')
    save(fig,'fig2_measurement_coverage')

    zz=primary[primary.protocol=='zero_risk_control'];fig,ax=plt.subplots(figsize=(9.2,3.7),layout='constrained')
    for layout in range(3):
        v=zz[zz.layout==layout].sort_values('intent')
        ax.plot(v.intent, v.saving_percent,marker='o',markerfacecolor='none',color=COLORS[layout],label=f'Layout {layout+1}')
    ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(3),['A: CPU≤40%, mem≤70%','B: CPU≤60%, mem≤80%','C: CPU≤80%, mem≤90%'])
    ax.set_ylabel('PSA1 saving vs query-only (%)');ax.set_title('ε=0; 60s refresh; Q=(4,2,1,3), R=0.05Q');style(ax);ax.legend(frameon=False)
    save(fig,'fig3_zero_budget_cost')

    # Show all three ages for the representative B predicate; same common calendar mask.
    risk=allrows[(allrows.scope=='all_intents_lags_common')&(allrows.method=='psa1')&(allrows.intent==1)&(allrows.costs=='heterogeneous')]
    fig,axs=plt.subplots(1,2,figsize=(11,3.8),sharey=True,layout='constrained')
    for ax,proto,title in zip(axs,['nominal_10pct','calibration_selected'],['Nominal ε=10%','Calibration-selected ε']):
        for layout in range(3):
            v=risk[(risk.protocol==proto)&(risk.layout==layout)].sort_values('lag_s')
            ax.plot(v.lag_s/60,100*v.conditional_miss,marker='o',markerfacecolor='none',color=COLORS[layout],label=f'Layout {layout+1}')
        ax.axhline(10,color='#b74148',ls='--',lw=1.1);ax.set_xscale('log');ax.set_xticks([1,10,60],['1','10','60'])
        ax.set_xlabel('Refresh age (minutes; log scale)');ax.set_title(title);style(ax)
    axs[0].set_ylabel('Misses / no-result stops (%)');axs[1].legend(frameon=False)
    save(fig,'fig4_holdout_risk')

    v=b[(b.costs=='equal')&(b.lag_s==60)]
    bb=v.groupby(['intent','split'])[['prior_brier','posterior_brier','observed_prevalence','prior']].mean().reset_index()
    bb.to_csv(D/'headline_brier_scores.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(11,3.4),sharex=True,layout='constrained')
    parts=['train','calibration','test']
    for it,ax in enumerate(axs):
        vv=bb[bb.intent==it].set_index('split').loc[parts]
        ax.plot(range(3),vv.prior_brier,marker='o',markerfacecolor='none',color=COLORS[1],label='Prior only')
        ax.plot(range(3),vv.posterior_brier,marker='o',markerfacecolor='none',color=COLORS[0],label='After refresh')
        ax.set_xticks(range(3),['Train','Calibrate','Test']);ax.set_title(f'Predicate {NAMES[it]}');style(ax)
    axs[0].set_ylabel('Mean domain Brier score');axs[-1].legend(frameon=False)
    save(fig,'fig5_posterior_brier')

    # Human-readable numeric exports retain missing entries and the actual calendar.
    exports=[]
    for k,h in enumerate(data['machine_ids']):
        exports.append(pd.DataFrame(dict(time_s=ts,machine_id=h,cpu_util_percent=data['cpu'][k],mem_util_percent=data['mem'][k],
                                         status_using=data['using'][k],source_failure_domain_2=data['failure_domain_2'][k])))
    pd.concat(exports,ignore_index=True).to_csv(D/'selected_measurements.csv.gz',index=False,compression='gzip',float_format='%.0f')
    states=np.load(D/'domain_states.npz');ex=[]
    for layout in range(3):
      for it in range(3):
        z=states[f'layout{layout}_intent{it}'];v=pd.DataFrame(z,columns=[f'domain_{i+1}' for i in range(4)])
        v.insert(0,'time_s',ts);v.insert(1,'layout',layout);v.insert(2,'intent',it);ex.append(v)
    pd.concat(ex,ignore_index=True).to_csv(D/'domain_states.csv.gz',index=False,compression='gzip')

    zp=primary[(primary.protocol=='zero_risk_control')&(primary.intent==2)]
    ztable=table(primary[primary.protocol=='zero_risk_control'],{'layout_label':'Elrendezés','intent_label':'Feltétel','n':'Tesztpont',
      'query_only_mean_cost':'Query-only','mean_cost':'PSA1','saving_percent':'Megtakarítás (%)','miss_n':'Elmulasztás'},
      {'n':'{:.0f}','query_only_mean_cost':'{:.4f}','mean_cost':'{:.4f}','saving_percent':'{:.2f}','miss_n':'{:.0f}'})
    counter=pd.DataFrame([{'model':'Philly, foglalási részminta','depth':v['depth'],'cost':v['policy_cost'],'gap':v['gap_percent']} for v in c1['depth_comparison']]+
                         [{'model':'Philly, teljes telemetria','depth':v['depth'],'cost':v['policy_cost'],'gap':v['gap_percent']} for v in c2['depth_comparison']])
    ctable=table(counter,{'model':'Tanult modell','depth':'Mélység','cost':'Várható költség','gap':'DP-gap (%)'},
                         {'depth':'{:.0f}','cost':'{:.6f}','gap':'{:.2f}'})
    # Compact root comparison, all alternatives remain in CSV.
    roots=[]
    for r in c1['root_actions']:
        if r['action'] in [['SKIP',0],['SKIP',3],['R',2]]:
            roots.append(dict(action=('SKIP' if r['action'][0]=='SKIP' else 'REFRESH')+f' D{r["action"][1]+1}',
                              heuristic=r['one_step_score'],exact=r['exact_action_value'],forced=r['forced_first_then_PSA1']))
    rtable=table(pd.DataFrame(roots),{'action':'Első akció','heuristic':'Egylépéses becslés','exact':'Egzakt folytatással','forced':'Utána PSA1-gyel'},
                 {'heuristic':'{:.6f}','exact':'{:.6f}','forced':'{:.6f}'})
    ptable=table(paired.loc[changes.index].reset_index(),{'profile':'Profil','layout':'Elrendezés','intent':'Feltétel','lag_s':'Jel kora (s)',
      'costs':'Költségprofil','selected_epsilon':'Választott ε','nominal_cost':'Nyers költség','selected_cost':'Kalibrált költség',
      'nominal_risk':'Nyers kockázat','selected_risk':'Kalibrált kockázat'},
      {'selected_epsilon':'{:.3f}','nominal_cost':'{:.4f}','selected_cost':'{:.4f}','nominal_risk':'{:.4%}','selected_risk':'{:.4%}'})
    missing=summary['audit']['missing_fraction_among_metadata_machines']*100
    report=f'''# PSA1-ellenpélda, Alibaba-2018-pilot és elkülönített kockázatkalibráció

2026. november 17. — NOMS-kutatási jelentés, v1

**A három kijelölt feladat elkészült.** Az ellenpéldát egzakt döntésfa- és regret-felbontással rekonstruáltam; az eredeti Alibaba-2018 géptáblák ellenőrzőösszegét és 246 934 820 használati rekordját auditáltam; 54 rögzített profilban, hét módszerrel, elkülönített tanítási–kalibrációs–tesztidőszakkal lefuttattam a pilotot. A költség és a feltételes elmulasztási arány külön kimenet. A címkézett domaincsoportok a valódi hibadomének géprészhalmazai; a keresési költségek modellezettek.

## 1. Mi okozza a 159%-os ellenpéldát?

Ez a korábbi Philly-pilot foglalási naplóval is összevethető részmintájának egyetlen, tanult, független négy-domaines modellje. **Nem a teljes mérési idősor átlagos teljesítménye, és nem modell–mérés kockázateltérés.** Q_i=1, R_i=0,05 és ε=0,10. A modellben az egzakt DP költsége 0,406522, a PSA1-é 1,053904; relatív különbség 159,25%.

A domainek jelölése itt D1–D4; a kód 0–3 indexelést használ. D1 kezdeti alkalmassági valószínűsége 4,8797%; negatív frissítés után 2,7389%-ra csökkenne. A PSA1 azonban az elején **visszavonhatatlanul SKIP-eli D1-et**, így ez a későbbi kockázatcsökkentés többé nem elérhető.

Az optimális policy egyik ekvivalens változata D4-et hagyja ki (1,2069%), majd D3, D2, D1 frissítését végzi a negatív ágakon. Három negatív frissítés után a kombinált maradékkockázat **9,5447%**, tehát megengedett a STOP. Ez az ág a modellben 72,7069% valószínűséggel érhető el, és 0,15 költségű. Pozitív jel után további pontos QUERY is szükséges; a teljes várható költség ezért 0,406522.

Ha viszont D1-et frissítés előtt elkötelezetten kihagyjuk, még mindhárom másik domain negatív frissítése után is **10,8646%** marad a kockázat. A megengedett STOP elérhetetlenné válik ezen az ágon; legalább egy további QUERY kell. A rossz kihagyás utáni PSA1 már optimális a megmaradt döntési térben: a teljes 0,647382 várható költségtöbblet az első akció egzakt Bellman-regretjéből származik.

{rtable}

Az egylépéses becslés két különböző kihagyást azonos értékűnek lát, miközben egzakt folytatásuk nagyon eltér. **A holtverseny másképp oldása önmagában nem javítás:** D4 első kihagyása után a PSA1 később még mindig meghozhatja a rossz D1-kihagyást. Sőt, kizárólag a legjobb első REFRESH kikényszerítése, utána ismét PSA1-gyel folytatva, ugyancsak 1,053904 költséget ad. A teljes döntéssorozat felismerése számít.

A base continuation legfeljebb egy közvetlen kezdeti SKIP-et és egy screening modult lát, aztán rögzített lekérdezési sort értékel. Nem értékeli elég jól a több egymást követő frissítés együttes, STOP-ot megnyitó hatását. A gyorsított és közvetlen base continuation korábbi ellenőrzése egyezett; a jelen felbontás ugyanezt a változatlan solvert használja.

![Az ellenpélda kockázati küszöbe és modellbeli költsége](fig1_counterexample_mechanism.png)

## 2. Mélyebb előretekintés: célzott kontroll, nem általános javítási tétel

Ugyanazt a PSA base continuationt Bellman-lépésekkel mélyítettem, és minden elért állapotban újra ugyanakkora mélységet alkalmaztam. A várható költségeket teljes döntésfa-integrálással számítottam.

{ctable}

A 159%-os esetben a kétlépéses policy már pontosan eléri a DP-t. A teljes telemetriából származó másik ellenpéldában a kettő csak kis javulás; három lépésnél 1,79%, négynél 1,36% marad a rés. **A mélység növelése nem jelent általános pontonkénti policy-dominanciát.** A modellbeli felső költségkorlát és az általános közeloptimalitás külön állítások. Új újdonságállítást itt nem teszek a lookahead vagy hybrid-DP módszerekre.

## 3. Alibaba-2018: eredeti forrás és adatminőségi audit

Forrás: [Alibaba clusterdata, v2018](https://github.com/alibaba/clusterdata/tree/3858992ca41bf92082e6d6522ddf0bebd14e838b/cluster-trace-v2018), rögzített commit `3858992ca41bf92082e6d6522ddf0bebd14e838b`. A `machine_meta.tar.gz` (92 432 bájt) és `machine_usage.tar.gz` (1 774 523 160 bájt) SHA-256 értékei megegyeztek a hivatalosan közölt értékekkel. A teljes használati CSV-t beolvastam; nem külső fél előfeldolgozott adatait használtam.

| Ellenőrzés | Eredmény |
|---|---:|
| Metaadat-sor | 17 592 |
| Metaadatban szereplő gép | 4 034 |
| Mérésekben szereplő gép | 4 023 |
| Használati rekord | 246 934 820 |
| Hibás időbélyeg / 10 másodperces rácson kívüli rekord | 0 / 0 |
| Ismétlődő gép–időpont rekord az elemzési ablakban | 0 |
| Ismétlődő metaadatbeli gép–időpont rekord | 0 |
| Hiányzó második hibadomén-címke a metaadat-sorokban | 34 |
| Tartományon kívüli CPU- / memóriaérték | 6 / 0 |
| Gépen belüli időrendi visszalépés a forrás fájlsorrendjében | 32 191 |
| Változó hibadomén-, CPU-kapacitás- vagy memóriakapacitás-címke | 0 |

A fájl sorrendjét nem tekintjük időrendnek: a feldolgozás gép és valódi időbélyeg szerint helyezi rácsra a méréseket. Nincs interpoláció vagy mérési forward-fill. A metaadat-állapotokat eseménynaplóként, az utolsó már megtörtént eseményből olvassuk ki. A megfigyelt rácspontok hiánya az összes metaadatbeli gép nyolcnapos rácsára vetítve {missing:.2f}%; ez tartalmazza a később belépő, illetve mérés nélküli gépeket is, tehát nem tisztán szenzorcsomag-vesztési arány.

Minden metaadatbeli gép 96 CPU-t és 100-as **normalizált** memóriakapacitást kapott. A memóriahasználat százalékos; ebből nem vezetünk le abszolút GiB értéket. A jelen lévő hibadomén-címkék között nincs eltérő értékű változás; a 34 hiányzó második címkét nem találgatjuk. A kiválasztás csak kezdetben már ismert címkéjű gépeket használ. A címkék publikált jelentéséből nem következik adminisztratív domainhatár vagy teljes hálózati topológia. A kezdeti metaadatban 3887 gép szerepel, további gépek később lépnek be; 127 metaadat-esemény már a nyolcnapos mérési ablak utánra esik, ezeket a kiértékelés nem használja jövőbeli információként.

192, hash-sorrendben kiválasztott jelöltből, kizárólag tanítási mérési lefedettség alapján 24 gépet választottam. A kezdeti `failure_domain_2` címkék első 12 elemét ugyancsak rögzített hash-sorrend adta. Három elrendezés, elrendezésenként négy forráscsoport, csoportonként két megfigyelt gép. A teljes hibadomén kapacitását nem reprezentáljuk ezzel.

![A kiválasztott gépek napi megfigyelési lefedettsége](fig2_measurement_coverage.png)

A korai időszak lefedettsége lényegesen rosszabb a későbbinél. A hiányok miatti szelekciós torzítás megmaradó korlát: a teljeseset-minták nem feltétlenül reprezentálják az összes klaszterállapotot. Az adatkonzisztencia több tekintetben kedvezőbb a Philly hibás kapacitáscímkéinél, de itt csak nyolc nap és normalizált CPU–memóriaadat áll rendelkezésre. Ez más szempontból jobb adat, nem minden feladatra jobb trace.

## 4. Rögzített feltételek, időbeli felosztás és módszerek

Az előzetes terv a használati idősor teljes beolvasása és a policy-futtatások előtt készült: `provenance/analysis_plan_before_usage_scan.md`. A néhány első forrássort és a metaadatokat addig már megismertük. Ez rögzített pilotterv, nem formális előregisztráció.

| Feltétel | CPU-terhelés legfeljebb | Memóriahasználat legfeljebb |
|---|---:|---:|
| A | 40% | 70% |
| B | 60% | 80% |
| C | 80% | 90% |

Egy domain akkor teljesíti a feltételt, ha a két kiválasztott gép közül legalább egy az adott mért pontban megfelel, és metaállapota `USING`. A hiányzó mérés ismeretlen; egy ismert negatív részkövetelmény vagy ismert pozitív gép logikailag rövidre zárhatja a predikátumot. **Ez mért állapotra épülő alkalmassági feltétel, nem foglalási/scheduler-garancia.**

Tanítás: [0,3 nap); kalibráció: [3 nap+1 óra,5 nap); teszt: [5 nap+1 óra,8 nap). A két egyórás rés lefedi a leghosszabb információkort. A modell paraméterei a tanítás után változatlanok maradnak. REFRESH-kor: 1, 10 vagy 60 perc. A QUERY a befejezett offline minta pontos predikátumát fedi fel, a keresés ideje alatt a vizsgált állapot rögzített.

Két költségprofil: Q=(1,1,1,1), illetve (4,2,1,3); R=0,05Q. Ezek modellezett, normalizált költségek. Hét módszer: query-only, risk-query, screening-greedy, H1, PSA1, egzakt Bellman-DP és a diagnosztikai kétlépéses PSA-rollout. A DP a tanult független modellre optimális, nem szükségszerűen a tesztidősor eloszlására.

Összesen **27 tanításból becsült valószínűségi modell × 2 költségprofil × 6 keret = 324 döntési modellváltozat**. Minden változat hét policyját mind a 256 igazság–refresh kimeneten enumeráltam; ez 580 608 determinisztikus epizód, nem ennyi független mérés. A teszt három protokollja: ε=0 költségkontroll, nyers ε=0,10, illetve kalibráción választott ε. A per-lag és közös időpontú érzékenységi táblák összesen 2268 eredménysort adnak. Az átfedő profilok és kimenetek nem független kísérletek.

Módszerenként azonos pontokat hasonlítunk. Az információkorok közös ismert tesztmintái elrendezésenként 21 975, 22 005 és 20 675 pontot tartalmaznak; a teljes, információkoronként ismert minták külön táblában maradnak.

## 5. Költség: külön, nulla megengedett elmulasztással

Az alábbi, előre kijelölt fő összehasonlítás egyperces jelet, heterogén költségeket és ε=0-t használ. A query-only és a PSA1 ugyanazon pontokon futott.

{ztable}

A változatosabb C feltételnél a megtakarítás **{zp.saving_percent.min():.2f}–{zp.saving_percent.max():.2f}%** a három elrendezésben, nulla elmulasztással. Ebben az összehasonlításban H1=PSA1. A teljes 54 profilban a PSA1 ε=0 megtakarítása **{zero.saving_percent.min():.2f}–{zero.saving_percent.max():.2f}%**: kedvezőtlen esetek is vannak. Ez az információ költségértékét mutatja, nem egységes PSA1-fölényt.

![Nulla kockázati keretű költségkontroll](fig3_zero_budget_cost.png)

A szigorú A feltétel többnyire üres jelölthalmazt ad; a B is ritka. Pozitív kockázati keretnél A egyes eseteiben 100% névleges költségmegtakarítás jelentkezik, mert a modell már a kezdeti állapotban megenged STOP-ot. **Ezt nem tekintjük látványos algoritmikus eredménynek.** Az egyik elrendezés A feltételében a tesztpontok 4,02%-ában mégis létezett megfelelő jelölt, amelyet az ilyen keresés elmulasztott. Az összes módszer ugyanilyen kezdeti STOP-ot alkalmazhat, tehát ez nem PSA1-specifikus nyereség.

Az ε=0 rácson PSA1 és DP realizált költsége {method_comparisons['zero_risk_control']['PSA1_equals_DP']}/54 profilban egyezett. Kétlépéses rollout a PSA1-hez képest egy esetben olcsóbb és négyben drágább lett a mérési idősoron. A tanult modell és a teszteloszlás eltérése mellett a mélyebb előretekintés sem ad automatikus realizált költségjavulást.

## 6. Kockázatkalibráció: rögzítés a teszt előtt

A tanításból becsült p, α, β változatlan maradt. A kalibrációs időszakon a {{0;0,01;0,025;0,05;0,075;0,10}} keretek közül módszerenként a legnagyobbat választottam, amely legalább 200 eredmény nélküli lezárást és legfeljebb 5% empirikus feltételes elmulasztást adott. Ennek hiányában ε=0 lett volna a választás. A 200-as minimum és az 5%-os kapu előzetes mérnöki döntés; **nem statisztikai tanúsítás** a későbbi 10%-os kockázatra.

A választások a teszt kiértékelése előtt fájlba kerültek. Rögzített SHA-256: `{e['calibration_choices_sha256']}`. A teszt nem módosította a gépválasztást, a küszöböket, a posteriorokat vagy a keretet.

A PSA1 kerete **{len(changes)}/54 profilban** szigorodott, a többi 49-ben maradt 0,10. Az összes változtatott profil:

{ptable}

A legmarkánsabb csökkenésnél a feltételes tesztarány 4,5265%-ról 0,1212%-ra esett, de a heterogén költség 2,0000-ről 5,9751-re nőtt. Ez költség–kockázat kompromisszum, nem egyoldalú javulás. A névleges és a kalibráción választott PSA1-rácson egyaránt mind az 54 pontbecslés 10% alatt maradt. Ez eltér a korábbi Philly-eredménytől; itt nem találtunk pontbecsléses túllépést.

![Kockázat azonos közös tesztidőpontokon, B feltétel](fig4_holdout_risk.png)

A legnagyobb, kalibráció után mért arány **{int(worst.miss_n)}/{int(worst.empty_stop_n)}={100*worst.conditional_miss:.2f}%**, leíró egyórás mozgóblokk-bootstrap 95%-os intervalluma **{100*worst.conditional_miss_lo:.2f}–{100*worst.conditional_miss_hi:.2f}%**. Összesen {summary['budget_calibration']['selected_profiles_CI_upper_over_10pct']}/54 profilban nyúlt az intervallum felső széle 10% fölé. A modellek összes STOP-ja teljesítette a saját posterior-korlátját, de az empirikus kapu és a pontbecslések **nem igazolnak általános trace-beli kockázati garanciát**.

A feltételes arány nevezője az eredmény nélküli lezárások száma, nem az összes kérés. Nulla nevező nincs nullának kezelve. Nulla eseménynél a közönséges empirikus blokkintervallum [0,0]-ra elfajulhat, ezt a CSV külön jelöli; ilyen intervallumot nem használunk nulla populációs kockázat bizonyítékaként. A kalibrációs minimum 200 korrelált tízmásodperces mintát is jelenthet; nem 200 független eseményt.

## 7. Posterior-kalibráció és eloszlásváltozás

A Brier-score külön értékeli a p prior és a friss jelből számított posterior előrejelzését. A pontonkénti reliability táblák a becsült és megfigyelt P(X=1|Y) értékeket is tartalmazzák, domainenként és időszakonként. Ezek leíró mutatók; a kisebb Brier-score önmagában nem residual-risk assurance.

![A prior és a frissített posterior Brier-score-ja](fig5_posterior_brier.png)

A C feltételnél, egyperces jel mellett a train-párok átlagos becsült priorja kb. 0,756; a kalibráció és teszt megfigyelt átlagos alkalmassága kb. 0,385 és 0,497. A frissítés jelentősen javítja a pontonkénti előrejelzést, de az időbeli eloszlásváltozás továbbra is jelen van. Az összes tartományi korreláció és a hiányok okozta mintaválasztás külön korlát; itt nem állítunk oksági szétválasztást.

## 8. Felhasználás a cikkben

1. **Algoritmikus állítás:** a korai, visszavonhatatlan SKIP és a több frissítés együttes STOP-értéke konkrét, egzaktul ellenőrzött korlátja a PSA1 egylépéses becslésének. A kétlépéses változat a nagy ellenpéldát feloldja, de nem általános optimumgarancia.
2. **Mérési állítás:** eredeti Alibaba CPU–memória-idősoron, forrásbeli hibadomén-címkékből képzett csoportokon, bizonyos feltételeknél kimutatható információs költségelőny. A jelen szűk pilot nem mutat erős, általános PSA1-többletet; a szélsőséges üres keresési eseteket külön kell kezelni.
3. **Módszertani állítás:** a költséget, a modellbeli kockázati keretet, a kalibrációs kaput és a teszten mért kockázatot külön kell közölni. Az időbeli elkülönítés a tesztre történő visszahangolást akadályozza meg; a helytelen modell vagy az eloszlásváltozás hatását nem szünteti meg.

A korábbi Philly továbbra is erősebb fő példa a kedvező költségeredményekhez. Az Alibaba-2018 ettől független, más erőforrástípusú kontroll és jól auditálható, rövid idősor. A két adatforrás nem azonos intentet vagy fizikai infrastruktúrát vizsgál; százalékaikat nem értelmezzük közvetlen versenyként. Valódi elosztott foglalási kísérletet, mért signalling-költséget vagy dinamikus GPU-VRAM-garanciát ez a munka sem ad.

## 9. Reprodukció és ellenőrzések

A csomagban szerepel a 24 gépes mért kivonat NPZ és importálható CSV formában, a hiányzó állapotokat is megtartó domain-idősor, az összes eredmény, a kalibrációs rács és befagyasztott választások, a Brier/reliability táblák, a tízperces naptári elegendő statisztikák, mindkét ellenpélda részletes állapotfája és root-akcióértékei, az eredeti adatforrások rögzített leírása és a kód.

Hat célzott ellenőrzés sikeres: hiányzó mérések logikája; valódi naptári késleltetés; a kalibrációs kapu és elégtelen minta visszatérése; ε=0 mind a 256 kimeneten; nem definiált kockázati nevező; a nagy ellenpélda teljes első-akció-regretje. A 324 modellben az enumerált DP-, PSA1- és kétlépéses költségek egyeztek a rekurzív értékekkel; a STOP- és rollout-korlátok teljesültek. A negatív és nulla költségelőnyű eseteket is megőriztem.

A teljes eredeti használati CSV 8 996 532 344 bájt; a reprodukciós ZIP a kicsi mért kivonatot tartalmazza. A teljes feldolgozás a hivatalos archívumokból a mellékelt letöltővel újra elvégezhető. A részletes parancsok a `README_HU.md` fájlban vannak. Ez kutatási pilot, az eredeti adatközlő nem hagyta jóvá benchmarkként.
'''
    (R/'NOMS2027_PSA1_Alibaba2018_calibration_v1_hu.md').write_text(report)
    env={m:__import__(m).__version__ for m in ['numpy','pandas','scipy','matplotlib']};env['python']=platform.python_version()
    (ROOT/'provenance/environment.json').write_text(json.dumps(env,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
