"""Primary report on all known telemetry; retain the job-paired sensitivity study."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import build_report as joint
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results_telemetry_full';O=ROOT/'report'
d=pd.read_csv(R/'policy_summary.csv');m=json.loads((R/'experiment_metadata.json').read_text())
psa=d[(d.method=='psa1')&(d.refresh_ratio==.05)]
core=psa[(psa.costs=='heterogeneous')&(psa.calibration=='rolling14d')]
zero=core[(core.lag_min==1)&(core.epsilon==0)].copy();zero['layout']=zero.replicate+1
base=d[(d.method=='query_only')&(d.costs=='heterogeneous')&(d.calibration=='rolling14d')&(d.lag_min==1)&(d.epsilon==0)&(d.refresh_ratio==.05)].set_index(['replicate','gpus_requested'])
zero['query_cost']=[base.loc[(r.replicate,r.gpus_requested),'mean_cost'] for r in zero.itertuples()]
zero['save_pct']=100*zero.saving_vs_query
ztable=joint.table(zero,['layout','gpus_requested','n','query_cost','mean_cost','save_pct','miss_n'],
 ['Elrendezés','Kért GPU/gép','Tesztpont','Query-only','PSA1','Megtakarítás','Elmulasztás'],
 [lambda x:str(int(x)),lambda x:str(int(x)),lambda x:str(int(x)),'.4f','.4f',lambda x:f'{x:.2f}%',lambda x:str(int(x))])
example=d[(d.replicate==0)&(d.gpus_requested==2)&(d.costs=='heterogeneous')&(d.lag_min==1)&(d.refresh_ratio==.05)&(d.calibration=='rolling14d')&(d.epsilon==.1)].copy()
example['save_pct']=100*example.saving_vs_query;example['risk_pct']=100*example.conditional_miss
etable=joint.table(example,['method','mean_cost','save_pct','miss_n','empty_stop_n','risk_pct'],
 ['Módszer','Realizált költség','Megtakarítás','Elmulasztás','Eredmény nélküli lezárás','Feltételes arány'],
 ['','.4f',lambda x:f'{x:.2f}%',lambda x:str(int(x)),lambda x:str(int(x)),lambda x:f'{x:.2f}%'])
fig,axs=plt.subplots(1,2,figsize=(11,4.4))
for ax,eps in zip(axs,[0.,.1]):
    for rep,col in enumerate(joint.colors):
        s=core[(core.replicate==rep)&(core.gpus_requested==2)&(core.epsilon==eps)].sort_values('lag_min')
        ax.plot(s.lag_min,100*s.saving_vs_query,marker='o',lw=2,color=col,label=f'Layout {rep+1}')
    ax.set_xscale('log');ax.set_xticks([1,5,15,60],['1','5','15','60']);ax.axhline(0,color='#777',lw=.8)
    ax.set_xlabel('Refresh age (minutes; log scale)');ax.set_ylabel('PSA1 saving vs query-only (%)')
    ax.set_title(f'2-GPU measured-state predicate, ε={eps:.2f}');ax.grid(axis='y',alpha=.2)
axs[1].legend(frameon=False)
fig.suptitle('Full known telemetry: Q=(4,2,1,3), R=0.05Q',y=1.02)
fig.tight_layout();fig.savefig(O/'fig4_full_telemetry_cost.png',dpi=180,bbox_inches='tight');plt.close(fig)
s=core[(core.gpus_requested==4)&(core.epsilon==.1)&(core.lag_min==5)].sort_values('replicate')
fig,ax=plt.subplots(figsize=(8.6,4.5));ax.bar(np.arange(3),100*s.conditional_miss,color=joint.colors,width=.55,alpha=.85)
for j,(_,v) in enumerate(s.iterrows()):
    ax.plot([j,j],[100*v.risk_block95_low,100*v.risk_block95_high],color='#222',lw=1.5)
    ax.annotate(f'{int(v.miss_n)} / {int(v.empty_stop_n)}',(j,100*v.conditional_miss),xytext=(0,7),textcoords='offset points',ha='center')
ax.axhline(10,color='#ac3939',ls='--',lw=1.5,label='Model budget ε=10%');ax.legend(frameon=False)
ax.set_xticks(np.arange(3),['Layout 1','Layout 2','Layout 3']);ax.set_ylabel('Misses / no-result stops (%)')
ax.set_title('Risk mismatch persists without job-log availability selection');ax.grid(axis='y',alpha=.2)
ax.text(0,-.19,'4-GPU predicate; 5-minute refresh; Q=(4,2,1,3); weekly calibration.\nIntervals: descriptive 7-day moving-block bootstrap, 1,000 replicates.',transform=ax.transAxes,fontsize=9)
fig.tight_layout();fig.savefig(O/'fig5_full_telemetry_risk.png',dpi=180,bbox_inches='tight');plt.close(fig)
risk=s.iloc[-1]
p=d.pivot(index='scenario',columns='method',values='mean_cost');mp=d.pivot(index='scenario',columns='method',values='model_expected_cost')
better=int((p.psa1<p.h1-1e-10).sum());worse=int((p.psa1>p.h1+1e-10).sum());same=int(((p.psa1-p.exact).abs()<1e-10).sum())
maxgap=100*(mp.psa1/mp.exact.replace(0,np.nan)-1).max();cnt=json.loads((R/'algorithmic_counterexample.json').read_text())
viol=int(((d.method=='psa1')&(d.epsilon==.1)&(d.conditional_miss>.1+1e-12)).sum())
audit=joint.audit;integ=joint.integrity
text=f'''# Microsoft Philly: adatminőségi audit és IA-OSR replay

2026. november 16. — első, reprodukálható változat

**Az eredeti Microsoft-adatokon az audit és a futtatások elkészültek.** Az elsődleges összehasonlítás a teljes ismert telemetriát használja: 224 konfiguráció és hat módszer, azaz 1344 eredménysor. Külön, további 448 konfiguráció vizsgálja a feladatnaplóval is párosítható részmintát. Összesen **672 kiértékelési konfiguráció és 4032 eredménysor** készült; az átfedő időpontok és paraméterváltozatok nem független kísérletek.

A fő eredmény: az információval javított sorrend megengedett kihagyás nélkül is költséget takarít meg; pozitív kockázati keretnél több esetben a PSA1 is javít a H1-hez képest. A becsült modell kockázati korlátja ugyanakkor nem teljesül automatikusan a későbbi mérési idősoron. Általános PSA1-közeloptimalitást nem állítunk.

## 1. Forrás, feldolgozás és adatminőség

A teljes eredeti [Microsoft Philly-archívumot](https://github.com/msr-fiddle/philly-traces) letöltöttem és ellenőriztem: 1 055 988 361 bájt, SHA-256 `2037ccf63a725a02be718b0e1aeee6142c90b7f62d0d849295120e9f6ba13d2c`, commit `29a1b87fa2d9ed80b83c9e3a37f3a88d382b031d`.

Ténylegesen feldolgozott bemenet: {audit['gpu']['rows']:,} GPU-telemetriarekord, {audit['memory']['rows']:,} hostmemória-rekord, {audit['jobs']['jobs']:,} feladat és {audit['jobs']['attempts']:,} futtatási kísérlet. A telemetria 552 gépazonosítót tartalmaz. A nyers forrás és a kutatás alapcikke: [Jeon és szerzőtársai, USENIX ATC 2019](https://www.usenix.org/conference/atc19/presentation/jeon).

A kísérletek 24, a tanítási időszakban jól megfigyelt gép nyolc-nyolc GPU-csatornáját használják. Először rögzített hash-sorrendben 96 gépazonosítót vettem, majd a 24-et kizárólag a november 1. előtti mérési lefedettségből választottam. Nem használtam későbbi alkalmasságot vagy algoritmuseredményt a gépválasztáshoz.

Az audit során három problémát kellett külön kezelni:

- **Hibás leltárcímkék.** A jobnaplóban megjelenő {integ['job_hosts']} gépből {integ['hosts_with_logged_gpu_index_exceeding_inventory']} olyan GPU-indexet is kapott, amely meghaladja a leltárban megadott darabszámot. Például a `m29` két GPU-sként szerepel, de nyolc megfigyelt GPU-csatornája és nyolcig terjedő kiosztása van. Ugyanezt a hivatalos adattár [8. hibajegye](https://github.com/msr-fiddle/philly-traces/issues/8) is jelzi. A kapacitás- és névleges VRAM-címkéket ezért nem használjuk; a gyökérokot nem találgatjuk.
- **Eltérő időjelölés.** A telemetria PDT/PST jelölése explicit. A jobnapló időzónája nincs dokumentálva. A YARN application ID-k ResourceManager-epochja és több első beküldés közötti 9–12 másodperces különbség UTC-t támogat. Ezt az időeltolási érzékenységvizsgálat is támogatja. Az UTC-értelmezés dokumentált következtetés; a bizonyítékokat CSV-k őrzik. Az application ID jelentését a [Hadoop API](https://hadoop.apache.org/docs/r2.7.7/api/org/apache/hadoop/yarn/api/records/ApplicationId.html) dokumentálja.
- **Hiányos foglalási napló.** {audit['jobs']['missing_start']} kísérletnél nincs értelmezhető kezdés, {audit['jobs']['missing_end']} esetben befejezés, {audit['jobs']['reversed_intervals']} esetben hibás a végpontok sorrendje. Az ilyen időszakok bizonytalanok maradnak. A nyilvános workload részhalmaz; a napló hiánya nem bizonyít üres GPU-t.

A részletes audit és a foglalási részminta eredményei a [külön mellékanyagban](Philly_joint_subset_details_hu.md) olvashatók. Az elsődleges, alább közölt kontroll a foglalási bizonytalanság miatt **nem dob el egyébként jól mért terhelési/memóriapontokat**.

## 2. A kísérlet pontos jelentése

Egy gép megfigyelési feltétele: legalább két, illetve négy GPU-n a mért terhelés legfeljebb 20%, és a host `mem_free/mem_total` aránya legalább 5%. Ez **mért állapotra épülő jelöltpredikátum**, nem új feladat scheduler által igazolt elhelyezhetősége. Dinamikus szabad GPU-VRAM-ot nem mérünk. Az abszolút hostmemória-egység feltételezése helyett dimenziótlan arányt használunk.

Három elrendezés készült; mindegyik négy virtuális domainből áll, domainenként két géppel. A 24 gépet rögzített, 2026-os magú permutáció osztja három diszjunkt nyolcgépes csoportba. A csoportok valódi méréseket használnak, de a domainhatárok modellezettek; nem állítunk valódi federációt vagy racktagságot.

A tanítás november 1. előtt történik, majd a teszt előtt egyórás rést hagyunk. A source percét csak a perc végén tekintjük elérhetőnek. A REFRESH az 1, 5, 15 vagy 60 perccel korábbi állapotjel. Minden elrendezésen belül a két intent, a négy késleltetés és minden algoritmus ugyanazokat a tesztidőpontokat használja. A teszthalmazok mérete 61 185, 62 788 és 64 129 perc; nincs interpoláció vagy forward-fill.

Két kalibráció: statikus, teszt előtti modell; illetve heti frissítés az előző 14 napból, legalább 100 páros megfigyelés hiányában kifejezett statikus visszatéréssel. Csak korábban bekövetkezett megfigyelések használhatók. A posteriorok Laplace-simított pontbecslések, nem statisztikai felső kockázati korlátok. A domainek függetlenségét a döntési modell feltételezi; a mérési adatokat nem függetlenítettük mesterségesen.

A QUERY-költség (1,1,1,1) vagy (4,2,1,3); a REFRESH ennek 5%-a, és az első elrendezés négy-GPU-s eseteiben 20%-os érzékenységvizsgálat is szerepel. Minden költség modellezett, normalizált érték; nincs mért hálózati RTT. Hat módszer: query-only, risk-aware query, screening-greedy, H1, PSA1 és egzakt Bellman-DP.

## 3. Költségelőny megengedett kihagyás nélkül

Az ε=0 kontrollban minden megmaradt bizonytalanságot pontos QUERY old fel. A refresh az ellenőrzések sorrendjét javíthatja, de kockázati SKIP vagy feloldatlan domain melletti STOP nincs. A költségelőny ezért nem az elfogadott elmulasztásokból ered.

Teljes ismert telemetria, egyperces refresh, adaptív kalibráció, Q=(4,2,1,3), R=0,05Q, ε=0:

{ztable}

A két-GPU-s esetekben a megtakarítás **13,70–20,72%**, a négy-GPU-s esetekben **6,37–18,33%** volt, a megfigyelt predikátumokra nulla elmulasztással. Ebben az ε=0 kontrollban a H1 és a PSA1 azonos: az eredmény az információval javított sorrend értékét mutatja, nem a PSA1 egyedi többletét. A nulla elmulasztás az offline, pontos QUERY-val feloldott predikátumokra vonatkozik; nem valós erőforrásfoglalási garancia.

![Költség és refresh-késleltetés a teljes telemetrián](fig4_full_telemetry_cost.png)

## 4. PSA1 a H1-hez és a többi baseline-hoz képest

Az első elrendezés két-GPU-s esete, egyperces refresh, adaptív kalibráció, Q=(4,2,1,3), R=0,05Q, ε=0,10:

{etable}

A PSA1 itt 2,1739, a H1 2,2883, a screening-greedy 2,6302, a DP 2,1151 költséget adott. A PSA1 a query-only-hoz képest 43,32% megtakarítást ért el, de a költség mellé a feltételes elmulasztási arányt is meg kell adni. A módszerek nem feltétlenül ugyanazokat a találatokat engedik el.

Az elsődleges 224 konfigurációban a PSA1 {better} esetben olcsóbb, {worse} esetben drágább volt a H1-nél; a többi esetben egyezett a realizált költség. A DP-vel {same} esetben egyezett. A tanult független modellek szerinti, időszakokkal súlyozott várható költségben a legnagyobb PSA1/DP-gap **{maxgap:.2f}%** volt. Az egzakt DP a becsült modellre optimális, nem szükségszerűen a későbbi mért eloszlásra.

## 5. A kockázati korlát nem vihető át automatikusan a trace-re

A harmadik elrendezés négy-GPU-s esetében, ötperces refresh-sel, heterogén QUERY-költséggel és adaptív kalibrációval a PSA1 **{int(risk.miss_n)}/{int(risk.empty_stop_n)} = {100*risk.conditional_miss:.2f}%** feltételes elmulasztási arányt adott ε=10% mellett. A leíró, hét napos mozgóblokk-bootstrap 95%-os intervalluma {100*risk.risk_block95_low:.2f}–{100*risk.risk_block95_high:.2f}%. A kockázati keret feletti pontbecslés tehát a teljes ismert telemetrián is megmarad; nem csak a foglalási naplóval párosított részminta jelensége.

![Kockázateltérés a teljes ismert telemetrián](fig5_full_telemetry_risk.png)

A 112 pozitív kockázati keretű PSA1-konfiguráció közül {viol} esetben volt a pontbecslés 10% felett. A nevező az eredmény nélküli lezárások száma, nem az összes kérés; az utóbbi arány külön CSV-oszlop. Ez nem a helyes modellre bizonyított korlát cáfolata: a becslési hiba, az időbeli változás és a domainek függősége egyaránt okozhat eltérést. A jelen vizsgálat e hatásokat még nem választja szét okságilag.

A blokk-bootstrap az időbeli összefüggésre választott skálájú leíró kiegészítés, nem állomásfüggetlen assurance-bizonyítvány. Nulla nevező nincs nullaként kezelve, nulla megfigyelt eseménynél pedig az empirikus intervallum elfajulhat. Minden döntési modellben minden STOP a saját modellje szerinti korláton belül maradt.

## 6. Kedvezőtlen algoritmikus esetek és a részminta szerepe

A teljes telemetrián talált legnagyobb egyedi, tanult modellbeli PSA1/DP-gap {100*cnt['relative_gap']:.2f}%. A gyorsított és közvetlen continuation {cnt['fast_vs_direct_continuation_checked_admissible_states']} megengedett állapotban egyezett a kedvezőtlen modellben.

A foglalási naplóval is párosítható részminta külön 448 konfigurációjában lényegesen rosszabb modell is előkerült: DP=0,406522, PSA1=1,053904, **159,25% egyedi időszakbeli gap**. Ennek a konfigurációnak az időszakokkal súlyozott modell szerinti gapje 16,79%. A közvetlen és gyorsított continuation 990 megengedett állapotban egyezett. Az ellenpélda és paraméterei a `results/algorithmic_counterexample.json` fájlban vannak; a teljes telemetria ellenpéldáját a `results_telemetry_full/algorithmic_counterexample.json` tartalmazza.

Az egyedi modell-gap nem a teljes trace átlagos gapje. A modell szerinti rollout-költségkorlát mindkét esetben teljesült; nem találtunk gyorsítási implementációs hibát. **Általános közeloptimalitási állításra nincs alap.** Az eredmény a több egymásra épülő REFRESH-művelet értékének célzott vizsgálatát indokolja; ez egyelőre kutatási irány, nem kész új algoritmus.

A foglalási részminta feltétele a mért terhelési/memóriakövetelményt a megerősített naplózott foglalások kizárásával egészíti ki, a hiányos rekordokat bizonytalannak hagyva. Sokkal kevesebb időpontot és gyakran szinte üres jelölthalmazt ad. A részminta részletes eredményeit ezért a [külön mellékanyagban](Philly_joint_subset_details_hu.md) kezeljük; nem ez adja az elsődleges költségállítások mintáját.

## 7. Mit ad ez a NOMS-cikkhez?

Az Eagle-pilotnál erősebb kiegészítő esettanulmány: eredeti forrásból feldolgozott, perces mérések, négy domain, sok negatív keresési helyzet, közös időpontokon vizsgált késleltetések és ε=0 kontroll. A friss információ költségértéke kimutatható; pozitív kockázati keretnél a PSA1 több esetben a H1-nél is jobb.

Az assurance szempontjából az eredmény a modellből a trace-re történő kockázatátvitel korlátját mutatja. A nagy megtakarításokat nem szabad igazoltan ε-megengedett, valós scheduler-teljesítményként közölni. A domainhatárok és hálózati költségek modellezettek, a leltárhibák miatt névleges VRAM-kapacitást nem állítunk, a workload naplója pedig részhalmaz.

Most a kedvezőtlen PSA1-modell szerkezetének és a kockázateltérés forrásainak elemzése adhat többet a cikkhez, mint újabb trace-ek önmagukban. A fő algoritmikus garanciák bizonyítéka továbbra is az analitikus eredmény és a pontosan körülhatárolt kontrollált modellvizsgálat.

## 8. Reprodukálás és ellenőrzések

A csomag tartalmazza a 24 gépes mért részhalmazt, a teljes és párosított minták összes eredményét, állapotexportját, napi statisztikáit, tanult modelljeit, mindkét kedvezőtlen esetet, a forrásauditot és a kódot. A nagy eredeti CSV-k az ellenőrzőösszeggel rögzített archívumból újra előállíthatók; az algoritmusok a mellékelt részhalmazzal új letöltés nélkül is futtathatók.

Nyolc célzott ellenőrzés sikeresen lefutott. Minden illesztett független négy-domaines modellben mind a 256 igazság–refresh kimenetet enumeráltuk, és a DP- és PSA1-költségeket a rekurzív értékfüggvénnyel is ellenőriztük. A mérési részhalmaz numerikus mezői float32 formában tárolódnak. A véletlen magok és a függőségek rögzítettek. Az eredeti adatok CC BY 4.0 forrásmegjelölése és a kötelező alapcikk-hivatkozás a README-ben és a `references.bib` fájlban szerepel.
'''
(O/'NOMS2027_Philly_audit_replay_v1_hu.md').write_text(text,encoding='utf-8')
zero.to_csv(R/'headline_zero_risk_control.csv',index=False);example.to_csv(R/'headline_psa_comparison.csv',index=False)
print('Primary report created:',O/'NOMS2027_Philly_audit_replay_v1_hu.md')
