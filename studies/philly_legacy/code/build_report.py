"""Create figures and a Hungarian research report from completed Philly runs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results';O=ROOT/'report';O.mkdir(exist_ok=True)
d=pd.read_csv(R/'policy_summary.csv');audit=json.loads((R/'audit_metadata.json').read_text())
meta=json.loads((R/'experiment_metadata.json').read_text());integrity=json.loads((R/'integrity_summary.json').read_text())
counter=json.loads((R/'algorithmic_counterexample.json').read_text())
fitted=json.loads((R/'fitted_models.json').read_text())
psa=d[d.method=='psa1'];primary=psa[(psa['mode']=='telemetry')&(psa.refresh_ratio==.05)]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors=['#146078','#c87931','#6e5b92']

cons=pd.read_csv(R/'inventory_job_telemetry_consistency.csv')
fig,ax=plt.subplots(figsize=(8.2,4.4))
bottom=np.zeros(2)
for count,col in [(2,'#9cb9cc'),(4,'#c87931'),(8,'#146078')]:
    vals=[int(((cons.inventory_gpus==n)&(cons.max_logged_gpu_id_plus1==count)).sum()) for n in [2,8]]
    ax.bar([0,1],vals,bottom=bottom,width=.55,color=col,label=f'Largest allocated GPU index + 1 = {count}')
    for j,v in enumerate(vals):
        if v and v<12:
            ax.annotate(str(v),(j,bottom[j]+v/2),xytext=(28,0),textcoords='offset points',ha='left',va='center',color='#222',fontweight='bold')
        elif v:ax.text(j,bottom[j]+v/2,str(v),ha='center',va='center',color='white' if count in [4,8] else '#222',fontweight='bold')
    bottom+=vals
ax.set_xticks([0,1],['Inventory: 2 GPUs','Inventory: 8 GPUs'])
ax.set_ylabel('Hosts appearing in job details')
ax.set_title('Published inventory capacity conflicts with job allocation records')
ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.13),fontsize=9)
fig.tight_layout();fig.savefig(O/'fig1_inventory_audit.png',dpi=180,bbox_inches='tight');plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(11,4.4))
for ax,eps in zip(axs,[0.,.1]):
    for rep,col in enumerate(colors):
        s=primary[(primary.replicate==rep)&(primary.gpus_requested==2)&(primary.epsilon==eps)
                  &(primary.calibration=='rolling14d')&(primary.costs=='heterogeneous')].sort_values('lag_min')
        ax.plot(s.lag_min,100*s.saving_vs_query,marker='o',lw=2,color=col,label=f'Layout {rep+1}')
    ax.axhline(0,color='#777',lw=.7);ax.set_xscale('log');ax.set_xticks([1,5,15,60],['1','5','15','60'])
    ax.set_xlabel('Age of refresh information (minutes; log scale)');ax.set_ylabel('PSA1 saving vs query-only (%)')
    ax.set_title(f'2-GPU predicate, ε={eps:.2f}');ax.grid(axis='y',alpha=.2)
axs[1].legend(frameon=False)
fig.suptitle('Measured GPU/host-memory predicate; Q=(4,2,1,3), R=0.05Q',y=1.02)
fig.tight_layout();fig.savefig(O/'fig2_cost_vs_staleness.png',dpi=180,bbox_inches='tight');plt.close(fig)

s=primary[(primary.gpus_requested==4)&(primary.epsilon==.1)&(primary.lag_min==1)
          &(primary.calibration=='rolling14d')&(primary.costs=='equal')].sort_values('replicate')
fig,ax=plt.subplots(figsize=(8.8,4.6))
xx=np.arange(3)
ax.bar(xx,100*s.conditional_miss,color=colors,width=.55,alpha=.85)
for j,(_,row) in enumerate(s.iterrows()):
    ax.plot([j,j],[100*row.risk_block95_low,100*row.risk_block95_high],color='#222',lw=1.6)
    ax.annotate(f'{int(row.miss_n)} / {int(row.empty_stop_n)}', (j,100*row.conditional_miss),xytext=(0,7),textcoords='offset points',ha='center')
ax.axhline(10,color='#ac3939',ls='--',lw=1.5,label='Model budget ε=10%')
ax.set_xticks(xx,['Layout 1','Layout 2','Layout 3']);ax.set_ylabel('Misses / no-result stops (%)')
ax.set_title('Empirical risk can exceed the model budget')
ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
ax.text(0,-.19,'Intervals: descriptive 7-day moving-block bootstrap, 1,000 replicates.\nMeasured-state predicates and incomplete public jobs do not certify scheduler availability.',transform=ax.transAxes,fontsize=9)
fig.tight_layout();fig.savefig(O/'fig3_empirical_risk.png',dpi=180,bbox_inches='tight');plt.close(fig)


def table(frame,cols,labels,formats):
    lines=['| '+' | '.join(labels)+' |','| '+' | '.join(['---']*len(cols))+' |']
    for _,row in frame.iterrows():
        vals=[]
        for c,fmt in zip(cols,formats):
            val=row[c]
            vals.append('nem becsülhető' if pd.isna(val) else fmt(val) if callable(fmt) else format(val,fmt))
        lines.append('| '+' | '.join(vals)+' |')
    return '\n'.join(lines)


zero=primary[(primary.epsilon==0)&(primary.calibration=='rolling14d')&(primary.costs=='heterogeneous')&(primary.lag_min==1)].copy()
zero['layout']=zero.replicate+1;zero['saving_pct']=100*zero.saving_vs_query
zero_table=table(zero,['layout','gpus_requested','n','mean_cost','saving_pct','miss_n'],
 ['Elrendezés','Kért GPU/gép','Tesztpont','PSA1-költség','Megtakarítás a query-only-hoz képest','Elmulasztás'],
 [lambda x:str(int(x)),lambda x:str(int(x)),lambda x:str(int(x)),'.4f',lambda x:f'{x:.2f}%',lambda x:str(int(x))])
example=d[(d.replicate==0)&(d.gpus_requested==2)&(d['mode']=='telemetry')&(d.lag_min==1)
          &(d.calibration=='rolling14d')&(d.costs=='heterogeneous')&(d.refresh_ratio==.05)&(d.epsilon==.1)].copy()
example['saving_pct']=100*example.saving_vs_query;example['risk_pct']=100*example.conditional_miss
example_table=table(example,['method','mean_cost','saving_pct','miss_n','empty_stop_n','risk_pct'],
 ['Módszer','Realizált költség','Megtakarítás','Elmulasztás','Eredmény nélküli lezárás','Feltételes arány'],
 ['','.4f',lambda x:f'{x:.2f}%',lambda x:str(int(x)),lambda x:str(int(x)),lambda x:f'{x:.2f}%'])
prev=pd.read_csv(R/'state_prevalence.csv')
prev=prev[prev.phase=='test'].groupby(['replicate','gpus_requested','mode']).agg(n=('n','first'),empty=('all_domains_empty','first')).reset_index()
prev['layout']=prev.replicate+1;prev['empty_pct']=100*prev['empty']
prev_table=table(prev,['layout','gpus_requested','mode','n','empty_pct'],
 ['Elrendezés','Kért GPU/gép','Predikátum','Közös tesztpont','Egyik domain sem alkalmas'],
 [lambda x:str(int(x)),lambda x:str(int(x)),'',lambda x:str(int(x)),lambda x:f'{x:.2f}%'])
p=d.pivot(index='scenario',columns='method',values='mean_cost')
mp=d.pivot(index='scenario',columns='method',values='model_expected_cost')
better=int((p.psa1<p.h1-1e-10).sum());worse=int((p.psa1>p.h1+1e-10).sum());same=int(((p.psa1-p.exact).abs()<1e-10).sum())
avgmax=100*(mp.psa1/mp.exact.replace(0,np.nan)-1).max()
riskcase=primary[(primary.replicate==1)&(primary.gpus_requested==4)&(primary.costs=='equal')&(primary.lag_min==1)&(primary.calibration=='rolling14d')&(primary.epsilon==.1)].iloc[0]
viol=int(((psa.epsilon==.1)&(psa.conditional_miss>.1+1e-12)).sum())
fallback=sum(bool(x['static_fallback']) for x in fitted)
mean_known=100*audit['mean_selected_test_coverage']
note=rf'''# Microsoft Philly: a foglalási naplóval párosított részminta részletei

2026. november 16. — első, reprodukálható változat

Ez a részletes mellékanyag a foglalási bizonytalanságot is kizáró közös részmintáról szól. A teljes ismert telemetriára készült elsődleges kontroll és az összesített értékelés a [főjelentésben](NOMS2027_Philly_audit_replay_v1_hu.md) található.

**A teljes eredeti archívum beszerzése, az adatminőségi audit és a négy-domaines algoritmus-összehasonlítás elkészült.** Összesen {meta['scenarios']} konfiguráció és hat módszer, azaz {len(d)} összesített eredménysor készült. A feldolgozásban talált leltárhibát nem javítottam feltételezett hardveradatokkal: a kísérlet a megfigyelt GPU-terhelési és hostmemória-feltételekre épül. A jobnapló külön, hiányos foglaltsági bizonyítékként szerepel.

## 1. Forrás és ténylegesen feldolgozott adatok

Hivatalos forrás: [Microsoft Research Philly traces](https://github.com/msr-fiddle/philly-traces), commit `29a1b87fa2d9ed80b83c9e3a37f3a88d382b031d`. A teljes eredeti `trace-data.tar.gz` mérete 1 055 988 361 bájt; SHA-256: `2037ccf63a725a02be718b0e1aeee6142c90b7f62d0d849295120e9f6ba13d2c`. A letöltött állomány mérete és ellenőrzőösszege egyezett a hivatalos Git LFS-bejegyzéssel.

A mérési adatokhoz tartozó alapcikk: M. Jeon és szerzőtársai, *Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads*, USENIX ATC 2019, 947–960. [Cikk és hivatkozás](https://www.usenix.org/conference/atc19/presentation/jeon). A nyilvános állomány a munkaterhelés részhalmaza, nem a teljes üzemeltetési állapotnapló.

Tényleges feldolgozás:

- {audit['gpu']['rows']:,} GPU-telemetriarekord, {audit['gpu']['hosts']} gépazonosító.
- {audit['memory']['rows']:,} hostmemória-rekord, {audit['memory']['hosts']} gépazonosító.
- {audit['jobs']['jobs']:,} feladat; {audit['jobs']['attempts']:,} futtatási kísérlet.
- Az elsődleges időrács 2017. október 3-tól december 16. kezdetéig tart. A nyers telemetria utolsó elérhető pontja december 15-én van; a rács későbbi hiányzó pontjait nem töltöttem ki.
- A kísérleti részhalmaz 24 gép nyolc-nyolc megfigyelt GPU-csatornáját használja. A két-/nyolc-GPU-s leltárcímkéket nem használtam a kapacitás meghatározására.

Az anonim gépazonosítók teljes halmazából rögzített hash-sorrendben 96 jelöltet vettem. A 24 gépet kizárólag a november 1. előtti, mind a nyolc GPU-csatornára és a hostmemóriára meglévő mérési lefedettség alapján választottam ki. Az állapotok kedvezősége és az algoritmusok későbbi teljesítménye nem szerepelt a kiválasztásban. A kiválasztott gépeken az átlagos teljes telemetria-lefedettség a tesztidőszakban {mean_known:.2f}%.

## 2. Az audit három fontos megállapítása

**A gépleltár kapacitáscímkéi nem megbízhatók.** Például a `m29` a leltárban két GPU-s gép, miközben a telemetriában nyolc GPU-csatornája, a jobnaplóban pedig nyolcig terjedő GPU-kiosztása szerepel. A jobnaplóban megjelenő {integrity['job_hosts']} gépből {integrity['hosts_with_logged_gpu_index_exceeding_inventory']} esetben található a leltár szerinti darabszámot meghaladó GPU-index. Ez {integrity['allocation_details_exceeding_inventory']:,} gépszintű kiosztási részrekordot érint. Ugyanezt a problémát a hivatalos adattár [nyitott 8. hibajegye](https://github.com/msr-fiddle/philly-traces/issues/8) is leírja. A gyökérokot az adatközlő megerősítése nélkül nem állítjuk; különösen nem feltételezünk önkényes azonosító-átszámozást.

![Leltár és kiosztott GPU-indexek](fig1_inventory_audit.png)

**Az időzónákat külön kellett rendezni.** A telemetria PDT/PST jelölése explicit. A jobnapló időpontjaiban nincs időzóna. A YARN application ID-k beágyazott ResourceManager-kezdőidejét összevetettem az első beküldésekkel: több `_0001` feladat 9–12 másodperccel követi a Unix-epochként értelmezett kezdőidőt, ami az UTC-értelmezést támogatja. A [Hadoop API](https://hadoop.apache.org/docs/r2.7.7/api/org/apache/hadoop/yarn/api/records/ApplicationId.html) dokumentálja az application ID ezen részének jelentését. **Az UTC a nyers adatokból levont, dokumentált következtetés, nem a Philly README-ben megadott időzóna.** Az eltolási érzékenységvizsgálat is a nulla UTC-eltolásnál mutatja a legerősebb terhelés–foglaltság elkülönülést. A bizonyítékokat két CSV őrzi; a végső feldolgozás UTC jobidőket használ.

**A foglalási napló hiányos.** {audit['jobs']['missing_start']} kísérletnél nem értelmezhető a kezdés, {audit['jobs']['missing_end']} esetben a befejezés, {audit['jobs']['reversed_intervals']} esetben a két végpont sorrendje hibás. A lezárt, érvényes intervallumok megerősített *naplózott* foglaltságot adnak; a hiányos/hibás végpontok bizonytalan időszakot jelölnek. Ez utóbbiakat nem tekintjük sem biztosan foglaltnak, sem szabadnak. A jól körülhatárolt GPU-intervallumok között {integrity['overlapping_bounded_gpu_intervals']} átfedő rekord is van; a foglaltsági feltétel ezek unióját használja. Az alacsony GPU-terhelés önmagában nem jelent kiosztható eszközt, és a nyilvános naplóból hiányzó feladat továbbra is jelen lehet a rendszerben.

## 3. A ténylegesen vizsgált feltételek

A két intent egyetlen gépen legalább két, illetve négy olyan GPU-t kíván, amelynek mért terhelése legfeljebb 20%; a host szabad memóriájának aránya legalább 5%. A szabad hostmemória aránya dimenziótlan, ezért nem kell a dokumentálatlan abszolút memóriaegységet feltételezni. **Dinamikus szabad GPU-VRAM-ot nem mérünk.**

Két állapotdefiníció szerepel:

1. `telemetry`: a fenti megfigyelt terhelési és hostmemória-feltétel. Ez alkalmas jelölteket ír le, nem scheduler által igazolt elhelyezhetőséget.
2. `logged_filter`: ugyanez, de a lezárt jobintervallum szerint foglalt GPU nem számítható be. A hiányos jobintervallummal érintett kártya státusza bizonytalan. Ha ebből az intent igazságértéke nem dönthető el, az adott állapot hiányzó marad.

A hiányzó méréseket és bizonytalan állapotokat nem helyettesítem sikertelenséggel. Egy kétgépes domain akkor biztosan pozitív, ha bármelyik gépe igazoltan teljesíti a megfigyelési feltételt; biztosan negatív csak akkor, ha mindkét gép feltétele eldönthetően hamis.

Mindhárom elrendezés négy virtuális domainből áll, domainenként két megfigyelt géppel. A 24 gépet rögzített, 2026-os véletlenmagú permutáció osztja három, egymástól diszjunkt nyolcgépes csoportba. **Ez modellezett csoportosítás, nem valódi federáció vagy ismert racktopológia.** A három elrendezés ugyanannak a klaszternek az időben korrelált részhalmaza, nem három független infrastruktúramérés.

November 1. előtt tanítunk, utána egyórás rést hagyunk. A refresh a domain 1, 5, 15 vagy 60 perccel korábbi állapotjelét fedi fel. Egy elrendezésen belül **mind a négy késleltetés, mindkét intent, mindkét állapotdefiníció és minden algoritmus ugyanazokat a tesztidőpontokat használja**. A bizonytalan foglalási adatok miatt ez jóval kevesebb a puszta telemetria-lefedettségből várható pontszámnál; a teljeseset-elemzés szelekciós hatását nem tekintjük elhanyagolhatónak.

Az állapotok változatossága a közös teszthalmazon:

{prev_table}

A `telemetry` esetben már gyakori, hogy egyik domain sem teljesíti a feltételt; ez az Eagle-pilotnál változatosabb keresési helyzetet ad. A foglalási szűrő után viszont több esetben szinte mindig üres a jelölthalmaz. Az ottani nagy költségcsökkenés gyakran pusztán a keresés korai elhagyását jelenti, ezért abból önmagában nem állítunk algoritmikus előnyt.

## 4. Döntési modell, baseline-ok és kontrollok

A QUERY/REFRESH/SKIP/STOP döntési modell és az eredeti, auditált PSA1-kód változatlan szemantikával fut. Hat módszer: query-only, risk-aware query, screening-greedy, H1-rollout, PSA1-rollout, egzakt Bellman-DP. Minden illesztett független domainmodellben mind a 256 lehetséges igazság–refresh kimenetet enumeráltuk, és a DP-, valamint PSA1-költséget a rekurzív értékfüggvénnyel is összevetettük.

A QUERY-költség vagy (1,1,1,1), vagy (4,2,1,3). A REFRESH a megfelelő QUERY-költség 5%-a; az első elrendezés négy-GPU-s eseteiben 20%-os érzékenységvizsgálat is készült. **Minden kommunikációs költség modellezett, normalizált érték; nincs mért hálózati RTT.** A keresés alatt az adott perc megfigyelési predikátuma rögzített.

Két kalibráció: statikus, teszt előtti modell; illetve heti frissítés az előző 14 napból. Ha a közös, ismert múltban nincs legalább 100 páros megfigyelés, az adaptív változat kifejezetten a teszt előtti statikus modellre tér vissza. Ez {fallback}/{len(fitted)} paraméter–időszak kombinációt érintett; a jelző a modellek között szerepel. Sem az újrabecslés, sem a visszatérés nem használ jövőbeli tesztadatot. A gyakoriságokat Laplace-simítás egészíti ki; ezek pontbecslések, nem kockázati felső korlátok.

## 5. Költségelőny megengedett kihagyás nélkül

Az ε=0 kontrollban sem SKIP, sem feloldatlan domain melletti kockázati STOP nem engedhető meg. A refresh az ellenőrzések sorrendjét javíthatja; minden ténylegesen megmaradt bizonytalanságot QUERY old fel. Az alábbi eredmény ezért nem a találatok kockázatos elengedéséből származik.

`telemetry`, egyperces refresh, adaptív kalibráció, Q=(4,2,1,3), R=0,05Q, ε=0:

{zero_table}

A két-GPU-s esetek 2,41–21,83% megtakarítást adtak a query-only sorrendhez képest, a megfigyelt predikátumokra nulla elmulasztással. Ebben a kontrollban a H1 és a PSA1 azonos volt; a nyereség az információval javított sorrend értékét mutatja, nem a PSA1 egyedi előnyét. A 21,83%-os eset leíró hét napos blokk-bootstrap intervalluma 5,27–31,26%; a három elrendezés eltérő eredménye és a modellezett költségprofil korlátozza az általánosítást. Az ε=0 eredmény az offline, pontos QUERY-val feloldott mérési predikátumokra vonatkozik, nem új feladatok valós erőforrásfoglalására.

![Költség és információelavulás](fig2_cost_vs_staleness.png)

## 6. PSA1 a korábbi heurisztikákhoz képest

Az első elrendezés két-GPU-s `telemetry` esete, egyperces refresh, adaptív kalibráció, Q=(4,2,1,3), R=0,05Q, ε=0,10:

{example_table}

Itt a PSA1 realizált költsége 2,3015, míg a H1-é 3,0987, a screening-greedyé 2,7823, az egzakt DP-é 2,2581. Ez már mutat PSA1-többletet az Eagle kiemelt esetével szemben. A költség és a feltételes elmulasztási arány együtt értékelendő: a módszerek eltérő találatokat hagyhatnak el, és a 10%-os korlát a tanult modellre, nem igazoltan a későbbi trace-re érvényes.

A teljes, {meta['scenarios']} konfigurációs rácsban a PSA1 {better} esetben olcsóbb, {worse} esetben drágább volt a H1-nél, a többi esetben egyezett a realizált költség. A PSA1 és a DP {same} konfigurációban adott azonos realizált költséget. A rács ismétlődő megfigyeléseket és kapcsolódó paraméterváltozatokat tartalmaz; ezek a darabszámok nem független statisztikai próbák.

**Lényeges kedvezőtlen esetet is találtunk.** Egy tanult, független négy-domaines modellben a DP várható költsége {counter['exact_expected_cost']:.6f}, a PSA1-é {counter['psa_expected_cost']:.6f}: az eltérés {100*counter['relative_gap']:.2f}%. Ez egyetlen kalibrációs időszak modellje; a konfiguráción belüli időszakokkal súlyozott, legnagyobb modell szerinti átlagos gap {avgmax:.2f}%. Az egyedi nagy eltérés nem azonos a teljes trace átlagával.

A gyorsított és közvetlen PSA1-continuation értéke a kedvezőtlen modell {counter['fast_vs_direct_continuation_checked_admissible_states']} megengedett állapotában egyezett. A modell szerinti rollout-költségkorlát is teljesült; a nagy gap tehát nem cáfolja ezt a korlátot, és a vizsgálat nem mutatott gyorsítási implementációs hibát. **Általános közel-optimalitási állítást ebből az eredményből nem lehet tenni.** A pontos paraméterek, gyökérállapotbeli döntések és értékek az `algorithmic_counterexample.json` fájlban szerepelnek. A közös REFRESH-döntések értékének elemzése indokolt következő algoritmikus feladat; ez itt még vizsgálati hipotézis.

## 7. A modell szerinti és a trace-en realizált kockázat

A feltételes elmulasztási arány nevezője az eredmény nélküli kereséslezárások száma, nem az összes kérés. Az utóbbi nevezővel számított arány külön oszlop a CSV-ben. A nulla nevező nincs nullaként kezelve.

A második elrendezés négy-GPU-s `telemetry` esetében, azonos QUERY-költségekkel, egyperces refresh-sel és adaptív kalibrációval a PSA1 **{int(riskcase.miss_n)}/{int(riskcase.empty_stop_n)} = {100*riskcase.conditional_miss:.2f}%** feltételes elmulasztási arányt adott ε=10% mellett. A lezárások {int(riskcase.stopping_days)} különböző napra esnek. A hét napos mozgóblokk-bootstrap leíró 95%-os intervalluma {100*riskcase.risk_block95_low:.2f}–{100*riskcase.risk_block95_high:.2f}%. Ez sokkal több negatív esemény, mint az Eagle-pilotban, és konkrétan megmutatja, hogy az alkalmazott pontbecsléses modell korlátja nem vihető át automatikusan a mérési idősorra.

![Tényleges feltételes kockázat](fig3_empirical_risk.png)

A 224 pozitív kockázati keretű PSA1-konfiguráció közül {viol} esetben volt a pontbecslés ε felett. **Ez nem az elméleti, helyes modellre kimondott feltételes korlát cáfolata.** A becslési hiba, a terhelés időbeli változása, a domainek függősége, valamint a kiválasztott közös mérési pontok szelekciós hatása egyaránt befolyásolhatja a tapasztalt kockázatot. A jelen kísérlet ezeket még nem választja szét okságilag.

A blokk-bootstrap figyelembe vesz egy választott időbeli összefüggési skálát, de nem ad állomásfüggetlen assurance-bizonyítványt; a hét napos blokkhossz érzékenységi választás. Nulla megfigyelt elmulasztásnál az empirikus bootstrap-intervallum elfajulhat. A kockázati mezők figyelmeztető státusza ezért megmarad. Minden módszer minden STOP-ja a saját modelljének megengedett kockázatát tartotta be az enumerációs ellenőrzésben.

## 8. Mit használhatunk a cikkben?

Az Eagle-nél erősebb kiegészítő eredményt kaptunk a friss információ értékére: több valóban negatív keresési helyzet, finomabb időfelbontás és ε=0 kontroll áll rendelkezésre. A választott kockázati keret mellett a PSA1 több esetben a H1-hez és a screening-greedyhez képest is javít. Ezzel együtt a módszerek és az egzakt DP közötti különbségeket nem szabad kizárólag a legkedvezőbb példával bemutatni.

Az assurance-állítás szempontjából a pozitív fő tanulság **nem a 10%-os garancia igazolása**, hanem az, hogy a belief-kalibráció és a modellhelyesség ellenőrzése elengedhetetlen. A leltárhiba miatt nem következtetünk a kártyák névleges VRAM-jára; a hiányos jobnapló miatt nem állítunk teljes scheduler-foglaltságot vagy valódi federált edge-validációt.

A következő érdemi munka a kimutatott kedvezőtlen PSA1-modell és a kockázatátvitel hibaforrásainak célzott elemzése. A jelen anyag reprodukálható telemetriaalapú esettanulmányként használható; a fő algoritmikus garanciák bizonyítéka továbbra is az analitikus eredmény és a pontosan körülhatárolt modellvizsgálat.

## 9. Reprodukálhatóság

A csomag tartalmazza a megfigyelt 24 gépes részhalmazt, a forrás- és integritási auditot, a 448 konfiguráció eredményeit, a közös tesztidőpontok állapotexportját, a napi statisztikákat, a tanult modelleket, a kedvezőtlen ellenpéldát és a futtatási kódot. A nagy eredeti CSV-k újra előállíthatók az ellenőrzőösszeggel rögzített eredeti archívumból; a részhalmazzal az algoritmusok újrafuttatásához nincs szükség ismételt teljes letöltésre.

Nyolc célzott ellenőrzés sikeresen lefutott: időzónák és DST, UTC epoch-példa, félig nyílt foglalási intervallumok, foglalt de alacsony terhelésű GPU, hiányzó mérés, bizonytalan feladatvég, közös késleltetési rács és négy-domaines Bellman-enumeráció. A részhalmaz float32 formában tárolja a mért numerikus mezőket; nincs interpoláció vagy munkaterhelés-generálás. A véletlen magok rögzítettek. Az eredeti adatok CC BY 4.0 licencének hivatkozását és a kötelező bibliográfiai adatokat a README és a `references.bib` tartalmazza.
'''
(O/'Philly_joint_subset_details_hu.md').write_text(note,encoding='utf-8')
zero.to_csv(R/'headline_zero_risk_control.csv',index=False)
example.to_csv(R/'headline_psa_comparison.csv',index=False)
print(O/'Philly_joint_subset_details_hu.md')
