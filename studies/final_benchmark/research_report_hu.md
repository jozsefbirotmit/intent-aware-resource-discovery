# IA-OSR: algoritmuscsalád és független szintetikus benchmark v1

2026. november 17. A `NOMS2027_Algorithm_Iteration_v2.zip` implementációját változatlan döntési szabályokkal használtuk tovább. A csomag a korábbi algoritmuskódot, a friss benchmarkot, teljes paraméterezést, eredményeket és új IEEE-szakasztervet tartalmazza. Az eredeti kézirat nem változott.

## 1. Modell és konfigurációk

Egy rögzített intent és discovery-epizód, független statikus Bernoulli domain-feasibility, ismert priorok és bináris megfigyelési csatornák, domainenként legfeljebb egy REFRESH, egzakt QUERY. A STOP sikertelen discovery esetén akkor admisszibilis, ha az összes megoldatlan domain (a kihagyottak is) együttes posterior kockázata legfeljebb epsilon. Ez a feltételes, megfigyelt történetre vonatkozó missed-resource kockázat, nem szolgáltatásmegbízhatóság vagy kalibrációhibára robusztus garancia.

A nulla költségű, információmentes, irrevokábilis SKIP az egzakt optimumhoz nem szükséges: a skip elhalasztása a STOP-ig bármely policy információs lépéseit reprodukálja, ugyanazon residual risk mellett. A referencia DP ezért a négy domainállapotot használja (frissítetlen, pozitív, negatív, sikertelen query). A PSA1 continuation továbbra is a skip-annotált állapottéren működik. Egy már elkövetett SKIP-et a tail nem von vissza. Az egzakt no-refresh referencia a 2^N maszk-DP; ha a risk nagyobb epsilonnál, a min_i q_i+(1-p_i)V(A\i) rekurziót használja. A kód `benchmark.py`.

H0(s) a PSA1 megvalósítható continuation-költség: legfeljebb egy upfront skip, legfeljebb egy screening modul (refresh, pozitívnál query, negatívnál admisszibilis skip), a többi aktív domain query-je a rögzített modulok c/s sorrendjében. Az egyes tervek folytatása a következő állapotban reprodukálható: a felhasznált upfront skip eltűnik a jelölttervből; a screening pozitív ágán marad a query, negatív ágán egy aktuálisan admisszibilis upfront skip; a már megkezdett query-only folytatás is megengedett. Korai STOP esetén zéró a folytatás. Ezért T H0 <= H0 (egzakt aritmetika mellett). H0^K(s)=V*(s), ha m(s)<=K, egyébként H0(s). Itt V* az aktuális committed SKIP állapot optimuma; m nem nő, így ugyanaz a supersolution tulajdonság fennmarad. T monotonitásából H2<=H1<=H0^K. U_d(s,a)=c(a)+E[H_{d-1}(s')], H_d=min_a U_d. A selective változat legfeljebb egy-egy legjobb Q/R/SKIP score-t mélyít U2-re. Más jelöltek U1-en maradnak; a minimum M<=H1. Véges aciklikus állapottérben visszafelé indukcióval V^pi<=M (az utód V^pi<=H1<=H0^K, ezért a kétféle score felső korlát). Ez nem monotóniatétel a realizált policyk költségeire és nem worst-case approximation ratio.

| Munkanév | K | Lookahead |
|---|---:|---|
| PSA1 | 0 | 1 |
| PSA1-E | 8 | 1 |
| PSA1-S | 8 | selective, legfeljebb 3 gyökérakció mélyítése |
| PSA1-A | 8 | multi-mechanism, teljes 2-lépéses mélyítés |

A K=8 esetben a korábbi 3-lépéses ág nem aktiválódhat nem egzakt állapotban. Az adaptív trigger a v2-ből változatlan: risk gate (két egyenként kihagyható domain additív riskje 12%-os környezetben a fennmaradó budgethez, legalább két refresh-elhető), near-best refresh (3% score-marzs), P4 query-only ordering gate (a net ordering-only VOI pozitív). A risk gate az előző feltételek mellett 3<=m<=16; a többi trigger is legfeljebb 16 aktív domainnél működik. Nem optimalizált küszöbök. Az exact tail LRU tábla-cache névleges korlátja 32 MiB, a többi Python cache nem korlátozott. A memoizálás egy epizódon át megmarad, epizódok között ürítjük. Az egzakt DP O(N 4^N) idő és O(4^N) értéktábla-memória; a PSA1 base O(N^2 log N) és az egy lépés O(N^3 log N), a fix mélységű teljes kétlépéses változat általános felső becslése O(N^4 log N), a legfeljebb három akciót tovább mélyítő változaté O(N^3 log N), K rögzített. Utóbbiak a döntési kiértékelésre vonatkoznak, nem a teljes policyfára. A 16-os cap miatt a tényleges nagy-N gyökér azonos egylépéses számítás.

## 2. Független tesztprotokoll

90 friss modell: N=8,10,12, egyenként 10 példával mindhárom családból (IID, extended skip boundary, informatív refresh). Az eredeti `iteration.generate` generátorok; az IID p~U[.05,.85], q log-uniform 10^U[-.2,1.1], r/q=10^U[-1.4,.15], alpha,beta~U[.65,.995], epsilon a {.01,.03,.05,.1,.2,.3} halmazból. A boundary 4-domaines, együttes SKIP-küszöb körüli példához további független paraméterű domaineket csatol; a probe család magasabb p és olcsóbb refresh, szigorú eps=.005, nem garantált P4. A teljes paraméterezés a JSONL-ben. A 73100000+n*10000+fi*1000+j seed eltér a korábbiakétól. K=8, gate-ek és cache változatlanok. Az egzakt várható policy-költség teljes döntésfa-integrálással (tail esetén DP-vel), az optimum no-SKIP DP-vel, a no-refresh referencia maszkkal. Minden 360 algoritmusértékelés az optimum és saját felső korlát közé esett 1e-7 abszolút toleranciával. N=8 esetén külön kell kezelni (a hibridek teljes egzakt megoldások). Az elsődleges értékelés N=10–12, 60 új modell; rétegzett, páros bootstrap 10 000 ismétléssel (seed 20261117), 6 méret-család rétegen belüli mintavételezés. A CI a mesterséges, kiegyensúlyozott benchmarkra, nem valós hálózati populációra vonatkozik.

## 3. Minőség és információérték

| N | PSA1 gap % | E | S | A |
|---:|---:|---:|---:|---:|
|8|5.711|0|0|0|
|10|4.966|2.835|2.347|2.107|
|12|6.317|5.983|3.984|3.739|
|10–12|5.642|4.409|3.165|2.923|

N=10–12, átlagos gap családonként: IID 0.771, 0.631, 0.142, 0.066%; boundary 12.587, 8.974, 6.557, 6.299%; informative refresh 3.567, 3.623, 2.797, 2.403%. Max gap: 25.011, 23.571, 17.391, 17.391%. A PSA1-S a PSA1-E-hez képest 40 esetben javított, 7-ben rontott, 13 egyezés; a PSA1-A a PSA1-S-hez képest 26 javulás, 11 romlás, 23 egyezés. Ez illeszkedik a felső korlát és policy-költség közötti különbséghez.

Páros átlagos gap-csökkenés (százalékpont): PSA1->E 1.232 [0.772,1.753]; E->S 1.244 [0.535,2.028]; E->A 1.486 [0.797,2.227]; S->A 0.242 [-0.170,0.658]. Utóbbi nem igazolja sem a különbséget, sem az ekvivalenciát.

Az egzakt no-refresh referencia ellen N=10–12 átlagos egyedi relatív megtakarítás (100*(1-C/C_noR), majd átlag): 16.182, 17.056, 18.024, 18.204%. Ezek a mesterséges családokban mért értékek, nem hálózati előrejelzések; egyetlen esetben sem volt negatív megtakarítás ezen a mintán, de ez nem általános garancia. Az optimális no-refresh policy kockázatkorlátja ugyanaz.

## 4. Online trajektória és első döntés

Modellenként két független Bernoulli feasibility vektor és feltételes megfigyelési vektor, ugyanazon világok a négy policyhoz. 720 policytrajektória, a 10–12 domaines részben 120/algoritmus. Minden epizód új cache, a JIT előre bemelegítve; a teljes fa-integrálástól elkülönített mérés. Az aktív döntések idejéből kivettük a végső sikertelen STOP ellenőrzést. Nincs hálózati idő, és egyetlen folyamatban fut a kísérlet; a mérések hardver- és környezetfüggők (AMD EPYC 9V74, Python/Numba; verziók environment.txt). Az online decision percentilisek az eseményeket egyenlő súllyal aggregálják, nem az epizódokat. A trajektóriákból nem becsüljük a várható költséget. Az egyedi realizált STOP posterior-kockázat ellenőrzése 720 esetben teljesült (numerikus 1e-12 tolerancia); mindössze 21 sikertelen befejezés volt, így az empirikus conditional miss-rate becslésre nem elég a mintaszám.

|Algoritmus|Medián gyökér ms|Aktív döntés medián ms|Aktív döntés P95 ms|Teljes trajektória medián ms|
|---|---:|---:|---:|---:|
|PSA1|1.165|0.682|3.296|3.010|
|E|1.248|0.659|18.354|3.904|
|S|17.132|2.347|41.622|34.097|
|A|84.157|9.638|140.086|110.777|

A tail cold build időt okozhat későbbi döntéseknél, míg a cached lookup olcsó. A teljes memóriát nem profilozzuk; peak_tail_table_bytes kizárólag az LRU tábláira vonatkozik.

## 5. Skálázás N=20–100

Méretenként 3 új IID és 3 low-feasibility stress instance. Utóbbi: p~U[.018,.026], q~U[4,12], r/q=.2, alpha=beta=.95, eps=.1; számos kihagyható domain és screen-jelölt. Csak a gyökérdöntést futtattuk PSA1-S-sel; m>16 esetén mind a négy policy ugyanarra az egy lépésre redukálódik. Nem igazolunk nagy-N optimalitást vagy teljes epizódidőt. Medián ms (IID/stress): 20: 10.55/20.24; 40: 36.59/157.00; 60:230.41/617.74; 80:212.32/1310.05; 100:1190.51/2643.81. A három adatpontból vett tartományt mutatja a skálázásábra, nem CI. Az IID idő nem monoton, a screen-admisszibilitás is befolyásolja a munkát. 100 stressz-domain esetén a több mint két másodperc korlátozza a time-critical felhasználhatóságot.

## 6. Publikációs döntés

A PSA1-S javasolt központi konfiguráció a kisebb/közepes modellekre, PSA1-E a számításkímélő referencia-konfiguráció, PSA1-A a drágább kontroll; N<=8 mellett az egzakt solver közvetlenül használható. Nem érdemes újabb heurisztikákat gyártani a mostani cikkhez. A konvergáló eredmény két konkrét fejlesztés hatását választja szét (tail és szelektív refinement). A 3600 mechanizmus-szintű vizsgálat, 4860 faktoros becslés és kalibrációs stressz továbbra is fontos; az Alibaba-pilot kiegészítő. A papírban a módszer erős oldala az ellenőrzött posterior risk és információszerzési költség együttes kezelése; a nagy-N minőség és stale/miscalibrated beliefs továbbra is korlát. A K és gate-ek nem optimálisnak bizonyítottak; a 60-as alminta kiegyensúlyozott stresszprobléma-minta, nem reprezentatív; a teljes adaptívhoz képest nem állítunk ekvivalenciát.
