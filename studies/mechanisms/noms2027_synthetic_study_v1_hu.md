# NOMS 2027 — Kontrollált szintetikus mechanizmusvizsgálat (v1)

**Adatforrás:** kizárólag szintetikus, modellen belül generált esetek. **Nem Alibaba-trace.**

## 1. Cél és módszer

A vizsgálat célja nem kizárólag egy átlagos nyereség kimutatása, hanem a PSA1 erősségeit és korlátait létrehozó mechanizmusok izolálása: P3 screening, P4 ordering-only refresh, sikertelen lekérdezés után halasztott refresh, SKIP-admissibility határ, valamint egy véletlen kontrollcsoport. A belső alapmodell: független domainek; egy pontos QUERY; legfeljebb egy bináris REFRESH domainenként; visszavonhatatlan SKIP; STOP ha a megmaradó kihagyási risk <= epsilon.

Közös algoritmusok: `HQ` (SKIP-et nem használó optimális fix query-sorrend), `greedy_skip` (heurisztikus risk-budget SKIP + query), **`no_refresh_opt` (egzakt optimális QUERY+SKIP REFRESH nélkül)**, H1 IA-OSR, PSA1, exponenciális skip-closed és egzakt Bellman DP. **Minden költség egzakt modellbeli várható költség**, nem véges replikációs akciótrajektória átlaga.

Öt seed, seed-enként 120 draw/család; a SKIP-küszöbnél mindegyik draw két epsilon-változat: összesen 3600 értékelt instance, amelyből 600+600 egymáshoz párosított. Az átlagos megtakarítás `100*(mean(query_only)-mean(PSA1))/mean(query_only)`. A zárójelezett CI-k 1500-ismétléses, instance bootstrap 95%-os intervallumai, nem a döntési folyamat futásának Monte Carlo-zaja.

## 2. Fő eredmények

| Család | N | PSA1 a query-onlyhoz képest | 95% bootstrap CI | Átlagos PSA1 gap az optimumhoz | PSA1 egzakt arány |
|---|---:|---:|---:|---:|---:|
| IID véletlen | 600 | 8.97% | [8.02%, 10.01%] | 0.46% | 74.0% |
| P4 ordering probe | 600 | 4.56% | [4.36%, 4.75%] | 0.00% | 100.0% |
| P3 screening | 600 | 17.28% | [16.86%, 17.71%] | 0.00% | 100.0% |
| Deferred refresh | 600 | 39.61% | [39.02%, 40.23%] | 0.00% | 100.0% |
| SKIP-küszöb (alatta) | 600 | 59.62% | [58.93%, 60.25%] | 3.95% | 45.3% |
| SKIP-küszöb (felette) | 600 | 64.09% | [63.47%, 64.67%] | 1.15% | 75.0% |

### Mechanizmusértelmezés

- **P4:** a generator garantálja az ordering crossingot és a két, szigorúan pozitív P4 margin-t. A 600/600 esetben az ordering probe olcsóbb volt mindkét fix Q-sorrendnél; PSA1=Bellman. Átlagos elméleti fix-sorrend-nyereség 0,5054 költségegység. Az eredmény **konstruált mechanizmusellenőrzés**, nem általános exactness-tétel.
- **P3:** a generátor minden esetben `L_i|j<q_i<U_i|j` tartományba helyezi a kétdomaines példát, biztosítja a negatív SKIP megengedettségét, és kizárja a másik domain olcsó refreshét. PSA1=Bellman 600/600. Az eredeti H1 is egy részükben eltérhet: ezen családban PSA1 átlagban 7,30%-kal olcsóbb H1-nél.
- **Deferred refresh:** mind a 600 esetben a PSA1 először `Q_j`-t választ és `Q_j` sikertelen ágában `R_i`-t, az egzakt optimumhoz hasonlóan. Az eredeti H1 is optimális: ez a dinamikus információszerzés előnyét, nem önmagában a PSA1 újdonságát mutatja.
- **SKIP-küszöb:** két olcsó posterior riskű, de drága domain együttes kihagyási riskje éppen a változtatott epsilon határon van. A 600 párnál a PSA1 költsége az epsilon megemelésekor **átlagosan 0.822 egységgel csökkent**, párosított 95% CI [0.773, 0.872]. Minden párban megváltozott a PSA1 expected cost (a seedek és paraméterezés szerint). Ez szándékosan határra hangolt stresszteszt.

### Fontos negatív eredmény

A PSA1 **nem uniformly near-optimal**. A SKIP-határ alatti 600 tesztben a PSA1 átlagos optimalitási eltérése 3.95%, de a legnagyobb **46.07%**. A skip-closed rollout a boundary alatt 166 esetben jobb PSA1-nél. Az IID kontrollban PSA1 19/600 esetben rosszabb H1-nél, tehát a base-policy javulása nem jelenti automatikusan a realizált rollout-javulást.

## 3. Faktoriális érzékenység: 4860 eset

Azonos 180 darab `p,q`-alappélda szerepel mind a 27 (epsilon, alpha=beta, r/q) kombinációban; 4 független seed x 45 eset. A true feasibility és a query-költség változatlan, csak az információ beszerzésének paramétereit variáljuk. Az alábbi különösen erős kontroll azonos epsilon mellett az **optimális QUERY+SKIP policy**, amelyben REFRESH tilos. Ez expliciten elválasztja a refresh hozzáadott értékét a közvetlen SKIP megengedésének előnyétől. `epsilon=0.10`:

| alpha=beta | r/q=0.05 | r/q=0.20 | r/q=0.50 |
|---:|---:|---:|---:|
| 0.70 | 7.2% | 4.0% | 1.1% |
| 0.85 | 19.2% | 10.7% | 4.1% |
| 0.97 | 48.5% | 32.2% | 10.3% |

Az információ megszerzésének plusz értéke a jelen generátoron belül nő a refresh megbízhatóságával, és csökken a relatív refresh-költséggel. Ezek a trendek nem automatikus univerzális tételek, hanem a megadott faktoriális tartományban mért mintázatok. A teljes 27 cella eredménye és CI-ja a CSV-ben.

## 4. Modellhibás belief és a kihagyási kockázat

375 alapfeladat x 4 pozitív log-odds eltolás (összesen 1500 true-policy értékelés). A controller a nominal p és nominal alpha/beta alapján dönt, a **valódi** p viszont `logit(p_true)=logit(p_nominal)+delta`; a refresh érzékenysége/specificitása változatlan. Az értékelő a valós döntési fában egzakt módon számolja a hibás lezárás valószínűségét, nem Monte Carlo-ban.

| Valódi prior log-odds eltolása | Kihagyási kockázat, feltéve hogy nem történt sikeres discovery | 95% bootstrap CI | Olyan instance-ek aránya, ahol az érték meghaladja a saját epsilonját |
|---:|---:|---:|---:|
| 0.0 | 9.3% | [8.9%, 9.7%] | 0.0% |
| 0.5 | 14.5% | [13.9%, 15.2%] | 73.3% |
| 1.0 | 22.0% | [21.2%, 23.0%] | 93.1% |
| 1.5 | 32.1% | [30.9%, 33.4%] | 99.5% |

Az átlagos instance-szintű epsilon 12,15% (nem közös epsilon): nulla eltolásnál **minden instance** az adott epsilon alatt maradt; pozitív log-odds eltérés után sok instance-ben már sérül a controller által nominal modellel tervezett risk bound. Mivel az epsilon instance-enként eltér, az aggregált feltételes risket **nem szabad önmagában** egyetlen egységes risk-szerződésként interpretálni.

## 5. Skálázhatóság

| Domainek N | Első PSA1 döntés medián idő [s] | Minimum [s] | Maximum [s] |
|---:|---:|---:|---:|
| 10 | 0.0026 | 0.0026 | 0.0028 |
| 20 | 0.0193 | 0.0190 | 0.0235 |
| 40 | 0.1503 | 0.1485 | 0.1610 |
| 60 | 0.5312 | 0.5093 | 0.5612 |
| 80 | 1.2389 | 1.2270 | 1.2619 |

Méretre jutó 8 stresszpélda; a medián log–log meredekség megközelítőleg 2,97. Ez összhangban van a PSA1 első döntésének egyszerű implementációban várható közel köbös skálázódásával, de nem helyettesíti a bizonyított `O(N^3 log N)` per-decision boundot. A benchmark a futtatási környezet és Python implementáció függvénye.

## 6. Mit érdemes közölni a NOMS-cikkben?

1. RQ1/RQ2: véletlen kontroll és *külön jelölt*, elméletvezérelt mechanizmuscsaládok; query-only **és** fairer SKIP-aware baseline; az átlag mellett 95%-os gap és worst-case ellenpélda.
2. RQ3: első döntés runtime növekedése, helyes base `O(N^2 log N)` versus action selection `O(N^3 log N)` megkülönböztetés; a teljes rollout fa várható költsége benchmark szinten más komplexitás.
3. RQ4: ε és refresh költség/pontosság faktoriális ábra; az ε-garancia **kalibrációs feltételhez kötött**.
4. Korlátok: szintetikus, független domainek, k=1, one-refresh, modellezett költségek; célszerű a trace-driven Alibaba-pilotot továbbra is csak kiegészítőnek tekinteni.

## 7. Reprodukálhatóság

A teljes parameterszintű CSV, a generátorok, algoritmuskód, öt unit test, táblák és ábrák a ZIP-ben vannak. A futtatási parancsokat lásd: `README_HU.md`. A benchmark nem ad új szerzőtársi vagy adatszolgáltatási korlátozást.

### Erősebb baseline: exact QUERY+SKIP, REFRESH nélkül

A `no_refresh_opt` azonos epsilon és minden köztes siker/sikertelenség mellett egzakt optimális, de kizárólag QUERY/SKIP döntéseket enged. Az információszerzés tisztább ablation-je, mint a greedy SKIP: például `epsilon=0.10`, `alpha=beta=0.97`, `r/q=0.20` esetén PSA1 átlagosan 32,17%-kal olcsóbb az **optimális refresh nélküli policy-nél**, míg a teljes Bellman-policy refreshből származó elméleti haszna 35,50%. PSA1 egyedi esetben lehet rosszabb a no-refresh optimal policy-nél, mivel maga nem egzakt; a fenti eredmények mintaátlagok.
