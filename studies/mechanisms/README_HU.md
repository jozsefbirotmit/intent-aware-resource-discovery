# IA-OSR / NOMS 2027 — mechanism-driven synthetic experiments (v1)

**Státusz:** reproducibilis, kontrollált *szintetikus* kísérleti kutatási csomag. Sem itt, sem az ábrákban **nincs Alibaba trace-adat**. A költségek az egzakt, modell szerinti várható költségek; a confidence intervalok a generált problémák eloszlására vonatkozó bootstrap-intervallumok, nem fizikai mérések mérési hibái.

## Fájlok

- `ia_osr_experiment.py` — változatlan korábbi egzakt Bellman, H1 és skip-closed implementáció.
- `ia_osr_psa_audit.py` — változatlan korábbi PSA1 referencia és gyorsított implementation.
- `mechanism_study.py` — új célzott generátorok, összehasonlító policyk, bootstrap, drift-vizsgálat, stresszteszt.
- `test_mechanisms.py` — P1, P2, P3, P4 és deferred-refresh kontrollok.
- `render_figures.py` — az öt ábra újragenerálása CSV-kből.
- `results/cases.csv` — 3600 közös algoritmikus teszt; teljes paraméterek a `parameters` mezőben.
- `results/family_summary.csv`, `results/boundary_pair_summary.csv` — aggregált statisztika és párosított SKIP-határ.
- `results/sensitivity_cases.csv` — 4860 teljes faktoriális futás, `sensitivity_summary.csv` — 27 cella.
- `results/calibration_drift_cases.csv` — 1500 true-vs-believed prior értékelés, `calibration_drift_summary.csv` — összesítés.
- `results/runtime_scaling.csv`, `results/runtime_scaling_summary.csv` — 40 futás, N=10..80.
- `results/figures/*.png` — publikációs célú ábrák.
- `noms2027_synthetic_study_v1_hu.md` — módszertan, eredmények, fenntartások.
- `noms2027_evaluation_draft.tex` — IEEE cikkbe beilleszthető angol subsection-szöveg (nem önállóan fordítandó).

## Függőségek

Python >= 3.10; az elsődleges optimalizáláshoz `numpy`; az ábrákhoz `pandas` és `matplotlib`.

```bash
python -m pip install numpy pandas matplotlib
python -m unittest test_mechanisms -v
```

## Az összes eredmény újragenerálása

A ZIP tartalmazza a kész kimeneteket, ezért futtatás nélkül is ellenőrizhetők. Az alábbi parancsok a **csomag gyökérkönyvtárából** indítandók:

```bash
python mechanism_study.py --out results --nseeds 5 --count 120 --robust-count 75 --bootstrap 1500
python -c "from mechanism_study import run_factorial;run_factorial('results',seeds=(1201,2201,3201,4201),count_per_seed=45,bootstrap=1000)"
python render_figures.py results
```

A `--out` új, üres könyvtárra is állítható. A futási idők mérési környezetfüggők; a többi statisztikát determinisztikus random seedek és determinisztikus bootstrap seed reprodukálják. A `mechanism_study.py` önálló futtatása a `run_factorial`-t nem hívja automatikusan, ez külön parancs.

A mechanizmuscsaládok:

1. `iid`: N=4 vagy 5, korábbi uniform/mixed eloszlás (baseline regeneráció).
2. `p4_probe`: N=2, **konstruált crossing és két pozitív P4 költségmargin**, ahol negatív refresh után sem SKIP-elhető domain.
3. `p3_screen`: N=2, **konstruált L<q_i<U**, negatív branch admissible.
4. `deferred`: N=2, előbb olcsó Q_j; ha sikertelen, akkor R_i.
5. `skip_boundary`: N=4, **600 párosított probléma**, minden egyéb paraméter azonos, az \(\varepsilon\) közvetlenül a két kihagyás riskjének alsó/felső oldalán van.

Minden `cases.csv`-beli algoritmikus expected costot a véges decision tree modellből számolunk (nem Monte Carlo-trace). A risk drift a *valódi* `p` priorok log-odds pozitív eltolásával modellezett **szándékos mismatch**; a controller továbbra is az eredeti beliefeket használja a döntéshez. A kontroll értékeli az így eredő tényleges terminal miss valószínűséget.

## Reprodukálhatósági és interpretációs figyelmeztetések

- A P3, P4 és deferred családok **matematikailag kedvező eseteket célzottan konstruálnak**; a 600/600 siker nem bizonyít általános közeloptimalitást.
- A SKIP-küszöb család szintén szándékosan extrém és nem reprezentatív.
- A `query_only` nem használ SKIP-et; fairer kockázati baseline a `greedy_skip`. Mindkettőt közöljük.
- A H1 és a PSA1 realizált expected costja között **nincs általános monoton dominancia**, még ha base continuationre van is rendezés.
- A PSA1 matematikai felső korlátja csak a helyesen definiált, független domainmodellben értendő; ezek a futások nem validálják valós hálózati telemetria kalibrációját.
- A kockázati tolerancia miss-discovery risk, **nem** szolgáltatás rendelkezésre állási SLA.
- A runtime egy **első döntés** számítását méri (nem az összes Bellman-ág enumerációját), kizárólag Python és adott hardver mellett.

## Erősebb baseline (v1 kiegészítés)

A `results/cases.csv` és `results/sensitivity_cases.csv` már tartalmazza a `no_refresh_opt` értékét. Ez a kis-N modellben **egzakt Bellman-optimum azzal a megszorítással, hogy REFRESH tilos**; azonos risk-budget mellett megengedett a SKIP és QUERY. A `sensitivity_summary.csv` `saving_vs_no_refresh_pct` oszlopa ezért a PSA1 hozzáadott értékét méri az optimális frissítés nélküli döntéssel szemben. Az `exact_value_of_refresh_pct` az egzakt teljes DP és az egzakt frissítés nélküli policy közötti különbség, így elméleti információszerzési érték a vizsgált modellben. A PSA1 nem garantáltan dominálja a no-refresh optimumot példányonként.
