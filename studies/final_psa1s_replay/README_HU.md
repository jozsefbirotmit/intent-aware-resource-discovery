# Végleges PSA1-S: célzott telemetria-visszajátszás

A csomag a végleges, feltöltött algoritmuskódot kapcsolja a korábbi Philly- és
Alibaba-2018-mintákhoz. A tanult modellek és az adatmaszkok változatlanok. A
futtatások során nem történt új algoritmus- vagy tesztküszöbhangolás.

Részletes jelentés: `report/NOMS2027_PSA1S_Frozen_Telemetry_Replay_v1_hu.md`.

## Reprodukció

Python 3.12, illetve a `requirements.txt` verziói. A fő benchmark Python 3.13-as
környezetével szemben itt új online futásidő-összehasonlítás nem készült.

A következő parancsokat a csomag gyökerében kell futtatni, lehetőleg másolaton,
mert újraírják a `results/` és `report/` generált fájljait. Az inputokat és a
végleges algoritmusforrásokat nem módosítják. A teljes nagy nyers trace-ek
letöltésére nincs szükség.

```bash
python -m pip install -r requirements.txt
python code/replay_philly.py
python code/replay_alibaba2018.py
python -m unittest discover -s code -p 'test_final_replay.py' -v
python code/build_replay_report.py
python code/validate_replay_outputs.py
```

A validator a csomag eredeti öt unit tesztjének mentett futását is ellenőrzi.
A fenti új unit tesztfutás stdout/stderr kimenete saját naplóba is menthető.
A benchmark teljes szintetikus futtatása most nem szükséges: a vendor-kód
entrypointjait a wrapper csak importálja, nem indítja el újra a 90 modelles
benchmarkot.

## Források és összevetés

- `vendor/final_algorithm/`: a felhasználó által feltöltött végleges benchmark Python-forrásai változatlanul. Az eredeti munkanevek PSA1, PSA1-E, PSA1-S, PSA1-A; a wrapper a PSA1-et és a PSA1-S-et futtatja.
- `input/philly/`: az elsődleges teljes ismert telemetriás minta állapot-idősora, 896 mentett időszakmodellje, eredeti eredménytáblája és metaadatai. A másodlagos foglalási részminta nem része a célzott újrafuttatásnak.
- `input/alibaba2018/`: a 27 valószínűségi modellből képzett 54 költségprofil mentett paraméterei; a teljes naptári domainállapot-rács, eredeti kalibrációs választások/rács és eredmények.
- `provenance/frozen_input_sources.json`: a rögzített inputok és a változatlan vendor-források mérete és SHA-256 értéke. A validator ezeket újra ellenőrzi.
- `provenance/replay_plan.md`: az új replay-eredmények előtt rögzített protokoll.

A nyers adatminőségi auditokat a korábbi csomagok tartalmazzák. Az örökölt
állapotpredikátumokat nem értelmezzük tényleges scheduler-elhelyezhetőségként.
Minden keresési költség modellezett, normalizált érték.

## Eredmények

- `results/philly/policy_summary.csv`: 224 konfiguráció × öt módszer = 1120 sor. Az összes módszer ugyanazt a tesztpontot és időszakonként ugyanazt a modellt használja.
- `results/alibaba2018/policy_summary.csv`: 54 profil × három protokoll × öt módszer × két megfigyelési maszk = 1620 sor. A `per_lag_known` az információkoronként teljes ismert minta; az `all_intents_lags_common` a korábbi közös minta az információkorok összevetéséhez.
- `results/*/model_tie_audit.json`: modell szerinti költségek, kezdeti döntések, a szelektív mélyítés inaktivitása és a 256 kimeneten összehasonlított költség/elmulasztás/STOP/fizetett akciósor eltérései.
- `results/philly/model_tie_audit.csv`, `results/alibaba2018/test_tie_audit.csv`: megfigyelt tesztidőpontokkal súlyozott eltérések. Az átfedő profilok számai nem független események.
- `results/alibaba2018/calibration_budget_grid.csv` és `calibration_choices_frozen_before_test.json`: a kizárólag kalibrációs szakaszon választott keretek, a tesztpolicyk értékelése előtt rögzítve.
- `results/*/legacy_reproduction_checks.json`: az eredeti query-only, PSA1 és régi egzakt DP költségeinek és találati számainak visszaellenőrzése.
- `results/philly/daily_policy_statistics.csv.gz`, `results/alibaba2018/calendar_policy_statistics.csv.gz`: a naptári blokk-bootstraphoz szükséges összesítések.
- `results/headline_summary.json`, `results/*/headline_zero_budget*.csv`: a jelentés fő összehasonlításainak pontos adatai.
- `report/fig*.png`, `report/fig*.pdf`: három ellenőrzött ábra raszteres és vektoros formában.
- `provenance/unit_test_run.txt`, `provenance/final_validation.json`: az elvégzett ellenőrzések.

## Fontos értelmezés

Mindkét vizsgált minta N=4-et használ. K=8 mellett a végleges PSA1-S kezdeti
STOP-ot vagy egzakt tailt alkalmaz; szelektív előretekintést nem. Az S és a
korábbi egzakt DP költsége az összes ellenőrzött kimeneten megegyezik, de az
optimális döntések és az elmulasztási arányok eltérhetnek.

A `no_refresh_opt` az ugyanazzal az epsilon-nal futó optimális, REFRESH
nélküli baseline. A `saving_vs_no_refresh_percent` minden sor saját
epsilon-jának megfelelő baseline-t használ, a módszerenként eltérő
kalibrált keretek esetén is. Ha ez a baseline költsége nulla, a relatív
megtakarítás nem definiált; üres CSV-mező marad.

A `conditional_miss` nevezője az `empty_stop_n`; a `miss_per_request`
nevezője az összes kérés. A nulla nevező és a nulla megfigyelt eseményből
elfajuló bootstrap-intervallum külön jelölést kap. A posterior-korlát minden
modellbeli STOP-nál teljesül, de ez nem automatikus mérésbeli garancia. A
leíró 95%-os intervallumok a Philly-n hét napos, az Alibaba-n egyórás naptári
mozgóblokkokból készültek, 1000 ismétléssel. Az időbeli eloszlásváltozással
szemben nem jelentenek statisztikai tanúsítványt.

Az új számok: a Philly végleges S legnagyobb feltételes aránya 18,67%; az
Alibaba végleges S legnagyobb aránya 10,30%. A korábbi 7,45%-os Alibaba-érték
a PSA1-re vonatkozott. A nulla keretű kontrollokban nincs megfigyelt
elmulasztás. A részletes feltételeket és korlátokat a jelentés tartalmazza.
