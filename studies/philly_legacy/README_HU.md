# Philly: adatminőségi audit és négy-domaines IA-OSR replay

A csomag a Microsoft által közzétett **eredeti Philly-archívum tényleges feldolgozásából** készült. A teljes archívum ellenőrzőösszege egyezett a hivatalos Git LFS-bejegyzéssel. A 24 gépes mért részhalmaz és az összes kísérleti eredmény benne van a csomagban.

**Értelmezési korlát:** megfigyelt GPU-terhelési és hostmemória-feltételeket vizsgálunk. A hibás gépleltár kapacitáscímkéit nem használjuk; a hiányos feladatnapló nem bizonyít teljes scheduler-foglaltságot. A kockázati korlát pontbecsléses modellekből történő átvihetősége több esetben nem teljesült. A jelentés mind a kedvező eredményeket, mind a nagy DP-gapet közli.

## Gyors reprodukálás a mellékelt mérési részhalmazzal

Python 3.12 és a `requirements.txt` csomagjai szükségesek:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s code -p 'test_philly.py' -v
python code/philly_replay.py --input results/philly_measured_subset.npz --out rerun_results_telemetry_full --telemetry-only --bootstrap 1000
python code/audit_counterexample.py --results rerun_results_telemetry_full
python code/philly_replay.py --input results/philly_measured_subset.npz --out rerun_results_joint --bootstrap 1000
python code/audit_counterexample.py --results rerun_results_joint
```

Az elsődleges, teljes ismert telemetriát használó kontroll a `results_telemetry_full` könyvtárban van: 224 konfiguráció × hat módszer = 1344 sor. A foglalási naplóval is párosítható részminta a `results` könyvtárban marad: 448 konfiguráció × hat módszer = 2688 sor. Összesen 672 kiértékelési konfiguráció és 4032 sor készült, átfedő mérésekkel. Az újraszámolt kimenetek a fenti külön `rerun_...` mappákba kerülnek. A kód minden illesztett modellben mind a 256 lehetséges négy-domaines igazság–refresh kimeneten ellenőrzi a DP- és PSA1-értékeket. Az eredeti jelentés és ábrák újragenerálásához:

```bash
python code/build_full_report.py
```

## Teljes feldolgozás az eredeti adatokból

A nyers bemenetek kb. 1,06 GB tömörített, összesen kb. 7,06 GB kibontott adatot igényelnek. Az előkészítés további köztes tömböket hoz létre; néhány GB szabad memória és legalább 10 GB szabad lemezterület célszerű. A CPU-telemetria is a hivatalos csomag része, de a mostani intentekhez nem használjuk.

```bash
python code/download_philly.py raw/trace-data.tar.gz
python code/extract_original.py raw/trace-data.tar.gz raw
python code/prepare_philly.py --raw raw --out results --job-timezone UTC
python code/audit_integrity.py --raw raw --out results
python code/philly_replay.py --input results/philly_measured_subset.npz --out results --bootstrap 1000
python code/audit_counterexample.py --results results
python code/philly_replay.py --input results/philly_measured_subset.npz --out results_telemetry_full --telemetry-only --bootstrap 1000
python code/audit_counterexample.py --results results_telemetry_full
python code/build_full_report.py
```

A fenti teljes feldolgozás ugyanabba a `results` könyvtárba ír; az eredeti eredményeket megőrző gyors változat külön célkönyvtárat használ. A `prepare_philly.py` a feldolgozott 96 jelölt gép mérési tömbjeit gyorsítótárazza. A tanítási lefedettségre épülő gépválasztás nem használ tesztbeli alkalmasságot vagy algoritmuseredményt.

## Fő fájlok

- `report/NOMS2027_Philly_audit_replay_v1_hu.md`: részletes magyar jelentés, költség- és kockázati ábrákkal.
- `report/Philly_joint_subset_details_hu.md`: a foglalási naplóval párosított részminta részletes auditja, további ábrákkal.
- `results_telemetry_full/policy_summary.csv`: az elsődleges, teljes ismert terhelési/memóriamintára készült kontroll eredményei.
- `results/policy_summary.csv`: a teljes összehasonlítás, költség és kockázat együtt.
- `results/philly_measured_subset.npz`: a valódi mérések kivonata, 24 gép × nyolc megfigyelt GPU-csatorna; hostmemória és naplózott/bizonytalan foglaltság.
- `results/domain_states.csv.gz`: minden elrendezés és intent megfigyelt állapotai, a hiányzó értékek és a közös tanítási/tesztmaszkok megőrzésével.
- `results/daily_policy_statistics.csv.gz`: napi elegendő statisztikák, a függő adatokon végzett blokk-bootstrap ellenőrzéséhez.
- `results/fitted_models.json`: paraméterek, korrelációk, kimeneti gyakoriságok, modell szerinti várható költségek és a statikus visszatérés jelzője.
- `results/algorithmic_counterexample.json`: a legnagyobb egyedi modellbeli PSA1/DP-gap, pontos paraméterekkel és a közvetlen/gyors continuation egyezőségi ellenőrzésével.
- `results_telemetry_full/algorithmic_counterexample.json`: a teljes telemetriás kontroll külön ellenpéldája; a két mintán mért modellek és gapek nem cserélhetők fel.
- `results/inventory_job_telemetry_consistency.csv`: a kapacitáscímkék és kiosztott GPU-indexek összevetése.
- `results/job_epoch_time_evidence.csv`: az application ID-k időbélyegei és az első megfigyelt beküldések kapcsolata.
- `results/job_time_alignment_sensitivity.csv`: a job–telemetria illesztés időeltolási érzékenysége.
- `provenance/original_files.json`: az eredeti kibontott fájlok mérete és SHA-256 értékei.
- `provenance/validation_log.txt`: a nyolc célzott ellenőrzés és a nagy modell-gap külön auditja.

## Fontos szemantika

A telemetria percét a perc végén tekintjük megfigyelhetőnek. A frissítés az 1, 5, 15 vagy 60 perccel korábbi domainpredikátumot adja vissza. Nincs interpoláció, forward-fill vagy szintetikus munkaterhelés. A hiányzó feladatvégek nem válnak önkényesen „szabad” vagy „foglalt” címkévé; a bizonytalan állapot külön megmarad.

A jobidők UTC-ként értelmezése a nyers application ID epoch-része és a beküldési idők alapján levont következtetés. A README nem közöl explicit jobidőzónát. A PDT/PST telemetriajelölések pontosan UTC-re konvertálódnak; a DST ismétlődő órája nem olvad össze.

A `conditional_miss` nevezője az `empty_stop_n`, a `miss_per_request` nevezője az összes tesztkérés. Az `epsilon=0` kontrollban minden megmaradt bizonytalanságot pontos QUERY old fel. Az `epsilon=0.1` esetben a modell szerinti korlát nem jelent igazolt trace-beli korlátot. Nulla nevezőnél a feltételes arány nincs definiálva; nulla megfigyelt elmulasztásnál a bootstrap-intervallum elfajulhat.

A két kalibráció statikus vagy heti frissítésű, előző 14 napra támaszkodó modell. Legalább 100 páros megfigyelés hiányában az utóbbi kifejezetten a teszt előtti statikus modellre tér vissza. A három elrendezés ugyanazon klaszter időben összefüggő, diszjunkt gépcsoportja; nem három független mérés. A költségek modellezettek.

## Adatforrás, licenc és hivatkozás

Forrás: [Microsoft Research Philly traces](https://github.com/msr-fiddle/philly-traces), commit `29a1b87fa2d9ed80b83c9e3a37f3a88d382b031d`. Eredeti archívum SHA-256: `2037ccf63a725a02be718b0e1aeee6142c90b7f62d0d849295120e9f6ba13d2c`. Az eredeti nyilvános adatok [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) licencűek. A csomag mérési kivonatát és állapotpredikátumait ezekből hoztuk létre; a módosításokat a kód és a jelentés részletezi. Ez nem Microsoft által jóváhagyott benchmark.

Kérjük hivatkozni: Myeongjae Jeon, Shivaram Venkataraman, Amar Phanishayee, Junjie Qian, Wencong Xiao és Fan Yang: *Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads*. USENIX ATC 2019, 947–960. A BibTeX-bejegyzés a `references.bib` fájlban szerepel.
