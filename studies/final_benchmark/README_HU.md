# Reprodukálhatóság

A korábbi v2 Python solver-kódok változatlanok. Új benchmark: `benchmark.py`, összesítés és ábrák: `summarize.py`. Python, numpy, numba, matplotlib szükséges.

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark.py 10
python summarize.py
```

A teljes futás felülírja a 3 JSONL eredményfájlt. A 10 itt a méret–család rétegenkénti esetszám; a summarizer a rögzített 90-es benchmarkot várja. A kimenetek: holdout.jsonl (modellek, paraméterek, egzakt költségek), online.jsonl (720 realizált trajektória), scaling.jsonl (30 gyökérdöntés), négy summary CSV, ábrák. `research_report_hu.md` a módszertani jegyzet. A `noms2027_algorithm_evaluation_v1.tex` önálló IEEE-formátumú kutatási részlet, később a fő kéziratba szerkeszthető; a három PNG vele egy könyvtárban legyen. A fő kéziratot nem cseréltük fel ezzel. Az időmérések nem hard real-time garanciák. Az archivált futás eredményei megtalálhatók, nem kell újrafuttatni a jelentés olvasásához.
