# Manuscript v8 result-to-command map

The original headline numbers can be checked against `python scripts/reproduce.py quick`; the numerical assertions encode the printed rounding, not new estimates. Tables below refer to the v8 manuscript labels. Each fresh run target creates a separate copy under `work/`.

| Manuscript item | Archived source | Fresh reproduction |
|---|---|---|
| Table II (`tab:quality`), mean/max gaps and savings | `final_benchmark/holdout.jsonl` | `benchmark`, `summarize.py` |
| Table II timing columns | `final_benchmark/online.jsonl` | `benchmark` (new hardware-dependent timings) |
| Stratified paired CIs and family means | `holdout.jsonl`, `paired_ci.csv` | `benchmark`, `summarize.py` (10,000 resamples) |
| Fig. 1 (`fig:scaling`) | `scaling.jsonl`, `scaling_summary.csv` | `benchmark`; `figures` regenerates archived numerical plotting data |
| Table III (`tab:tracezero`) | `final_psa1s_replay/results/{philly,alibaba2018}/policy_summary.csv` | `telemetry` with locally generated Alibaba inputs |
| Fig. 2 (`fig:tracerisk`), positive risk and zero-denominator exclusions | same policy summaries, calendar/day aggregates | `telemetry` with locally generated Alibaba inputs; `figures` regenerates plotting data |
| Cost vs signal age, negative savings, positive risk example, extremes and CIs | same final policy summaries | `telemetry` with locally generated Alibaba inputs |
| Exact-tail model checks and calibration choices | `model_tie_audit.json`, `calibration_budget_grid.csv` | `telemetry` with locally generated Alibaba inputs |
| P3/P4/deferred mechanism checks | `mechanisms/results/cases.csv` | `mechanisms` |
| Signal quality factorial comparison | `mechanisms/results/sensitivity_cases.csv` | `mechanisms` |
| Prior log-odds shift risk | `mechanisms/results/calibration_drift_cases.csv` | `mechanisms` |
| Alibaba Brier scores and prevalence | `alibaba2018_legacy/results/alibaba2018/brier_scores.csv` | `alibaba-prepare` |
| Table I algorithm variants and Appendix B flowchart | manuscript, final solver sources, TikZ | not numerical claims |

All paths in the archived-source column are relative to `studies/`. The quick command writes `work/paper_results/` with the numeric table data, paired CIs, risk extrema, plotting data and two independently rendered figures. The exact v8 figure assets remain in `manuscript/results/figures/`.

For a Philly-only replay use `telemetry-philly`. For the full joint replay and Alibaba refit use the instructions in `DATA.md`; Alibaba source and derived inputs are not included in this distribution.

## Manuscript v10 scientific-audit additions (artifact v1.1.0)

| Revised manuscript claim | Source and reproduction |
|---|---|
| HQ/H1 mean per-instance savings on 60 holdout models | `audit/baseline_60.csv`; `audit/check_baselines.py` |
| Independent small-state DP and continuation checks | `audit/check_small_states.py`, historical log |
| Zero-budget grid signs, coverage, calibration budget counts, risk denominators and bootstrap examples | `audit/check_telemetry_summaries.py`, `audit/telemetry_summary.json` |
| Synthetic aggregate savings, episode timing and prior-drift interpretation | `audit/check_synthetic_summaries.py` and original benchmark summaries |

The v8 manuscript snapshot and original numerical results are unchanged. Consult `audit/README.md` for the supplementary reproduction commands and scope.
