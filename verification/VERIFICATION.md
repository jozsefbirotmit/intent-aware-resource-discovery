# Historical verification report — 25 November 2026

**Distribution note (public candidate 0.2):** this report describes the prior full local v0.1 verification. Detailed rerun directories and logs are retained in the full local archive, not distributed here; the summary JSON comparisons, aggregate paper results, and source code remain. Evidence paths to excluded files below refer to the local archive, not this candidate. Alibaba record-level and fitted input data must be regenerated locally. The new quickstart and staging checks are reported separately in `PUBLIC_CANDIDATE_CHECK.md`.

Scope: NOMS 2027 manuscript v8, local artifact candidate 0.1. This report records the runs actually performed while assembling the candidate, not the original studies' validation assertions alone.

| Test | Actual result | Evidence |
|---|---|---|
| Final synthetic benchmark | 90 generated parameter sets, 360 policy costs/bounds/gaps, 720 trajectories (cost, success/miss, residual model risk, decision depths), 30 scaling root decisions and bounds match archived results | `benchmark_comparison.json`, `rerun/final_benchmark/`, `logs/benchmark.log` |
| Final telemetry policy replay | All 1120 Philly and 1620 Alibaba policy-summary rows match archived outputs, including integer counts, costs, risk and bootstrap intervals | `telemetry_comparison.json`, `rerun/telemetry/` |
| Final replay consistency | 23 frozen input/vendor hashes, 1220 exact-tail models, 270 calibration choices, zero model STOP-bound violations; 5 unit tests pass | `telemetry_validation.json`, `logs/replay_tests.log`, `logs/replay_validate.log` |
| Controlled mechanisms | 3600 case rows, 4860 factorial rows, 1500 drift rows, 27 factorial summary rows and 4 drift summary rows match originals; 5 unit tests pass | `mechanisms_comparison.json`, `rerun/mechanisms/`, `logs/mechanisms.log` |
| Refit from Philly measured subset | 896 fitted period models, exact equality of domain-state grid and 1344 legacy policy summary rows; parameters within 1e-12 | `philly_refit_comparison.json`, `logs/philly_refit.log` |
| Refit from Alibaba measured subset | 54 fitted cost-profile models, exact equality of state/mask arrays, 2268 legacy policy rows and Brier-score table; parameters within 1e-12 | `alibaba_refit_comparison.json`, `logs/alibaba_refit.log` |
| Paper headline assertions | All tested rounded v8 values pass; two numerical tables and both performance figure datasets regenerated | `paper_results.log`, `paper_results/` |
| Quickstart driver | Executed with pinned installed dependencies; both test suites pass | `logs/quickstart.log` |

Numerical CSV comparisons use absolute and relative tolerances 1e-9, NaNs matched at the same positions. Deterministic benchmark parameters/actions and discrete outcomes are compared exactly. This is reproducibility of the original code and design, not independent scientific validation of the model, theorem proofs or dataset generalizability. The stored results from fresh runs are separate from the originals.

## Key replicated claims

- 60 N=10–12 instances: mean gaps PSA1 5.6416849%, PSA1-E 4.4093944%, PSA1-S 3.1653095%, PSA1-A 2.9229263%. PSA1-S mean saving vs same-budget no-refresh optimum 18.0240332%; max gap 17.3907003%.
- PSA1-S vs PSA1-E: 1.2441 percentage-point mean improvement, paired stratified 95% interval [0.5354, 2.0276]; 40 better, 7 worse, 13 ties.
- Zero-risk savings: Philly 2-GPU 14.38, 17.27, 21.15%; 4-GPU 6.85, 13.89, 18.73%; Alibaba C 2.54, 3.52, 3.04%; no observed misses in those comparisons.
- Nominal 10% tolerance: worst conditional miss rates Philly 1996/10692 = 18.67%, Alibaba 2412/23425 = 10.30%. Exceedance counts 34/112 and 1/54. Exact tail never invokes selective refinement at N=4.

## Runtime caveat

The archived timing columns in the paper are correctly recomputed from the original observations. In the present rerun, executed partly concurrently on shared resources, first-decision medians for the four methods were 1.75, 1.73, 25.49 and 124.32 ms (N=10–12), and N=100 PSA1-S root medians were 1.77 s IID and 2.39 s stress. These new times are descriptive only; no timing-equality assertion was made and the original paper timings are unchanged. The fresh timing data are in `rerun/final_benchmark/online_summary.csv` and `scaling_summary.csv`.

## Not completed / release gates

- Original multi-gigabyte source traces were not downloaded and processed again in this assembly. Their original source URLs, preprocessing code, hashes and measured-subset inputs are retained. Replay from preserved machine-level measurements, rather than only saved result CSVs, was independently executed.
- The new figure renderer regenerates numerical contents; it is not a byte-exact reconstruction of the original v8 PDF appearance. Original assets are included.
- A fresh independent virtual environment/container installation was not performed; the installed versions were checked. Re-run on a separate machine for external artifact evaluation.
- GitHub repository and Zenodo DOI not created. Software license, coauthor approval, Alibaba v2018 data redistribution, and review anonymity need resolution before publication. See `docs/RELEASE_CHECKLIST.md`.
- No changes to manuscript v8; no new results added to it.
