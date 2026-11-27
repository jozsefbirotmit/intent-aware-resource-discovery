# Scientific audit supplement — artifact v1.1.0

This directory extends the complete v1.0.0 artifact (https://doi.org/10.5281/zenodo.24583528) for manuscript v10 of "Intent-Aware Sequential Resource Discovery under Uncertain Domain Knowledge". The original release, solver and experimental inputs are unchanged.

## Reproduction

Use the dependencies of the archived final benchmark (Python, NumPy, Numba, pandas). From this directory, with REPO pointing to the root of the v1.1.0 checkout (or original v1.0.0 checkout for the underlying studies):

```bash
python check_baselines.py "$REPO/studies/final_benchmark" baseline_60_rerun.csv
python check_small_states.py "$REPO/studies/final_benchmark"
python check_telemetry_summaries.py "$REPO" --out telemetry_summary_rerun.json
python check_synthetic_summaries.py "$REPO"
```

Compare the regenerated baseline CSV with baseline_60.csv (floating-point tolerance may be required across platforms). HQ uses a query-only nominal continuation; H1 permits one screening module. Both use the same admissible first-action set and replan after observations. This is a retrospective comparison on the same 60 held-out models at N=10,12, with exact expected costs, no new timing comparison or paired confidence intervals for HQ/H1. The main four-algorithm benchmark remains unchanged. The earlier independent DP check is supplied with its original log, not asserted to have been rerun for WP4.

The telemetry summary script uses the frozen aggregate policy summaries and the Philly domain-state calendar grid. The Alibaba denominator (25,560 ten-second test bins) follows the archived chronological split. No Alibaba measurements, time-indexed states or fitted beliefs are redistributed. The WP4 numerical checks were executed on the equivalent retained frozen replay archive with --replay-root; the HQ/H1 rerun used the vendored frozen solver there and reproduced the audit CSV byte for byte. The original synthetic aggregate CSVs were inspected directly at tag v1.0.0; the standalone synthetic assertion script was supplied for a full local checkout and was not executed on such a checkout in WP4. Existing installed dependencies were used; no fresh environment installation was attempted in WP4.

The local SHA256SUMS verifies the audit directory. The repository-level manifest covers both the original artifact and this extension. The new Zenodo version DOI is assigned upon release archiving.
