# Data provenance and acquisition

This public-release candidate includes Philly measured subsets and derived data for offline replay. Alibaba measurement and derived-input files are excluded. It does not include multi-gigabyte original archives. This assembly did not repeat the original downloads or the full-record audits. The retained original preprocessing scripts and hashes document these steps. Data in the legacy directories and frozen replay inputs remain distinct from simulated data; do not apply a software license to them.

## Philly

- Source: https://github.com/msr-fiddle/philly-traces, pinned commit `29a1b87fa2d9ed80b83c9e3a37f3a88d382b031d`.
- Original archive SHA-256 `2037ccf63a725a02be718b0e1aeee6142c90b7f62d0d849295120e9f6ba13d2c`.
- Upstream CC BY 4.0, https://creativecommons.org/licenses/by/4.0/. Cite Myeongjae Jeon, Shivaram Venkataraman, Amar Phanishayee, Junjie Qian, Wencong Xiao and Fan Yang, *Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads*, USENIX ATC 2019, pp. 947–960. The measured subset and derived domain states are adaptations: 24 selected machines, time alignment, binning, utilization/memory predicates and masks. No upstream endorsement.
- Approximately 1.06 GB compressed, 7.06 GB extracted, at least 10 GB disk advised by the original study.

In a disposable copy of `studies/philly_legacy/`:

```bash
python code/download_philly.py raw/trace-data.tar.gz
python code/extract_original.py raw/trace-data.tar.gz raw
python code/prepare_philly.py --raw raw --out results --job-timezone UTC
python code/audit_integrity.py --raw raw --out results
python code/philly_replay.py --input results/philly_measured_subset.npz --out results_telemetry_full --telemetry-only --bootstrap 1000
```

Compare generated domain states, fitted models and policy summaries with the preserved files. The secondary job-log subset is retained for provenance, not used by the final paper replay.

## Alibaba 2018

Source: https://github.com/alibaba/clusterdata/tree/3858992ca41bf92082e6d6522ddf0bebd14e838b/cluster-trace-v2018. The provider encourages research use but explicit redistribution permission for the measurement records and derivatives was not established. The public candidate therefore includes no machine-level measurements, time-indexed domain states, calendar statistics or fitted model input files. Aggregate research results (`policy_summary.csv`, `brier_scores.csv`) remain available. Original source hashes are preserved in `studies/alibaba2018_legacy/provenance/`.

Obtain `machine_meta.tar.gz` and `machine_usage.tar.gz` from the provider after reviewing its terms and completing any applicable survey. The acquisition script can validate their original hashes and download from the official endpoint where permitted. Approximately 1.78 GB compressed; the approximately 9 GB usage table is streamed. Run these commands from the repository root (choose a fresh work directory):

```bash
python studies/alibaba2018_legacy/code/download_alibaba2018.py --raw work/raw/alibaba2018 --provenance work/raw/alibaba2018/provenance
python scripts/reproduce.py alibaba-prepare --alibaba-raw work/raw/alibaba2018 --work work/alibaba-build
python scripts/reproduce.py telemetry --alibaba-results work/alibaba-build/alibaba2018_legacy/results/alibaba2018 --work work/joint-replay
```

The preparation target selects the original training-coverage-based subset, reconstructs domain states, fits models, runs the legacy experiment, and compares the regenerated policy and Brier summaries against the archived aggregates. The joint replay stages the six locally generated input files in its own work directory, checks the legacy aggregates again, preserves the original frozen-input hash manifest separately, and records the actual hashes of the staged inputs. It then replays both datasets and compares the final policy summaries to the frozen research results. Byte equality of regenerated inputs is not assumed. This new raw-data-to-staging workflow was not executed on the complete multi-GB source archives during public-candidate assembly; original local reruns started at the retained measured subsets.

Keep `work/` outside version control. Do not upload local Alibaba source or derived data to GitHub or Zenodo without explicit permission.
