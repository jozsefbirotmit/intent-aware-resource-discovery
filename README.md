# Intent-Aware Sequential Resource Discovery under Uncertain Domain Knowledge

**Artifact version 1.1.0 — complete v1.0.0 artifact plus the scientific-audit supplement for manuscript v10.**

Dániel Ficzere, Attila Frankó, Gergely Hollósi, Pál Varga, József Bíró.

This artifact includes the algorithm implementations, synthetic benchmark, Philly telemetry subset, manuscript figures, and aggregate research results. Alibaba-2018 machine data, time-indexed derived states and fitted inputs are **not redistributed**. They must be regenerated locally from the provider's source under its terms. The full local v0.1 archive is retained separately by the authors. The original v1.0.0 release is archived at https://doi.org/10.5281/zenodo.24583528. The manuscript snapshot under `manuscript/` is the historical v8 snapshot; the audit extension supports the revised v10 results.

## Quickstart

Python 3.13.5, from the repository root:

```bash
python -m venv .venv
# activate the environment (PowerShell: .venv\Scripts\Activate.ps1)
python -m pip install -r requirements.txt
python scripts/reproduce.py quick
```

`quick` regenerates the headline numerical results and figures from archived aggregate/per-instance research results and runs ten unit tests. It does not rerun the full experiments.

```bash
python scripts/reproduce.py benchmark
python scripts/reproduce.py mechanisms
python scripts/reproduce.py telemetry-philly
python scripts/reproduce.py refit
python scripts/reproduce.py figures
```

Use different `--work` directories for repeated or overlapping targets. `refit` rebuilds Philly domain states and models from the included measured subset. New timing values depend on the hardware. For the complete Alibaba preprocessing and joint telemetry replay, follow [DATA.md](docs/DATA.md). For the mapping from manuscript claims to commands, see [PAPER_MAP.md](docs/PAPER_MAP.md).

## Scientific audit extension (v1.1.0)

The [audit directory](audit/README.md) contains the HQ/H1 baseline benchmark on the original 60 holdout instances, an independent small-state DP verification, and checks of the revised telemetry and synthetic summaries. It adds no new third-party records and changes no original solver, benchmark or replay inputs.

Run from the repository root, using the same installed dependencies:

```bash
python audit/check_baselines.py studies/final_benchmark work/baseline_60_rerun.csv
python audit/check_small_states.py studies/final_benchmark
python audit/check_telemetry_summaries.py . --out work/telemetry_summary_rerun.json
python audit/check_synthetic_summaries.py .
```

Create `work/` first if it does not exist. See the audit README for the distinction between historical and newly executed checks.

## Evidence and limitations

The synthetic benchmark tests selective lookahead. The four-domain telemetry replay tests the exact-tail policy and information value, not selective refinement. Costs are normalized model costs, not measured signaling delays. Model-based stopping risk is not an empirical risk guarantee. The [verification report](verification/VERIFICATION.md) describes the earlier full local reruns and this distribution's exclusions. Original solver implementations are retained; only the reproduction driver has been adapted for locally supplied Alibaba inputs.

## Data attribution and license

Philly data: Microsoft Research, [Philly traces](https://github.com/msr-fiddle/philly-traces), CC BY 4.0. Cite Jeon et al., USENIX ATC 2019. Included measurements and predicates are adapted subsets (selection, time alignment, binning, resource predicates and masks). See [DATA.md](docs/DATA.md). No upstream endorsement is implied. Alibaba source: [Alibaba Cluster Trace Program v2018](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018). Only original research-level aggregate outputs and acquisition/reproduction code are distributed, not the underlying records. No third-party data is relicensed. The original software and associated software documentation are distributed under the [MIT License](LICENSE). This license does not cover datasets, manuscript text, figures, or other research outputs. The Philly data and their adaptations remain under CC BY 4.0. Cite the version-specific Zenodo DOI once the release has been archived.
