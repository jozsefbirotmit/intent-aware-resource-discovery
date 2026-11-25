# Intent-Aware Sequential Resource Discovery under Uncertain Domain Knowledge

**Public-release candidate 0.2 — matched to manuscript v8 (2026-11-25).**

Dániel Ficzere, Attila Frankó, Gergely Hollósi, Pál Varga, József Bíró.

This candidate includes the algorithm implementations, synthetic benchmark, Philly telemetry subset, manuscript figures, and aggregate research results. Alibaba-2018 machine data, time-indexed derived states and fitted inputs are **not redistributed**. They must be regenerated locally from the provider's source under its terms. The full local v0.1 archive is retained separately by the authors. No GitHub release or Zenodo DOI has yet been created.

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

## Evidence and limitations

The synthetic benchmark tests selective lookahead. The four-domain telemetry replay tests the exact-tail policy and information value, not selective refinement. Costs are normalized model costs, not measured signaling delays. Model-based stopping risk is not an empirical risk guarantee. The [verification report](verification/VERIFICATION.md) describes the earlier full local reruns and this distribution's exclusions. Original solver implementations are retained; only the reproduction driver has been adapted for locally supplied Alibaba inputs.

## Data attribution and license

Philly data: Microsoft Research, [Philly traces](https://github.com/msr-fiddle/philly-traces), CC BY 4.0. Cite Jeon et al., USENIX ATC 2019. Included measurements and predicates are adapted subsets (selection, time alignment, binning, resource predicates and masks). See [DATA.md](docs/DATA.md). No upstream endorsement is implied. Alibaba source: [Alibaba Cluster Trace Program v2018](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2018). Only original research-level aggregate outputs and acquisition/reproduction code are distributed, not the underlying records. No third-party data is relicensed. Software license selection and final release metadata remain to be confirmed before making this candidate public.
