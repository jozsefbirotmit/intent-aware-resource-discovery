# Public-release candidate 0.2 checks

Date: 2026-11-25. Based on the full local artifact v0.1. The source measurements and time-indexed Alibaba-derived inputs are deliberately excluded; aggregate research outputs are retained. This is a conservative distribution decision, not a determination that upstream redistribution is prohibited.

- `python scripts/reproduce.py quick`: PASS in the existing Python 3.13.5 environment. All frozen v8 numerical assertions passed; ten unit tests passed (5 mechanism, 5 final replay). Table and figure data were regenerated.
- `python scripts/reproduce.py stage-alibaba --alibaba-results <local-v0.1-legacy-results> --work <temporary-directory>`: PASS. Compared 2268 policy-summary rows and 648 Brier-score rows with the retained aggregate reference. All six staged input files were byte-identical to the original frozen-input hashes. This checks staging with the historical local results, not with a newly downloaded raw trace.
- Full original-trace download, preprocessing, and the new end-to-end joint replay from regenerated raw inputs were not rerun in assembling this reduced candidate. Earlier full local replay/refit verification is documented in VERIFICATION.md. Timing equality is not claimed. A new independent environment was not installed during this assembly.
- No repository, release, or DOI has yet been created. Software license selection and the final review-policy check remain open.
