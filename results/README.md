# Results files

All files here are aggregate outputs; no patient-level data are included.

| File | Status |
|---|---|
| `revise_primary_mimic.json` | Current: primary models, continuous nadir, severe hypomagnesaemia, negative control, E-value, sensitivity analyses, three-arm comparison, covariate-set head-to-head |
| `revise_primary_eicu.json` | Current: eICU-CRD replication |
| `revise_gradients_tz.json` | Current: dose and order-count gradients restricted to prescriptions issued on or before time zero |
| `revise_gradients_table1.json` | Partly superseded: timing gradient and cumulative-incidence inputs are current; the dose and order-count gradients in this file use whole-window prescriptions and are retained for provenance only. Use `revise_gradients_tz.json` for those quantities. |
| `revise_af_mimic.json` | Current: exploratory atrial fibrillation and landmark analysis |
| `revise_flow_mimic.json`, `revise_flow_eicu.json` | Current: cohort flow |
| `revise_table1.csv` | Current: Table 1 |
| `revise_cum_incidence.csv` | Current: cumulative-incidence series |
| `revise_hyperk_probe.json` | Current: negative-control sensitivity without the admission-year term |
| `faers_ror.csv`, `faers_drug_counts.json` | Current: FAERS reporting odds ratios and report counts |
