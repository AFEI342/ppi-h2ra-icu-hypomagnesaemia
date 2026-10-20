# -*- coding: utf-8 -*-
# --- repository paths (override with environment variables) ---
import os
ROOT = os.environ.get("PPI_MG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get("PPI_MG_DATA", os.path.join(ROOT, "data"))
RESULTS = os.environ.get("PPI_MG_RESULTS", os.path.join(ROOT, "results"))
FIGURES = os.environ.get("PPI_MG_FIGURES", os.path.join(ROOT, "figures"))
MIMIC_ZIP = os.environ.get("MIMIC_IV_ZIP", os.path.join(DATA, "mimic-iv-3.1.zip"))
EICU_DIR = os.environ.get("EICU_CRD_DIR", os.path.join(DATA, "eicu-crd-2.0"))
# ------------------------------------------------------------
"""105 (REVISION, v2): rebuild MIMIC-IV analysis set, reviewer-aligned time-zero design.

Design after revision:
  - time zero = ICU admission + 24 h (end of the baseline window).
  - cohort restricted to patients whose FIRST acid-suppressant order falls in
    [icu_intime - 48 h, time zero]  -> exposure is fixed and knowable at time
    zero and strictly precedes every outcome measurement.
  - covariates: all known at time zero.  Co-prescriptions = already started on
    or before time zero (was: any use during the whole ICU stay).
    n_distinct_drugs (admission-wide) dropped.
  - baseline magnesium = max Mg in [icu_intime - 6 h, icu_intime + 24 h]
    (published definition); outcome = Mg <= 1.7 mg/dL (min) in
    (icu_intime + 24 h, icu_outtime + 48 h].
  - baseline creatinine = pre-admission value when available (cre_base_prior),
    falling back to the first-24 h minimum; disclosed in Methods.
  - admission year recovered from anchor_year / anchor_year_group (MIMIC dates
    are per-patient shifted).
Sensitivity columns also built here:
  - pre_rx_<k>: co-prescription capped at the index order (stricter).
  - mg_base_strict: max Mg in the 24 h before the index order only (subset).
  - mg_min_idx / hypo_idx: outcome window re-anchored to index
    (max(admission, index) + 24 h).
"""
import os, json, zipfile
import pandas as pd, numpy as np

OUT = DATA
RES = RESULTS

A = pd.read_parquet(os.path.join(OUT, "analysis_mimic.parquet"))
labs = pd.read_parquet(os.path.join(OUT, "labs_k_mg.parquet"))
aki = pd.read_parquet(os.path.join(OUT, "mimic_hadm_aki2.parquet"),
                      columns=["hadm_id", "cre_base_prior", "cre_base_low"])
print("cohort", A.shape, "labs", labs.shape)

# ---------- cohort: exclusive users, first order knowable by time zero ----------
q = A[((A["exp_ppi"] > 0) & (A["exp_h2b"] == 0)) | ((A["exp_h2b"] > 0) & (A["exp_ppi"] == 0))].copy()
q["treat"] = ((q["exp_ppi"] > 0) & (q["exp_h2b"] == 0)).astype(int)
q["index_time"] = q[["first_ppi", "first_h2b"]].min(axis=1)
q["time_zero"] = q["icu_intime"] + pd.Timedelta(hours=24)
n0 = len(q)
q = q[(q["index_time"] >= q["icu_intime"] - pd.Timedelta(hours=48)) &
      (q["index_time"] <= q["time_zero"])].copy()
print("exclusive users: %d | with index in [-48h, +24h]: %d (dropped %d late/very-early)" %
      (n0, len(q), n0 - len(q)))

# ---------- index-anchored sensitivity outcome (window re-anchored to index) ----------
s = labs[labs["lab"] == "Mg"].merge(
    q[["subject_id", "hadm_id", "icu_intime", "icu_outtime", "index_time"]], on="subject_id", how="inner")
s = s[(s["charttime"] >= s["icu_intime"] - pd.Timedelta(hours=6)) &
      (s["charttime"] <= s["icu_outtime"] + pd.Timedelta(hours=48))]
base_strict = s[(s["charttime"] >= s["index_time"] - pd.Timedelta(hours=24)) &
                (s["charttime"] <= s["index_time"])].groupby("hadm_id")["valuenum"].max().rename("mg_base_strict")
out_start = pd.concat([s["icu_intime"], s["index_time"]], axis=1).max(axis=1) + pd.Timedelta(hours=24)
s2 = s.assign(out_start=out_start)
post = s2[s2["charttime"] > s2["out_start"]]
outmin_idx = post.groupby("hadm_id")["valuenum"].min().rename("mg_min_idx")
nout_idx = post.groupby("hadm_id")["valuenum"].count().rename("n_mg_post_idx")
q = q.merge(pd.concat([base_strict, outmin_idx, nout_idx], axis=1).reset_index(), on="hadm_id", how="left")
q["hypo_idx"] = (q["mg_min_idx"] <= 1.7).astype(float)
q.loc[q["mg_min_idx"].isna(), "hypo_idx"] = np.nan

# ---------- negative control: hyperkalaemia, primary window ----------
sk = labs[labs["lab"] == "K"].merge(
    q[["subject_id", "hadm_id", "icu_intime", "icu_outtime", "time_zero"]], on="subject_id", how="inner")
sk = sk[(sk["charttime"] >= sk["icu_intime"] - pd.Timedelta(hours=6)) &
        (sk["charttime"] <= sk["icu_outtime"] + pd.Timedelta(hours=48))]
kmax = sk[sk["charttime"] > sk["time_zero"]].groupby("hadm_id")["valuenum"].max().rename("k_max")
q["hyperk"] = (q["k_max"] >= 5.5).astype(float)
q.loc[q["k_max"].isna(), "hyperk"] = np.nan

# ---------- time-zero covariates ----------
COMEDS = ["furo", "thiazide", "mra", "aminoglycoside", "amphotericin", "cnimod", "digoxin"]
for k in COMEDS:
    q[f"pre0_{k}"] = ((q[f"first_{k}"].notna()) & (q[f"first_{k}"] <= q["time_zero"])).astype(int)
    q[f"pre_rx_{k}"] = ((q[f"first_{k}"].notna()) & (q[f"first_{k}"] <= q["index_time"])).astype(int)
aki2 = aki.set_index("hadm_id")
q["cre_base_rev"] = q["hadm_id"].map(aki2["cre_base_prior"]).fillna(q["hadm_id"].map(aki2["cre_base_low"]))

# real admission year (MIMIC shifts dates per patient)
Z = zipfile.ZipFile(MIMIC_ZIP)
pts = pd.read_csv(Z.open("mimic-iv-3.1/hosp/patients.csv.gz"), compression="gzip",
                  usecols=["subject_id", "anchor_year", "anchor_year_group"])
pts["yr_start"] = pts["anchor_year_group"].astype(str).str.split(" - ").str[0].astype(int)
q = q.drop(columns=[c for c in ["anchor_year", "yr_start"] if c in q.columns])
q = q.merge(pts[["subject_id", "anchor_year", "yr_start"]], on="subject_id", how="left")
q["adm_year"] = q["yr_start"] + (pd.to_datetime(q["admittime"]).dt.year - q["anchor_year"])
print("adm_year range:", int(q["adm_year"].min()), "-", int(q["adm_year"].max()))
print("cre_base_rev notna: %.1f%%" % (100 * q["cre_base_rev"].notna().mean()))
print("strict pre-index Mg baseline available: %.1f%%" % (100 * q["mg_base_strict"].notna().mean()))

q.to_parquet(os.path.join(OUT, "revise_mimic.parquet"), index=False)
flow = {"exclusive": int(n0), "primary_cohort": int(len(q)),
        "with_outcome": int(q["hypo"].notna().sum()),
        "events": int(q["hypo"].sum()), "treated": int(q["treat"].sum()),
        "late_orders_dropped": int(n0 - len(q))}
json.dump(flow, open(os.path.join(RES, "revise_flow_mimic.json"), "w"), indent=1)
print("flow:", flow)
print("saved revise_mimic.parquet", q.shape)
