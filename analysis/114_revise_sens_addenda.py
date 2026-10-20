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
"""114 (REVISION r3): reproducible addenda requested by GPT audit.

(a) Measurement-count-adjusted outcome model  -> revise_primary_mimic.json
    sensitivity.measurement_count_adjusted / nadir_measurement_count_adjusted
(b) AKI trio (adjusted / AKI-free restricted / within-AKI) -> sensitivity.aki_*
(c) Exclusivity-free sensitivity (GPT audit item 3): all acid-suppressant users
    (both-class patients included), exposure = class of the FIRST order,
    time-zero covariates identical -> sensitivity.first_order_assignment
(d) revise_flow_eicu.json with explicit provenance for every flow number.
"""
import os, json
import numpy as np, pandas as pd
import statsmodels.api as sm

OUT = DATA
RES = RESULTS

COMORB = ["ckd", "esrd_dialysis", "diabetes", "hf", "htn", "cirrhosis", "gi_ulcer_bleed", "cancer",
          "sepsis", "diarrhea", "malnutrition", "af_arrhythmia", "copd", "alcohol"]
COMEDS = ["furo", "thiazide", "mra", "aminoglycoside", "amphotericin", "cnimod", "digoxin"]
COV_TZ = (["age", "female", "emergency", "mg_base", "logcre"] + COMORB +
          ["pre0_" + k for k in COMEDS] + ["adm_year"])

def prep(df):
    df = df.copy()
    df["logcre"] = np.log(df["cre_base_rev"].clip(0.2, 20))
    return df.dropna(subset=[c for c in COV_TZ if c != "logcre"] + ["logcre", "hypo"]).copy()

def iptw(d, cov=COV_TZ):
    keep = []
    for c in cov:
        v = d[c]
        if v.nunique() < 2:
            continue
        if v.dtype.kind in "if" and v.dropna().isin([0, 1]).all() and min(v.sum(), (1 - v).sum()) < 10:
            continue
        keep.append(c)
    X = sm.add_constant(d[keep].astype(float))
    ps = sm.Logit(d["treat"], X).fit(disp=0).predict(X)
    p_tr = d["treat"].mean()
    w = np.where(d["treat"] == 1, p_tr / ps, (1 - p_tr) / (1 - ps))
    return d, np.clip(w, *np.quantile(w, [0.01, 0.99]))

def wted_or(d, w, extra=None):
    cols = ["treat"] + (["n_mg_post"] if extra == "nmg" else ["aki"] if extra == "aki" else [])
    m = sm.GLM(d["hypo"], sm.add_constant(d[cols].astype(float)), family=sm.families.Binomial(),
               freq_weights=w).fit(cov_type="HC0")
    term = "treat"
    o = float(np.exp(m.params[term])); ci = np.exp(m.conf_int().loc[term])
    return {"OR": round(o, 2), "CI": [round(float(ci[0]), 2), round(float(ci[1]), 2)], "n": int(len(d))}

P = json.load(open(os.path.join(RES, "revise_primary_mimic.json")))
sens = P["sensitivity"]

# ---------- (a) measurement-count adjusted ----------
Q = pd.read_parquet(os.path.join(OUT, "revise_mimic.parquet"))
q = prep(Q[Q["hypo"].notna()].copy())
d, w = iptw(q.dropna(subset=["n_mg_post"]))
sens["measurement_count_adjusted"] = wted_or(d, w, extra="nmg")
lin = sm.WLS(d["mg_min"], sm.add_constant(d[["treat", "n_mg_post"]].astype(float)), weights=w).fit(cov_type="HC0")
cb = lin.conf_int().loc["treat"]
sens["nadir_measurement_count_adjusted"] = {"beta": round(float(lin.params["treat"]), 3),
                                            "CI": [round(float(cb[0]), 3), round(float(cb[1]), 3)]}

# ---------- (b) AKI trio ----------
aki = pd.read_parquet(os.path.join(OUT, "mimic_hadm_aki2.parquet"),
                      columns=["hadm_id", "aki", "cre_base_prior", "cre_base_low"])
q = q.merge(aki, on="hadm_id", how="left")
d, w = iptw(q)
sens["aki_adjusted"] = wted_or(d, w, extra="aki")
d0, w0 = iptw(q[q["aki"] == 0])
sens["aki_free_restricted"] = wted_or(d0, w0)
d1, w1 = iptw(q[q["aki"] == 1])
sens["within_aki"] = wted_or(d1, w1)

# ---------- (c) exclusivity-free: first-order assignment on ALL users ----------
A = pd.read_parquet(os.path.join(OUT, "analysis_mimic.parquet"))
A = A[(A["exp_ppi"] > 0) | (A["exp_h2b"] > 0)].copy()
A["treat"] = (A["first_ppi"].notna() & ((A["first_h2b"].isna()) | (A["first_ppi"] < A["first_h2b"]))).astype(int)
A["index_time"] = A[["first_ppi", "first_h2b"]].min(axis=1)
A["time_zero"] = A["icu_intime"] + pd.Timedelta(hours=24)
A = A[(A["index_time"] >= A["icu_intime"] - pd.Timedelta(hours=48)) & (A["index_time"] <= A["time_zero"])].copy()
tz = A["time_zero"]
for k in COMEDS:
    A[f"pre0_{k}"] = ((A[f"first_{k}"].notna()) & (A[f"first_{k}"] <= tz)).astype(int)
ak = aki.set_index("hadm_id")
A["cre_base_rev"] = A["hadm_id"].map(ak["cre_base_prior"]).fillna(A["hadm_id"].map(ak["cre_base_low"]))
import zipfile
Z = zipfile.ZipFile(MIMIC_ZIP)
pts = pd.read_csv(Z.open("mimic-iv-3.1/hosp/patients.csv.gz"), compression="gzip",
                  usecols=["subject_id", "anchor_year", "anchor_year_group"])
pts["yr_start"] = pts["anchor_year_group"].astype(str).str.split(" - ").str[0].astype(int)
A = A.drop(columns=[c for c in ["anchor_year", "yr_start"] if c in A.columns])
A = A.merge(pts[["subject_id", "anchor_year", "yr_start"]], on="subject_id", how="left")
A["adm_year"] = A["yr_start"] + (pd.to_datetime(A["admittime"]).dt.year - A["anchor_year"])
qa = prep(A[A["hypo"].notna()].copy())
da, wa = iptw(qa)
sens["first_order_assignment"] = wted_or(da, wa)
sens["first_order_assignment"]["both_class_n"] = int((qa["exp_ppi"].notna() & (qa["exp_ppi"] > 0) & (qa["exp_h2b"] > 0)).sum())
print("first_order_assignment:", sens["first_order_assignment"])

P["sensitivity"] = sens
json.dump(P, open(os.path.join(RES, "revise_primary_mimic.json"), "w"), indent=1)
for k in ["measurement_count_adjusted", "aki_adjusted", "aki_free_restricted", "within_aki", "first_order_assignment"]:
    print(k, sens[k])

# ---------- (d) eICU flow json with provenance ----------
old = json.load(open(os.path.join(RES, "eicu_flow.json")))
E = pd.read_parquet(os.path.join(OUT, "revise_eicu.parquet"), columns=["patientunitstayid", "hospitalid"])
flow = {"screened_stays": old["screened_stays"],          # unchanged screening (old eicu_flow.json)
        "acid_suppressant": old["acid_suppressant"],      # unchanged screening (old eicu_flow.json)
        "exclusive_users": old["ppi_only"] + old["h2ra_only"],  # = 47,314 + 24,677 (old eicu_flow.json)
        "with_outcome_old_design": int(old["analysed"]),  # old analysis set (analysis_eicu.parquet)
        "first_order_in_window": int(len(E)),             # recomputed in 108 (revise_eicu.parquet)
        "analysed": int(len(E)),
        "hospitals": int(E["hospitalid"].nunique()),
        "dropped_out_of_window": int(old["analysed"] - len(E)),
        "provenance": "screened/exposed/exclusive = eicu_flow.json (screening definitions unchanged); "
                      "in-window and analysed = scripts/108_revise_eicu.py (revise_eicu.parquet)"}
json.dump(flow, open(os.path.join(RES, "revise_flow_eicu.json"), "w"), indent=1)
print("flow_eicu:", flow)
print("DONE")
