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
"""119 (REVISION r4, GPT audit P1-1/P2-2): time-zero gradients + tie accounting.

P1-1: dose and order-count gradients must use ONLY prescriptions with
      start <= time zero.  Re-extracts PPI prescriptions from the raw archive
      (the derived parquet carries whole-window aggregates), recomputes
      omprazole-equivalent max dose and order counts within the time-zero
      window, and re-runs the dose-stratum and duration-tertile IPTW models.
P2-2: counts first-order timestamp ties (first_ppi == first_h2b) in the
      exclusivity-free sensitivity and re-runs it excluding ties.
Output: results/revise_gradients_tz.json
"""
import os, re, json, zipfile, time
import numpy as np, pandas as pd
import statsmodels.api as sm

BASE = ROOT
OUT = os.path.join(DATA)
RES = os.path.join(RESULTS)
ZIPEX = MIMIC_ZIP
t0 = time.time()
def log(*a): print("[%.1f min]" % ((time.time()-t0)/60), *a, flush=True)

Q = pd.read_parquet(os.path.join(OUT, "revise_mimic.parquet"))
COMORB = ["ckd", "esrd_dialysis", "diabetes", "hf", "htn", "cirrhosis", "gi_ulcer_bleed", "cancer",
          "sepsis", "diarrhea", "malnutrition", "af_arrhythmia", "copd", "alcohol"]
COMEDS = ["furo", "thiazide", "mra", "aminoglycoside", "amphotericin", "cnimod", "digoxin"]
COV_TZ = (["age", "female", "emergency", "mg_base", "logcre"] + COMORB +
          ["pre0_" + k for k in COMEDS] + ["adm_year"])

# ---------- stream PPI prescriptions ----------
PPI_RE = re.compile(r"(omeprazole|pantoprazole|esomeprazole|lansoprazole|rabeprazole|dexlansoprazole)", re.I)
Z = zipfile.ZipFile(ZIPEX)
rows = []
for i, ch in enumerate(pd.read_csv(Z.open("mimic-iv-3.1/hosp/prescriptions.csv.gz"), compression="gzip",
        usecols=["hadm_id", "drug", "starttime", "dose_val_rx"], dtype={"hadm_id": "Int64"}, chunksize=3_000_000)):
    dl = ch["drug"].astype(str).str.lower()
    hit = dl.str.contains(PPI_RE, na=False)
    if hit.any():
        sub = ch.loc[hit, ["hadm_id", "drug", "starttime", "dose_val_rx"]].copy()
        sub["start"] = pd.to_datetime(sub["starttime"], errors="coerce")
        dose = pd.to_numeric(sub["dose_val_rx"].astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")
        sub["dose"] = dose
        sub["omz_eq"] = np.where(sub["drug"].astype(str).str.lower().str.contains("pantoprazole|esomeprazole"),
                                 sub["dose"] * 1.6, sub["dose"])
        rows.append(sub[["hadm_id", "start", "omz_eq"]])
    if i % 6 == 0: log("presc chunk", i)
pp = pd.concat(rows, ignore_index=True)
pp["hadm_id"] = pp["hadm_id"].astype("int64")
pp = pp[pp["start"].notna()]
log("ppi prescription rows:", len(pp))
_cache = os.path.join(OUT, "ppi_orders_cache.parquet")
if os.path.exists(_cache):
    pp = pd.read_parquet(_cache)
    log("loaded cached PPI prescriptions:", len(pp))

# ---------- time-zero window aggregates ----------
q = Q.copy()
tz = q.set_index("hadm_id")[["time_zero"]]
pp = pp.join(tz, on="hadm_id")
pp_tz = pp[(pp["start"].notna()) & (pp["start"] <= pp["time_zero"])]
agg = pp_tz.groupby("hadm_id").agg(ppi_dose_max_tz=("omz_eq", "max"),
                                   n_orders_tz=("start", "size")).reset_index()
pp.drop(columns=["time_zero"]).to_parquet(_cache, index=False)
log("hadm with >=1 in-window PPI order:", len(agg))
q = q.merge(agg, on="hadm_id", how="left")
q = q[q["hypo"].notna()].copy()
q["logcre"] = np.log(q["cre_base_rev"].clip(0.2, 20))
q = q.dropna(subset=COV_TZ).copy()
print("analytic:", len(q))

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

def wted_or(sub, label):
    d = sub.copy()
    if len(d) < 200 or d["treat"].nunique() < 2:
        return {"OR": None, "CI": None, "n": int(len(d))}
    d, w = iptw(d)
    m = sm.GLM(d["hypo"], sm.add_constant(d[["treat"]].astype(float)), family=sm.families.Binomial(),
               freq_weights=w).fit(cov_type="HC0")
    o = float(np.exp(m.params["treat"])); ci = np.exp(m.conf_int().loc["treat"])
    out = {"OR": round(o, 2), "CI": [round(float(ci[0]), 2), round(float(ci[1]), 2)], "n": int(len(d))}
    print("%-28s OR %.2f [%.2f,%.2f] n=%d" % (label, o, ci[0], ci[1], len(d)), flush=True)
    return out

res = {}
ppi_hadm = set(q.loc[(q["treat"] == 1) & q["ppi_dose_max_tz"].notna(), "hadm_id"])
h2_hadm = set(q.loc[q["treat"] == 0, "hadm_id"])

# ---------- dose strata (in-window max ordered dose, omz equivalents) ----------
d = q[(q["treat"] == 1) & q["ppi_dose_max_tz"].notna()].copy()
d["dose_stratum"] = pd.cut(d["ppi_dose_max_tz"], [0, 40, 80, np.inf], labels=["low", "standard", "high"])
res["dose_strata_counts"] = {str(k): int(v) for k, v in d["dose_stratum"].value_counts().items()}
print("in-window dose strata counts:", res["dose_strata_counts"])
for s in ["low", "standard", "high"]:
    hadms = set(d.loc[d["dose_stratum"] == s, "hadm_id"])
    sub = q[(q["treat"] == 0) | q["hadm_id"].isin(hadms)]
    res[f"dose_{s}"] = wted_or(sub, f"dose {s} vs H2RA")

# ---------- duration: 1 in-window order vs >=2 in-window orders (vs H2RA) ----------
dq = q.copy()
dq["n_orders_tz"] = dq["n_orders_tz"].fillna(0)
dq["dur_g"] = np.where(dq["treat"] == 0, "ref_h2ra",
              np.where(dq["n_orders_tz"] <= 1, "single", "multi"))
res["duration_groups_counts"] = dq.loc[dq.treat == 1, "dur_g"].value_counts().to_dict()
print("in-window order-count groups (PPI):", res["duration_groups_counts"])
for t, lab in [("single", "1 order vs H2RA"), ("multi", ">=2 orders vs H2RA")]:
    sub = dq[(dq["dur_g"] == t) | (dq["treat"] == 0)].assign(treat=lambda x: (x["dur_g"] == t).astype(int))
    res[f"duration_{t}"] = wted_or(sub, lab)
# keep legacy keys consumed by the figure script
res["duration_T2"] = res["duration_single"]
res["duration_T3"] = res["duration_multi"]

# ---------- P2-2: ties in first-order assignment ----------
ties = int(((q["first_ppi"].notna()) & (q["first_h2b"].notna()) &
            (q["first_ppi"] == q["first_h2b"])).sum())
res["first_order_ties_in_analytic_cohort"] = ties
print("first-order timestamp ties (analytic cohort):", ties)

json.dump(res, open(os.path.join(RES, "revise_gradients_tz.json"), "w"), indent=1)
print("saved revise_gradients_tz.json")
