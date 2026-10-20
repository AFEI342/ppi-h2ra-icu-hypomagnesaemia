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
"""108 (REVISION): eICU-CRD replication rebuilt on the time-zero design.

Mirrors the revised MIMIC-IV design:
  - exposure restricted to first acid-suppressant order within
    [-48 h, +24 h] of the unit stay (offsets in minutes);
  - baseline Mg = max in [-12 h, +24 h];  outcome = min Mg in (+24 h,
    unit discharge + 48 h];  hypomagnesaemia = nadir <= 1.7 mg/dL;
  - covariates known at time zero only: demographics, baseline labs,
    stay-level diagnosis flags (disclosed limitation), furosemide started
    on or before time zero.  No post-baseline measurement count.
Hospital-clustered standard errors throughout.
"""
import os, time, json
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats

EICU = EICU_DIR
OUT = DATA
RES = RESULTS
t0 = time.time()
def log(*a): print("[%.1f min]" % ((time.time() - t0) / 60), *a, flush=True)

E0 = pd.read_parquet(os.path.join(OUT, "analysis_eicu.parquet"))
print("previous analysis stays:", len(E0))

# ---------- exposure restriction to time-zero-knowable orders ----------
idx = E0[["first_ppi", "first_h2b"]].min(axis=1)
keep = (idx >= -2880) & (idx <= 1440)          # -48 h .. +24 h in minutes
E = E0[keep].copy()
E["index_off"] = idx[keep]
print("after exposure-window restriction: %d (dropped %d)" % (len(E), (~keep).sum()))

# ---------- magnesium rows for these stays ----------
mg = []
for i, ch in enumerate(pd.read_csv(os.path.join(EICU, "lab.csv.gz"),
        usecols=["patientunitstayid", "labname", "labresult", "labresultoffset"], chunksize=4_000_000, low_memory=False)):
    s = ch[ch["labname"].astype(str).str.lower().str.strip() == "magnesium"]
    if len(s):
        s = s.copy(); s["v"] = pd.to_numeric(s["labresult"], errors="coerce")
        s = s[(s["v"] > 0.2) & (s["v"] < 5)]
        mg.append(s[["patientunitstayid", "v", "labresultoffset"]])
    if i % 8 == 0: log("lab chunk", i)
mgl = pd.concat(mg, ignore_index=True)
mgl = mgl[mgl["patientunitstayid"].isin(set(E["patientunitstayid"]))]
log("Mg rows:", len(mgl))

# time-zero design windows (minutes from unit admission)
E2 = E[["patientunitstayid", "unitdischargeoffset"]].merge(mgl, on="patientunitstayid", how="inner")
base = E2[(E2["labresultoffset"] >= -720) & (E2["labresultoffset"] <= 1440)].groupby("patientunitstayid")["v"].max().rename("mg_base_tz")
post = E2[(E2["labresultoffset"] > 1440) & (E2["labresultoffset"] <= E2["unitdischargeoffset"] + 2880)]
outmin = post.groupby("patientunitstayid")["v"].min().rename("mg_min_tz")
nout = post.groupby("patientunitstayid")["v"].count().rename("n_mg_tz")
E = E.merge(base.reset_index(), on="patientunitstayid", how="left").merge(
    outmin.reset_index(), on="patientunitstayid", how="left").merge(
    nout.reset_index(), on="patientunitstayid", how="left")
E["hypo"] = (E["mg_min_tz"] <= 1.7).astype(float)
E.loc[E["mg_min_tz"].isna(), "hypo"] = np.nan
log("with baseline+outcome: %d | hypo rate %.1f%%" % (int(E["hypo"].notna().sum()), 100 * E["hypo"].mean()))

# furosemide started on or before time zero (offset of first order already on file)
E["pre0_furo"] = ((E["first_furo"].notna()) & (E["first_furo"] <= 1440)).astype(int)

DXFLAG = ["ckd", "esrd", "diabetes", "hf", "cirrhosis", "cancer", "sepsis", "copd", "gi_bleed", "alcohol"]
COV = ["age_n", "female", "mg_base_tz", "cre_base", "pre0_furo"] + DXFLAG
E = E.dropna(subset=["hypo"] + COV).copy()
log("analysis stays: %d | treated %d | hospitals %d" % (len(E), int(E["treat"].sum()), E["hospitalid"].nunique()))

def orci(m, term="treat"):
    o = float(np.exp(m.params[term])); ci = np.exp(m.conf_int().loc[term])
    return round(o, 2), [round(float(ci[0]), 2), round(float(ci[1]), 2)]

def gee(y, cov, wcol=None, label=""):
    if wcol is None:
        m = sm.GEE(E[y], sm.add_constant(E[cov + ["treat"]].astype(float)), groups=E["hospitalid"],
                   family=sm.families.Binomial(), cov_struct=sm.cov_struct.Independence()).fit()
    else:
        m = sm.GLM(E[y], sm.add_constant(E[cov + ["treat"]].astype(float)), family=sm.families.Binomial(),
                   freq_weights=E[wcol]).fit(cov_type="cluster", cov_kwds={"groups": E["hospitalid"]})
    o, ci = orci(m)
    print("  %-34s OR %.2f [%.2f,%.2f]" % (label, o, ci[0], ci[1]), flush=True)
    return {"OR": o, "CI": ci}

res = {}
res["crude"] = gee("hypo", [], label="crude")
res["multivariable_cluster"] = gee("hypo", COV, label="multivariable (clustered)")
# IPTW
X = sm.add_constant(E[COV].astype(float))
ps = sm.Logit(E["treat"], X).fit(disp=0).predict(X)
E["ps"] = ps
p_tr = E["treat"].mean()
w = np.where(E["treat"] == 1, p_tr / ps, (1 - p_tr) / (1 - ps))
E["w_tr"] = np.clip(w, *np.quantile(w, [0.01, 0.99]))
u = stats.mannwhitneyu(E.loc[E.treat == 1, "ps"], E.loc[E.treat == 0, "ps"]).statistic
res["ps_cstat"] = round(float(u / (E["treat"].sum() * (1 - E["treat"]).sum())), 3)
res["IPTW_cluster"] = gee("hypo", [], wcol="w_tr", label="IPTW (clustered) - primary")
lin = sm.GEE(E["mg_min_tz"], sm.add_constant(E[["treat"]].astype(float)), groups=E["hospitalid"],
             family=sm.families.Gaussian(), freq_weights=E["w_tr"]).fit()
res["mg_nadir_diff"] = {"beta": round(float(lin.params["treat"]), 3),
                        "CI": [round(float(lin.conf_int().loc["treat"][0]), 3), round(float(lin.conf_int().loc["treat"][1]), 3)]}
# measurement frequency (descriptive)
res["n_mg_by_group"] = {"ppi": round(float(E.loc[E.treat == 1, "n_mg_tz"].mean()), 2),
                        "h2ra": round(float(E.loc[E.treat == 0, "n_mg_tz"].mean()), 2)}
for k, v in res.items(): print(k, v)
E.to_parquet(os.path.join(OUT, "revise_eicu.parquet"), index=False)
json.dump(res, open(os.path.join(RES, "revise_primary_eicu.json"), "w"), indent=1)
log("DONE")
