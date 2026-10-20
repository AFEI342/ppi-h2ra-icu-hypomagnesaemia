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
"""106 (REVISION): primary and sensitivity models on the time-zero design (MIMIC-IV).

Primary covariate set = variables known at time zero (ICU admission + 24 h):
demographics, baseline Mg/Cr, ICD comorbidity flags, co-prescriptions started
on or before time zero, admission year.  n_distinct_drugs removed.
The published covariate set (full-stay co-prescriptions + n_distinct_drugs) is
reproduced on the same revised cohort for head-to-head comparison, and on the
original cohort to confirm the published numbers.
"""
import os, json
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats

OUT = DATA
RES = RESULTS

Q = pd.read_parquet(os.path.join(OUT, "revise_mimic.parquet"))
A_OLD = pd.read_parquet(os.path.join(OUT, "analysis_mimic.parquet"))

COMEDS = ["furo", "thiazide", "mra", "aminoglycoside", "amphotericin", "cnimod", "digoxin"]
COMORB = ["ckd", "esrd_dialysis", "diabetes", "hf", "htn", "cirrhosis", "gi_ulcer_bleed", "cancer",
          "sepsis", "diarrhea", "malnutrition", "af_arrhythmia", "copd", "alcohol"]

q = Q[Q["hypo"].notna()].copy()
q["logcre"] = np.log(q["cre_base_rev"].clip(0.2, 20))
COV_TZ = (["age", "female", "emergency", "mg_base", "logcre"] + COMORB +
          ["pre0_" + k for k in COMEDS] + ["adm_year"])
q = q.dropna(subset=["hypo"] + COV_TZ).copy()
print("revised analytic n =", len(q), "| treated =", int(q["treat"].sum()),
      "| events =", int(q["hypo"].sum()), "| rate = %.1f%%" % (100 * q["hypo"].mean()))

def orci(m, term="treat"):
    o = float(np.exp(m.params[term])); ci = np.exp(m.conf_int().loc[term])
    return round(o, 2), [round(float(ci[0]), 2), round(float(ci[1]), 2)], float(m.pvalues[term])

def evalue(orv, lo):
    f = lambda o: o + np.sqrt(o * (o - 1))
    return round(float(f(orv) if orv > 1 else f(1 / orv)), 2), round(float(f(lo) if lo > 1 else f(1 / lo)), 2)

def iptw_frame(d, cov):
    d = d.copy()
    # drop covariates with no variation or near-separation (protects the Logit fit)
    keep = []
    for c in cov:
        v = d[c]
        if v.nunique() < 2:
            continue
        if v.dtype.kind in "if" and v.dropna().isin([0, 1]).all() and min(v.sum(), (1 - v).sum()) < 10:
            continue
        keep.append(c)
    dropped = [c for c in cov if c not in keep]
    if dropped:
        print("  [iptw_frame] dropped low-variance covariates:", dropped)
    cov = keep
    d.attrs["cov_used"] = list(cov)
    X = sm.add_constant(d[cov].astype(float))
    ps = sm.Logit(d["treat"], X).fit(disp=0).predict(X)
    d = d.copy(); d["ps"] = ps
    p_tr = d["treat"].mean()
    w = np.where(d["treat"] == 1, p_tr / ps, (1 - p_tr) / (1 - ps))
    lo, hi = np.quantile(w, [0.01, 0.99])
    d["w_tr"] = np.clip(w, lo, hi)
    u = stats.mannwhitneyu(d.loc[d.treat == 1, "ps"], d.loc[d.treat == 0, "ps"]).statistic
    d.attrs["cstat"] = float(u / (d["treat"].sum() * (1 - d["treat"]).sum()))
    return d

res = {}
d = iptw_frame(q, COV_TZ)
print("PS c-statistic:", round(d.attrs["cstat"], 3))

res["crude"] = dict(zip(["OR", "CI", "p"], orci(sm.Logit(d["hypo"], sm.add_constant(d[["treat"]].astype(float))).fit(disp=0))))
res["multivariable"] = dict(zip(["OR", "CI", "p"], orci(sm.Logit(d["hypo"], sm.add_constant(d[COV_TZ + ["treat"]].astype(float))).fit(disp=0))))
m3 = sm.GLM(d["hypo"], sm.add_constant(d[["treat"]].astype(float)), family=sm.families.Binomial(),
            freq_weights=d["w_tr"]).fit(cov_type="HC0")
res["IPTW"] = dict(zip(["OR", "CI", "p"], orci(m3)))
res["doubly_robust"] = dict(zip(["OR", "CI", "p"], orci(sm.GLM(d["hypo"], sm.add_constant(d[COV_TZ + ["treat"]].astype(float)),
        family=sm.families.Binomial(), freq_weights=d["w_tr"]).fit(cov_type="HC0"))))
# PSM 1:1
d["lps"] = np.log(d["ps"] / (1 - d["ps"]))
cal = 0.2 * d["lps"].std()
tr = d[d.treat == 1].sort_values("lps"); ct = d[d.treat == 0].sort_values("lps").reset_index(drop=True)
used, pairs = set(), []
for _, r in ct.iterrows():
    cand = tr[(tr["lps"] - r["lps"]).abs() <= cal]
    cand = cand[~cand.index.isin(used)]
    if len(cand):
        best = (cand["lps"] - r["lps"]).abs().idxmin()
        used.add(best); pairs.append(r.name)
md = pd.concat([tr.loc[list(used)], ct.loc[pairs]]).reset_index(drop=True)
res["PSM_1to1"] = dict(zip(["OR", "CI", "p"], orci(sm.Logit(md["hypo"], sm.add_constant(md[["treat"]].astype(float))).fit(disp=0))))
res["PSM_1to1"]["n_pairs"] = len(md) // 2
# overlap weights
ow = np.where(d["treat"] == 1, 1 - d["ps"], d["ps"])
mow = sm.GLM(d["hypo"], sm.add_constant(d[["treat"]].astype(float)), family=sm.families.Binomial(),
             freq_weights=ow).fit(cov_type="HC0")
res["overlap_weights"] = dict(zip(["OR", "CI", "p"], orci(mow)))
# continuous + severe + negative control (IPTW)
lin = sm.WLS(d["mg_min"], sm.add_constant(d[["treat"]].astype(float)), weights=d["w_tr"]).fit(cov_type="HC0")
res["mg_nadir_diff"] = {"beta": round(float(lin.params["treat"]), 3),
                        "CI": [round(float(lin.conf_int().loc["treat"][0]), 3), round(float(lin.conf_int().loc["treat"][1]), 3)]}
linadj = sm.WLS(d["mg_min"], sm.add_constant(d[COV_TZ + ["treat"]].astype(float)), weights=d["w_tr"]).fit(cov_type="HC0")
res["mg_nadir_diff_adj"] = {"beta": round(float(linadj.params["treat"]), 3),
                            "CI": [round(float(linadj.conf_int().loc["treat"][0]), 3), round(float(linadj.conf_int().loc["treat"][1]), 3)]}
dsev = d.dropna(subset=["hypo_sev"])
msev = sm.GLM(dsev["hypo_sev"], sm.add_constant(dsev[["treat"]].astype(float)), family=sm.families.Binomial(),
              freq_weights=dsev["w_tr"]).fit(cov_type="HC0")
res["severe_hypo_IPTW"] = dict(zip(["OR", "CI", "p"], orci(msev)))
dk = d.dropna(subset=["hyperk"])
mk = sm.GLM(dk["hyperk"], sm.add_constant(dk[["treat"]].astype(float)), family=sm.families.Binomial(),
            freq_weights=dk["w_tr"]).fit(cov_type="HC0")
res["hyperk_IPTW"] = dict(zip(["OR", "CI", "p"], orci(mk)))
res["E_value_IPTW"] = list(evalue(res["IPTW"]["OR"], res["IPTW"]["CI"][0]))
# measurement frequency (descriptive)
res["n_mg_post_by_group"] = {"ppi": round(float(d.loc[d.treat == 1, "n_mg_post"].mean()), 2),
                             "h2ra": round(float(d.loc[d.treat == 0, "n_mg_post"].mean()), 2)}

# ---------- sensitivity analyses ----------
sens = {}
# (1) co-prescriptions capped at the index order
COV_RX = (["age", "female", "emergency", "mg_base", "logcre"] + COMORB +
          ["pre_rx_" + k for k in COMEDS] + ["adm_year"])
dr = iptw_frame(q, COV_RX)
mr = sm.GLM(dr["hypo"], sm.add_constant(dr[["treat"]].astype(float)), family=sm.families.Binomial(),
            freq_weights=dr["w_tr"]).fit(cov_type="HC0")
sens["comed_capped_at_index"] = dict(zip(["OR", "CI", "p"], orci(mr)))
# (2) strict pre-index baseline magnesium subset
qs = q[q["mg_base_strict"].notna()].copy()
qs["mg_base"] = qs["mg_base_strict"]
ds = iptw_frame(qs, COV_TZ)
ms = sm.GLM(ds["hypo"], sm.add_constant(ds[["treat"]].astype(float)), family=sm.families.Binomial(),
            freq_weights=ds["w_tr"]).fit(cov_type="HC0")
sens["strict_preindex_baseline"] = dict(zip(["OR", "CI", "p"], orci(ms)))
sens["strict_preindex_baseline"]["n"] = int(len(ds))
# (3) outcome window re-anchored to the index order
qi = q[q["hypo_idx"].notna()].copy()
di = iptw_frame(qi, COV_TZ)
mi = sm.GLM(di["hypo_idx"], sm.add_constant(di[["treat"]].astype(float)), family=sm.families.Binomial(),
            freq_weights=di["w_tr"]).fit(cov_type="HC0")
sens["index_anchored_outcome"] = dict(zip(["OR", "CI", "p"], orci(mi)))
sens["index_anchored_outcome"]["n"] = int(len(di))
# (4) first post-baseline value
qf = q[q["mg_first"].notna()].copy()
qf["hypo_first"] = (qf["mg_first"] <= 1.7).astype(float)
df_ = iptw_frame(qf, COV_TZ)
mf = sm.GLM(df_["hypo_first"], sm.add_constant(df_[["treat"]].astype(float)), family=sm.families.Binomial(),
            freq_weights=df_["w_tr"]).fit(cov_type="HC0")
sens["first_value_outcome"] = dict(zip(["OR", "CI", "p"], orci(mf)))
# (5) LOS >= 2 days
ql = q[q["los_d"] >= 2].copy()
dl = iptw_frame(ql, COV_TZ)
ml = sm.GLM(dl["hypo"], sm.add_constant(dl[["treat"]].astype(float)), family=sm.families.Binomial(),
            freq_weights=dl["w_tr"]).fit(cov_type="HC0")
sens["los_ge2d"] = dict(zip(["OR", "CI", "p"], orci(ml)))
# (6) drop year covariate
COV_NOYR = [c for c in COV_TZ if c != "adm_year"]
dn = iptw_frame(q, COV_NOYR)
mn = sm.GLM(dn["hypo"], sm.add_constant(dn[["treat"]].astype(float)), family=sm.families.Binomial(),
            freq_weights=dn["w_tr"]).fit(cov_type="HC0")
sens["no_year_covariate"] = dict(zip(["OR", "CI", "p"], orci(mn)))
# (7) year strata
med_yr = q["adm_year"].median()
for lbl, sub in [("year_early", q[q["adm_year"] <= med_yr]), ("year_late", q[q["adm_year"] > med_yr])]:
    dy = iptw_frame(sub, COV_NOYR)
    my = sm.GLM(dy["hypo"], sm.add_constant(dy[["treat"]].astype(float)), family=sm.families.Binomial(),
                freq_weights=dy["w_tr"]).fit(cov_type="HC0")
    sens[lbl] = dict(zip(["OR", "CI", "p"], orci(my)))
    sens[lbl]["n"] = int(len(dy))
res["sensitivity"] = sens
res["adm_year_median_split"] = float(med_yr)
# descriptive: PPI share by year
res["ppi_share_by_year"] = {int(y): [int(g["size"]), round(float(g["mean"]), 3)]
                            for y, g in q.groupby("adm_year")["treat"].agg(["size", "mean"]).iterrows()}

# ---------- head-to-head: published covariate set on the revised cohort ----------
COV_OLD = (["age", "female", "emergency", "mg_base", "logcre"] + COMORB +
           ["exp_" + k for k in COMEDS] + ["n_distinct_drugs"])
dold = q.dropna(subset=["n_distinct_drugs"]).copy()
dold = iptw_frame(dold, COV_OLD)
mold = sm.GLM(dold["hypo"], sm.add_constant(dold[["treat"]].astype(float)), family=sm.families.Binomial(),
              freq_weights=dold["w_tr"]).fit(cov_type="HC0")
res["published_covars_on_revised_cohort"] = dict(zip(["OR", "CI", "p"], orci(mold)))
res["published_covars_on_revised_cohort"]["n"] = int(len(dold))

# ---------- reproduce published model on the original cohort ----------
ao = A_OLD[((A_OLD["exp_ppi"] > 0) & (A_OLD["exp_h2b"] == 0)) | ((A_OLD["exp_h2b"] > 0) & (A_OLD["exp_ppi"] == 0))].copy()
ao["treat"] = ((ao["exp_ppi"] > 0) & (ao["exp_h2b"] == 0)).astype(int)
ao["logcre"] = np.log(ao["cre_base"].clip(0.2, 20))
ao = ao.dropna(subset=["hypo"] + COV_OLD).copy()
da = iptw_frame(ao, COV_OLD)
ma = sm.GLM(da["hypo"], sm.add_constant(da[["treat"]].astype(float)), family=sm.families.Binomial(),
            freq_weights=da["w_tr"]).fit(cov_type="HC0")
res["published_model_reproduced"] = dict(zip(["OR", "CI", "p"], orci(ma)))
res["published_model_reproduced"]["n"] = int(len(da))

# ---------- three-arm comparison (no acid suppression as reference) ----------
import zipfile as _zf
aki = pd.read_parquet(os.path.join(OUT, "mimic_hadm_aki2.parquet"),
                      columns=["hadm_id", "cre_base_prior", "cre_base_low"])
_Z = _zf.ZipFile(MIMIC_ZIP)
_pts = pd.read_csv(_Z.open("mimic-iv-3.1/hosp/patients.csv.gz"), compression="gzip",
                   usecols=["subject_id", "anchor_year", "anchor_year_group"])
_pts["yr_start"] = _pts["anchor_year_group"].astype(str).str.split(" - ").str[0].astype(int)
_aki2 = aki.set_index("hadm_id")

def _tz_covars(df):
    df = df.copy()
    tz = df["icu_intime"] + pd.Timedelta(hours=24)
    for k in COMEDS:
        df[f"pre0_{k}"] = ((df[f"first_{k}"].notna()) & (df[f"first_{k}"] <= tz)).astype(int)
    df["cre_base_rev"] = df["hadm_id"].map(_aki2["cre_base_prior"]).fillna(df["hadm_id"].map(_aki2["cre_base_low"]))
    df = df.drop(columns=[c for c in ["anchor_year", "yr_start"] if c in df.columns])
    df = df.merge(_pts[["subject_id", "anchor_year", "yr_start"]], on="subject_id", how="left")
    df["adm_year"] = df["yr_start"] + (pd.to_datetime(df["admittime"]).dt.year - df["anchor_year"])
    return df

nonusers = A_OLD[(A_OLD["exp_ppi"] == 0) & (A_OLD["exp_h2b"] == 0)]
allh = pd.concat([_tz_covars(q), _tz_covars(nonusers)], ignore_index=True)
allh["treat_ppi"] = allh["exp_ppi"].astype(int)
allh["treat_h2ra"] = allh["exp_h2b"].astype(int)
allh["logcre"] = np.log(allh["cre_base_rev"].clip(0.2, 20))
COV_3ARM = (["age", "female", "emergency", "mg_base", "logcre"] + COMORB +
            ["pre0_" + k for k in COMEDS] + ["adm_year"])
allh = allh.dropna(subset=["hypo"] + COV_3ARM).copy()
X3 = sm.add_constant(allh[COV_3ARM + ["treat_ppi", "treat_h2ra"]].astype(float))
m3a = sm.Logit(allh["hypo"], X3).fit(disp=0)
r3 = {}
for t in ["treat_ppi", "treat_h2ra"]:
    o = float(np.exp(m3a.params[t])); ci = np.exp(m3a.conf_int().loc[t])
    r3[t] = {"OR": round(o, 2), "CI": [round(float(ci[0]), 2), round(float(ci[1]), 2)]}
res["three_arm_adjusted"] = r3
res["three_arm_adjusted"]["n"] = int(len(allh))

for k, v in res.items():
    if not isinstance(v, dict) or k in ("ppi_share_by_year", "sensitivity"):
        continue
    print(k, v)
print("--- sensitivity ---")
for k, v in sens.items(): print(k, v)
json.dump(res, open(os.path.join(RES, "revise_primary_mimic.json"), "w"), indent=1)
print("saved revise_primary_mimic.json")
