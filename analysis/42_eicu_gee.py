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
"""42: eICU replication - PPI vs H2B hypomagnesaemia with IPTW + hospital-clustered GEE.
"""
import os, time, json
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats

EICU = EICU_DIR
OUT = DATA
RES = RESULTS
t0 = time.time()
def log(*a): print("[%.1f min]" % ((time.time()-t0)/60), *a, flush=True)

pat = pd.read_csv(os.path.join(EICU, "patient.csv.gz"),
                  usecols=["patientunitstayid", "age", "gender", "unitdischargeoffset", "hospitalid"], low_memory=False)
pat["age_n"] = pd.to_numeric(pat["age"].astype(str).str.replace(">", "", regex=False), errors="coerce")
pat["female"] = pat["gender"].astype(str).str.upper().str.startswith("F").astype(int)

PPI = r"(omeprazole|pantoprazole|esomeprazole|lansoprazole|rabeprazole|dexlansoprazole)"
H2B = r"(famotidine|ranitidine|cimetidine|nizatidine)"
FURO = r"(furosemide|lasix)"
med_rows = []
for ch in pd.read_csv(os.path.join(EICU, "medication.csv.gz"),
        usecols=["patientunitstayid", "drugname", "drugstartoffset"], chunksize=2_000_000, low_memory=False):
    d = ch["drugname"].astype(str).str.lower()
    isppi = d.str.contains(PPI, na=False); ish2 = d.str.contains(H2B, na=False); isf = d.str.contains(FURO, na=False)
    sub = ch[isppi | ish2 | isf].copy()
    if len(sub):
        sub["cls"] = np.where(isppi[sub.index], "ppi", np.where(ish2[sub.index], "h2b", "furo"))
        med_rows.append(sub[["patientunitstayid", "cls", "drugstartoffset"]])
med = pd.concat(med_rows, ignore_index=True)
first = med.pivot_table(index="patientunitstayid", columns="cls", values="drugstartoffset", aggfunc="min").reset_index()
first = first.rename(columns={"ppi": "first_ppi", "h2b": "first_h2b", "furo": "first_furo"})
first["exp_ppi"] = first["first_ppi"].notna().astype(int)
first["exp_h2b"] = first["first_h2b"].notna().astype(int)
first["exp_furo"] = first["first_furo"].notna().astype(int)
log("med stays:", len(first))

mg_rows, cr_rows = [], []
for i, ch in enumerate(pd.read_csv(os.path.join(EICU, "lab.csv.gz"),
        usecols=["patientunitstayid", "labname", "labresult", "labresultoffset"], chunksize=4_000_000, low_memory=False)):
    ln = ch["labname"].astype(str).str.lower().str.strip()
    s = ch[ln.isin(["magnesium", "creatinine"])]
    if len(s):
        s = s.copy(); s["lab"] = np.where(s["labname"].astype(str).str.lower().str.strip() == "magnesium", "Mg", "Cr")
        s["v"] = pd.to_numeric(s["labresult"], errors="coerce")
        s = s[s["v"].notna()]
        mg_rows.append(s[s["lab"] == "Mg"]); cr_rows.append(s[s["lab"] == "Cr"])
    if i % 8 == 0: log("lab chunk", i)
mgl = pd.concat(mg_rows, ignore_index=True); mgl = mgl[(mgl["v"] > 0.2) & (mgl["v"] < 5)]
crl = pd.concat(cr_rows, ignore_index=True); crl = crl[(crl["v"] > 0.1) & (crl["v"] < 30)]
log("Mg rows:", len(mgl), "| Cr rows:", len(crl))

# comorbidities from diagnosisstring
dx = pd.read_csv(os.path.join(EICU, "diagnosis.csv.gz"), usecols=["patientunitstayid", "diagnosisstring"], low_memory=False)
ds = dx["diagnosisstring"].astype(str).str.lower()
DXFLAG = {"ckd": "chronic kidney disease|renal failure|renal insufficiency", "esrd": "esrd|end-stage renal|end stage renal",
          "diabetes": "diabetes", "hf": "heart failure|congestive heart failure", "cirrhosis": "cirrhosis|cirrhotic",
          "cancer": "neoplasm|metastatic|carcinoma|malignan", "sepsis": "sepsis|septic", "copd": "chronic obstructive|copd",
          "gi_bleed": "gastrointestinal bleed|gi bleed|gastrointestinal hemorrhage", "alcohol": "alcohol"}
for k, patn in DXFLAG.items():
    dx[k] = ds.str.contains(patn, na=False).astype(int)
dxf = dx.groupby("patientunitstayid")[[k for k in DXFLAG]].max().reset_index()
print("comorbidity counts:", {k: int(dxf[k].sum()) for k in DXFLAG}, flush=True)

d = first.merge(pat, on="patientunitstayid", how="inner").merge(mgl, on="patientunitstayid", how="inner")
d = d[(d["labresultoffset"] >= -720) & (d["labresultoffset"] <= d["unitdischargeoffset"] + 2880)]
base = d[d["labresultoffset"] <= 1440].groupby("patientunitstayid")["v"].max().rename("mg_base")
out = d[d["labresultoffset"] > 1440].groupby("patientunitstayid")["v"].min().rename("mg_min")
nmg = d[d["labresultoffset"] > 1440].groupby("patientunitstayid")["v"].count().rename("n_mg")
cr1 = crl[crl["labresultoffset"] <= 1440].groupby("patientunitstayid")["v"].min().rename("cre_base")
Q = first.merge(pat, on="patientunitstayid", how="left").merge(dxf, on="patientunitstayid", how="left").set_index("patientunitstayid").join([base, out, nmg, cr1]).dropna(subset=["mg_base", "mg_min"])
Q["hypo"] = (Q["mg_min"] <= 1.7).astype(int)
Q = Q[((Q["exp_ppi"] > 0) & (Q["exp_h2b"] == 0)) | ((Q["exp_h2b"] > 0) & (Q["exp_ppi"] == 0))].copy()
Q["treat"] = ((Q["exp_ppi"] > 0) & (Q["exp_h2b"] == 0)).astype(int)
for k in DXFLAG: Q[k] = Q.get(k, pd.Series(0, index=Q.index)).fillna(0).astype(int)
COV = ["age_n", "female", "mg_base", "cre_base", "exp_furo", "n_mg"] + list(DXFLAG)
Q = Q.dropna(subset=["hypo"] + COV).copy()
log("analysis stays:", len(Q), "| treated:", int(Q["treat"].sum()), "| hypo rate: %.1f%%" % (100 * Q["hypo"].mean()))
print("hospitals:", Q["hospitalid"].nunique())

res = {}
# crude
m = sm.GLM(Q["hypo"], sm.add_constant(Q[["treat"]].astype(float)), family=sm.families.Binomial()).fit()
o = float(np.exp(m.params["treat"])); ci = np.exp(m.conf_int().loc["treat"])
res["crude"] = {"OR": round(o, 2), "CI": [round(float(ci[0]), 2), round(float(ci[1]), 2)]}
# multivariable GEE cluster-robust by hospital
m2 = sm.GEE(Q["hypo"], sm.add_constant(Q[COV + ["treat"]].astype(float)),
            groups=Q["hospitalid"], family=sm.families.Binomial(), cov_struct=sm.cov_struct.Independence()).fit()
o = float(np.exp(m2.params["treat"])); ci = np.exp(m2.conf_int().loc["treat"])
res["multivariable_GEE"] = {"OR": round(o, 2), "CI": [round(float(ci[0]), 2), round(float(ci[1]), 2)], "p": float(m2.pvalues["treat"])}
# PS + IPTW within eICU
X = sm.add_constant(Q[COV].astype(float))
ps = sm.Logit(Q["treat"], X).fit(disp=0).predict(X)
Q["ps"] = ps
p_tr = Q["treat"].mean()
Q["w"] = np.where(Q["treat"] == 1, p_tr / Q["ps"], (1 - p_tr) / (1 - Q["ps"]))
lo, hi = Q["w"].quantile([0.01, 0.99]); Q["w"] = Q["w"].clip(lo, hi)
m3 = sm.GLM(Q["hypo"], sm.add_constant(Q[["treat"]].astype(float)), family=sm.families.Binomial(),
            freq_weights=Q["w"]).fit(cov_type="HC0")
o = float(np.exp(m3.params["treat"])); ci = np.exp(m3.conf_int().loc["treat"])
res["IPTW"] = {"OR": round(o, 2), "CI": [round(float(ci[0]), 2), round(float(ci[1]), 2)]}
# IPTW + hospital-clustered robust SE (weighted GLM with cluster-robust covariance)
m4 = sm.GLM(Q["hypo"], sm.add_constant(Q[COV + ["treat"]].astype(float)), family=sm.families.Binomial(),
            freq_weights=Q["w"]).fit(cov_type="cluster", cov_kwds={"groups": Q["hospitalid"]})
o = float(np.exp(m4.params["treat"])); ci = np.exp(m4.conf_int().loc["treat"])
res["IPTW_cluster"] = {"OR": round(o, 2), "CI": [round(float(ci[0]), 2), round(float(ci[1]), 2)], "p": float(m4.pvalues["treat"])}
# multivariable + cluster-robust (same estimator family as above, unweighted)
m5 = sm.GLM(Q["hypo"], sm.add_constant(Q[COV + ["treat"]].astype(float)), family=sm.families.Binomial()).fit(
    cov_type="cluster", cov_kwds={"groups": Q["hospitalid"]})
o = float(np.exp(m5.params["treat"])); ci = np.exp(m5.conf_int().loc["treat"])
res["multivariable_cluster"] = {"OR": round(o, 2), "CI": [round(float(ci[0]), 2), round(float(ci[1]), 2)], "p": float(m5.pvalues["treat"])}
# continuous
lin = sm.GEE(Q["mg_min"], sm.add_constant(Q[["treat"]].astype(float)), groups=Q["hospitalid"],
             family=sm.families.Gaussian(), freq_weights=Q["w"]).fit()
res["mg_nadir_diff"] = {"beta": round(float(lin.params["treat"]), 3),
                        "CI": [round(float(lin.conf_int().loc["treat"][0]), 3), round(float(lin.conf_int().loc["treat"][1]), 3)]}
uf = stats.mannwhitneyu(Q.loc[Q.treat == 1, "ps"], Q.loc[Q.treat == 0, "ps"]).statistic
res["ps_cstat"] = round(float(uf / (Q["treat"].sum() * (1 - Q["treat"]).sum())), 4)
for k, v in res.items(): print(k, v)
Q.reset_index().to_parquet(os.path.join(OUT, "analysis_eicu.parquet"), index=False)
json.dump(res, open(os.path.join(RES, "primary_eicu.json"), "w"), indent=1)
log("DONE")
