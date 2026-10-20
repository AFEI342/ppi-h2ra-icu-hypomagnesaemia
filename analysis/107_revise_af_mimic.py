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
"""107 (REVISION): exploratory downstream outcome - new-onset atrial fibrillation.

Reviewer #1 asked for an outcome plausibly downstream of hypomagnesaemia, e.g.
arrhythmia.  Exploratory only (not a prespecified endpoint, single centre,
rhythm documented at nursing discretion); reported in the supplement.

Definitions (time-zero design):
  - baseline AF: any AF/flutter rhythm record on or before time zero
    (ICU admission + 24 h) -> excluded from the new-onset analysis.
  - new AF: first AF/flutter record in (time_zero, icu_outtime + 24 h].
  - PPI vs H2RA: IPTW with the revised time-zero covariate set.
  - landmark: early hypomagnesaemia (min Mg in (time_zero, +24 h] <= 1.7)
    -> AF after time_zero + 48 h (adjusted logistic, class-adjusted).
"""
import os, json
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats

OUT = DATA
RES = RESULTS

Q = pd.read_parquet(os.path.join(OUT, "revise_mimic.parquet"))
rh = pd.read_parquet(os.path.join(OUT, "mimic_rhythm.parquet"))
print("cohort", Q.shape, "rhythm", rh.shape)

COMORB = ["ckd", "esrd_dialysis", "diabetes", "hf", "htn", "cirrhosis", "gi_ulcer_bleed", "cancer",
          "sepsis", "diarrhea", "malnutrition", "af_arrhythmia", "copd", "alcohol"]
COMEDS = ["furo", "thiazide", "mra", "aminoglycoside", "amphotericin", "cnimod", "digoxin"]
q = Q[Q["hypo"].notna()].copy()
q["logcre"] = np.log(q["cre_base_rev"].clip(0.2, 20))
COV_TZ = (["age", "female", "emergency", "mg_base", "logcre"] + COMORB +
          ["pre0_" + k for k in COMEDS] + ["adm_year"])
q = q.dropna(subset=COV_TZ).copy()

# ---------- AF flags ----------
rh["value_u"] = rh["value"].astype(str).str.upper().str.strip()
AF_PAT = r"ATRIAL FIB|AFIB|A ?FIB|^\s*AF\b|ATRIAL FLUTTER|A ?FLUTTER"
rh["is_af"] = rh["value_u"].str.contains(AF_PAT, regex=True, na=False)
rh["charttime"] = pd.to_datetime(rh["charttime"])
m = rh[rh["is_af"]].merge(q[["stay_id", "icu_outtime", "time_zero"]], on="stay_id", how="inner")
base_af = set(m.loc[m["charttime"] <= m["time_zero"], "stay_id"])
new_af = set(m.loc[(m["charttime"] > m["time_zero"]) &
                   (m["charttime"] <= m["icu_outtime"] + pd.Timedelta(hours=24)), "stay_id"])
q["af_baseline"] = q["stay_id"].isin(base_af).astype(int)
q["af_new"] = np.where(q["af_baseline"] == 1, np.nan, q["stay_id"].isin(new_af).astype(float))
print("baseline AF: %d | eligible AF-free: %d | new AF: %d" %
      (int(q["af_baseline"].sum()), int(q["af_new"].notna().sum()), int(np.nansum(q["af_new"]))))

# ---------- propensity weights (same as primary) ----------
X = sm.add_constant(q[COV_TZ].astype(float))
ps = sm.Logit(q["treat"], X).fit(disp=0).predict(X)
q["ps"] = ps
p_tr = q["treat"].mean()
w = np.where(q["treat"] == 1, p_tr / ps, (1 - p_tr) / (1 - ps))
q["w_tr"] = np.clip(w, *np.quantile(w, [0.01, 0.99]))

res = {}
daf = q[q["af_new"].notna()]
maf = sm.GLM(daf["af_new"], sm.add_constant(daf[["treat"]].astype(float)), family=sm.families.Binomial(),
             freq_weights=daf["w_tr"]).fit(cov_type="HC0")
o = float(np.exp(maf.params["treat"])); ci = np.exp(maf.conf_int().loc["treat"])
res["ppi_vs_h2ra_new_af_IPTW"] = {"n": int(len(daf)), "events": int(daf["af_new"].sum()),
                                  "OR": round(o, 2), "CI": [round(float(ci[0]), 2), round(float(ci[1]), 2)]}
print("PPI vs H2RA -> new AF (IPTW):", res["ppi_vs_h2ra_new_af_IPTW"])

# ---------- landmark: early hypo -> later AF ----------
labs = pd.read_parquet(os.path.join(OUT, "labs_k_mg.parquet"))
s = labs[labs["lab"] == "Mg"].merge(q[["subject_id", "stay_id", "icu_outtime", "time_zero"]], on="subject_id", how="inner")
s = s[(s["charttime"] > s["time_zero"]) & (s["charttime"] <= s["time_zero"] + pd.Timedelta(hours=24))]
early = s.groupby("stay_id")["valuenum"].min().rename("mg_early")
q2 = q.merge(early, on="stay_id", how="left")
q2["hypo_early"] = (q2["mg_early"] <= 1.7).astype(float)
m2 = rh[rh["is_af"]].merge(q2[["stay_id", "icu_outtime", "time_zero"]], on="stay_id", how="inner")
af_after48 = set(m2.loc[(m2["charttime"] > m2["time_zero"] + pd.Timedelta(hours=48)) &
                        (m2["charttime"] <= m2["icu_outtime"] + pd.Timedelta(hours=24)), "stay_id"])
q2["af_after48"] = np.where(q2["af_baseline"] == 1, np.nan, q2["stay_id"].isin(af_after48).astype(float))
dl = q2.dropna(subset=["hypo_early", "af_after48"])
ml = sm.GLM(dl["af_after48"], sm.add_constant(dl[["hypo_early", "treat"] + COV_TZ].astype(float)),
            family=sm.families.Binomial()).fit(cov_type="HC0")
o = float(np.exp(ml.params["hypo_early"])); ci = np.exp(ml.conf_int().loc["hypo_early"])
res["landmark_hypo_early_to_af"] = {"n": int(len(dl)), "events": int(dl["af_after48"].sum()),
                                    "OR": round(o, 2), "CI": [round(float(ci[0]), 2), round(float(ci[1]), 2)]}
print("landmark early hypo -> AF after 48 h:", res["landmark_hypo_early_to_af"])

json.dump(res, open(os.path.join(RES, "revise_af_mimic.json"), "w"), indent=1)
print("saved revise_af_mimic.json")
