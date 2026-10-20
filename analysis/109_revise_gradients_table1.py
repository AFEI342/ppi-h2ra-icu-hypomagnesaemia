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
"""109 (REVISION): gradients, Table 1 and cumulative-incidence data on the
time-zero design (MIMIC-IV).

Timing gradient redefined: under the revised cohort every first order falls in
[-48 h, +24 h], so the old early/late (>48 h) contrast no longer exists.
New contrast: therapy already started BEFORE ICU admission (index < 0 min,
i.e. home/ED continuity) versus started during the first ICU day.
Dose strata: maximum ordered PPI dose in omeprazole equivalents per dose
(<=40 / 41-80 / >80 mg).  Duration: tertiles of PPI order count.
"""
import os, json
import numpy as np, pandas as pd
import statsmodels.api as sm

OUT = DATA
RES = RESULTS

Q = pd.read_parquet(os.path.join(OUT, "revise_mimic.parquet"))
COMORB = ["ckd", "esrd_dialysis", "diabetes", "hf", "htn", "cirrhosis", "gi_ulcer_bleed", "cancer",
          "sepsis", "diarrhea", "malnutrition", "af_arrhythmia", "copd", "alcohol"]
COMEDS = ["furo", "thiazide", "mra", "aminoglycoside", "amphotericin", "cnimod", "digoxin"]
q = Q[Q["hypo"].notna()].copy()
q["logcre"] = np.log(q["cre_base_rev"].clip(0.2, 20))
COV_TZ = (["age", "female", "emergency", "mg_base", "logcre"] + COMORB +
          ["pre0_" + k for k in COMEDS] + ["adm_year"])
q = q.dropna(subset=COV_TZ).copy()
print("analytic:", len(q))

# propensity weights (identical specification to the primary model)
X = sm.add_constant(q[COV_TZ].astype(float))
ps = sm.Logit(q["treat"], X).fit(disp=0).predict(X)
q["ps"] = ps
p_tr = q["treat"].mean()
w = np.where(q["treat"] == 1, p_tr / ps, (1 - p_tr) / (1 - ps))
q["w_tr"] = np.clip(w, *np.quantile(w, [0.01, 0.99]))

def wted_or(sub, cov, label):
    d = sub.copy()
    if len(d) < 200 or d["treat"].nunique() < 2:
        return {"OR": None, "CI": None, "n": int(len(d))}
    keep = []
    for cc in cov:
        v = d[cc]
        if v.nunique() < 2:
            continue
        if v.dtype.kind in "if" and v.dropna().isin([0, 1]).all() and min(v.sum(), (1 - v).sum()) < 10:
            continue
        keep.append(cc)
    Xs = sm.add_constant(d[keep].astype(float))
    p2 = sm.Logit(d["treat"], Xs).fit(disp=0).predict(Xs)
    p2t = d["treat"].mean()
    w2 = np.where(d["treat"] == 1, p2t / p2, (1 - p2t) / (1 - p2))
    w2 = np.clip(w2, *np.quantile(w2, [0.01, 0.99]))
    m = sm.GLM(d["hypo"], sm.add_constant(d[["treat"]].astype(float)), family=sm.families.Binomial(),
               freq_weights=w2).fit(cov_type="HC0")
    o = float(np.exp(m.params["treat"])); ci = np.exp(m.conf_int().loc["treat"])
    out = {"OR": round(o, 2), "CI": [round(float(ci[0]), 2), round(float(ci[1]), 2)], "n": int(len(d))}
    print("%-30s OR %.2f [%.2f,%.2f] n=%d" % (label, o, ci[0], ci[1], len(d)), flush=True)
    return out

res = {}

# ---------- timing: pre-admission start vs in-ICU start ----------
q["pre_adm_start"] = (q["index_time"] < q["icu_intime"]).astype(int)
ppi = q[q["treat"] == 1]
res["timing_pre_admission_vs_inicu"] = {
    "ppi_pre_adm": wted_or(q[q["pre_adm_start"] == 1], COV_TZ, "PPI started before ICU"),
    "ppi_in_icu": wted_or(q[q["pre_adm_start"] == 0], COV_TZ, "PPI started in ICU"),
    "share_pre_adm": round(float(ppi["pre_adm_start"].mean()), 3),
    "hypo_rate_pre_adm": round(float(q.loc[(q.treat == 1) & (q.pre_adm_start == 1), "hypo"].mean()), 3),
    "hypo_rate_in_icu": round(float(q.loc[(q.treat == 1) & (q.pre_adm_start == 0), "hypo"].mean()), 3),
    "hypo_rate_h2ra": round(float(q.loc[q.treat == 0, "hypo"].mean()), 3)}

# ---------- dose strata (max ordered dose, omeprazole equivalents) ----------
d = q[q["treat"] == 1].copy()
if "ppi_dose_max" in d.columns and d["ppi_dose_max"].notna().sum() > 500:
    d["dose_stratum"] = pd.cut(d["ppi_dose_max"], [0, 40, 80, np.inf], labels=["low", "standard", "high"])
    res["dose_strata_counts"] = {str(k): int(v) for k, v in d["dose_stratum"].value_counts().items()}
    for s in ["low", "standard", "high"]:
        sub = q[(q["treat"] == 1) & (q["hadm_id"].isin(d.loc[d["dose_stratum"] == s, "hadm_id"])) |
                (q["treat"] == 0)]
        res[f"dose_{s}"] = wted_or(sub, COV_TZ, f"dose {s} vs H2RA")

# ---------- duration tertiles (PPI order count) ----------
n_col = "n_ppi" if "n_ppi" in q.columns else None
print("duration column:", n_col)
if n_col:
    dq = q.copy()
    dq["n_ppi"] = dq[n_col].fillna(0)
    t1 = dq.loc[dq["treat"] == 1, "n_ppi"]
    terts = t1.quantile([1 / 3, 2 / 3]).tolist()
    dq["dur_t"] = np.where(dq["treat"] == 0, "ref_h2ra",
                  np.where(dq["n_ppi"] <= terts[0], "T1", np.where(dq["n_ppi"] <= terts[1], "T2", "T3")))
    res["duration_tertile_cuts"] = [float(terts[0]), float(terts[1])]
    for t in ["T2", "T3"]:
        sub = dq[(dq["dur_t"] == t) | (dq["treat"] == 0)]
        sub = sub.assign(treat=(sub["dur_t"] == t).astype(int))
        res[f"duration_{t}"] = wted_or(sub, COV_TZ, f"duration {t} vs H2RA")

# ---------- Table 1 ----------
rows = []
def num(v, f="%.2f"):
    return f % v if pd.notna(v) else ""

def add_row(label, series_t, series_c, kind="cont"):
    if kind == "cont":
        rows.append({"Characteristic": label,
                     "PPI": num(series_t.mean()) + " (" + num(series_t.std()) + ")",
                     "H2RA": num(series_c.mean()) + " (" + num(series_c.std()) + ")"})
    else:
        rows.append({"Characteristic": label,
                     "PPI": "%d (%.1f%%)" % (series_t.sum(), 100 * series_t.mean()),
                     "H2RA": "%d (%.1f%%)" % (series_c.sum(), 100 * series_c.mean())})

t = q[q["treat"] == 1]; c = q[q["treat"] == 0]
add_row("Age, years", t["age"], c["age"])
add_row("Female sex", t["female"], c["female"], "bin")
add_row("Emergency admission", t["emergency"], c["emergency"], "bin")
add_row("Baseline magnesium, mg/dL", t["mg_base"], c["mg_base"])
add_row("Baseline creatinine, mg/dL", t["cre_base_rev"], c["cre_base_rev"])
for k in COMORB:
    add_row(k.replace("_", " ").capitalize(), t[k], c[k], "bin")
for k in COMEDS:
    add_row("Pre-time-zero " + k, t["pre0_" + k], c["pre0_" + k], "bin")
add_row("Admission year", t["adm_year"], c["adm_year"])
add_row("ICU length of stay, days", t["los_d"], c["los_d"])
add_row("Post-baseline magnesium measurements", t["n_mg_post"], c["n_mg_post"])
pd.DataFrame(rows).to_csv(os.path.join(RES, "revise_table1.csv"), index=False)

# ---------- cumulative incidence of a first low value (daily, from time zero) ----------
labs = pd.read_parquet(os.path.join(OUT, "labs_k_mg.parquet"))
s = labs[labs["lab"] == "Mg"].merge(
    q[["subject_id", "hadm_id", "treat", "time_zero", "icu_outtime"]], on="subject_id", how="inner")
s = s[(s["charttime"] > s["time_zero"]) & (s["charttime"] <= s["icu_outtime"] + pd.Timedelta(hours=48))]
low = s[s["valuenum"] <= 1.7].sort_values("charttime").groupby("hadm_id").agg(
    first_low=("charttime", "min"), treat=("treat", "first"), time_zero=("time_zero", "first")).reset_index()
low["days"] = (low["first_low"] - pd.to_datetime(low["time_zero"])).dt.total_seconds() / 86400
den = q.groupby("treat")["hadm_id"].count()
cum = []
for tr in [1, 0]:
    n_tot = int(den[tr])
    sub = low[low["treat"] == tr]
    for day in range(0, 15):
        n_cum = int((sub["days"] <= day).sum())
        cum.append({"treat": tr, "day": day, "cum": round(100 * n_cum / n_tot, 1), "n": n_tot})
pd.DataFrame(cum).to_csv(os.path.join(RES, "revise_cum_incidence.csv"), index=False)
res["cum_incidence_d7"] = {("ppi" if tr else "h2ra"): [c["cum"] for c in cum if c["treat"] == tr and c["day"] == 7][0]
                           for tr in [1, 0]}
res["cum_incidence_end"] = {}
for tr in [1, 0]:
    sub = low[low["treat"] == tr]
    res["cum_incidence_end"]["ppi" if tr else "h2ra"] = round(100 * len(sub) / int(den[tr]), 1)

for k, v in res.items():
    if k not in ("dose_strata_counts", "ppi_share_by_year"):
        print(k, v)
json.dump(res, open(os.path.join(RES, "revise_gradients_table1.json"), "w"), indent=1)
print("saved revise_gradients_table1.json / revise_table1.csv / revise_cum_incidence.csv")
