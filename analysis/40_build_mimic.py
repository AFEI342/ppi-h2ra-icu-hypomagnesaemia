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
"""40: Build final MIMIC-IV analysis dataset for PPI vs H2B / hypomagnesaemia.
Exposures, co-meds, comorbidities (ICD), demographics, baseline Mg/Cr, outcome.
Output: analysis_mimic.parquet  (one row per hadm, exclusive acid-suppressant users)
"""
import zipfile, os, time, re, json
import pandas as pd, numpy as np

ZIP = MIMIC_ZIP
OUT = DATA
RES = RESULTS
t0 = time.time()
def log(*a): print("[%.1f min]" % ((time.time()-t0)/60), *a, flush=True)
Z = zipfile.ZipFile(ZIP); P = "mimic-iv-3.1/"

cohort = pd.read_parquet(os.path.join(OUT, "cohort_flags_k_mg.parquet"))
labs = pd.read_parquet(os.path.join(OUT, "labs_k_mg.parquet"))
log("cohort", cohort.shape, "labs", labs.shape)

# ---------- medications ----------
DRUGS = {"ppi": r"(omeprazole|pantoprazole|esomeprazole|lansoprazole|rabeprazole|dexlansoprazole)",
         "h2b": r"(famotidine|ranitidine|cimetidine|nizatidine)",
         "furo": r"(furosemide|lasix)",
         "thiazide": r"(hydrochlorothiazide|chlorthalidone|metolazone|indapamide)",
         "mra": r"(spironolactone|eplerenone)",
         "aminoglycoside": r"(gentamicin|tobramycin|amikacin|streptomycin|neomycin|plazomicin)",
         "amphotericin": r"amphotericin",
         "cnimod": r"(tacrolimus|cyclosporine)",
         "digoxin": r"digoxin"}
RD = re.compile("|".join(f"(?P<{k}>{v})" for k, v in DRUGS.items()), re.I)
rows = []
for i, ch in enumerate(pd.read_csv(Z.open(P+"hosp/prescriptions.csv.gz"), compression="gzip",
        usecols=["hadm_id", "drug", "starttime", "stoptime", "dose_val_rx", "dose_unit_rx", "route"],
        dtype={"hadm_id": "Int64"}, chunksize=3_000_000)):
    ch["drug_l"] = ch["drug"].astype(str).str.lower()
    ch = ch[ch["drug_l"].str.contains(RD, na=False)]
    if len(ch):
        ch = ch.assign(cls=ch["drug_l"].map(lambda d: (RD.search(d).lastgroup if RD.search(d) else None)))
        ch = ch[ch["cls"].notna()].copy()
        ch["start"] = pd.to_datetime(ch["starttime"], errors="coerce")
        ch["stop"] = pd.to_datetime(ch["stoptime"], errors="coerce")
        rows.append(ch[["hadm_id", "cls", "drug", "start", "stop", "dose_val_rx", "dose_unit_rx", "route"]])
    if i % 6 == 0: log("presc chunk", i)
med = pd.concat(rows, ignore_index=True)
med["hadm_id"] = med["hadm_id"].astype("int64")
med = med[med["start"].notna()]
log("med rows:", len(med))

A = cohort.merge(med, on="hadm_id", how="left", suffixes=("", "_m"))
# within ICU window (+allow start up to 48h before ICU for "early")
A_w = A[(A["cls"].isna()) | ((A["stop"].isna() | (A["stop"] >= A["icu_intime"] - pd.Timedelta(hours=48))) &
                             (A["start"] <= A["icu_outtime"]))]
g = A_w.dropna(subset=["cls"]).groupby(["hadm_id", "cls"]).agg(
    first=("start", "min"), last=("stop", "max"), n=("start", "size")).reset_index()
ex = g.pivot(index="hadm_id", columns="cls", values="first").reset_index()
dur = g.pivot(index="hadm_id", columns="cls", values="n").reset_index()
ex.columns = ["hadm_id"] + ["first_" + c for c in ex.columns[1:]]
dur.columns = ["hadm_id"] + ["n_" + c for c in dur.columns[1:]]
A = cohort.merge(ex, on="hadm_id", how="left").merge(dur, on="hadm_id", how="left")
for c in [c for c in A.columns if c.startswith("first_") or c.startswith("n_")]:
    if c.startswith("n_"): A[c] = A[c].fillna(0).astype(int)
A["exp_ppi"] = A["first_ppi"].notna().astype(int)
A["exp_h2b"] = A["first_h2b"].notna().astype(int)
for k in ["furo", "thiazide", "mra", "aminoglycoside", "amphotericin", "cnimod", "digoxin"]:
    A[f"exp_{k}"] = A[f"first_{k}"].notna().astype(int) if f"first_{k}" in A.columns else 0
# PPI dose intensity: sum of numeric dose_val_rx per hadm (mg/day proxy) and timing vs ICU
ppi_med = med[med["cls"] == "ppi"].copy()
ppi_med["dose_num"] = pd.to_numeric(ppi_med["dose_val_rx"].astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")
ppi_med["omz_eq"] = np.where(ppi_med["drug"].astype(str).str.lower().str.contains("pantoprazole|esomeprazole"), ppi_med["dose_num"] * 1.6, ppi_med["dose_num"])
ppiagg = ppi_med.groupby("hadm_id").agg(ppi_dose_mean=("omz_eq", "mean"), ppi_dose_max=("omz_eq", "max"),
                                        ppi_orders=("omz_eq", "size")).reset_index()
A = A.merge(ppiagg, on="hadm_id", how="left")

# ---------- comorbidities ----------
ICD = {"ckd": [r"^N18", r"^585", r"^N19", r"^586"], "esrd_dialysis": [r"^N186", r"^5856", r"^Z992", r"^V451"],
       "diabetes": [r"^E1[0-4]", r"^250"], "hf": [r"^I50", r"^428"], "htn": [r"^I1[0-3]", r"^40[1-5]"],
       "cirrhosis": [r"^K74", r"^571"], "gi_ulcer_bleed": [r"^K2[5-7]", r"^K92", r"^53[1-4]", r"^578"],
       "cancer": [r"^C[0-9]"], "sepsis": [r"^A4[01]", r"^R65", r"^038", r"^9959"], "diarrhea": [r"^A09", r"^K52", r"^78791", r"^R197"],
       "malnutrition": [r"^E4[346]", r"^263"], "af_arrhythmia": [r"^I48", r"^4273"], "copd": [r"^J4[0-4]", r"^49[1-6]"],
       "alcohol": [r"^F10", r"^K70", r"^291", r"^303"], "ckd_stage5": [r"^N185", r"^5855"]}
dx = pd.read_csv(Z.open(P+"hosp/diagnoses_icd.csv.gz"), compression="gzip",
                 usecols=["hadm_id", "icd_code", "icd_version"], dtype={"hadm_id": "int64", "icd_code": str})
flags = {}
m_code = dx["icd_code"].astype(str).str.upper().str.replace(".", "", regex=False)
for name, pats in ICD.items():
    hit = pd.Series(False, index=dx.index)
    for pat in pats:
        hit |= m_code.str.match(pat.lstrip("^"), na=False)
    flags[name] = dx.loc[hit].groupby("hadm_id").size().rename(name)
dxf = pd.concat(flags.values(), axis=1).fillna(0).astype(int).reset_index() if flags else pd.DataFrame({"hadm_id": []})
A = A.merge(dxf, on="hadm_id", how="left")
for name in ICD: A[name] = A.get(name, pd.Series(0, index=A.index)).fillna(0).astype(int)

# verify icd regex sanity
chk = {name: int(A[name].sum()) for name in ICD}
log("comorbidity counts:", chk)

# ---------- demographics, baseline ----------
pts = pd.read_csv(Z.open(P+"hosp/patients.csv.gz"), compression="gzip", usecols=["subject_id", "gender", "anchor_age", "anchor_year"])
adm = pd.read_csv(Z.open(P+"hosp/admissions.csv.gz"), compression="gzip", usecols=["hadm_id", "admittime", "admission_type", "hospital_expire_flag"], dtype={"hadm_id": "int64"})
adm["admittime"] = pd.to_datetime(adm["admittime"])
A = A.merge(pts, on="subject_id", how="left").merge(adm, on="hadm_id", how="left")
A["age"] = A["anchor_age"] + (A["admittime"].dt.year - A["anchor_year"])
A["female"] = A["gender"].astype(str).str.upper().str.startswith("F").astype(int)
A["emergency"] = (~A["admission_type"].astype(str).str.upper().str.contains("ELECTIVE")).astype(int)
A["los_d"] = (A["icu_outtime"] - A["icu_intime"]).dt.total_seconds() / 86400

# baseline creatinine from earlier AKI file
aki = pd.read_parquet(os.path.join(OUT, "mimic_hadm_aki2.parquet"))
aki = aki.drop(columns=[c for c in aki.columns if "|" in c])
A = A.merge(aki[["hadm_id", "cre_base", "n_distinct_drugs"]], on="hadm_id", how="left")

# Mg outcome + pre-ICU Mg
s = labs[labs["lab"] == "Mg"]
sm = s.merge(A[["subject_id", "hadm_id", "icu_intime", "icu_outtime", "first_ppi", "first_h2b"]], on="subject_id", how="inner")
sm = sm[(sm["charttime"] >= sm["icu_intime"] - pd.Timedelta(hours=6)) & (sm["charttime"] <= sm["icu_outtime"] + pd.Timedelta(hours=48))]
base = sm[sm["charttime"] <= sm["icu_intime"] + pd.Timedelta(hours=24)].groupby("hadm_id")["valuenum"].max().rename("mg_base")
post = sm[sm["charttime"] > sm["icu_intime"] + pd.Timedelta(hours=24)]
outmin = post.groupby("hadm_id")["valuenum"].min().rename("mg_min")
firstv = post.sort_values("charttime").groupby("hadm_id")["valuenum"].first().rename("mg_first")
nout = post.groupby("hadm_id")["valuenum"].count().rename("n_mg_post")
A = A.merge(pd.concat([base, outmin, firstv, nout], axis=1).reset_index(), on="hadm_id", how="left")
A["hypo"] = (A["mg_min"] <= 1.7).astype(float)
A["hypo_sev"] = (A["mg_min"] <= 1.5).astype(float)
A.loc[A["mg_min"].isna(), ["hypo", "hypo_sev"]] = np.nan

# K negative control
sk = labs[labs["lab"] == "K"]
skm = sk.merge(A[["subject_id", "hadm_id", "icu_intime", "icu_outtime"]], on="subject_id", how="inner")
skm = skm[(skm["charttime"] >= skm["icu_intime"] - pd.Timedelta(hours=6)) & (skm["charttime"] <= skm["icu_outtime"] + pd.Timedelta(hours=48))]
kmax = skm[skm["charttime"] > skm["icu_intime"] + pd.Timedelta(hours=24)].groupby("hadm_id")["valuenum"].max().rename("k_max")
A = A.merge(kmax.reset_index(), on="hadm_id", how="left")
A["hyperk"] = (A["k_max"] >= 5.5).astype(float)

A.to_parquet(os.path.join(OUT, "analysis_mimic.parquet"), index=False)
log("saved analysis_mimic.parquet", A.shape)
print("exclusive groups:", ((A.exp_ppi > 0) & (A.exp_h2b == 0)).sum(), "PPI-only |",
      ((A.exp_h2b > 0) & (A.exp_ppi == 0)).sum(), "H2B-only |",
      ((A.exp_ppi > 0) & (A.exp_h2b > 0)).sum(), "both")
print("with outcome:", A["hypo"].notna().sum(), "| hypo rate:", round(100 * A["hypo"].mean(), 1))
log("DONE")
