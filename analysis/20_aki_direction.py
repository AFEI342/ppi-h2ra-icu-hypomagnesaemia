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
"""Stage-2 direction probe: KDIGO(creatinine) AKI rates for combo-exposed vs mono-exposed ICU hadms.
Reuses mimic_combo_overlap.parquet (hadm-level overlap hours). Streams labevents for creatinine.
"""
import zipfile, os, time, json
import pandas as pd
import numpy as np

ZIP = MIMIC_ZIP
OUT = DATA
RES = RESULTS
t0 = time.time()
def log(*a): print("[%.1f min]" % ((time.time() - t0) / 60), *a, flush=True)

Z = zipfile.ZipFile(ZIP)
P = "mimic-iv-3.1/"
def read(member, **kw):
    return pd.read_csv(Z.open(P + member), compression="gzip", low_memory=False, **kw)

mf = pd.read_parquet(os.path.join(OUT, "mimic_combo_overlap.parquet"))
log("overlap matrix:", mf.shape)

patients = read("hosp/patients.csv.gz", usecols=["subject_id", "gender", "anchor_age", "anchor_year"])
adm = read("hosp/admissions.csv.gz", usecols=["subject_id", "hadm_id", "admittime", "dischtime", "admission_type", "hospital_expire_flag"])
icu = read("icu/icustays.csv.gz", usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "first_careunit"])
for c in ["admittime", "dischtime"]: adm[c] = pd.to_datetime(adm[c])
for c in ["intime", "outtime"]: icu[c] = pd.to_datetime(icu[c])

# one ICU stay per hadm: longest
icu["los"] = (icu["outtime"] - icu["intime"]).dt.total_seconds() / 3600
icu1 = icu.sort_values("los", ascending=False).drop_duplicates("hadm_id")
icu1 = icu1.rename(columns={"intime": "icu_intime", "outtime": "icu_outtime"})
mf2 = mf.drop(columns=[c for c in ["intime", "outtime", "n_stay"] if c in mf.columns])
coh = mf2.merge(icu1[["hadm_id", "subject_id", "stay_id", "icu_intime", "icu_outtime", "first_careunit"]], on="hadm_id", how="inner")
coh = coh.merge(adm[["subject_id", "hadm_id", "admittime", "admission_type", "hospital_expire_flag"]], on=["hadm_id", "subject_id"], how="left")
coh = coh.merge(patients, on="subject_id", how="left")
coh["year"] = coh["admittime"].dt.year
coh["age"] = coh["anchor_age"] + (coh["year"] - coh["anchor_year"])
coh = coh[coh["age"] >= 18].copy()
log("cohort hadms:", len(coh))

# ---- creatinine streaming (full scan, keep only cohort subjects) ----
subj = set(coh["subject_id"].astype(int))
d_lab = read("hosp/d_labitems.csv.gz")
cre_ids = d_lab[(d_lab["label"].astype(str).str.contains("creatinine", case=False, na=False)) &
                (d_lab["fluid"].astype(str).str.lower() == "blood")]["itemid"].astype(int).tolist()
label_map = d_lab.set_index("itemid")["label"].to_dict()
log("serum creatinine itemids:", [(i, label_map[i]) for i in cre_ids])
rows = []
for i, ch in enumerate(pd.read_csv(Z.open(P + "hosp/labevents.csv.gz"), compression="gzip",
        usecols=["subject_id", "itemid", "charttime", "valuenum"],
        dtype={"subject_id": int, "itemid": int, "valuenum": float}, chunksize=3_000_000)):
    g = ch[ch["itemid"].isin(cre_ids) & ch["valuenum"].notna() & (ch["valuenum"] > 0) & (ch["valuenum"] < 80)]
    g = g[g["subject_id"].isin(subj)]
    if len(g):
        rows.append(g)
    if i % 5 == 0:
        log("  lab chunk", i, "kept:", sum(len(r) for r in rows))
cre = pd.concat(rows, ignore_index=True)
cre["charttime"] = pd.to_datetime(cre["charttime"])
cre = cre.sort_values(["subject_id", "charttime"])
log("creatinine rows:", len(cre))

# ---- KDIGO per hadm ----
c = coh.merge(cre, on="subject_id", how="left")
prior = c[(c["charttime"] < c["admittime"]) & (c["charttime"] >= c["admittime"] - pd.Timedelta(days=365))]
base_prior = prior.groupby("hadm_id")["valuenum"].median().rename("cre_base_prior")
inwin = c[(c["charttime"] >= c["icu_intime"] - pd.Timedelta(hours=6)) & (c["charttime"] <= c["icu_outtime"] + pd.Timedelta(hours=48))]
base_low = inwin[inwin["charttime"] <= inwin["icu_intime"] + pd.Timedelta(hours=24)].groupby("hadm_id")["valuenum"].min().rename("cre_base_low")
peak = inwin.groupby("hadm_id")["valuenum"].max().rename("cre_peak")
q = coh.set_index("hadm_id").join([base_prior, base_low, peak])
q["cre_base"] = q["cre_base_prior"].fillna(q["cre_base_low"])
q["aki_ratio"] = q["cre_peak"] / q["cre_base"]
q["aki"] = ((q["aki_ratio"] >= 1.5) | ((q["cre_peak"] - q["cre_base"]) >= 0.3)).fillna(False)
q = q[q["cre_base"].notna() & q["cre_peak"].notna()].copy()
log("hadms with usable creatinine:", len(q), "| AKI rate overall: %.1f%%" % (100 * q["aki"].mean()))
q.reset_index().to_parquet(os.path.join(OUT, "mimic_hadm_aki.parquet"), index=False)

# ---- contrasts ----
PAIRS = [("ceftriaxone", "ppi"), ("ceftriaxone", "h2b"), ("vancomycin", "ppi"),
         ("vancomycin", "piptazo"), ("acei_arb", "nsaid"), ("vancomycin", "ceftriaxone")]
res = {}
def rate(mask):
    s = q[mask]
    return len(s), (100 * s["aki"].mean() if len(s) else float("nan"))

import math
def orci(a_aki, a_n, b_aki, b_n):
    # crude OR with Woolf CI
    a1, a0, b1, b0 = a_aki, a_n - a_aki, b_aki, b_n - b_aki
    if min(a1, a0, b1, b0) == 0: return None, None
    orv = (a1 * b0) / (a0 * b1)
    se = math.sqrt(1/a1 + 1/a0 + 1/b1 + 1/b0)
    return orv, (math.exp(math.log(orv) - 1.96*se), math.exp(math.log(orv) + 1.96*se))

for a, b in PAIRS:
    hasA = q[f"{a}|{b}"] > 0  # any overlap
    combo = q[f"{a}|{b}"] >= 24
    n_c, r_c = rate(combo)
    n_a, r_a = rate((~hasA))
    o, ci = orci(int(q.loc[combo, "aki"].sum()), n_c, int(q.loc[~hasA, "aki"].sum()), n_a)
    res[f"{a} x {b}"] = {"n_combo>=24h": n_c, "aki_rate_combo_%": round(r_c, 1),
                         "n_no_overlap": n_a, "aki_rate_nooverlap_%": round(r_a, 1),
                         "crude_OR": (round(o, 2) if o else None), "OR_CI": ([round(x, 2) for x in ci] if ci else None)}

# cleaner contrasts: combo vs each mono using per-class exposure flags
# rebuild class-level exposure flags from overlap matrix (any overlap with each partner implies class presence)
cls_partners = {}
for a, b in PAIRS:
    cls_partners.setdefault(a, set()).add(b)
    cls_partners.setdefault(b, set()).add(a)
for cls, partners in cls_partners.items():
    q[f"exp_{cls}"] = q[[f"{cls}|{p}" if f"{cls}|{p}" in q.columns else f"{p}|{cls}" for p in partners]].max(axis=1) > 0

for a, b in PAIRS:
    combo = q[f"{a}|{b}"] >= 24
    mono_a = q[f"exp_{a}"] & (~combo)
    mono_b = q[f"exp_{b}"] & (~combo)
    for tag, m in [("A-mono", mono_a), ("B-mono", mono_b)]:
        n_c, r_c = rate(combo); n_m, r_m = rate(m)
        if n_m < 50: continue
        o, ci = orci(int(q.loc[combo, "aki"].sum()), n_c, int(q.loc[m, "aki"].sum()), n_m)
        res[f"{a} x {b} vs {tag}"] = {"n_combo": n_c, "rate_combo": round(r_c,1), "n_mono": n_m, "rate_mono": round(r_m,1),
                                      "OR": (round(o,2) if o else None), "CI": ([round(x,2) for x in ci] if ci else None)}
json.dump(res, open(os.path.join(RES, "aki_direction.json"), "w"), indent=1)
print("="*100)
for k, v in res.items():
    print(k, v)
log("DONE")
