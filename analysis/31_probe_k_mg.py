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
"""Stage-5 probe: two candidates in one pass.
  A) TMP-SMX in ICU -> hyperkalemia (K trajectory; dose split; RAASi/MRA co-exposure)
  B) PPI vs H2B in ICU -> hypomagnesemia (active comparator; Mg trajectory)
Saves per-hadm exposure flags for reuse.
"""
import zipfile, os, time, re, json
import pandas as pd, numpy as np

ZIP = MIMIC_ZIP
OUT = DATA
RES = RESULTS
t0 = time.time()
def log(*a): print("[%.1f min]" % ((time.time()-t0)/60), *a, flush=True)

Z = zipfile.ZipFile(ZIP); P = "mimic-iv-3.1/"
icu = pd.read_csv(Z.open(P+"icu/icustays.csv.gz"), compression="gzip", low_memory=False,
                  usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"])
icu["intime"] = pd.to_datetime(icu["intime"]); icu["outtime"] = pd.to_datetime(icu["outtime"])
icu["los"] = (icu["outtime"] - icu["intime"]).dt.total_seconds()/3600
icu1 = icu.sort_values("los", ascending=False).drop_duplicates("hadm_id").rename(
    columns={"intime": "icu_intime", "outtime": "icu_outtime"})
cohort = icu1[["subject_id", "hadm_id", "stay_id", "icu_intime", "icu_outtime"]].copy()
cohort["hadm_id"] = cohort["hadm_id"].astype("int64")
log("ICU hadms:", len(cohort))

PAT = {"tmp_smx": r"(trimethoprim|sulfamethoxazole|bactrim)",
       "ppi": r"(omeprazole|pantoprazole|esomeprazole|lansoprazole|rabeprazole|dexlansoprazole)",
       "h2b": r"(famotidine|ranitidine|cimetidine|nizatidine)",
       "mra": r"(spironolactone|eplerenone)",
       "acei_arb": r"(lisinopril|enalapril|captopril|ramipril|losartan|valsartan|irbesartan|candesartan|telmisartan|olmesartan)",
       "piptazo": r"(piperacillin|tazobactam|zosyn)"}
RE = re.compile("|".join(f"(?P<{k}>{v})" for k, v in PAT.items()), re.I)

rowsK = []   # key meds with dose
for i, ch in enumerate(pd.read_csv(Z.open(P+"hosp/prescriptions.csv.gz"), compression="gzip",
        usecols=["hadm_id", "drug", "starttime", "stoptime", "dose_val_rx", "dose_unit_rx"],
        dtype={"hadm_id": "Int64"}, chunksize=3_000_000)):
    ch["drug_l"] = ch["drug"].astype(str).str.lower()
    ch = ch[ch["drug_l"].str.contains(RE, na=False)]
    if len(ch):
        ch = ch.assign(cls=ch["drug_l"].map(lambda d: (RE.search(d).lastgroup if RE.search(d) else None)))
        ch = ch[ch["cls"].notna()]
        ch["start"] = pd.to_datetime(ch["starttime"], errors="coerce")
        ch["stop"] = pd.to_datetime(ch["stoptime"], errors="coerce")
        rowsK.append(ch[["hadm_id", "cls", "drug", "start", "stop", "dose_val_rx", "dose_unit_rx"]])
    if i % 6 == 0: log("presc chunk", i)
presc = pd.concat(rowsK, ignore_index=True)
presc["hadm_id"] = presc["hadm_id"].astype("int64")
presc = presc[presc["start"].notna()]
log("presc rows:", len(presc))

mf = presc.merge(cohort[["hadm_id", "icu_intime", "icu_outtime"]], on="hadm_id", how="inner")
mf = mf[(mf["stop"].isna() | (mf["stop"] >= mf["icu_intime"])) & (mf["start"] <= mf["icu_outtime"])]
flags = mf.groupby(["hadm_id", "cls"]).size().unstack(fill_value=0).add_prefix("n_").reset_index()
flags.columns = [c if c == "hadm_id" else c.replace("n_", "exp_") for c in flags.columns]
log("exposed hadms by class:")
print(flags[[c for c in flags.columns if c.startswith("exp_")]].gt(0).sum().to_string(), flush=True)

# TMP dose split: parse "160" vs "80" TMP component from dose_val_rx when unit mg and drug contains SMX/TMP
tmp = mf[mf["cls"] == "tmp_smx"].copy()
tmp["dose_num"] = pd.to_numeric(tmp["dose_val_rx"].astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")
tmp["high_dose"] = (tmp["dose_num"].fillna(0) >= 160).astype(int)   # 160mg TMP per dose = treatment dose
hd = tmp.groupby("hadm_id")["high_dose"].max().rename("tmp_high_dose")
flags = flags.merge(hd, on="hadm_id", how="left")
flags["tmp_high_dose"] = flags["tmp_high_dose"].fillna(0).astype(int)

# ---- labs: K and Mg (fluid == Blood!) ----
dl = pd.read_csv(Z.open(P+"hosp/d_labitems.csv.gz"), compression="gzip")
def ids_for(name):
    s = dl[(dl["label"].astype(str).str.contains(name, case=False, na=False)) &
           (dl["fluid"].astype(str).str.lower() == "blood")]
    return s["itemid"].astype(int).tolist(), s["label"].tolist()
k_ids, k_lab = ids_for("potassium")
mg_ids, mg_lab = ids_for("magnesium")
log("K itemids:", list(zip(k_ids, k_lab)), "| Mg itemids:", list(zip(mg_ids, mg_lab)))
keep_ids = set(k_ids) | set(mg_ids)
subj = set(cohort["subject_id"].astype(int))
rows = []
for i, ch in enumerate(pd.read_csv(Z.open(P+"hosp/labevents.csv.gz"), compression="gzip",
        usecols=["subject_id", "itemid", "charttime", "valuenum"],
        dtype={"subject_id": int, "itemid": int, "valuenum": float}, chunksize=3_000_000)):
    g = ch[ch["itemid"].isin(keep_ids) & ch["valuenum"].notna() & ch["subject_id"].isin(subj)]
    if len(g): rows.append(g)
    if i % 8 == 0: log("lab chunk", i)
labs = pd.concat(rows, ignore_index=True)
labs["charttime"] = pd.to_datetime(labs["charttime"])
labs["lab"] = np.where(labs["itemid"].isin(k_ids), "K", "Mg")
labs = labs[labs["valuenum"] > 0]
log("lab rows:", len(labs), "| K:", int((labs['lab']=='K').sum()), "| Mg:", int((labs['lab']=='Mg').sum()))
labs.to_parquet(os.path.join(OUT, "labs_k_mg.parquet"), index=False)

A = cohort.merge(flags, on="hadm_id", how="left").fillna({c: 0 for c in flags.columns if c != "hadm_id"})
A.to_parquet(os.path.join(OUT, "cohort_flags_k_mg.parquet"), index=False)

c = A.merge(labs, on="subject_id", how="inner")
inw = c[(c["charttime"] >= c["icu_intime"] - pd.Timedelta(hours=6)) & (c["charttime"] <= c["icu_outtime"] + pd.Timedelta(hours=48))]
res = {}

def rate_table(lab, exposed_mask, label, thr):
    s = inw[inw["lab"] == lab]
    if lab == "K":
        base = s[(s["charttime"] <= s["icu_intime"] + pd.Timedelta(hours=24))].groupby("hadm_id")["valuenum"].min()
        out = s[(s["charttime"] > s["icu_intime"] + pd.Timedelta(hours=24))].groupby("hadm_id")["valuenum"].max()
    else:
        base = s[(s["charttime"] <= s["icu_intime"] + pd.Timedelta(hours=24))].groupby("hadm_id")["valuenum"].max()
        out = s[(s["charttime"] > s["icu_intime"] + pd.Timedelta(hours=24))].groupby("hadm_id")["valuenum"].min()
    q = pd.DataFrame({"base": base, "out": out}).join(A.set_index("hadm_id")[["exp_" + k for k in ["tmp_smx", "ppi", "h2b", "mra", "acei_arb"]] + ["tmp_high_dose"]], how="inner")
    q["hit"] = ((q["out"] >= thr) if lab == "K" else (q["out"] <= thr))
    q["delta"] = q["out"] - q["base"]
    ex = q.index.isin(A.loc[exposed_mask, "hadm_id"])
    r1, r0 = q.loc[ex, "hit"].mean(), q.loc[~ex, "hit"].mean()
    print("="*100, flush=True)
    print(f"{label}: N_lab={len(q)} | exposed={ex.sum()} | {lab}-event rate exposed={100*r1:.1f}% vs unexposed={100*r0:.1f}%", flush=True)
    print(f"  delta {lab} mean exposed={q.loc[ex,'delta'].mean():.3f} vs unexposed={q.loc[~ex,'delta'].mean():.3f}", flush=True)
    res[label] = {"n": int(len(q)), "n_exposed": int(ex.sum()), "rate_exposed": round(100*float(r1), 1),
                  "rate_unexposed": round(100*float(r0), 1),
                  "delta_exposed": round(float(q.loc[ex, "delta"].mean()), 3),
                  "delta_unexposed": round(float(q.loc[~ex, "delta"].mean()), 3)}
    return q

A_ex = A["exp_tmp_smx"] > 0
A_ex_high = (A["exp_tmp_smx"] > 0) & (A["tmp_high_dose"] == 1)
A_ex_low = (A["exp_tmp_smx"] > 0) & (A["tmp_high_dose"] == 0)
qk = rate_table("K", A_ex, "TMP-SMX any vs none (K>=5.5)", 5.5)
rate_table("K", A_ex_high, "TMP-SMX high-dose vs none", 5.5)
rate_table("K", A_ex_low, "TMP-SMX low-dose vs none", 5.5)
rate_table("K", A_ex & (A["exp_mra"] > 0), "TMP-SMX + MRA vs none", 5.5)
rate_table("K", A_ex & (A["exp_acei_arb"] > 0), "TMP-SMX + RAASi vs none", 5.5)
rate_table("K", A_ex, "TMP-SMX any vs none (K>=6.0 severe)", 6.0)

# B: PPI vs H2B on Mg
b = inw[(inw["lab"] == "Mg") & ((A.set_index("hadm_id").loc[inw["hadm_id"], "exp_ppi"] > 0) | (A.set_index("hadm_id").loc[inw["hadm_id"], "exp_h2b"] > 0))]
mg = qk  # placeholder
label = "PPI vs H2B (Mg<=1.7)"
s = inw[inw["lab"] == "Mg"]
base = s[(s["charttime"] <= s["icu_intime"] + pd.Timedelta(hours=24))].groupby("hadm_id")["valuenum"].max()
out = s[(s["charttime"] > s["icu_intime"] + pd.Timedelta(hours=24))].groupby("hadm_id")["valuenum"].min()
q = pd.DataFrame({"base": base, "out": out}).join(A.set_index("hadm_id")[["exp_ppi", "exp_h2b"]], how="inner")
q["hit"] = q["out"] <= 1.7
ppi, h2 = q["exp_ppi"] > 0, q["exp_h2b"] > 0
print("="*100, flush=True)
print(f"Mg cohort: N_lab={len(q)} | PPI={int(ppi.sum())} H2B={int(h2.sum())}", flush=True)
print(f"  hypomagnesemia(<=1.7): PPI={100*q.loc[ppi,'hit'].mean():.1f}% vs H2B={100*q.loc[h2,'hit'].mean():.1f}%", flush=True)
print(f"  Mg drop mean: PPI={(q.loc[ppi,'out']-q.loc[ppi,'base']).mean():.3f} vs H2B={(q.loc[h2,'out']-q.loc[h2,'base']).mean():.3f}", flush=True)
res[label] = {"ppi_rate": round(100*float(q.loc[ppi,'hit'].mean()),1), "h2b_rate": round(100*float(q.loc[h2,'hit'].mean()),1),
              "ppi_n": int(ppi.sum()), "h2b_n": int(h2.sum())}
json.dump(res, open(os.path.join(RES, "probe_k_mg.json"), "w"), indent=1)
log("DONE")
