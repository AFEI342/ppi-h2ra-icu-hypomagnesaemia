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
"""90: hard-outcome extraction for the MIMIC cohort (heart rhythm -> new AF; ventilation; mortality)."""
import zipfile, os, time
import pandas as pd, numpy as np

ZIP = MIMIC_ZIP
OUT = DATA
t0 = time.time()
def log(*a): print("[%.1f min]" % ((time.time() - t0) / 60), *a, flush=True)
Z = zipfile.ZipFile(ZIP); P = "mimic-iv-3.1/"

# ---- inspect item ids ----
di = pd.read_csv(Z.open(P + "icu/d_items.csv.gz"), compression="gzip")
for kw in ["Heart Rhythm", "Atrial Fibrillation", "Invasive Ventilation", "Non-invasive Ventilation", "Ventilator Mode", "O2 Delivery Device"]:
    sub = di[di["label"].astype(str).str.contains(kw, case=False, na=False)]
    print(f"--- {kw} ---")
    print(sub[["itemid", "label", "category", "unitname"]].head(6).to_string(index=False))

coh = pd.read_parquet(os.path.join(OUT, "analysis_mimic.parquet"),
                      columns=["subject_id", "hadm_id", "stay_id", "icu_intime", "icu_outtime"])
subj = set(coh["subject_id"].astype(int))
stay2hadm = dict(zip(coh["stay_id"], coh["hadm_id"]))

# ---- heart rhythm (220048) ----
rhythm_rows = []
for i, ch in enumerate(pd.read_csv(Z.open(P + "icu/chartevents.csv.gz"), compression="gzip",
        usecols=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "value"],
        dtype={"subject_id": int, "hadm_id": "Int64", "stay_id": "Int64", "itemid": int}, chunksize=5_000_000)):
    s = ch[(ch["itemid"] == 220048) & (ch["subject_id"].isin(subj))]
    if len(s):
        rhythm_rows.append(s[["subject_id", "hadm_id", "stay_id", "charttime", "value"]])
    if i % 4 == 0: log("chartevents chunk", i, "kept", sum(len(r) for r in rhythm_rows))
rh = pd.concat(rhythm_rows, ignore_index=True) if rhythm_rows else pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "charttime", "value"])
rh.to_parquet(os.path.join(OUT, "mimic_rhythm.parquet"), index=False)
log("rhythm rows:", len(rh))
if len(rh):
    print(rh["value"].astype(str).str.strip().value_counts().head(12).to_string())

# ---- procedureevents: ventilation ----
proc_rows = []
for i, ch in enumerate(pd.read_csv(Z.open(P + "icu/procedureevents.csv.gz"), compression="gzip",
        dtype={"subject_id": int, "hadm_id": "Int64", "stay_id": "Int64", "itemid": int}, chunksize=2_000_000, low_memory=False)):
    s = ch[ch["subject_id"].isin(subj)]
    if len(s):
        proc_rows.append(s[["subject_id", "hadm_id", "stay_id", "starttime", "endtime", "itemid", "value", "valueuom"]])
pr = pd.concat(proc_rows, ignore_index=True) if proc_rows else pd.DataFrame()
pr.to_parquet(os.path.join(OUT, "mimic_procedures.parquet"), index=False)
log("procedure rows (cohort):", len(pr))
if len(pr):
    labs = di.set_index("itemid")["label"].to_dict()
    top = pr["itemid"].value_counts().head(15)
    print(pd.DataFrame({"itemid": top.index, "label": [labs.get(i, "?") for i in top.index], "n": top.values}).to_string(index=False))
log("DONE")
