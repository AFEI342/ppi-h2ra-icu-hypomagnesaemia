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
"""Stage-1 exposure probe: DDI-combo ICU exposures in MIMIC-IV 3.1 (zip streaming) + eICU 2.0.
No extraction to disk; only compact per-admission flags parquet in project dir.
Positive control pair = vancomycin x pip/tazo (known AKI synergism).
Negative control pair = ceftriaxone x H2 blocker (mechanistically unrelated).
"""
import zipfile, os, time, json, re
import pandas as pd

ZIP = MIMIC_ZIP
EICU = EICU_DIR
OUT = DATA
RES = RESULTS
for d in (OUT, RES):
    os.makedirs(d, exist_ok=True)
t0 = time.time()
def log(*a): print("[%.1f min]" % ((time.time() - t0) / 60), *a, flush=True)

CLASS_PAT = {
    "ceftriaxone": r"ceftriaxone",
    "vancomycin":  r"vancomycin",
    "ppi":         r"(omeprazole|pantoprazole|esomeprazole|lansoprazole|rabeprazole|dexlansoprazole)",
    "h2b":         r"(famotidine|ranitidine|cimetidine|nizatidine)",
    "piptazo":     r"(piperacillin|tazobactam|zosyn)",
    "acei_arb":    r"(captopril|enalapril|lisinopril|ramipril|quinapril|perindopril|fosinopril|trandolapril|benazepril|moexipril|losartan|valsartan|irbesartan|candesartan|telmisartan|olmesartan|eprosartan|azilsartan)",
    "nsaid":       r"(ibuprofen|ketorolac|naproxen|indomethacin|diclofenac|celecoxib|meloxicam|piroxicam|nabumetone|etodolac|sulindac|ketoprofen|flurbiprofen|mefenamic|meclofenamate|tolmetin|diflunisal|oxaprozin)",
    "furosemide":  r"(furosemide|lasix)",
}
COMBO_RE = re.compile("|".join(f"(?P<{k}>{v})" for k, v in CLASS_PAT.items()), re.I)

# ---------------- MIMIC ----------------
Z = zipfile.ZipFile(ZIP)
P = "mimic-iv-3.1/"
def read(member, **kw):
    return pd.read_csv(Z.open(P + member), compression="gzip", low_memory=False, **kw)

icustays = read("icu/icustays.csv.gz", usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"])
icustays["intime"] = pd.to_datetime(icustays["intime"]); icustays["outtime"] = pd.to_datetime(icustays["outtime"])
icustays["los_h"] = (icustays["outtime"] - icustays["intime"]).dt.total_seconds() / 3600
log("icu stays:", len(icustays), "hadm:", icustays["hadm_id"].nunique())

# per-hadm ICU window (earliest intime / latest outtime), keep hadm with ANY stay
w = icustays.groupby("hadm_id").agg(intime=("intime", "min"), outtime=("outtime", "max"), n_stay=("stay_id", "count")).reset_index()

chunks = []
for i, ch in enumerate(pd.read_csv(Z.open(P + "hosp/prescriptions.csv.gz"), compression="gzip",
        usecols=["subject_id", "hadm_id", "drug", "starttime", "stoptime", "route"],
        dtype={"subject_id": int, "hadm_id": "Int64"}, chunksize=2_000_000)):
    ch["drug_l"] = ch["drug"].astype(str).str.lower()
    ch = ch[ch["drug_l"].str.contains(COMBO_RE, na=False, regex=True)]
    if len(ch):
        chunks.append(ch)
    if i % 5 == 0:
        log("  presc chunk", i, "kept rows:", sum(len(c) for c in chunks))
presc = pd.concat(chunks, ignore_index=True); del chunks
log("prescriptions kept:", len(presc))

# map each kept row to class(es)
def classify(d):
    m = COMBO_RE.search(d)
    return m.lastgroup if m else None
presc["cls"] = presc["drug_l"].map(classify)
presc = presc[presc["cls"].notna()].copy()
presc["starttime"] = pd.to_datetime(presc["starttime"], errors="coerce")
presc["stoptime"] = pd.to_datetime(presc["stoptime"], errors="coerce")
presc = presc[presc["starttime"].notna() & presc["stoptime"].notna()]
presc = presc[presc["stoptime"] > presc["starttime"]]
log("presc rows with valid interval:", len(presc), "| class counts:")
print(presc["cls"].value_counts().to_string())

# overlap per (hadm, classA, classB): union intervals per class then check pairwise overlap >= X hours
def overlap_hours(pairs1, pairs2):
    """max total overlap hours between two interval sets (vectorised brute force; sets are small)"""
    if len(pairs1) == 0 or len(pairs2) == 0:
        return 0.0
    a = pairs1; b = pairs2
    tot = 0.0
    for i in range(len(a)):
        for j in range(len(b)):
            lo = max(a.starttime.iloc[i], b.starttime.iloc[j])
            hi = min(a.stoptime.iloc[i], b.stoptime.iloc[j])
            if hi > lo:
                tot += (hi - lo).total_seconds() / 3600
    return tot

mimic_flags = []
PCTS = [("ceftriaxone", "ppi"), ("ceftriaxone", "h2b"), ("vancomycin", "ppi"), ("vancomycin", "piptazo"),
        ("acei_arb", "nsaid"), ("vancomycin", "ceftriaxone")]
hadm_list = list(w["hadm_id"])
grp = {k: v for k, v in presc.groupby(["hadm_id", "cls"])}
log("computing overlaps for", len(hadm_list), "hadms ...")
for hi, hadm in enumerate(hadm_list):
    row = {"hadm_id": hadm}
    for a, b in PCTS:
        ga = grp.get((hadm, a)); gb = grp.get((hadm, b))
        row[f"{a}|{b}"] = overlap_hours(ga, gb) if (ga is not None and gb is not None) else 0.0
    mimic_flags.append(row)
    if hi % 20000 == 0:
        log("  hadm", hi)
mf = pd.DataFrame(mimic_flags).merge(w, on="hadm_id", how="left")
mf.to_parquet(os.path.join(OUT, "mimic_combo_overlap.parquet"), index=False)
log("MIMIC overlap matrix saved:", mf.shape)

res = {"n_icu_hadm": int(len(mf))}
for a, b in PCTS:
    ov = mf[f"{a}|{b}"]
    res[f"MIMIC {a} x {b}"] = {"gte24h": int((ov >= 24).sum()), "gte48h": int((ov >= 48).sum()),
                               "gte72h": int((ov >= 72).sum()), "any": int((ov > 0).sum())}

# ---------------- eICU ----------------
pat = pd.read_csv(os.path.join(EICU, "patient.csv.gz"),
                  usecols=["patientunitstayid", "age", "unitdischargeoffset", "hospitalid"], low_memory=False)
log("eICU stays:", len(pat))
med_rows = []
for ch in pd.read_csv(os.path.join(EICU, "medication.csv.gz"),
        usecols=["patientunitstayid", "drugname", "drugstartoffset", "drugstopoffset"],
        chunksize=2_000_000, low_memory=False):
    ch["drug_l"] = ch["drugname"].astype(str).str.lower()
    ch = ch[ch["drug_l"].str.contains(COMBO_RE, na=False, regex=True)]
    if len(ch):
        med_rows.append(ch)
m = pd.concat(med_rows, ignore_index=True); del med_rows
m["cls"] = m["drug_l"].map(classify)
m = m[m["cls"].notna() & m["drugstartoffset"].notna() & m["drugstopoffset"].notna()]
m = m[m["drugstopoffset"] > m["drugstartoffset"]]
log("eICU med rows kept:", len(m))
print(m["cls"].value_counts().to_string())
m["starttime"] = pd.to_timedelta(m["drugstartoffset"], unit="m")
m["stoptime"] = pd.to_timedelta(m["drugstopoffset"], unit="m")

grp = {k: v for k, v in m.groupby(["patientunitstayid", "cls"])}
eicu_flags = []
uids = pat["patientunitstayid"].tolist()
for ui, uid in enumerate(uids):
    row = {"patientunitstayid": uid}
    for a, b in PCTS:
        ga = grp.get((uid, a)); gb = grp.get((uid, b))
        row[f"{a}|{b}"] = overlap_hours(ga, gb) if (ga is not None and gb is not None) else 0.0
    eicu_flags.append(row)
    if ui % 40000 == 0:
        log("  eICU uid", ui)
ef = pd.DataFrame(eicu_flags)
ef.to_parquet(os.path.join(OUT, "eicu_combo_overlap.parquet"), index=False)
res["n_eicu_stays"] = int(len(ef))
for a, b in PCTS:
    ov = ef[f"{a}|{b}"]
    res[f"eICU {a} x {b}"] = {"gte24h": int((ov >= 24).sum()), "gte48h": int((ov >= 48).sum()),
                              "gte72h": int((ov >= 72).sum()), "any": int((ov > 0).sum())}

json.dump(res, open(os.path.join(RES, "exposure_probe.json"), "w"), indent=1)
log("=== EXPOSURE PROBE SUMMARY ===")
for k, v in res.items():
    print(k, v)
log("DONE")
