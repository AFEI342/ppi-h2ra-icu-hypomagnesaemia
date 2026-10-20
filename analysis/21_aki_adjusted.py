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
"""Stage-3: adjusted AKI models for combos + PPI-vs-H2B active-comparator contrast.
One prescriptions pass -> per-hadm intervals for 5 classes + polypharmacy proxy.
"""
import zipfile, os, time, json
import pandas as pd, numpy as np
import statsmodels.api as sm

ZIP = MIMIC_ZIP
OUT = DATA
RES = RESULTS
t0 = time.time()
def log(*a): print("[%.1f min]" % ((time.time()-t0)/60), *a, flush=True)

Z = zipfile.ZipFile(ZIP); P = "mimic-iv-3.1/"
PAT = {"ceftriaxone": r"ceftriaxone", "vancomycin": r"vancomycin", "ppi": r"(omeprazole|pantoprazole|esomeprazole|lansoprazole|rabeprazole|dexlansoprazole)",
       "h2b": r"(famotidine|ranitidine|cimetidine|nizatidine)", "piptazo": r"(piperacillin|tazobactam|zosyn)"}
import re
RE = re.compile("|".join(f"(?P<{k}>{v})" for k, v in PAT.items()), re.I)

iv = {k: [] for k in PAT}          # (hadm, start, stop) per class
drugpairs = []                      # (hadm, drug) for polypharmacy
for i, ch in enumerate(pd.read_csv(Z.open(P+"hosp/prescriptions.csv.gz"), compression="gzip",
        usecols=["hadm_id", "drug", "starttime", "stoptime"], dtype={"hadm_id": "Int64"}, chunksize=3_000_000)):
    ch["drug_l"] = ch["drug"].astype(str).str.lower()
    dp = ch[["hadm_id", "drug_l"]].dropna(subset=["hadm_id"])
    dp = dp.rename(columns={"drug_l": "drug"})
    drugpairs.append(dp.drop_duplicates())
    sub = ch[ch["drug_l"].str.contains(RE, na=False)]
    if len(sub):
        sub = sub.copy()
        sub["cls"] = sub["drug_l"].map(lambda d: (RE.search(d).lastgroup if RE.search(d) else None))
        sub = sub[sub["cls"].notna()]
        st = pd.to_datetime(sub["starttime"], errors="coerce"); sp = pd.to_datetime(sub["stoptime"], errors="coerce")
        sub = sub.assign(start=st, stop=sp)
        sub = sub[sub["start"].notna() & sub["stop"].notna() & (sub["stop"] > sub["start"])]
        for cls, g in sub.groupby("cls"):
            iv[cls].append(g[["hadm_id", "start", "stop"]])
    if i % 5 == 0:
        log("presc chunk", i)
dpm = pd.concat(drugpairs, ignore_index=True).drop_duplicates()
poly = dpm.groupby("hadm_id").size().rename("n_distinct_drugs").reset_index()
poly["hadm_id"] = poly["hadm_id"].astype("int64")
log("polypharmacy computed, hadms:", len(poly))
IV = {}
for k, v in iv.items():
    if v:
        vv = pd.concat(v, ignore_index=True)
        vv["hadm_id"] = vv["hadm_id"].astype("int64")
        IV[k] = vv
    else:
        IV[k] = pd.DataFrame(columns=["hadm_id", "start", "stop"])

def ov_hours(g1, g2):
    """given two DataFrames for same hadm, sum pairwise overlap hours (ns int math)"""
    tot = 0.0
    s2 = g2["start"].astype("int64").values
    e2 = g2["stop"].astype("int64").values
    for _, r1 in g1.iterrows():
        lo = np.maximum(r1["start"].value, s2); hi = np.minimum(r1["stop"].value, e2)
        d = (hi - lo) / 3.6e12  # ns -> hours
        tot += d[d > 0].sum()
    return float(tot)

NEED = [("ceftriaxone","ppi"), ("ceftriaxone","h2b"), ("vancomycin","ppi"), ("vancomycin","piptazo"), ("vancomycin","h2b")]
g1 = {k: {h: v for h, v in vv.groupby("hadm_id")} for k, vv in IV.items()}
aki = pd.read_parquet(os.path.join(OUT, "mimic_hadm_aki.parquet"))
aki = aki.drop(columns=[c for c in aki.columns if "|" in c])  # drop stale overlap cols
rows = []
for hadm in aki["hadm_id"].astype("int64"):
    r = {}
    for a, b in NEED:
        ga = g1.get(a, {}).get(hadm); gb = g1.get(b, {}).get(hadm)
        r[f"{a}|{b}"] = ov_hours(ga, gb) if (ga is not None and gb is not None) else 0.0
    rows.append(r)
ovd = pd.DataFrame(rows)
aki2 = pd.concat([aki.reset_index(drop=True), ovd], axis=1).merge(poly, on="hadm_id", how="left")
aki2.to_parquet(os.path.join(OUT, "mimic_hadm_aki2.parquet"), index=False)
log("merged analysis file:", aki2.shape)

def logit(df, expo, label):
    d = df.dropna(subset=["aki", expo, "age", "cre_base", "n_distinct_drugs"]).copy()
    d["female"] = (d["gender"].astype(str).str.upper().str.startswith("F")).astype(int)
    d["logcre"] = np.log(d["cre_base"].clip(0.2, 20))
    d["emergency"] = (~d["admission_type"].astype(str).str.upper().str.contains("ELECTIVE")).astype(int)
    d["expo"] = d[expo].astype(int)
    X = sm.add_constant(d[["expo", "age", "female", "logcre", "emergency", "n_distinct_drugs"]])
    m = sm.Logit(d["aki"].astype(int), X).fit(disp=0)
    orv = np.exp(m.params["expo"]); ci = np.exp(m.conf_int().loc["expo"])
    base = d["aki"].mean()
    print(f"  {label}: n={len(d)}, events={int(d['aki'].sum())}, crude={100*base:.1f}%, adjOR={orv:.2f} [{ci[0]:.2f},{ci[1]:.2f}]", flush=True)
    return {"n": int(len(d)), "adjOR": round(float(orv), 3), "CI": [round(float(ci[0]), 3), round(float(ci[1]), 3)]}

out = {}
print("[CONTRAST 1] combo>=24h vs rest of cohort (adjusted)", flush=True)
for a, b in [("ceftriaxone","ppi"), ("ceftriaxone","h2b"), ("vancomycin","ppi"), ("vancomycin","piptazo"), ("vancomycin","h2b")]:
    aki2[f"combo_{a}_{b}"] = (aki2[f"{a}|{b}"] >= 24).astype(int)
    out[f"combo {a}x{b} vs rest"] = logit(aki2, f"combo_{a}_{b}", f"{a} x {b} >=24h")

print("[CONTRAST 2] within-index-drug: PPI vs H2B (active comparator)", flush=True)
cro = aki2[(aki2["ceftriaxone|ppi"] >= 24) | (aki2["ceftriaxone|h2b"] >= 24)].copy()
cro["expo"] = (cro["ceftriaxone|ppi"] >= 24).astype(int)
out["CRO users: PPI vs H2B"] = logit(cro, "expo", "ceftriaxone users: PPI vs H2B")
van = aki2[(aki2["vancomycin|ppi"] >= 24) | (aki2["vancomycin|h2b"] >= 24)].copy()
van["expo"] = (van["vancomycin|ppi"] >= 24).astype(int)
out["VAN users: PPI vs H2B"] = logit(van, "expo", "vancomycin users: PPI vs H2B")

print("[CONTRAST 3] within-PPI-users: ceftriaxone vs no ceftriaxone (dose-response prep)", flush=True)
jp = aki2[aki2["ceftriaxone|ppi"] > 0].copy()
for lo, hi, tag in [(0, 24, "1-24h"), (24, 72, "24-72h"), (72, 1e9, ">72h")]:
    jp[f"ov_{tag}"] = ((jp["ceftriaxone|ppi"] >= lo) & (jp["ceftriaxone|ppi"] < hi)).astype(int)
    out[f"CROxPPI overlap {tag} vs rest of PPI cohort"] = logit(jp, f"ov_{tag}", f"overlap {tag}")

json.dump(out, open(os.path.join(RES, "aki_adjusted.json"), "w"), indent=1)
log("DONE")
