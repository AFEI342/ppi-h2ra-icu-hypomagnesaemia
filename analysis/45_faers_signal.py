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
"""45: FAERS leg - ROR (with 95% CI) for magnesium-related adverse event reporting,
PPIs vs H2RAs, using exact openFDA totals (report counts).
"""
import urllib.request, urllib.parse, json, time, math, os
import pandas as pd

RES = RESULTS
BASE = "https://api.fda.gov/drug/event.json"
PTS = ["Hypomagnesaemia", "Blood magnesium decreased", "Magnesium deficiency", "Hypocalcaemia"]

def call(search, count=None, retries=4):
    url = BASE + "?search=" + urllib.parse.quote(search, safe="+:")
    if count: url += "&count=" + urllib.parse.quote(count, safe="+.")
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 epi"})
            return json.loads(urllib.request.urlopen(req, timeout=60).read())
        except urllib.error.HTTPError as e:
            if e.code == 404: return {"meta": {"results": {"total": 0}}, "results": []}
            time.sleep(2)
        except Exception: time.sleep(2)
    return None

def total(search):
    d = call(search); return None if d is None else d.get("meta", {}).get("results", {}).get("total")

def pt_total(drug, pt):
    return total(f'patient.drug.medicinalproduct:{drug}+AND+patient.reaction.reactionmeddrapt:"{pt}"')

DRUGS_PPI = ["omeprazole", "pantoprazole", "esomeprazole", "lansoprazole", "rabeprazole"]
DRUGS_H2B = ["famotidine", "ranitidine", "cimetidine", "nizatidine"]
rows = []
cache = {}
for drug in DRUGS_PPI + DRUGS_H2B:
    N = total(f"patient.drug.medicinalproduct:{drug}"); time.sleep(0.4)
    ptc = {}
    for pt in PTS:
        ptc[pt] = pt_total(drug, pt); time.sleep(0.4)
    cache[drug] = {"N": N, "pt": ptc}
    rows.append({"drug": drug, "class": "PPI" if drug in DRUGS_PPI else "H2RA", "N_total": N, **{f"n_{pt}": ptc[pt] for pt in PTS}})

df = pd.DataFrame(rows)

def ror_ci(a, n1, b, n2):
    a1, a0, b1, b0 = a, n1 - a, b, n2 - b
    if min(a1, a0, b1, b0) <= 0: return None, None, None
    orv = (a1 * b0) / (a0 * b1)
    se = math.sqrt(1/a1 + 1/a0 + 1/b1 + 1/b0)
    return orv, (math.exp(math.log(orv) - 1.96*se), math.exp(math.log(orv) + 1.96*se)), se

# comparator = famotidine (most used H2RA, still marketed)
fam = cache["famotidine"]
out = []
for drug in DRUGS_PPI:
    d = cache[drug]
    for label, pts in [("Hypomagnesaemia", ["Hypomagnesaemia"]),
                       ("Mg-decreased (incl. blood Mg decreased)", ["Hypomagnesaemia", "Blood magnesium decreased", "Magnesium deficiency"])]:
        a = sum(d["pt"][p] for p in pts); b_ = fam["pt"]
        c = sum(b_[p] for p in pts)
        orv, ci, se = ror_ci(a, d["N"], c, fam["N"])
        out.append({"drug": drug, "outcome": label, "n_drug": a, "rate_drug_%": round(100*a/d["N"], 3),
                    "n_fam": c, "rate_fam_%": round(100*c/fam["N"], 3),
                    "ROR": round(orv, 2), "CI_low": round(ci[0], 2), "CI_high": round(ci[1], 2)})
o = pd.DataFrame(out)
print(o.to_string(index=False), flush=True)
o.to_csv(os.path.join(RES, "faers_ror.csv"), index=False)

# class-level (sum across drugs; multi-drug reports double-counted -> acknowledged)
for cls, drugs in [("PPI", DRUGS_PPI), ("H2RA", DRUGS_H2B)]:
    N = sum(cache[d]["N"] for d in drugs)
    a = sum(sum(cache[d]["pt"][p] for p in ["Hypomagnesaemia"]) for d in drugs)
    print(cls, "sum-N:", N, "sum-Hypomagnesaemia:", a, "rate/1000:", round(1000*a/N, 2))
json.dump(cache, open(os.path.join(RES, "faers_drug_counts.json"), "w"), indent=1)
print("saved")
