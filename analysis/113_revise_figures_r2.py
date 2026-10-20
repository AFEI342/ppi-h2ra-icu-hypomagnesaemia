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
"""113 (REVISION r2): regenerate Figures 1-6 fixing all visual-gate findings.
Replaces 110 output.  300 dpi PNG into figs_revise/.
"""
import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = ROOT
RES = os.path.join(RESULTS)
OUTF = FIGURES
os.makedirs(OUTF, exist_ok=True)
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans", "axes.linewidth": 0.8})
RED, BLUE, TEAL, GREY = "#C0392B", "#2860A8", "#148F77", "#8E8E8E"

prim = json.load(open(os.path.join(RES, "revise_primary_mimic.json")))
eic = json.load(open(os.path.join(RES, "revise_primary_eicu.json")))
grad_tz = json.load(open(os.path.join(RES, "revise_gradients_tz.json")))
grad = json.load(open(os.path.join(RES, "revise_gradients_table1.json")))
flow_m = json.load(open(os.path.join(RES, "revise_flow_mimic.json")))
flow_e = json.load(open(os.path.join(RES, "revise_flow_eicu.json")))
sens = prim["sensitivity"]
af = json.load(open(os.path.join(RES, "revise_af_mimic.json")))

def orci(d):
    return d["OR"], d["CI"][0], d["CI"][1]

# ---------------- Figure 1 ----------------
fig = plt.figure(figsize=(7.2, 5.6))
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
def box(x, y, w, h, text, fc="white", ec="black", fs=7.2):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.9, color="black"))
ax.text(0.18, 0.975, "MIMIC-IV (primary)", fontsize=9.5, fontweight="bold", ha="center")
ax.text(0.015, 0.975, "a", fontsize=11, fontweight="bold", ha="left", va="center")
box(0.02, 0.895, 0.32, 0.055, "85,242 ICU admissions screened")
arrow(0.18, 0.895, 0.18, 0.858)
box(0.02, 0.802, 0.32, 0.055, "62,859 with acid suppression")
arrow(0.18, 0.802, 0.18, 0.765)
box(0.02, 0.709, 0.32, 0.055, "54,454 exclusive users of one class")
arrow(0.18, 0.709, 0.18, 0.672)
box(0.02, 0.604, 0.32, 0.068, f"{flow_m['primary_cohort']:,} first order within\n[-48 h, +24 h] of ICU admission")
arrow(0.18, 0.604, 0.18, 0.567)
box(0.02, 0.492, 0.32, 0.075, f"39,594 analysed\n(PPI {21447:,} / H2RA {18147:,})\nhypomagnesaemia 26.8%", fc="#F2F2F2")
ax.text(0.82, 0.975, "eICU-CRD (replication)", fontsize=9.5, fontweight="bold", ha="center")
ax.text(0.655, 0.975, "b", fontsize=11, fontweight="bold", ha="left", va="center")
box(0.66, 0.895, 0.32, 0.055, "200,859 unit stays screened")
arrow(0.82, 0.895, 0.82, 0.858)
box(0.66, 0.802, 0.32, 0.055, "81,032 with acid suppression")
arrow(0.82, 0.802, 0.82, 0.765)
box(0.66, 0.709, 0.32, 0.055, f"{flow_e['exclusive_users']:,} exclusive users of one class")
arrow(0.82, 0.709, 0.82, 0.672)
box(0.66, 0.604, 0.32, 0.068, f"{flow_e['first_order_in_window']:,} first order within\n[-48 h, +24 h] of unit admission")
arrow(0.82, 0.604, 0.82, 0.567)
box(0.66, 0.492, 0.32, 0.075, f"{flow_e['analysed']:,} analysed ({flow_e['hospitals']} hospitals)\nhypomagnesaemia 35.6%", fc="#F2F2F2")
ax.plot([0.02, 0.98], [0.44, 0.44], lw=0.8, color="black")
ax.text(0.5, 0.415, "Design schematic (both databases)", fontsize=9.5, fontweight="bold", ha="center")
ax.text(0.015, 0.415, "c", fontsize=11, fontweight="bold", ha="left", va="center")
ax.annotate("", xy=(0.90, 0.30), xytext=(0.06, 0.30), arrowprops=dict(arrowstyle="-|>", lw=1.1, color="black"))
for x, lab in [(0.06, "ICU\nadmission"), (0.33, "baseline window\n(max Mg, min Cr)"),
               (0.58, "time zero = +24 h\n(first order fixed)"), (0.90, "outcome window\n(min Mg after +24 h)")]:
    ax.plot([x, x], [0.285, 0.315], lw=0.9, color="black")
    ax.text(x, 0.22, lab, ha="center", va="top", fontsize=7.2)
ax.text(0.06, 0.325, "index order may start up to 48 h before", fontsize=6.8, ha="left", style="italic", color="0.3")
ax.text(0.5, 0.11, "FAERS: spontaneous-report component (per-drug reporting odds ratios vs famotidine)",
        fontsize=8, ha="center", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.5"))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.savefig(os.path.join(OUTF, "F1_study_design.png"), dpi=300)
plt.close(fig)
print("F1 ok")

# ---------------- Figure 2 ----------------
cum = pd.read_csv(os.path.join(RES, "revise_cum_incidence.csv"))
Q = pd.read_parquet(os.path.join(DATA, "revise_mimic.parquet"), columns=["treat", "mg_min"])
E = pd.read_parquet(os.path.join(DATA, "revise_eicu.parquet"), columns=["treat", "mg_min_tz"])
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.4))
ax = axes[0]
for tr, col, lab in [(1, RED, "PPI"), (0, BLUE, "H2RA")]:
    s = cum[cum["treat"] == tr]
    ax.plot(s["day"], s["cum"], color=col, lw=1.6, label=lab)
ax.axhline(31.2, color=RED, lw=0.7, ls=":"); ax.axhline(21.7, color=BLUE, lw=0.7, ls=":")
ax.set_ylim(0, 44)
ax.text(13.6, 32.6, "31.2%", color=RED, fontsize=7.5, ha="right")
ax.text(13.6, 23.1, "21.7%", color=BLUE, fontsize=7.5, ha="right")
ax.set_xlabel("Days from time zero"); ax.set_ylabel("Cumulative incidence (%)")
ax.set_title("a  First magnesium \u2264 1.7 mg/dL (MIMIC-IV)", loc="left", fontsize=8.8)
ax.legend(frameon=False, loc="lower right"); ax.set_xlim(0, 14)
ax = axes[1]
bins = np.arange(0.9, 3.6, 0.1)
for tr, col, lab in [(1, RED, "PPI"), (0, BLUE, "H2RA")]:
    ax.hist(Q.loc[(Q.treat == tr), "mg_min"].dropna(), bins=bins, density=True, histtype="step",
            color=col, lw=1.5, label=lab)
ax.axvline(1.7, color="0.3", lw=0.8, ls="--")
ax.set_xlabel("Magnesium nadir (mg/dL)"); ax.set_ylabel("Density")
ax.set_title("b  Nadir distribution (MIMIC-IV)", loc="left", fontsize=8.8)
ax.legend(frameon=False)
ax = axes[2]
for tr, col, lab in [(1, RED, "PPI"), (0, BLUE, "H2RA")]:
    ax.hist(E.loc[(E.treat == tr), "mg_min_tz"].dropna(), bins=bins, density=True, histtype="step",
            color=col, lw=1.5, ls="--", label=lab)
ax.axvline(1.7, color="0.3", lw=0.8, ls="--")
ax.set_xlabel("Magnesium nadir (mg/dL)"); ax.set_ylabel("Density")
ax.set_title("c  Nadir distribution (eICU-CRD)", loc="left", fontsize=8.8)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(OUTF, "F2_discovery.png"), dpi=300)
plt.close(fig)
print("F2 ok")

# ---------------- Figure 3 ----------------
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6), gridspec_kw={"width_ratios": [1.35, 1, 1]})
ax = axes[0]
rows = [("Crude", *orci(prim["crude"]), RED), ("Multivariable", *orci(prim["multivariable"]), RED),
        ("IPTW (primary)", *orci(prim["IPTW"]), RED), ("Doubly robust", *orci(prim["doubly_robust"]), RED),
        ("PS matched", *orci(prim["PSM_1to1"]), RED), ("Overlap weights", *orci(prim["overlap_weights"]), RED),
        ("eICU crude", *orci(eic["crude"]), TEAL), ("eICU multivariable", *orci(eic["multivariable_cluster"]), TEAL),
        ("eICU IPTW", *orci(eic["IPTW_cluster"]), TEAL)]
for i, (lab, o, lo, hi, col) in enumerate(rows):
    y = len(rows) - i
    ax.plot([lo, hi], [y, y], color=col, lw=1.5)
    ax.plot([o], [y], "o", color=col, ms=3.5)
    ax.text(hi + 0.03, y, f"{o:.2f} [{lo:.2f}, {hi:.2f}]", va="center", fontsize=7.2)
ax.axvline(1, color="0.4", lw=0.7, ls="--")
ax.set_yticks(range(1, len(rows) + 1)); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8)
ax.set_xlabel("Odds ratio (95% CI)"); ax.set_xlim(0.9, 2.4)
ax.set_title("a  PPI vs H2RA, hypomagnesaemia", loc="left", fontsize=8.8)
ax = axes[1]
ds = [("Low dose", *orci(grad_tz["dose_low"])), ("Standard", *orci(grad_tz["dose_standard"])), ("High dose", *orci(grad_tz["dose_high"]))]
for i, (lab, o, lo, hi) in enumerate(ds):
    y = i + 1
    ax.plot([lo, hi], [y, y], color=RED, lw=1.5); ax.plot([o], [y], "o", color=RED, ms=4)
    ax.text(hi + 0.04, y, f"{o:.2f}", va="center", fontsize=7.5)
ax.axvline(1, color="0.4", lw=0.7, ls="--")
ax.set_yticks([1, 2, 3]); ax.set_yticklabels([d[0] for d in ds])
ax.set_xlabel("Odds ratio vs H2RA"); ax.set_xlim(0.9, 2.55)
ax.set_title("b  Dose gradient", loc="left", fontsize=8.8)
ax = axes[2]
sec = [("Severe hypo\n(\u22641.5 mg/dL)", *orci(prim["severe_hypo_IPTW"]), RED),
       ("Hyperkalaemia\n(negative control)", *orci(prim["hyperk_IPTW"]), GREY)]
for i, (lab, o, lo, hi, col) in enumerate(sec):
    y = i + 1
    ax.plot([lo, hi], [y, y], color=col, lw=1.5); ax.plot([o], [y], "o", color=col, ms=4)
    ax.text(hi + 0.03, y, f"{o:.2f}", va="center", fontsize=7.5)
ax.text(0.02, 0.97, f"nadir {prim['mg_nadir_diff_adj']['beta']:+.3f} mg/dL\n(adjusted, weighted)", fontsize=7.0,
        transform=ax.transAxes, ha="left", va="top", color="0.25")
ax.axvline(1, color="0.4", lw=0.7, ls="--")
ax.set_yticks([1, 2]); ax.set_yticklabels([s[0] for s in sec], fontsize=7.6)
ax.set_xlabel("Odds ratio"); ax.set_xlim(0.55, 1.75); ax.set_ylim(0.45, 2.55)
ax.set_title("c  Secondary outcomes", loc="left", fontsize=8.8)
fig.tight_layout()
fig.savefig(os.path.join(OUTF, "F3_core_claim.png"), dpi=300)
plt.close(fig)
print("F3 ok")

# ---------------- Figure 4 ----------------
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
ax = axes[0]
t = grad["timing_pre_admission_vs_inicu"]
tm = [("PPI started\nbefore ICU", *orci(t["ppi_pre_adm"]), RED, t["hypo_rate_pre_adm"]),
      ("PPI started\nin ICU", *orci(t["ppi_in_icu"]), RED, t["hypo_rate_in_icu"]),
      ("H2RA\n(reference)", 1.0, 1.0, 1.0, BLUE, t["hypo_rate_h2ra"])]
for i, (lab, o, lo, hi, col, rate) in enumerate(tm):
    y = i + 1
    if o != 1.0:
        ax.plot([lo, hi], [y, y], color=col, lw=1.5); ax.plot([o], [y], "o", color=col, ms=4)
        ax.text(hi + 0.03, y, f"{o:.2f}", va="center", fontsize=7.5)
    else:
        ax.plot([1], [y], "s", color=col, ms=5)
    ax.text(0.05, y - 0.26, f"crude {100*rate:.1f}%", fontsize=7.2, color="black",
            transform=ax.get_yaxis_transform())
ax.axvline(1, color="0.4", lw=0.7, ls="--")
ax.set_yticks([1, 2, 3]); ax.set_yticklabels([x[0] for x in tm], fontsize=8)
ax.set_xlabel("Odds ratio vs H2RA"); ax.set_xlim(0.85, 1.95); ax.set_ylim(0.45, 3.55)
ax.set_title("a  Start timing", loc="left", fontsize=8.8)
ax = axes[1]
du = [("1 order", *orci(grad_tz["duration_single"])), ("≥2 orders", *orci(grad_tz["duration_multi"]))]
for i, (lab, o, lo, hi) in enumerate(du):
    y = 3 - i
    ax.plot([lo, hi], [y, y], color=RED, lw=1.5); ax.plot([o], [y], "o", color=RED, ms=4)
    ax.text(hi + 0.03, y, f"{o:.2f}", va="center", fontsize=7.5)
ax.plot([1], [1], "s", color=BLUE, ms=5)
ax.axvline(1, color="0.4", lw=0.7, ls="--")
ax.set_yticks([1, 2, 3]); ax.set_yticklabels(["H2RA (ref)"] + [d[0] for d in du][::-1], fontsize=8)
ax.set_xlabel("Odds ratio vs H2RA"); ax.set_xlim(0.9, 2.0)
ax.set_title("b  In-window PPI order count", loc="left", fontsize=8.8)
ax = axes[2]
er = pd.read_parquet(os.path.join(DATA, "revise_eicu.parquet"), columns=["treat", "hypo"])
rates = [100 * er.loc[er.treat == 1, "hypo"].mean(), 100 * er.loc[er.treat == 0, "hypo"].mean()]
ns = [int((er.treat == 1).sum()), int((er.treat == 0).sum())]
ax.bar([0, 1], rates, color=[RED, BLUE], width=0.55)
for i, (r, n) in enumerate(zip(rates, ns)):
    ax.text(i, r + 1.2, f"{r:.1f}% (n={n:,})", ha="center", fontsize=7.5)
ax.set_xticks([0, 1]); ax.set_xticklabels(["PPI", "H2RA"])
ax.set_ylabel("Hypomagnesaemia (%)"); ax.set_ylim(0, max(rates) * 1.35)
ax.set_title("c  Crude rates, eICU-CRD", loc="left", fontsize=8.8)
fig.tight_layout()
fig.savefig(os.path.join(OUTF, "F4_gradients.png"), dpi=300)
plt.close(fig)
print("F4 ok")

# ---------------- Figure 5 ----------------
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axes[0]
items = [("Co-meds capped at index order", *orci(sens["comed_capped_at_index"])),
         ("Strict pre-index baseline Mg", *orci(sens["strict_preindex_baseline"])),
         ("Outcome re-anchored to index", *orci(sens["index_anchored_outcome"])),
         ("First post-baseline value", *orci(sens["first_value_outcome"])),
         ("Stay \u2265 2 days", *orci(sens["los_ge2d"])),
         ("Without admission-year term", *orci(sens["no_year_covariate"])),
         ("Admission years (early half)", *orci(sens["year_early"])),
         ("Admission years (late half)", *orci(sens["year_late"])),
         ("Published covariate set", *orci(prim["published_covars_on_revised_cohort"]))]
for i, (lab, o, lo, hi) in enumerate(items):
    y = len(items) - i
    col = GREY if lab.startswith("Published") else RED
    ax.plot([lo, hi], [y, y], color=col, lw=1.5); ax.plot([o], [y], "o", color=col, ms=3.5)
    ax.text(hi + 0.02, y, f"{o:.2f}", va="center", fontsize=7.2)
ax.axvline(1, color="0.4", lw=0.7, ls="--"); ax.axvline(prim["IPTW"]["OR"], color=RED, lw=0.7, ls=":")
ax.set_yticks(range(1, len(items) + 1)); ax.set_yticklabels([i[0] for i in items][::-1], fontsize=7.6)
ax.set_xlabel("Odds ratio (IPTW unless stated)"); ax.set_xlim(1.0, 1.85)
ax.set_title("a  Sensitivity analyses", loc="left", fontsize=8.8)
ax = axes[1]
ax.axis("off")
ax.set_title("b  Measurement, E-value and arrhythmia", loc="left", fontsize=8.8)
ax.text(0.02, 0.97, "Measurement frequency\n(post-baseline values per stay)", fontsize=8.4, fontweight="bold", va="top")
ax.text(0.02, 0.80, f"MIMIC-IV:  PPI {prim['n_mg_post_by_group']['ppi']}  vs  H2RA {prim['n_mg_post_by_group']['h2ra']}\n"
                    f"eICU-CRD:  PPI {eic['n_mg_by_group']['ppi']}  vs  H2RA {eic['n_mg_by_group']['h2ra']}", fontsize=8, va="top")
ax.text(0.02, 0.55, "E-value (primary IPTW)", fontsize=8.4, fontweight="bold", va="top")
ax.text(0.02, 0.44, f"{prim['E_value_IPTW'][0]}  (lower limit {prim['E_value_IPTW'][1]})", fontsize=8, va="top")
ax.text(0.02, 0.22, "Exploratory downstream outcome", fontsize=8.4, fontweight="bold", va="top")
ax.text(0.02, 0.10, f"new-onset AF: OR {af['ppi_vs_h2ra_new_af_IPTW']['OR']};  "
                    f"early hypo \u2192 AF: OR {af['landmark_hypo_early_to_af']['OR']}", fontsize=7.6, va="top")
fig.tight_layout()
fig.savefig(os.path.join(OUTF, "F5_credibility.png"), dpi=300)
plt.close(fig)
print("F5 ok")

# ---------------- Figure 6 (rebuilt with revised numbers) ----------------
fr = pd.read_csv(os.path.join(RESULTS, "faers_ror.csv")) if os.path.exists(os.path.join(RESULTS, "faers_ror.csv")) else None
fr = pd.read_csv(os.path.join(RES, "faers_ror.csv"))
fr = fr[fr["outcome"].astype(str).str.strip() == "Hypomagnesaemia"]
_nm = {"omeprazole": "Omeprazole", "pantoprazole": "Pantoprazole", "esomeprazole": "Esomeprazole",
       "lansoprazole": "Lansoprazole", "rabeprazole": "Rabeprazole"}
faers = [(_nm[str(d).lower()], float(r), float(lo), float(hi))
         for d, r, lo, hi in zip(fr["drug"], fr["ROR"], fr["CI_low"], fr["CI_high"])]
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), gridspec_kw={"width_ratios": [1.2, 1]})
ax = axes[0]
for i, (lab, o, lo, hi) in enumerate(faers):
    y = len(faers) - i
    ax.plot([lo, hi], [y, y], color=RED, lw=1.5); ax.plot([o], [y], "o", color=RED, ms=4)
    ax.text(hi + 0.06, y, f"{o:.2f} [{lo:.2f}, {hi:.2f}]", va="center", fontsize=7.2)
ax.axvline(1, color="0.4", lw=0.7, ls="--")
ax.set_yticks(range(1, len(faers) + 1)); ax.set_yticklabels([f[0] for f in faers][::-1], fontsize=8)
ax.set_xlabel("Reporting odds ratio vs famotidine (95% CI)"); ax.set_xlim(0.8, 5.6)
ax.set_title("a  FAERS, magnesium-related reports", loc="left", fontsize=8.8)
ax = axes[1]
tri = [("MIMIC-IV\n(n = 39,594)", *orci(prim["IPTW"]), RED),
       ("eICU-CRD\n(n = 22,166; 117 hospitals)", *orci(eic["IPTW_cluster"]), TEAL)]
for i, (lab, o, lo, hi, col) in enumerate(tri):
    y = len(tri) - i
    ax.plot([lo, hi], [y, y], color=col, lw=1.5); ax.plot([o], [y], "o", color=col, ms=4)
    ax.text(hi + 0.02, y, f"{o:.2f}", va="center", fontsize=7.5)
y = 0
ax.plot([2.24, 3.53], [y, y], color=GREY, lw=1.5)
ax.plot([2.885], [y], "o", color=GREY, ms=4)
ax.text(2.885, -0.30, "FAERS ROR 2.24\u20133.53\n(6,148 reports)", fontsize=6.8, color=GREY, ha="center", va="top")
ax.axvline(1, color="0.4", lw=0.7, ls="--")
ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["FAERS\n(spontaneous reports)", tri[1][0], tri[0][0]], fontsize=7.4)
ax.set_xlabel("Odds / reporting odds ratio"); ax.set_xlim(0.8, 3.9); ax.set_ylim(-0.85, 2.7)
ax.set_title("b  Three-source triangulation", loc="left", fontsize=8.8)
fig.tight_layout()
fig.savefig(os.path.join(OUTF, "F6_extrapolation.png"), dpi=300)
plt.close(fig)
print("F6 ok (revised numbers)")
