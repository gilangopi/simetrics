"""
make_paradigm_simple.py , clean single-model version of fig8 (fig8b) with the threshold % labelled.

Shows ONE baseline (HistGB, the best single dispatcher) as a closed-loop policy: its 5 operating
points (each = "service the top X% riskiest by the model's score"), the calendar curve, and the
target zone. Each point is annotated with its threshold percentage so the knob is visible.

Run:  python viz/make_paradigm_simple.py     (~2-3 min: trains HistGB + sweeps 5 thresholds)
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import simetrics
from viz.ml_policy import train_clf, frontier
from evaluate import build_xy

OUT = os.path.join(ROOT, "viz", "figures"); CACHE = os.path.join(OUT, "_paradigm_pts.json")
OLD = "#566573"; NEW = "#1e8449"; MLC = "#8c510a"
QLEVELS = [0.60, 0.74, 0.84, 0.91, 0.96]      # top 40% .. top 4%
XMAX = 3700


def main():
    d = json.load(open(CACHE))
    reactive = tuple(d["reactive"]); prev = [tuple(x) for x in d["prev"]]
    bound_f = [tuple(x) for x in d["bound_f"]]

    HFCACHE = os.path.join(OUT, "_hgb_frontier.json")
    if os.path.exists(HFCACHE) and "--recompute" not in sys.argv:
        fr_all = json.load(open(HFCACHE)); print("loaded cached HGB frontier (--recompute to refresh)")
    else:
        print("training HistGB + sweeping 5 thresholds (seeds 6,7,8) ...")
        top = simetrics.build_topology("oberrhein"); n = len(top["ids"])
        Xva, _, _ = build_xy([7], "reactive", n, top)
        clf = train_clf(top, "hgb")
        fr_all = frontier(top, clf, QLEVELS, seeds=[6, 7, 8], Xref=Xva)
        json.dump(fr_all, open(HFCACHE, "w"))
    fr = sorted([r for r in fr_all if r["visits"] <= XMAX], key=lambda r: r["visits"])
    for r in fr:
        print(f"  top {100*(1-r['q']):4.0f}%: visits {r['visits']:5.0f}  SAIFI {r['saifi']:.2f}")

    LANGS = {
        "id": dict(out="fig8b_paradigm_shift_simple.png", target="WILAYAH\nSASARAN",
                   reactive="Reaktif: SAIFI ≈ 8,7 (di luar skala)",
                   prev=("Preventif 1×", "Preventif 2×", "Preventif 4×"),
                   dial="servis top {:.0f}%",
                   xlabel="Jumlah kunjungan pemeliharaan terencana per 5 tahun  ( →  makin banyak kunjungan )",
                   ylabel="SAIFI — rata-rata listrik padam\n(kali per pelanggan per tahun)  ( ↓  makin andal )",
                   leg_old="Cara lama — reaktif & preventif",
                   leg_ml="ML dasar (HistGB) — kebijakan; label = ambang",
                   leg_tgt="Wilayah sasaran — target GNN+MARL"),
        "en": dict(out="fig8b_paradigm_shift_simple_en.png", target="TARGET\nREGION",
                   reactive="Reactive: SAIFI ≈ 8.7 (off scale)",
                   prev=("Preventive 1x", "Preventive 2x", "Preventive 4x"),
                   dial="service top {:.0f}%",
                   xlabel="Planned maintenance visits per 5 years  ( →  more visits )",
                   ylabel="SAIFI — interruptions per customer per year  ( ↓  more reliable )",
                   leg_old="Status quo — reactive & preventive",
                   leg_ml="ML baseline: HistGB (policy); label = threshold",
                   leg_tgt="Target region — topology-aware policy (open)"),
    }
    for L in LANGS.values():
        fig, ax = plt.subplots(figsize=(10.5, 6.6))
        ms = sorted([(r["visits"], r["saifi"]) for r in fr], key=lambda p: p[1])
        bs = sorted(bound_f, key=lambda p: p[1])
        lo = max(ms[0][1], bs[0][1]); hi = min(ms[-1][1], bs[-1][1]); grid = np.linspace(lo, hi, 80)
        xb = np.interp(grid, [p[1] for p in bs], [p[0] for p in bs])
        xm = np.interp(grid, [p[1] for p in ms], [p[0] for p in ms])
        ax.fill_betweenx(grid, xb, xm, color=NEW, alpha=0.16, zorder=1)
        ax.plot(xb, grid, color=NEW, lw=2.4, zorder=2)
        ax.text(np.median(xb) + 240, (lo + hi)/2 + 0.05, L["target"], color="#14532d",
                fontsize=12.5, fontweight="bold", ha="center", va="center", zorder=8)
        ax.plot([p[0] for p in [reactive]+prev], [p[1] for p in [reactive]+prev],
                ls="--", lw=1.8, color=OLD, zorder=3, alpha=0.85)
        for p, name in zip(prev, L["prev"]):
            ax.scatter(*p, s=150, marker="o", color=OLD, zorder=4, edgecolor="white", linewidth=1.1)
            ax.annotate(name, xy=p, xytext=(p[0], p[1]+0.16), ha="center", fontsize=9.5,
                        color="#222", fontweight="bold")
        ax.annotate(L["reactive"], xy=(40, 4.4), fontsize=10, color=OLD, fontweight="bold")
        ax.plot([r["visits"] for r in fr], [r["saifi"] for r in fr], ls="-", lw=2.5, color=MLC,
                marker="s", markersize=9, markeredgecolor="white", markeredgewidth=0.9, zorder=6)
        for r in fr:
            dy = -0.26 if r["visits"] > 1500 else 0.13       # high-visit labels go below (avoid prev labels)
            ax.annotate(L["dial"].format(100*(1-r["q"])), xy=(r["visits"], r["saifi"]),
                        xytext=(r["visits"]+75, r["saifi"]+dy), fontsize=9, color=MLC, fontweight="bold")
        ax.set_xlabel(L["xlabel"], fontsize=12)
        ax.set_ylabel(L["ylabel"], fontsize=12)
        ax.set_xlim(-150, XMAX); ax.set_ylim(0, 4.6); ax.grid(alpha=0.3); ax.tick_params(labelsize=10)
        handles = [
            Line2D([0], [0], ls="--", marker="o", color=OLD, lw=1.8, markersize=9,
                   markeredgecolor="white", label=L["leg_old"]),
            Line2D([0], [0], ls="-", marker="s", color=MLC, lw=2.5, markersize=9,
                   markeredgecolor="white", label=L["leg_ml"]),
            Patch(facecolor=NEW, alpha=0.30, edgecolor=NEW, label=L["leg_tgt"]),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=10.5, framealpha=0.95)
        fig.savefig(os.path.join(OUT, L["out"]), dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  saved -> viz/figures/{L['out']}")


if __name__ == "__main__":
    main()
