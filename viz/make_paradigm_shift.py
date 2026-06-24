"""
make_paradigm_shift.py , the reframing in one trade-off plane (reliability vs effort).

Honest, MEASURED layers (computed by viz/_precompute_paradigm.py, cached to _paradigm_pts.json):
  1. CARA LAMA (calendar-driven): reactive + preventive 1x/2x/4x.
  2. ML DASAR , every leaderboard baseline (LogReg, Random Forest, HistGB, MLP) run as a closed-loop
     POLICY (observables only, no hidden RUL); threshold-swept -> each model's real (visits, SAIFI)
     frontier. They beat calendar but none reaches the ceiling.
  3. WILAYAH SASARAN (target): the shaded zone between the best basic-ML frontier and the
     best-achievable bound -> the room a topology-aware GNN+MARL aims to win.

No title/caption (annotate on your own slide); axis labels + legend + region label only.

Run:  python viz/_precompute_paradigm.py   # once (heavy: trains 4 models, runs closed-loop sweeps)
      python viz/make_paradigm_shift.py     # renders instantly from the cache
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, time, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
OUT = os.path.join(ROOT, "viz", "figures"); os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "_paradigm_pts.json")
OLD = "#566573"; NEW = "#1e8449"
MODEL_STYLE = {                       # warm family = "basic ML"
    "logreg": ("#e08214", "o", "LogReg"),
    "rf":     ("#b30000", "^", "Random Forest"),
    "hgb":    ("#8c510a", "s", "HistGB"),
    "mlp":    ("#d4a017", "D", "MLP"),
}


def pareto(points):
    out, vmin = [], 1e18
    for v, s in sorted(points, key=lambda p: p[1]):
        if v < vmin - 1e-9:
            out.append((v, s)); vmin = v
    return out


def _band(ax, bound_f, right_f):
    ms = sorted(right_f, key=lambda p: p[1]); bs = sorted(bound_f, key=lambda p: p[1])
    lo = max(ms[0][1], bs[0][1]); hi = min(ms[-1][1], bs[-1][1])
    grid = np.linspace(lo, hi, 80)
    xb = np.interp(grid, [p[1] for p in bs], [p[0] for p in bs])
    xm = np.interp(grid, [p[1] for p in ms], [p[0] for p in ms])
    ax.fill_betweenx(grid, xb, xm, color=NEW, alpha=0.16, zorder=1)
    ax.plot(xb, grid, color=NEW, lw=2.4, zorder=2)
    return lo, hi, xb


def _cara_lama(ax, reactive, prev):
    ax.plot([p[0] for p in [reactive]+prev], [p[1] for p in [reactive]+prev],
            ls="--", lw=1.8, color=OLD, zorder=3, alpha=0.85)
    for p, name in zip(prev, ("Preventif 1×", "Preventif 2×", "Preventif 4×")):
        ax.scatter(*p, s=150, marker="o", color=OLD, zorder=4, edgecolor="white", linewidth=1.1)
        ax.annotate(name, xy=p, xytext=(p[0], p[1]+0.16), ha="center", fontsize=9.5,
                    color="#222", fontweight="bold")
    ax.annotate("Reaktif: SAIFI ≈ 8,7 (di luar skala)", xy=(40, 4.4), fontsize=10,
                color=OLD, fontweight="bold")


def _axes(ax):
    ax.set_xlabel("Jumlah kunjungan pemeliharaan terencana per 5 tahun  ( →  makin banyak kunjungan )",
                  fontsize=12)
    ax.set_ylabel("SAIFI — rata-rata listrik padam\n(kali per pelanggan per tahun)  ( ↓  makin andal )",
                  fontsize=12)
    ax.set_xlim(-150, 3700); ax.set_ylim(0, 4.6)
    ax.grid(alpha=0.3); ax.tick_params(labelsize=10)


def render_simple(reactive, prev, bound_f, best_ml):
    """Clean sibling: one best-ML envelope curve instead of all four models."""
    fig, ax = plt.subplots(figsize=(10.5, 6.6))
    lo, hi, xb = _band(ax, bound_f, best_ml)
    ax.text(np.median(xb) + 250, (lo + hi)/2 + 0.05, "WILAYAH\nSASARAN", color="#14532d",
            fontsize=12.5, fontweight="bold", ha="center", va="center", zorder=8)
    _cara_lama(ax, reactive, prev)
    env = sorted([p for p in best_ml if p[0] <= 3700], key=lambda p: p[0])
    ax.plot([p[0] for p in env], [p[1] for p in env], ls="-", lw=2.6, color="#8c510a",
            marker="s", markersize=8.5, markeredgecolor="white", markeredgewidth=0.9, zorder=6)
    _axes(ax)
    handles = [
        Line2D([0], [0], ls="--", marker="o", color=OLD, lw=1.8, markersize=9,
               markeredgecolor="white", label="Cara lama — reaktif & preventif"),
        Line2D([0], [0], ls="-", marker="s", color="#8c510a", lw=2.6, markersize=9,
               markeredgecolor="white", label="ML dasar terbaik (RF / HistGB)"),
        Patch(facecolor=NEW, alpha=0.30, edgecolor=NEW, label="Wilayah sasaran — target GNN+MARL"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=10.5, framealpha=0.95)
    fig.savefig(os.path.join(OUT, "fig8b_paradigm_shift_simple.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved -> viz/figures/fig8b_paradigm_shift_simple.png")


def render_en(reactive, prev, bound_f, models, best_ml):
    """English-labelled twin of fig8 for the paper -> fig8_paradigm_shift_en.png."""
    fig, ax = plt.subplots(figsize=(11, 6.8))
    ms = sorted(best_ml, key=lambda p: p[1]); bs = sorted(bound_f, key=lambda p: p[1])
    lo = max(ms[0][1], bs[0][1]); hi = min(ms[-1][1], bs[-1][1]); grid = np.linspace(lo, hi, 80)
    x_bound = np.interp(grid, [p[1] for p in bs], [p[0] for p in bs])
    x_ml = np.interp(grid, [p[1] for p in ms], [p[0] for p in ms])
    ax.fill_betweenx(grid, x_bound, x_ml, color=NEW, alpha=0.15, zorder=1)
    ax.plot(x_bound, grid, color=NEW, lw=2.4, zorder=2)
    ax.text(np.median(x_bound) + 250, (lo + hi)/2 + 0.05, "TARGET\nREGION", color="#14532d",
            fontsize=12, fontweight="bold", ha="center", va="center", zorder=9)
    ax.plot([p[0] for p in [reactive]+prev], [p[1] for p in [reactive]+prev],
            ls="--", lw=1.8, color=OLD, zorder=3, alpha=0.85)
    for p, name in zip(prev, ("Preventive 1x", "Preventive 2x", "Preventive 4x")):
        ax.scatter(*p, s=140, marker="o", color=OLD, zorder=4, edgecolor="white", linewidth=1.1)
        ax.annotate(name, xy=p, xytext=(p[0], p[1] + 0.15), ha="center", fontsize=9,
                    color="#222", fontweight="bold")
    ax.annotate("Reactive: SAIFI ≈ 8.7 (off scale)", xy=(40, 4.4), fontsize=10,
                color=OLD, fontweight="bold")
    handles = [Line2D([0], [0], ls="--", marker="o", color=OLD, lw=1.8, markersize=9,
                      markeredgecolor="white", label="Status quo — reactive & preventive (calendar)")]
    for name in ("logreg", "rf", "hgb", "mlp"):
        if name not in models:
            continue
        c, mk, lab = MODEL_STYLE[name]
        fr = sorted([p for p in models[name] if p[0] <= 3700], key=lambda p: p[0])
        if not fr:
            continue
        ax.plot([p[0] for p in fr], [p[1] for p in fr], ls="-", lw=1.7, color=c, marker=mk,
                markersize=7.5, markeredgecolor="white", markeredgewidth=0.8, zorder=6)
        handles.append(Line2D([0], [0], ls="-", marker=mk, color=c, lw=1.7, markersize=8,
                              markeredgecolor="white", label=f"ML baseline: {lab}"))
    handles.append(Patch(facecolor=NEW, alpha=0.30, edgecolor=NEW,
                         label="Target region — topology-aware policy (open)"))
    ax.set_xlabel("Planned maintenance visits per 5 years  ( →  more visits )", fontsize=12)
    ax.set_ylabel("SAIFI — interruptions per customer per year  ( ↓  more reliable )", fontsize=12)
    ax.set_xlim(-150, 3700); ax.set_ylim(0, 4.6); ax.grid(alpha=0.3); ax.tick_params(labelsize=10)
    ax.legend(handles=handles, loc="upper right", fontsize=9.5, framealpha=0.95)
    fig.savefig(os.path.join(OUT, "fig8_paradigm_shift_en.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved -> viz/figures/fig8_paradigm_shift_en.png")


def main():
    if not os.path.exists(CACHE):
        sys.exit("no cache yet , run first:  python viz/_precompute_paradigm.py")
    d = json.load(open(CACHE))
    reactive = tuple(d["reactive"]); prev = [tuple(x) for x in d["prev"]]
    bound_f = [tuple(x) for x in d["bound_f"]]
    models = {k: [tuple(x) for x in v] for k, v in
              d.get("models", {"hgb": d.get("ml_f", [])}).items() if v}
    best_ml = pareto([p for fr in models.values() for p in fr])
    print(f"models: {list(models)}; best-ML envelope {[(round(v),round(s,2)) for v,s in best_ml]}")

    fig, ax = plt.subplots(figsize=(11, 6.8))

    # ---- WILAYAH SASARAN: between best basic-ML frontier (right) and best-achievable bound (left) ----
    ms = sorted(best_ml, key=lambda p: p[1]); bs = sorted(bound_f, key=lambda p: p[1])
    lo = max(ms[0][1], bs[0][1]); hi = min(ms[-1][1], bs[-1][1])
    grid = np.linspace(lo, hi, 80)
    x_bound = np.interp(grid, [p[1] for p in bs], [p[0] for p in bs])
    x_ml = np.interp(grid, [p[1] for p in ms], [p[0] for p in ms])
    ax.fill_betweenx(grid, x_bound, x_ml, color=NEW, alpha=0.15, zorder=1)
    ax.plot(x_bound, grid, color=NEW, lw=2.4, zorder=2)
    ax.text(np.median(x_bound) + 250, (lo + hi)/2 + 0.05, "WILAYAH\nSASARAN", color="#14532d",
            fontsize=12, fontweight="bold", ha="center", va="center", zorder=9)

    # ---- CARA LAMA: preventive frontier (reactive off-scale, noted) ----
    pv = [reactive] + prev
    ax.plot([p[0] for p in pv], [p[1] for p in pv], ls="--", lw=1.8, color=OLD, zorder=3, alpha=0.85)
    for p, name in zip(prev, ("Preventif 1×", "Preventif 2×", "Preventif 4×")):
        ax.scatter(*p, s=140, marker="o", color=OLD, zorder=4, edgecolor="white", linewidth=1.1)
        ax.annotate(name, xy=p, xytext=(p[0], p[1] + 0.15), ha="center", fontsize=9,
                    color="#222", fontweight="bold")
    ax.annotate("Reaktif: SAIFI ≈ 8,7 (di luar skala)", xy=(40, 4.4), fontsize=10,
                color=OLD, fontweight="bold")

    # ---- ML DASAR: each baseline as a closed-loop policy ----
    handles = [Line2D([0], [0], ls="--", marker="o", color=OLD, lw=1.8, markersize=9,
                      markeredgecolor="white", label="Cara lama — reaktif & preventif")]
    for name in ("logreg", "rf", "hgb", "mlp"):
        if name not in models:
            continue
        c, mk, lab = MODEL_STYLE[name]
        fr = sorted([p for p in models[name] if p[0] <= 3700], key=lambda p: p[0])  # clip off-chart
        if not fr:
            continue
        ax.plot([p[0] for p in fr], [p[1] for p in fr], ls="-", lw=1.7, color=c, marker=mk,
                markersize=7.5, markeredgecolor="white", markeredgewidth=0.8, zorder=6)
        handles.append(Line2D([0], [0], ls="-", marker=mk, color=c, lw=1.7, markersize=8,
                              markeredgecolor="white", label=f"ML dasar: {lab}"))
    handles.append(Patch(facecolor=NEW, alpha=0.30, edgecolor=NEW,
                         label="Wilayah sasaran — target GNN+MARL"))

    ax.set_xlabel("Jumlah kunjungan pemeliharaan terencana per 5 tahun  ( →  makin banyak kunjungan )",
                  fontsize=12)
    ax.set_ylabel("SAIFI — rata-rata listrik padam\n(kali per pelanggan per tahun)  ( ↓  makin andal )",
                  fontsize=12)
    ax.set_xlim(-150, 3700); ax.set_ylim(0, 4.6)
    ax.grid(alpha=0.3); ax.tick_params(labelsize=10)
    ax.legend(handles=handles, loc="upper right", fontsize=9.5, framealpha=0.95)

    fig.savefig(os.path.join(OUT, "fig8_paradigm_shift.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  saved -> viz/figures/fig8_paradigm_shift.png")
    render_en(reactive, prev, bound_f, models, best_ml)
    # (fig8b clean single-model version with threshold labels: viz/make_paradigm_simple.py)


if __name__ == "__main__":
    main()
