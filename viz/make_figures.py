"""
make_figures.py , generate all SiMetrics benchmark figures and save them to viz/figures/.

Run from anywhere:
    python viz/make_figures.py
    python viz/make_figures.py --case oberrhein --seeds 5     # options

Figures produced (PNG, 150 dpi):
  fig1_five_year_journey      , the 5-year story: fleet health (RUL) ups & downs, cumulative
                                customer-interruptions, and failures-per-quarter by cause.
  fig2_maintenance_sweetspot  , the sweet spot: reliability vs maintenance effort (Pareto), and the
                                over-maintenance(wasted-RUL) vs under-maintenance(failures) trade-off.
  fig3_roc_pr                 , Task-1 ROC curves + Precision-Recall curves for the reference baselines.
  fig4_leaderboard            , Task-1 AUC/AP bar chart (learning vs no-learning references).
  fig5_topological_impact     , downstream-customers per asset (why topology-aware targeting matters).

Each figure is independent (one failing won't stop the others). All data comes from the shipped
dataset + the official harness , no hidden inputs.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, json, argparse, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (simetrics/)
sys.path.insert(0, ROOT)
os.chdir(ROOT)                                                       # so relative data paths resolve
BM = "data/simetrics_benchmark"
OUT = os.path.join(ROOT, "viz", "figures")
os.makedirs(OUT, exist_ok=True)

EV_WEAR, EV_EXT, EV_MAINT, EV_REPAIR, EV_TRIP, EV_RECLOSE = 0, 1, 2, 3, 4, 5
C_REACT, C_PREV, C_PRED = "#d62728", "#1f77b4", "#2ca02c"


def _run(case, policy, seed):
    return np.load(f"{BM}/data/{case}_{policy}_seed{seed:02d}.npz", allow_pickle=True)


def _total_customers(case):
    return int(np.load(f"{BM}/topology_{case}.npz", allow_pickle=True)["total_customers"])


def save(fig, name):
    p = os.path.join(OUT, name + ".png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  saved -> viz/figures/{name}.png")


# ----------------------------------------------------------------------------- FIG 1
def fig_journey(case):
    total = _total_customers(case)
    rea, pre = _run(case, "reactive", 1), _run(case, "preventive", 1)
    n_days = rea["rul"].shape[0]; t = np.arange(n_days)/365.0
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # (A) fleet mean RUL , the ups & downs of asset health
    ax[0].plot(t, rea["rul"].astype(float).mean(1), color=C_REACT, lw=1.2, label="reactive (run-to-failure)")
    ax[0].plot(t, pre["rul"].astype(float).mean(1), color=C_PREV, lw=1.2, label="preventive 2x/yr")
    ax[0].set_ylabel("mean RUL across all assets\n(health, 10=new, 0=dead)"); ax[0].legend(loc="upper right", fontsize=8)
    ax[0].set_title(f"SiMetrics , 5-year journey ({case})", fontweight="bold")
    ax[0].grid(alpha=0.3)

    # (B) cumulative customer-interruptions per customer (cumulative SAIFI)
    ax[1].plot(t, rea["ci_cum"].astype(float)/total, color=C_REACT, lw=1.5, label="reactive")
    ax[1].plot(t, pre["ci_cum"].astype(float)/total, color=C_PREV, lw=1.5, label="preventive 2x/yr")
    ax[1].set_ylabel("cumulative interruptions\nper customer (= cum. SAIFI)"); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    # (C) reactive failures per quarter, stacked by cause
    binw = 91; nb = n_days//binw
    centers = (np.arange(nb)*binw + binw/2)/365.0
    counts = {c: np.zeros(nb) for c in (EV_WEAR, EV_EXT, EV_TRIP)}
    for d, a, c in rea["events"]:
        if c in counts and d//binw < nb:
            counts[c][d//binw] += 1
    bottom = np.zeros(nb)
    for c, lab, col in [(EV_WEAR, "wear-out", "#8c564b"), (EV_EXT, "external", "#7f7f7f"),
                        (EV_TRIP, "overload trip", "#ff7f0e")]:
        ax[2].bar(centers, counts[c], width=binw/365*0.9, bottom=bottom, label=lab, color=col)
        bottom += counts[c]
    ax[2].set_ylabel("failures per quarter\n(reactive)"); ax[2].set_xlabel("year")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3, axis="y")
    fig.tight_layout(); save(fig, "fig1_five_year_journey")


# ----------------------------------------------------------------------------- FIG 2
def fig_sweetspot(case, seeds):
    import simetrics
    top = simetrics.build_topology(case)
    configs = [("reactive", {}, "reactive", C_REACT),
               ("preventive", dict(rounds_per_year=1), "prev 1x/yr", C_PREV),
               ("preventive", dict(rounds_per_year=2), "prev 2x/yr", C_PREV),
               ("preventive", dict(rounds_per_year=4), "prev 4x/yr", C_PREV),
               ("predictive", dict(predict_thresh=0.35), "pred p>.35", C_PRED),
               ("predictive", dict(predict_thresh=0.55), "pred p>.55", C_PRED)]
    pts = []
    for pol, kw, lab, col in configs:
        rs = [simetrics.simulate(top, pol, s, years=5, record=False, **kw)["summary"]
              for s in range(1, seeds+1)]
        m = lambda k: float(np.mean([r[k] for r in rs]))
        pts.append(dict(lab=lab, col=col, saifi=m("saifi"), visits=m("visits"),
                        wasted=m("wasted_life"), fails=m("failures")))
        print(f"    [{lab}] SAIFI {pts[-1]['saifi']:.2f} | visits {pts[-1]['visits']:.0f} | "
              f"wasted {pts[-1]['wasted']:.0f} | fails {pts[-1]['fails']:.0f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    # (A) reliability vs maintenance effort (Pareto)
    for p in pts:
        ax[0].scatter(p["visits"], p["saifi"], s=90, color=p["col"], zorder=3, edgecolor="k", lw=0.5)
        ax[0].annotate(p["lab"], (p["visits"], p["saifi"]), textcoords="offset points",
                       xytext=(6, 5), fontsize=8)
    ax[0].set_xlabel("planned maintenance visits (5 yr)"); ax[0].set_ylabel("SAIFI (lower = better)")
    ax[0].set_title("Reliability vs maintenance effort\n(lower-left = better)", fontweight="bold")
    ax[0].grid(alpha=0.3)

    # (B) the SWEET SPOT: over-maintenance (wasted RUL) vs under-maintenance (failures)
    for p in pts:
        ax[1].scatter(p["wasted"], p["fails"], s=90, color=p["col"], zorder=3, edgecolor="k", lw=0.5)
        ax[1].annotate(p["lab"], (p["wasted"], p["fails"]), textcoords="offset points",
                       xytext=(6, 5), fontsize=8)
    ax[1].set_xlabel("wasted usable life (Σ RUL discarded) , OVER-maintenance")
    ax[1].set_ylabel("unplanned failures , UNDER-maintenance")
    ax[1].set_title("The maintenance sweet spot (t_optimal)\nknee = best balance", fontweight="bold")
    ax[1].grid(alpha=0.3)
    fig.suptitle(f"SiMetrics , maintenance maturity spectrum ({case}, mean of {seeds} seeds)",
                 fontweight="bold", y=1.02)
    fig.tight_layout(); save(fig, "fig2_maintenance_sweetspot")


# ----------------------------------------------------------------------------- FIG 3 + 4
def fig_roc_pr_and_bar(case):
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    import evaluate as E
    d = E.task1_data(case)
    yte = d["yte"]
    models = [("logreg", "Logistic Reg."), ("rf", "Random Forest"),
              ("hgb", "HistGradBoost"), ("mlp", "MLP 2x64")]
    cols = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    scored = []
    for name, lab in models:
        f = E.fit_model(name, 0, d["Xtr"], d["ytr"], d["Xva"], d["yva"])
        s = f(d["Xte"]); scored.append((lab, s,
            roc_auc_score(yte, s), average_precision_score(yte, s)))
        print(f"    [{lab}] AUC {scored[-1][2]:.3f} | AP {scored[-1][3]:.3f}")

    # ---- FIG 3: ROC + PR ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for (lab, s, auc, ap), c in zip(scored, cols):
        fpr, tpr, _ = roc_curve(yte, s); ax[0].plot(fpr, tpr, color=c, lw=1.6, label=f"{lab} (AUC {auc:.3f})")
        rec, prec, _ = precision_recall_curve(yte, s); ax[1].plot(rec, prec, color=c, lw=1.6, label=f"{lab} (AP {ap:.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="random")
    ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate")
    ax[0].set_title("Task 1 , ROC", fontweight="bold"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].axhline(yte.mean(), color="k", ls="--", lw=0.8, label=f"prevalence ({yte.mean():.2f})")
    ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
    ax[1].set_title("Task 1 , Precision-Recall", fontweight="bold"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.suptitle(f"SiMetrics Task 1 (failure within 90 d) , {case}, test seed 8", fontweight="bold", y=1.02)
    fig.tight_layout(); save(fig, "fig3_roc_pr")

    # ---- FIG 4: leaderboard bar (learning + no-learning references) ----
    refs = [("days-since-svc", d["Xte"][:, 0]), ("mean-load-60d", d["Xte"][:, 1]),
            ("consumed-proxy", d["Xte"][:, 6])]
    names, aucs, aps, isml = [], [], [], []
    for lab, col in refs:
        names.append(lab); aucs.append(roc_auc_score(yte, col)); aps.append(average_precision_score(yte, col)); isml.append(False)
    for lab, s, auc, ap in scored:
        names.append(lab); aucs.append(auc); aps.append(ap); isml.append(True)
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5))
    b1 = ax.bar(x-w/2, aucs, w, label="ROC-AUC", color=["#9ecae1" if not m else "#3182bd" for m in isml])
    b2 = ax.bar(x+w/2, aps, w, label="Avg Precision", color=["#a1d99b" if not m else "#31a354" for m in isml])
    best = max(a for a, m in zip(aucs, isml) if m)
    ax.axhline(best, color="#d62728", ls="--", lw=1, label=f"best AUC to beat ({best:.3f})")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("score"); ax.set_ylim(0, 1.0); ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"SiMetrics Task 1 leaderboard , {case} (light = no-learning reference, dark = ML)",
                 fontweight="bold")
    for b in list(b1)+list(b2):
        ax.annotate(f"{b.get_height():.2f}", (b.get_x()+b.get_width()/2, b.get_height()),
                    textcoords="offset points", xytext=(0, 2), ha="center", fontsize=7)
    fig.tight_layout(); save(fig, "fig4_leaderboard")


# ----------------------------------------------------------------------------- FIG 5
def fig_impact(case):
    top = np.load(f"{BM}/topology_{case}.npz", allow_pickle=True)
    down = np.sort(top["downstream_customers"].astype(float))[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].bar(np.arange(len(down)), down, color="#6a51a3")
    ax[0].set_xlabel("asset (sorted by impact)"); ax[0].set_ylabel("downstream customers")
    ax[0].set_title("Topological impact is heavy-tailed", fontweight="bold"); ax[0].grid(alpha=0.3, axis="y")
    ax[1].hist(down, bins=30, color="#6a51a3"); ax[1].set_xlabel("downstream customers per asset")
    ax[1].set_ylabel("# assets"); ax[1].set_title("Distribution of per-asset impact", fontweight="bold")
    ax[1].grid(alpha=0.3, axis="y")
    share = down[:max(1, len(down)//10)].sum()/down.sum()*100
    fig.suptitle(f"SiMetrics , why topology-aware targeting matters ({case}): "
                 f"top 10% of assets carry {share:.0f}% of customer-impact", fontweight="bold", y=1.02)
    fig.tight_layout(); save(fig, "fig5_topological_impact")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="oberrhein")
    ap.add_argument("--seeds", type=int, default=5, help="seeds for the sweet-spot spectrum (fig2)")
    a = ap.parse_args()
    t0 = time.time()
    print(f"=== SiMetrics figures , case={a.case} -> viz/figures/ ===")
    for name, fn in [("fig1 five-year journey", lambda: fig_journey(a.case)),
                     ("fig2 maintenance sweet spot", lambda: fig_sweetspot(a.case, a.seeds)),
                     ("fig3/4 ROC-PR + leaderboard", lambda: fig_roc_pr_and_bar(a.case)),
                     ("fig5 topological impact", lambda: fig_impact(a.case))]:
        try:
            print(f"[{name}]"); fn()
        except Exception as e:
            print(f"  !! skipped ({type(e).__name__}: {e})")
    print(f"done in {time.time()-t0:.0f}s , see viz/figures/")
