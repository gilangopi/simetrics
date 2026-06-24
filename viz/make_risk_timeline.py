"""
make_risk_timeline.py , the TEMPORAL view of Task 1 (what the pipeline actually uses).

For one representative asset, plot each baseline classifier's predicted P(wear-failure within H=90 days)
over time (the risk score that the maintenance policy thresholds). Shaded bands = the TRUE 90-day
pre-failure windows (where the correct answer is "yes, failing soon"); red markers = actual failures.
A good model's risk should rise INSIDE the shaded bands and stay low outside.

RUL itself is a hidden label and is NOT shown here , this is purely the observable-driven risk score.

Run:  python viz/make_risk_timeline.py     (~30-60 s: trains 4 classifiers, scores one asset)
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import simetrics
from evaluate import build_xy, fit_model, H
from viz.make_rul_trajectory import asset_timeline, STEP

OUT = os.path.join(ROOT, "viz", "figures"); os.makedirs(OUT, exist_ok=True)
BM = "data/simetrics_benchmark"; CASE = "oberrhein"; TEST_SEED = 8
STYLE = {"logreg": ("#e08214", "LogReg"), "rf": ("#b30000", "Random Forest"),
         "hgb": ("#8c510a", "HistGB"), "mlp": ("#1f78b4", "MLP")}


def main():
    import time; t0 = time.time()
    print("building topology + training 4 classifiers (reactive seeds 1-6) ...")
    top = simetrics.build_topology(CASE); n = len(top["ids"])
    Xtr, ytr, _ = build_xy(range(1, 7), "reactive", n, top)
    Xva, yva, _ = build_xy([7], "reactive", n, top)
    clfs = {k: fit_model(k, 0, Xtr, ytr, Xva, yva) for k in STYLE}

    z = np.load(f"{BM}/data/{CASE}_reactive_seed{TEST_SEED:02d}.npz", allow_pickle=True)
    ev = z["events"]
    wear = np.zeros(n, int)
    for d, a, c in ev:
        if c == 0:
            wear[int(a)] += 1
    a = int(np.argmax(wear))                       # asset with the most wear failures (clear story)
    fails = sorted(int(d) for d, aa, c in ev if int(aa) == a and c == 0)
    ts, Xa, _, _ = asset_timeline(z, a, top, STEP)
    print(f"  asset #{a} ({top['ids'][a]}): {len(fails)} wear failures; scoring {len(ts)} timesteps")

    fig, ax = plt.subplots(figsize=(11, 5.2))
    # true 90-day pre-failure windows (the positive-label zones) + actual failures
    for i, d in enumerate(fails):
        ax.axvspan(max(0, d - H)/365.0, d/365.0, color="#d62728", alpha=0.12,
                   label="true 90-day pre-failure window" if i == 0 else None)
        ax.plot(d/365.0, 1.0, "v", color="#d62728", ms=9,
                label="actual failure" if i == 0 else None)
    # each classifier's risk score over time
    for k, (c, lab) in STYLE.items():
        ax.plot(ts, np.asarray(clfs[k](Xa), float), color=c, lw=1.7, label=lab)

    ax.set_xlabel("time (years)", fontsize=12)
    ax.set_ylabel("predicted P(failure within 90 days)", fontsize=12)
    ax.set_ylim(-0.03, 1.05); ax.set_xlim(0, ts.max() if len(ts) else 5)
    ax.grid(alpha=0.3); ax.tick_params(labelsize=10)
    ax.legend(loc="upper left", fontsize=9.5, ncol=2, framealpha=0.95)
    fig.savefig(os.path.join(OUT, "fig9_risk_timeline.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> viz/figures/fig9_risk_timeline.png   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
