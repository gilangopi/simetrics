"""
make_rul_trajectory.py , the "stock-chart" view of a continuous prediction.

A classification AUC (Task 1) is threshold-free ranking; it does NOT look like a price chart.
A *continuous* prediction does: this figure overlays a reference model's PREDICTED RUL on the
hidden GROUND-TRUTH RUL of a single asset across the full 5-year run, exactly the way a forecast
is drawn over a price series , the gauge falls as the asset ages, jumps back up at each repair/
service, and the model's line tries to track it.

  fig7_rul_trajectory  , top: true vs predicted RUL (Task 1b reference regressors) for one asset.
                         bottom: the Task-1 risk score P(fail <=90d) for the SAME asset over time,
                         with the true 90-day pre-failure windows shaded (what the AUC ranks).

HONESTY: the predicted lines here are the REGRESSOR twins of the SAME four model families on the
Task-1 leaderboard (linear/Ridge, Random Forest, HistGB, MLP), trained on observable features only.
They are the bar to beat , NOT the proposed GNN+MARL framework, which is not trained yet. All inputs
are observables; `rul` is used only as the target/label, never as a feature.

Run:  python viz/make_rul_trajectory.py            (from anywhere)
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import simetrics
from evaluate import build_xy, fit_model, FEATURE_NAMES, H   # reuse the official Task-1 classifier

BM = "data/simetrics_benchmark"
OUT = os.path.join(ROOT, "viz", "figures"); os.makedirs(OUT, exist_ok=True)
CASE = "oberrhein"
W_LONG, W_SHORT = 60, 14
STEP = 7
TRAIN_SEEDS, TEST_SEED = range(1, 7), 8
EV_WEAR, EV_EXT, EV_MAINT, EV_REPAIR = 0, 1, 2, 3
C_TRUE, C_HGB, C_RIDGE, C_RISK = "#1f77b4", "#ff7f0e", "#7f7f7f", "#9467bd"


def faa(hot):
    return np.exp(15000.0/383.0 - 15000.0/(hot + 273.0))


def _run(seed, policy="reactive"):
    return np.load(f"{BM}/data/{CASE}_{policy}_seed{seed:02d}.npz", allow_pickle=True)


def feats_targets(seeds, n_assets, top, step):
    """Featurize all asset-days -> (X observable, y=true RUL). Mirrors evaluate.build_xy features."""
    raw, down = top["raw"], top["down"]
    X, Y = [], []
    for sd in seeds:
        z = _run(sd); load = z["loading"].astype(np.float32); hot = z["hotspot"].astype(np.float32)
        rul = z["rul"].astype(np.float32); ev = z["events"]; n_days = load.shape[0]
        resets = [[0] for _ in range(n_assets)]
        for d, a, c in ev:
            if c in (EV_MAINT, EV_REPAIR): resets[int(a)].append(int(d))
        resets = [np.array(sorted(r)) for r in resets]
        for t in range(W_LONG, n_days, step):
            mload = load[t-W_LONG:t].mean(0); sload = load[t-W_LONG:t].std(0)
            m14 = load[t-W_SHORT:t].mean(0); mhot = hot[t-W_LONG:t].mean(0)
            xhot = hot[t-W_LONG:t].max(0); mfaa = faa(mhot)
            for a in range(n_assets):
                last = resets[a][resets[a] <= t]; dsr = float(t - (last[-1] if len(last) else 0))
                X.append([dsr, mload[a], sload[a], m14[a]-mload[a], mhot[a], xhot[a],
                          dsr*mfaa[a], raw[a], down[a]/1000.0])
                Y.append(float(rul[t, a]))
    return np.array(X, np.float32), np.array(Y, np.float32)


def asset_timeline(z, a, top, step):
    """Dense feature rows + true RUL for ONE asset across the whole run."""
    load = z["loading"].astype(np.float32)[:, a]; hot = z["hotspot"].astype(np.float32)[:, a]
    rul = z["rul"].astype(np.float32)[:, a]; ev = z["events"]; n_days = len(load)
    resets = [0]
    for d, aa, c in ev:
        if int(aa) == a and c in (EV_MAINT, EV_REPAIR): resets.append(int(d))
    resets = np.array(sorted(resets)); raw = top["raw"][a]; down = top["down"][a]/1000.0
    ts, X, tr = [], [], []
    for t in range(W_LONG, n_days, step):
        w = load[t-W_LONG:t]; mload = w.mean(); sload = w.std(); m14 = load[t-W_SHORT:t].mean()
        mhot = hot[t-W_LONG:t].mean(); xhot = hot[t-W_LONG:t].max(); mfaa = faa(mhot)
        last = resets[resets <= t]; dsr = float(t - (last[-1] if len(last) else 0))
        X.append([dsr, mload, sload, m14-mload, mhot, xhot, dsr*mfaa, raw, down])
        ts.append(t/365.0); tr.append(float(rul[t]))
    return np.array(ts), np.array(X, np.float32), np.array(tr), rul


def main():
    import time; t0 = time.time()
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    print(f"building topology ({CASE}) ...")
    top = simetrics.build_topology(CASE); n = len(top["ids"])

    print("featurizing + training Task-1b RUL regressors (same families as the leaderboard) ...")
    Xtr, ytr = feats_targets(TRAIN_SEEDS, n, top, STEP)
    Xte, yte = feats_targets([TEST_SEED], n, top, STEP)
    sc = StandardScaler().fit(Xtr)

    # one regressor per leaderboard family (LogReg's regression cousin = linear/Ridge); flag = needs scaling
    reg = {}
    reg["Linear (Ridge)"] = (Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr), True)
    reg["Random Forest"]  = (RandomForestRegressor(n_estimators=200, min_samples_leaf=5,
                                                   n_jobs=-1, random_state=0).fit(Xtr, ytr), False)
    reg["HistGB"]         = (HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08,
                                                   random_state=0).fit(Xtr, ytr), False)
    reg["MLP (2x64)"]     = (MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=300,
                                          early_stopping=True, random_state=0).fit(sc.transform(Xtr), ytr), True)

    def rpredict(ms, A):
        m, scaled = ms
        return np.clip(m.predict(sc.transform(A) if scaled else A), 0, 10)

    mae = {name: float(np.mean(np.abs(rpredict(ms, Xte) - yte))) for name, ms in reg.items()}
    print("  test RUL MAE (units 0-10, lower = better):")
    for name in sorted(mae, key=mae.get):
        print(f"    {name:<16} {mae[name]:.2f}")

    print("training the Task-1 risk classifier (P fail<=90d) for the risk panel ...")
    Xc, yc, _ = build_xy(TRAIN_SEEDS, "reactive", n, top)
    Xcv, ycv, _ = build_xy([7], "reactive", n, top)
    risk_f = fit_model("hgb", 0, Xc, yc, Xcv, ycv)

    # pick a representative asset in the TEST run: most wear failures (clear saw-tooth)
    z = _run(TEST_SEED); ev = z["events"]
    wear = np.zeros(n, int)
    for d, a, c in ev:
        if c == EV_WEAR: wear[int(a)] += 1
    a = int(np.argmax(wear))
    fails = sorted(int(d) for d, aa, c in ev if int(aa) == a and c == EV_WEAR)
    resets = sorted(int(d) for d, aa, c in ev if int(aa) == a and c in (EV_MAINT, EV_REPAIR))
    print(f"  chosen asset #{a} ({top['ids'][a]}): {wear[a]} wear-failures over 5 yr")

    ts, Xa, tr, rul_daily = asset_timeline(z, a, top, STEP)
    risk = risk_f(Xa)
    days = np.arange(len(rul_daily))/365.0
    REG_COLORS = {"Linear (Ridge)": "#7f7f7f", "Random Forest": "#2ca02c",
                  "HistGB": "#ff7f0e", "MLP (2x64)": "#9467bd"}
    REG_STYLE = {"Linear (Ridge)": "--"}

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.1, 1]))

    # ---- (A) RUL: ground truth vs the four leaderboard families as regressors (the "stock chart") ----
    ax[0].plot(days, rul_daily.astype(float), color=C_TRUE, lw=2.2, zorder=5,
               label="TRUE RUL (hidden ground truth)")
    for name, ms in reg.items():
        ax[0].plot(ts, rpredict(ms, Xa), color=REG_COLORS[name], lw=1.3,
                   ls=REG_STYLE.get(name, "-"), alpha=0.9, label=f"{name} , MAE {mae[name]:.2f}")
    for i, d in enumerate(fails):
        ax[0].plot(d/365.0, 0, "v", color="#d62728", ms=9, label="actual failure" if i == 0 else None)
    for i, d in enumerate(resets):
        ax[0].axvline(d/365.0, color="#2ca02c", lw=0.6, alpha=0.45, label="service / repair (RUL reset)" if i == 0 else None)
    ax[0].set_ylabel("RUL , 'fuel gauge'\n(10 = new, 0 = dead)")
    ax[0].set_ylim(-0.6, 10.5); ax[0].grid(alpha=0.3)
    ax[0].set_title(f"SiMetrics , continuous RUL forecast vs ground truth  (asset {top['ids'][a]}, {CASE}, test seed {TEST_SEED})",
                    fontweight="bold")
    ax[0].legend(loc="upper right", fontsize=8, ncol=2)

    # ---- (B) the Task-1 risk score the AUC actually ranks ----
    ax[1].plot(ts, risk, color=C_RISK, lw=1.6, label="risk score , P(fail ≤90 d), HistGB classifier (leaderboard Task-1)")
    for i, d in enumerate(fails):
        lo = max(0, d-H)/365.0
        ax[1].axvspan(lo, d/365.0, color="#d62728", alpha=0.12,
                      label="true 90-day pre-failure window (positive label)" if i == 0 else None)
        ax[1].plot(d/365.0, 1.0, "v", color="#d62728", ms=8)
    ax[1].set_ylabel("P(fail ≤90 d)"); ax[1].set_xlabel("time (years)")
    ax[1].set_ylim(-0.03, 1.05); ax[1].grid(alpha=0.3)
    ax[1].legend(loc="upper left", fontsize=8)

    fig.text(0.5, -0.02,
             "Top: the four Task-1 leaderboard families as RUL regressors (Task 1b). Bottom: the Task-1 risk score. "
             "Reference baselines = the bar to beat; the proposed GNN+MARL model is not trained yet. "
             "Inputs are observables only; RUL is the label.",
             ha="center", fontsize=8, style="italic", color="#555555")

    p = os.path.join(OUT, "fig7_rul_trajectory.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  saved -> viz/figures/fig7_rul_trajectory.png   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
