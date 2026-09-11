"""
rul_regression.py , Task-1b reference: regress the hidden RUL from the SAME nine observable
features as Task-1, and report MAE / RMSE on the 0-10 RUL scale.

Protocol matches Task-1: train on reactive seeds 1-6, test on reactive seed 8. RUL is a LABEL
(the target), never a feature. This reproduces the paper's Task-1b regression result.
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import simetrics
from evaluate import faa_proxy, W_LONG, W_SHORT, H, STEP, BM, CASE, TRAIN_SEEDS, TEST_SEED


def build_xy_rul(seeds, policy, n_assets, top, case=CASE):
    """Identical sampling/features to evaluate.build_xy, but the target is the hidden RUL[t, a]."""
    X, T = [], []
    basel, down = top["raw"], top["down"]
    for sd in seeds:
        z = np.load(f"{BM}/data/{case}_{policy}_seed{sd:02d}.npz", allow_pickle=True)
        load = z["loading"].astype(np.float32); hot = z["hotspot"].astype(np.float32)
        ev = z["events"]; rul = z["rul"].astype(np.float32)
        n_days = load.shape[0]
        resets = [[0] for _ in range(n_assets)]
        for d, a, c in ev:
            if c in (2, 3):                       # maintenance / repair-complete => age reset
                resets[int(a)].append(int(d))
        resets = [np.array(sorted(r)) for r in resets]
        for t in range(W_LONG, n_days - H, STEP):
            mload = load[t-W_LONG:t].mean(0); sload = load[t-W_LONG:t].std(0)
            mload14 = load[t-W_SHORT:t].mean(0)
            mhot = hot[t-W_LONG:t].mean(0); xhot = hot[t-W_LONG:t].max(0)
            mfaa = faa_proxy(mhot)
            for a in range(n_assets):
                last = resets[a][resets[a] <= t]
                dsr = float(t - (last[-1] if len(last) else 0))
                X.append([dsr, mload[a], sload[a], mload14[a]-mload[a], mhot[a], xhot[a],
                          dsr*mfaa[a], basel[a], down[a]/1000.0])
                T.append(float(rul[t, a]))         # hidden ground-truth RUL = target (never an input)
    return np.array(X, np.float32), np.array(T, np.float32)


def main():
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    top = simetrics.build_topology(CASE); n = len(top["ids"])
    print(f"building T1b data (train seeds {list(TRAIN_SEEDS)} / test seed {TEST_SEED}) ...")
    Xtr, ytr = build_xy_rul(TRAIN_SEEDS, "reactive", n, top)
    Xte, yte = build_xy_rul([TEST_SEED], "reactive", n, top)
    print(f"  train rows {len(ytr):,} | test rows {len(yte):,} | test RUL range "
          f"[{yte.min():.2f}, {yte.max():.2f}], mean {yte.mean():.2f}")
    sc = StandardScaler().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    models = [
        ("Linear Regressor", LinearRegression(), True),
        ("Random Forest",    RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                    n_jobs=-1, random_state=0), False),
        ("HistGB",           HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08,
                                                           random_state=0), False),
        ("MLP (2x64)",       MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=300,
                                          early_stopping=True, random_state=0), True),
    ]
    print(f"\n=== SiMetrics Task-1b : RUL regression (MAE / RMSE, 0-10 scale) , "
          f"{CASE}, test seed {TEST_SEED} ===")
    for name, m, scale in models:
        Xt, Xv = (Xtr_s, Xte_s) if scale else (Xtr, Xte)
        m.fit(Xt, ytr); p = m.predict(Xv)
        mae = mean_absolute_error(yte, p); rmse = mean_squared_error(yte, p) ** 0.5
        print(f"  {name:<18} MAE {mae:.3f} | RMSE {rmse:.3f}")


if __name__ == "__main__":
    main()
