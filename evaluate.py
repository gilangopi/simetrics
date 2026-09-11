"""
evaluate.py , the OFFICIAL SiMetrics evaluation harness (one entrypoint for all scoring).

Run from the repo root:
    python evaluate.py                 # trains the reference baselines and prints the leaderboard

Or plug in YOUR OWN model (see examples/your_model_template.py):
    from evaluate import task1_data, score_task1
    d = task1_data()                                  # official featurized train/val/test
    my_model.fit(d["Xtr"], d["ytr"])
    scores = my_model.predict_proba(d["Xte"])[:, 1]
    print(score_task1(d["yte"], scores))              # {'auc':..., 'ap':...}  compare to leaderboard

WHY a single harness: a benchmark is only meaningful if everyone scores on the SAME split with the
SAME metric. This file fixes the protocol so any new model's numbers are directly comparable to the
shipped baselines (and to each other).

Tasks (see README):
  T1  failure prognosis  , node x time, predict wear-out failure within H=90 days from OBSERVABLES
                           only (loading/hot-spot/event history/static physical features). Metric:
                           ROC-AUC + Average Precision. THIS file's main flow.
  T1b RUL regression     , predict the hidden `rul` label. (use d['top'] + the raw run arrays.)
  T2  trip-cause attrib. , explanation: which out-of-service asset caused each overload trip
                           (ground truth in `trip_causes`). Metric: precision@k. (helper below.)
  T3  maintenance policy , choose visits under a budget; the SIMULATOR scores SAIFI/SAIDI/visits
                           (score_policy below runs the reference rule-based policies).

Protocol: train seeds 1-6, validation seed 7, test seed 8 (per the dataset README). Ground-truth
`rul`/`trip_causes` are LABELS , never feed them as model inputs.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, json, time, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # repo root -> import simetrics
import numpy as np

BM = "data/simetrics_benchmark"          # dataset root (run from the repo root)
CASE = "oberrhein"                        # evaluation grid (cigre is small, for development)
H = 90                                    # T1 prediction horizon (days)
W_LONG, W_SHORT = 60, 14                  # feature windows (days)
STEP = 14                                 # sample every STEP days
MODEL_SEEDS = [0, 1, 2]                   # model re-seeds (data itself is fixed by its own seed)
TRAIN_SEEDS, VAL_SEED, TEST_SEED = range(1, 7), 7, 8

FEATURE_NAMES = ["days_since_reset", "mean_load_60d", "std_load_60d", "trend_load_14v60",
                 "mean_hot_60d", "max_hot_60d", "consumed_proxy", "base_loading", "downstream_k"]


# ---------------------------------------------------------------------------- T1 featurizer
def faa_proxy(hot):
    return np.exp(15000.0/383.0 - 15000.0/(hot + 273.0))


def build_xy(seeds, policy, n_assets, top, case=CASE):
    """Official Task-1 featurizer. Returns X (rows x 9 features), y (binary), groups (seed).
    Uses ONLY observable signals (loading, hot-spot, maintenance/repair events, static features)."""
    X, Y, G = [], [], []
    basel, down = top["raw"], top["down"]
    for sd in seeds:
        z = np.load(f"{BM}/data/{case}_{policy}_seed{sd:02d}.npz", allow_pickle=True)
        load = z["loading"].astype(np.float32); hot = z["hotspot"].astype(np.float32); ev = z["events"]
        n_days = load.shape[0]
        resets = [[0] for _ in range(n_assets)]
        wearfail = [[] for _ in range(n_assets)]
        for d, a, c in ev:
            if c in (2, 3):                       # maintenance or repair-complete => reset
                resets[int(a)].append(int(d))
            elif c == 0:                          # wear failure (the label)
                wearfail[int(a)].append(int(d))
        resets = [np.array(sorted(r)) for r in resets]
        wearfail = [np.array(f) for f in wearfail]
        for t in range(W_LONG, n_days - H, STEP):
            mload = load[t-W_LONG:t].mean(axis=0); sload = load[t-W_LONG:t].std(axis=0)
            mload14 = load[t-W_SHORT:t].mean(axis=0)
            mhot = hot[t-W_LONG:t].mean(axis=0); xhot = hot[t-W_LONG:t].max(axis=0)
            mfaa = faa_proxy(mhot)
            for a in range(n_assets):
                last = resets[a][resets[a] <= t]
                dsr = float(t - (last[-1] if len(last) else 0))
                X.append([dsr, mload[a], sload[a], mload14[a]-mload[a], mhot[a], xhot[a],
                          dsr*mfaa[a], basel[a], down[a]/1000.0])
                Y.append(1 if np.any((wearfail[a] > t) & (wearfail[a] <= t+H)) else 0)
                G.append(sd)
    return np.array(X, np.float32), np.array(Y, np.int8), np.array(G)


def task1_data(case=CASE):
    """Convenience: official Task-1 train/val/test arrays + the static graph `top`.
    Build your own features instead? Use d['top'] (ids, node_features, elec_edges, coupling_edges)
    and the raw run files in data/simetrics_benchmark/data/."""
    import simetrics
    top = simetrics.build_topology(case); n = len(top["ids"])
    Xtr, ytr, _ = build_xy(TRAIN_SEEDS, "reactive", n, top, case)
    Xva, yva, _ = build_xy([VAL_SEED], "reactive", n, top, case)
    Xte, yte, _ = build_xy([TEST_SEED], "reactive", n, top, case)
    Xpe, ype, _ = build_xy([TEST_SEED], "preventive", n, top, case)   # censored-regime transfer
    return dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte, Xpe=Xpe, ype=ype,
                feature_names=FEATURE_NAMES, top=top, case=case)


# ---------------------------------------------------------------------------- metrics
def score_task1(y_true, y_score):
    """Official Task-1 metric. y_score = predicted probability of failure-within-H per row."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    return {"auc": float(roc_auc_score(y_true, y_score)),
            "ap": float(average_precision_score(y_true, y_score))}


def eval_auc_ap(y, p):                      # back-compat alias used by examples/
    s = score_task1(y, p); return s["auc"], s["ap"]


def score_task2(pred_causes, true_causes_npz, k=1):
    """Task-2 (trip-cause attribution): precision@k. pred_causes = dict {(day,tripped): [asset,...]}
    ranked best-first. true = the run's `trip_causes` array (day, tripped, cause)."""
    truth = {}
    for d, trp, cause in true_causes_npz:
        truth.setdefault((int(d), int(trp)), set()).add(int(cause))
    hit = tot = 0
    for key, gt in truth.items():
        ranked = pred_causes.get(key, [])[:k]
        if ranked:
            hit += len(set(ranked) & gt) / max(1, min(k, len(gt))); tot += 1
    return {"precision_at_k": (hit/tot if tot else 0.0), "k": k, "n_trips": tot}


def score_policy(policy="preventive", seeds=range(1, 9), case=CASE, years=5, **kw):
    """Task-3 reference scoring: run a built-in rule-based policy and report mean reliability/cost.
    (Custom learned policies require extending simetrics.engine.simulate; see README.)"""
    import simetrics
    top = simetrics.build_topology(case)
    rs = [simetrics.simulate(top, policy, s, years=years, record=False, **kw)["summary"] for s in seeds]
    return {k: float(np.mean([r[k] for r in rs]))
            for k in ["saifi", "saidi", "failures", "visits", "wasted_life"]}


# ---------------------------------------------------------------------------- reference model zoo
def fit_model(name, seed, Xtr, ytr, Xva, yva):
    """Train one reference baseline; returns f(raw_X) -> failure-probability scores."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    sc = None
    if name in ("logreg", "mlp"):
        sc = StandardScaler().fit(Xtr); Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
    if name == "logreg":
        m = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed).fit(Xtr, ytr)
        f = lambda A: m.predict_proba(A)[:, 1]
    elif name == "rf":
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                                   class_weight="balanced_subsample", random_state=seed).fit(Xtr, ytr)
        f = lambda A: m.predict_proba(A)[:, 1]
    elif name == "hgb":
        sw = np.where(ytr == 1, (ytr == 0).sum()/max((ytr == 1).sum(), 1), 1.0)
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08,
                                           random_state=seed).fit(Xtr, ytr, sample_weight=sw)
        f = lambda A: m.predict_proba(A)[:, 1]
    elif name == "mlp":
        import torch, torch.nn as nn
        torch.manual_seed(seed); np.random.seed(seed)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        net = nn.Sequential(nn.Linear(Xtr.shape[1], 64), nn.ReLU(), nn.Dropout(0.2),
                            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1)).to(dev)
        posw = torch.tensor([(ytr == 0).sum()/max((ytr == 1).sum(), 1)], device=dev, dtype=torch.float32)
        lossf = nn.BCEWithLogitsLoss(pos_weight=posw); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        Xt = torch.tensor(Xtr, device=dev); yt = torch.tensor(ytr, device=dev, dtype=torch.float32)
        Xv = torch.tensor(Xva, device=dev)
        best_auc, best_state, patience = -1, None, 0; idx = np.arange(len(Xt))
        for _ in range(60):
            net.train(); np.random.shuffle(idx)
            for b in range(0, len(idx), 4096):
                j = idx[b:b+4096]; opt.zero_grad()
                lossf(net(Xt[j]).squeeze(1), yt[j]).backward(); opt.step()
            net.eval()
            with torch.no_grad():
                pv = torch.sigmoid(net(Xv).squeeze(1)).cpu().numpy()
            auc = eval_auc_ap(yva, pv)[0]
            if auc > best_auc + 1e-4:
                best_auc, best_state, patience = auc, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                patience += 1
                if patience >= 8:
                    break
        net.load_state_dict(best_state); net.eval()
        def f(A):
            import torch as T
            with T.no_grad():
                return T.sigmoid(net(T.tensor(A, device=dev)).squeeze(1)).cpu().numpy()
    else:
        raise ValueError(name)
    if sc is not None:
        raw = f; f = lambda A: raw(sc.transform(A))
    return f


# ---------------------------------------------------------------------------- leaderboard
def leaderboard(case=CASE, models=("logreg", "rf", "hgb", "mlp"), save="results/leaderboard.csv"):
    """Train the reference baselines and print the official Task-1 leaderboard (the bar to beat)."""
    t0 = time.time()
    d = task1_data(case)
    print(f"=== SiMetrics Task-1 leaderboard , {case}, horizon {H}d, split 1-6/7/8 ===")
    print(f"rows: train {len(d['ytr']):,} ({100*d['ytr'].mean():.1f}% pos) | "
          f"test {len(d['yte']):,} ({100*d['yte'].mean():.1f}% pos)")
    print("\nNo-learning single-feature references:")
    for label, col in [("days-since-reset alone", 0), ("mean-loading-60d alone", 1),
                       ("consumed-life proxy alone", 6)]:
        s = score_task1(d["yte"], d["Xte"][:, col])
        print(f"  {label:<28} AUC {s['auc']:.3f} | AP {s['ap']:.3f}")
    rows = []
    for name in models:
        te, ap_, pe = [], [], []
        for sd in MODEL_SEEDS:
            f = fit_model(name, sd, d["Xtr"], d["ytr"], d["Xva"], d["yva"])
            s = score_task1(d["yte"], f(d["Xte"])); te.append(s["auc"]); ap_.append(s["ap"])
            pe.append(score_task1(d["ype"], f(d["Xpe"]))["auc"])
        r = dict(model=name, test_auc=float(np.mean(te)), test_auc_std=float(np.std(te)),
                 test_ap=float(np.mean(ap_)), transfer_preventive_auc=float(np.mean(pe)))
        rows.append(r)
        print(f"  {name:<8} TEST AUC {r['test_auc']:.3f}±{r['test_auc_std']:.3f} | "
              f"AP {r['test_ap']:.3f} | transfer->preventive AUC {r['transfer_preventive_auc']:.3f}")
    best = max(rows, key=lambda r: r["test_auc"])
    print(f"\nBEST baseline: {best['model']} , TEST AUC {best['test_auc']:.3f}. "
          f"Beat this (same split & metric) to claim improvement.")
    if save:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        with open(save, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"saved -> {save}  ({time.time()-t0:.0f}s)")
    return rows


if __name__ == "__main__":
    leaderboard()
