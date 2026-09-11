"""
model_as_policy.py , run a TRAINED baseline model as a closed-loop maintenance policy.

The engine's `predictive` policy is an oracle (reads true RUL). Here we instead let a model trained on
OBSERVABLES decide: each day the model scores every asset's P(wear-fail <=90d) from the same 9 features
as evaluate.build_xy (computed from the rolling observable history the engine keeps), and we service
assets whose score exceeds a threshold. Sweeping the threshold traces the model's real (visits, SAIFI)
frontier , a genuine ML-as-policy result, not the oracle.

HONEST: model trained on reactive seeds 1-6 (observables only; never the hidden RUL); applied as policy
on held-out seeds. This is the realistic deploy setting (train on history, act forward).
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import simetrics
from evaluate import build_xy, fit_model, faa_proxy

W_LONG, W_SHORT = 60, 14


def make_score_fn(clf, raw, down):
    """clf: f(X)->P(fail<=90d). Returns score_fn matching the engine's model-policy hook."""
    def score_fn(day, lh, hh, last_maint):
        n = lh.shape[1]
        if day < W_LONG:
            return np.zeros(n)
        L = lh[day-W_LONG:day]; H = hh[day-W_LONG:day]
        mload = L.mean(0); sload = L.std(0); m14 = lh[day-W_SHORT:day].mean(0)
        mhot = H.mean(0); xhot = H.max(0); mfaa = faa_proxy(mhot)
        lm = np.where(last_maint < 0, 0.0, last_maint); dsr = day - lm     # match build_xy reset origin
        X = np.column_stack([dsr, mload, sload, m14-mload, mhot, xhot,
                             dsr*mfaa, raw, down/1000.0]).astype(np.float32)
        return clf(X)
    return score_fn


def train_clf(top, name="hgb"):
    n = len(top["ids"])
    Xtr, ytr, _ = build_xy(range(1, 7), "reactive", n, top)
    Xva, yva, _ = build_xy([7], "reactive", n, top)
    return fit_model(name, 0, Xtr, ytr, Xva, yva)


def frontier(top, clf, qlevels, seeds, Xref, years=5):
    """Event-driven policy with PER-MODEL thresholds = quantiles of that model's own score
    ("service when risk enters the model's top (1-q)%"). Calibration-free & comparable across models.
    Sweep q -> the model's real (visits, SAIFI) frontier."""
    raw, down = top["raw"], top["down"]
    sfn = make_score_fn(clf, raw, down)
    sc = np.asarray(clf(Xref), float)
    out = []
    for q in qlevels:
        thr = float(np.quantile(sc, q))
        rs = [simetrics.simulate(top, "model", s, years=years, record=False,
                                 score_fn=sfn, predict_thresh=thr)["summary"] for s in seeds]
        out.append(dict(q=q, thr=thr,
                        visits=float(np.mean([r["visits"] for r in rs])),
                        saifi=float(np.mean([r["saifi"] for r in rs])),
                        saidi=float(np.mean([r["saidi"] for r in rs]))))
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    print("building topology + training HistGB on reactive seeds 1-6 ...")
    top = simetrics.build_topology("oberrhein")
    clf = train_clf(top, "hgb")
    print(f"  trained ({time.time()-t0:.0f}s). HistGB frontier at the paper's operating points "
          f"(top 4/9/16/26/40%, seeds 1-8):")
    from evaluate import build_xy
    n = len(top["ids"]); Xref, _, _ = build_xy([7], "reactive", n, top)   # reference score set
    for r in frontier(top, clf, [0.96, 0.91, 0.84, 0.74, 0.60], seeds=range(1, 9), Xref=Xref):
        print(f"    top {100*(1-r['q']):4.0f}%: visits {r['visits']:6.0f}  "
              f"SAIFI {r['saifi']:.2f}  SAIDI {r['saidi']:.1f}")
    print(f"  total {time.time()-t0:.0f}s")
