"""
_precompute_paradigm.py , compute every operating point for fig8 and cache to _paradigm_pts.json.

Heavy (trains 4 baselines + runs each as a closed-loop policy across thresholds/seeds). Run once;
make_paradigm_shift.py then renders instantly from the cache.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, time, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import simetrics
from evaluate import build_xy, fit_model
from viz.ml_policy import frontier

OUT = os.path.join(ROOT, "viz", "figures"); os.makedirs(OUT, exist_ok=True)
CACHE = os.path.join(OUT, "_paradigm_pts.json")
CASE = "oberrhein"; YEARS = 5
SEEDS = list(range(1, 13))
ML_SEEDS = [6, 7, 8]
BOUND_THR = [0.15, 0.20, 0.25, 0.30, 0.35, 0.45, 0.55]
ML_QLEVELS = [0.60, 0.74, 0.84, 0.91, 0.96]    # per-model score quantiles (top 40%..4%)
MODELS = ["logreg", "rf", "hgb", "mlp"]


def mean_pt(top, policy, **kw):
    rs = [simetrics.simulate(top, policy, s, years=YEARS, record=False, **kw)["summary"] for s in SEEDS]
    return (float(np.mean([r["visits"] for r in rs])), float(np.mean([r["saifi"] for r in rs])))


def pareto(points):
    out, vmin = [], 1e18
    for v, s in sorted(points, key=lambda p: p[1]):
        if v < vmin - 1e-9:
            out.append([v, s]); vmin = v
    return out


def main():
    t0 = time.time()
    print(f"building topology ({CASE}) ...", flush=True)
    top = simetrics.build_topology(CASE)
    reactive = list(mean_pt(top, "reactive"))
    prev = [list(mean_pt(top, "preventive", rounds_per_year=r)) for r in (1, 2, 4)]
    bound_f = pareto([mean_pt(top, "predictive", predict_thresh=t) for t in BOUND_THR])
    print(f"  old + bound done ({time.time()-t0:.0f}s). featurizing for training ...", flush=True)

    n = len(top["ids"])
    Xtr, ytr, _ = build_xy(range(1, 7), "reactive", n, top)
    Xva, yva, _ = build_xy([7], "reactive", n, top)

    models = {}
    for name in MODELS:
        ts = time.time()
        clf = fit_model(name, 0, Xtr, ytr, Xva, yva)
        fr = frontier(top, clf, ML_QLEVELS, ML_SEEDS, Xref=Xva, years=YEARS)
        models[name] = pareto([(d["visits"], d["saifi"]) for d in fr])
        print(f"  {name:<7} frontier {[(round(v),round(s,2)) for v,s in models[name]]} "
              f"({time.time()-ts:.0f}s)", flush=True)

    json.dump({"reactive": reactive, "prev": prev, "bound_f": bound_f,
               "models": models, "ml_f": models.get("hgb", [])}, open(CACHE, "w"))
    print(f"saved -> {CACHE}   (total {time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
