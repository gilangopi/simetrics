"""
spectrum_unified.py , maintenance-maturity spectrum on the UNIFIED benchmark engine.

One physics source (simetrics/engine.py), multi-seed, with the delta metrics that separate the regimes:
unplanned failures (reactive's weakness), wasted usable life + visits (preventive's),
RUL@maintenance (how early you act). Predictive here = perfect-PoF ideal threshold ,
an UPPER BOUND for prediction-based policies (real models see noisy observables).
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from simetrics.engine import build_topology, simulate

CASE = "oberrhein"; SEEDS = list(range(1, 13)); YEARS = 5


def agg(policy, **kw):
    rs = [simulate(top, policy, s, years=YEARS, record=False, **kw)["summary"] for s in SEEDS]
    out = {}
    for k in ["saifi", "saidi", "failures", "wear_failures", "external_failures",
              "trip_failures", "visits", "wasted_life", "rul_at_maint"]:
        v = np.array([r[k] for r in rs], float); out[k] = (v.mean(), v.std())
    return out


def cv(t): m, s = t; return 0 if m == 0 else 100*s/abs(m)


if __name__ == "__main__":
    top = build_topology(CASE)
    rows = [
        ("(a) reactive",            "reactive",   {}),
        ("(b) preventive 1x/yr",    "preventive", dict(rounds_per_year=1)),
        ("(b) preventive 2x/yr",    "preventive", dict(rounds_per_year=2)),
        ("(b) preventive 4x/yr",    "preventive", dict(rounds_per_year=4)),
        ("(c) predictive p>0.35",   "predictive", dict(predict_thresh=0.35)),
        ("(c) predictive p>0.55",   "predictive", dict(predict_thresh=0.55)),
    ]
    print(f"MAINTENANCE-MATURITY SPECTRUM , unified engine, {CASE}, {YEARS} yr, "
          f"mean over {len(SEEDS)} seeds (CV%):\n")
    print(f"{'strategy':<24}{'SAIFI':>15}{'SAIDI(h)':>14}{'unplanned':>12}{'visits':>9}"
          f"{'wasted-life':>13}{'RUL@maint':>11}")
    A = {}
    for label, pol, kw in rows:
        a = agg(pol, **kw); A[label] = a
        print(f"{label:<24}{a['saifi'][0]:>8.3f}(±{cv(a['saifi']):>3.0f}%)"
              f"{a['saidi'][0]:>8.2f}(±{cv(a['saidi']):>2.0f}%)"
              f"{a['failures'][0]:>8.0f}(±{cv(a['failures']):>2.0f}%)"
              f"{a['visits'][0]:>9.0f}"
              f"{a['wasted_life'][0]:>11.0f}  "
              f"{a['rul_at_maint'][0]:>9.2f}")

    re, p2, pr = A["(a) reactive"], A["(b) preventive 2x/yr"], A["(c) predictive p>0.35"]
    def dpc(a, b, k): return 100*(b[k][0]-a[k][0])/a[k][0]
    print("\nDELTA (the maturity-curve gaps, mean over seeds):")
    print(f"  (a)->(b)  reactive->preventive2x : SAIFI {dpc(re,p2,'saifi'):+.0f}% | "
          f"unplanned {dpc(re,p2,'failures'):+.0f}% | cost: {p2['visits'][0]:.0f} visits, "
          f"wasted-life {p2['wasted_life'][0]:.0f} RUL-yr")
    print(f"  (b)->(c)  preventive2x->predictive: SAIFI {dpc(p2,pr,'saifi'):+.0f}% | "
          f"visits {dpc(p2,pr,'visits'):+.0f}% | wasted-life {dpc(p2,pr,'wasted_life'):+.0f}%")
    print(f"  (c)->(d)  predictive->prescriptive: OPEN PROBLEM , add impact (downstream "
          f"customers), crew/travel budgets, t_optimal trade-off => the GNN+MARL research gap.")
    print("\nHONEST NOTES: predictive uses the TRUE PoF (ideal upper bound); external failures "
          f"(~{re['external_failures'][0]:.0f}) + contingency trips (reactive ~{re['trip_failures'][0]:.0f}) "
          "are not directly preventable; SAIFI floor != 0. Trips REWARD preventing high-impact "
          "primaries (fewer outages => fewer transfer-overloads).")
