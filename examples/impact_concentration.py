"""
impact_concentration.py , heavy-tail check behind the paper's "concentrated impact" finding.

PRIMARY metric (matches the paper's ~24%): the downstream-customer *exposure* is heavy-tailed across
assets. Ranking line-assets by their downstream-customer count and taking the top decile shows that a
small fraction of assets sit on a large share of all customers. Because a line's downstream-customer
count is exactly its SAIFI/SAIDI weight, this is the impact heavy-tail that makes topology-aware,
impact-weighted maintenance decisions matter.

CONTEXT metric: weighting each asset by how often it *actually* fails over the reactive runs
(failure count x downstream customers) concentrates the realized interruptions even further.
"""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)

BM = "data/simetrics_benchmark"
CASE = "oberrhein"
FAIL_CODES = (0, 1, 4)     # 0 wear-out, 1 external, 4 overload-trip , all interrupt customers


def top_share(weights, frac):
    order = np.argsort(weights)[::-1]
    k = int(round(frac * len(weights)))
    return k, 100.0 * weights[order[:k]].sum() / weights.sum()


def main(seeds=range(1, 9), decile=0.10):
    tz = np.load(f"{BM}/topology_{CASE}.npz", allow_pickle=True)
    down = tz["downstream_customers"].astype(float)
    n = len(down)

    # PRIMARY , static downstream-customer exposure concentration (the paper's ~24%)
    k, share = top_share(down, decile)
    print(f"=== SiMetrics concentrated impact ({CASE}, {n} line-assets) ===")
    print(f"  [downstream-customer exposure]  top {k} assets (top {round(100*k/n)}%) hold "
          f"{share:.1f}% of all downstream customers")
    for frac in (0.05, 0.20):
        kk, s = top_share(down, frac)
        print(f"       (top {round(100*kk/n)}% = {kk} assets -> {s:.1f}%)")

    # CONTEXT , failure-weighted realized interruptions over the reactive runs
    ci = np.zeros(n); total_events = 0
    for sd in seeds:
        z = np.load(f"{BM}/data/{CASE}_reactive_seed{sd:02d}.npz", allow_pickle=True)
        for d, a, c in z["events"]:
            if int(c) in FAIL_CODES:
                ci[int(a)] += down[int(a)]; total_events += 1
    kk, s = top_share(ci, decile)
    print(f"  [failure-weighted, reactive seeds {list(seeds)}]  {total_events} failure events; "
          f"top {kk} assets carry {s:.1f}% of realized customer-interruptions")


if __name__ == "__main__":
    main()
