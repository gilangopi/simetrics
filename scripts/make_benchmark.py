"""
make_benchmark.py , generate the shareable SiMetrics distribution benchmark dataset.

5-year daily trajectories on standard feeders (CIGRE MV, mv_oberrhein), two BASELINE policies:
  - reactive   : run-to-failure  -> uncensored degradation data (best for learning RUL models)
  - preventive : static time-based schedule (the *current industrial era*) -> realistically
                 CENSORED data (maintenance interrupts degradation before failure)
Separate file per (case, policy, seed). Topology (features/edges/coupling) shipped per case.
Seeds 1-6 = train, 7 = val, 8 = test (documented split).

Predictive/prescriptive policies are deliberately NOT shipped as data: they are the open
problem the benchmark asks users to solve, scored against these baselines.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from simetrics.engine import build_topology, simulate, ENGINE_VERSION

OUT = "data/simetrics_benchmark"
SEEDS = list(range(1, 9))                       # 1-6 train | 7 val | 8 test
YEARS = 5
CASES = ["cigre", "oberrhein"]
POLICIES = [("reactive", {}), ("preventive", dict(rounds_per_year=2))]


def main():
    os.makedirs(f"{OUT}/data", exist_ok=True)
    rows = []
    for case in CASES:
        t0 = time.time()
        print(f"\n=== {case}: building topology + tie-transfer coupling (power-flow probing) ===")
        top = build_topology(case, verbose=True)
        ce = sorted((x, y, v) for x, ys in top["coupling"].items() for y, v in ys.items())
        np.savez_compressed(
            f"{OUT}/topology_{case}.npz",
            ids=top["ids"], node_features=top["feat"].astype(np.float32),
            feature_names=np.array(top["feat_names"]),
            elec_edges=top["elec_edges"],
            coupling_edges=np.array([(x, y) for x, y, v in ce], np.int32).reshape(-1, 2),
            coupling_dload=np.array([v for x, y, v in ce], np.float32),
            downstream_customers=top["down"].astype(np.float32),
            total_customers=top["total"], label=top["label"], engine_version=ENGINE_VERSION)
        print(f"  topology saved ({time.time()-t0:.0f}s): {len(top['ids'])} assets, "
              f"{len(top['elec_edges'])} electrical edges, {len(ce)} coupling edges, "
              f"{top['total']:,} customers")
        for policy, kw in POLICIES:
            for sd in SEEDS:
                r = simulate(top, policy, sd, years=YEARS, record=True, **kw)
                s = r["summary"]
                np.savez_compressed(
                    f"{OUT}/data/{case}_{policy}_seed{sd:02d}.npz",
                    loading=r["loading"].astype(np.float16),
                    hotspot=r["hotspot"].astype(np.float16),
                    in_service=r["in_service"],
                    rul=r["rul"].astype(np.float16),            # HIDDEN ground-truth labels
                    events=r["events"],                          # (day, asset, code)
                    trip_causes=r["trip_causes"],                # (day, tripped, cause) explanations
                    trip_cause_load=r["trip_cause_load"],        # cause contribution (loading frac)
                    ci_cum=r["ci_cum"], cmh_cum=r["cmh_cum"],
                    summary=json.dumps(s))
                rows.append(s)
                print(f"  {case:<10} {policy:<10} seed {sd}: SAIFI {s['saifi']:.3f} | "
                      f"SAIDI {s['saidi']:.2f} h | fail {s['failures']:>3} "
                      f"(wear {s['wear_failures']}, ext {s['external_failures']}, "
                      f"trip {s['trip_failures']}) | "
                      f"visits {s['visits']:>4} | lambda {s['lambda_per_asset_yr']:.3f}/asset/yr")

    # ---- summary.csv ----
    import csv
    keys = ["case", "policy", "seed", "years", "rounds_per_year", "saifi", "saidi", "failures",
            "wear_failures", "external_failures", "trip_failures", "visits", "wasted_life",
            "wasted_life_per_asset_yr", "rul_at_maint", "lambda_per_asset_yr", "engine_version"]
    with open(f"{OUT}/summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- aggregate table (mean over seeds) for README + console ----
    print("\n=== BASELINE TABLE (mean +/- std over seeds) ===")
    agg_lines = []
    for case in CASES:
        for policy, _ in POLICIES:
            sel = [r for r in rows if r["case"] == case and r["policy"] == policy]
            def ms(key):
                v = np.array([r[key] for r in sel]); return v.mean(), v.std()
            line = (f"{case:<10} {policy:<10} SAIFI {ms('saifi')[0]:6.3f}±{ms('saifi')[1]:.3f} | "
                    f"SAIDI {ms('saidi')[0]:6.2f}±{ms('saidi')[1]:.2f} h | "
                    f"failures {ms('failures')[0]:5.1f}±{ms('failures')[1]:.1f} | "
                    f"visits {ms('visits')[0]:6.1f} | lambda {ms('lambda_per_asset_yr')[0]:.3f}")
            print("  " + line); agg_lines.append(line)

    write_readme(agg_lines)
    print(f"\nDataset + README written to {OUT}/")


def write_readme(agg_lines):
    import simetrics.engine as E
    card = open(os.path.join(os.path.dirname(__file__), "..", "simetrics",
                             "engine.py"), encoding="utf-8").read()
    card = card.split('"""')[1]                  # the engine docstring IS the model card
    table = "\n".join("    " + l for l in agg_lines)
    readme = f"""# SiMetrics , an open benchmark for predictive maintenance on distribution feeders
Version {ENGINE_VERSION} · generated by `scripts/make_benchmark.py` (fully reproducible)

## What this is
5-year, daily-resolution simulated operation of standard MV distribution feeders
(CIGRE MV, mv_oberrhein) under the two maintenance regimes that bracket today's industrial
practice:
- **reactive** , run-to-failure: *uncensored* degradation trajectories (every asset that ages,
  fails). Best for training degradation/RUL models.
- **preventive** , static time-based schedule (2 visits/asset/year, staggered): the *current
  industrial era*. Realistically *censored* data , maintenance interrupts degradation before
  failure, exactly the data quality a utility's historian actually contains.

**The open problem for users:** beat these baselines , decide *which asset to maintain when
(and with which crew)* to minimise SAIFI/SAIDI at equal or lower visit budget. Ground-truth RUL
is included for training/evaluation but is NOT observable in deployment , honest methods should
treat it as a label, not a feature.

## Tasks (OGB-style taxonomy)
| Task | Level | Type | Target | Recommended metric |
|---|---|---|---|---|
| **T1 failure prognosis** | node x time | binary cls | wear-failure within H days (H=90 default) from observables | ROC-AUC + average precision (positives ~5%) |
| **T1b RUL regression** | node x time | regression | hidden `rul` label | MAE |
| **T2 trip-cause attribution** | event | explanation | `trip_causes` ground truth: which outaged asset(s) caused each overload trip | precision@k vs cause set |
| **T3 maintenance policy** | system | sequential decision | choose visits under a budget; simulator scores SAIFI / SAIDI / visits / wasted-life | SAIFI + SAIDI across a range of visit budgets vs the shipped baselines |
Protocol: train seeds 1-6, validate 7, test 8; report mean ± std over >=3 model seeds (the reference baselines use 3); never feed
`rul`/`trip_causes` as input features (labels only).

## Ground-truth explanations (PowerGraph-style, at distribution level)
Every overload trip (event code 4) carries its CAUSE SET in `trip_causes` (day, tripped_asset,
cause_asset) with `trip_cause_load` (the transferred-loading contribution computed from power
flow). This enables benchmarking explainability methods (does your model identify the true
causal neighbour?) , analogous to PowerGraph's cascading-edge masks [Varbella et al., NeurIPS
2024], which provide empirical explanations for TRANSMISSION cascades. SiMetrics provides them
for DISTRIBUTION contingency trips embedded in multi-year maintenance trajectories.

## Positioning vs existing public datasets
| Dataset | Level | What it covers | What it does NOT cover (SiMetrics's focus) |
|---|---|---|---|
| **PowerGraph** (NeurIPS'24) | transmission | PF/OPF snapshots, cascading-failure graphs + explanations | no asset aging/time, no maintenance, no SAIFI/SAIDI, no decision layer |
| **Grid2Op / L2RPN** | transmission | RL for topology/redispatch operation | no asset health, no maintenance scheduling, no customer-weighted reliability |
| **SimBench** | distribution | static grids + load/RES time series | no failures, no aging, no maintenance, not ML-task-ready |
| **PSML / EGS** | mixed | PMU-style time series / synthetic ops | no maintenance decision task, no downstream-customer impact |
| **NASA C-MAPSS** | component | run-to-failure RUL trajectories | single machine, no network/topology, no system-level reliability |
| **SiMetrics (this)** | **distribution** | **5-yr asset-health trajectories + maintenance regimes + IEEE-1366 customer impact + contingency coupling + trip explanations** | , |

## Files
```
topology_<case>.npz      static: node_features (z, length, ampacity, vn, base loading,
                         downstream customers), feature_names, elec_edges (line-graph),
                         coupling_edges + coupling_dload (tie-transfer matrix, from power flow),
                         downstream_customers, total_customers
data/<case>_<policy>_seed<NN>.npz
                         loading   (days x assets, float16)  , SCADA-like, with sensor noise
                         hotspot   (days x assets, float16)  , thermal proxy
                         in_service(days x assets, uint8)
                         rul       (days x assets, float16)  , HIDDEN ground truth (labels)
                         events    (k x 3 int32: day, asset, code 0=wear-fail 1=external-fail
                                    2=planned-maintenance 3=repair-complete 4=overload-trip
                                    5=reclose-after-trip)
                         trip_causes (m x 3 int32: day, tripped, cause) , ground-truth explanations
                         trip_cause_load (m float32) , each cause's transferred-loading share
                         ci_cum / cmh_cum (cumulative customer-interruptions / customer-hours)
                         summary   (json: SAIFI, SAIDI, failures, visits, wasted-life, lambda)
summary.csv              all runs, one row each
```
Split: seeds 1-6 train | 7 val | 8 test (per case x policy).

## Baseline numbers (mean ± std over 8 seeds, {YEARS} years)
{table}

Reliability sanity (honest anchoring): published MV failure rates (e.g. RBTS ~0.04-0.065
/km/yr) describe fleets under *preventive* maintenance , and the preventive runs' wear+external
rate lands near that range (see summary.csv for exact values). The reactive runs sit far above
it BY DESIGN: run-to-failure is the counterfactual upper bracket no utility actually operates.
Two failure channels are unpreventable-by-maintenance BY DESIGN: the external floor
(~0.02 /asset/yr, age-independent) and contingency overload TRIPS (cascade channel , they
depend on what else is out, not on the asset's own condition). Perfect SAIFI=0 is impossible;
trips REWARD policies that prevent the high-impact primary failures that trigger them.

## Quickstart
```python
import numpy as np, json
top = np.load("topology_oberrhein.npz", allow_pickle=True)
run = np.load("data/oberrhein_reactive_seed01.npz", allow_pickle=True)
X = run["loading"]          # (1825, n_assets) observable
y = run["rul"]              # (1825, n_assets) ground-truth labels
ev = run["events"]          # failures / maintenance events
print(json.loads(str(run["summary"])))
```

## Model card (verbatim from the engine , every equation & constant)
```{card}```

## Validation
Every release is gated by a 10-check validation suite (`scripts/validate_benchmark.py`):
bit-identical reproducibility, multi-seed stability, maintenance monotonicity, policy-invariant
external floor, wear-out prognosability, cascade-substrate lift vs null-graph + stripped-engine
controls, baseline separability, learnability probe (logistic AUC), file integrity.
Results: see `VALIDATION.md` in this folder.
"""
    with open(f"{OUT}/README.md", "w", encoding="utf-8") as fh:
        fh.write(readme)


if __name__ == "__main__":
    main()
