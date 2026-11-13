# Paper &harr; Code Map

This maps every equation, task, table, and figure in the paper *"SiMetrics: A Simulated Reliability
Benchmark Dataset for Predictive Maintenance in Power Distribution Systems"* to the **function that
produces it** and the **command that reproduces it**.

Reported values live in the paper; this repo lets anyone re-run the code that yields them. Runs are
deterministic per `(grid, policy, seed)`, so results are reproducible (closed-loop policy figures may
vary slightly with the reference set and seed aggregation).

## Reproduce the paper

```bash
pip install -r requirements.txt

python scripts/make_benchmark.py         # build the dataset  ->  data/simetrics_benchmark/
python scripts/validate_benchmark.py     # Table I  (10-check validation)
python evaluate.py                       # Table III (Task-1 leaderboard)
python examples/rul_regression.py        # Task-1b  RUL regression        (Sec. VI-B)
python examples/impact_concentration.py  # concentrated impact            (Sec. VI-C)
python examples/model_as_policy.py       # Table IV  policy frontier      (Sec. VI-D)
```

**Data flow:** `make_benchmark.py` runs the simulator (`engine.simulate`) and writes the dataset;
everything else loads those arrays. Every physics constant lives in `simetrics/params.py`.

## 1 &middot; Simulator physics (Sec. III-B)

| Eq. | Quantity | Function |
| :---: | --- | --- |
| (1) | daily stress fraction | `engine.py` &rarr; `simulate` |
| (2) | hot-spot temperature | `engine.py` &rarr; `hotspot` |
| (3) | Arrhenius aging factor | `engine.py` &rarr; `faa` |
| text | wear &amp; RUL update | `engine.py` &rarr; `wear_per_day`, `simulate` |
| (4) | probability of failure | `engine.py` &rarr; `simulate` |
| (5) | wear-out hazard | `engine.py` &rarr; `simulate` |
| text | external-floor hazard | `engine.py` &rarr; `simulate` |
| (6) | overload-trip probability | `engine.py` &rarr; `simulate` |
| (7) | transferred contingency load | `engine.py` &rarr; `simulate` |
| (8) | tie-transfer coupling matrix | `engine.py` &rarr; `_build_coupling` |
| (9) | SAIFI / SAIDI | `engine.py` &rarr; `simulate` |
| (10) | wasted usable life | `engine.py` &rarr; `simulate` |

**Constants** (all in `simetrics/params.py`):

| Paper symbol | Code constant |
| --- | --- |
| &sigma; (load scale) | `LOAD_SCALE` |
| S<sub>floor</sub>, S<sub>cap</sub> | `S_FLOOR`, `S_CAP` |
| c(t) (demand curve) | `SEASON_*`, `WEEK_*`, `DEMAND_*` |
| T<sub>ambient</sub> | `HOT_AMBIENT` (constant 30 &deg;C) |
| K<sub>lin</sub>, K<sub>nonlin</sub>, &gamma; | `HOT_LIN`, `HOT_NONLIN`, `HOT_EXP` |
| B, T<sub>ref</sub> | `FAA_B`, `FAA_TREF` (see note) |
| K<sub>cal</sub>, K<sub>th</sub> | `K_CAL`, `K_TH` |
| PoF window (RUL = 4) | `RUL_FULL` |
| &beta;<sub>fail</sub>, &mu;<sub>maint</sub> (0.04 within 70 d) | `FAIL_BASE`, `PROT_MULT`, `PROTECT_D` |
| h<sub>ext</sub> (0.02/365 per day) | `EXT_DAILY` |
| K<sub>trip</sub> | `TRIP_K` |
| &tau;<sub>couple</sub> | `COUPLE_TH` |

**Note on Eq. (3).** The paper writes the exact Kelvin form (T<sub>ref</sub> = 383.15 K, T + 273.15). The
code uses the rounded 383 / 273. Both give FAA = 1 at 110 &deg;C, and the rounded constants are the ones
that generated every number reported in the paper.

## 2 &middot; Networks, dataset, observables (Sec. III-A, III-C)

| Item | Function |
| --- | --- |
| networks (CIGRE MV, MV Oberrhein) | `network.py` &rarr; `build_distribution`, then `engine.py` &rarr; `build_topology` |
| dataset, split, bit-identical runs | `scripts/make_benchmark.py` &rarr; `main` |
| observables vs. hidden labels | `engine.py` &rarr; `simulate` (record block) |

## 3 &middot; Tasks and metrics (Sec. III-D)

| Task | Function |
| --- | --- |
| T1, failure prognosis | `evaluate.py` &rarr; `build_xy`, `fit_model`, `score_task1`, `leaderboard` |
| T1b, RUL regression | `examples/rul_regression.py` |
| T2, trip-cause attribution | `evaluate.py` &rarr; `score_task2` |
| T3, maintenance policy | `evaluate.py` &rarr; `score_policy`; `engine.py` &rarr; `simulate(policy="predictive")`; `examples/model_as_policy.py` &rarr; `frontier` |

## 4 &middot; Where each result comes from

| Paper item | Function | Reproduce |
| --- | --- | --- |
| Table III, Task-1 leaderboard | `evaluate.py` &rarr; `leaderboard` | `python evaluate.py` |
| Table I, validation suite | `scripts/validate_benchmark.py` | `python scripts/validate_benchmark.py` |
| Cascade signal (Sec. VI-C) | `validate_benchmark.py` &rarr; `temporal_lift` | `python scripts/validate_benchmark.py` |
| Concentrated impact (Sec. VI-C) | `examples/impact_concentration.py` | `python examples/impact_concentration.py` |
| Task-1b RUL MAE (Sec. VI-B) | `examples/rul_regression.py` | `python examples/rul_regression.py` |
| Table IV, maintenance policies | `examples/model_as_policy.py` &rarr; `frontier`; `evaluate.py` &rarr; `score_policy` | `python examples/model_as_policy.py` |
| Reliability baselines (SAIFI/SAIDI) | `scripts/make_benchmark.py` &rarr; `summary.csv` | `python scripts/make_benchmark.py` |

## 5 &middot; Reproducing Table IV (cost-reliability frontier)

Operating points are the top **4 / 9 / 16 / 26 / 40 %** of assets by predicted risk
(`q = 0.96, 0.91, 0.84, 0.74, 0.60`). `python examples/model_as_policy.py` prints the HistGB frontier;
the full table is:

```python
import simetrics, evaluate
from examples.model_as_policy import train_clf, frontier

top = simetrics.build_topology("oberrhein")
Xref, _, _ = evaluate.build_xy([7], "reactive", len(top["ids"]), top)     # reference score set

for name in ("hgb", "mlp"):                                               # learned policies
    clf = train_clf(top, name)
    for r in frontier(top, clf, [0.96, 0.91, 0.84, 0.74, 0.60], seeds=range(1, 9), Xref=Xref):
        print(name, r)

from evaluate import score_policy                                         # calendar policies
for rpy in (1, 2, 4):
    print("preventive", rpy, score_policy("preventive", rounds_per_year=rpy))
print("reactive", score_policy("reactive"))
# oracle row: simetrics.simulate(top, "predictive", seed, predict_thresh=...) swept over the threshold
```

## 6 &middot; Figures

| Fig. | Source |
| :---: | --- |
| 1 | `engine.py` &rarr; `simulate` (five-year output) |
| 2 | `examples/model_as_policy.py` (baseline policy flow) |
| 3 | `examples/model_as_policy.py` &rarr; `make_score_fn` (risk scores over time) |
| 4 | `examples/model_as_policy.py` &rarr; `frontier` (cost-reliability plane) |
