---
license: cc-by-4.0
pretty_name: "SiMetrics , Distribution-Grid Reliability Benchmark"
language:
  - en
tags:
  - predictive-maintenance
  - power-distribution
  - reliability
  - saifi-saidi
  - prognostics
  - graph
  - time-series
size_categories:
  - 1K<n<10K
---

# SiMetrics , a simulated reliability benchmark for predictive maintenance on distribution grids

SiMetrics is a fully reproducible, physics-informed **simulator + dataset** for benchmarking
machine-learning methods on medium-voltage **distribution** feeders, scored on the reliability
metrics utilities actually care about: **SAIFI / SAIDI** (IEEE 1366). This repository hosts the
**frozen dataset**; the simulator, evaluation harness, and reference baselines that produced it live
in the code repository (link below).

It is built **entirely on public test networks** (CIGRE MV, mv_oberrhein via pandapower) and
synthetic, standards-anchored physics. **No proprietary or utility operational data is used.**

## What's inside
34 array files = **2 topology + 32 runs** (2 grids × {reactive, preventive} × 8 seeds).
- **Split (documented):** seeds **1–6 train, 7 validation, 8 test**.
- Each run holds SCADA-like **observables** (`loading`, `hotspot`, `in_service`), the hidden
  ground-truth **label** `rul` (never an input), `events`, and trip-cause **explanations**
  (`trip_causes`, `trip_cause_load`).

```
topology_cigre.npz  topology_oberrhein.npz   # static graph: node features, electrical + coupling edges
data/<case>_<policy>_seed<NN>.npz            # 32 daily 5-year trajectories
summary.csv  DATASHEET.md  VALIDATION.md
```

## Load it
```python
from huggingface_hub import hf_hub_download
import numpy as np, json
REPO = "<your-username>/simetrics"                          # <- edit
top = np.load(hf_hub_download(REPO, "topology_oberrhein.npz", repo_type="dataset"), allow_pickle=True)
run = np.load(hf_hub_download(REPO, "data/oberrhein_reactive_seed01.npz", repo_type="dataset"), allow_pickle=True)
X, y = run["loading"], run["rul"]      # observable input ; hidden label (do NOT use y as input)
print(json.loads(str(run["summary"])))
```

## Tasks & the bar to beat (reference baselines, oberrhein, test = seed 8)
| Task | Type | Metric | Best reference baseline |
|---|---|---|---|
| **T1** failure prognosis | binary (fail ≤90 d) | ROC-AUC / AP | **MLP 0.843 AUC** (all topology-blind) |
| **T1b** RUL regression | regression | MAE / RMSE | **MLP / HistGB ≈ 1.20** RUL-units |
| **T2** trip-cause attribution | explanation | precision@k | 713 ground-truth cause rows |
| **T3** maintenance policy | sequential | SAIFI/SAIDI @ iso-budget | reactive 8.7 / preventive 0.9 (SAIFI) |

All reference baselines are **topology-blind**, yet the data contains a verified spatial cascade
signal (neighbour-failure lift 2.31× vs 1.49× null). Exploiting the graph is the open opportunity.

## Scope
A **simulation** benchmark with a **calibrated-synthetic** wear law anchored to the IEEE C57.91 form
(not fitted to real failure histories, which are not public). Failures include an unpreventable
external floor and contingency overload trips, so SAIFI = 0 is impossible by design. Field validation
on a real network is out of scope (it would be a pilot study). Full details: `DATASHEET.md`.

## Reproduce / extend
The simulator + official evaluation harness (`evaluate.py`) + validation suite are in the code
repository: **`<github-url>`**. Every run is **bit-identical** for a given (case, policy, seed), so the
dataset can be regenerated exactly with `python scripts/make_benchmark.py`.
