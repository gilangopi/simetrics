# SiMetrics
### An open **sim**ulation benchmark for distribution-grid reliability **metrics** (predictive maintenance)

*An open research project on distribution-grid reliability benchmarking by Gilang Aulia Muhamad, Khalid Habib and Candra Restu Winata (PLN Icon Plus).*

[![Paper](https://img.shields.io/badge/IEEE%20Xplore-ICT--PEP%202026-00629B)](https://ieeexplore.ieee.org/document/11762037)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FICT--PEP71341.2026.11762037-blue)](https://doi.org/10.1109/ICT-PEP71341.2026.11762037)
[![License: MIT](https://img.shields.io/badge/code-MIT-green)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

**Paper:** [*SiMetrics: A Simulated Reliability Benchmark Dataset for Predictive Maintenance in Power Distribution Systems*](https://ieeexplore.ieee.org/document/11762037), IEEE ICT-PEP 2026. See [How to cite](#how-to-cite).

SiMetrics is a fully reproducible, physics-informed **simulator + dataset** for benchmarking
machine-learning methods on medium-voltage **distribution** feeders, scored on the reliability
metrics utilities actually care about: **SAIFI / SAIDI** (IEEE 1366). It ships the **simulator**, a
5-year **dataset**, an **official evaluation harness**, and **reference baselines** , so a new model's
results are directly comparable to everyone else's.

It is a self-contained academic benchmark built entirely on **public** test networks (CIGRE MV,
mv_oberrhein via pandapower) and synthetic, standards-anchored physics. **No proprietary or utility
operational data is used.**

---

## Why it exists
Adjacent public resources cover transmission operation (PowerGraph, Grid2Op), static distribution
snapshots (SimBench), single-component RUL (NASA C-MAPSS), or real distribution-transformer records for
failure classification (the Cauca transformer dataset, Bravo M. et al., *Data in Brief* 2021). To our
knowledge none of them combines **asset aging over time + maintenance regimes + customer-weighted
reliability (SAIFI/SAIDI) + contingency coupling + a maintenance-policy task** in one ML-ready package,
which is the combination SiMetrics provides. Real records such as the Cauca dataset remain the natural
complement for testing whether methods developed here transfer to field data.

## Install
```bash
pip install -r requirements.txt          # numpy, pandapower, networkx, scikit-learn, scipy
# optional (for the MLP baseline): pip install torch
```

## The dataset
**Location: `data/simetrics_benchmark/`** , 34 array files (2 topology + 32 runs = 2 grids ×
{reactive, preventive} × 8 seeds). Split: **seeds 1-6 train, 7 validation, 8 test**. See
`data/README.md` for how to obtain/regenerate it and `data/simetrics_benchmark/DATASHEET.md` for the
full datasheet. Each run file holds SCADA-like observables (`loading`, `hotspot`, `in_service`),
hidden ground-truth `rul` (a **label**, never an input), `events`, and trip-cause `explanations`.

> **Note:** the `.npz` files are **not committed to git** (see `.gitignore`); regenerate them
> bit-for-bit with `python scripts/make_benchmark.py`, or download the frozen release from Hugging Face
> once it is public.

```python
import numpy as np, json
top = np.load("data/simetrics_benchmark/topology_oberrhein.npz", allow_pickle=True)
run = np.load("data/simetrics_benchmark/data/oberrhein_reactive_seed01.npz", allow_pickle=True)
X, y = run["loading"], run["rul"]        # observable input ; hidden label
```

## Tasks
| Task | Level | Type | Target | Metric |
|---|---|---|---|---|
| **T1 failure prognosis** | node×time | binary | wear-failure within H=90 d, from observables | ROC-AUC + Average Precision |
| **T1b RUL regression** | node×time | regression | hidden `rul` | MAE / RMSE |
| **T2 trip-cause attribution** | event | explanation | which out asset caused each overload trip (`trip_causes`) | precision@k |
| **T3 maintenance policy** | system | sequential | choose visits under budget → simulator scores SAIFI/SAIDI | SAIFI + SAIDI at iso-budget |

---

## ▶ Evaluate your model & beat the baselines
Everything is scored through **one official harness, `evaluate.py`**, so your numbers are comparable to
the leaderboard. Run from the repo root.

**1) See the bar to beat:**
```bash
python evaluate.py            # trains the reference baselines, prints the Task-1 leaderboard
```

**2) Benchmark YOUR model (3 lines):**
```python
from evaluate import task1_data, score_task1
d = task1_data()                                   # official train/val/test (split 1-6/7/8)
my_model.fit(d["Xtr"], d["ytr"])                   # <- your model here (any library)
scores = my_model.predict_proba(d["Xte"])[:, 1]    # P(fail within 90 d) per test row
print(score_task1(d["yte"], scores))               # {'auc': ..., 'ap': ...}  -> compare below
```
A ready-to-edit starting point: **`examples/your_model_template.py`**.

**3) Using topology / a GNN?** `d["top"]` gives the graph (`ids`, `node_features` = z, length,
ampacity, vn, base-loading, downstream-customers; `elec_edges`; `coupling_edges` = contingency
neighbours). Build your own spatio-temporal features from the raw run arrays, produce scores for the
test rows, and score with `score_task1` for a fair comparison.

**Rules for a valid submission:** train on seeds 1-6 only; never feed `rul`/`trip_causes` as inputs
(they are labels); report mean ± std over ≥3 model seeds; use `evaluate.py`'s metric and split.

### Leaderboard , Task 1 (oberrhein, test = seed 8; reproduce with `python evaluate.py`)
| Method | Topology-aware? | Test AUC | Test AP |
|---|---|---|---|
| days-since-service only (no learning) | n/a | 0.470 | 0.048 |
| mean-loading-60d only (no learning) | n/a | 0.773 | 0.163 |
| consumed-life proxy (physics, no learning) | n/a | 0.798 | 0.284 |
| Logistic Regression | no | 0.814 | 0.313 |
| Random Forest | no | 0.816 | 0.323 |
| HistGradientBoosting | no | 0.837 | 0.354 |
| **MLP (2×64)** , best baseline | no | **0.843** | **0.356** |
| *your topology-aware model* | yes | *> 0.843 ?* | *> 0.356 ?* |

> All reference baselines are **topology-blind** (per-asset tabular), yet the dataset contains a
> verified spatial cascade signal (neighbour-failure lift 2.31× vs 1.49× null, z=8.8). Exploiting the
> graph is the open opportunity. *(Run a trained model as a Task-3 maintenance policy:
> `examples/model_as_policy.py`.)*

---

## Reproduce & verify
```bash
python scripts/make_benchmark.py       # regenerate the dataset bit-for-bit (~2-3 min)
python scripts/validate_benchmark.py   # 10-check validation suite -> VALIDATION.md (expects 10/10)
```
Every run is bit-identical for a given (case, policy, seed). The physics, every equation and every
constant, is documented in **`FORMULATION.md`**; all constants live in **`simetrics/params.py`**
(no hidden numbers). **Every result in the paper is traced to the exact function that produces it in
[`PAPER_TO_CODE.md`](PAPER_TO_CODE.md)**, with a one-line command to reproduce each number.

## Scope & honesty
A **simulation** benchmark with a **calibrated synthetic** wear law anchored to the IEEE C57.91 form
(not fitted to real failure histories, which are not public). Anchors: aging-acceleration FAA = 1
exactly at 110 °C (C57.91 reference); preventive failure rate ≈ 0.05 /asset/yr (within published RBTS
range). Failures include an **unpreventable external floor** and **contingency overload trips**, so
SAIFI = 0 is impossible by design. Field validation on a real network is out of scope (it would be a
pilot study). Full limitations: `FORMULATION.md` §14 and the datasheet.

---

## How to cite
If you use SiMetrics in your research, please cite the paper:

> G. A. Muhamad, K. Habib, and C. R. Winata, "SiMetrics: A Simulated Reliability Benchmark Dataset for
> Predictive Maintenance in Power Distribution Systems," in *2026 International Conference on Technology
> and Policy in Energy and Electric Power (ICT-PEP)*, Banten, Indonesia, 2026, pp. 168-173,
> doi: [10.1109/ICT-PEP71341.2026.11762037](https://doi.org/10.1109/ICT-PEP71341.2026.11762037).

```bibtex
@inproceedings{muhamad2026simetrics,
  author    = {Muhamad, Gilang Aulia and Habib, Khalid and Winata, Candra Restu},
  title     = {{SiMetrics}: A Simulated Reliability Benchmark Dataset for Predictive Maintenance in Power Distribution Systems},
  booktitle = {2026 International Conference on Technology and Policy in Energy and Electric Power (ICT-PEP)},
  address   = {Banten, Indonesia},
  publisher = {IEEE},
  year      = {2026},
  pages     = {168--173},
  doi       = {10.1109/ICT-PEP71341.2026.11762037},
  url       = {https://ieeexplore.ieee.org/document/11762037}
}
```

A machine-readable `CITATION.cff` is included, so GitHub's **"Cite this repository"** button gives the same reference.

## License
Code is released under the **MIT License** (see `LICENSE`). The accompanying dataset is released under
**CC-BY-4.0**. SiMetrics uses only public test networks and synthetic, standards-anchored physics, and no
proprietary or utility operational data.