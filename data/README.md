# data/ , WHERE THE TRAINABLE DATASET LIVES

**The machine-learning dataset is in `data/simetrics_benchmark/`.** That is what you load to train
and evaluate models.

## Exact layout
```
data/simetrics_benchmark/
  topology_cigre.npz                 # static graph: node features, electrical edges, coupling edges, downstream customers
  topology_oberrhein.npz
  data/                              # <-- the per-run TRAJECTORIES (the training arrays)
    cigre_reactive_seed01.npz  ... cigre_reactive_seed08.npz
    cigre_preventive_seed01.npz ... cigre_preventive_seed08.npz
    oberrhein_reactive_seed01.npz ... oberrhein_reactive_seed08.npz
    oberrhein_preventive_seed01.npz ... oberrhein_preventive_seed08.npz
  summary.csv                        # one row per run (SAIFI, SAIDI, failures, visits, lambda, ...)
  README.md  DATASHEET.md  VALIDATION.md
```
- **34 array files** = 2 topology + 32 runs (2 grids × 2 policies × 8 seeds).
- **Split (documented):** seeds 1–6 = train, 7 = validation, 8 = test.
- Each run file holds `loading`, `hotspot`, `in_service` (SCADA-like observables), `rul` (hidden
  ground-truth label), `events`, `trip_causes` + `trip_cause_load` (explanations), `ci_cum`, `cmh_cum`,
  `summary`. Full field spec: the benchmark `README.md` and `DATASHEET.md` in that folder.

## Ways to obtain the data
1. **It is already here** (if this copy of the repo includes `data/simetrics_benchmark/`). Just load it
   , see the quickstart in `data/simetrics_benchmark/README.md`.
2. **Regenerate it bit-for-bit** (one command, fully reproducible , the canonical source):
   ```
   cd ..                              # run from the simetrics/ repo root
   python scripts/make_benchmark.py   # (re)writes data/simetrics_benchmark/  (~2-3 min)
   ```
   Then verify it: `python scripts/validate_benchmark.py` (expects 10/10 PASS).

## Note on git
The large `.npz` arrays are excluded from git by `.gitignore` (a code repo should not carry binary
data). The canonical distribution channel is **(2) regenerate**. If you prefer
to commit the arrays directly into your fork, delete the two `*.npz` lines from `.gitignore`.

## Quickstart
```python
import numpy as np, json
top = np.load("data/simetrics_benchmark/topology_oberrhein.npz", allow_pickle=True)
run = np.load("data/simetrics_benchmark/data/oberrhein_reactive_seed01.npz", allow_pickle=True)
X = run["loading"]          # (1825, n_assets)  observable input
y = run["rul"]              # (1825, n_assets)  hidden ground-truth label (do NOT use as input)
events = run["events"]      # (k, 3): day, asset, code
print(json.loads(str(run["summary"])))
```
