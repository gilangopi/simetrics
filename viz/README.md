# viz/ , figures for SiMetrics

Two scripts, both save to `viz/figures/`. Require `matplotlib` (`pip install matplotlib`). All
inputs come from the shipped dataset (`data/simetrics_benchmark/`) + the official harness
(`evaluate.py`) + real pandapower power-flow solves , no hidden data.

```bash
python viz/make_figures.py     [--case oberrhein --seeds 5]   # statistical figures (fig1-5)
python viz/make_topology.py                                   # network maps + power-flow GIF (fig6 + gif)
```

### Statistical figures (`make_figures.py`)
| File | What it shows |
|---|---|
| `fig1_five_year_journey.png` | the 5-year story: fleet health (RUL) ups & downs, cumulative customer-interruptions, failures-per-quarter by cause (wear / external / trip) |
| `fig2_maintenance_sweetspot.png` | reliability vs maintenance effort (Pareto), and the **sweet spot** , over-maintenance (wasted RUL) vs under-maintenance (failures), with the t_optimal knee |
| `fig3_roc_pr.png` | Task-1 ROC curves + Precision-Recall curves for the reference baselines |
| `fig4_leaderboard.png` | Task-1 AUC / AP bar chart (no-learning references vs ML baselines, with the bar-to-beat) |
| `fig5_topological_impact.png` | downstream-customers per asset (heavy-tailed) , why topology-aware targeting matters |

### Network maps + power-flow animation (`make_topology.py`)
| File | What it shows |
|---|---|
| `fig6_topology_cigre.png` | schematic map of the CIGRE MV feeder (nodes sized by downstream customers, lines colored by loading %, substation starred) |
| `fig6_topology_oberrhein.png` | real-geography map of the mv_oberrhein feeder (same legend) |
| `powerflow_contingency.gif` | **mechanism explainer (one curated fault, real power-flow):** (A) electricity flows out from the substation ring-by-ring with per-line loading %; (B) a high-impact line **fails** (red ✗) → downstream customers go **dark**; (C) a **tie switch** closes → neighbours **absorb** the transferred load, loading jumps (+delta%), overload flagged red. Illustrates the cascade; NOT a timeline. |
| `powerflow_replay_cigre.gif` | **the actual 5-year run replayed** (CIGRE, reactive seed 1): each edge colored AND labeled by its failure risk **PoF = 1 − RUL/4** (so the number explains the color); red ✗ = a failure that period; grey = out for repair; live counters (year, failures-so-far, cumulative interruptions/customer). 100% read from the saved run file. |
| `powerflow_replay_oberrhein.gif` | same 5-year replay on the real-geography 174-asset feeder (riskiest edges labeled to stay readable). |

The two `powerflow_replay_*.gif` ARE the simulation (straight from the dataset). The
`powerflow_contingency.gif` is a curated single-fault explainer (real physics, one event).
Figures are regenerated deterministically from the dataset; re-run any time after `make_benchmark.py`.
