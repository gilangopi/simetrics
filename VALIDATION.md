# Validation report , SiMetrics v0.3.0

Suite: `scripts/validate_benchmark.py` · case `oberrhein` · 10/10 PASS

| id | check | verdict | detail |
|---|---|---|---|
| V1 | bit-identical reproducibility | **PASS** | two seed-3 runs identical = True |
| V2a | stability reactive (n=20) | **PASS** | SAIFI 8.682 run-CV 5% (rare-event floor ~7%) | 8-seed reference SEM 2% | failures 301 CV 5% |
| V2b | stability preventive (n=20) | **PASS** | SAIFI 0.805 run-CV 29% (rare-event floor ~21%) | 8-seed reference SEM 10% | failures 39 CV 23% |
| V3 | more maintenance -> fewer wear failures | **PASS** | wear failures @1x/2x/4x per yr = 34.1 / 18.1 / 9.6 |
| V4 | external failures = policy-invariant floor | **PASS** | expected ~17.4; observed reactive 18.0 | preventive 16.4 |
| V5 | wear-out failures are prognosable | **PASS** | mean PoF at wear-failure 0.56 vs fleet-average PoF 0.07 (7.8x concentrated) |
| V6 | GNN substrate: neighbour-failure lift | **PASS** | W=7d: lift TRUE-graph 2.31x | null-graph 1.49±0.09 (z=8.8) | coupling-stripped engine 0.98x |
| V7 | reactive vs preventive separable | **PASS** | SAIFI 8.68 vs 0.81 | paired-t 67.6, p=4.2e-24 |
| V8 | prediction task is learnable from observables | **PASS** | fail-in-90d AUC: reactive test-seed 0.789 (train n=125,280, pos 5.5%) | transfer to censored preventive 0.643 |
| V9 | dataset + explanation integrity | **PASS** | 32 run files + 2 topology files OK, version 0.3.0; 713 trip-cause explanation rows, all trips explained & causes verified out = True |
