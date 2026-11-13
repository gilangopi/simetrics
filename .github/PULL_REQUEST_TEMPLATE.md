## Summary
<!-- What does this pull request change, and why? -->

## Type of change
- [ ] New model or baseline
- [ ] Leaderboard result
- [ ] Bug fix
- [ ] Documentation
- [ ] Simulator or dataset change (discussed and agreed in issue #___)
- [ ] Other:

## Checklist
- [ ] I followed the evaluation protocol in `CONTRIBUTING.md`: trained on seeds 1 to 6, validated on 7, tested on 8.
- [ ] I did **not** use `rul` or `trip_causes` (or anything derived from them) as model inputs.
- [ ] I scored results with `evaluate.py` and report mean ± std over at least 3 model seeds.
- [ ] `python scripts/validate_benchmark.py` reports 10/10 (if I touched `simetrics/`, `scripts/`, or `evaluate.py`).
- [ ] `python evaluate.py` still reproduces the reference baselines.
- [ ] I did not commit generated files (`*.npz`, figures, results).
- [ ] I updated the documentation if behaviour changed.

## How to reproduce
<!-- The exact command(s) a reviewer can run to verify this change or result. -->
```bash

```

## Results (if applicable)
| Method | Topology-aware? | Test AUC (mean ± std) | Test AP (mean ± std) | Model seeds |
|---|---|---|---|---|
|  |  |  |  |  |
