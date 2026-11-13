---
name: Leaderboard submission
about: Submit a result for the SiMetrics leaderboard
title: "[Leaderboard] <method name>"
labels: leaderboard
---

## Method
- **Name:**
- **One-line description:**
- **Topology-aware (uses the graph)?** yes / no
- **Task:** T1 failure prognosis / T1b RUL regression / T2 trip-cause attribution / T3 maintenance policy
- **Code link:**

## Results
| Metric | Mean ± std | Model seeds |
|---|---|---|
| Test AUC |  |  |
| Test AP |  |  |

## How to reproduce
```bash
# the exact command(s) that produce the numbers above
```
- Package versions:
- Hardware and approximate runtime (optional):

## Protocol checklist
- [ ] Trained on seeds 1 to 6, validated on seed 7, tested on seed 8 (no tuning on the test seed).
- [ ] Did **not** use `rul` or `trip_causes` (or anything derived from them) as inputs.
- [ ] Scored with `evaluate.py` (official split and metrics).
- [ ] Reported mean ± std over at least 3 model seeds.

A maintainer reruns the command before adding the result to the leaderboard.
