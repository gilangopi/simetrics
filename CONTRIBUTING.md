# Contributing to SiMetrics

Thank you for your interest in SiMetrics. Contributions of all sizes are welcome: new models, leaderboard
results, bug fixes, documentation, and new test cases.

SiMetrics is a **benchmark**, so its value depends on every result being comparable and reproducible.
Most of the rules below exist to protect that. Please read them before opening a pull request.

## Ways to contribute

| Contribution | How |
|---|---|
| Report a bug | Open an issue with the **Bug report** template |
| Submit a leaderboard result | Open an issue with the **Leaderboard submission** template, or a pull request adding your row |
| Add a model or baseline | Pull request, usually as a new script in `examples/` |
| Fix a bug or improve code | Pull request |
| Improve documentation | Pull request |
| Propose a change to the simulator or dataset | **Open an issue first** and discuss it before writing code (see below) |
| Ask a question | Open an issue |

## Setup

```bash
git clone https://github.com/<your-username>/simetrics.git   # your fork
cd simetrics
pip install -r requirements.txt          # optional, for the MLP baseline: pip install torch
python scripts/make_benchmark.py         # regenerate the dataset bit-for-bit (about 2 to 3 minutes)
python scripts/validate_benchmark.py     # validation suite, expects 10/10
python evaluate.py                       # reproduces the reference baselines
```

## Workflow

1. **Fork** the repository and create a branch for your change, for example `add-gnn-baseline`.
2. Make your change, keeping it focused on one thing.
3. Run the checks in [Before you open a pull request](#before-you-open-a-pull-request).
4. Open a **pull request** against `main` and fill in the template.
5. A maintainer reviews it. Pull requests are merged with **squash merge**, so your change lands as a
   single commit.

## Evaluation protocol (required for any reported result)

Results are only comparable if everyone follows the same protocol:

- **Split:** train on seeds **1 to 6**, validate on seed **7**, test on seed **8**. Never tune on the test seed.
- **Labels are never inputs:** `rul` and `trip_causes` are ground-truth labels. Using them, or anything
  derived from them, as model inputs invalidates the result.
- **Official metrics:** score with `evaluate.py` (for Task 1, `task1_data()` and `score_task1()`), so
  metrics and splits match the leaderboard exactly.
- **Variance:** report **mean ± std over at least 3 model seeds**.
- **Reproducibility:** include the exact command or script that produces your numbers, and the package
  versions you used.

`examples/your_model_template.py` is a ready-to-edit starting point that already follows the protocol.

## Changes to the simulator or dataset

The engine (`simetrics/`) and its constants (`simetrics/params.py`) define the dataset. Changing them
changes the data, which would make new results incomparable with the existing leaderboard and with the
published paper. Therefore:

- **Open an issue first** describing the change and why it is needed.
- Every numeric constant lives in `simetrics/params.py` with its unit and justification. Do not add
  hidden literals in the engine.
- Randomness must stay deterministic per `(case, policy, seed)`, using the seeded `RandomState`.
- If an agreed change alters the generated data, bump `ENGINE_VERSION` in `simetrics/params.py`, and update
  `FORMULATION.md` and `PAPER_TO_CODE.md` where they describe the changed behaviour.

## Before you open a pull request

- [ ] `python scripts/validate_benchmark.py` still reports **10/10** (required if you touched `simetrics/`,
      `scripts/`, or `evaluate.py`).
- [ ] `python evaluate.py` still reproduces the reference baseline numbers in the README.
- [ ] New code follows the style of the surrounding code, with short comments where the reasoning is not obvious.
- [ ] Documentation is updated if behaviour changed.
- [ ] Large generated files (`*.npz`, figures, results) are **not** committed; they are listed in `.gitignore`.

## Adding a leaderboard result

Include:

- Method name and a one-line description.
- Whether it is **topology-aware** (uses the graph) or not.
- Test AUC and AP as **mean ± std**, and the number of model seeds.
- The command or script that reproduces it, and a link to your code.

A maintainer reruns the command before adding the row. Results that cannot be reproduced are not listed.

## Reporting bugs

Use the **Bug report** issue template. The most helpful reports include the exact command you ran, the
full error message, your operating system and Python version, and whether `scripts/validate_benchmark.py`
passes on your machine.

## License

By contributing, you agree that your contributions are licensed under the repository's **MIT License**
(code). The generated dataset is released under **CC BY 4.0**.

## Citation

If your work builds on SiMetrics, please cite the paper. See [How to cite](README.md#how-to-cite).
