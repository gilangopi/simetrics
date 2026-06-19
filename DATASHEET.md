# Datasheet , SiMetrics

## Motivation
**Why was the dataset created?** To fill a verified gap: no public, ML-task-ready benchmark exists
for *maintenance decision-making* on distribution grids evaluated on customer-weighted reliability
(SAIFI/SAIDI, IEEE 1366). Adjacent datasets cover transmission operation (PowerGraph, Grid2Op),
static distribution snapshots (SimBench), or single-component RUL (NASA C-MAPSS) , none combine
asset aging over time, maintenance regimes, downstream-customer impact, and contingency coupling.

## Composition
**What do the instances represent?** Each run file = one 5-year, daily-resolution operation of a
standard MV feeder (CIGRE MV, 12 line-assets; mv_oberrhein, 174 line-assets) under one maintenance
policy (reactive or preventive 2x/yr) and one random seed. Per asset-day: SCADA-like loading (with
sensor noise), hot-spot temperature proxy, in-service flag, and hidden ground-truth RUL. Event
streams record failures (wear / external / overload-trip), maintenance, repairs, recloses.
Every overload trip carries a ground-truth cause set (which outaged asset(s) transferred the load).
**How many instances?** 32 run files (2 grids x 2 policies x 8 seeds) + 2 topology files;
e.g. oberrhein: 174 assets x 1825 days per run.
**Is anything confidential?** No. 100% synthetic, generated from published test networks. No utility's operational data is included or was used.

## Collection / generation process
**How was the data generated?** By `simetrics/engine.py` (model card embedded in its docstring):
AC power flow (pandapower) -> per-asset loading; seasonal+weekly demand curve; IEEE C57.91-form
Arrhenius thermal aging + calendar aging; three failure channels (end-of-life wear hazard,
age-independent external floor, contingency overload trips with a no-retrip cooldown); radial
outage propagation to downstream customers (IEEE 1366); tie-switch restoration found by power-flow
probing. Every random element draws from a single seeded RNG: same seed = bit-identical file.
**Calibration anchors (honest):** preventive-fleet failure rate ~0.05 /asset/yr matches published
RBTS MV rates; reactive is the run-to-failure counterfactual (~0.34); trips are a 15-26% minority
share. The wear law is calibrated-synthetic (anchored to the C57.91 FORM, not fitted to real
failure histories, none are public).

## Preprocessing / labels
Observables (`loading`, `hotspot`, `in_service`, `events`) are what a SCADA historian could
plausibly contain. `rul` and `trip_causes` are ground-truth LABELS, not observable in deployment;
honest methods must not use them as input features.

## Uses
**Intended:** T1 failure prognosis (node-time classification/regression), T2 trip-cause
attribution (explainability), T3 maintenance-policy optimisation scored by SAIFI/SAIDI at
iso-budget (the open problem). Train/val/test = seeds 1-6 / 7 / 8.
**Out-of-scope / misuse:** estimating any real utility's reliability indices; claiming real-world
validation (this is a simulation benchmark , field validation requires a pilot); safety-critical
deployment decisions.

## Known limitations (verbatim honesty)
Synthetic wear calibration; demand = one snapshot x seasonal curve (no AMI profiles); no
weather/vegetation model; first-success tie restoration only; customers proportional to load
(600/MW); cosmetic coordinates; chronic normal-state overloads in the stressed scenario are
assumed operator-managed (they age assets but never trip); CIGRE case is small (12 assets) ,
use it for development, oberrhein for evaluation.
