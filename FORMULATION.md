### SiMetrics v0.3.0

> **How to read:** every "guardrail" (physics constraint) follows a fixed pattern ,
> **WHAT (plain language) -> INPUT -> FORMULA -> CONSTANTS (value + reason + standard) -> OUTPUT ->
> THEORY/STANDARD -> CODE LOCATION.** Every constant lives in `simetrics/params.py` (no hidden
> numbers). Sections §10 (wasted-life) and §11 (seed) are explained extra-slowly because they cause
> the most confusion.

## §0. Big picture (pipeline)
```
INPUT (public, NO utility data):
  pandapower test feeder (CIGRE MV / mv_oberrhein) + one AC power flow
        │
        ▼
[1] per-asset base loading (raw) ──[2] daily demand curve──┐
        │                                                  │
        ▼                                                  ▼
[3] daily STRESS s ──►[4] hot-spot temp ──►[5] aging (Arrhenius + calendar) ──► RUL decreases
        │                                                  │
        │                                                  ▼
        │                                         [6] PoF = 1 - RUL/4
        ▼                                                  │
[8] coupling matrix (load transfer)                        ▼
        │                          [7] 3 failure channels: wear-out / external / overload-trip
        ▼                                                  │
[7c] overload excess ──────────────────────────────────►  │
                                                           ▼
                          [9] outage → downstream customers → SAIFI / SAIDI (IEEE 1366)
                          [10] maintenance → RUL reset → "wasted usable life"
OUTPUT: daily trajectories (loading, hot-spot, RUL, in-service) + events + SAIFI/SAIDI + trip causes
EVERYTHING is driven by ONE seed (§11) → reproducible.
```

---

## §1. Per-asset base loading (`raw`) , from power flow
**What:** we solve **one** AC power flow on the test feeder; each line gets a "loading_percent"
(percent of its current rating). That is its base thermal load.
**Input:** `net` (pandapower) → `pp.runpp(net)` → `net.res_line.loading_percent`.
**Formula:**
```
raw[i] = loading_percent[i] / 100        (fraction 0..1; NaN → 0)
```
**Output:** vector `raw` (per line-asset). **Theory:** AC power flow (Newton–Raphson).
**Code:** `simetrics/engine.py` `build_topology()`; `simetrics/network.py` `pp.runpp`.

## §2. Daily demand curve (seasonal + weekly + noise)
**What:** load is not static; it rises/falls daily with season and weekday, plus small noise.
**Input:** day index `t = 0..n_days-1`; seeded RNG (for the noise).
**Formula:**
```
curve[t] = clip( 1 + 0.20·sin(2π t/365 − π/2) + 0.07·sin(2π t/7) + N(0, 0.03) ,  0.6 , 1.4 )
```
**Constants:** `SEASON_AMP=0.20` (±20% annual swing), `SEASON_PHASE=−π/2` (minimum at day 0),
`WEEK_AMP=0.07` (±7% weekly), `DEMAND_NOISE=0.03` (±3% daily noise), `DEMAND_CLIP=(0.6,1.4)` (bound so
noise stays realistic). **Theory:** load-profile decomposition into annual+weekly harmonics + noise.
**Code:** `engine.py` `simulate()`.

## §3. Daily stress `s` (+ `ambient`, `s_raw`)
**What:** base load × scenario × curve, plus transferred load from out-of-service neighbours, then
clipped. We separate THREE quantities for honesty:
**Formula:**
```
ambient = LOAD_SCALE · curve[t] · raw                 # normal-state load (no contingency)
s_raw   = LOAD_SCALE · curve[t] · (raw + extra)        # load including contingency transfer
s       = clip(s_raw, 0.20, 1.00) ;  s[out] = 0        # stress used for aging & sensor logging
```
`extra[j]` = total load transferred to neighbour j from all currently-OUT assets (see §8).
**Constants:** `LOAD_SCALE={cigre:1.0, oberrhein:2.0}` (oberrhein snapshot is lightly loaded → ×2 =
"stressed aging fleet" scenario, **disclosed, not hidden**); `S_FLOOR=0.20` (floor , feeder sections
always carry some load); `S_CAP=1.00` (cap , protection limits sustained loading above rating, so the
aging input saturates at rated load). **Output:** `s` (drives §4–§5), `ambient`/`s_raw` (drive §7c).
**Code:** `engine.py` `simulate()`.

## §4. Hot-spot temperature
**What:** higher load → hotter conductor.
**Formula:**
```
hotspot(s) = 30 + 70·s + 30·s^1.6        (°C)
```
**Constants:** `HOT_AMBIENT=30` (°C base), `HOT_LIN=70` (linear, copper-loss-like), `HOT_NONLIN=30`,
`HOT_EXP=1.6` (super-linear, I²R-like). At full load (s=1): 30+70+30 = **130 °C**.
**Theory:** transformer/conductor thermal-rise model (loading-guide form). **Code:** `engine.py` `hotspot()`.

## §5. Aging: Arrhenius (IEEE C57.91) + calendar → RUL decreases
**What:** Arrhenius chemistry , aging rate rises **exponentially** with temperature. Plus calendar
aging (assets age even when idle).
**Formula:**
```
FAA(T)          = exp( 15000/383 − 15000/(T+273) )     # IEEE C57.91 aging acceleration factor
wear_per_day(s) = K_CAL + K_TH · FAA(hotspot(s))        # RUL-units lost per day
RUL            ← max(0, RUL − wear_per_day(s))          # each day
```
**Constants & KEY ANCHOR:** `FAA_B=15000` K (Arrhenius constant, C57.91); **`FAA_TREF=383` K = 110 °C
= the IEEE C57.91 normal-insulation-life reference hot-spot** → **FAA = 1 exactly at 110 °C** (verified:
`faa(110)=1.0000`). The paper (Eq. 3) writes the exact Kelvin form, T<sub>ref</sub> = 383.15 K with
T + 273.15; the code uses the rounded 383 / 273. Both give FAA = 1 at 110 °C, and the rounded constants
are the ones that generated every number reported in the paper. `K_CAL=10/(25·365)≈0.001096` (a new asset, RUL=10, lasts ~25 yr if idle);
`K_TH=0.0018` (calibrated so a new asset at sustained 100% loading reaches end-of-life in ~2 yr).
**Honest note:** this is a **calibrated SYNTHETIC** wear law **anchored to the C57.91 FORM**, not fitted
to any utility's failure history (such data is not public). **Code:** `engine.py` `faa()`, `wear_per_day()`.

## §6. Health → probability of failure (PoF)
**Formula:** `PoF = clip(1 − RUL/4, 0, 1)`. **Constant:** `RUL_FULL=4.0` (PoF ramps over the last 4
RUL-units). RUL≥4 → PoF=0 (healthy); RUL=0 → PoF=1 (critical). **Code:** `engine.py`.

### §6a. What the headline AUC measures (vs RUL) — read this before quoting 0.843
The reference leaderboard scores **Task 1 (failure prognosis)**, which is a **binary classification**,
not a lifetime regression. For each `(asset, day)` row it predicts `P(a wear-out failure occurs within
the next H=90 days)` from **observables only** (loading/hot-spot windows, days-since-service, static
physical features — see `evaluate.py:build_xy`). The model output is therefore a **risk score in
`[0,1]` per asset per day** — not a class label, and not a calendar lifetime. ROC-AUC then measures
*ranking* quality (threshold-free): the probability that a truly-failing row is scored above a
non-failing one; Average Precision is reported alongside because positives are rare (~5%). The label
counts **wear failures only** (event code 0); the unpreventable external floor (§7b) and contingency
trips (§7c) are excluded, since no maintenance policy can pre-empt them.

**RUL is a different, separate task.** `rul` is the hidden ground-truth "fuel gauge" (0–10 RUL-units,
**not years** — §10), shipped as a *label*. Regressing it is **Task 1b** (metric: MAE, paper Sec. VI-B), scored by
`examples/rul_regression.py`; it is a separate, optional task. So the headline number
**AUC = 0.843 is the 90-day failure-classification score, never an RUL accuracy.** A continuous RUL
prediction — the predicted-vs-true "fuel gauge" trajectory rising and falling over time, the way a
forecast is overlaid on a price series — is an optional Task-1b deliverable, left open for users to
implement.

### §6b. Why a high AUC does NOT make a good policy (prediction ≠ decision)
A recurring, important result in this benchmark: the model with the best Task-1 ROC-AUC (MLP, 0.843)
can be one of the **worst** maintenance policies (reproducible with `examples/model_as_policy.py`, which
runs each trained model as a closed-loop policy and traces its visits–SAIFI frontier). This is not a
bug , it is the reason the benchmark scores the **decision** (Task 3), not just **prediction** (Task 1).
ROC-AUC and policy-SAIFI measure fundamentally different things:

- **Threshold-free ranking vs a real cutoff.** AUC only asks "is a failing asset-day ranked above a
  healthy one?"; it never commits to a cutoff. A policy must. A model can rank well yet have no cutoff
  that cleanly separates "visit" from "skip" at a sensible budget , its scores may pile healthy assets
  near risky ones, so catching the few failures also flags a flood of healthy ones (over-servicing).
  Tree models (RF/HistGB) concentrate scores and isolate the risky tail; linear/MLP do not, and
  over-service massively in our runs.
- **Distribution shift / feedback loop.** AUC is measured on **static reactive-run data** , the regime
  the model trained on. Once the model *drives* maintenance, assets get serviced and rarely reach
  end-of-life, so the feature distribution becomes a healthier, censored regime the model never saw.
  Flexible models (MLP) extrapolate unpredictably off their training manifold; tree partitions degrade
  more gracefully. The on-distribution AUC edge need not survive the shift.
- **Temporal consistency.** A policy acts every day; a score that jitters day-to-day mistimes visits.
  AUC, computed on independently sampled rows, is blind to temporal stability.
- **Objective mismatch.** AUC rewards correct ranking of *all* asset-days equally; SAIFI rewards
  preventing the *few high-customer-impact* failures at low visit cost. Optimizing discrimination is
  not optimizing the operational objective.

**Takeaway.** A high AUC is necessary-ish but **not sufficient** for a good decision policy , the classic
*offline-metric vs online-decision gap*. This is exactly why SiMetrics scores the decision (Task 3:
SAIFI/SAIDI at iso-budget), and why a sound approach should **learn the policy** (e.g. MARL on the
SAIFI objective) rather than bolt a threshold onto a good classifier. *(Caveat: the specific
over-servicing of linear/MLP also reflects the simple threshold rule and thresholds taken from the
reactive distribution; better per-regime calibration would help them somewhat , but the gap, and the
lesson, are real.)*

## §7. Three failure channels (honest: no policy can reach SAIFI=0)
### §7a. Wear-out , prognosable
```
daily_hazard = PoF² · FAIL_BASE · (PROT_MULT if serviced ≤70 d ago, else 1)
wear_fail = (rng.random < daily_hazard)
```
**Constants:** `FAIL_BASE=0.02`; the **PoF² (squared)** makes failures concentrate at end-of-life
(→ prognosable, the basis of PdM); `PROT_MULT=0.04` (a freshly serviced asset is 25× safer) for
`PROTECT_D=70` days. **Theory:** bathtub-curve wear-out region (Weibull β>1). **Code:** `engine.py`.
### §7b. External (weather / third-party / animals) , unpreventable
```
ext_fail = (rng.random < EXT_DAILY) ;  EXT_DAILY = 0.02/365  (≈0.02 /asset/yr)
```
**Age-INDEPENDENT** → a **hard floor**: no maintenance policy can remove it → SAIFI=0 is impossible
**by design** (anti-gaming). **Theory:** homogeneous Poisson process. **Code:** `engine.py`.
### §7c. Overload TRIP (contingency cascade channel)
**What:** when an asset is out, its load transfers to neighbours (§8); if a neighbour exceeds its
rating **because of the contingency**, its protection relay trips.
```
excess = max(0, s_raw − max(1.0, ambient))     # ONLY contingency-induced excess
trip   = (not in cooldown) ∧ (rng.random < TRIP_K · excess)
```
**Constants:** `TRIP_K=0.015` (LOW by design , emergency ratings + operator load-shedding absorb most
contingency overloads → trips are a ~10–20% minority of outages); `TRIP_OUT_DAYS=1` (recloses next
day, no damage); `TRIP_COOLDOWN_D=7` (no re-trip , operators do not cycle a breaker back into the same
overload → **at most one trip per asset per contingency**). Normal-state (planning) overloads are
assumed operator-managed and do **not** trip (N-1 convention). **Each trip records its cause set**
(ground-truth explanation: which out asset(s) transferred the load + how much). **Theory:** N-1
contingency / cascading failure. **Code:** `engine.py`.

## §8. Coupling (load-transfer) matrix , derived from REAL power flow
**What:** "if asset X goes out and is switched over (close a tie), how much does each neighbour's
loading rise?" , computed once up front with real power flow.
**Procedure:** for each line X: (1) set `in_service=False`; (2) find orphaned buses; (3) close
normally-open ties one by one, re-solve `runpp`; (4) the first tie that restores supply → record each
neighbour's loading rise; if `Δ > COUPLE_TH (5 points %)` → store `C[X][Y] = Δ/100`.
**Output:** sparse matrix `C` (used by §3 `extra` and §7c). **Code:** `engine.py` `_build_coupling()`.

## §9. Outage → downstream customers → SAIFI / SAIDI (IEEE 1366)
**What:** radial network without FLISR → one asset failing de-energizes **all customers downstream of it**.
**Formula:**
```
downstream_customers[asset] = Σ customers in its subtree     (BFS from the substation)
duration = RESTORE_TRIP_HRS(2h) if trip ; RESTORE_TIE_HRS(3h) if a restoring tie exists ; else RESTORE_NOTIE_HRS(8h)
SAIFI = Σ(downstream_customers per failure) / total_customers / years
SAIDI = Σ(downstream_customers · duration) / total_customers / years
```
**Constants:** `CUST_PER_MW=600`; restore durations above. **Theory:** **IEEE 1366** reliability indices.
**Code:** `network.py` (downstream subtree), `engine.py`.

## §10. Maintenance & "WASTED USABLE LIFE" , explained EXTRA-slowly
**Picture RUL as a fuel gauge for an asset's life:** 10 = full tank (new), 0 = empty (dead). **Its unit
is "RUL-units" (NOT calendar years)** , because 1 RUL-unit equals a different amount of calendar time
depending on load (a heavily loaded asset burns RUL faster; recall §5: new = ~25 yr idle, or ~2 yr at
full load). So "RUL-units" is a **relative efficiency measure**, fair to compare **across policies**
(same scale), not an absolute year count.

**When a planned maintenance** touches an asset, we reset its RUL to `MAINT_RUL=(6..9)`. But the asset
**still had remaining life** before the reset , that remainder is **thrown away**. That is "wasted
usable life":
```
wasted_life_of_one_visit = RUL of the asset AT THE MOMENT it is visited
TOTAL wasted life          = Σ (RUL at visit time)  over ALL planned visits
```
**Worked example (oberrhein, preventive 2×/yr):** ~1,745 visits over 5 yr, average RUL-at-service
≈ **6.77** (meaning: on average assets are serviced while still ~68% healthy , wasteful!). Total
≈ 1,745 × 6.77 ≈ **11,802 RUL-units** (this is the number in the spectrum table). Compare:
- **reactive:** 0 planned visits → wasted **0** (but pays with many failures instead).
- **predictive-ideal p>0.35:** services at RUL≈**2.45** (just before failure) → only ~175 visits ×
  2.45 ≈ **428** wasted.
**Meaning:** "wasted life" is the **"Usable Region"** of the P-F / maturity curve , life you already paid
for but did not use. Preventive is wasteful (services healthy assets); predictive is lean (services just
before failure). **Code:** `engine.py` (`wasted += RUL[k].sum()`).

## §11. SEED
**Step 1 , why computer "randomness" is not truly random.** A computer uses a **formula** (a PRNG, e.g.
Mersenne Twister) that emits a sequence that **looks random** but is **entirely determined** by one
starting number = the **seed**.
**Step 2 , the most concrete analogy (forget card-shuffling):** imagine one **thick printed book of
millions of random-looking digits**. **The seed = the page number you start reading from.** Start at
page 1 → you always read the exact same digits, in order. Start at page 7 → a different (equally
random-looking) stream. **Same page → same stream, on any computer, every time.**
**Step 3 , in OUR simulation, "digits from the book" are used at EXACTLY 4 places:**
  (a) **initial age of each asset** , `rng.uniform(1,10)` for 174 assets → which assets are "born old";
  (b) **daily demand noise** , `rng.normal(0,0.03)` for 1,825 days;
  (c) **daily dice roll per asset** , `rng.random()`: given its hazard, does it actually fail today;
  (d) **RUL after service/repair** , `rng.uniform(...)`.
**Step 4 , the consequence:** the seed fixes the **ENTIRE 5-year "story"** (which assets weaken first,
exactly which day failures land). Same seed → identical story (we prove it: **V1 = bit-identical**).
Different seed → a different but **statistically equivalent** "parallel 5 years".
**Step 5 , why this is MANDATORY:**
  1. **Reproducible:** anyone runs seed=1 → identical numbers → checkable (a requirement of science).
  2. **Fair (common random numbers):** test reactive/preventive/predictive **all on the same seed** →
     they face the **same would-be failures** → the difference is the **STRATEGY**, not luck.
  3. **Not a fluke:** one seed could be a lucky/unlucky year. So we run **8–20 seeds** (8–20 parallel
     years), take the **average** (trustworthy) + the **spread / CV** (how much luck matters). Evidence:
     **V2** , reactive CV 5% (stable), preventive CV 29% (≈ the theoretical rare-event floor 21%;
     real-world annual SAIFI is noisy like this too , what must be stable is the multi-seed mean, SEM ≤10%).
**Code:** `engine.py` `rng = np.random.RandomState(seed)` then all `rng.*`.

## §12. Outputs (what is produced, per file)
- `loading` (days×assets): stress `s` + sensor noise `N(0,0.01)` , **SCADA-like observable**.
- `hotspot` (days×assets): temperature `hotspot(s)`.
- `in_service` (days×assets, 0/1).
- `rul` (days×assets): **hidden ground truth** (label; must **not** be a model input , the real world
  does not observe it).
- `events` (day, asset, code): 0=wear-fail 1=external 2=maintenance 3=repair-done 4=trip 5=reclose.
- `trip_causes` (day, tripped, cause) + `trip_cause_load`: **ground-truth explanations**.
- `summary` (json): SAIFI, SAIDI, failures per channel, visits, wasted-life, λ.

## §13. Calibration anchors & validation (why you can trust it)
- **FAA=1 at 110 °C** (IEEE C57.91) , numerically verified.
- **Preventive λ ≈ 0.047 /asset/yr** = right in the published RBTS range (0.04–0.065) , and **RBTS
  statistics come from preventively-maintained fleets**, so the apples-to-apples comparison is our
  preventive run.
- **Validation suite 10/10** (`scripts/validate_benchmark.py` → `VALIDATION.md`): reproducibility,
  stability, maintenance monotonicity, policy-invariant external floor, prognosability (PoF@fail 0.56 =
  7.8× fleet mean), cascade substrate (lift 2.31× vs null 1.49, z=8.8; stripped-engine control 0.98),
  separability (p≈4×10⁻²⁴), learnability (AUC 0.79), integrity + 713 verified explanation rows.

## §14. Known limitations
Synthetic calibrated wear (not fitted to real utility failure data); demand = one snapshot × seasonal
curve (no AMI profiles); no weather/vegetation model; coupling uses the first-success tie only;
customers proportional to load (600/MW); cosmetic coordinates; chronic normal-state overloads in the
stressed scenario are assumed operator-managed (they age assets but never trip); CIGRE is small (12
assets , for development), oberrhein (174 , for evaluation). Field validation = a Pilot phase (outside
the scope of this simulation benchmark).
