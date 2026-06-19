"""
params.py , SINGLE SOURCE OF TRUTH for every numeric constant in the SiMetrics simulator.

Nothing in the engine is a hidden literal: every coefficient lives here with its value, unit,
meaning, and the justification / standard it is anchored to. The engine imports these names;
change a value here and the whole benchmark changes consistently.

Anchors used: IEEE C57.91 (transformer thermal aging / Arrhenius), IEEE 1366 (SAIFI/SAIDI),
RBTS / CIGRE reliability surveys (failure rates), N-1 contingency convention (cascade trips).
"""
import math

ENGINE_VERSION = "0.3.0"

# =====================================================================================
# 1. NETWORK -> CUSTOMERS
# =====================================================================================
CUST_PER_MW = 600          # customers per MW of connected load (SAIFI/SAIDI weighting).
                           # ~ order-of-magnitude for MV feeders; only relative weights matter.

# =====================================================================================
# 2. LOADING / STRESS  (power-flow base loading -> daily stress fraction s)
# =====================================================================================
LOAD_SCALE = {             # scenario knob: multiply the power-flow base loading per case.
    "cigre": 1.0,          #   CIGRE MV snapshot is already realistically loaded.
    "oberrhein": 2.0,      #   mv_oberrhein snapshot is lightly loaded -> x2 = "stressed aging
}                          #   fleet" scenario (DISCLOSED, not hidden). Unknown case -> 1.0.
S_FLOOR = 0.20             # min stress fraction: even off-peak sections carry non-trivial load.
S_CAP   = 1.00             # stress clipped to 100% of thermal rating; protection limits sustained
                           # loading above nameplate, so the aging input saturates at rated load.

# demand curve: 1 + seasonal + weekly + noise, clipped (dimensionless multiplier on base load)
SEASON_AMP    = 0.20       # +/-20% annual peak-to-trough demand swing
SEASON_PHASE  = -math.pi/2 # annual minimum at day 0, peak mid-year
SEASON_PERIOD = 365        # days (annual cycle)
WEEK_AMP      = 0.07       # +/-7% weekday/weekend modulation
WEEK_PERIOD   = 7          # days (weekly cycle)
DEMAND_NOISE  = 0.03       # +/-3% Gaussian day-to-day demand noise (seeded)
DEMAND_CLIP   = (0.6, 1.4) # bound the multiplier so noise cannot create unrealistic excursions

# =====================================================================================
# 3. HOT-SPOT TEMPERATURE   hotspot(s) = HOT_AMBIENT + HOT_LIN*s + HOT_NONLIN*s**HOT_EXP
# =====================================================================================
HOT_AMBIENT = 30.0         # degC, no-load base / ambient hot-spot
HOT_LIN     = 70.0         # degC per unit loading fraction (linear copper-loss-like term)
HOT_NONLIN  = 30.0         # degC scale of the super-linear (I^2R-like) term
HOT_EXP     = 1.6          # exponent of the super-linear term
# At s=1.0 (rated): hotspot = 30 + 70 + 30 = 130 degC (an above-normal but standard loading-guide
# hot-spot value); aging then accelerates strongly per Arrhenius below.

# =====================================================================================
# 4. AGING  (IEEE C57.91 Arrhenius aging-acceleration factor + calendar aging)
#    FAA(T) = exp(FAA_B/FAA_TREF - FAA_B/(T + CELSIUS_K))
#    wear_per_day(s) = K_CAL + K_TH * FAA(hotspot(s))      [RUL-units lost per day]
# =====================================================================================
FAA_B     = 15000.0        # Arrhenius constant (K) from IEEE C57.91 thermal-aging model
FAA_TREF  = 383.0          # reference absolute temp = 110 degC = IEEE C57.91 normal-insulation-life
                           # reference (FAA=1 exactly at the hot-spot of 110 degC)
CELSIUS_K = 273.0          # Celsius -> Kelvin offset
K_CAL     = 10.0/(25*365)  # calendar aging: a new asset (RUL=10) would last ~25 yr at zero thermal
                           # stress (age even when idle). ~ 0.001096 RUL/day.
K_TH      = 0.0018         # thermal-aging weight: calibrated so a new asset at sustained 100%
                           # loading (FAA at 130 degC) reaches end-of-life in ~2 yr.

# =====================================================================================
# 5. HEALTH STATE  (RUL -> PoF)
# =====================================================================================
RUL_INIT = (1.0, 10.0)     # initial RUL drawn uniformly per asset -> a MIXED-AGE fleet at t=0
RUL_FULL = 4.0             # PoF = clip(1 - RUL/RUL_FULL, 0, 1): PoF ramps over the last 4 RUL-units
MAINT_RUL = (6.0, 9.0)     # planned maintenance restores RUL to this range (refurbish, not new)
EMERG_RUL = (5.0, 8.0)     # emergency repair after a failure restores RUL to this range

# =====================================================================================
# 6. FAILURE CHANNELS
# =====================================================================================
# (1) wear-out: daily hazard = PoF^2 * FAIL_BASE  (squared => failures concentrate at end-of-life)
FAIL_BASE = 0.02           # wear-out hazard scale (daily prob at PoF=1, before protection)
PROTECT_D = 70             # days a freshly serviced asset is "protected"
PROT_MULT = 0.04           # within the protection window, wear hazard x0.04 (25x safer)
# (2) external (weather / third-party / animals): age-INDEPENDENT Poisson floor
EXT_DAILY = 0.02/365       # ~0.02 failures/asset/yr ; no maintenance policy can prevent these
# (3) overload trip (contingency cascade channel): daily trip prob = TRIP_K * excess,
#     excess = max(0, s_with_transfer - max(1.0, s_ambient))   (only CONTINGENCY-induced excess)
TRIP_K          = 0.015    # low BY DESIGN: emergency ratings + operator shedding absorb most
                           # contingency overloads -> trips are a ~10-20% minority of outages.
TRIP_OUT_DAYS   = 1        # a tripped asset recloses next day (no damage, no RUL change)
TRIP_COOLDOWN_D = 7        # no re-trip window after reclose (one trip per asset per contingency)

# =====================================================================================
# 7. OUTAGE DURATIONS (hours of customer interruption -> SAIDI) & repair time
# =====================================================================================
REPAIR_DAYS       = 3      # days an asset is OUT after a (wear/external) failure
RESTORE_TIE_HRS   = 3.0    # customer interruption if a normally-open tie can pick up the section
RESTORE_NOTIE_HRS = 8.0    # ... if no tie can restore it (wait for the repair crew)
RESTORE_TRIP_HRS  = 2.0    # interruption for an overload trip (shedding / partial transfer)

# =====================================================================================
# 8. COUPLING (tie-transfer) precompute
# =====================================================================================
COUPLE_TH = 5.0            # loading-POINTS (%) a neighbour must rise by, after failing X and
                           # closing the restoring tie, to be recorded as a coupling edge of X.

# =====================================================================================
# 9. OBSERVABILITY / DEFAULTS
# =====================================================================================
SCADA_NOISE = 0.01         # Gaussian noise added to the logged loading (SCADA sensor realism)
DEFAULT_YEARS = 5
DEFAULT_ROUNDS_PER_YEAR = 2   # preventive: visits per asset per year (staggered static schedule)
DEFAULT_PREDICT_THRESH = 0.35 # predictive (experiments only): maintain when PoF > this
