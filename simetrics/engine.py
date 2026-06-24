"""
engine.py , UNIFIED benchmark-grade distribution PdM simulation engine (SiMetrics).

Single source of truth for the shareable benchmark. ALL numeric constants live in params.py;
this file contains only the (transparent) logic. See FORMULATION.md for equation-by-
equation walkthrough of every physics-informed guardrail below.

MODEL CARD (honest, complete , everything a user needs to trust/criticise the data)
-----------------------------------------------------------------------------------
Network        : standard pandapower test feeders (CIGRE MV without DER; mv_oberrhein).
                 Topology, line impedances, lengths, ampacities, voltages are REAL network data.
Loading        : one AC power flow (pandapower runpp) gives each line's base loading fraction;
                 a seasonal+weekly demand curve (+3% Gaussian noise) modulates it daily.
                 LOAD_SCALE creates a "stressed aging fleet" scenario for mv_oberrhein whose
                 snapshot is lightly loaded (documented scenario knob, NOT hidden).
Stress cap     : loading is clipped to S_CAP (=1.0) of thermal rating.
Aging          : RUL decreases daily by wear = K_CAL + K_TH*FAA(hotspot(s)); FAA is the IEEE
                 C57.91 Arrhenius aging-acceleration form (FAA=1 at 110 degC reference).
                 Calibrated synthetic wear anchored to the C57.91 form , NOT fitted to utility data.
Failures       : (1) WEAR-OUT hazard PoF^2*FAIL_BASE (end-of-life concentrated => prognosable);
                 (2) EXTERNAL Poisson floor EXT_DAILY (age-independent, unpreventable);
                 (3) OVERLOAD TRIP TRIP_K*max(0, s_transfer - max(1, s_ambient)) (N-1 cascade;
                 one trip per asset per contingency via cooldown). PROT_MULT during PROTECT_D.
Outage impact  : radial; a failure interrupts ALL downstream customers (IEEE 1366). Restore time
                 = RESTORE_TIE_HRS if a tie can pick up the section, else RESTORE_NOTIE_HRS;
                 a trip = RESTORE_TRIP_HRS.
Coupling       : while an asset is out, transferred load raises specific neighbours' loading
                 (coupling matrix precomputed by failing each line in power flow + closing the
                 restoring tie). Drives the spatial-correlation substrate.
Maintenance    : restores RUL to MAINT_RUL (planned) or EMERG_RUL (after failure repair).
Explanations   : every overload trip carries a GROUND-TRUTH CAUSE SET (out asset(s) whose
                 transferred load created the excess) + each cause's transferred-loading share.
Randomness     : one seeded RNG => same seed = bit-identical run; different seeds = statistically
                 equivalent "parallel grids".
Known limits   : synthetic wear calibration; demand = single snapshot x seasonal curve (no AMI);
                 no weather/vegetation model; coupling uses first-success tie only; customers
                 proportional to load (600/MW); coordinates cosmetic; chronic normal-state
                 overloads in the stressed scenario are assumed operator-managed (age, never trip).
"""
import math
import numpy as np
import pandapower as pp
import pandapower.networks as pn

from .network import build_distribution
from .params import *   # noqa: F401,F403  (all constants; see params.py for values+justification)


def hotspot(s):
    """Conductor hot-spot temperature (degC) from loading fraction s."""
    s = np.clip(s, 0.0, None)
    return HOT_AMBIENT + HOT_LIN*s + HOT_NONLIN*np.power(s, HOT_EXP)


def faa(T):
    """IEEE C57.91 aging acceleration factor at hot-spot temperature T (degC). FAA=1 at 110 degC."""
    return np.exp(FAA_B/FAA_TREF - FAA_B/(np.asarray(T) + CELSIUS_K))


def wear_per_day(s):
    """RUL-units consumed per day at stress s = calendar aging + thermal (Arrhenius) aging."""
    return K_CAL + K_TH*faa(hotspot(s))


def _maker(case):
    return pn.create_cigre_network_mv(with_der=False) if case.startswith("cigre") else pn.mv_oberrhein()


def _supplied(net):
    return set(net.res_bus.index[~net.res_bus.vm_pu.isna()])


def _build_coupling(case, ids, verbose=False):
    """Fail each line -> find a normally-open tie that restores the orphaned section -> record
    which neighbours' loading rises by > COUPLE_TH points (all via real power flow, in-place)."""
    net = _maker(case); pp.runpp(net)
    base_load = net.res_line.loading_percent.copy()
    base_sup = _supplied(net)
    ties = {int(sw): int(net.switch.at[sw, "element"]) for sw in net.switch.index
            if net.switch.at[sw, "et"] == "l" and not bool(net.switch.at[sw, "closed"])}
    tie_lines = set(ties.values())
    C = {}
    probe = [int(i) for i in ids if int(i) not in tie_lines]
    for cnt, L in enumerate(probe):
        if verbose and cnt % 25 == 0:
            print(f"    coupling probe {cnt}/{len(probe)}")
        net.line.at[L, "in_service"] = False
        ok = True
        try:
            pp.runpp(net)
        except Exception:
            ok = False
        if ok:
            orphan = base_sup - _supplied(net)
            if orphan:
                for sw in ties:
                    net.switch.at[sw, "closed"] = True
                    good = True
                    try:
                        pp.runpp(net)
                    except Exception:
                        good = False
                    if good and orphan.issubset(_supplied(net)):
                        nl = net.res_line.loading_percent
                        bump = {}
                        for j in ids:
                            j = int(j)
                            if j == L:
                                continue
                            dd = float(nl.get(j, np.nan)) - float(base_load.get(j, np.nan))
                            if dd == dd and dd > COUPLE_TH:
                                bump[j] = round(dd/100.0, 4)
                        if bump:
                            C[L] = bump
                        net.switch.at[sw, "closed"] = False
                        break
                    net.switch.at[sw, "closed"] = False
        net.line.at[L, "in_service"] = True
    return C


def build_topology(case, verbose=False):
    """Static benchmark topology: assets (lines with downstream customers), physical features,
    electrical line-graph edges, tie-transfer coupling matrix, downstream-customer weights."""
    d = build_distribution(case, seed=0)
    net = d["net"]
    lines = [a for a in d["assets"] if a["kind"] == "line" and a["downstream"] > 0]
    ids = [int(a["idx"]) for a in lines]
    down = np.array([a["downstream"] for a in lines], float)
    ll = net.res_line.loading_percent
    raw, feat = [], []
    for a in lines:
        i = a["idx"]; r = net.line.loc[i]
        L = float(r.length_km) if r.length_km == r.length_km and r.length_km > 0 else 1.0
        rr, xx = float(r.r_ohm_per_km)*L, float(r.x_ohm_per_km)*L
        amp = float(r.max_i_ka) if r.max_i_ka == r.max_i_ka else 0.0
        vn = float(net.bus.at[int(r.from_bus), "vn_kv"])
        lo = float(ll.get(i, 0.0) or 0.0)/100.0
        lo = 0.0 if lo != lo else lo
        raw.append(lo)
        feat.append([rr, xx, math.hypot(rr, xx), L, amp, vn, lo, float(a["downstream"])])
    raw = np.array(raw)
    bus2 = {}
    for k, a in enumerate(lines):
        for b in (a["u"], a["v"]):
            bus2.setdefault(b, []).append(k)
    eedges = sorted({(min(x, y), max(x, y)) for ls in bus2.values() for x in ls for y in ls if x != y})
    if verbose:
        print(f"  [{d['label']}] {len(ids)} line-assets; probing tie-transfer coupling...")
    Cid = _build_coupling(case, ids, verbose=verbose)
    id2k = {i: k for k, i in enumerate(ids)}
    C = {id2k[X]: {id2k[Y]: v for Y, v in ys.items() if Y in id2k} for X, ys in Cid.items() if X in id2k}
    return dict(case=case, label=d["label"], ids=np.array(ids, int), down=down, raw=raw,
                feat=np.array(feat, float),
                feat_names=["r_ohm", "x_ohm", "z_ohm", "length_km", "ampacity_ka", "vn_kv",
                            "base_loading", "downstream_customers"],
                elec_edges=np.array(eedges, int).reshape(-1, 2), coupling=C,
                total=int(d["total_customers"]))


# event codes in the recorded dataset
EV_FAIL_WEAR, EV_FAIL_EXT, EV_MAINT, EV_REPAIR_DONE, EV_TRIP, EV_RECLOSE = 0, 1, 2, 3, 4, 5


def simulate(top, policy, seed, years=DEFAULT_YEARS, rounds_per_year=DEFAULT_ROUNDS_PER_YEAR,
             predict_thresh=DEFAULT_PREDICT_THRESH, record=True, score_fn=None):
    """One full run. Returns summary + (if record) daily logs of SCADA-like observables and
    hidden ground-truth RUL. policy in {'reactive','preventive','predictive'}."""
    rng = np.random.RandomState(seed)
    n = len(top["ids"]); n_days = int(years*365)
    scale = LOAD_SCALE.get(top["case"], 1.0)
    raw, down, total, C = top["raw"], top["down"], top["total"], top["coupling"]
    restorable = np.array([k in C for k in range(n)])
    rul = rng.uniform(*RUL_INIT, n)
    last_maint = np.full(n, -10**9, float)
    out_until = np.full(n, -1, int)              # asset is OUT while day < out_until
    t = np.arange(n_days)
    curve = np.clip(1 + SEASON_AMP*np.sin(2*np.pi*t/SEASON_PERIOD + SEASON_PHASE)
                    + WEEK_AMP*np.sin(2*np.pi*t/WEEK_PERIOD)
                    + rng.normal(0, DEMAND_NOISE, n_days), DEMAND_CLIP[0], DEMAND_CLIP[1])
    offsets = (np.arange(n) * (365.0/rounds_per_year/n)).astype(int)
    interval = int(round(365.0/rounds_per_year))
    if record:
        load_log = np.zeros((n_days, n), np.float32)
        hot_log = np.zeros((n_days, n), np.float32)
        rul_log = np.zeros((n_days, n), np.float32)
        insrv_log = np.zeros((n_days, n), np.uint8)
        ci_cum = np.zeros(n_days, np.float64)     # cumulative customer-interruptions
        cmh_cum = np.zeros(n_days, np.float64)    # cumulative customer-hours
    # model-as-policy: a trained model decides WHEN from OBSERVABLE history (additive; does not
    # touch the other policies' RNG/logic). score_fn(day, load_hist, hot_hist, last_maint) -> risk.
    model_mode = (policy == "model")
    if model_mode and score_fn is None:
        raise ValueError("policy 'model' requires score_fn(day, load_hist, hot_hist, last_maint)")
    if model_mode:
        mload_hist = np.zeros((n_days, n), np.float32)
        mhot_hist = np.zeros((n_days, n), np.float32)
    events = []
    trip_causes = []                 # (day, tripped_asset, cause_asset) ground-truth explanation
    trip_cause_load = []             # transferred-loading contribution of that cause
    saifi_n = saidi_n = 0.0
    fails_wear = fails_ext = fails_trip = visits = 0
    wasted = 0.0; rul_at_maint = []

    trip_out = np.zeros(n, bool)                 # current outage is a trip (no damage)
    last_trip = np.full(n, -10**9)               # for the no-retrip cooldown
    for day in range(n_days):
        # outages ending today: damaged assets return refurbished; tripped assets just reclose
        done = out_until == day
        if done.any():
            for kk in np.where(done)[0]:
                if trip_out[kk]:
                    trip_out[kk] = False
                    events.append((day, int(kk), EV_RECLOSE))
                else:
                    rul[kk] = rng.uniform(*EMERG_RUL); last_maint[kk] = day
                    events.append((day, int(kk), EV_REPAIR_DONE))
        out = out_until > day

        # transferred load from currently-out assets onto their tie-coupled neighbours
        extra = np.zeros(n)
        for k in np.where(out)[0]:
            for j, v in C.get(int(k), {}).items():
                extra[j] += v
        ambient = scale*curve[day]*raw
        s_raw = scale*curve[day]*(raw + extra)
        s = np.clip(s_raw, S_FLOOR, S_CAP)
        s[out] = 0.0

        # planned maintenance (policy decides WHEN)
        if policy == "preventive":
            due = (day >= offsets) & ((day - offsets) % interval == 0) & ~out
        elif policy == "predictive":
            pof_now = np.clip(1 - rul/RUL_FULL, 0, 1)
            due = (pof_now > predict_thresh) & ~out
        elif policy == "model":
            # event-driven: service an asset when the model's risk score crosses a threshold
            # (score_fn returns 0 before enough history, so no visits during warm-up).
            probs = np.asarray(score_fn(day, mload_hist, mhot_hist, last_maint), float)
            due = (probs > predict_thresh) & ~out
        elif policy == "reactive":
            due = np.zeros(n, bool)
        else:
            raise ValueError(policy)
        if due.any():
            k = np.where(due)[0]
            wasted += float(rul[k].sum()); visits += len(k)
            rul_at_maint.extend(rul[k].tolist())
            for kk in k:
                events.append((day, int(kk), EV_MAINT))
            rul[k] = rng.uniform(*MAINT_RUL, len(k)); last_maint[k] = day

        # aging (C57.91-anchored wear)
        w = wear_per_day(s); w[out] = 0.0
        rul = np.maximum(0.0, rul - w)

        # failures: wear-out (EOL-concentrated) + external floor + CONTINGENCY overload trips
        pof = np.clip(1 - rul/RUL_FULL, 0, 1)
        prot = (day - last_maint) <= PROTECT_D
        p_wear = (pof**2)*FAIL_BASE*np.where(prot, PROT_MULT, 1.0)
        f_wear = (~out) & (rng.random(n) < p_wear)
        f_ext = (~out) & ~f_wear & (rng.random(n) < EXT_DAILY)
        excess = np.maximum(0.0, s_raw - np.maximum(1.0, ambient))   # contingency-induced only
        excess[out] = 0.0
        cooldown = (day - last_trip) <= TRIP_COOLDOWN_D
        f_trip = (~out) & ~f_wear & ~f_ext & ~cooldown & (rng.random(n) < TRIP_K*excess)
        f = f_wear | f_ext | f_trip
        if f.any():
            k = np.where(f)[0]
            saifi_n += float(down[k].sum())
            hrs = np.where(f_trip[k], RESTORE_TRIP_HRS,
                           np.where(restorable[k], RESTORE_TIE_HRS, RESTORE_NOTIE_HRS))
            saidi_n += float((down[k]*hrs).sum())
            out_until[k] = np.where(f_trip[k], day + TRIP_OUT_DAYS, day + REPAIR_DAYS)
            trip_out[k] = f_trip[k]
            tripped = k[f_trip[k]]
            last_trip[tripped] = day
            for kk in tripped:       # ground-truth cause set: out assets transferring onto kk
                for j in np.where(out)[0]:
                    v = C.get(int(j), {}).get(int(kk), 0.0)
                    if v > 0:
                        trip_causes.append((day, int(kk), int(j)))
                        trip_cause_load.append(v)
            fails_wear += int(f_wear.sum()); fails_ext += int(f_ext.sum())
            fails_trip += int(f_trip.sum())
            for kk in k:
                events.append((day, int(kk),
                               EV_FAIL_WEAR if f_wear[kk] else (EV_FAIL_EXT if f_ext[kk] else EV_TRIP)))

        if record:
            load_log[day] = s + rng.normal(0, SCADA_NOISE, n)   # SCADA-like sensor noise
            hot_log[day] = hotspot(s)
            rul_log[day] = rul
            insrv_log[day] = (~(out_until > day)).astype(np.uint8)
            ci_cum[day] = saifi_n; cmh_cum[day] = saidi_n

        if model_mode:               # keep observable history for the model's next decisions
            mload_hist[day] = s + rng.normal(0, SCADA_NOISE, n)   # same SCADA-noise form as training data
            mhot_hist[day] = hotspot(s)

    yrs = n_days/365.0
    summary = dict(case=top["case"], policy=policy, seed=int(seed), years=years,
                   rounds_per_year=(rounds_per_year if policy == "preventive" else None),
                   predict_thresh=(predict_thresh if policy == "predictive" else None),
                   saifi=saifi_n/total/yrs, saidi=saidi_n/total/yrs,
                   failures=fails_wear + fails_ext + fails_trip, wear_failures=fails_wear,
                   external_failures=fails_ext, trip_failures=fails_trip, visits=visits,
                   wasted_life=wasted, wasted_life_per_asset_yr=wasted/n/yrs,
                   rul_at_maint=(float(np.mean(rul_at_maint)) if rul_at_maint else 0.0),
                   lambda_per_asset_yr=(fails_wear + fails_ext + fails_trip)/n/yrs,
                   engine_version=ENGINE_VERSION)
    out = dict(summary=summary, events=np.array(events, np.int32).reshape(-1, 3),
               trip_causes=np.array(trip_causes, np.int32).reshape(-1, 3),
               trip_cause_load=np.array(trip_cause_load, np.float32))
    if record:
        out.update(loading=load_log, hotspot=hot_log, rul=rul_log, in_service=insrv_log,
                   ci_cum=ci_cum.astype(np.float32), cmh_cum=cmh_cum.astype(np.float32))
    return out
