"""
lead_time_analysis.py , the DELTA-t analysis: from model alarm to actual failure.

This operationalises the maturity-curve figure:
  - panel (b) preventive : maintain far too early  -> large "Usable Region" thrown away
  - panel (c) predictive : alarm shortly before a PREDICTABLE failure (delta-t small but >0)
  - panel (d) t_optimal  : the balance point , enough lead time to act, minimal life wasted

For every ACTUAL wear failure in the test run (seed 8, reactive), we ask each trained model:
  WHEN did your alarm first fire (score >= theta) in the asset's current life-cycle?
    delta_t      = failure_day - first_alarm_day      (lead time, days)
    RUL@alarm    = ground-truth RUL at the alarm day  (usable life discarded if we maintain then,
                   in the SAME units as the maturity spectrum's RUL@maint , ideal p>.35 = 2.45)
  Detection     = an alarm fired before the failure (within the same life-cycle)
  Alarm budget  = theta set per model on the VALIDATION seed to an equal 5% asset-day alarm rate
                  (fair comparison , same crew workload).
"Closest to t_optimal" = high detection with delta_t >= 7d (actionable: weekly dispatch) and the
LOWEST RUL@alarm (least usable life wasted). External failures and contingency trips are excluded:
they are age-independent BY DESIGN and no condition model can foresee them.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from evaluate import (build_xy, fit_model, faa_proxy, BM, CASE, W_LONG, W_SHORT)

T0 = W_LONG                      # first scoreable day
ALARM_RATE = 0.05                # equal alarm budget (fraction of asset-days flagged)
ACTIONABLE_MIN = 7               # days , next weekly dispatch can still reach the asset


def daily_features(z, top):
    """Vectorised DAILY feature matrix with EXACTLY the training semantics of build_xy."""
    load = z["loading"].astype(np.float32); hot = z["hotspot"].astype(np.float32)
    n_days, n = load.shape
    basel, down = top["raw"].astype(np.float32), top["down"].astype(np.float32)
    cs = np.vstack([np.zeros((1, n), np.float64), np.cumsum(load, axis=0)])
    cs2 = np.vstack([np.zeros((1, n), np.float64), np.cumsum(load.astype(np.float64)**2, axis=0)])
    csh = np.vstack([np.zeros((1, n), np.float64), np.cumsum(hot, axis=0)])
    t = np.arange(T0, n_days)
    m60 = (cs[t] - cs[t-W_LONG]) / W_LONG
    v60 = np.clip((cs2[t] - cs2[t-W_LONG])/W_LONG - m60**2, 0, None)
    s60 = np.sqrt(v60)
    m14 = (cs[t] - cs[t-W_SHORT]) / W_SHORT
    mh60 = (csh[t] - csh[t-W_LONG]) / W_LONG
    swv = sliding_window_view(hot, W_LONG, axis=0).max(axis=-1)     # row i = max over days [i, i+59]
    xh60 = swv[t - W_LONG]                                           # window ending at t-1
    # last reset (day 0 + maintenance/repair events), inclusive of t , as in training
    reset_mark = np.zeros((n_days, n), np.int32)
    for d, a, c in z["events"]:
        if c in (2, 3):
            reset_mark[int(d), int(a)] = int(d)
    last_reset = np.maximum.accumulate(reset_mark, axis=0)[t]
    dsr = (t[:, None] - last_reset).astype(np.float32)
    mfaa = faa_proxy(mh60)
    X = np.stack([dsr, m60, s60, m14 - m60, mh60, xh60, dsr*mfaa,
                  np.broadcast_to(basel, m60.shape), np.broadcast_to(down/1000.0, m60.shape)],
                 axis=-1).astype(np.float32)                         # (T, n, 9)
    return X, last_reset


def lifecycle_first_alarm(S, last_reset, day_f, a, theta):
    """First alarm day within the asset's current life-cycle (after its last reset)."""
    r = int(last_reset[day_f - 1 - T0, a]) if day_f - 1 >= T0 else 0
    lo = max(T0, r)
    win = S[lo - T0: day_f - T0, a]
    hits = np.where(win >= theta)[0]
    return (lo + hits[0]) if len(hits) else None


if __name__ == "__main__":
    from simetrics.engine import build_topology
    t_start = time.time()
    top = build_topology(CASE); n_assets = len(top["ids"])

    print(f"=== Lead-time (delta-t) analysis , {CASE}, equal alarm budget {ALARM_RATE:.0%} ===")
    Xtr, ytr, _ = build_xy(range(1, 7), "reactive", n_assets, top)
    Xva, yva, _ = build_xy([7], "reactive", n_assets, top)

    zte = np.load(f"{BM}/data/{CASE}_reactive_seed08.npz", allow_pickle=True)
    zva = np.load(f"{BM}/data/{CASE}_reactive_seed07.npz", allow_pickle=True)
    Xd_te, lr_te = daily_features(zte, top)
    Xd_va, _ = daily_features(zva, top)
    Tn, n, F = Xd_te.shape
    rul_te = zte["rul"].astype(np.float32)
    wear_fails = [(int(d), int(a)) for d, a, c in zte["events"] if c == 0 and d > T0 + 1]
    n_days = zte["loading"].shape[0]
    # daily label for precision: wear-fail within (t, t+90]
    y_daily = np.zeros((Tn, n), bool)
    for d, a in wear_fails:
        lo = max(T0, d - 90); y_daily[lo - T0: d - T0, a] = True

    print(f"test wear failures: {len(wear_fails)} | scored asset-days: {Tn*n:,}")
    print(f"\n{'model':<8}{'theta':>8}{'detect%':>9}{'med dt':>8}{'IQR dt':>12}{'RUL@alarm':>11}"
          f"{'dt>=7d%':>9}{'precision':>11}")
    rows = []
    for name in ["logreg", "rf", "hgb", "mlp"]:
        f = fit_model(name, 0, Xtr, ytr, Xva, yva)
        Sva = f(Xd_va.reshape(-1, F)).reshape(Tn, n)
        theta = float(np.quantile(Sva, 1 - ALARM_RATE))
        Ste = f(Xd_te.reshape(-1, F)).reshape(Tn, n)
        dts, ruls, miss = [], [], 0
        for d, a in wear_fails:
            ta = lifecycle_first_alarm(Ste, lr_te, d, a, theta)
            if ta is None:
                miss += 1
            else:
                dts.append(d - ta); ruls.append(float(rul_te[ta, a]))
        dts = np.array(dts); ruls = np.array(ruls)
        alarms = Ste >= theta
        precision = float(y_daily[alarms].mean()) if alarms.any() else 0.0
        det = 1 - miss/len(wear_fails)
        q1, q3 = (np.percentile(dts, 25), np.percentile(dts, 75)) if len(dts) else (0, 0)
        actionable = float((dts >= ACTIONABLE_MIN).mean()) if len(dts) else 0.0
        rows.append(dict(model=name, theta=theta, detect=det, med_dt=float(np.median(dts)),
                         q1=q1, q3=q3, rul_alarm=float(ruls.mean()), actionable=actionable,
                         precision=precision))
        print(f"{name:<8}{theta:>8.3f}{100*det:>8.0f}%{np.median(dts):>8.0f}"
              f"{q1:>6.0f}-{q3:<5.0f}{ruls.mean():>10.2f}{100*actionable:>8.0f}%{precision:>10.2f}")

    print(f"\nReference points (same RUL units): ideal-predictive p>0.35 maintains at RUL 2.45; "
          f"p>0.55 at 1.73; preventive-2x wastes RUL 6.77 per visit (maturity spectrum, 12 seeds).")
    best = min([r for r in rows if r["detect"] >= 0.6 and r["actionable"] >= 0.6],
               key=lambda r: r["rul_alarm"], default=None)
    if best:
        print(f"Closest to t_optimal at this alarm budget: **{best['model']}** , detects "
              f"{100*best['detect']:.0f}% of wear failures, median lead {best['med_dt']:.0f} d, "
              f"acts at RUL {best['rul_alarm']:.2f} (vs ideal 2.45, preventive 6.77).")
    # ---- the empirical t_optimal FRONTIER: sweep the alarm budget (threshold) ----
    # The budget is the t_optimal knob: tighter budget -> later alarms (less wasted RUL, lower
    # detection); looser -> earlier (preventive-ish). Where to sit on this frontier, jointly with
    # crew budget and downstream-customer impact, IS the prescriptive (panel-d) problem.
    BUDGETS = [0.01, 0.02, 0.05, 0.10]
    frontier = []
    print("\nFRONTIER (detection vs lead vs wasted RUL at equal alarm budgets):")
    print(f"{'model':<8}{'budget':>7}{'detect%':>9}{'med dt':>8}{'RUL@alarm':>11}")
    for name in ["logreg", "rf", "hgb", "mlp"]:
        f = fit_model(name, 0, Xtr, ytr, Xva, yva)
        Sva = f(Xd_va.reshape(-1, F)).reshape(Tn, n)
        Ste = f(Xd_te.reshape(-1, F)).reshape(Tn, n)
        for budget in BUDGETS:
            th = float(np.quantile(Sva, 1 - budget))
            dts2, ruls2, miss2 = [], [], 0
            for d, a in wear_fails:
                ta = lifecycle_first_alarm(Ste, lr_te, d, a, th)
                if ta is None:
                    miss2 += 1
                else:
                    dts2.append(d - ta); ruls2.append(float(rul_te[ta, a]))
            det2 = 1 - miss2/len(wear_fails)
            frontier.append(dict(model=name, budget=budget, detect=det2,
                                 med_dt=float(np.median(dts2)) if dts2 else 0.0,
                                 rul_alarm=float(np.mean(ruls2)) if ruls2 else 0.0,
                                 n_detected=len(dts2)))
            print(f"{name:<8}{budget:>6.0%}{100*det2:>8.0f}%{frontier[-1]['med_dt']:>8.0f}"
                  f"{frontier[-1]['rul_alarm']:>11.2f}")
    import csv
    with open("results/lead_time_analysis.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows:
            w.writerow(r)
    with open("results/lead_time_frontier.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(frontier[0].keys())); w.writeheader()
        for r in frontier:
            w.writerow(r)
    print(f"saved -> results/lead_time_analysis.csv + results/lead_time_frontier.csv "
          f"({time.time()-t_start:.0f}s)")
