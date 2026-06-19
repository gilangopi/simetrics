"""
make_topology.py , draw the grid topology and an animated power-flow contingency GIF.

Run from anywhere:
    python viz/make_topology.py

Outputs (to viz/figures/):
  fig6_topology_cigre.png      , schematic map of the CIGRE MV feeder
  fig6_topology_oberrhein.png  , real-geography map of the mv_oberrhein feeder
  powerflow_contingency.gif    , the contingency story on CIGRE MV:
       (A) electricity flows OUT from the substation, ring by ring, with per-line loading %;
       (B) a high-impact line FAILS (red alert) -> its downstream customers go dark;
       (C) a normally-open tie switch closes -> neighbours ABSORB the transferred load,
           their loading jumps (+delta%), any overload is flagged red.

All numbers (loading %, who-feeds-whom, the transferred load) come from real pandapower AC
power-flow solves , nothing is hand-drawn.
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import pandapower as pp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from simetrics.network import build_distribution
from simetrics.engine import _maker, _supplied

OUT = os.path.join(ROOT, "viz", "figures"); os.makedirs(OUT, exist_ok=True)
BM = "data/simetrics_benchmark"          # dataset root (CWD is ROOT)
CMAP = plt.cm.RdYlGn_r          # green = lightly loaded, red = heavily loaded
NORM = Normalize(0, 100)
GOLD, DARK, REDX, TIEG = "#f2c200", "#9e9e9e", "#d62728", "#2ca02c"


def _lines(net):
    return [(int(i), int(r["from_bus"]), int(r["to_bus"])) for i, r in net.line.iterrows()]


# ----------------------------------------------------------------- STATIC TOPOLOGY
def topology_static(case):
    d = build_distribution(case); net = d["net"]; pos = d["positions"]; pp.runpp(net)
    lines = _lines(net); load = net.res_line.loading_percent
    bd = d["bus_downstream"]; srcs = d["sources"]
    maxd = max(bd.values()) or 1
    fig, ax = plt.subplots(figsize=(11, 8))
    label = case != "oberrhein"                      # too many lines to label on oberrhein
    for idx, u, v in lines:
        if u not in pos or v not in pos:
            continue
        (x1, y1), (x2, y2) = pos[u], pos[v]
        lo = float(load.get(idx, 0) or 0)
        ax.plot([x1, x2], [y1, y2], color=CMAP(NORM(lo)), lw=2.2, zorder=1, solid_capstyle="round")
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2, f"{lo:.0f}%", fontsize=6, color="#333",
                    ha="center", va="center", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))
    for b, (x, y) in pos.items():
        if b in srcs:
            continue
        ax.scatter(x, y, s=30 + 220*(bd.get(b, 0)/maxd), color="#4575b4",
                   edgecolor="k", lw=0.4, zorder=3)
    for s in srcs:
        if s in pos:
            ax.scatter(*pos[s], marker="*", s=600, color=GOLD, edgecolor="k", lw=0.8,
                       zorder=5, label="substation (source)")
    sm = ScalarMappable(NORM, CMAP); sm.set_array([])
    fig.colorbar(sm, ax=ax, label="line loading % (AC power flow)", shrink=0.7)
    handles = [Line2D([], [], marker="*", color="w", markerfacecolor=GOLD, markeredgecolor="k",
                      markersize=18, label="substation (electricity source)"),
               Line2D([], [], marker="o", color="w", markerfacecolor="#4575b4", markeredgecolor="k",
                      markersize=10, label="bus (node size = downstream customers)"),
               Line2D([], [], color=CMAP(NORM(20)), lw=3, label="line (color = loading %)")]
    ax.legend(handles=handles, loc="best", fontsize=9, framealpha=0.9)
    ax.set_title(f"SiMetrics topology , {d['label']}  "
                 f"({d['n_bus']} buses, {d['n_line']} lines)", fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    p = os.path.join(OUT, f"fig6_topology_{case}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  saved -> viz/figures/fig6_topology_{case}.png")


# ----------------------------------------------------------------- CONTINGENCY SCENARIO
def find_scenario(case):
    base = _maker(case); pp.runpp(base)
    bl = base.res_line.loading_percent.copy(); base_sup = _supplied(base)
    ties = {int(sw): int(base.switch.at[sw, "element"]) for sw in base.switch.index
            if base.switch.at[sw, "et"] == "l" and not bool(base.switch.at[sw, "closed"])}
    tie_lines = set(ties.values())
    cand = [int(i) for i in base.line.index if int(i) not in tie_lines]
    best = None
    for L in cand:
        n2 = _maker(case); n2.line.at[L, "in_service"] = False
        try: pp.runpp(n2)
        except Exception: continue
        orphans = base_sup - _supplied(n2)
        if not orphans:
            continue
        for sw, Lt in ties.items():
            n3 = _maker(case); n3.line.at[L, "in_service"] = False; n3.switch.at[sw, "closed"] = True
            try: pp.runpp(n3)
            except Exception: continue
            if orphans.issubset(_supplied(n3)):
                post = n3.res_line.loading_percent
                risers = {int(j): float(post[j]-bl[j]) for j in base.line.index
                          if j != L and float(post.get(j, 0)-bl.get(j, 0)) > 5}
                mb = max(risers.values()) if risers else 0
                if best is None or mb > best["maxb"]:
                    best = dict(L=L, tie_line=Lt, orphans=set(int(b) for b in orphans),
                                post={int(j): float(post[j]) for j in base.line.index},
                                risers=risers, maxb=mb)
                break
    best["base"] = {int(j): float(bl[j]) for j in base.line.index}
    return best


def gif_contingency(case="cigre"):
    d = build_distribution(case); net = d["net"]; pos = d["positions"]
    lines = _lines(net); src = sorted(d["sources"])[0]
    bc = d["bus_customers"]
    sc = find_scenario(case)
    L, tie_L, orphans = sc["L"], sc["tie_line"], sc["orphans"]
    base, post, risers = sc["base"], sc["post"], sc["risers"]
    uv = {idx: (u, v) for idx, u, v in lines}
    fu, fv = uv[L]; darkN = sum(bc.get(b, 0) for b in orphans)

    # BFS rings from the substation over the full OPERATING graph (includes the trafo edge that
    # feeds the MV busbar, so the source is reachable; line-only graph would miss it).
    G = d["graph"].copy()
    if "SRC" in G:
        G.remove_node("SRC")
    depth = nx.shortest_path_length(G, src)
    maxring = max(depth.values()) if depth else 1
    ring_of = {idx: max(depth.get(u, 0), depth.get(v, 0)) for idx, u, v in lines}

    # frame plan
    frames = [dict(p="A", r=r) for r in range(maxring+1)] + [dict(p="A", r=maxring)]*2
    frames += [dict(p="B1")]*2 + [dict(p="B2")]*3
    frames += [dict(p="C1")]*2 + [dict(p="C2")]*4

    fig, ax = plt.subplots(figsize=(10, 8))
    sm = ScalarMappable(NORM, CMAP); sm.set_array([])
    fig.colorbar(sm, ax=ax, label="line loading % (AC power flow)", shrink=0.7)

    def seg(idx):
        u, v = uv[idx]; return pos[u], pos[v]

    def draw(i):
        fr = frames[i]; ph = fr["p"]; ax.cla(); ax.set_xticks([]); ax.set_yticks([])
        dark = orphans if ph in ("B2", "C1") else set()      # dark = customers cut (before restore)
        loadmap = post if ph == "C2" else base
        energized = (set(ring_of[idx] <= fr["r"] and idx for idx, _, _ in lines) - {False}
                     if ph == "A" else set(idx for idx, _, _ in lines))
        for idx, u, v in lines:
            if idx == L and ph != "A":                       # the failed asset
                (x1, y1), (x2, y2) = seg(idx)
                ax.plot([x1, x2], [y1, y2], color=REDX, lw=3.5, ls=":", zorder=2)
                ax.scatter((x1+x2)/2, (y1+y2)/2, marker="X", s=240, color=REDX,
                           edgecolor="k", lw=1, zorder=6)
                continue
            (x1, y1), (x2, y2) = seg(idx)
            on = (idx in energized)
            col = CMAP(NORM(loadmap.get(idx, 0))) if on else "#dddddd"
            w = 2.0
            if ph == "A" and ring_of[idx] == fr["r"]:        # wavefront glow
                w = 4.5; col = GOLD
            if ph == "C2" and idx in risers:                 # neighbours that absorbed load
                w = 3.0 + min(3, risers[idx]/20)
            over = on and loadmap.get(idx, 0) > 100
            ax.plot([x1, x2], [y1, y2], color=("#000" if over else col),
                    lw=(w+2 if over else w), zorder=2, solid_capstyle="round")
            if over:
                ax.plot([x1, x2], [y1, y2], color=REDX, lw=w, zorder=3)
            if on and idx != L:                              # the calculated number
                lab = f"{loadmap.get(idx,0):.0f}%"
                if ph == "C2" and idx in risers:
                    lab += f"  +{risers[idx]:.0f}"
                ax.text((x1+x2)/2, (y1+y2)/2, lab, fontsize=6.5,
                        color="#b30000" if (ph == "C2" and idx in risers) else "#222",
                        ha="center", va="center", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
        # transformer connector(s) so the substation source is visibly wired to the MV busbar
        for _, tr in net.trafo.iterrows():
            hv, lv = int(tr["hv_bus"]), int(tr["lv_bus"])
            if hv in pos and lv in pos:
                (x1, y1), (x2, y2) = pos[hv], pos[lv]
                ax.plot([x1, x2], [y1, y2], color="#555", lw=3, zorder=1, solid_capstyle="round")
        # the restoring tie
        if ph in ("C1", "C2"):
            (x1, y1), (x2, y2) = seg(tie_L)
            ax.plot([x1, x2], [y1, y2], color=TIEG, lw=3, ls=("--" if ph == "C1" else "-"), zorder=2)
        # buses
        for b, (x, y) in pos.items():
            if b == src:
                continue
            c = DARK if b in dark else ("#4575b4" if (ph != "A" or depth.get(b, 99) <= fr["r"]) else "#dddddd")
            ax.scatter(x, y, s=70, color=c, edgecolor="k", lw=0.4, zorder=3)
        ax.scatter(*pos[src], marker="*", s=650, color=GOLD, edgecolor="k", lw=0.9, zorder=7)
        titles = {"A": ("Electricity flows OUT from the substation",
                        f"energizing ring {fr.get('r',0)}/{maxring} , numbers = line loading %"),
                  "B1": ("⚠  ASSET FAILURE", f"line (bus {fu}-{fv}) has tripped"),
                  "B2": ("⚠  CUSTOMERS IN THE DARK", f"{darkN:,} customers downstream lose power"),
                  "C1": ("Reconfiguration", "closing a normally-open TIE switch (green)"),
                  "C2": ("Neighbours ABSORB the transferred load",
                         "their loading jumps (+delta%); black/red = overload")}
        t, sub = titles[ph]
        ax.set_title(f"{t}\n{sub}", fontweight="bold", fontsize=12,
                     color=REDX if ph in ("B1", "B2") else "#222")
        handles = [Line2D([], [], marker="*", color="w", markerfacecolor=GOLD, markeredgecolor="k",
                          markersize=16, label="substation (source)"),
                   Line2D([], [], marker="o", color="w", markerfacecolor="#4575b4", markersize=10,
                          label="energized bus"),
                   Line2D([], [], marker="o", color="w", markerfacecolor=DARK, markersize=10,
                          label="de-energized (dark)"),
                   Line2D([], [], color=REDX, lw=3, ls=":", label="failed line"),
                   Line2D([], [], color=TIEG, lw=3, label="tie switch (backup path)"),
                   Line2D([], [], color="#555", lw=3, label="transformer (HV->MV)")]
        ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=700)
    p = os.path.join(OUT, "powerflow_contingency.gif")
    anim.save(p, writer=animation.PillowWriter(fps=1.6)); plt.close(fig)
    print(f"  saved -> viz/figures/powerflow_contingency.gif  "
          f"(fail line {L} bus {fu}-{fv}; {darkN:,} dark; tie line {tie_L}; "
          f"max neighbour +{sc['maxb']:.0f} pts)")


# ----------------------------------------------------------------- 5-YEAR REPLAY (real run file)
def replay_gif(case, policy="reactive", seed=1, every=30, fps=6):
    """Replay an ACTUAL saved 5-year run, time-stepped. Edges are colored by each asset's failure
    risk PoF, and LABELED with that PoF number (so you see exactly why the color is what it is).
    Red X = a failure during that period; grey = out for repair; live counters in the title.
    100% read from the saved run file , zero curation, no power-flow re-solve."""
    import bisect
    d = build_distribution(case); net = d["net"]; pos = d["positions"]; src = sorted(d["sources"])[0]
    top = np.load(f"{BM}/topology_{case}.npz", allow_pickle=True)
    ids = [int(i) for i in top["ids"]]; total = int(top["total_customers"])
    z = np.load(f"{BM}/data/{case}_{policy}_seed{seed:02d}.npz", allow_pickle=True)
    rul = z["rul"].astype(float); insrv = z["in_service"]; ci = z["ci_cum"].astype(float)
    fev = sorted((int(dd), int(a)) for dd, a, c in z["events"] if c in (0, 1, 4))
    fdays = [dd for dd, _ in fev]
    n_days, n = rul.shape
    seg = {}
    for k, li in enumerate(ids):
        r = net.line.loc[li]; u, v = int(r["from_bus"]), int(r["to_bus"])
        if u in pos and v in pos:
            seg[k] = (pos[u], pos[v])
    NRISK = Normalize(0, 1); CMR = plt.cm.RdYlGn_r            # 0 green (healthy) -> 1 red (failing)
    label_all = (case == "cigre")                            # oberrhein: label the riskiest only
    frames = list(range(0, n_days, every)) + [n_days-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    sm = ScalarMappable(NRISK, CMR); sm.set_array([])
    fig.colorbar(sm, ax=ax, label="failure risk  PoF = 1 - RUL/4   (0 healthy -> 1 about to fail)", shrink=0.7)

    def draw(fi):
        t = frames[fi]; ax.cla(); ax.set_xticks([]); ax.set_yticks([])
        pof = np.clip(1 - rul[t]/4.0, 0, 1); out = insrv[t] == 0
        win = set(a for dd, a in fev if t-every < dd <= t)
        topk = set(sorted(range(n), key=lambda k: -pof[k])[:8])
        for _, tr in net.trafo.iterrows():
            hv, lv = int(tr["hv_bus"]), int(tr["lv_bus"])
            if hv in pos and lv in pos:
                (x1, y1), (x2, y2) = pos[hv], pos[lv]
                ax.plot([x1, x2], [y1, y2], color="#555", lw=2, zorder=1)
        for k, ((x1, y1), (x2, y2)) in seg.items():
            col = "#cccccc" if out[k] else CMR(NRISK(pof[k]))
            ax.plot([x1, x2], [y1, y2], color=col, lw=(1.4 if out[k] else 2.6),
                    zorder=2, solid_capstyle="round")
            if k in win:
                ax.scatter((x1+x2)/2, (y1+y2)/2, marker="X", s=150, color=REDX,
                           edgecolor="k", lw=0.8, zorder=6)
            if label_all or k in topk or k in win:
                ax.text((x1+x2)/2, (y1+y2)/2, f"{pof[k]:.2f}", fontsize=6.5,
                        color=("#b30000" if k in win else "#222"), ha="center", va="center",
                        zorder=5, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
        for b, (x, y) in pos.items():
            if b != src:
                ax.scatter(x, y, s=32, color="#4575b4", edgecolor="k", lw=0.3, zorder=3)
        ax.scatter(*pos[src], marker="*", s=600, color=GOLD, edgecolor="k", lw=0.8, zorder=7)
        cumf = bisect.bisect_right(fdays, t)
        ax.set_title(f"{d['label']} , 5-year replay ({policy}, seed {seed})\n"
                     f"Year {t/365:.1f}   |   failures so far: {cumf}   |   "
                     f"cumulative interruptions / customer: {ci[t]/total:.2f}",
                     fontweight="bold", fontsize=12)
        handles = [Line2D([], [], marker="*", color="w", markerfacecolor=GOLD, markeredgecolor="k",
                          markersize=15, label="substation"),
                   Line2D([], [], marker="X", color="w", markerfacecolor=REDX, markeredgecolor="k",
                          markersize=11, label="failure this period"),
                   Line2D([], [], color="#cccccc", lw=3, label="out for repair"),
                   Line2D([], [], color="#555", lw=3, label="transformer"),
                   Line2D([], [], color=CMR(0.2), lw=3, label="line (color+number = PoF)")]
        ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000/fps)
    p = os.path.join(OUT, f"powerflow_replay_{case}.gif")
    anim.save(p, writer=animation.PillowWriter(fps=fps)); plt.close(fig)
    print(f"  saved -> viz/figures/powerflow_replay_{case}.gif  "
          f"({len(frames)} frames, {len(fdays)} failures over 5 yr)")


if __name__ == "__main__":
    print("=== SiMetrics topology + power-flow GIFs -> viz/figures/ ===")
    for case in ("cigre", "oberrhein"):
        try:
            print(f"[topology {case}]"); topology_static(case)
        except Exception as e:
            print(f"  !! skipped ({type(e).__name__}: {e})")
    try:
        print("[contingency mechanism GIF , cigre]"); gif_contingency("cigre")
    except Exception as e:
        print(f"  !! contingency GIF skipped ({type(e).__name__}: {e})")
    for case in ("cigre", "oberrhein"):
        try:
            print(f"[5-year replay GIF , {case}]"); replay_gif(case)
        except Exception as e:
            print(f"  !! replay skipped ({type(e).__name__}: {e})")
    print("done , see viz/figures/")
