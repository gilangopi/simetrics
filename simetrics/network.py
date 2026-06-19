"""
network.py , distribution-grid foundation (conventional / radial, NO smart-grid features).

Loads a standard MV distribution feeder (pandapower) and computes the ONE thing that makes
SAIFI/SAIDI correct at distribution: for every asset (line section / transformer), WHICH
customers are DOWNSTREAM of it , i.e. who loses power if that asset fails (no FLISR / no
reconfiguration; a fault isolates its whole downstream subtree).

Networks:
  - 'cigre'     : CIGRE MV (15 bus, clean radial) , for development
  - 'oberrhein' : mv_oberrhein (179 bus, real MV network) , for scale / realism
All numeric constants live in simetrics/params.py (here: CUST_PER_MW).
"""
import json
import numpy as np
import networkx as nx
import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top

from .params import CUST_PER_MW


def build_distribution(case="cigre", seed=42):
    rng = np.random.RandomState(seed)
    if case in ("cigre", "cigre_mv"):
        net = pn.create_cigre_network_mv(with_der=False); label = "CIGRE MV"
    elif case in ("oberrhein", "mv_oberrhein"):
        net = pn.mv_oberrhein(); label = "mv_oberrhein"
    else:
        raise ValueError(f"unknown case '{case}'")
    try:
        pp.runpp(net)
    except Exception:
        pass

    sources = set(int(b) for b in net.ext_grid.bus.values)          # feeding substation(s)
    Gs = nx.Graph(top.create_nxgraph(net, respect_switches=True))    # operating (radial) topology

    # customers per bus (proportional to connected load)
    bus_cust = {int(b): 0 for b in net.bus.index}
    for _, ld in net.load.iterrows():
        bus_cust[int(ld.bus)] = bus_cust.get(int(ld.bus), 0) + max(1, int(float(ld.p_mw) * CUST_PER_MW))

    # virtual super-source over all feeding substations, then a radial tree
    Gs.add_node("SRC")
    for s in sources:
        if s in Gs:
            Gs.add_edge("SRC", s)
    comp = nx.node_connected_component(Gs, "SRC")
    tree = nx.bfs_tree(Gs.subgraph(comp), "SRC")

    # subtree customer totals (post-order) -> downstream customers of every edge's child
    subtree = {}
    for node in reversed(list(nx.topological_sort(tree))):
        subtree[node] = bus_cust.get(node, 0) + sum(subtree[c] for c in tree.successors(node))

    def downstream(u, v):
        if tree.has_edge(u, v): return int(subtree.get(v, 0))
        if tree.has_edge(v, u): return int(subtree.get(u, 0))
        return 0                                # non-tree edge (loop): no single-point isolation

    assets = []
    for idx, r in net.line.iterrows():
        u, v = int(r["from_bus"]), int(r["to_bus"])
        assets.append(dict(kind="line", idx=int(idx), u=u, v=v, downstream=downstream(u, v)))
    for idx, r in net.trafo.iterrows():
        u, v = int(r["hv_bus"]), int(r["lv_bus"])
        assets.append(dict(kind="trafo", idx=int(idx), u=u, v=v, downstream=downstream(u, v)))

    # positions: parse GeoJSON in net.bus['geo'] (oberrhein = real lat/lon; CIGRE = schematic)
    pos = {}
    if "geo" in net.bus.columns:
        for b in net.bus.index:
            g = net.bus.at[b, "geo"]
            if isinstance(g, str) and g:
                try:
                    c = json.loads(g)["coordinates"]
                    pos[int(b)] = (float(c[0]), float(c[1]))
                except Exception:
                    pass
    if len(pos) < len(net.bus):
        sub = Gs.copy(); sub.remove_node("SRC") if "SRC" in sub else None
        lay = nx.kamada_kawai_layout(sub)
        for k, v in lay.items():
            pos.setdefault(int(k), (float(v[0]), float(v[1])))

    bus_downstream = {int(b): int(subtree.get(b, 0)) for b in pos}   # customers cut if bus de-energized

    return dict(net=net, label=label, graph=Gs, sources=sources, bus_customers=bus_cust,
                assets=assets, positions=pos, bus_downstream=bus_downstream,
                total_customers=int(sum(bus_cust.values())),
                n_bus=len(net.bus), n_line=len(net.line), n_trafo=len(net.trafo))


if __name__ == "__main__":
    for case in ["cigre", "oberrhein"]:
        d = build_distribution(case)
        downs = sorted(d["assets"], key=lambda a: -a["downstream"])
        print(f"\n=== {d['label']} ===")
        print(f"  {d['n_bus']} bus | {d['n_line']} line | {d['n_trafo']} trafo | "
              f"~{d['total_customers']:,} customers | sources at bus {sorted(d['sources'])}")
        print("  TOP-5 assets by downstream customers (most critical for SAIFI/SAIDI):")
        for a in downs[:5]:
            print(f"    {a['kind']:5s} {a['idx']:>3} (bus {a['u']}-{a['v']}): {a['downstream']:>7,} downstream")
