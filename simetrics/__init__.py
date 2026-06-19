"""
SiMetrics , an open benchmark for predictive maintenance on distribution feeders.

Public API:
    from simetrics import build_topology, simulate, params
    top = build_topology("oberrhein")
    run = simulate(top, "reactive", seed=1, years=5)
"""
from . import params
from .params import ENGINE_VERSION
from .network import build_distribution
from .engine import (build_topology, simulate, hotspot, faa, wear_per_day,
                     EV_FAIL_WEAR, EV_FAIL_EXT, EV_MAINT, EV_REPAIR_DONE, EV_TRIP, EV_RECLOSE)

__version__ = ENGINE_VERSION
__all__ = ["params", "ENGINE_VERSION", "__version__", "build_distribution", "build_topology",
           "simulate", "hotspot", "faa", "wear_per_day",
           "EV_FAIL_WEAR", "EV_FAIL_EXT", "EV_MAINT", "EV_REPAIR_DONE", "EV_TRIP", "EV_RECLOSE"]
