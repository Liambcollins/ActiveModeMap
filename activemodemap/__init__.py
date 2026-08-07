"""ActiveModeMap — rapid mode-shape mapping in dynamic AFM by active learning.

Core (cross-platform):
  forward_model, virtual_afm, inference, acquisition, baselines, hybrid, loop
      physics-informed active-learning pipeline (simulation + benchmarks)
  lowrank
      model-light low-rank mode-shape reconstruction (used on real data)
  online
      Instrument abstraction + VirtualInstrument + live plotting

Hardware (Windows / Igor only, imported lazily):
  asylum
      AFMLaserSweepAutomation + AsylumInstrument.  Access via
      `from activemodemap.asylum import AsylumInstrument` on the instrument PC.
"""

from .forward_model import EBForwardModel, ProbeGeometry
from .virtual_afm import VirtualAFM
from .inference import PhysicsPosterior
from .loop import run_loop
from .lowrank import (LowRankModeMap, reconstruct_map, d_optimal_order,
                      chebyshev_basis, dns_from_map, resonance_index)
from .online import Instrument, VirtualInstrument, plot_state

__all__ = [
    "EBForwardModel", "ProbeGeometry", "VirtualAFM", "PhysicsPosterior",
    "run_loop", "LowRankModeMap", "reconstruct_map", "d_optimal_order",
    "chebyshev_basis", "dns_from_map", "resonance_index",
    "Instrument", "VirtualInstrument", "plot_state",
]
