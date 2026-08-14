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
                      chebyshev_basis, dns_from_map, resonance_index,
                      dns_branch, band_mask, spatial_null,
                      classify_null, classify_null_from_map)
from .series import (Condition, make_conditions, run_series, reconstruct_series,
                     separate_channels, estimate_v_cpd, channel_spots, separate_domains,
                     save_checkpoint, load_checkpoint)
from .online import (Instrument, VirtualInstrument, plot_state,
                     LiveModeMapPlot, dense_reference_sweep, compare_to_dense, dense_grid,
                     PositionScale, fit_position_scale, fit_position_scale_optical,
                     peak_in_band)

__all__ = [
    "EBForwardModel", "ProbeGeometry", "VirtualAFM", "PhysicsPosterior",
    "run_loop", "LowRankModeMap", "reconstruct_map", "d_optimal_order",
    "chebyshev_basis", "dns_from_map", "resonance_index",
    "dns_branch", "band_mask", "spatial_null",
    "classify_null", "classify_null_from_map",
    "Instrument", "VirtualInstrument", "plot_state",
    "LiveModeMapPlot", "dense_reference_sweep", "compare_to_dense", "dense_grid",
    "PositionScale", "fit_position_scale", "fit_position_scale_optical",
    "peak_in_band",
    "Condition", "make_conditions", "run_series", "reconstruct_series",
    "separate_channels", "estimate_v_cpd", "channel_spots", "separate_domains",
    "save_checkpoint", "load_checkpoint",
]
