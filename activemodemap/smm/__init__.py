"""Sparse mode mapping (SMM) — the operational, map-route calibration workflow.

This package wraps the pieces of ``activemodemap`` that the FMM/SMM paper's
*map route* needs into a small number of stage functions that a notebook (or a
GUI) can call one after another:

    connect  -> preflight -> capture -> analyse -> pick_position -> goto_and_tune -> images

Every stage reads one `SMMConfig`, writes its result as JSON/NPZ into the
campaign folder, and can be re-run from its own checkpoint. The acquisition
stages end by setting the bias to 0 V and withdrawing, whether they finish or
are interrupted. The analysis is pure numpy (Chebyshev low-rank interpolation
along the lever) so everything runs inside the kernel that owns the instrument.

Nothing in here needs Windows or Igor unless an `AsylumInstrument` is built;
`ReplayInstrument` replays an archived checkpoint so the whole workflow can be
rehearsed off the microscope.
"""
from .config import SMMConfig, ProbeConfig, CampaignConfig, CaptureConfig, PreflightConfig, \
    AnalysisConfig, ImagingConfig, LimitsConfig, load_config
from .instruments import ReplayInstrument, make_instrument
from .analysis import (channel_table, fit_delta, apply_delta, emap_along_lever, recommend_positions,
                       analyse_survey, deff_qs_pm_per_V, deff_cr_pm_per_V)
from .live import LiveCapture
from . import stages

__all__ = ["SMMConfig", "ProbeConfig", "CampaignConfig", "CaptureConfig", "PreflightConfig",
           "AnalysisConfig", "ImagingConfig", "LimitsConfig", "load_config",
           "ReplayInstrument", "make_instrument",
           "channel_table", "fit_delta", "apply_delta", "emap_along_lever", "recommend_positions",
           "analyse_survey", "deff_qs_pm_per_V", "deff_cr_pm_per_V",
           "LiveCapture", "stages"]
