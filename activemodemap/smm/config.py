"""One configuration object for the whole SMM calibration campaign.

Everything a user might want to change lives here, with units in the field name.
The notebook builds one `SMMConfig`, every stage reads it, and `cfg.save()` writes
it next to the data so a campaign is reproducible. Nothing downstream hard-codes
a number that is in this file.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ProbeConfig:
    name: str = "SCM-PIT"
    k_N_per_m: float = 1.70                 # thermal spring constant (for the deflection shown next to a load)
    f0_free_Hz: float = 63.8e3              # free resonance (bookkeeping only)
    length_um: float = 232.0                # stage x of the FREE END when the laser starts there (x = 0 is the clamp)
    clamp_offset_um: float = 0.0            # stage x of the clamp, if known (x_clamp = x - offset); 0 = unknown
    hw_sign_toward_free_end: int = +1       # DoLDMove(+x) moves toward the tip on this Cypher (+1) or away (-1)
    # prior frequency windows for the contact resonances (refined by pre-flight)
    cr_bands_Hz: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "cr1": (230e3, 340e3), "cr2": (780e3, 1000e3), "cr3": (1550e3, 1850e3)})
    qs_band_Hz: Tuple[float, float] = (15e3, 45e3)     # quasi-static reference band
    amp_divisor: float = 32.0               # lock-in amplitude factor G: metres = V * InvOLS / G


@dataclass
class CampaignConfig:
    data_dir: str = r"D:\User Data\Liam\ActiveModeMap\SMM_campaign"   # everything this campaign writes goes here
    base_filename: str = "SMM"              # Igor base name for tunes / images
    tag: str = ""                           # free text written into the manifest
    virtual: bool = False                   # True: no hardware; replay `replay_checkpoint` or simulate
    replay_checkpoint: Optional[str] = None # an archived two-domain checkpoint to replay (virtual=True)
    replay_invols_json: Optional[str] = None  # optional {"x_um": invols} or preflight_result.json for the replay
    replay_delay_s: float = 0.0             # pause per replayed condition, to watch the live plot fill


@dataclass
class PreflightConfig:
    run: bool = True                        # False: use x_span_um below instead of walking the lever
    step_um: float = 10.0                   # walk step toward the base
    x_floor_um: float = 30.0                # never go closer to the clamp than this
    min_snr_dB: float = 8.0                 # stop when the stronger of CR1/CR2 falls below this
    invols_bounds_m_per_V: Tuple[float, float] = (1e-8, 5e-5)
    invols_jump: float = 3.0                # stop on a >3x jump in InvOLS between steps
    x_span_um: Optional[Tuple[float, float]] = None   # (x_lo, x_hi) used when run=False or as the result


@dataclass
class CaptureConfig:
    load_nN: float = 500.0
    drive_V: float = 1.0                    # V_ac for every tune (also used in the pm/V conversion)
    bias_V: List[float] = field(default_factory=lambda: [-9, -6, -3, 0, 3, 6, 9])
    bias_order: str = "interleaved"         # "interleaved" (recommended) or "given"
    spots: Tuple[int, int] = (1, 2)         # marked spots on the two antiparallel domains (domain 1, domain 2)
    n_positions: int = 8                    # equispaced over the reachable span
    positions_um: Optional[List[float]] = None   # explicit stage positions instead of n_positions
    order: str = "coarse_to_fine"           # "coarse_to_fine" (ends, middle, quarters...), "free_end_first", "base_first", "as_given"
    ref_bias_V: float = 0.0                 # the reference condition (position bookkeeping, live map)
    stop_when_stable: bool = False          # end early once E(x) stops changing (see below)
    stable_rel: float = 0.05                # ... median |dE/E| over the lever below this ...
    stable_over: int = 2                    # ... for this many consecutive positions
    min_positions: int = 5                  # never stop before this many positions
    checkpoint_name: str = "two_domain_survey_checkpoint.npz"


@dataclass
class AnalysisConfig:
    cr_band_frac: Tuple[float, float] = (0.80, 1.25)  # CR1 analysis band as a fraction of the measured CR1
    track_peaks: bool = True                # read every bias condition at its own CR1 peak (paper convention)
    fit_delta: bool = True                  # domain-potential correction P = P_raw - delta * b (fitted in QS)
    delta_exclude_tip_um: float = 0.0       # ignore positions within this distance of the free end in the delta fit
    rank: int = 4                           # Chebyshev spatial rank for the along-lever interpolation
    grid_step_um: float = 1.0               # output grid along the lever


@dataclass
class ImagingConfig:
    # ---- quasi-static reference frame at the electrostatic blind spot (no correction needed there)
    qs_x_um: Optional[float] = None         # None = the blind spot found by the analysis (b_QS/P_QS smallest)
    qs_bias_plan: List[str] = field(default_factory=lambda: ["0V"])          # "0V", "Vcpd", "Vhalf", "0V_repeat"
    f_qs_Hz: float = 20e3                   # quasi-static drive
    image_base: str = "QImg"                # Igor base name for single-frequency frames
    frame: Optional[Dict[str, float]] = None   # ScanSize, XOffset, YOffset, ScanAngle, ScanPoints, ScanLines, ScanRate; None = MasterPanel as is
    scan_timeout_s: float = 1200.0
    # ---- band-excitation frames at interior stops with a known enhancement
    be_positions_um: Optional[List[float]] = None   # stage positions you choose; None = recommended from E_map(x)
    n_be_positions: int = 2                 # how many to recommend
    be_min_spacing_um: float = 20.0         # recommended stops at least this far apart
    tol_pct: float = 5.0                    # E may change by at most this over +/- drift_um ...
    drift_um: float = 2.0                   # ... the laser drift budget
    min_E_frac: float = 0.5                 # and E >= this fraction of the max on the validated span
    edge_exclude_um: Optional[float] = None # keep this far from the ends of the measured span (None = half the spacing)
    loo_tol: float = 0.15                   # only between positions whose leave-one-out E agrees to this
    be_bias_plan: List[str] = field(default_factory=lambda: ["0V", "Vcpd", "0V_repeat"])
    be_px: int = 128                        # BE frame pixels per side
    be_repeats: int = 1
    be_width_x_fwhm: float = 10.0           # BE band width = this x the CR1 FWHM from the tune at the stop ...
    be_width_Hz: Tuple[float, float] = (20e3, 80e3)   # ... clipped to this range
    be_acquire: str = "smm_be_adapter:be_frame"       # "module:function" on the instrument PC (see be_adapter_template.py)
    cr1_search_frac: Tuple[float, float] = (0.85, 1.15)   # where to look for CR1 in the tune at a stop
    tune_after_each_stop: bool = True       # re-tune after the frames (drift bracket)


@dataclass
class LimitsConfig:
    max_abs_bias_V: float = 10.0
    max_load_nN: float = 1500.0
    max_drive_V: float = 3.0


@dataclass
class SMMConfig:
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    campaign: CampaignConfig = field(default_factory=CampaignConfig)
    preflight: PreflightConfig = field(default_factory=PreflightConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    imaging: ImagingConfig = field(default_factory=ImagingConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)

    # ---- paths -------------------------------------------------------------------------
    def path(self, *parts) -> str:
        return os.path.join(self.campaign.data_dir, *parts)

    @property
    def preflight_json(self): return self.path("01_preflight", "preflight_result.json")
    @property
    def preflight_npz(self): return self.path("01_preflight", "preflight_walk_checkpoint.npz")
    @property
    def survey_npz(self): return self.path("10_survey", self.capture.checkpoint_name)
    @property
    def analysis_json(self): return self.path("20_analysis", "emap_analysis.json")
    @property
    def analysis_npz(self): return self.path("20_analysis", "emap_maps.npz")
    @property
    def plan_json(self): return self.path("30_plan", "imaging_plan.json")
    @property
    def images_json(self): return self.path("40_images", "images.json")
    @property
    def manifest_json(self): return self.path("manifest.json")
    @property
    def config_json(self): return self.path("smm_config.json")

    def make_dirs(self):
        for d in ("01_preflight", "10_survey", "20_analysis", "30_plan", "40_images", "logs"):
            os.makedirs(self.path(d), exist_ok=True)

    # ---- validation --------------------------------------------------------------------
    def check(self):
        """Refuse obviously unsafe or inconsistent settings before anything moves."""
        c, L = self.capture, self.limits
        bad = []
        if max(abs(float(v)) for v in c.bias_V) > L.max_abs_bias_V:
            bad.append(f"a survey bias exceeds limits.max_abs_bias_V = {L.max_abs_bias_V} V")
        if c.load_nN > L.max_load_nN:
            bad.append(f"capture.load_nN {c.load_nN} > limits.max_load_nN {L.max_load_nN}")
        if c.drive_V > L.max_drive_V:
            bad.append(f"capture.drive_V {c.drive_V} > limits.max_drive_V {L.max_drive_V}")
        if len(set(float(v) for v in c.bias_V)) < 3:
            bad.append("need at least 3 distinct biases for the Z = a + bV fit")
        if c.spots[0] == c.spots[1]:
            bad.append("capture.spots must name two different marked spots")
        if c.positions_um is None and c.n_positions < self.analysis.rank + 1:
            bad.append(f"n_positions {c.n_positions} must exceed analysis.rank {self.analysis.rank}")
        if c.bias_order not in ("interleaved", "given"):
            bad.append("capture.bias_order must be 'interleaved' or 'given'")
        if not self.preflight.run and self.preflight.x_span_um is None:
            bad.append("preflight.run is False but preflight.x_span_um is not set")
        if self.campaign.virtual and self.campaign.replay_checkpoint is None:
            bad.append("virtual=True needs campaign.replay_checkpoint (an archived two-domain survey)")
        if bad:
            raise ValueError("config problems:\n  - " + "\n  - ".join(bad))
        return True

    # ---- (de)serialisation -------------------------------------------------------------
    def to_dict(self):
        return asdict(self)

    def save(self, path=None):
        path = path or self.config_json
        os.makedirs(os.path.dirname(path), exist_ok=True)
        d = self.to_dict(); d["_saved"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1, default=_json_default)
        return path

    @classmethod
    def from_dict(cls, d):
        d = dict(d); d.pop("_saved", None)
        kw = {}
        for name, sub in (("probe", ProbeConfig), ("campaign", CampaignConfig), ("preflight", PreflightConfig),
                          ("capture", CaptureConfig), ("analysis", AnalysisConfig), ("imaging", ImagingConfig),
                          ("limits", LimitsConfig)):
            kw[name] = _build(sub, d.get(name, {}))
        return cls(**kw)

    def describe(self) -> str:
        """Plain-English summary for the notebook."""
        c, p, a = self.capture, self.probe, self.analysis
        n_cond = len(c.bias_V) * 2
        npos = len(c.positions_um) if c.positions_um else c.n_positions
        defl_nm = c.load_nN * 1e-9 / p.k_N_per_m * 1e9
        lines = [
            f"Probe {p.name}: k = {p.k_N_per_m} N/m, lever free end at stage x = {p.length_um} um",
            f"Campaign folder: {self.campaign.data_dir}" + ("   [VIRTUAL / replay]" if self.campaign.virtual else ""),
            f"Capture: {npos} laser positions x {n_cond} conditions "
            f"({len(c.bias_V)} biases {sorted(c.bias_V)} V on spots {c.spots}), "
            f"load {c.load_nN:.0f} nN (deflection {defl_nm:.0f} nm), V_ac {c.drive_V} V",
            f"Estimated time: ~{npos * (1.0 + 0.3 * n_cond):.0f} min "
            f"(about 1 min per position + ~20 s per condition)",
            f"Analysis: QS band {p.qs_band_Hz[0] / 1e3:.0f}-{p.qs_band_Hz[1] / 1e3:.0f} kHz, "
            f"CR1 band {a.cr_band_frac[0]:.2f}-{a.cr_band_frac[1]:.2f} x f_CR1, rank {a.rank}, "
            f"{'peak-tracked' if a.track_peaks else 'fixed-bin'} CR1, "
            f"{'with' if a.fit_delta else 'without'} domain-potential correction",
            f"Imaging: QS frame at the electrostatic blind spot"
            + (f" (x = {self.imaging.qs_x_um} um)" if self.imaging.qs_x_um else " (from the analysis)")
            + f"; BE frames at " + (f"{self.imaging.be_positions_um} um" if self.imaging.be_positions_um else
                                   f"{self.imaging.n_be_positions} recommended stops (E flat to {self.imaging.tol_pct:.0f} % "
                                   f"over +/-{self.imaging.drift_um:.0f} um, >= {100 * self.imaging.min_E_frac:.0f} % of max E)"),
        ]
        return "\n".join(lines)


def _json_default(o):
    if is_dataclass(o):
        return asdict(o)
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
    except Exception:
        pass
    return str(o)


def _build(cls, d):
    fields = {f.name for f in cls.__dataclass_fields__.values()}
    kw = {k: v for k, v in d.items() if k in fields}
    obj = cls(**kw)
    # JSON turns tuples into lists; restore the ones we compare as tuples
    for f in cls.__dataclass_fields__.values():
        v = getattr(obj, f.name)
        if isinstance(v, list) and f.name in ("qs_band_Hz", "cr_band_frac", "cr1_search_frac",
                                               "invols_bounds_m_per_V", "spots", "x_span_um"):
            setattr(obj, f.name, tuple(v))
        if f.name == "cr_bands_Hz" and isinstance(v, dict):
            setattr(obj, f.name, {k: tuple(b) for k, b in v.items()})
    return obj


def load_config(path) -> SMMConfig:
    with open(path, "r", encoding="utf-8") as f:
        return SMMConfig.from_dict(json.load(f))
