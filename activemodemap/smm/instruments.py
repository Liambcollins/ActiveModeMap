"""Instrument construction for the SMM workflow, plus a replay instrument.

`make_instrument(cfg)` returns either an `AsylumInstrument` (Windows + Igor) or a
`ReplayInstrument` that serves spectra from an archived two-domain checkpoint.
Both expose the same calls the stages use, so the notebook is identical on and
off the microscope.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from ..series import load_checkpoint


class _ReplayAutomation:
    """Just enough of `AFMLaserSweepAutomation` for the stages to call."""

    def __init__(self, file_loc, base_filename):
        self.file_loc, self.base_filename = file_loc, base_filename
        self.igor = None
        self.invols_bounds = (1e-8, 5e-5)

    def withdraw(self): pass
    def set_folder(self): os.makedirs(self.file_loc, exist_ok=True)
    def get_gmv(self): return dict(SpringConstant=None, InvOLS=None, DeflectionSetpointVolts=None,
                                  DriveAmplitude=None, DriveFrequency=None)


class ReplayInstrument:
    """Replays an archived two-domain survey as if it were being measured.

    * A request at stage position x returns the archived spectrum at the NEAREST
      archived position (the distance is reported in the record).
    * A requested (bias, spot) that was archived is returned as is; any other bias
      is synthesised from the archived Z = a + bV fit at that spot, so a different
      bias list still produces a consistent survey.
    * `noise_rel` adds complex Gaussian noise relative to the band RMS, so repeated
      replays are not bit-identical.
    * `invols_by_x` (dict "x": m/V, or a preflight_result.json) gives InvOLS at
      each position; otherwise a nominal 1e-5 m/V is used and flagged.
    """

    def __init__(self, checkpoint_path, file_loc, base_filename="REPLAY", invols_by_x=None,
                 span_um=None, x_limits_um=None, delay_s=0.0, noise_rel=0.0, seed=0):
        ck = load_checkpoint(checkpoint_path)
        self.src = checkpoint_path
        self.x_src = np.asarray(ck["x_um"], float)
        self.freq = np.asarray(ck["freq_Hz"], float)
        self.Z_src = np.asarray(ck["Z"])
        self.conds_src = ck["conditions"]
        self.a = _ReplayAutomation(file_loc, base_filename)
        self.records = []
        self.delay_s, self.noise_rel = float(delay_s), float(noise_rel)
        self.rng = np.random.default_rng(seed)
        self.current_x = float(self.x_src.max())
        self.current_spot = None
        self.load_nN = float(self.conds_src[0].get("load_nN", 0.0))
        self.dc_bias_V = 0.0
        self.span_um = float(span_um if span_um is not None else self.x_src.max())
        self.x_limits_um = tuple(x_limits_um) if x_limits_um else (float(self.x_src.min()), float(self.x_src.max()))
        self._invols = None
        self._scan_index = 0
        self.engaged = False
        self._invols_tab = _load_invols(invols_by_x)
        sp = np.diff(np.sort(self.x_src))
        self.max_dx_um = float(0.6 * np.median(sp)) if sp.size else 10.0
        # per-spot complex bias fits for synthesising unarchived biases
        self._fits = {}
        for spot in sorted({c.get("spot") for c in self.conds_src}):
            idx = [i for i, c in enumerate(self.conds_src) if c.get("spot") == spot]
            V = np.array([self.conds_src[i]["bias_V"] for i in idx], float)
            A = np.stack([np.ones_like(V), V], 1)
            Z = self.Z_src[idx].reshape(len(idx), -1)
            coef = np.linalg.lstsq(A, Z, rcond=None)[0]
            self._fits[spot] = (coef[0].reshape(self.Z_src.shape[1:]), coef[1].reshape(self.Z_src.shape[1:]))

    # ---- bookkeeping calls used by the stages -----------------------------------------
    def set_dc_bias(self, volts, settle_s=0.0): self.dc_bias_V = float(volts)
    def set_load(self, load_nN, reengage=True, settle_s=0.0):
        self.load_nN = float(load_nN); self.engaged = True
        return 0.1 * load_nN / 500.0          # a fake setpoint in volts
    def goto_spot(self, spot, withdraw_first=True): self.current_spot = int(spot); self.engaged = False
    def set_position(self, x_um, confirm=False): self.current_x = float(x_um)
    def preview_moves(self, xs): print("  (replay) moves:", [f"{x:.1f}" for x in xs])
    def _check_travel(self, x):
        lo, hi = self.x_limits_um
        if not (lo - 1e-6 <= x <= hi + 1e-6):
            raise RuntimeError(f"x = {x:.1f} um outside travel limits {self.x_limits_um}")

    def _nearest(self, x):
        j = int(np.argmin(np.abs(self.x_src - x)))
        dx = float(self.x_src[j] - x)
        if abs(dx) > self.max_dx_um:
            # beyond the archived span the real instrument would lose the laser; mimic that
            raise RuntimeError(f"replay has no spectrum within {self.max_dx_um:.0f} um of x = {x:.1f} um "
                               f"(archived {self.x_src.min():.1f}-{self.x_src.max():.1f} um)")
        return j, dx

    def _goto_and_prepare(self, x_um, capture_image=False):
        self._check_travel(x_um)
        self.current_x = float(x_um)
        j, dx = self._nearest(x_um)
        self._invols = self._invols_at(x_um)
        self.engaged = False
        return f"X{x_um:05.1f}"

    def _invols_at(self, x_um):
        """InvOLS from the table, log-interpolated in x (it changes ~10x along the lever); 1e-5 if no table."""
        if not self._invols_tab:
            return 1e-5
        j, _ = self._nearest(x_um)                      # same archived stop as the spectrum, when tabulated
        key = round(float(self.x_src[j]), 1)
        if key in self._invols_tab:
            return float(self._invols_tab[key])
        xs = np.array(sorted(self._invols_tab)); v = np.array([self._invols_tab[k] for k in xs])
        return float(np.exp(np.interp(x_um, xs, np.log(v))))

    def _spectrum(self, x_um, bias_V, spot):
        j, dx = self._nearest(x_um)
        match = [i for i, c in enumerate(self.conds_src)
                 if c.get("spot") == spot and abs(c["bias_V"] - bias_V) < 1e-9]
        if match:
            Z = self.Z_src[match[0], j].copy()
        else:
            if spot not in self._fits:
                raise RuntimeError(f"replay has no spot {spot} (archived: {sorted(self._fits)})")
            a, b = self._fits[spot]
            Z = a[j] + b[j] * bias_V
        if self.noise_rel > 0:
            s = self.noise_rel * np.sqrt(np.mean(np.abs(Z) ** 2))
            Z = Z + s * (self.rng.standard_normal(Z.shape) + 1j * self.rng.standard_normal(Z.shape)) / np.sqrt(2)
        rec = dict(position_x_um=self.current_x, replay_source_x_um=float(self.x_src[j]), replay_dx_um=dx,
                   dc_bias_V=bias_V, load_nN=self.load_nN, sample_spot=spot, invols_m_per_V=self._invols,
                   spring_N_per_m=None, resonance_freq_Hz=float(self.freq[np.argmax(np.abs(Z))]),
                   q_factor=np.nan, n_tune_points=int(self.freq.size), tune_f_lo_Hz=float(self.freq[0]),
                   tune_f_hi_Hz=float(self.freq[-1]), n_fit_points=int(self.freq.size), tune_completed=True,
                   tune_file=None, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), setpoint_V=0.1)
        self.records.append(rec)
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        return self.freq, Z, rec

    def _tune_and_read(self, label, setpoint_V):
        return self._spectrum(self.current_x, self.dc_bias_V, self.current_spot)

    # ---- the two calls the acquisition loop uses ---------------------------------------
    def measure_at(self, x_um):
        self._goto_and_prepare(x_um)
        self.set_load(self.load_nN)
        f, Z, rec = self._spectrum(x_um, self.dc_bias_V, self.current_spot)
        return f, Z, rec

    def measure_conditions_at(self, x_um, conditions, reengage_on_load_change=True, verbose=True):
        self._goto_and_prepare(x_um)
        out = {}
        for i, cond in enumerate(conditions):
            spot = getattr(cond, "spot", None)
            if spot is not None and spot != self.current_spot:
                self.goto_spot(spot)
            self.set_dc_bias(cond.bias_V); self.set_load(cond.load_nN)
            f, Z, rec = self._spectrum(x_um, cond.bias_V, spot)
            out[i] = (f, Z, rec)
            if verbose:
                print(f"      cond {i}: {cond.bias_V:+.2f} V, {cond.load_nN:.0f} nN, spot {spot} -> "
                      f"{f.size} pts (replayed from x = {rec['replay_source_x_um']:.1f} um)")
        self.engaged = False
        return out

    def tune_eigenmode(self, position_label="", scan_index=0, save_tune_data=True):
        f, Z, rec = self._spectrum(self.current_x, self.dc_bias_V, self.current_spot)
        return dict(tune_data=dict(frequency=f, Z=Z), resonance_freq=rec["resonance_freq_Hz"],
                    q_factor=np.nan, tune_completed=True, tune_file=None, freq=f, Z=Z)

    def close(self): pass


def _load_invols(src):
    """Accept None, a dict {x: invols}, or a path to a JSON with 'invols_by_x' (preflight) or x keys."""
    if src is None:
        return {}
    if isinstance(src, dict):
        d = src
    else:
        with open(src, "r", encoding="utf-8") as f:
            d = json.load(f)
        if "invols_by_x" in d:
            d = d["invols_by_x"]
        elif "rows" in d:        # s3_transfer-style rows
            d = {r["x_um"]: r["invols"] for r in d["rows"] if r.get("invols")}
    return {round(float(k), 1): float(v) for k, v in d.items() if v is not None}


def make_instrument(cfg, existing=None, verbose=True):
    """Build (or adopt) the instrument described by `cfg`.

    * `existing` — an instrument already in the notebook namespace is adopted,
      never rebuilt (laser position bookkeeping is software-tracked).
    * `cfg.campaign.virtual` — a `ReplayInstrument` on `cfg.campaign.replay_checkpoint`.
    * otherwise — connect to Igor and build an `AsylumInstrument` with the probe
      parameters from `cfg` (Windows only).
    """
    cfg.make_dirs()
    if existing is not None:
        inst = existing
        inst.a.file_loc = cfg.campaign.data_dir
        inst.a.base_filename = cfg.campaign.base_filename
        inst.a.set_folder()
        if verbose:
            print(f"adopted existing instrument: x = {inst.current_x} um, spot = {inst.current_spot}, "
                  f"load = {inst.load_nN} nN, bias = {inst.dc_bias_V:+.2f} V, limits = {inst.x_limits_um}")
        return inst

    cam, p = cfg.campaign, cfg.probe
    if cam.virtual:
        inst = ReplayInstrument(cam.replay_checkpoint, cam.data_dir, cam.base_filename,
                                invols_by_x=cam.replay_invols_json, span_um=p.length_um,
                                x_limits_um=(0.0, p.length_um), delay_s=cam.replay_delay_s)
        if verbose:
            print(f"VIRTUAL instrument replaying {os.path.basename(cam.replay_checkpoint)}: "
                  f"{inst.x_src.size} archived positions {inst.x_src.min():.1f}-{inst.x_src.max():.1f} um, "
                  f"{len(inst.conds_src)} conditions")
        return inst

    import win32com.client                                    # Windows + Igor only
    from ..asylum import AFMLaserSweepAutomation, AsylumInstrument
    igor = win32com.client.Dispatch("IgorPro.Application")
    automation = AFMLaserSweepAutomation(igor, cam.data_dir, cam.base_filename,
                                         log_filename=os.path.join(cam.data_dir, "logs", f"{cam.base_filename}_log"))
    automation.eigenmode_center_freq = float(np.mean(p.cr_bands_Hz["cr1"]))
    automation.resonance_band_Hz = None
    automation.autowedge_pause = 15.0
    automation.wait_for_tune_complete = True
    automation.tune_timeout_s = 600.0
    automation.verbose_tune = True
    automation.spot_timeout_s = 90.0
    automation.spot_stable_tol_V = 2e-3
    automation.invols_bounds = tuple(cfg.preflight.invols_bounds_m_per_V)
    automation.set_folder()
    L = float(p.length_um)
    inst = AsylumInstrument(automation, load_nN=cfg.capture.load_nN, dc_bias_V=0.0, recalibrate_each=True,
                            analysis_band_Hz=None, start_at="free_end", span_um=L, x_limits_um=(0.0, L),
                            hw_sign_toward_free_end=p.hw_sign_toward_free_end, apply_bias=False)
    inst.set_position(L, confirm=True)          # the laser IS at the free end when this runs
    if verbose:
        print(f"connected to Igor; instrument built with the laser at the free end (x = {L:.0f} um)")
        inst.preview_moves([L, L - 10.0])       # the second entry must read 'toward base'
        gmv = inst.a.get_gmv()
        k = gmv.get("SpringConstant")
        print(f"MasterPanel: k = {k} N/m (config {p.k_N_per_m}), InvOLS = {gmv.get('InvOLS')}, "
              f"setpoint = {gmv.get('DeflectionSetpointVolts')} V")
        if k and abs(k - p.k_N_per_m) / p.k_N_per_m > 0.10:
            print("  *** panel spring constant differs from the config by >10 % -- check before trusting loads ***")
    return inst
