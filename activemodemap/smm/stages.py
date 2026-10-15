"""The SMM calibration campaign as stage functions — one notebook cell each.

    inst   = stages.connect(cfg)                      # or adopt an existing `inst`
    pf     = stages.preflight(inst, cfg)              # reachable span, CR1, InvOLS(x)
    series = stages.capture(inst, cfg)                # two-domain sparse survey, live plot
    res    = stages.analyse(cfg)                      # delta, V_cpd, E_map(x), blind spot, d_eff
    plan   = stages.plan_imaging(cfg)                 # QS stop at the blind spot + BE stops with E
    frames = stages.run_imaging(inst, cfg)            # per stop: laser, InvOLS, engage, tune, frames -> d_eff
    stages.closeout(inst, cfg)                        # bias 0 V, withdraw, manifest

Every stage: reads `cfg`, accepts keyword overrides for a one-off change
(`stages.capture(inst, cfg, n_positions=6)`), writes its result into the campaign
folder, logs to logs/<stage>.txt, and — for the stages that move the instrument —
ends with the bias at 0 V and the tip withdrawn, finished or not.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import replace

import numpy as np

from ..series import make_conditions, run_series, load_checkpoint
from .config import SMMConfig, _json_default
from .instruments import make_instrument
from .analysis import analyse_survey, summary_text, peak, find_cr1
from .live import LiveCapture
from . import imaging


# ----------------------------------------------------------------------------------- #
#  logging / safety scaffolding                                                         #
# ----------------------------------------------------------------------------------- #
def say(msg=""):
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


class _Tee:
    def __init__(self, path, out=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", buffering=1, encoding="utf-8"); self.out = out or sys.stdout
    def write(self, s): self.out.write(s); self.f.write(s)
    def flush(self): self.out.flush(); self.f.flush()
    def close(self):
        try: self.f.close()
        except Exception: pass


def finish(inst, tag=""):
    """The standing rule: bias to 0 V, withdraw. Never inst.close()."""
    if inst is None:
        return
    try: inst.set_dc_bias(0.0)
    except Exception as e: print("  bias reset:", e)
    try: inst.a.withdraw()
    except Exception as e: print("  withdraw:", e)
    say(f"=== {tag} finished (tip withdrawn, bias 0 V) ===")


class Stage:
    """with Stage('CAPTURE', cfg, inst): ...  -> tee to logs/, always finish() on exit when withdraw=True."""

    def __init__(self, tag, cfg, inst=None, withdraw=True):
        self.tag, self.cfg, self.inst, self.withdraw = tag, cfg, inst, withdraw
        self.log = cfg.path("logs", f"{tag.lower().replace(' ', '_')}.txt")

    def __enter__(self):
        self.tee = _Tee(self.log); self.old = sys.stdout; sys.stdout = self.tee; self.t0 = time.time()
        say("=" * 78); say(f"{self.tag}   ({self.cfg.campaign.data_dir})"); say("=" * 78)
        return self

    def __exit__(self, et, ev, tb):
        if et is KeyboardInterrupt:
            say("*** interrupted by the user")
        elif et is not None:
            say(f"*** FAILED: {et.__name__}: {ev}"); traceback.print_tb(tb)
        mins = (time.time() - self.t0) / 60
        if self.withdraw:
            finish(self.inst, f"{self.tag} ({mins:.1f} min)")
        else:
            say(f"=== {self.tag} done ({mins:.1f} min) ===")
        sys.stdout = self.old; self.tee.close()
        return et is KeyboardInterrupt            # swallow Ctrl-C after the clean-up; re-raise real errors


def _with(cfg, section, **ovr):
    """A copy of cfg with keyword overrides applied to one section."""
    if not ovr:
        return cfg
    sec = getattr(cfg, section)
    bad = [k for k in ovr if not hasattr(sec, k)]
    if bad:
        raise TypeError(f"{section} has no setting(s) {bad}; valid: {list(sec.__dataclass_fields__)}")
    return replace(cfg, **{section: replace(sec, **ovr)})


def _dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    return path


def _load(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------------- #
#  0  connect                                                                           #
# ----------------------------------------------------------------------------------- #
def connect(cfg: SMMConfig, existing=None):
    cfg.check(); cfg.make_dirs(); cfg.save()
    print(cfg.describe()); print()
    inst = make_instrument(cfg, existing=existing)
    span = cfg.preflight.x_span_um
    pf = _load(cfg.preflight_json)
    if pf and pf.get("x_lo_um") is not None:
        inst.x_limits_um = (float(pf["x_lo_um"]), float(pf["x_hi_um"]))
        print(f"travel limits from the saved pre-flight: {inst.x_limits_um}")
    elif span:
        inst.x_limits_um = (float(span[0]), float(span[1]))
        print(f"travel limits from config: {inst.x_limits_um}")
    return inst


# ----------------------------------------------------------------------------------- #
#  1  pre-flight                                                                        #
# ----------------------------------------------------------------------------------- #
def _peaks_in_bands(freq, amp, bands):
    good = np.isfinite(freq) & np.isfinite(amp) & (amp > 0)
    floor = np.quantile(amp[good & (freq > 50e3)], 0.2) if (good & (freq > 50e3)).any() else np.quantile(amp[good], 0.2)
    out = {}
    for name, (lo, hi) in bands.items():
        m = good & (freq >= lo) & (freq <= hi)
        if not m.any():
            out[name] = dict(f_Hz=None, snr_dB=-np.inf, fwhm_Hz=None); continue
        pk = peak(freq, amp.astype(complex), (lo, hi))
        out[name] = dict(f_Hz=pk["f_Hz"], snr_dB=float(20 * np.log10(pk["amp"] / floor)), fwhm_Hz=pk["fwhm_Hz"], Q=pk["Q"])
    return out


def preflight(inst, cfg: SMMConfig, spot=None, **ovr):
    """Walk the laser from the free end toward the base until detection fails.

    Writes preflight_result.json (span, CR1-3, InvOLS(x)) and the walk spectra; sets
    `inst.x_limits_um`. With `cfg.preflight.run = False` it records the configured span
    and a free-end tune only.
    """
    cfg = _with(cfg, "preflight", **ovr); P, c, pr = cfg.preflight, cfg.capture, cfg.probe
    spot = c.spots[0] if spot is None else spot
    L = float(inst.span_um or pr.length_um)
    result = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), probe=pr.name, k_N_per_m=pr.k_N_per_m, load_nN=c.load_nN,
                  spot=spot, probe_L_um=L, x_hi_um=L, x_lo_um=None, step_um=P.step_um, stop_reason=None,
                  cr1_Hz=None, cr2_Hz=None, cr3_Hz=None, cr1_fwhm_Hz=None, invols_by_x={}, cr_by_x={})
    xs, Zs, freq_ref = [], [], None

    def measure(x):
        inst._goto_and_prepare(x, capture_image=True)
        sp = inst.set_load(c.load_nN, reengage=True)
        freq, Z, rec = inst._tune_and_read(f"PF_X{x:05.1f}", sp)
        inst.a.withdraw(); time.sleep(0.5)
        return freq, Z, rec, _peaks_in_bands(freq, np.abs(Z), pr.cr_bands_Hz)

    with Stage("PRE-FLIGHT", cfg, inst):
        say(f"load {c.load_nN:.0f} nN (deflection {c.load_nN * 1e-9 / pr.k_N_per_m * 1e9:.0f} nm), spot {spot}, "
            f"lever free end at x = {L:.0f} um")
        inst.x_limits_um = (0.0, L); inst.load_nN = c.load_nN; inst.set_dc_bias(0.0)
        say(f"GoToSpot({spot}) -- the image with the marked spots must be open in Igor")
        inst.goto_spot(spot)
        say(f"--- free end, x = {L:.1f} um ---")
        freq, Z, rec, pk = measure(L); freq_ref = freq
        xs.append(L); Zs.append(Z); result["invols_by_x"][f"{L:.1f}"] = rec["invols_m_per_V"]; result["cr_by_x"][f"{L:.1f}"] = pk
        say(f"InvOLS {rec['invols_m_per_V']}, {rec['n_tune_points']} pts "
            f"{rec['tune_f_lo_Hz'] / 1e3:.0f}-{rec['tune_f_hi_Hz'] / 1e3:.0f} kHz")
        for name, p in pk.items():
            if p["f_Hz"]:
                say(f"   {name.upper()}: {p['f_Hz'] / 1e3:8.2f} kHz  FWHM {p['fwhm_Hz'] / 1e3 if p['fwhm_Hz'] else float('nan'):6.2f} kHz  {p['snr_dB']:5.1f} dB")
        if pk["cr1"]["snr_dB"] < 3 and pk.get("cr2", {}).get("snr_dB", -np.inf) < 3:
            raise RuntimeError("neither CR1 nor CR2 visible at the free end -- is the tip engaged and the tune window right?")
        result.update(cr1_Hz=pk["cr1"]["f_Hz"], cr1_fwhm_Hz=pk["cr1"]["fwhm_Hz"],
                      cr2_Hz=pk.get("cr2", {}).get("f_Hz"), cr3_Hz=pk.get("cr3", {}).get("f_Hz"))
        if rec["tune_f_hi_Hz"] < 1.5e6:
            say(f"*** tune window tops out at {rec['tune_f_hi_Hz'] / 1e3:.0f} kHz; higher modes will be missed ***")

        x_lo = L
        if not P.run:
            x_lo = float(P.x_span_um[0]); result["stop_reason"] = "walk disabled; span from config"
        else:
            last_inv = rec["invols_m_per_V"]; x = L - P.step_um
            while x >= P.x_floor_um - 1e-9:
                say(f"--- x = {x:.1f} um ({L - x:.0f} um from the free end) ---")
                try:
                    freq, Z, rec, pk = measure(x)
                except Exception as e:
                    result["stop_reason"] = f"x={x:.1f}: {type(e).__name__}: {e}"; say("STOP: " + result["stop_reason"]); break
                v = rec.get("invols_m_per_V"); lo_b, hi_b = P.invols_bounds_m_per_V
                snr = max(pk["cr1"]["snr_dB"], pk.get("cr2", {}).get("snr_dB", -np.inf))
                say(f"InvOLS {v}, CR1 {(pk['cr1']['f_Hz'] or 0) / 1e3:.2f} kHz {pk['cr1']['snr_dB']:.1f} dB, best SNR {snr:.1f} dB")
                if v is None or not np.isfinite(v) or not (lo_b < v < hi_b):
                    result["stop_reason"] = f"x={x:.1f}: InvOLS {v} outside bounds {P.invols_bounds_m_per_V}"; say("STOP: " + result["stop_reason"]); break
                if last_inv and (v / last_inv > P.invols_jump or last_inv / v > P.invols_jump):
                    result["stop_reason"] = f"x={x:.1f}: InvOLS jumped {last_inv:.3e} -> {v:.3e}"; say("STOP: " + result["stop_reason"]); break
                if snr < P.min_snr_dB:
                    result["stop_reason"] = f"x={x:.1f}: SNR {snr:.1f} dB < {P.min_snr_dB} dB"; say("STOP: " + result["stop_reason"]); break
                xs.append(x); Zs.append(Z); result["invols_by_x"][f"{x:.1f}"] = v; result["cr_by_x"][f"{x:.1f}"] = pk
                last_inv = v; x_lo = x; x -= P.step_um
            else:
                result["stop_reason"] = f"reached x_floor_um = {P.x_floor_um}"; say("STOP: " + result["stop_reason"])
        result["x_lo_um"] = float(x_lo); result["n_good_positions"] = len(xs)
        inst.x_limits_um = (float(x_lo), L)
        say("=" * 78)
        say(f"REACHABLE SPAN x = {x_lo:.0f} .. {L:.0f} um ({L - x_lo:.0f} um, {len(xs)} good positions); "
            f"CR1 {result['cr1_Hz'] / 1e3:.2f} kHz")
        say("=" * 78)
        if xs:
            os.makedirs(os.path.dirname(cfg.preflight_npz), exist_ok=True)
            np.savez(cfg.preflight_npz, x_um=np.array(xs), freq_Hz=np.asarray(freq_ref), Z=np.array(Zs),
                     load_nN=c.load_nN, spot=spot)
        _dump(cfg.preflight_json, result)
        say(f"-> {cfg.preflight_json}")
    if result["x_lo_um"] is None:
        raise RuntimeError("pre-flight did not establish a span")
    return result


# ----------------------------------------------------------------------------------- #
#  2  two-domain sparse capture                                                         #
# ----------------------------------------------------------------------------------- #
def survey_positions(cfg: SMMConfig, x_lo, x_hi):
    c = cfg.capture
    xs = np.array(c.positions_um, float) if c.positions_um else np.round(np.linspace(x_lo, x_hi, c.n_positions), 1)
    if c.order == "free_end_first":
        xs = np.sort(xs)[::-1]
    elif c.order == "base_first":
        xs = np.sort(xs)
    elif c.order == "coarse_to_fine":
        xs = coarse_to_fine(np.sort(xs)[::-1])
    elif c.order != "as_given":
        raise ValueError("capture.order must be coarse_to_fine, free_end_first, base_first or as_given")
    return xs


def coarse_to_fine(xs):
    """Both ends first, then the midpoint, then the quarter points ... so that stopping early
    still leaves positions spread over the whole span (what `stop_when_stable` needs)."""
    xs = list(xs)
    if len(xs) <= 2:
        return np.array(xs)
    order = [0, len(xs) - 1]
    segments = [(0, len(xs) - 1)]
    while segments:
        nxt = []
        for a, b in segments:
            if b - a < 2:
                continue
            m = (a + b) // 2
            order.append(m); nxt += [(a, m), (m, b)]
        segments = nxt
    return np.array([xs[i] for i in order])


def capture(inst, cfg: SMMConfig, live=True, resume=True, **ovr):
    """The two-domain sparse survey: every (bias, spot) condition at each laser position.

    Checkpointed after every position (resume=True continues an interrupted run). The
    live figure updates after every position; `capture.stop_when_stable` ends the run
    once E(x) has stopped changing, a file called STOP in the campaign folder ends it
    after the current position, Ctrl-C after the current condition.
    """
    cfg = _with(cfg, "capture", **ovr); cfg.check(); c = cfg.capture
    pf = _load(cfg.preflight_json) or {}
    lo, hi = inst.x_limits_um if inst.x_limits_um else (pf.get("x_lo_um"), pf.get("x_hi_um"))
    if lo is None:
        raise RuntimeError("no reachable span: run preflight() or set preflight.x_span_um")
    xs = survey_positions(cfg, lo, hi)
    conds = make_conditions(c.bias_V, [c.load_nN], spots=list(c.spots), bias_order=c.bias_order)
    ref = next((i for i, k in enumerate(conds) if abs(k.bias_V - c.ref_bias_V) < 1e-9 and k.spot == c.spots[0]), 0)
    f_cr1 = pf.get("cr1_Hz")
    stop_file = cfg.path("STOP")
    if os.path.exists(stop_file):
        os.remove(stop_file)
    rec_json = cfg.path("10_survey", "survey_records.json")

    def invols_by_x():
        d = {}
        for r in inst.records:
            if r.get("invols_m_per_V") is not None:
                d[round(float(r["position_x_um"]), 1)] = float(r["invols_m_per_V"])
        for k, v in (pf.get("invols_by_x") or {}).items():
            d.setdefault(round(float(k), 1), float(v))
        return d

    lv = LiveCapture(cfg, f_cr1_Hz=f_cr1, invols_getter=invols_by_x, x_limits=(lo, hi), stop_file=stop_file) if live else None

    def on_position(series):
        _dump(rec_json, dict(invols_by_x=invols_by_x(), records=inst.records[-400:]))
        if lv is not None:
            lv.update(series)

    series = None
    with Stage("CAPTURE", cfg, inst):
        say(f"{xs.size} positions {xs.tolist()} (order: {c.order})")
        say(f"{len(conds)} conditions per position: " + ", ".join(str(k) for k in conds))
        say(f"checkpoint: {cfg.survey_npz}    (create {stop_file} to stop after the current position)")
        series = run_series(inst, xs, conds, ref_index=ref, rank=cfg.analysis.rank, checkpoint_path=cfg.survey_npz,
                            resume=resume, positions_um=xs, verbose=True, on_position=on_position,
                            should_stop=(lv.should_stop if lv else None))
        say(f"{series['x_um'].size} positions x {len(conds)} conditions x {series['freq_Hz'].size} frequencies")
        if lv is not None:
            lv.save(cfg.path("10_survey", "capture_live.png"))
    if series is not None:
        series["live"] = lv
    return series


# ----------------------------------------------------------------------------------- #
#  3  analysis                                                                          #
# ----------------------------------------------------------------------------------- #
def analyse(cfg: SMMConfig, series=None, show=True, **ovr):
    """delta, V_cpd, E_map(x) with leave-one-out, d_eff per position; writes 20_analysis/."""
    cfg = _with(cfg, "analysis", **ovr)
    if series is None:
        series = load_checkpoint(cfg.survey_npz)
    pf = _load(cfg.preflight_json) or {}
    recs = _load(cfg.path("10_survey", "survey_records.json")) or {}
    inv = {float(k): v for k, v in (recs.get("invols_by_x") or {}).items()}
    for k, v in (pf.get("invols_by_x") or {}).items():
        inv.setdefault(float(k), v)
    lims = (pf.get("x_lo_um"), pf.get("x_hi_um")) if pf.get("x_lo_um") is not None else None
    res = analyse_survey(series, cfg, f_cr1_Hz=pf.get("cr1_Hz"), invols_by_x=inv, x_limits=lims, verbose=True)
    t, em = res["table"], res["emap"]
    out = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), f_cr1_Hz=res["f_cr1_Hz"], cr_band_Hz=res["cr_band_Hz"],
               qs_band_Hz=res["qs_band_Hz"], delta=dict(res["delta"], D=abs(res["delta"]["D"])), v_cpd_V=res["v_cpd_V"],
               v_cpd_spread_V=res["v_cpd_spread_V"], Q_median=float(np.nanmedian(t["Q"])),
               deff_qs_median_pm_per_V=float(np.nanmedian(res["deff_qs_pm_per_V"])),
               deff_qs_sd_pct=float(100 * np.nanstd(res["deff_qs_pm_per_V"]) / np.nanmedian(res["deff_qs_pm_per_V"])),
               loo_ratio_median=float(np.nanmedian(em["loo_ratio"])),
               loo_ratio_range=[float(np.nanmin(em["loo_ratio"])), float(np.nanmax(em["loo_ratio"]))],
               invols_known=res["invols_known"], recommended=res["recommend"]["recommended"], best=res["recommend"]["best"],
               fallback=res["recommend"]["fallback"],
               esbs={k: v for k, v in res["esbs"].items() if not isinstance(v, np.ndarray)},
               per_position=[dict(x_um=float(t["x_um"][j]), invols=t["invols"][j], f_cr_Hz=t["f_cr_Hz"][j], Q=t["Q"][j],
                                  E_raw=t["E_raw"][j], E_meas=res["corr"]["E_meas"][j],
                                  E_loo=float(np.interp(t["x_um"][j], em["x_meas"], em["E_loo"])),
                                  v_cpd_qs=t["v_cpd_qs"][j], ratio_qs=t["ratio_qs"][j], ratio_cr=t["ratio_cr"][j],
                                  flip_qs_deg=t["flip_qs"][j], flip_cr_deg=t["flip_cr"][j],
                                  deff_qs_pm_per_V=res["deff_qs_pm_per_V"][j], deff_cr_pm_per_V=res["deff_cr_pm_per_V"][j])
                             for j in range(t["x_um"].size)],
               E_grid=dict(x_um=em["x_grid"], E=em["E"], E_lo=em["E_lo"], E_hi=em["E_hi"]))
    _dump(cfg.analysis_json, out)
    os.makedirs(os.path.dirname(cfg.analysis_npz), exist_ok=True)
    np.savez_compressed(cfg.analysis_npz, x_grid=em["x_grid"], E=em["E"], E_lo=em["E_lo"], E_hi=em["E_hi"],
                        Pcr_rec=em["Pcr_rec"], Pqs_rec=em["Pqs_rec"], f_cr_Hz=em["f_cr_Hz"], f_qs_Hz=em["f_qs_Hz"],
                        x_meas=em["x_meas"], Pcr_meas=em["Pcr_meas"], Pqs_meas=em["Pqs_meas"])
    print(f"\n-> {cfg.analysis_json}")
    if show:
        lv = LiveCapture(cfg, f_cr1_Hz=res["f_cr1_Hz"], invols_getter=lambda: inv, x_limits=lims, show=False)
        lv.update(series); lv.draw(series, res); lv.save(cfg.path("20_analysis", "emap_analysis.png"))
    res["saved"] = out
    return res


# ----------------------------------------------------------------------------------- #
#  4  imaging plan: QS at the electrostatic blind spot, BE at interior stops              #
# ----------------------------------------------------------------------------------- #
def _E_at(an, x):
    xg, E = np.asarray(an["E_grid"]["x_um"], float), np.asarray(an["E_grid"]["E"], float)
    return float(np.interp(x, xg, E))


def _variation_at(an, x, drift_um):
    xg, E = np.asarray(an["E_grid"]["x_um"], float), np.asarray(an["E_grid"]["E"], float)
    m = np.abs(xg - x) <= drift_um + 1e-9
    return float(100 * (E[m].max() - E[m].min()) / _E_at(an, x)) if m.sum() > 1 else float("nan")


def plan_imaging(cfg: SMMConfig, be_positions_um=None, qs_x_um=None, **ovr):
    """Decide where to image, from the analysis.

    * QS stop: the electrostatic blind spot (b_QS/P_QS smallest; usually the free end on a stiff
      lever). A single-frequency QS frame there needs no electrostatic correction; the residual
      error delta*|b/P| is recorded. Override with `qs_x_um` (or cfg.imaging.qs_x_um).
    * BE stops: `be_positions_um` (or cfg.imaging.be_positions_um) if you choose them, otherwise
      the recommended stops (largest validated E, flat over the drift budget, spaced). E_map, its
      drift sensitivity and whether the leave-one-out test validates that stretch are recorded
      for each.
    Writes 30_plan/imaging_plan.json and prints the plan.
    """
    cfg = _with(cfg, "imaging", **ovr); im = cfg.imaging
    an = _load(cfg.analysis_json)
    if an is None:
        raise RuntimeError("run analyse() first")
    xg = np.asarray(an["E_grid"]["x_um"], float); lo, hi = xg.min(), xg.max()
    es = an["esbs"]
    xq = float(qs_x_um if qs_x_um is not None else (im.qs_x_um if im.qs_x_um is not None else es["x_esbs_um"]))
    xq = float(np.clip(xq, lo, hi))
    stops = [dict(kind="qs", stage_um=xq, E_map=1.0, E_at_x=_E_at(an, xq), bias_plan=list(im.qs_bias_plan),
                  source="override" if (qs_x_um is not None or im.qs_x_um is not None) else "blind spot",
                  b_over_P=es["r_at_stop"], raw_error_pct=es["raw_error_pct"], blind_spot_crossing=es["crossing"])]
    chosen = be_positions_um if be_positions_um is not None else im.be_positions_um
    if chosen:
        bes = [dict(stage_um=float(x), source="chosen") for x in chosen]
    else:
        bes = [dict(stage_um=b["stage_um"], source="recommended") for b in (an["best"] or an.get("fallback", []))[:im.n_be_positions]]
        if not an["best"] and bes:
            print(f"** no stop meets the flatness criterion ({im.tol_pct:.0f} % over +/-{im.drift_um:.0f} um); "
                  f"using the flattest allowed stops **")
    xs_meas = np.array([r["x_um"] for r in an["per_position"]]); loo = np.array([r["E_meas"] / r["E_loo"] for r in an["per_position"]])
    o = np.argsort(xs_meas); xs_meas, loo = xs_meas[o], loo[o]
    for b in bes:
        x = b["stage_um"]
        if not (lo <= x <= hi):
            raise ValueError(f"BE stop x = {x} um is outside the measured span {lo:.0f}-{hi:.0f} um")
        i = int(np.searchsorted(xs_meas, x)); i0, i1 = max(i - 1, 0), min(i, xs_meas.size - 1)
        validated = bool(abs(loo[i0] - 1) <= im.loo_tol and abs(loo[i1] - 1) <= im.loo_tol)
        b.update(kind="be", E_map=_E_at(an, x), variation_pct=_variation_at(an, x, im.drift_um), loo_validated=validated,
                 loo_ratios_bracketing=[float(loo[i0]), float(loo[i1])], bias_plan=list(im.be_bias_plan))
    stops += bes
    plan = dict(planned=time.strftime("%Y-%m-%d %H:%M:%S"), v_cpd_V=an["v_cpd_V"], f_cr1_survey_Hz=an["f_cr1_Hz"],
                Q_survey=an["Q_median"], deff_qs_survey_pm_per_V=an["deff_qs_median_pm_per_V"], stops=stops)
    _dump(cfg.plan_json, plan)
    print(f"{'stop':>4} {'kind':>4} {'x (um)':>8} {'E_map':>7} {'E +/- over drift':>16}  {'biases':<22} note")
    for k, st in enumerate(stops):
        if st["kind"] == "qs":
            print(f"{k:>4} {'QS':>4} {st['stage_um']:8.1f} {'1 (QS)':>7} {'':>16}  {str(st['bias_plan']):<22} "
                  f"blind spot ({st['source']}): |b/P| {st['b_over_P']:.3f}, uncorrected error ~{st['raw_error_pct']:.1f} %"
                  + ("" if st["blind_spot_crossing"] else "; no true null in the span, end stop used"))
        else:
            print(f"{k:>4} {'BE':>4} {st['stage_um']:8.1f} {st['E_map']:7.0f} {st['variation_pct']:15.1f} %  {str(st['bias_plan']):<22} "
                  f"{st['source']}; LOO {'validated' if st['loo_validated'] else '** NOT validated **'} "
                  f"({st['loo_ratios_bracketing'][0]:.2f}, {st['loo_ratios_bracketing'][1]:.2f})")
    print(f"\nV_cpd = {plan['v_cpd_V']:+.2f} V; survey QS d_eff {plan['deff_qs_survey_pm_per_V']:.2f} pm/V  -> {cfg.plan_json}")
    return plan


# ----------------------------------------------------------------------------------- #
#  5  go to a stop and tune (tip stays engaged for the frames)                           #
# ----------------------------------------------------------------------------------- #
def goto_and_tune(inst, cfg: SMMConfig, x, tag="tune_before", withdraw=False, record_to=None):
    """Laser -> x, AutoWedge + InvOLS, engage at the capture load, one saved tune.

    Returns dict(f_cr1_Hz, fwhm_Hz, Q, invols_m_per_V, ...). The tip is left ENGAGED unless
    withdraw=True. `record_to` (a dict) receives the result under `tag`.
    """
    x = float(x); c, im = cfg.capture, cfg.imaging
    plan = _load(cfg.plan_json) or {}
    f_guess = plan.get("f_cr1_survey_Hz") or (_load(cfg.preflight_json) or {}).get("cr1_Hz") or float(np.mean(cfg.probe.cr_bands_Hz["cr1"]))
    search = (im.cr1_search_frac[0] * f_guess, im.cr1_search_frac[1] * f_guess)
    out = None
    with Stage(f"GOTO+{tag.upper()}", cfg, inst, withdraw=withdraw):
        say(f"laser -> stage x = {x:.1f} um, AutoWedge + InvOLS, engage {c.load_nN:.0f} nN, tune")
        inst.set_dc_bias(0.0)
        if inst.current_spot != c.spots[0]:
            inst.goto_spot(c.spots[0])
        inst.a.withdraw(); time.sleep(1); inst._goto_and_prepare(x, capture_image=True)
        if inst._invols is None:
            raise RuntimeError("InvOLS None at the stop")
        sp = inst.set_load(c.load_nN, reengage=True)
        t = inst.tune_eigenmode(position_label=f"{tag}_X{x:05.1f}", scan_index=900, save_tune_data=True)
        td = t["tune_data"]
        if "Z" in td:
            freq, Z = np.asarray(td["frequency"], float), np.asarray(td["Z"])
        else:
            from ..asylum import tune_to_complex
            freq, Z = tune_to_complex(td, band_Hz=None)
        pk = peak(freq, Z, search)
        floor = float(np.median(np.abs(Z)[(freq > 50e3)])) if (freq > 50e3).any() else float(np.median(np.abs(Z)))
        out = dict(stage_um=x, f_cr1_Hz=pk["f_Hz"], fwhm_Hz=pk["fwhm_Hz"], Q=pk["Q"], peak_over_median=pk["amp"] / floor,
                   amp_cr1_V=pk["amp"], invols_m_per_V=float(inst._invols), setpoint_V=float(sp),
                   time=time.strftime("%Y-%m-%d %H:%M:%S"))
        say(f"InvOLS {inst._invols:.4e} m/V, CR1 {pk['f_Hz'] / 1e3:.3f} kHz, FWHM {pk['fwhm_Hz'] / 1e3:.2f} kHz, "
            f"Q {pk['Q']:.0f}, peak/median {out['peak_over_median']:.1f}")
        if out["peak_over_median"] < 3:
            say("** WARNING: weak CR1 peak -- check the tune window and the engage")
        if record_to is not None:
            record_to[tag] = out
        if not withdraw:
            say(f"tip is ENGAGED at x = {x:.1f} um, bias 0 V")
    return out


# ----------------------------------------------------------------------------------- #
#  6  frames at every planned stop                                                       #
# ----------------------------------------------------------------------------------- #
def _bias_for(name, v_cpd):
    if name.endswith("Vcpd"): return float(v_cpd)
    if name.endswith("Vhalf"): return float(v_cpd) / 2
    return 0.0


def run_imaging(inst, cfg: SMMConfig, stops=None, **ovr):
    """For each planned stop: laser -> stop, InvOLS, engage, tune, frames, (tune-after), withdraw.

    QS stop : single-frequency frames at imaging.f_qs_Hz through Igor (one per entry in the bias plan),
              converted with E = 1 (blind spot: no electrostatic correction).
    BE stops: one BE frame per bias-plan entry through the adapter named in imaging.be_acquire, band
              centred on the CR1 from the tune at that stop, width = be_width_x_fwhm x FWHM (clipped);
              converted by phase-coherent domain averaging + SHO fit with E = E_map(stop).
    `stops` = indices into the plan to run a subset (e.g. [0] for the QS frame only). Results
    accumulate in 40_images/images.json; a failed conversion keeps the file path and moves on.
    """
    cfg = _with(cfg, "imaging", **ovr); im, c = cfg.imaging, cfg.capture
    plan = _load(cfg.plan_json)
    if plan is None:
        raise RuntimeError("run plan_imaging() first")
    sel = list(range(len(plan["stops"]))) if stops is None else list(stops)
    log = _load(cfg.images_json) or dict(frames=[], stops={})
    virtual = getattr(inst.a, "igor", None) is None
    be_acquire = None
    if any(plan["stops"][k]["kind"] == "be" for k in sel) and not virtual:
        be_acquire = imaging.load_be_acquire(im.be_acquire)          # fail before anything moves
    v_cpd = plan["v_cpd_V"]
    out_dir = cfg.path("40_images")
    for k in sel:
        st = plan["stops"][k]; x = st["stage_um"]; tagk = f"stop{k}_{st['kind']}_x{x:05.1f}"
        rec = log["stops"].setdefault(tagk, dict(stop=st))
        before = goto_and_tune(inst, cfg, x, tag=f"{tagk}_before", record_to=rec)
        with Stage(f"FRAMES {tagk}", cfg, inst, withdraw=False):
            if virtual:
                say("virtual instrument: frames are not acquired; the plan and tunes are exercised only")
            elif st["kind"] == "qs":
                frame = im.frame or imaging.read_frame(inst)
                say(f"scan frame: {frame}")
                if im.frame:
                    imaging.set_frame(inst, im.frame, say=say)
                for name in st["bias_plan"]:
                    vdc = _bias_for(name, v_cpd); t0 = time.time()
                    path = imaging.single_frequency_frame(inst, im.image_base, vdc, im.f_qs_Hz, c.drive_V, im.scan_timeout_s, say=say)
                    e = dict(stop=k, kind="qs", name=name, path=path, stage_um=x, bias_V=vdc, drive_Hz=im.f_qs_Hz, drive_V=c.drive_V,
                             E_used=1.0, invols=before["invols_m_per_V"], minutes=(time.time() - t0) / 60, time=time.strftime("%H:%M:%S"))
                    try:
                        d = imaging.frame_deff(path, c.drive_V, 1.0)
                        e.update({kk: v for kk, v in d.items() if kk != "deff_map"})
                        say(f"    d_eff QS: domain 1 {d.get('domain1_median', float('nan')):.2f}, domain 2 "
                            f"{d.get('domain2_median', float('nan')):.2f}, mean {d.get('mean_of_domains', d['median_all']):.2f} pm/V "
                            f"(survey {plan['deff_qs_survey_pm_per_V']:.2f}; uncorrected error here ~{st['raw_error_pct']:.1f} %)")
                    except Exception as ex:
                        say(f"    (conversion failed: {type(ex).__name__}: {ex})")
                    log["frames"].append(e); _dump(cfg.images_json, log)
            else:
                f_c = before["f_cr1_Hz"]; w = float(np.clip(im.be_width_x_fwhm * before["fwhm_Hz"], *im.be_width_Hz))
                say(f"BE band {f_c / 1e3:.2f} kHz +/- {w / 2e3:.1f} kHz ({im.be_width_x_fwhm:.0f} x FWHM {before['fwhm_Hz'] / 1e3:.2f} kHz), "
                    f"{im.be_px} px, {im.be_repeats} repeat(s), V_ac {c.drive_V} V, E_map {st['E_map']:.0f}")
                for name in st["bias_plan"]:
                    vdc = _bias_for(name, v_cpd); t0 = time.time()
                    say(f"--- BE frame {name}: V_dc {vdc:+.3f} V ---")
                    inst.set_dc_bias(vdc); time.sleep(1.0)
                    path = be_acquire(inst=inst, name=f"{tagk}_{name}", v_dc=vdc, v_ac=c.drive_V, f_center_Hz=f_c, f_width_Hz=w,
                                      out_dir=out_dir, px=im.be_px, repeats=im.be_repeats, imaging_config=im.__dict__)
                    e = dict(stop=k, kind="be", name=name, path=str(path), stage_um=x, bias_V=vdc, f_center_Hz=f_c, f_width_Hz=w,
                             drive_V=c.drive_V, px=im.be_px, repeats=im.be_repeats, E_used=st["E_map"], invols=before["invols_m_per_V"],
                             Q_tune=before["Q"], minutes=(time.time() - t0) / 60, time=time.strftime("%H:%M:%S"))
                    try:
                        d = imaging.be_deff(path, st["E_map"], before["invols_m_per_V"], c.drive_V, cfg.probe.amp_divisor)
                        e.update({kk: v for kk, v in d.items() if kk != "path"})
                        say(f"    d_eff BE: {d['deff_pm_per_V']:.2f} pm/V (domains {d['domain1_pm_per_V']:.2f} / {d['domain2_pm_per_V']:.2f}, "
                            f"common mode/P {d['common_over_P']:.2f}, f0 {d['f0_domain1_Hz'] / 1e3:.2f} kHz, Q {d['Q_domain1']:.0f}; "
                            f"survey QS {plan['deff_qs_survey_pm_per_V']:.2f})")
                    except Exception as ex:
                        say(f"    (conversion failed: {type(ex).__name__}: {ex} -- file kept, convert offline)")
                    log["frames"].append(e); _dump(cfg.images_json, log)
                inst.set_dc_bias(0.0)
        if im.tune_after_each_stop:
            after = goto_and_tune(inst, cfg, x, tag=f"{tagk}_after", withdraw=True, record_to=rec)
            print(f"stop {k}: CR1 drift {after['f_cr1_Hz'] - before['f_cr1_Hz']:+.0f} Hz (FWHM {before['fwhm_Hz']:.0f} Hz), "
                  f"InvOLS {before['invols_m_per_V']:.3e} -> {after['invols_m_per_V']:.3e}, "
                  f"CR1 amplitude after/before {after['amp_cr1_V'] / before['amp_cr1_V']:.2f}")
        else:
            finish(inst, tagk)
        _dump(cfg.images_json, log)
    print(f"-> {cfg.images_json}")
    return log


# ----------------------------------------------------------------------------------- #
#  7  close-out                                                                          #
# ----------------------------------------------------------------------------------- #
def closeout(inst, cfg: SMMConfig):
    """Bias 0 V, withdraw, manifest; prints a one-table summary of every frame's d_eff."""
    finish(inst, "CLOSE-OUT")
    log = _load(cfg.images_json)
    if log and log.get("frames"):
        print(f"{'stop':>4} {'kind':>4} {'x (um)':>7} {'bias V':>7} {'E':>6} {'d_eff pm/V':>11}  file")
        for e in log["frames"]:
            d = e.get("deff_pm_per_V", e.get("mean_of_domains", e.get("median_all", float("nan"))))
            print(f"{e['stop']:>4} {e['kind']:>4} {e['stage_um']:7.1f} {e['bias_V']:+7.2f} {e['E_used']:6.0f} {d:11.2f}  {os.path.basename(str(e['path']))}")
    return manifest(cfg)


def manifest(cfg: SMMConfig):
    m = dict(written=time.strftime("%Y-%m-%d %H:%M:%S"), config=cfg.to_dict(),
             preflight=_load(cfg.preflight_json), analysis={k: v for k, v in (_load(cfg.analysis_json) or {}).items()
                                                            if k not in ("E_grid", "per_position")},
             plan=_load(cfg.plan_json), images=_load(cfg.images_json),
             files={k: os.path.exists(p) for k, p in (("survey", cfg.survey_npz), ("analysis_maps", cfg.analysis_npz),
                                                     ("preflight_walk", cfg.preflight_npz))})
    _dump(cfg.manifest_json, m)
    print(f"manifest -> {cfg.manifest_json}")
    return m
