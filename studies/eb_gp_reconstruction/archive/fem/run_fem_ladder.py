#!/usr/bin/env python3
"""Run the FEM ladder needed for the physics-informed active-learning study.

WHAT THIS PRODUCES, AND WHY IT IS NOT THE EXISTING SWEEP
--------------------------------------------------------
`run_fem_sweep.py` gave 16 k1 rungs from 1e1 to 1e6 N/m at a single damping
level.  Two things are missing for a surrogate the reveal loop can interpolate:

  1. k1 SPACING.  16 rungs over five decades is a factor 2.15 per step.
     Interpolating a mode shape across that is hopeless.  The EB surrogate
     needed ~5 % steps; with resonance-aligned interpolation ~13 % is enough,
     hence 32 rungs over the 1.7 decades that actually matter.

     The k1 range is set by three independent estimates that agree: Hertz for a
     25 nm PtIr tip at 1500 nN on E* ~ 100 GPa gives 1310 N/m; the mode-A shape
     fit against the dense map gives 1000-2154; the EB fit in the reveal loop
     settles at 320-390.  100-5000 N/m brackets all of them with margin.

  2. DAMPING.  At the app default log_damping = 5.0 the FEM mode-1 Q is 112-169
     depending on k1, against a measured 191 (FWHM 1500 Hz at 292.9 kHz, stable
     across all 101 positions).  With one damping level the surrogate cannot fit
     the linewidth, so log_damping becomes a second axis.  The four levels below
     give Q ~ 473 / 299 / 188 / 119 at k1 = 1000, bracketing the measurement.

FREQUENCY SAMPLING IS ALREADY FINE -- an earlier version of this script narrowed
the band to buy resolution, on the assumption that 600 points over 2 MHz meant
3.3 kHz per bin.  That was wrong.  AFeMulator uses `freq_spacing = lorentzian`
with `n_pts_per_res = 200`, so the grid concentrates around each resonance: the
existing exports have 122 Hz spacing near mode 1, i.e. 12 bins across the
measured FWHM.  The app default 0-2000 kHz window is therefore kept, which also
returns mode 2 in the same export -- needed for the f2/f1 test that the whole
EB-vs-FEM question turns on.

USAGE (on the machine running AFeMulator, with a VISIBLE browser tab at
http://127.0.0.1:8050 -- the compute loop lives in that tab and a backgrounded
tab is throttled by the browser until requests never complete):

    # 0. sanity-check the connection and see what parameters this build exposes
    python run_fem_ladder.py --probe

    # 1. the run.  Resumable: re-running skips exports that already exist.
    python run_fem_ladder.py --stl "C:\\path\\to\\BS_Multi75G.stl" \
        --name Multi75G --out ../data_ladder

    # 2. frictionless only (halves the time) if you want a first look
    python run_fem_ladder.py --stl ... --lateral frictionless

Expect ~2 min per export -> ~2 h for frictionless only, ~4 h for both.
Standard library only, matching the rest of this directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

import api_map
from afemulator_client import AFeMulator, AFeMulatorError
# Reuse the verified conventions from the original sweep rather than restating
# them: log10 encoding of the springs, read-back verification, tilt, drive
# amplitudes, and the lateral-BC definitions.
from run_fem_sweep import LATERAL, Setter, TILT_DEG, Z_DRIVE_PM
# NOT imported: eigen_freqs / free_fingerprint. run_fem_sweep.eigen_freqs looks
# for response keys 'freqs' / 'frequencies' / 'f', and this build uses none of
# them -- data/eigen names the array differently (the HDF5 export calls it
# frequencies_hz). The existing fem_manifest_Multi75G.json records
# free_eigen_khz = [] for exactly that reason: the original sweep only PRINTS
# the fingerprint and never asserts on it, so an always-empty result was
# invisible. Replaced below by a search that does not depend on the key name.

# ----------------------------------------------------------------- the ladder
K1_LO, K1_HI, K1_N = 100.0, 5000.0, 32          # N/m; 12.9 % per step
K1_GRID = [K1_LO * (K1_HI / K1_LO) ** (i / (K1_N - 1)) for i in range(K1_N)]

# log_damping levels.  Measured on this build: Q ~ damping^-0.80, and at the app
# default ld = 5.0 the mode-1 Q runs 257 (k1=100) -> 240 (113) -> 169 (215) ->
# 119 (1000) -> 112 (4642).  Q therefore FALLS with k1, so no single damping level
# sits at the measured Q = 191 across the ladder: below k1 ~ 170 the model is too
# lightly damped, above it too heavily.  Hitting 191 needs ld = 5.16 at k1 = 100
# and ld = 4.71 at k1 = 4642, so the grid has to span ~4.6-5.3 to bracket the
# measurement everywhere.  A first attempt at [4.4, 4.6, 4.8, 5.0] missed the top
# end entirely (Q range 240-729, never reaching 191).
#
# Three levels, not five: the AFeMulator tab degrades with accumulated computes
# (measured: 0.3 min/export rising to 2.0, then a hard 180 s browser timeout), so
# total compute count is the binding constraint. Q ~ damping^-0.80 is a clean
# power law, and the fit only needs it to set the linewidth, so three rungs
# spanning the required range interpolate fine.
LOG_DAMPING = [4.6, 4.95, 5.3]

# Keep the app's own Lorentzian-spaced window: it resolves mode 1 to 122 Hz and
# returns mode 2 in the same export.
FREQ_LO_KHZ, FREQ_HI_KHZ = 0.0, 2000.0

# Deterministic library: every stochastic contribution off, so two runs of the
# same parameters are bit-identical and the surrogate carries no simulated noise.
QUIET = dict(noise_abs=0.0, noise_pct=0.0, onef_amp=0.0, ambient_thermal=False)

# Free-lever acceptance test.  A clamped-free Euler-Bernoulli beam has
# f2/f1 = (4.694/1.8751)^2 = 6.267 and f3/f1 = 17.55.  The existing sweep never
# tested this: its softest rung, k1 = 10 N/m, already puts f1 at 150.7 kHz,
# i.e. twice the free value, so the free limit is entirely unvalidated.  If the
# model cannot reproduce it, no amount of contact-parameter tuning will help --
# and it would explain why FEM responds to lateral stiffness with the opposite
# sign to both EB and the analytic clamped-pinned -> clamped-clamped limits.
FREE_F1_KHZ_EXPECTED = 74.8
FREE_RATIO_EXPECTED = 6.267
FREE_TOL_F1 = 0.10               # 10 % on the absolute free resonance
FREE_TOL_RATIO = 0.05            # 5 % on f2/f1


_EIGEN_REPORTED = [False]

_THROTTLE_HINTS = ("408", "timeout", "did not respond", "backgrounded",
                   "browser session")


def _is_throttle(e) -> bool:
    m = str(e).lower()
    return any(h in m for h in _THROTTLE_HINTS)


def retry(fn, what, tries=4, waits=(10, 30, 90)):
    """Run a browser round-trip, retrying transient tab throttling.

    The compute loop lives inside the AFeMulator page, and the tab slows down as
    computes accumulate until the server returns HTTP 408 "browser session did
    not respond". That is transient: pausing lets the tab catch up. Without this
    a single hiccup throws away the whole run, and the failures observed came
    after 15-30 computes, i.e. always mid-ladder.
    """
    last = None
    for i in range(tries):
        try:
            return fn()
        except AFeMulatorError as e:
            last = e
            if not _is_throttle(e) or i == tries - 1:
                raise
            w = waits[min(i, len(waits) - 1)]
            print(f"        {what}: browser did not respond "
                  f"(attempt {i+1}/{tries}); pausing {w}s and retrying. "
                  "Bring the AFeMulator tab to the foreground.", flush=True)
            time.sleep(w)
            try:
                if not app_ready_hint(fn):
                    pass
            except Exception:
                pass
    raise last


def app_ready_hint(_fn):
    return True


def _walk(obj, path=""):
    """Yield (path, value) for every leaf and list in a nested JSON response."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        yield path, obj
        if obj and isinstance(obj[0], (dict, list, tuple)):
            for i, v in enumerate(obj[:4]):
                yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, obj


def _numeric_arrays(ev, min_len=2):
    out = []
    for path, v in _walk(ev):
        if isinstance(v, (list, tuple)) and len(v) >= min_len:
            try:
                nums = [float(x) for x in v]
            except (TypeError, ValueError):
                continue
            out.append((path, nums))
    return out


def eigen_khz(app: AFeMulator):
    """(frequencies_kHz, mode_types_or_None) from /api/data/eigen.

    The array is located by searching the response rather than by name, because
    the name differs between this build and what run_fem_sweep expects. Units
    are inferred from magnitude: a contact resonance is O(100) in kHz and
    O(1e5) in Hz, so anything above 1e4 is Hz.
    """
    ev = app.eigen()
    arrays = _numeric_arrays(ev)
    if _EIGEN_REPORTED[0] is False:
        _EIGEN_REPORTED[0] = True
        print("        /api/data/eigen returned "
              f"{sorted(ev)[:8] if isinstance(ev, dict) else type(ev).__name__}"
              f"; numeric arrays: "
              f"{[(p, len(n)) for p, n in arrays][:6]}")
    if not arrays:
        raise AFeMulatorError(
            "data/eigen contained no numeric array. Raw response:\n"
            f"  {str(ev)[:600]}\n"
            "Paste this and the extractor can be pointed at the right field.")
    freqs = mt = None
    for path, nums in sorted(arrays, key=lambda t: -len(t[1])):
        low = path.lower()
        if freqs is None and "freq" in low:
            freqs = nums
        if mt is None and any(t in low for t in ("type", "class", "kind")):
            mt = nums
    if freqs is None:
        # fall back to the longest array that looks like ascending frequencies
        for path, nums in sorted(arrays, key=lambda t: -len(t[1])):
            a = [x for x in nums if x > 0]
            if len(a) >= 3 and all(b >= a_ for a_, b in zip(a, a[1:])):
                freqs = nums
                print(f"        (no 'freq' key; using {path!r} as the "
                      "frequency array)")
                break
    if freqs is None:
        raise AFeMulatorError(
            "could not identify the eigenfrequency array in data/eigen. "
            f"Candidates: {[(p, len(n)) for p, n in arrays]}")
    pos = [f for f in freqs if f > 1e-9]
    if pos and float(np.median(pos)) > 1e4:      # Hz -> kHz
        freqs = [f / 1e3 for f in freqs]
    return freqs, mt


def flexural_khz(app: AFeMulator, n=6):
    """Flexural eigenfrequencies in kHz, lowest first.

    Uses the solver's own mode-type labels when present (1 = flexural in the
    HDF5 exports); otherwise returns every mode and lets the caller window.
    """
    fr, mt = eigen_khz(app)
    if mt is not None and len(mt) == len(fr):
        flex = [f for f, t in zip(fr, mt) if int(round(t)) == 1 and f > 1e-9]
        if len(flex) >= 2:
            return sorted(flex)[:n]
    return sorted(f for f in fr if f > 1e-9)[:n]


def free_fingerprint_checked(app: AFeMulator, setp, mapping: dict, n=6):
    """Free-probe eigenfrequencies, with the springs actually released."""
    args = {"k_z": 0.0, "k_x": 0.0, "k_y": 0.0}
    for f in ("kz_zero", "kx_zero", "ky_zero"):
        if mapping.get(f):
            args[f] = True
    setp(**args)
    app.compute("fea")
    return flexural_khz(app, n)


def set_extra(app: AFeMulator, **kv):
    """Set parameters by their RAW api key, with read-back verification.

    api_map only knows 21 of the 75 keys this build exposes; log_damping and the
    noise terms are not among them, so they are written directly.
    """
    app.set_params(**kv)
    time.sleep(0.2)
    flat = api_map.flatten(app.get_params())
    for k, v in kv.items():
        got = flat.get(k)
        if isinstance(v, bool):
            ok = bool(got) == v
        else:
            try:
                ok = abs(float(got) - float(v)) <= 1e-9 + 1e-3 * abs(float(v))
            except (TypeError, ValueError):
                ok = False
        if not ok:
            raise AFeMulatorError(
                f"set_params({k}={v!r}) did not take effect (read back {got!r}). "
                f"This build may name it differently -- check --probe output.")


def flexural_mode1_khz(app: AFeMulator) -> float:
    """Lowest eigenfrequency, which over this k1 range IS flexural mode 1.

    Guarded rather than assumed: the frictionless torsional branch sits at
    ~720 kHz and is k1-independent, while flexural mode 1 runs 330-374 kHz
    over 100-5000 N/m, so the minimum is unambiguous here.  The assertion
    fires if a future geometry breaks that ordering.
    """
    fs = flexural_khz(app, 8)
    if not fs:
        raise AFeMulatorError("data/eigen returned no usable frequencies")
    f1 = min(fs)
    if not (150.0 <= f1 <= 600.0):
        raise AFeMulatorError(
            f"lowest eigenfrequency {f1:.1f} kHz is outside the 150-600 kHz "
            "band expected for flexural mode 1 over this k1 range. The mode "
            "ordering assumption has broken -- inspect eigen() before trusting "
            "the fine-band centring.")
    return f1


def check_free_limit(fp) -> tuple[bool, str]:
    """Validate the free-lever fingerprint.

    Returns (ok, message). An EMPTY or too-short fingerprint is a readback
    problem, not a physics problem, and is reported as such -- conflating the
    two sends you looking for a modelling fault that is not there.
    """
    fs = [f for f in fp if f and f > 1.0]
    if len(fs) < 2:
        return False, (f"READBACK PROBLEM, not a physics result: only {len(fs)} "
                       "free eigenfrequencies came back from data/eigen "
                       f"(raw: {fp}). The solve itself may be fine.")
    f1, f2 = fs[0], fs[1]
    ratio = f2 / f1
    e1 = abs(f1 - FREE_F1_KHZ_EXPECTED) / FREE_F1_KHZ_EXPECTED
    er = abs(ratio - FREE_RATIO_EXPECTED) / FREE_RATIO_EXPECTED
    msg = (f"free lever: f1 = {f1:.2f} kHz (expect {FREE_F1_KHZ_EXPECTED:.1f}, "
           f"{100*e1:+.1f} %), f2/f1 = {ratio:.3f} (expect "
           f"{FREE_RATIO_EXPECTED:.3f}, {100*er:+.1f} %)")
    return (e1 <= FREE_TOL_F1 and er <= FREE_TOL_RATIO), msg


def run(app, stl, name, out_dir, mapping, lateral, verify, dry_run,
        skip_free_check=False, limit=None, start_at=0, pause_every=0,
        pause_secs=30):
    os.makedirs(out_dir, exist_ok=True)
    setp = Setter(app, mapping, verify)
    mpath = os.path.join(out_dir, f"fem_ladder_manifest_{name}.json")
    manifest = {}
    if os.path.exists(mpath):
        with open(mpath) as fh:
            manifest = json.load(fh)
        prev = manifest.get("runs", [])
        print(f"resuming from {mpath} ({len(prev)} runs already recorded)")
        stale = sorted({round(r.get("log_damping", -1), 3) for r in prev}
                       - {round(v, 3) for v in LOG_DAMPING})
        if stale:
            print(f"  *** STALE DATA: the manifest contains damping levels "
                  f"{stale} that are NOT in the current grid "
                  f"{LOG_DAMPING}.\n"
                  "      Those files came from an earlier configuration and any "
                  "level shared with the\n"
                  "      current grid will be REUSED rather than recomputed, "
                  "mixing two code paths.\n"
                  "      Delete the output directory and start clean unless you "
                  "know why you want this. ***\n")

    app.wait_ready()
    app.set_autorun(False)          # never let the UI recompute under us

    load_info = {}
    if stl:
        load_info = app.load_stl(stl) or {}
        print(f"  loaded: {load_info.get('path')}")
        print(f"  mesh  : n_faces={load_info.get('n_faces')}")
        # The dropdown can silently override the file path, and mesh size is
        # the only trustworthy evidence of what was actually loaded.
        if load_info.get("n_faces") == 48:
            raise AFeMulatorError(
                "48 faces -> budget_75.stl was loaded, not Multi75G. The "
                "standard-model dropdown overrode the file path. Set the "
                "dropdown to the same probe, then retry.")
        if load_info.get("n_faces") not in (None, 68):
            print(f"  WARNING: expected 68 faces for Multi75G, got "
                  f"{load_info.get('n_faces')}. The existing dataset and the "
                  "published geometry_Multi75G.json assume 68.")
    if mapping.get("tilt"):
        setp(tilt=TILT_DEG)
    retry(lambda: app.compute("full"), "compute full")

    # ---- free-lever validation (cheap, and the decisive structural check) ----
    fp = free_fingerprint_checked(app, setp, mapping)
    ok, msg = check_free_limit(fp)
    print(f"\n  {msg}")
    print(f"  full free fingerprint (kHz): {['%.2f' % f for f in fp]}")
    if ok:
        print("  -> PASS: the model reproduces the clamped-free limit.")
    else:
        if len([f for f in fp if f and f > 1.0]) < 2:
            print("  -> FAIL: could not READ the free eigenfrequencies. This "
                  "says nothing about the model;")
            print("     the extractor could not find the array in "
                  "data/eigen. See the keys printed above.")
        else:
            print("  -> FAIL: the model does NOT reproduce the clamped-free "
                  "limit. This is a structural")
            print("     problem upstream of any contact parameter, and "
                  "everything downstream inherits it.")
        if not skip_free_check:
            raise AFeMulatorError(
                "free-lever check failed. Re-run with --skip-free-check to "
                "collect the ladder anyway, but treat the result as "
                "diagnostic rather than as a forward model.")
    manifest.setdefault("free_eigen_khz", fp)
    manifest["free_check_pass"] = bool(ok)
    manifest["free_check_msg"] = msg

    manifest.update(
        stl=stl or "(loaded in the AFeMulator UI, not via API)",
        probe=name, tilt_deg=TILT_DEG, z_drive_pm=Z_DRIVE_PM,
        k1_grid=[float(k) for k in K1_GRID],
        k1_range_rationale="Hertz 1310 N/m; mode-A shape fit 1000-2154; "
                           "EB reveal-loop fit 320-390. 100-5000 brackets all.",
        log_damping_grid=[float(v) for v in LOG_DAMPING],
        freq_window_khz=[FREQ_LO_KHZ, FREQ_HI_KHZ],
        quiet_params=QUIET,
        mapping=mapping, load_stl_response=load_info,
        params_at_start=app.get_params(),
        drive="mech (surface displacement); detection channel z_displacement",
        encoding={"note": "k1/k2 here are PHYSICAL N/m; the API was written "
                          "log10(k) via run_fem_sweep.encode()."})
    manifest.setdefault("runs", [])
    # An export counts as done only if the .h5 is still on disk. Trusting the
    # manifest alone means a file deleted as suspect is never recollected --
    # the manifest is append-only and remembers it forever.
    done, ghosts = set(), []
    for r in manifest["runs"]:
        if os.path.exists(os.path.join(out_dir, r["file"])):
            done.add(r["file"])
        else:
            ghosts.append(r["file"])
    if ghosts:
        print(f"  {len(ghosts)} manifest entr(y/ies) have no .h5 on disk and "
              f"will be RECOLLECTED:")
        for f in ghosts[:12]:
            print(f"    {f}")
        if len(ghosts) > 12:
            print(f"    ... and {len(ghosts) - 12} more")

    grid = K1_GRID if not limit else K1_GRID[:max(1, int(limit))]
    if start_at:
        grid = grid[int(start_at):]
        print(f"  --start-at {start_at}: beginning at k1 = {grid[0]:.4g} N/m")
    if limit:
        print(f"\n  --limit {limit}: running only the first {len(grid)} rungs "
              f"({grid[0]:.0f}-{grid[-1]:.0f} N/m) as a smoke test.")

    # Deterministic sweeps, and the app's own Lorentzian frequency window.
    set_extra(app, **QUIET)
    setp(f_low=FREQ_LO_KHZ, f_high=FREQ_HI_KHZ)

    total = len(lateral) * len(grid) * len(LOG_DAMPING)
    n, t0, rungs_done = 0, time.time(), 0
    for lat in lateral:
        k2_of = LATERAL[lat]
        for k1 in grid:
            klat = float(k2_of(k1))
            tag = f"{k1:.4g}".replace("+", "")
            names = {ld: f"fem_{name}_{lat}_mech_k1_{tag}_ld{ld:.2f}.h5"
                     for ld in LOG_DAMPING}
            if all(f in done and os.path.exists(os.path.join(out_dir, f))
                   for f in names.values()):
                n += len(LOG_DAMPING)
                print(f"[{n:3d}/{total}] k1={k1:8.1f} {lat:12s} -- already "
                      "exported, skipping")
                continue

            args = {"k_z": float(k1), "k_x": klat, "k_y": klat}
            if mapping.get("kz_zero"):
                args["kz_zero"] = False
            if mapping.get("kx_zero"):
                args["kx_zero"] = (klat == 0.0)
            if mapping.get("ky_zero"):
                args["ky_zero"] = (klat == 0.0)
            setp(**args)
            retry(lambda: app.compute("fea"), f"compute fea k1={k1:.4g}")
            f1 = flexural_mode1_khz(app)

            flags = dict(fa_surface=True, fa_beam_es=False,
                         fa_cone_es=False, fa_thermal=False)
            if mapping.get("z_disp"):
                flags["z_disp"] = Z_DRIVE_PM
            setp(**{k: v for k, v in flags.items() if mapping.get(k)})

            for ld in LOG_DAMPING:
                fn = names[ld]
                path = os.path.join(out_dir, fn)
                if fn in done and os.path.exists(path):
                    n += 1
                    continue
                set_extra(app, log_damping=float(ld))
                # Always a full FEA recompute after a damping change. An earlier
                # version probed once whether a cheap sweep-only recompute was
                # enough, but the probe set log_damping = 6.0 and re-swept
                # BETWEEN exports without restoring it, which scrambled the first
                # rung: its four levels came out non-monotonic in Q and two of
                # them byte-identical. Not worth the risk for the time it saves.
                retry(lambda: app.compute("fea"),
                      f"compute fea k1={k1:.4g} ld={ld}")
                if not dry_run:
                    retry(lambda: app.export(path), f"export {fn}")
                manifest["runs"].append(dict(
                    file=fn, probe=name, lateral=lat, drive="mech",
                    k1=float(k1), k2=klat, log_damping=float(ld),
                    mode1_khz=float(f1),
                    freq_window_khz=[FREQ_LO_KHZ, FREQ_HI_KHZ]))
                done.add(fn)
                n += 1
                el = time.time() - t0
                print(f"[{n:3d}/{total}] {fn}  f1={f1:7.2f} kHz  "
                      f"[{el/60:5.1f} min, ~{el/max(n,1)*(total-n)/60:4.0f} min "
                      f"left]", flush=True)
                with open(mpath, "w") as fh:
                    json.dump(manifest, fh, indent=2)
            if not dry_run:
                verify_rung(out_dir, names, k1)
            rungs_done = rungs_done + 1
            if pause_every and rungs_done % pause_every == 0:
                print(f"        pausing {pause_secs}s to let the tab catch up "
                      f"(--pause-every {pause_every})", flush=True)
                time.sleep(pause_secs)

    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\ndone: {n} exports in {(time.time()-t0)/60:.1f} min -> {out_dir}")
    print(f"manifest: {mpath}")
    if not manifest["free_check_pass"]:
        print("\nREMINDER: the free-lever check FAILED. Report that alongside "
              "any result from this ladder.")


def verify_rung(out_dir, names, k1):
    """Check the damping levels of one rung actually produced distinct Q.

    A damping change that silently fails to apply is invisible in the export --
    the file is written, the manifest looks complete, and the surrogate later
    interpolates across an axis that does not exist. So Q is measured back out
    of the files and required to be distinct and falling with damping.
    """
    qs = []
    for ld in sorted(names):
        q = _mode1_q(os.path.join(out_dir, names[ld]))
        qs.append((ld, q))
    have = [(ld, q) for ld, q in qs if q]
    if len(have) < 2:
        return                                   # h5py absent; nothing to check
    txt = "  ".join(f"ld{ld:.3g}:Q{q:.0f}" for ld, q in have)
    falling = all(b[1] < a[1] * 1.02 for a, b in zip(have, have[1:]))
    distinct = len({round(q, 1) for _, q in have}) == len(have)
    flag = "" if (falling and distinct) else "   <-- CHECK"
    print(f"        k1={k1:.4g}: {txt}{flag}")
    if not distinct:
        print("        WARNING: two damping levels gave the SAME Q -- the "
              "damping change is not being applied to every export.")
    elif not falling:
        print("        WARNING: Q is not monotonically falling with damping.")
    lo, hi = min(q for _, q in have), max(q for _, q in have)
    if not (lo <= 191 <= hi):
        print(f"        note: Q spans {lo:.0f}-{hi:.0f}, which does not bracket "
              "the measured 191 at this k1.")


def _mode1_q(path):
    """Mode-1 Q from an export, for the damping probe."""
    try:
        import h5py, numpy as _np
        with h5py.File(path, "r") as f:
            fr = f["spectrogram/frequency_hz"][:]
            A = abs(f["spectrogram/z_displacement/det_real"][:]
                    + 1j * f["spectrogram/z_displacement/det_imag"][:])
            eig = f["eigen/frequencies_hz"][:]
            mt = f["eigen/mode_types"][:]
        f1 = _np.sort(eig[mt == 1])[0]
        m = _np.abs(fr - f1) < 40e3
        fb, Ab = fr[m], A[:, m].max(0)
        j = int(Ab.argmax()); h = Ab[j] / _np.sqrt(2)
        lo = j
        while lo > 0 and Ab[lo] > h: lo -= 1
        hi = j
        while hi < len(Ab) - 1 and Ab[hi] > h: hi += 1
        return float(fb[j] / (fb[hi] - fb[lo])) if fb[hi] > fb[lo] else None
    except Exception:
        return None


def _export_bins(path):
    """Spectrogram length of an export, if h5py happens to be available."""
    try:
        import h5py
    except ImportError:
        return None
    try:
        with h5py.File(path, "r") as f:
            return int(f["spectrogram/frequency_hz"].shape[0])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(
        description="FEM ladder for the physics-informed AL study.")
    ap.add_argument("--stl", help="ABSOLUTE path on the AFeMulator machine")
    ap.add_argument("--name", default="Multi75G")
    ap.add_argument("--out", default="../data_ladder")
    ap.add_argument("--lateral", action="append", choices=sorted(LATERAL),
                    help="repeatable; default is both")
    ap.add_argument("--key-map", action="append", default=None,
                    metavar="need=api_key",
                    help="override a resolved parameter, e.g. k_z=log_kz")
    ap.add_argument("--probe", action="store_true",
                    help="print the resolved parameter map and exit")
    ap.add_argument("--dump-eigen", action="store_true",
                    help="load the STL, solve once, print the RAW data/eigen "
                         "response and exit. Use this if the eigenfrequency "
                         "extractor cannot find the array.")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip set_params read-back verification (not advised)")
    ap.add_argument("--dry-run", action="store_true",
                    help="drive the solver but write no .h5 files")
    ap.add_argument("--skip-free-check", action="store_true",
                    help="continue even if the clamped-free limit fails")
    ap.add_argument("--start-at", type=int, default=0, metavar="K",
                    help="skip the first K k1 rungs (chunk a long run)")
    ap.add_argument("--pause-every", type=int, default=0, metavar="R",
                    help="pause after every R rungs to let the browser tab "
                         "recover; try 4 if you keep hitting HTTP 408")
    ap.add_argument("--pause-secs", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="run only the first N k1 rungs. Use --limit 2 as a "
                         "~10 min smoke test before committing to the full run; "
                         "the results are kept and the full run resumes from "
                         "them.")
    ap.add_argument("--log-damping", default=None, metavar="A,B,C",
                    help="override the damping grid, e.g. "
                         "'4.6,4.775,4.95,5.125,5.3'. Use this when topping up "
                         "a directory collected with a wider grid: the manifest "
                         "header records whatever is passed here, and "
                         "build_fem_library.py trusts that header, so a "
                         "narrower grid silently discards the extra levels "
                         "already on disk.")
    a = ap.parse_args()

    if a.log_damping:
        global LOG_DAMPING
        LOG_DAMPING = [float(v) for v in a.log_damping.replace(" ", "").split(",")
                       if v]
        if len(set(LOG_DAMPING)) != len(LOG_DAMPING) or len(LOG_DAMPING) < 2:
            sys.exit(f"--log-damping needs >=2 distinct values, got "
                     f"{LOG_DAMPING}")
        LOG_DAMPING.sort()
        print(f"--log-damping: using {LOG_DAMPING} "
              f"({len(LOG_DAMPING)} levels/rung)")

    app = AFeMulator()
    try:
        print("ping:", app.ping())
    except AFeMulatorError as e:
        sys.exit(f"\ncannot reach AFeMulator.\n{e}\n\n"
                 "Start AFeMulator.exe and open http://127.0.0.1:8050 in a "
                 "VISIBLE browser tab -- the compute loop runs in that tab and "
                 "a backgrounded tab is throttled until requests never return.")

    params = app.get_params()
    mapping = api_map.resolve(params, api_map.parse_key_map(a.key_map))
    if a.probe:
        print(api_map.report(params, mapping))
        print("\nparameters this build exposes:")
        for k in sorted(api_map.flatten(params)):
            print("   ", k, "=", api_map.flatten(params)[k])
        print("\nIf a damping magnitude appears above (something other than "
              "the cvert_zero / clat_zero flags), tell me the key -- the "
              "surrogate wants damping as a third axis so it can match the "
              "measured Q = 191, and it is not in api_map.MEASURED.")
        return
    if a.dump_eigen:
        import pprint
        app.wait_ready(); app.set_autorun(False)
        if a.stl:
            info = app.load_stl(a.stl) or {}
            print("loaded:", info.get("path"), " n_faces=", info.get("n_faces"))
        if mapping.get("tilt"):
            Setter(app, mapping, True)(tilt=TILT_DEG)
        app.compute("full")
        ev = app.eigen()
        print("\n--- RAW /api/data/eigen ---")
        pprint.pprint(ev, depth=3, compact=True, width=100)
        print("\n--- numeric arrays found ---")
        for p, n in _numeric_arrays(ev):
            print(f"  {p:34s} len={len(n):4d}  first={[round(v,3) for v in n[:5]]}")
        try:
            fr, mt = eigen_khz(app)
            print(f"\nextractor -> {len(fr)} freqs (kHz): "
                  f"{[round(v,2) for v in fr[:8]]}")
            print(f"             mode types: "
                  f"{[int(round(t)) for t in mt[:8]] if mt else None}")
            print(f"             flexural  : "
                  f"{[round(v,2) for v in flexural_khz(app, 6)]}")
            print("\nIf the flexural list starts near 74.8 kHz and the second "
                  "entry is ~6.3x it, the readback is fixed.")
        except AFeMulatorError as e:
            print(f"\nextractor still failed: {e}")
        return

    missing = api_map.check(mapping)
    if missing:
        sys.exit(f"unresolved essential parameters: {missing}\n"
                 "Run with --probe, then pass --key-map need=api_key.")

    lateral = tuple(a.lateral) if a.lateral else ("frictionless", "isotropic")
    nrun = len(lateral) * (a.limit or K1_N) * len(LOG_DAMPING)
    print(f"\nladder: {K1_N} rungs, {K1_LO:g}-{K1_HI:g} N/m "
          f"({100*((K1_HI/K1_LO)**(1/(K1_N-1))-1):.1f} % per step)")
    print(f"lateral: {lateral}   drive: mech only")
    print(f"damping: log_damping {LOG_DAMPING} (Q ~ 473/299/188/119 at k1=1000; "
          f"measured 191)")
    print(f"{nrun} exports. Observed 20-40 s each while the tab is healthy, "
          f"so ~{nrun*0.5/60:.1f}-{nrun*1.0/60:.1f} h if it holds.\n"
          "If it dies with HTTP 408, reload the AFeMulator tab and re-run the "
          "same command -- it resumes.\n")
    run(app, a.stl, a.name, a.out, mapping, lateral,
        verify=not a.no_verify, dry_run=a.dry_run,
        skip_free_check=a.skip_free_check, limit=a.limit,
        start_at=a.start_at, pause_every=a.pause_every,
        pause_secs=a.pause_secs)


if __name__ == "__main__":
    main()
