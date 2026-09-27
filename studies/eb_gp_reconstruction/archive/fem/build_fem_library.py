#!/usr/bin/env python3
"""Turn the FEM ladder into a surrogate library for the AL replay.

Consumes what run_fem_ladder.py exports and writes a single .npz shaped
(k1, damping, u, position) -- exactly what physrec.py already loads for EB, so
the FEM arm drops into the existing reveal loop with no other changes.

Two things this does that a naive loader would get wrong:

1. RESONANCE-ALIGN before interpolating.  Each rung is resampled onto
   u = f / f_res(k1, damping) so that interpolating across k1 does not mix peaks
   sitting at different frequencies.  Skipping this step cost 3.8 % rms and 42 %
   worst-case on the EB library; with it the same library validated to 0.1 %.
   The app's Lorentzian frequency grid is non-uniform (122 Hz near mode 1,
   12 kHz far from it), which is handled by interpolating on the raw grid.

2. MAP THE POSITION AXIS.  AFeMulator reports x relative to the CONTACT point
   (-219.6 -> +15.8 um on a 240.76 um beam with contact at 225.0).  The measured
   lever is 225 um with the tip set back 10.9 um, so its contact is at 214.1 um.
   The two are put on a common footing by eta = x_from_clamp / contact_x, which
   pins the contact point at eta = 1 in both frames.  The residual mismatch is
   the overhang fraction -- 6.5 % (model) vs 4.8 % (ours) -- which is a caveat to
   report, not something this script can remove.

    python build_fem_library.py --data ../data_ladder --name Multi75G \
        --lateral frictionless --out femlib.npz
"""
# Paths are resolved through config.py -- set AMM_* environment
# variables or edit that file to point at your data.
import sys as _sys, os as _os
_D = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_D, 'config.py')):
    _D = _os.path.dirname(_D)
_sys.path[:0] = [_D, _os.path.join(_D, 'src'),
                 _os.path.join(_D, 'figures')]
from config import DENSE_GRID_A, FEM_LADDER, EB_GEOMETRY, OUT, FIG, add_eb_to_path



import argparse, json, os, sys
import numpy as np

try:
    import h5py
except ImportError:
    sys.exit("h5py required: pip install h5py")

MEAS_CX_UM = 214.1        # our lever: 225 um long, tip set back 10.9 um
U_LO, U_HI, U_N = 0.55, 2.05, 1700
MODE1_WINDOW_KHZ = (150.0, 600.0)


def load_export(path, chan):
    with h5py.File(path, "r") as f:
        fr = f["spectrogram/frequency_hz"][:]
        xf = f["spectrogram/position_x_um"][:]
        A = np.abs(f[f"spectrogram/{chan}/det_real"][:]
                   + 1j * f[f"spectrogram/{chan}/det_imag"][:])
        eig = f["eigen/frequencies_hz"][:]
        mt = f["eigen/mode_types"][:] if "eigen/mode_types" in f else None
    return fr, xf, A, eig, mt


def refine_peak(fr, prof, lo_hz, hi_hz):
    """Parabolically refined peak of `prof` inside a frequency window."""
    m = (fr >= lo_hz) & (fr <= hi_hz)
    if m.sum() < 3:
        return float("nan")
    f, p = fr[m], prof[m]
    j = int(np.argmax(p))
    if 0 < j < len(p) - 1:
        y0, y1, y2 = p[j-1], p[j], p[j+1]
        den = y0 - 2*y1 + y2
        if abs(den) > 1e-30:
            return float(f[j] + 0.5 * (y0 - y2) / den * (f[j] - f[j-1]))
    return float(f[j])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="Multi75G")
    ap.add_argument("--lateral", default="frictionless")
    ap.add_argument("--chan", default="z_displacement",
                    help="detection channel; displacement beat slope by 4-36x "
                         "on this instrument, so do not change without reason")
    ap.add_argument("--x-meas", default=None,
                    help=".npy of the 101 calibrated measured positions (um "
                         "from clamp). Defaults to a 123.31-225.75 um ramp.")
    ap.add_argument("--contact-x-model", type=float, default=225.0)
    ap.add_argument("--out", default="femlib.npz")
    ap.add_argument("--min-rungs", type=int, default=8, metavar="N",
                    help="refuse to build from fewer than N complete k1 rungs "
                         "(default 8). Lower it only to smoke-test the "
                         "pipeline on a partial ladder.")
    a = ap.parse_args()

    man = json.load(open(os.path.join(
        a.data, f"fem_ladder_manifest_{a.name}.json")))
    if not man.get("free_check_pass", True):
        print("WARNING: this ladder was collected with a FAILING free-lever "
              "check.\n  " + man.get("free_check_msg", "") +
              "\n  Treat anything built from it as diagnostic.\n")

    runs = [r for r in man["runs"] if r["lateral"] == a.lateral]
    if not runs:
        sys.exit(f"no runs with lateral={a.lateral!r} in the manifest")

    # --- stale-run guard -------------------------------------------------
    # data_ladder accumulates across runs, and the manifest is appended to,
    # never rewritten. Only the damping levels of the CURRENT grid are
    # trusted; levels left over from an earlier grid are dropped, otherwise
    # `complete` below would demand levels that only the oldest rungs have.
    grid = [round(float(d), 6) for d in man.get("log_damping_grid", [])]
    if grid:
        extra = sorted({round(r["log_damping"], 6) for r in runs} - set(grid))
        if extra:
            print(f"dropping {len(extra)} damping level(s) not in the current "
                  f"grid (left over from an earlier run): {extra}")
        runs = [r for r in runs if round(r["log_damping"], 6) in set(grid)]
        lds = sorted(grid)
    else:
        lds = sorted({round(r["log_damping"], 6) for r in runs})

    k1s = sorted({round(r["k1"], 6) for r in runs})
    # later manifest entries win, so a re-collected export supersedes the
    # original without needing the old line removed
    idx = {}
    for r in runs:
        idx[(round(r["k1"], 6), round(r["log_damping"], 6))] = r
    missing_file = [k for k, r in idx.items()
                    if not os.path.exists(os.path.join(a.data, r["file"]))]
    if missing_file:
        print(f"dropping {len(missing_file)} manifest entr(y/ies) whose .h5 is "
              f"gone (deleted as suspect): "
              f"{sorted((round(k), d) for k, d in missing_file)}")
        for k in missing_file:
            del idx[k]

    complete = [k for k in k1s if all((k, d) in idx for d in lds)]
    if len(complete) < len(k1s):
        miss = sorted(set(k1s) - set(complete))
        print(f"skipping {len(miss)} rungs missing a damping level: "
              f"{[round(k) for k in miss]}")
    k1s = complete
    if len(k1s) < a.min_rungs:
        sys.exit(f"only {len(k1s)} complete rungs -- too few to interpolate "
                 f"(need {a.min_rungs}; --min-rungs lowers the bar for a "
                 f"smoke test on a partial ladder)")

    x_meas = (np.load(a.x_meas) if a.x_meas
              else np.linspace(123.31, 225.75, 101))
    eta = x_meas / MEAS_CX_UM
    x_model_um = eta * a.contact_x_model

    UG = np.linspace(U_LO, U_HI, U_N)
    LOGU = np.zeros((len(k1s), len(lds), U_N, len(x_model_um)), dtype=np.float32)
    RES = np.zeros((len(k1s), len(lds)))
    rows, worst_nan = [], 0.0

    for i, k1 in enumerate(k1s):
        for j, ld in enumerate(lds):
            r = idx[(k1, ld)]
            fr, xf, A, eig, mt = load_export(
                os.path.join(a.data, r["file"]), a.chan)
            x_clamp = a.contact_x_model + xf
            o = np.argsort(x_clamp)
            x_clamp, A = x_clamp[o], A[o]

            fres = refine_peak(fr, A.max(0),
                               MODE1_WINDOW_KHZ[0]*1e3, MODE1_WINDOW_KHZ[1]*1e3)
            if not np.isfinite(fres):
                sys.exit(f"{r['file']}: no mode-1 peak in "
                         f"{MODE1_WINDOW_KHZ} kHz")
            RES[i, j] = fres

            target = UG * fres
            inside = (target >= fr.min()) & (target <= fr.max())
            worst_nan = max(worst_nan, float(np.mean(~inside)))
            # interpolate each position onto the aligned frame, in log amplitude
            block = np.full((len(x_clamp), U_N), np.nan)
            for p in range(len(x_clamp)):
                block[p, inside] = np.interp(target[inside], fr,
                                             np.maximum(A[p], 1e-300))
            for u in range(U_N):
                col = block[:, u]
                LOGU[i, j, u] = (np.log(np.interp(x_model_um, x_clamp, col))
                                 if np.all(np.isfinite(col)) else np.nan)
            rows.append((k1, ld, fres, _q(fr, A.max(0), fres)))

    print(f"\n{'k1 (N/m)':>9s} {'log_damp':>9s} {'f_res kHz':>10s} {'Q':>7s}")
    for k1, ld, fres, q in rows:
        print(f"{k1:9.1f} {ld:9.2f} {fres/1e3:10.2f} "
              f"{q if q else float('nan'):7.0f}")

    mono = bool(np.all(np.diff(RES.mean(1)) > 0))
    print(f"\nf_res monotonic in k1: {mono}  "
          f"({RES.mean(1)[0]/1e3:.1f} -> {RES.mean(1)[-1]/1e3:.1f} kHz)")
    # Q must fall as damping rises, within every rung. A rung that violates
    # this was collected with a corrupted damping value (e.g. a probe that set
    # log_damping and never restored it) -- f_res cannot detect that, since
    # damping does not move the resonance.
    bad_q = []
    for i, k1 in enumerate(k1s):
        qr = [q for (kk, ld, fr_, q) in rows if kk == k1]
        if len(qr) == len(lds) and all(qr) and \
           any(b >= a_ for a_, b in zip(qr, qr[1:])):
            bad_q.append((round(k1), [round(q) for q in qr]))
    if bad_q:
        print(f"\nWARNING: Q is not strictly decreasing with damping on "
              f"{len(bad_q)} rung(s) -- suspect a contaminated export:")
        for k1, qr in bad_q:
            print(f"  k1={k1:5d}  Q={qr}")
    else:
        print("\nQ strictly decreasing with damping on every rung: True")

    qs = [q for *_, q in rows if q]
    if qs:
        print(f"Q range across the library: {min(qs):.0f} - {max(qs):.0f}   "
              f"(measured 191 -- must be bracketed)")
        if not (min(qs) <= 191 <= max(qs)):
            print("  WARNING: the measured Q is OUTSIDE the library range. "
                  "Shift LOG_DAMPING in run_fem_ladder.py and re-run.")
    if worst_nan > 0.02:
        print(f"WARNING: up to {100*worst_nan:.1f}% of the u grid falls outside "
              "the exported frequency window.")
    if not mono:
        print("WARNING: f_res is not monotonic in k1. Resonance-aligned "
              "interpolation assumes it is -- inspect the table above.")

    # GG holds the PHYSICAL damping (N s/m^3), not log10 of it, so that
    # physrec's LGG = ln(GG) axis means the same thing for FEM as it does for
    # EB and a fitted log_g is comparable between the two arms.
    GG = 10.0 ** np.array(lds)
    free = man.get("free_eigen_khz") or []
    np.savez_compressed(
        a.out, K1=np.array(k1s), GG=GG, log10_damping=np.array(lds), UG=UG,
        RES=RES, RES_K=RES.mean(1), logamp_u=LOGU,
        x_model=x_model_um, x_meas=x_meas, eta=eta,
        F_REF=float(free[0] * 1e3) if free else float("nan"),
        lateral=a.lateral, chan=a.chan, source=os.path.abspath(a.data),
        free_check_pass=bool(man.get("free_check_pass", True)),
        note="FEM surrogate in the resonance-aligned frame u=f/f_res. Axes "
             "(k1, log_damping, u, position). GG is physical damping in "
             "N s/m^3 (log10_damping holds the grid as collected). F_REF is "
             "the FEM free-lever f1 in Hz, for f_free_of().")
    print(f"\nwrote {a.out}: {LOGU.shape} = (k1, log_damping, u, position)")
    print("load it with physrec.use_library('<this file>') -- it is already in "
          "the resonance-aligned frame, so the reveal loop runs against FEM "
          "with no other change.")


def _q(fr, prof, fres):
    m = np.abs(fr - fres) < 40e3
    if m.sum() < 5:
        return None
    fb, Ab = fr[m], prof[m]
    j = int(Ab.argmax()); h = Ab[j] / np.sqrt(2)
    lo = j
    while lo > 0 and Ab[lo] > h: lo -= 1
    hi = j
    while hi < len(Ab) - 1 and Ab[hi] > h: hi += 1
    return float(fb[j] / (fb[hi] - fb[lo])) if fb[hi] > fb[lo] else None


if __name__ == "__main__":
    main()
