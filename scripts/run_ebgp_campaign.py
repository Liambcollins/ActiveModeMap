r"""EB+GP (physics-driven) active-learning campaign -- live instrument runner.

Counterpart to the rank-6 model-free run that produced
`sparse_capture_100nN_0V_checkpoint.npz`. Same grid, same condition, same spot,
same cap, so the two are position-matched and directly comparable; only the
selector differs.

RUN IT FROM THE KERNEL THAT ALREADY HOLDS `inst` (notebook 07's kernel):

    %run -i "C:/Users/Asylum User/Desktop/STAFF Software/Liam/ActiveModeMap/scripts/run_ebgp_campaign.py"

`-i` matters: the script uses the existing `inst` rather than building a second
AsylumInstrument. AsylumInstrument tracks the laser position in software from
relative moves, so constructing a fresh one after a run that left the laser
mid-span (it is at x = 75 um now, where the 250 nN survey finished) would offset
every subsequent move. Do not build a new instrument unless you pass
`x_start_um=inst.current_x`.

DO NOT run this in a worker thread. The Igor COM object was created on the kernel's
main thread; calling it from a new thread raises "CoInitialize has not been called"
(verified 2026-09-19 -- it failed on the first withdraw, before any motion). `%run -i`
executes on the main thread, which is what you want. The kernel is busy for the
whole campaign; progress is mirrored to the log file below so it can be followed
from outside the kernel (e.g. `tail -f` on the mounted folder).

PRE-FLIGHT -- two things this script cannot check for itself:
  1. The Igor image window with spots 1 and 2 marked must still be OPEN. The
     instrument is parked on spot 2; the dense map and rank-6 run were on spot 1,
     so the first position triggers GoToSpot(1). If that window was closed since
     the 250 nN survey, GoToSpot raises an Igor alert (which blocks) or silently
     does nothing -- and the campaign would then run on spot 2 while believing it
     is on spot 1. Check the window before launching.
  2. Wideband tune settings are as the dense map left them (verified 2026-09-19
     20:30: SweepWidth 1.1 MHz, 47040 points). If anything was retuned since, the
     spectra will not share the dense map's frequency grid.

Safety: the tip is withdrawn after every position by `measure_conditions_at`, so
the long posterior fits happen with the tip off the surface. The finally block
resets the bias to 0 V and withdraws; it deliberately does NOT call inst.close().

Timing (offline replay, 2026-09-19): first fit ~3.5 min cold; steps 4-5 also cold
(~3.5 min each, red_chi2 > 4 forces a rescan); from ~n=6 fits warm-start in
0-60 s and the EIG scan is ~10 s. Measurement is ~70 s/position. Expect ~60-70 min
for 30 points including the drift check. Every point is checkpointed, so it is
safe to interrupt -- the checkpoint holds everything measured up to that moment.
"""
import os
import sys
import time

import numpy as np

from activemodemap.ebgp_capture import run_ebgp_capture
from activemodemap.series import make_conditions, save_checkpoint

OUT = r'D:\User Data\Liam\ActiveModeMap\DomainsBPPPCONTAU'
CKPT = os.path.join(OUT, 'ebgp_capture_100nN_0V_checkpoint.npz')
LOG = os.path.join(OUT, 'ebgp_capture_100nN_0V_log.txt')
DRIFT_NPZ = os.path.join(OUT, 'ebgp_drift_check_100nN_0V.npz')
DENSE = os.path.join(OUT, 'dense_map_100nN_0V_checkpoint.npz')

X_GRID = np.arange(100.0, 446.0, 1.0)      # the dense map's own candidate grid
BAND_HZ = (50.2e3, 79.8e3)                 # CR1; drives selection only
MAX_POS = 30                               # matched to the rank-6 run
LOAD_NN = 100.0
BIAS_V = 0.0
SPOT = 1                                   # the dense map / rank-6 spot (inst is on 2)

DRIFT_FIRST = False                        # already done 19:49-19:53 -> ebgp_drift_check_100nN_0V.npz
DRIFT_XS = (150.0, 275.0, 400.0)
DRIFT_MAX_NRMSE = 0.25                     # above this the dense map is not a valid reference
FORCE = True                               # gate tripped at 0.353 (x=400, +117 Hz on a Q~130 line);
                                           # Liam's call 2026-09-19: run now, coarse reference after


class Tee:
    """Mirror stdout to a file.

    Captures the CURRENT sys.stdout at construction. Under ipykernel that is the
    notebook's OutStream; sys.__stdout__ would be the Jupyter server's console,
    and the cell would show nothing for an hour.
    """

    def __init__(self, path):
        self.f = open(path, 'a', buffering=1)
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def drift_check(inst, cond):
    """Re-measure a few dense-map positions and compare, before trusting the reference.

    The dense map is ~15 h old and CR1 already moved 63.47 -> 65.0 kHz between the
    Sep-17 and Sep-19 sweeps, so this is worth the ~4 minutes it costs. The first
    position here is also what moves the sample to spot 1.

    Returns (worst_in_band_NRMSE, rows). The re-measured spectra are saved to
    DRIFT_NPZ as the only same-session reference.
    """
    band = (55e3, 75e3)
    d = np.load(DENSE, allow_pickle=False)
    xg, fr = d['x_um'], d['freq_Hz']
    Zref = d['Z'][0]                       # load the 220 MB member ONCE
    del d
    mr = (fr >= band[0]) & (fr <= band[1])
    worst, rows, saved = 0.0, [], {}
    print('--- drift check against the dense map (spot 1, 100 nN, 0 V) ---')
    for xv in DRIFT_XS:
        j = int(np.argmin(np.abs(xg - xv)))
        zr = Zref[j]
        got = inst.measure_conditions_at(xv, cond, verbose=False)
        if got.get(0) is None:
            print(f'  x={xv:.0f}: FAILED')
            continue
        f, Z, _ = got[0]
        saved[xv] = (f, Z)
        m = (f >= band[0]) & (f <= band[1])
        jn = int(np.argmax(np.abs(Z[m])))
        jr = int(np.argmax(np.abs(zr[mr])))
        f_now, f_ref = f[m][jn], fr[mr][jr]
        amp_ratio = float(np.abs(Z[m])[jn] / np.abs(zr[mr])[jr])
        zi = (np.interp(fr[mr], f[m], Z[m].real) + 1j * np.interp(fr[mr], f[m], Z[m].imag))
        nr = float(np.sqrt(np.mean(np.abs(zi - zr[mr]) ** 2))
                   / np.sqrt(np.mean(np.abs(zr[mr]) ** 2)))
        worst = max(worst, nr)
        rows.append(dict(x=xv, f_now_Hz=float(f_now), f_ref_Hz=float(f_ref),
                         amp_ratio=amp_ratio, nrmse=nr))
        print(f'  x={xv:.0f}: {f_now/1e3:.3f} kHz (ref {f_ref/1e3:.3f}, {f_now-f_ref:+.0f} Hz) | '
              f'amp {amp_ratio:.3f}x | band NRMSE {nr:.3f}')
    if saved:
        np.savez_compressed(DRIFT_NPZ,
                            x_um=np.array(sorted(saved)),
                            freq_Hz=saved[sorted(saved)[0]][0],
                            Z=np.array([saved[k][1] for k in sorted(saved)]),
                            rows=str(rows))
        print(f'  saved {os.path.basename(DRIFT_NPZ)}')
    print(f'  worst band NRMSE {worst:.3f} (threshold {DRIFT_MAX_NRMSE})')
    return worst, rows


def main(inst):
    cond = make_conditions([BIAS_V], [LOAD_NN], spots=[SPOT])
    measured = {}
    out = None

    def measure_fn(x_um):
        got = inst.measure_conditions_at(x_um, cond, verbose=True)
        if got.get(0) is None:
            return None
        freq, Z = got[0][0], got[0][1]
        return freq, Z

    def on_point(step, x_um, freq, Z):
        # same checkpoint layout as the dense map and the rank-6 sparse capture
        measured[float(x_um)] = {0: (freq, Z)}
        save_checkpoint(CKPT, measured, cond, 0)

    print(f"\n=== EB+GP CAMPAIGN started {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"instrument before: x={inst.current_x} um, spot={getattr(inst, 'current_spot', None)}, "
          f"load={inst.load_nN} nN, bias={inst.dc_bias_V:+.1f} V")
    print(f"campaign: spot {SPOT}, {LOAD_NN:.0f} nN, {BIAS_V:+.1f} V, up to {MAX_POS} positions")
    print(f"checkpoint: {CKPT}")
    print(f"settings: DRIFT_FIRST={DRIFT_FIRST}, FORCE={FORCE}, DRIFT_MAX_NRMSE={DRIFT_MAX_NRMSE}")

    try:
        if DRIFT_FIRST:
            worst, _ = drift_check(inst, cond)
            if worst > DRIFT_MAX_NRMSE and not FORCE:
                print(f"\nDRIFT CHECK FAILED: worst band NRMSE {worst:.3f} > {DRIFT_MAX_NRMSE}. "
                      "The 15 h old dense map is not a valid reference for a position-matched "
                      "comparison. Not starting the campaign. Set FORCE=True to override, or "
                      "re-take a coarse dense reference first.")
                return None
            print("drift check passed; starting the campaign\n")
        out = run_ebgp_capture(measure_fn, X_GRID, band_Hz=BAND_HZ,
                               max_pos=MAX_POS, on_point=on_point, verbose=True)
    finally:
        try:
            inst.set_dc_bias(0.0)
        except Exception as e:
            print('bias reset:', e)
        try:
            inst.a.withdraw()
        except Exception as e:
            print('withdraw:', e)
        print(f"=== EB+GP CAMPAIGN finished {time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"(tip withdrawn, bias 0 V) ===")

    if out is None:
        return None
    xs = out['x_sel']
    print(f"\nEB+GP capture: {len(xs)} positions -> {[round(v, 1) for v in sorted(xs)]}")
    if out['blocked']:
        print(f"failed/blocked positions: {out['blocked']}")
    if len(xs) > 1:
        gaps = np.diff(sorted(xs))
        print(f"spread: min gap {gaps.min():.1f} um, median {np.median(gaps):.1f} um, "
              f"max {gaps.max():.1f} um")
        print("(the rank-6 run gave min 1.0, median 1.0, max 46.0 -- ~7 clusters)")
    fits = [h for h in out['history'] if 'setback_um' in h]
    if fits:
        last = fits[-1]
        print(f"final fit: setback {last['setback_um']:.2f} um, zeta {last['zeta']:.4f}, "
              f"red_chi2 {last['red_chi2']:.2f}, EB in-sample {last['crmse_eb_insample']:.4f}")
        print("NOTE: that residual is IN-SAMPLE. The number to compare against the "
              "rank-6 run's 0.319 is the out-of-sample error against the dense map, "
              "computed offline.")
    return out


if __name__ == '__main__':
    if 'inst' not in globals():
        raise SystemExit(
            "No `inst` in the namespace. Run this with `%run -i run_ebgp_campaign.py` "
            "from the kernel that holds the AsylumInstrument, so the laser position "
            "tracking stays correct.")
    _tee = Tee(LOG)
    _old = sys.stdout
    sys.stdout = _tee
    try:
        result = main(globals()['inst'])
    finally:
        sys.stdout = _old
        _tee.flush()
