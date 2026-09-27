r"""Same-hour validation set for the 12-position fast wideband run.

WHY THIS EXISTS
  The fast run (10:09-10:23 today, wideband_fast_12pos_100nN_0V) was scored
  against coarse_ref_5um_100nN_0V, taken 00:00-01:20 -- about nine hours earlier.
  Decomposing that comparison gave:

      reconstruction only ............ 0.116   (rank 8 from 12 columns)
      9 h instrument drift ........... 0.295   (the same 12 positions, measured
                                                twice, no reconstruction at all)
      live run vs reference .......... 0.320   (both, adding in quadrature)

  The drift term dominates and it is not a gain error -- CR1 moved -70 Hz, CR3
  -164 Hz, CR5 +444 Hz, and removing a global complex gain makes the agreement
  worse (0.295 -> 0.396). So the cross-capture number understates the method.

  This run measures 15 positions that the fast run never visited, within the same
  hour, so the fast run's rank-8 reconstruction can be scored against a
  contemporaneous ground truth and the drift floor drops out.

DESIGN
  15 positions, none of them in the 12-point design, minimum 5 um away from any
  of them, ~25 um pitch spanning 105-440 um, measured 440 -> 105 (the direction
  every other sweep used). ~1.15 min/position -> ~17 min.

  440, 415, 390, 365, 340, 320, 295, 270, 245, 220, 200, 175, 150, 125, 105

  All fifteen lie on the 5 um reference grid, so they are ALSO directly
  comparable to the 00:00 reference -- which gives a second, independent estimate
  of how much the instrument moved between then and now.

WHAT IT DOES NOT DO
  It measures nothing the reconstruction uses. The rank-8 fit stays exactly as it
  was, built from the 12 fast-run positions only. This set is held out by
  construction, not by subsetting after the fact.

RUN IT FROM THE KERNEL THAT HOLDS `inst`, ON THE MAIN THREAD:

    %run -i "C:/Users/Asylum User/Desktop/STAFF Software/Liam/ActiveModeMap/scripts/run_wideband_validation.py"

PRE-FLIGHT: same two unverifiable items as before -- the Igor image window with
spots 1 and 2 marked must still be open, and the wideband tune must be untouched
(1.1 MHz sweep, 47040 points). The fast run confirmed both at 10:23; if nothing
has been retuned or closed since, they still hold.

Safety: measure_conditions_at withdraws after every position. The finally block
resets bias to 0 V and withdraws; it deliberately does NOT call inst.close().
Checkpointed every position and resumable.
"""
import os
import sys
import time

import numpy as np

from activemodemap.series import make_conditions, run_series

OUT = r'D:\User Data\Liam\ActiveModeMap\DomainsBPPPCONTAU'
CKPT = os.path.join(OUT, 'wideband_valid_15pos_100nN_0V_checkpoint.npz')
LOG = os.path.join(OUT, 'wideband_valid_15pos_100nN_0V_log.txt')

X_FIXED = np.array([440.0, 415.0, 390.0, 365.0, 340.0, 320.0, 295.0, 270.0,
                    245.0, 220.0, 200.0, 175.0, 150.0, 125.0, 105.0])
LOAD_NN, BIAS_V, SPOT = 100.0, 0.0, 1

# the 12 positions the fast run measured -- this set must not touch any of them
FAST_X = np.array([445.0, 410.0, 380.0, 350.0, 315.0, 285.0,
                   255.0, 225.0, 190.0, 160.0, 130.0, 100.0])


class Tee:
    def __init__(self, path):
        self.f = open(path, 'a', buffering=1)
        self.out = sys.stdout          # the notebook stream, not sys.__stdout__

    def write(self, s):
        self.out.write(s); self.f.write(s)

    def flush(self):
        self.out.flush(); self.f.flush()


def main(inst):
    sep = float(np.min(np.abs(X_FIXED[:, None] - FAST_X[None, :])))
    if sep < 4.99:
        raise SystemExit(f"validation set is only {sep:.1f} um from a fast-run position -- "
                         "it would not be held out")
    cond = make_conditions([BIAS_V], [LOAD_NN], spots=[SPOT])
    t_start = time.time()
    print(f"\n=== WIDEBAND VALIDATION started {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"{X_FIXED.size} held-out positions {X_FIXED[0]:.0f} -> {X_FIXED[-1]:.0f} um "
          f"(~{abs(np.median(np.diff(X_FIXED))):.0f} um pitch), {LOAD_NN:.0f} nN, "
          f"{BIAS_V:+.1f} V, spot {SPOT}")
    print(f"min separation from any fast-run position: {sep:.1f} um (none shared)")
    print(f"instrument before: x={inst.current_x} um, spot={getattr(inst, 'current_spot', None)}, "
          f"load={inst.load_nN} nN, bias={inst.dc_bias_V:+.1f} V")
    print(f"checkpoint: {CKPT}")
    res = None
    try:
        res = run_series(inst, X_FIXED, cond, positions_um=X_FIXED,
                         checkpoint_path=CKPT, resume=True, verbose=True)
    finally:
        try:
            inst.set_dc_bias(0.0)
        except Exception as e:
            print('bias reset:', e)
        try:
            inst.a.withdraw()
        except Exception as e:
            print('withdraw:', e)
        mins = (time.time() - t_start) / 60.0
        print(f"=== WIDEBAND VALIDATION finished {time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"-- {mins:.1f} min (tip withdrawn, bias 0 V) ===")
    if res is not None:
        print(f"captured {len(res['x_um'])} positions x {len(res['freq_Hz'])} frequencies")
        print("Score the fast run's rank-8 reconstruction at these 15 positions: the "
              "9 h drift floor of 0.295 is gone, so what is left is the method.")
    return res


if __name__ == '__main__':
    if 'inst' not in globals():
        raise SystemExit("No `inst` in the namespace -- run with `%run -i` from the kernel that holds it.")
    _tee = Tee(LOG); _old = sys.stdout; sys.stdout = _tee
    try:
        result = main(globals()['inst'])
    finally:
        sys.stdout = _old; _tee.flush()
