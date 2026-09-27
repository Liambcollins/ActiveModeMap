r"""Fastest live wideband transfer-function recovery -- 12 fixed positions.

THE CLAIM THIS RUN SUPPORTS
  The full cantilever wideband transfer function (100 Hz - 1.1 MHz, 47040 bins,
  all five contact resonances CR1-CR5) recovered from 12 measured positions in
  ~14 minutes, beating the published N=14 / rank 15 benchmark on every metric
  and the 346-position dense map (410.6 min) by ~29x in wall clock.

WHY THERE IS NO ACTIVE LEARNING HERE
  Active learning was the right tool for the CR1-band physics fit and the wrong
  one for this claim. Measured offline against the same-session 5 um reference
  (70 positions, held-out scoring, wideband_heldout/robust/floor.json):

    design        N  rank | wideband  dB med  <3dB  phase
    equispaced   12    8  |   0.108    0.27   0.99   1.8 deg
    rank6-live   17    8  |   0.105    0.24   0.99   1.7 deg
    EIG-live     18    9  | 448.2     20.90   0.34  56.4 deg   <- unusable

  EIG positions cluster where the CR1 Jacobian is informative; low-rank
  Chebyshev reconstruction needs SPREAD, so the design matrix goes singular and
  the wideband reconstruction diverges. Equispaced is not a compromise here, it
  is the optimum -- and it costs zero seconds of in-loop computation, which is
  where the EB+GP campaign spent 3.5 h on 30 points.

DESIGN (validated, phasing-robust: offsets 0/1/2 give 0.108 / 0.114 / 0.123)
  12 positions, ~30 um pitch, 445 -> 100 um, the direction the dense map used.
  Rank 8 is the knee: rank 7 gives 0.17, rank 9 gives no improvement, and rank 8
  holds from N=11 up. N=10 still works but becomes phasing-sensitive (0.13-0.17).

RECONSTRUCTION IS FREE AND HAPPENS AFTERWARDS
  reconstruct_map() at rank 8 is an SVD least-squares over a Chebyshev basis:
  no fitting, no model, ~0.1 s for all 47040 frequency bins. Nothing is computed
  between positions, so wall clock is exactly 12 x measurement time.

RUN IT FROM THE KERNEL THAT HOLDS `inst`, ON THE MAIN THREAD:

    %run -i "C:/Users/Asylum User/Desktop/STAFF Software/Liam/ActiveModeMap/scripts/run_wideband_fast.py"

PRE-FLIGHT -- the same two things this script cannot check for itself:
  1. The Igor image window with spots 1 and 2 marked must still be OPEN, or
     GoToSpot(1) blocks on an alert or silently no-ops and the run happens on
     whatever spot the stage is already on.
  2. Wideband tune must be as the dense map left it: SweepWidth 1.1 MHz,
     47040 points. Anything retuned since breaks the shared frequency grid.

Safety: measure_conditions_at withdraws after every position. The finally block
resets bias to 0 V and withdraws; it deliberately does NOT call inst.close().
Checkpointed every position and resumable, so it is safe to interrupt.
"""
import os
import sys
import time

import numpy as np

from activemodemap.series import make_conditions, run_series

OUT = r'D:\User Data\Liam\ActiveModeMap\DomainsBPPPCONTAU'
CKPT = os.path.join(OUT, 'wideband_fast_12pos_100nN_0V_checkpoint.npz')
LOG = os.path.join(OUT, 'wideband_fast_12pos_100nN_0V_log.txt')

# the validated N=12 design, measured 445 -> 100 um (same direction as the dense map)
X_FIXED = np.array([445.0, 410.0, 380.0, 350.0, 315.0, 285.0,
                    255.0, 225.0, 190.0, 160.0, 130.0, 100.0])
RANK = 8                       # for the offline reconstruction; nothing uses it in the loop
LOAD_NN, BIAS_V, SPOT = 100.0, 0.0, 1


class Tee:
    def __init__(self, path):
        self.f = open(path, 'a', buffering=1)
        self.out = sys.stdout          # the notebook stream, not sys.__stdout__

    def write(self, s):
        self.out.write(s); self.f.write(s)

    def flush(self):
        self.out.flush(); self.f.flush()


def main(inst):
    cond = make_conditions([BIAS_V], [LOAD_NN], spots=[SPOT])
    t_start = time.time()
    print(f"\n=== WIDEBAND FAST started {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"{X_FIXED.size} fixed positions {X_FIXED[0]:.0f} -> {X_FIXED[-1]:.0f} um "
          f"(~{abs(np.median(np.diff(X_FIXED))):.0f} um pitch), {LOAD_NN:.0f} nN, "
          f"{BIAS_V:+.1f} V, spot {SPOT}")
    print(f"reconstruct offline at rank {RANK}; nothing is computed between positions")
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
        print(f"=== WIDEBAND FAST finished {time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"-- {mins:.1f} min wall clock (tip withdrawn, bias 0 V) ===")
    if res is not None:
        n, nf = len(res['x_um']), len(res['freq_Hz'])
        print(f"captured {n} positions x {nf} frequencies "
              f"({(time.time()-t_start)/60/max(n,1):.2f} min/position)")
        print("THIS IS THE PAPER NUMBER: wall clock above vs 410.6 min for the "
              "346-position dense map.")
    return res


if __name__ == '__main__':
    if 'inst' not in globals():
        raise SystemExit("No `inst` in the namespace -- run with `%run -i` from the kernel that holds it.")
    _tee = Tee(LOG); _old = sys.stdout; sys.stdout = _tee
    try:
        result = main(globals()['inst'])
    finally:
        sys.stdout = _old; _tee.flush()
