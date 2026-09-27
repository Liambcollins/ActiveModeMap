r"""Stage 2b -- held-out validation for 20_wideband_fast.py, measured WITHIN THE HOUR.

Positions: the midpoints between consecutive fast-run positions (N_FAST - 1 of
them), so every one is half a pitch (~7 um) from any design position -- held out
by construction. Same spot, load, bias, tune window. Measured free end -> base.

Score afterwards: rank-4/5 Chebyshev reconstruction from the fast run, evaluated
here. Drift dominates any comparison against data more than an hour or two old,
which is why this runs immediately after 20.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scmpit_common as C
from activemodemap.series import make_conditions, run_series

inst = globals().get('inst')
P = C.load_preflight()
C.check_inst(inst, P)

LOAD_NN, BIAS_V, SPOT = 500.0, 0.0, 1
FAST_X = np.load(os.path.join(C.FILE_LOC, 'wideband_fast_design_x_um.npy'))
X_FIXED = np.round(0.5 * (FAST_X[:-1] + FAST_X[1:]), 1)          # midpoints, free end -> base
sep = float(np.min(np.abs(X_FIXED[:, None] - FAST_X[None, :])))
CKPT = os.path.join(C.FILE_LOC, f'wideband_valid_{X_FIXED.size}pos_500nN_0V_checkpoint.npz')
LOG = os.path.join(C.FILE_LOC, f'wideband_valid_{X_FIXED.size}pos_500nN_0V_log.txt')

_tee = C.Tee(LOG); _old = sys.stdout; sys.stdout = _tee
t0 = time.time()
res = None
try:
    if sep < 3.0:
        raise SystemExit(f'validation set only {sep:.1f} um from a design position -- not held out')
    cond = make_conditions([BIAS_V], [LOAD_NN], spots=[SPOT])
    C.stamp(f'=== STAGE 2b WIDEBAND VALIDATION  {X_FIXED.size} held-out positions '
            f'{X_FIXED[0]:.1f} -> {X_FIXED[-1]:.1f} um, min separation {sep:.1f} um, '
            f'{LOAD_NN:.0f} nN, {BIAS_V:+.1f} V, spot {SPOT} ===')
    C.stamp(f'checkpoint: {CKPT}')
    res = run_series(inst, X_FIXED, cond, positions_um=X_FIXED,
                     checkpoint_path=CKPT, resume=True, verbose=True)
    C.stamp(f"captured {len(res['x_um'])} positions x {len(res['freq_Hz'])} frequencies")
finally:
    C.finish(inst, t0, 'STAGE 2b WIDEBAND VALIDATION')
    sys.stdout = _old; _tee.close()

wideband_valid_res = res
