r"""Stage 3 -- dense reference map at 500 nN, 1 um pitch over the reachable span.

Reference for scoring stages 2 and 4. Same code path as run_coarse_reference.py:
run_series with a fixed design, no convergence stop, checkpointed every position
and resumable. Free end -> base, spot 1, 0 V. ~1.15 min/position, so a ~100 um
span is ~100 positions ~ 2 h. Runs unattended; the finally block parks the tip.

Pitch: 1 um is safe on the DoLDMove dead-band (06_reanalysis: 0.5 um is not).
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

PITCH_UM = 1.0
LOAD_NN, BIAS_V, SPOT = 500.0, 0.0, 1
X_FIXED = np.round(np.arange(P['x_hi_um'], P['x_lo_um'] - 1e-9, -PITCH_UM), 1)   # free end -> base
CKPT = os.path.join(C.FILE_LOC, 'dense_map_1um_500nN_0V_checkpoint.npz')
LOG = os.path.join(C.FILE_LOC, 'dense_map_1um_500nN_0V_log.txt')

_tee = C.Tee(LOG); _old = sys.stdout; sys.stdout = _tee
t0 = time.time()
res = None
try:
    cond = make_conditions([BIAS_V], [LOAD_NN], spots=[SPOT])
    C.stamp(f'=== STAGE 3 DENSE MAP  {X_FIXED.size} positions {X_FIXED[0]:.0f} -> {X_FIXED[-1]:.0f} um '
            f'at {PITCH_UM:.0f} um, {LOAD_NN:.0f} nN, {BIAS_V:+.1f} V, spot {SPOT}  '
            f'(~{X_FIXED.size * 1.15:.0f} min) ===')
    C.stamp(f'instrument before: x={inst.current_x} um, spot={inst.current_spot}')
    C.stamp(f'checkpoint: {CKPT}')
    res = run_series(inst, X_FIXED, cond, positions_um=X_FIXED,
                     checkpoint_path=CKPT, resume=True, verbose=True)
    C.stamp(f"dense map: {len(res['x_um'])} positions x {len(res['freq_Hz'])} frequencies")
finally:
    C.finish(inst, t0, 'STAGE 3 DENSE MAP')
    sys.stdout = _old; _tee.close()

dense_map_res = res
