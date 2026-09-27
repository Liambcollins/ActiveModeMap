r"""Stage 2a -- live wideband capture on the SCM-PIT probe: N_FAST equispaced positions.

Protocol from wideband-live-12pos-fast-recovery-2026-09-20, budget shrunk for the
easy SCM-PIT case (two/three modes, short span, no interior higher-mode nodes;
r = 4 from 8 positions gave 99 % within 3 dB offline). Equispaced over the
reachable span, measured free end -> base, spot 1, 500 nN, 0 V. NO active
learning, NOTHING computed between positions; reconstruct afterwards at rank 4-5
(escalate N only if 21_wideband_validation.py's held-out score demands it).

Tune window: whatever the panel holds (2 MHz, 10 s). The validation run must use
the same window -- do not retune between 20 and 21.
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

N_FAST = 8
LOAD_NN, BIAS_V, SPOT = 500.0, 0.0, 1
X_FIXED = np.round(np.linspace(P['x_hi_um'], P['x_lo_um'], N_FAST), 1)   # free end -> base
CKPT = os.path.join(C.FILE_LOC, f'wideband_fast_{N_FAST}pos_500nN_0V_checkpoint.npz')
LOG = os.path.join(C.FILE_LOC, f'wideband_fast_{N_FAST}pos_500nN_0V_log.txt')
np.save(os.path.join(C.FILE_LOC, 'wideband_fast_design_x_um.npy'), X_FIXED)

_tee = C.Tee(LOG); _old = sys.stdout; sys.stdout = _tee
t0 = time.time()
res = None
try:
    cond = make_conditions([BIAS_V], [LOAD_NN], spots=[SPOT])
    C.stamp(f'=== STAGE 2a WIDEBAND FAST  {X_FIXED.size} positions {X_FIXED[0]:.1f} -> {X_FIXED[-1]:.1f} um '
            f'(pitch {abs(np.median(np.diff(X_FIXED))):.1f} um), {LOAD_NN:.0f} nN, {BIAS_V:+.1f} V, spot {SPOT} ===')
    C.stamp(f'instrument before: x={inst.current_x} um, spot={inst.current_spot}')
    C.stamp(f'checkpoint: {CKPT}')
    res = run_series(inst, X_FIXED, cond, positions_um=X_FIXED,
                     checkpoint_path=CKPT, resume=True, verbose=True)
    n, nf = len(res['x_um']), len(res['freq_Hz'])
    C.stamp(f'captured {n} positions x {nf} frequencies ({(time.time() - t0) / 60 / max(n, 1):.2f} min/position)')
finally:
    C.finish(inst, t0, 'STAGE 2a WIDEBAND FAST')
    sys.stdout = _old; _tee.close()

wideband_fast_res = res
