r"""R2 extension, stage 4c -- bracketing mode-shape walk after the load work.

The load ladder and the bias-vs-load survey are only interpretable if the contact
state did not move while they ran. The ladder's own probe checks watch the free
end after every load; this closes the bracket properly by repeating the full
8-position walk at 0 V / 500 nN -- the same measurement the R2 pre-flight, the
dense map and the R2 close-out all made, so it drops straight into the existing
CR1(x) comparison.

If CR1 comes back near 295.4 kHz with a spread of a few hundred Hz, the whole
load extension sits on the same contact state as the rest of R2 and can be
reported alongside it. If it has moved, the ladder is still valid as a load
series but stops being comparable to the earlier stages, and the paper has to
say so.

8 positions x ~1.2 min ~ 10 min.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scmpit_common as C
from activemodemap.series import make_conditions, run_series

R2_ROOT = r'D:\User Data\Liam\ActiveModeMap\DomainsB_SCMPIT_R2'
OUT = os.path.join(R2_ROOT, '42_postcheck')

inst = globals().get('inst')
igor = inst.a.igor
if not os.path.isdir(OUT):
    raise SystemExit(f'output folder missing: {OUT}')
C.FILE_LOC = OUT
C.PREFLIGHT_JSON = os.path.join(R2_ROOT, '01_preflight', 'preflight_result.json')
inst.a.file_loc = OUT
inst.a.set_folder()

P = C.load_preflight()
C.check_inst(inst, P)

LOAD_NN, BIAS_V, SPOT, DRIVE_V = 500.0, 0.0, 1, 1.0
X_POS = np.round(np.linspace(P['x_hi_um'], P['x_lo_um'], 8), 1)          # free end -> base
CKPT = os.path.join(C.FILE_LOC, 'postcheck_walk_8pos_checkpoint.npz')
LOG = os.path.join(C.FILE_LOC, 'postcheck_r2_log.txt')
CR1_WIN = (255e3, 330e3)

_tee = C.Tee(LOG); _old = sys.stdout; sys.stdout = _tee
t0 = time.time()
res = None
try:
    C.stamp('=' * 78)
    C.stamp(f'R2 EXT STAGE 4c  POST-LOAD MODE-SHAPE WALK, {X_POS.size} positions '
            f'{X_POS[0]:.0f} -> {X_POS[-1]:.0f} um, {LOAD_NN:.0f} nN, {BIAS_V:+.1f} V, spot {SPOT}')
    C.stamp(f'data -> {C.FILE_LOC}')
    C.stamp(f'reference: R2 pre-flight CR1 {P["cr1_Hz"] / 1e3:.2f} kHz, '
            f'R2 dense map 295.35 +- 0.15 kHz, R2 close-out walk 295.58 (spread 0.20)')
    C.stamp('=' * 78)
    igor.Execute(f'PV("DriveAmplitude", {DRIVE_V})'); time.sleep(0.5)
    C.stamp(f'DriveAmplitude reads {inst.a.get_gmv().get("DriveAmplitude")}')
    cond = make_conditions([BIAS_V], [LOAD_NN], spots=[SPOT])
    res = run_series(inst, X_POS, cond, positions_um=X_POS,
                     checkpoint_path=CKPT, resume=True, verbose=True)

    F = np.asarray(res['freq_Hz'], float)
    Z = np.asarray(res['Z'])[0]
    x = np.asarray(res['x_um'], float)
    o = np.argsort(x); x, Z = x[o], Z[o]
    m = (F >= CR1_WIN[0]) & (F <= CR1_WIN[1])
    f_pk, a_pk = [], []
    for j in range(x.size):
        a = np.abs(Z[j, m]); k = int(np.argmax(a))
        f_pk.append(float(F[m][k])); a_pk.append(float(a[k]))
    f_pk = np.array(f_pk)
    C.stamp('')
    for xx, ff, aa in zip(x, f_pk, a_pk):
        C.stamp(f'   x {xx:7.1f} um   CR1 {ff / 1e3:8.2f} kHz   |Z| {aa * 1e3:8.4f} mV')
    C.stamp('')
    C.stamp(f'CR1 mean {f_pk.mean() / 1e3:.2f} kHz, spread {(f_pk.max() - f_pk.min()) / 1e3:.2f} kHz')
    drift = (f_pk.mean() - 295.35e3) / 295.35e3
    C.stamp(f'vs the R2 dense map mean: {100 * drift:+.2f} %  '
            + ('-- contact state UNCHANGED, the load extension is comparable to the rest of R2'
               if abs(drift) < 0.01 else
               '-- *** CONTACT STATE MOVED: the load data is still valid but is NOT '
               'directly comparable to the earlier R2 stages ***'))
    C.stamp('=' * 78)
except Exception as e:
    C.stamp(f'*** FAILED: {type(e).__name__}: {e}')
    import traceback; traceback.print_exc(); raise
finally:
    C.finish(inst, t0, 'R2 EXT STAGE 4c POST-LOAD WALK')
    sys.stdout = _old; _tee.close()

postcheck_res = res
