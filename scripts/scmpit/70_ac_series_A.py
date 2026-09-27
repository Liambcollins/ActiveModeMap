r"""Stage 7 -- AC drive-amplitude series at position A, both domains, two biases.

Liam asked for more data points on the drive-linearity curve. Stages 5/6 only
had three anchors at position A (1 V, 2 V from stage 5; 30 mV from stage 6, and
that one didn't resolve per-domain because it was a full raster image). This
does real spectroscopy (measure_conditions_at, same method as the stage-1 bias
survey) at the two marked sample spots (spot 1 / spot 2 = the two domains),
at the SAME laser position A (154.9 um, E ~ 200, good SNR down to 30 mV per
stage 6), sweeping V_ac from 5 mV to 2 V (15 points, log-spaced), at both 0 V
(to get the AC-linearity curve itself) and the stage-5 extrapolated null
+1.33 V (to check the null holds at every drive level, not just 30 mV and 1 V).

V_ac is NOT a `Condition` field (series.py only has bias/load/spot) -- it's set
directly via PV() before each drive level's pass, using the SAME run_series /
measure_conditions_at path as every other survey so the checkpoint format,
InvOLS handling, and standing safety rules are unchanged. One checkpoint per
V_ac (like the load ladder saved one per load).

15 V_ac x 4 conditions (2 spots x 2 biases) x ~24 s/condition (stage-1 rate)
~ 24 min + moves/tunes. Standing rules: finally -> bias 0, withdraw.
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scmpit_common as C
from activemodemap.series import make_conditions, run_series

inst = globals().get('inst'); igor = inst.a.igor
P = C.load_preflight(); C.check_inst(inst, P)

X_A = 154.9
LOAD_NN = 500.0
V_CPD_TRUE = 1.33
VAC_LIST = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
LOG = os.path.join(C.FILE_LOC, 'ac_series_A_log.txt')
_tee = C.Tee(LOG); _old = sys.stdout; sys.stdout = _tee
say = C.stamp


def pv(k, v): igor.Execute(f'PV("{k}", {v})')


def verify_drive(vac, tol=0.01):
    gmv = inst.a.get_gmv()
    got = gmv.get('DriveAmplitude')
    ok = got is not None and abs(got - vac) <= max(tol * vac, 1e-4)
    say(f'    DriveAmplitude asked {vac:.4g}  reads {got}  {"ok" if ok else "** MISMATCH"}')
    return ok


def ckpt(vac):
    return os.path.join(C.FILE_LOC, f'ac_series_A_vac{vac*1000:07.2f}mV_checkpoint.npz')


t_start = time.time()
results = {}
try:
    C.stamp('=' * 78)
    C.stamp(f'STAGE 7  AC DRIVE SERIES AT POSITION A ({X_A} um): '
            f'{len(VAC_LIST)} Vac levels x 2 spots x 2 biases (0 V, {V_CPD_TRUE} V)')
    C.stamp(f'Vac list (V): {VAC_LIST}')
    C.stamp('=' * 78)
    cond = make_conditions([0.0, V_CPD_TRUE], [LOAD_NN], spots=[1, 2])
    for vac in VAC_LIST:
        say(f'\n--- Vac = {vac*1000:.1f} mV ---')
        pv('DriveAmplitude', vac); time.sleep(0.5)
        if not verify_drive(vac):
            say('    retrying drive-amplitude set once...')
            pv('DriveAmplitude', vac); time.sleep(0.5)
            verify_drive(vac)
        t_v = time.time()
        results[vac] = run_series(inst, [X_A], cond, positions_um=[X_A],
                                   checkpoint_path=ckpt(vac), resume=True, verbose=True)
        say(f'Vac {vac*1000:.1f} mV done in {(time.time() - t_v) / 60:.1f} min -> {ckpt(vac)}')
    say(f'\nAC series complete: {len(results)} drive levels x 4 conditions')
except Exception as e:
    say(f'*** FAILED: {type(e).__name__}: {e}')
    import traceback; traceback.print_exc(); raise
finally:
    C.finish(inst, t_start, 'STAGE 7 AC DRIVE SERIES')
    sys.stdout = _old; _tee.close()

ac_series_res = results
