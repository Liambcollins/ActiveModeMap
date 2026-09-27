r"""R2 stage 7 -- AC drive-amplitude series at position A, both domains, two biases.

Identical design to R1 stage 7, with V_cpd read from `vcpd_current.json` instead
of hard-coded, and two extra points at the bottom of the range.

R1 established that the apparent d33 rise below ~30-50 mV is magnitude-detection
noise-floor bias (|signal + noise| >= |noise| biases a weak amplitude high, never
low), not real nonlinearity, and that everything from 50 mV to 2 V is flat. R2
adds 2 mV and 3 mV so the noise-floor branch has enough points to be fitted
rather than merely asserted -- if it really is the magnitude-detection floor, the
apparent amplitude should approach a constant (the noise level) as V_ac -> 0,
which three more decades-low points can actually test.

V_ac is not a `Condition` field (series.py carries bias/load/spot only), so it is
set directly via PV() before each drive level's pass, using the same run_series /
measure_conditions_at path as every other survey -- checkpoint format, InvOLS
handling and standing safety rules unchanged. One checkpoint per V_ac.

17 V_ac x 4 conditions (2 spots x 2 biases) x ~24 s ~ 30 min + moves/tunes.
Standing rules: finally -> bias 0, withdraw.
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scmpit_common as C
from activemodemap.series import make_conditions, run_series

if not C.FILE_LOC.rstrip('\\/').endswith('_R2'):
    raise SystemExit(f'R2 guard: C.FILE_LOC is {C.FILE_LOC!r} -- run 00_setup_r2.py first')

inst = globals().get('inst'); igor = inst.a.igor
P = C.load_preflight(); C.check_inst(inst, P)

VCPD_JSON = os.path.join(C.FILE_LOC, 'vcpd_current.json')
if not os.path.exists(VCPD_JSON):
    raise SystemExit(f'{VCPD_JSON} missing -- the V_cpd chain did not run')
_vc = json.load(open(VCPD_JSON, encoding='utf-8'))
V_CPD_TRUE = float(_vc['v_cpd_V'])
if not (-5.0 < V_CPD_TRUE < 5.0):
    raise SystemExit(f'V_cpd {V_CPD_TRUE} from {VCPD_JSON} is not physical')

X_A = 154.9
LOAD_NN = 500.0
VAC_LIST = [0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2,
            0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
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
    C.stamp(f'R2 STAGE 7  AC DRIVE SERIES AT POSITION A ({X_A} um): '
            f'{len(VAC_LIST)} V_ac levels x 2 spots x 2 biases (0 V, {V_CPD_TRUE:+.3f} V)')
    C.stamp(f'V_cpd source: {_vc.get("source")}')
    C.stamp(f'V_ac list (mV): {[round(v*1e3, 1) for v in VAC_LIST]}')
    C.stamp('=' * 78)
    cond = make_conditions([0.0, V_CPD_TRUE], [LOAD_NN], spots=[1, 2])
    for vac in VAC_LIST:
        say(f'\n--- V_ac = {vac*1000:.1f} mV ---')
        pv('DriveAmplitude', vac); time.sleep(0.5)
        if not verify_drive(vac):
            say('    retrying drive-amplitude set once...')
            pv('DriveAmplitude', vac); time.sleep(0.5)
            verify_drive(vac)
        t_v = time.time()
        results[vac] = run_series(inst, [X_A], cond, positions_um=[X_A],
                                  checkpoint_path=ckpt(vac), resume=True, verbose=True)
        say(f'V_ac {vac*1000:.1f} mV done in {(time.time() - t_v) / 60:.1f} min -> {ckpt(vac)}')
    say(f'\nAC series complete: {len(results)} drive levels x {len(cond)} conditions')
except Exception as e:
    say(f'*** FAILED: {type(e).__name__}: {e}')
    import traceback; traceback.print_exc(); raise
finally:
    try:
        pv('DriveAmplitude', 1.0)          # leave the panel at the campaign reference drive
    except Exception:
        pass
    C.finish(inst, t_start, 'R2 STAGE 7 AC DRIVE SERIES')
    sys.stdout = _old; _tee.close()

ac_series_res = results
