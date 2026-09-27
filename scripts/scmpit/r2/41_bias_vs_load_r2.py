r"""R2 extension, stage 4b -- the two-domain bias survey repeated at three loads.

This is the measurement neither campaign has ever made, and it is the one the
electrostatic-versus-piezoresponse story actually needs. R1's ladder ran at 0 V
only, so it gave P(x, load) and CR1(load) but no channel ratio; the bias surveys
ran at 500 nN only, so they gave |b|/|P| at one load. The prediction on record
(handoff, and R1 stage 1) is that the electrostatic term loads against the
CONTACT stiffness while the piezoresponse does not, so |b|/|P| should FALL as
load rises. Three loads at one position tests that directly.

Position A only (154.9 stage-um): away from the CR1 node, the most stable
enhancement in the dataset, and the position every other R2 result is anchored
to. Both marked domains, bias -9..+9 V interleaved so drift shows as scatter
rather than as a bias slope. One checkpoint per load.

V_cpd is NOT imported. The +-9 V ladder determines it independently at each load,
which is also a three-point check on whether V_cpd itself depends on load -- it
should not, since the contact potential is an electrostatic property of the
tip-sample pair, not of how hard they are pressed together.

3 loads x 14 conditions x ~25 s ~ 18 min + re-engages ~ 25 min.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scmpit_common as C
from activemodemap.series import make_conditions, run_series

R2_ROOT = r'D:\User Data\Liam\ActiveModeMap\DomainsB_SCMPIT_R2'
OUT = os.path.join(R2_ROOT, '41_bias_vs_load')

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

X_A = 154.9
LOADS_NN = [100.0, 300.0, 500.0]
BIAS_V = [-9, -6, -3, 0, 3, 6, 9]
BIAS_ORDER = 'interleaved'
DRIVE_V = 1.0
DNS_BAND_HZ = C.cr1_band(P, 0.80, 1.25)
LOG = os.path.join(C.FILE_LOC, 'bias_vs_load_r2_log.txt')


def ckpt(load):
    return os.path.join(C.FILE_LOC, f'bias_vs_load_r2_{load:04.0f}nN_checkpoint.npz')


_tee = C.Tee(LOG); _old = sys.stdout; sys.stdout = _tee
t0 = time.time()
results = {}
try:
    C.stamp('=' * 78)
    C.stamp(f'R2 EXT STAGE 4b  BIAS SURVEY AT {len(LOADS_NN)} LOADS, position A ({X_A} um)')
    C.stamp(f'loads {[int(l) for l in LOADS_NN]} nN, bias {BIAS_V} V ({BIAS_ORDER}), '
            f'spots 1 and 2, V_ac {DRIVE_V} V')
    C.stamp(f'data -> {C.FILE_LOC}')
    C.stamp('tests whether |b|/|P| falls with load, and whether V_cpd depends on load (it should not)')
    C.stamp('=' * 78)
    igor.Execute(f'PV("DriveAmplitude", {DRIVE_V})'); time.sleep(0.5)
    got = inst.a.get_gmv().get('DriveAmplitude')
    C.stamp(f'DriveAmplitude reads {got}')
    if got is None or abs(got - DRIVE_V) > 0.01:
        raise RuntimeError(f'drive amplitude is {got}, expected {DRIVE_V}')

    for load in LOADS_NN:
        t_l = time.time()
        C.stamp(f'\n--- load {load:.0f} nN '
                f'(deflection {load * 1e-9 / C.K_LEVER_N_PER_M * 1e9:.0f} nm) ---')
        cond = make_conditions(BIAS_V, [load], spots=[1, 2], bias_order=BIAS_ORDER)
        results[load] = run_series(inst, [X_A], cond, positions_um=[X_A], dns_band_Hz=DNS_BAND_HZ,
                                   checkpoint_path=ckpt(load), resume=True, verbose=True)
        C.stamp(f'load {load:.0f} nN done in {(time.time() - t_l) / 60:.1f} min -> {ckpt(load)}')

        # quick in-line readout so the log alone tells the story
        F = np.asarray(results[load]['freq_Hz'], float)
        Z = np.asarray(results[load]['Z'])[:, 0, :]
        bias = np.array([c.bias_V for c in cond], float)
        spot = np.array([c.spot for c in cond], int)
        qm = (F >= 15e3) & (F <= 45e3)
        zq = Z[:, qm].mean(axis=-1)
        s1, s2 = spot == 1, spot == 2
        (b1, a1) = np.polyfit(bias[s1], zq[s1], 1)
        (b2, a2) = np.polyfit(bias[s2], zq[s2], 1)
        Pp, bb = (a1 - a2) / 2.0, (b1 + b2) / 2.0
        C.stamp(f'   quasi-static: V_cpd {np.real(-(a1 + a2) / (2 * bb)):+.3f} V   '
                f'|b|/|P| {abs(bb) / abs(Pp):.3f}   '
                f'flip {np.degrees(abs(np.angle(a1) - np.angle(a2))):.1f} deg')
    C.stamp(f'\nbias-vs-load complete: {len(results)} loads x 14 conditions')
except Exception as e:
    C.stamp(f'*** FAILED: {type(e).__name__}: {e}')
    import traceback; traceback.print_exc(); raise
finally:
    C.finish(inst, t0, 'R2 EXT STAGE 4b BIAS VS LOAD')
    sys.stdout = _old; _tee.close()

bias_vs_load_r2_res = results
