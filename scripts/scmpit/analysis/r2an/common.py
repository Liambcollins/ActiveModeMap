import json, os
import numpy as np

HOME = os.path.expanduser('~')
R1 = os.path.join(HOME, 'mnt/ActiveModeMap/DomainsB_SCMPIT')
R2 = os.path.join(HOME, 'mnt/ActiveModeMap/DomainsB_SCMPIT_R2')
OUT = os.path.join(HOME, 'r2an')

P1 = dict(  # R1, after the per-stage reorganisation
    preflight=os.path.join(R1, '01_preflight/preflight_result.json'),
    bias=os.path.join(R1, '10_bias_survey/domains_bias_checkpoint_500nN_scmpit.npz'),
    wb_fast=os.path.join(R1, '20_wideband_fast/wideband_fast_8pos_500nN_0V_checkpoint.npz'),
    wb_val=os.path.join(R1, '21_wideband_validation/wideband_valid_7pos_500nN_0V_checkpoint.npz'),
    dense=os.path.join(R1, '30_dense_grid/dense_map_1um_500nN_0V_checkpoint.npz'),
    img=os.path.join(R1, '50_quant_imaging/image_cr1_quant.json'),
    lowac=os.path.join(R1, '60_lowac_30mV/lowac_cr1_A.json'),
    acdir=os.path.join(R1, '70_ac_series'),
    co_ref=os.path.join(R1, '80_closeout/closeout_ref_A_checkpoint.npz'),
    co_walk=os.path.join(R1, '80_closeout/closeout_wideband_8pos_checkpoint.npz'),
    co_json=os.path.join(R1, '80_closeout/closeout_cal.json'))

P2 = dict(
    preflight=os.path.join(R2, 'preflight_result.json'),
    bias=os.path.join(R2, 'domains_bias_checkpoint_500nN_scmpit.npz'),
    wb_fast=os.path.join(R2, 'wideband_fast_8pos_500nN_0V_checkpoint.npz'),
    wb_val=os.path.join(R2, 'wideband_valid_7pos_500nN_0V_checkpoint.npz'),
    dense=os.path.join(R2, 'dense_map_1um_500nN_0V_checkpoint.npz'),
    img=os.path.join(R2, 'image_cr1_quant.json'),
    lowac=os.path.join(R2, 'lowac_cr1_A.json'),
    acdir=R2,
    co_ref=os.path.join(R2, 'closeout_ref_A_checkpoint.npz'),
    co_walk=os.path.join(R2, 'closeout_wideband_8pos_checkpoint.npz'),
    co_json=os.path.join(R2, 'closeout_cal.json'))

CLAMP = 6.1                     # stage_um - CLAMP = distance from clamp
QS_BAND = (15e3, 45e3)
# CR bands must be wide enough for BOTH campaigns: R1 285/886/1706, R2 295/913/1831 kHz
BANDS = dict(cr1=(255e3, 330e3), cr2=(840e3, 960e3), cr3=(1650e3, 1900e3))


def load_ckpt(path):
    d = np.load(path, allow_pickle=True)
    F = np.asarray(d['freq_Hz'], float)
    Z = np.asarray(d['Z'])
    x = np.asarray(d['x_um'], float)
    cond = json.loads(str(d['conditions'])) if 'conditions' in d.files else None
    return F, Z, x, cond


def peak_in(F, z, band):
    m = (F >= band[0]) & (F <= band[1])
    if not m.any():
        return None
    a = np.abs(z[..., m])
    k = int(np.argmax(a))
    return dict(f_Hz=float(F[m][k]), amp=float(a[k]),
                phase_deg=float(np.degrees(np.angle(z[..., m][k]))),
                snr=float(a[k] / max(np.median(a), 1e-18)))


def decompose(bias, z1, z2):
    (b1, a1) = np.polyfit(bias, z1, 1)
    (b2, a2) = np.polyfit(bias, z2, 1)
    P = (a1 - a2) / 2.0
    b = (b1 + b2) / 2.0
    return dict(v_cpd=float(np.real(-(a1 + a2) / (2.0 * b))),
                P_abs=float(abs(P)), b_abs=float(abs(b)),
                ratio=float(abs(b) / abs(P)) if abs(P) else float('nan'),
                flip_deg=float(np.degrees(abs(np.angle(a1) - np.angle(a2)))),
                b_bal=float(abs(b2) / abs(b1)) if abs(b1) else float('nan'))


def jdump(obj, name):
    p = os.path.join(OUT, name)
    json.dump(obj, open(p, 'w'), indent=1, default=float)
    print(f'  -> {p}')
