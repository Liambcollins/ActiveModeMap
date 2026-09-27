r"""R2 stage 1b -- recover V_cpd from R2's OWN bias survey. Analysis only, no instrument.

The single clearest lesson of R1: a V_cpd measured hours earlier, across anything
that changes the tip, is not valid later. So R2 never imports one. This script
runs straight after the bias survey, reads its checkpoint, and writes
`vcpd_current.json`, which every downstream stage reads instead of a hard-coded
constant.

Channel choice: the quasi-static plateau (15-45 kHz), not the CR1 peak. R1's
close-out compared both on identical data and the quasi-static band was the
sound one -- it reproduced the published stage-1 d33 (7.84 vs 7.90 pm/V) and its
V_cpd landed on the independent stage-7 value (+1.19 vs +1.22 V), while a
single-bin CR1 estimate sat ~0.17 V low. CR1 is still computed and logged for
comparison, just not used.

Runs in the relay kernel but touches no COM and no hardware, so it is safe
anywhere in the queue.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scmpit_common as C

if not C.FILE_LOC.rstrip('\\/').endswith('_R2'):
    raise SystemExit(f'R2 guard: C.FILE_LOC is {C.FILE_LOC!r} -- run 00_setup_r2.py first')

CKPT = os.path.join(C.FILE_LOC, 'domains_bias_checkpoint_500nN_scmpit.npz')
OUT = os.path.join(C.FILE_LOC, 'vcpd_current.json')
LOG = os.path.join(C.FILE_LOC, 'vcpd_from_stage1_log.txt')
X_A_TARGET = 154.9                 # the campaign's workhorse position
QS_BAND = (15e3, 45e3)
CR1_WIN = (255e3, 325e3)

_tee = C.Tee(LOG); _old = sys.stdout; sys.stdout = _tee
say = C.stamp


def decompose(bias, z1, z2):
    """Z_s(V) = a_s + b_s V for the two domains -> P, b, V_cpd."""
    (b1, a1) = np.polyfit(bias, z1, 1)
    (b2, a2) = np.polyfit(bias, z2, 1)
    P = (a1 - a2) / 2.0
    b = (b1 + b2) / 2.0
    return dict(v_cpd=float(np.real(-(a1 + a2) / (2.0 * b))),
                P_abs=float(abs(P)), b_abs=float(abs(b)),
                ratio=float(abs(b) / abs(P)) if abs(P) > 0 else float('nan'),
                flip_deg=float(np.degrees(abs(np.angle(a1) - np.angle(a2)))),
                b_bal=float(abs(b2) / abs(b1)) if abs(b1) > 0 else float('nan'))


try:
    say('=' * 78)
    say('R2 STAGE 1b  V_cpd FROM R2\'S OWN BIAS SURVEY')
    say('=' * 78)
    if not os.path.exists(CKPT):
        raise SystemExit(f'bias-survey checkpoint missing: {CKPT}')

    d = np.load(CKPT, allow_pickle=True)
    F = np.asarray(d['freq_Hz'], float)
    Z = np.asarray(d['Z'])                      # (n_cond, n_pos, n_freq)
    x = np.asarray(d['x_um'], float)
    cond = json.loads(str(d['conditions']))
    bias = np.array([c['bias_V'] for c in cond], float)
    spot = np.array([c['spot'] for c in cond], int)
    say(f'checkpoint: {Z.shape[0]} conditions x {Z.shape[1]} positions x {Z.shape[2]} freqs')
    say(f'positions: {np.round(x, 1).tolist()}')

    j = int(np.argmin(np.abs(x - X_A_TARGET)))
    say(f'using position index {j} (x = {x[j]:.1f} um, target {X_A_TARGET})')

    s1, s2 = spot == 1, spot == 2
    if not (s1.any() and s2.any()):
        raise SystemExit('need both spots in the survey to separate the domains')

    results = {}
    for tag, band in (('qs', QS_BAND), ('cr1_band', CR1_WIN)):
        m = (F >= band[0]) & (F <= band[1])
        if tag == 'qs':
            zz = Z[:, j, m].mean(axis=-1)                       # flat plateau -> average it
        else:
            k = int(np.argmax(np.abs(Z[np.argmax(bias == 0), j, m])))
            zz = Z[:, j, m][:, k]                               # the CR1 peak bin at 0 V
        r = decompose(bias[s1], zz[s1], zz[s2])
        results[tag] = r
        say(f'  {tag:9s}  V_cpd {r["v_cpd"]:+7.3f} V   |b|/|P| {r["ratio"]:.3f}   '
            f'flip {r["flip_deg"]:6.1f} deg   |b2/b1| {r["b_bal"]:.2f}')

    v = results['qs']['v_cpd']
    sane = np.isfinite(v) and -5.0 < v < 5.0
    if not sane:
        raise SystemExit(f'quasi-static V_cpd {v} is not physical -- refusing to write it')
    if not (150.0 < results['qs']['flip_deg'] < 210.0):
        say(f'  NOTE: domain flip {results["qs"]["flip_deg"]:.0f} deg is outside 150-210 -- '
            f'the two spots may not be cleanly antiparallel domains. V_cpd still written.')

    payload = dict(v_cpd_V=float(v), source='R2_stage1_quasistatic_band',
                   position_um=float(x[j]), band_Hz=list(QS_BAND),
                   cr1_band_v_cpd_V=float(results['cr1_band']['v_cpd']),
                   channel_ratio=float(results['qs']['ratio']),
                   flip_deg=float(results['qs']['flip_deg']),
                   written=time.strftime('%Y-%m-%d %H:%M:%S'))
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=1)
    say('')
    say(f'V_cpd for the rest of R2: {v:+.3f} V  ->  {OUT}')
    say(f'(R1 for reference: +0.77 V morning, +1.33 V post-ladder, +1.19 V close-out)')
    say('=' * 78)
except Exception as e:
    say(f'*** FAILED: {type(e).__name__}: {e}')
    say('    downstream stages will fall back to whatever vcpd_current.json already holds')
    import traceback; traceback.print_exc()
    raise
finally:
    sys.stdout = _old; _tee.close()
