r"""Stage 4 -- load dependence on ONE domain (spot 1), 50 -> 1500 nN. RUNS LAST.

Liam, 2026-09-20: extend the ladder to 1500 nN even though it may damage the
probe, so it is the final stage. Design choices that follow from that:

  * LOAD IS THE OUTER LOOP. Each load is a complete 8-position pass (own
    checkpoint) before the next, higher load starts -- so the lower loads are all
    measured before the tip has ever seen 750+ nN, and a damaged tip contaminates
    only the loads above the damage, never the ones below.
  * PROBE CHECK between loads >= CHECK_ABOVE_NN: one measurement at the free end
    at 500 nN (the pre-flight/dense-map condition). CR1 and InvOLS are compared to
    the pre-flight values and printed; a CR1 shift beyond CR1_WARN_FRAC is flagged
    loudly. It does not stop the ladder (that is the user's decision) -- it stamps
    the log so the analysis knows where the tip changed.
  * Positions: the same 8 as stage 1 (equispaced over the reachable span), free
    end first, so |b|/|P| from stage 1 gets its load axis at identical positions.

Deflections at k = 1.697 N/m: 29 / 88 / 177 / 295 / 442 / 589 / 884 nm.
8 positions x 7 loads ~ 65 min + 4 probe checks ~ 5 min.
"""
import json
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

LOADS_NN = [50.0, 150.0, 300.0, 500.0, 750.0, 1000.0, 1500.0]
BIAS_V, SPOT = 0.0, 1
CHECK_ABOVE_NN = 500.0
CHECK_LOAD_NN = 500.0
CR1_WARN_FRAC = 0.03
X_POS = np.round(np.linspace(P['x_lo_um'], P['x_hi_um'], 8), 1)[::-1]      # stage-1 positions, free end first
DNS_BAND_HZ = C.cr1_band(P, 0.75, 1.30)                                     # CR1 stiffens with load
LOG = os.path.join(C.FILE_LOC, 'load_ladder_spot1_log.txt')
CHECKS_JSON = os.path.join(C.FILE_LOC, 'load_ladder_probe_checks.json')


def ckpt(load):
    return os.path.join(C.FILE_LOC, f'load_ladder_spot1_{load:04.0f}nN_checkpoint.npz')


def probe_check(after_load):
    """Free end, 500 nN: CR1 and InvOLS vs pre-flight. Returns a dict."""
    inst.load_nN = CHECK_LOAD_NN
    inst.set_dc_bias(0.0)
    freq, Z, rec = inst.measure_at(P['x_hi_um'])
    pk = C.find_peaks_simple(freq, np.abs(Z), n=3, min_sep_Hz=60e3, min_snr_dB=6.0)
    cr1 = next((p for p in pk if abs(p['f_Hz'] - P['cr1_Hz']) < 0.2 * P['cr1_Hz']), None)
    d = dict(after_load_nN=after_load, time=time.strftime('%H:%M:%S'),
             invols=rec['invols_m_per_V'], invols_preflight=P['invols_by_x'].get(f"{P['x_hi_um']:.1f}"),
             cr1_Hz=(cr1['f_Hz'] if cr1 else None), cr1_preflight_Hz=P['cr1_Hz'],
             cr1_fwhm_Hz=(cr1['fwhm_Hz'] if cr1 else None), peaks=pk)
    if cr1:
        shift = (cr1['f_Hz'] - P['cr1_Hz']) / P['cr1_Hz']
        d['cr1_shift_frac'] = shift
        flag = '  *** CR1 SHIFT > %.0f %% -- TIP MAY HAVE CHANGED ***' % (100 * CR1_WARN_FRAC) \
            if abs(shift) > CR1_WARN_FRAC else ''
        C.stamp(f'PROBE CHECK after {after_load:.0f} nN: CR1 {cr1["f_Hz"] / 1e3:.2f} kHz '
                f'({100 * shift:+.2f} % vs pre-flight), FWHM {cr1["fwhm_Hz"] / 1e3:.2f} kHz, '
                f'InvOLS {rec["invols_m_per_V"]:.3e}{flag}')
    else:
        C.stamp(f'PROBE CHECK after {after_load:.0f} nN: *** CR1 NOT FOUND near {P["cr1_Hz"] / 1e3:.0f} kHz ***')
    return d


_tee = C.Tee(LOG); _old = sys.stdout; sys.stdout = _tee
t0 = time.time()
checks, results = [], {}
try:
    C.stamp('=' * 78)
    C.stamp(f'STAGE 4  LOAD LADDER ON SPOT {SPOT}: {[int(l) for l in LOADS_NN]} nN, load = OUTER loop')
    C.stamp(f'positions (free end first): {X_POS.tolist()}')
    C.stamp(f'analysis band {DNS_BAND_HZ[0] / 1e3:.0f}-{DNS_BAND_HZ[1] / 1e3:.0f} kHz; probe check at '
            f'{P["x_hi_um"]:.0f} um / {CHECK_LOAD_NN:.0f} nN after every load >= {CHECK_ABOVE_NN:.0f} nN')
    C.stamp('=' * 78)
    for load in LOADS_NN:
        t_l = time.time()
        C.stamp(f'--- load {load:.0f} nN (deflection {load * 1e-9 / C.K_LEVER_N_PER_M * 1e9:.0f} nm) ---')
        cond = make_conditions([BIAS_V], [load], spots=[SPOT])
        results[load] = run_series(inst, X_POS, cond, positions_um=X_POS, dns_band_Hz=DNS_BAND_HZ,
                                   checkpoint_path=ckpt(load), resume=True, verbose=True)
        C.stamp(f'load {load:.0f} nN done in {(time.time() - t_l) / 60:.1f} min -> {ckpt(load)}')
        if load >= CHECK_ABOVE_NN and load != LOADS_NN[-1]:
            checks.append(probe_check(load))
            with open(CHECKS_JSON, 'w', encoding='utf-8') as f:
                json.dump(checks, f, indent=1, default=str)
    checks.append(probe_check(LOADS_NN[-1]))
    with open(CHECKS_JSON, 'w', encoding='utf-8') as f:
        json.dump(checks, f, indent=1, default=str)
    C.stamp(f'ladder complete: {len(results)} loads x {X_POS.size} positions; probe checks -> {CHECKS_JSON}')
finally:
    C.finish(inst, t0, 'STAGE 4 LOAD LADDER')
    sys.stdout = _old; _tee.close()

load_ladder_res = results
