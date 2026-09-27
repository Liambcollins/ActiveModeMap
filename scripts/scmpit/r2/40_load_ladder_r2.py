r"""R2 extension, stage 4 -- load ladder on the STABLE tip, 50 -> 500 nN.

R1's ladder went to 1500 nN and that is where the tip changed: CR1 jumped +3.2 %,
InvOLS jumped 10x, V_cpd moved +0.56 V, and every stage afterwards sat on a
different contact state than the ones before. This ladder deliberately stops at
500 nN -- the load every other R2 stage used -- so the load axis is measured on
the same contact state as the rest of the R2 dataset and stays comparable to it.

Design carried over from R1 unchanged, because it worked:
  * LOAD IS THE OUTER LOOP. Each load is a complete 8-position pass with its own
    checkpoint before the next, higher load starts, so a tip change contaminates
    only the loads above it and never the ones below.
  * PROBE CHECK after EVERY load this time, not just the high ones. Each check is
    one free-end measurement at 500 nN compared against the R2 pre-flight CR1 and
    InvOLS. At R1's load range the checks were a damage alarm; here they are the
    evidence that nothing changed, which is what the load result rests on.
  * Positions: the same 8 the R2 bias survey and dense map used, free end first,
    so |b|/|P| and the mode shape get their load axis at identical positions.

Deflections at k = 1.697 N/m: 29 / 59 / 118 / 206 / 295 nm.
8 positions x 5 loads ~ 48 min + 5 probe checks ~ 6 min.

Writes straight into its own experiment folder -- R2's per-stage layout already
exists and files cannot be moved within it afterwards.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scmpit_common as C
from activemodemap.series import make_conditions, run_series

R2_ROOT = r'D:\User Data\Liam\ActiveModeMap\DomainsB_SCMPIT_R2'
OUT = os.path.join(R2_ROOT, '40_load_ladder')

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

LOADS_NN = [50.0, 100.0, 200.0, 350.0, 500.0]
BIAS_V, SPOT = 0.0, 1
DRIVE_V = 1.0
CHECK_LOAD_NN = 500.0
CR1_WARN_FRAC = 0.02
X_POS = np.round(np.linspace(P['x_lo_um'], P['x_hi_um'], 8), 1)[::-1]   # free end first
DNS_BAND_HZ = C.cr1_band(P, 0.80, 1.25)
LOG = os.path.join(C.FILE_LOC, 'load_ladder_r2_log.txt')
CHECKS_JSON = os.path.join(C.FILE_LOC, 'load_ladder_r2_probe_checks.json')


def ckpt(load):
    return os.path.join(C.FILE_LOC, f'load_ladder_r2_{load:04.0f}nN_checkpoint.npz')


def probe_check(after_load):
    inst.load_nN = CHECK_LOAD_NN
    inst.set_dc_bias(0.0)
    freq, Z, rec = inst.measure_at(P['x_hi_um'])
    pk = C.find_peaks_simple(freq, np.abs(Z), n=3, min_sep_Hz=60e3, min_snr_dB=6.0)
    cr1 = next((p for p in pk if abs(p['f_Hz'] - P['cr1_Hz']) < 0.2 * P['cr1_Hz']), None)
    d = dict(after_load_nN=after_load, time=time.strftime('%H:%M:%S'),
             invols=rec['invols_m_per_V'],
             invols_preflight=(P.get('invols_by_x') or {}).get(f"{P['x_hi_um']:.1f}"),
             cr1_Hz=(cr1['f_Hz'] if cr1 else None), cr1_preflight_Hz=P['cr1_Hz'],
             cr1_fwhm_Hz=(cr1['fwhm_Hz'] if cr1 else None), peaks=pk)
    if cr1:
        shift = (cr1['f_Hz'] - P['cr1_Hz']) / P['cr1_Hz']
        d['cr1_shift_frac'] = shift
        flag = ('  *** CR1 SHIFT > %.0f %% -- TIP MAY HAVE CHANGED ***' % (100 * CR1_WARN_FRAC)
                if abs(shift) > CR1_WARN_FRAC else '')
        C.stamp(f'PROBE CHECK after {after_load:.0f} nN: CR1 {cr1["f_Hz"] / 1e3:.2f} kHz '
                f'({100 * shift:+.2f} % vs R2 pre-flight), FWHM {cr1["fwhm_Hz"] / 1e3:.2f} kHz, '
                f'Q~{cr1["f_Hz"] / max(cr1["fwhm_Hz"], 1):.0f}, '
                f'InvOLS {rec["invols_m_per_V"]:.3e}{flag}')
    else:
        C.stamp(f'PROBE CHECK after {after_load:.0f} nN: '
                f'*** CR1 NOT FOUND near {P["cr1_Hz"] / 1e3:.0f} kHz ***')
    return d


_tee = C.Tee(LOG); _old = sys.stdout; sys.stdout = _tee
t0 = time.time()
checks, results = [], {}
try:
    C.stamp('=' * 78)
    C.stamp(f'R2 EXT STAGE 4  LOAD LADDER, SPOT {SPOT}: {[int(l) for l in LOADS_NN]} nN, '
            f'load = OUTER loop, capped at {LOADS_NN[-1]:.0f} nN')
    C.stamp(f'data -> {C.FILE_LOC}')
    C.stamp(f'positions (free end first): {X_POS.tolist()}')
    C.stamp(f'R2 pre-flight CR1 {P["cr1_Hz"] / 1e3:.2f} kHz; analysis band '
            f'{DNS_BAND_HZ[0] / 1e3:.0f}-{DNS_BAND_HZ[1] / 1e3:.0f} kHz; '
            f'probe check at {P["x_hi_um"]:.0f} um / {CHECK_LOAD_NN:.0f} nN after EVERY load')
    C.stamp('=' * 78)
    igor.Execute(f'PV("DriveAmplitude", {DRIVE_V})'); time.sleep(0.5)
    C.stamp(f'DriveAmplitude reads {inst.a.get_gmv().get("DriveAmplitude")}')
    for load in LOADS_NN:
        t_l = time.time()
        C.stamp(f'--- load {load:.0f} nN '
                f'(deflection {load * 1e-9 / C.K_LEVER_N_PER_M * 1e9:.0f} nm) ---')
        cond = make_conditions([BIAS_V], [load], spots=[SPOT])
        results[load] = run_series(inst, X_POS, cond, positions_um=X_POS, dns_band_Hz=DNS_BAND_HZ,
                                   checkpoint_path=ckpt(load), resume=True, verbose=True)
        C.stamp(f'load {load:.0f} nN done in {(time.time() - t_l) / 60:.1f} min -> {ckpt(load)}')
        checks.append(probe_check(load))
        with open(CHECKS_JSON, 'w', encoding='utf-8') as f:
            json.dump(checks, f, indent=1, default=str)
    C.stamp(f'ladder complete: {len(results)} loads x {X_POS.size} positions; '
            f'probe checks -> {CHECKS_JSON}')
except Exception as e:
    C.stamp(f'*** FAILED: {type(e).__name__}: {e}')
    import traceback; traceback.print_exc(); raise
finally:
    C.finish(inst, t0, 'R2 EXT STAGE 4 LOAD LADDER')
    sys.stdout = _old; _tee.close()

load_ladder_r2_res = results
