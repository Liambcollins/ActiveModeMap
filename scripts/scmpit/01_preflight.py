r"""Pre-flight for the SCM-PIT probe: CR1/CR2/CR3 at 500 nN, and the reachable span.

What it does, in one engage sequence on SPOT 1:
  1. At the free end (x = L): AutoWedge, InvOLS, engage at 500 nN, one 2 MHz tune.
     Reports the three strongest resonances (CR1, CR2, CR3), CR1 FWHM, InvOLS.
  2. Walks the laser toward the BASE in STEP_UM steps, measuring at each step
     (same recipe: withdraw, move, AutoWedge, InvOLS, engage, tune, withdraw).
     Stops at the first position where the detection fails:
       - InvOLS None / outside (1e-8, 1e-5) m/V, or a >3x jump from the last good one
       - CR1 peak < MIN_SNR_dB above the floor
       - a tune / engage exception
     or at X_FLOOR_UM. The last GOOD position is x_lo.
  3. Withdraws and writes preflight_result.json + preflight_walk_checkpoint.npz
     (every good spectrum -- a free 10 um coarse map at 500 nN).

Time: ~1.2 min per position; a 232 um lever reachable to ~125 um is ~12 positions.

Requires: `inst` (00_setup_inst.py), the marked-spot image window OPEN (GoToSpot 1),
tune panel at 2 MHz sweep width / 10 s (Liam set both).
"""
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scmpit_common as C

inst = globals().get('inst')
C.check_inst(inst)

LOAD_NN = 500.0
SPOT = 1
STEP_UM = 10.0
X_FLOOR_UM = 30.0            # never walk closer to the base than this, whatever the signal says
MIN_SNR_dB = 8.0             # on the STRONGER of CR1/CR2 -- CR1 is near its node at the free end
# prior bands from the free-end tune at 12:21 (CR1 285.4 / CR2 885.6 / CR3 1703.8 kHz)
CR_BANDS_HZ = {'cr1': (230e3, 340e3), 'cr2': (780e3, 1000e3), 'cr3': (1550e3, 1850e3)}
INVOLS_BOUNDS = (1e-8, 1e-5)
INVOLS_JUMP = 3.0
L = float(inst.span_um or C.PROBE_L_UM_FALLBACK)

LOG = os.path.join(C.FILE_LOC, 'preflight_log.txt')
CKPT = os.path.join(C.FILE_LOC, 'preflight_walk_checkpoint.npz')
_tee = C.Tee(LOG); _old = sys.stdout; sys.stdout = _tee
t0 = time.time()

xs, Zs, freq_ref, recs, peaks_by_x = [], [], None, [], {}
result = dict(date=time.strftime('%Y-%m-%d %H:%M:%S'), probe='SCM-PIT', k_N_per_m=C.K_LEVER_N_PER_M,
              f0_free_Hz=C.F0_FREE_HZ, load_nN=LOAD_NN, spot=SPOT, probe_L_um=L,
              x_hi_um=L, x_lo_um=None, step_um=STEP_UM, stop_reason=None,
              cr1_Hz=None, cr2_Hz=None, cr3_Hz=None, cr1_fwhm_Hz=None, peaks_free_end=None)


def measure(x):
    """One full position: returns (freq, Z, rec, peaks) or raises."""
    label = inst._goto_and_prepare(x, capture_image=True)
    sp = inst.set_load(LOAD_NN, reengage=True)
    freq, Z, rec = inst._tune_and_read(label, sp)
    inst.a.withdraw(); time.sleep(1.0)
    pk = peaks_in_bands(freq, np.abs(Z))
    return freq, Z, rec, pk


def peaks_in_bands(freq, amp):
    """Strongest point inside each prior band, with SNR over the >50 kHz floor and FWHM."""
    freq = np.asarray(freq, float); amp = np.asarray(amp, float)
    good = np.isfinite(freq) & np.isfinite(amp) & (amp > 0)
    floor = np.quantile(amp[good & (freq > 50e3)], 0.2)
    out = []
    for name, (lo, hi) in CR_BANDS_HZ.items():
        m = good & (freq >= lo) & (freq <= hi)
        if not m.any():
            out.append(dict(name=name, f_Hz=None, snr_dB=-np.inf, fwhm_Hz=None)); continue
        idx = np.where(m)[0]; j = idx[np.argmax(amp[idx])]
        half = amp[j] / np.sqrt(2.0)
        a = j
        while a > idx[0] and amp[a] > half: a -= 1
        b = j
        while b < idx[-1] and amp[b] > half: b += 1
        out.append(dict(name=name, f_Hz=float(freq[j]), amp=float(amp[j]),
                        snr_dB=float(20 * np.log10(amp[j] / floor)), fwhm_Hz=float(freq[b] - freq[a])))
    return out


def invols_ok(rec, last):
    v = rec.get('invols_m_per_V')
    if v is None or not np.isfinite(v) or not (INVOLS_BOUNDS[0] < v < INVOLS_BOUNDS[1]):
        return False, f'InvOLS {v} outside bounds'
    if last and (v / last > INVOLS_JUMP or last / v > INVOLS_JUMP):
        return False, f'InvOLS jumped {last:.3e} -> {v:.3e}'
    return True, ''


try:
    C.stamp('=' * 78)
    C.stamp(f'PRE-FLIGHT  SCM-PIT on DomainsB  {LOAD_NN:.0f} nN, spot {SPOT}, lever {L:.0f} um (provisional)')
    C.stamp(f'deflection at {LOAD_NN:.0f} nN: {LOAD_NN * 1e-9 / C.K_LEVER_N_PER_M * 1e9:.0f} nm')
    C.stamp('=' * 78)
    inst.x_limits_um = (0.0, L)            # pre-flight is the one stage allowed to explore
    inst.load_nN = LOAD_NN
    inst.set_dc_bias(0.0)
    C.stamp(f'GoToSpot({SPOT}) -- the marked-spot image window must be open')
    inst.goto_spot(SPOT)

    # ---- 1. free end -----------------------------------------------------------
    x = L
    C.stamp(f'--- free end x = {x:.1f} um ---')
    freq, Z, rec, pk = measure(x)
    freq_ref = freq
    xs.append(x); Zs.append(Z); recs.append(rec); peaks_by_x[x] = pk
    C.stamp(f'InvOLS {rec["invols_m_per_V"]} m/V, panel k {rec["spring_N_per_m"]}, '
            f'setpoint {rec["setpoint_V"]:.3f} V, {rec["n_tune_points"]} pts '
            f'{rec["tune_f_lo_Hz"] / 1e3:.0f}-{rec["tune_f_hi_Hz"] / 1e3:.0f} kHz')
    C.stamp('resonances at the free end (strongest point in each prior band):')
    for p in pk:
        if p['f_Hz'] is None:
            C.stamp(f'   {p["name"].upper()}: no data in band'); continue
        C.stamp(f'   {p["name"].upper()}: {p["f_Hz"] / 1e3:8.2f} kHz   FWHM {p["fwhm_Hz"] / 1e3:6.2f} kHz   '
                f'Q~{p["f_Hz"] / max(p["fwhm_Hz"], 1):5.0f}   {p["snr_dB"]:5.1f} dB')
    byname = {p['name']: p for p in pk}
    if byname['cr1']['snr_dB'] < 3 and byname['cr2']['snr_dB'] < 3:
        raise RuntimeError('neither CR1 nor CR2 visible at the free end -- is the tip engaged, '
                           'is the tune panel on the 2 MHz window?')
    result['peaks_free_end'] = pk
    result['cr1_Hz'] = byname['cr1']['f_Hz']; result['cr1_fwhm_Hz'] = byname['cr1']['fwhm_Hz']
    result['cr2_Hz'] = byname['cr2']['f_Hz']; result['cr3_Hz'] = byname['cr3']['f_Hz']
    result['cr1_snr_free_end_dB'] = byname['cr1']['snr_dB']
    C.stamp(f'CR2/CR1 = {result["cr2_Hz"] / result["cr1_Hz"]:.3f} '
            f'(earlier SCM-PIT 904/293 = 3.085; clamped-pinned EB 3.24)')
    if rec['tune_f_hi_Hz'] < 1.5e6:
        C.stamp(f'*** tune window tops out at {rec["tune_f_hi_Hz"] / 1e3:.0f} kHz -- '
                f'CR3 will not be captured; expected 2 MHz ***')

    # ---- 2. walk toward the base ----------------------------------------------
    last_invols = rec['invols_m_per_V']
    x_lo = L
    x = L - STEP_UM
    while x >= X_FLOOR_UM - 1e-9:
        C.stamp(f'--- x = {x:.1f} um ({L - x:.0f} um from the free end) ---')
        try:
            freq, Z, rec, pk = measure(x)
        except Exception as e:
            result['stop_reason'] = f'x={x:.1f}: {type(e).__name__}: {e}'
            C.stamp(f'STOP: {result["stop_reason"]}')
            break
        ok, why = invols_ok(rec, last_invols)
        bn = {p['name']: p for p in pk}
        snr = max(bn['cr1']['snr_dB'], bn['cr2']['snr_dB'])
        C.stamp(f'InvOLS {rec["invols_m_per_V"]}, '
                f'CR1 {(bn["cr1"]["f_Hz"] or 0) / 1e3:.2f} kHz {bn["cr1"]["snr_dB"]:.1f} dB, '
                f'CR2 {(bn["cr2"]["f_Hz"] or 0) / 1e3:.2f} kHz {bn["cr2"]["snr_dB"]:.1f} dB, '
                f'CR3 {(bn["cr3"]["f_Hz"] or 0) / 1e3:.2f} kHz {bn["cr3"]["snr_dB"]:.1f} dB')
        if not ok:
            result['stop_reason'] = f'x={x:.1f}: {why}'
            C.stamp(f'STOP: {result["stop_reason"]}')
            break
        if snr < MIN_SNR_dB:
            result['stop_reason'] = f'x={x:.1f}: best of CR1/CR2 SNR {snr:.1f} dB < {MIN_SNR_dB} dB'
            C.stamp(f'STOP: {result["stop_reason"]}')
            break
        xs.append(x); Zs.append(Z); recs.append(rec); peaks_by_x[x] = pk
        last_invols = rec['invols_m_per_V']
        x_lo = x
        x -= STEP_UM
    else:
        result['stop_reason'] = f'reached X_FLOOR_UM = {X_FLOOR_UM}'
        C.stamp(f'STOP: {result["stop_reason"]}')

    result['x_lo_um'] = float(x_lo)
    result['n_good_positions'] = len(xs)
    result['invols_by_x'] = {f'{xx:.1f}': r['invols_m_per_V'] for xx, r in zip(xs, recs)}
    for nm in ('cr1', 'cr2', 'cr3'):
        result[f'{nm}_by_x'] = {f'{xx:.1f}': next((p['f_Hz'] for p in peaks_by_x[xx] if p['name'] == nm), None)
                                for xx in xs}
        result[f'{nm}_snr_by_x'] = {f'{xx:.1f}': next((p['snr_dB'] for p in peaks_by_x[xx] if p['name'] == nm), None)
                                    for xx in xs}
    C.stamp('=' * 78)
    C.stamp(f'REACHABLE SPAN: x = {x_lo:.0f} .. {L:.0f} um  ({L - x_lo:.0f} um, '
            f'{100 * (L - x_lo) / L:.0f} % of the lever), {len(xs)} good positions')
    C.stamp(f'CR1 {result["cr1_Hz"] / 1e3:.2f} kHz (FWHM {result["cr1_fwhm_Hz"] / 1e3:.2f} kHz)   '
            f'CR2 {(result["cr2_Hz"] or 0) / 1e3:.2f} kHz   CR3 {(result["cr3_Hz"] or 0) / 1e3:.2f} kHz')
    C.stamp('=' * 78)

    # bookkeeping for every later stage
    inst.x_limits_um = (float(x_lo), float(L))
    C.stamp(f'inst.x_limits_um -> {inst.x_limits_um}')

finally:
    try:
        if xs:
            np.savez(CKPT, x_um=np.array(xs), freq_Hz=np.asarray(freq_ref),
                     Z=np.array(Zs), load_nN=LOAD_NN, spot=SPOT,
                     records=json.dumps(recs, default=str), peaks=json.dumps(peaks_by_x, default=str))
            C.stamp(f'walk checkpoint -> {CKPT}')
        with open(C.PREFLIGHT_JSON, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=1, default=str)
        C.stamp(f'pre-flight result -> {C.PREFLIGHT_JSON}')
    except Exception:
        traceback.print_exc()
    C.finish(inst, t0, 'PRE-FLIGHT')
    sys.stdout = _old; _tee.close()

if result['x_lo_um'] is None:
    raise SystemExit('pre-flight did not establish a span')
