r"""Stage 8 -- CLOSE-OUT CALIBRATION for the SCM-PIT probe (end of the 2026-09-20 campaign).

Everything downstream of the load ladder (stages 5-7) depends on calibration
constants that demonstrably MOVED during the session: V_cpd walked 0.77 -> 1.33
-> 1.22 V, InvOLS jumped across the 750-1500 nN passes, the CR1 enhancement at
position B went stale by 7-9x, and the two domains' relative CR1 amplitude
ranking actually inverted between stage 5 and stage 7. None of those numbers has
an end-of-campaign anchor. This stage supplies one, so every earlier result can
be quoted with a measured drift bound instead of an assumption.

Three parts, in the only order that works (the free-air check needs the tip OUT,
so it goes last):

  PART 1 -- reference-condition repeat at position A (154.9 um).
      EXACTLY the stage-1 recipe: spots 1 and 2, bias [-9,-6,-3,0,3,6,9] V
      interleaved, 500 nN, V_ac = 1.000 V. Same conditions, same method, same
      position, ~9 h later. Gives a clean end-of-day V_cpd, P, b and d33 that is
      directly differenceable against this morning's stage-1 numbers.
      NOTE: stage 7 left DriveAmplitude at 2.0 V. This script puts it back to
      1.000 V and VERIFIES the readback before measuring -- a silent 2 V would
      make the whole comparison meaningless.

  PART 2 -- mode-shape walk repeat, 8 positions, 0 V, 500 nN, spot 1, V_ac 1 V.
      Same X grid as stage 2a (linspace(x_hi, x_lo, 8), free end -> base). Tells
      whether the CR1 frequency/amplitude profile ALONG the beam has changed
      shape, which separates "the contact got stiffer" from "the cantilever
      itself changed".

  PART 3 -- final detection calibration, then withdraw.
      One more AutoWedge + InvOLS force curve at position A, and the panel
      spring constant, recorded against the pre-flight values. Then withdraw.
      A best-effort free-air tune follows, clearly flagged: the drive on this
      setup is the ELECTRICAL (tip-bias) drive, so with the tip off the surface
      there is no mechanical excitation and this tune may well return only the
      noise floor. It is included because it is nearly free and a visible
      thermal peak near 63.8 kHz would be a bonus data point; it is NOT relied
      on. A proper thermal spring-constant re-cal is a manual Thermal-panel
      step -- deliberately not driven blind over COM here.

Standing campaign rules preserved verbatim: bias to 0 V and withdraw in the
`finally` block, never inst.close(), all COM on the main thread.

Runtime ~20 min.
"""
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scmpit_common as C
# --- R2 guard -------------------------------------------------------------
# Byte-identical to the R1 script apart from this block. Refuses to run unless
# 00_setup_r2.py has repointed the campaign at the R2 data folder, so a failed
# setup aborts the queue instead of writing a second campaign into R1's folders.
if not C.FILE_LOC.rstrip('\\/').endswith('_R2'):
    raise SystemExit(f'R2 guard: C.FILE_LOC is {C.FILE_LOC!r} -- run 00_setup_r2.py first')
from activemodemap.series import make_conditions, run_series

inst = globals().get('inst')
igor = inst.a.igor
P = C.load_preflight()
C.check_inst(inst, P)

# ---- design -----------------------------------------------------------------
X_A = 154.9                                   # position A, the campaign's workhorse
LOAD_NN = 500.0
VAC_REF = 1.000                               # stage-1 drive amplitude; stage 7 left 2.0 V
BIAS_V = [-9, -6, -3, 0, 3, 6, 9]
BIAS_ORDER = 'interleaved'
N_WALK = 8
X_WALK = np.round(np.linspace(P['x_hi_um'], P['x_lo_um'], N_WALK), 1)   # free end -> base
FREEAIR_DRIVE_V = 0.05                        # small, in case the free-air tune does respond

CKPT_REF = os.path.join(C.FILE_LOC, 'closeout_ref_A_checkpoint.npz')
CKPT_WALK = os.path.join(C.FILE_LOC, f'closeout_wideband_{N_WALK}pos_checkpoint.npz')
OUT_JSON = os.path.join(C.FILE_LOC, 'closeout_cal.json')
LOG = os.path.join(C.FILE_LOC, 'closeout_cal_log.txt')

_tee = C.Tee(LOG); _old = sys.stdout; sys.stdout = _tee
say = C.stamp
t0 = time.time()

out = dict(date=time.strftime('%Y-%m-%d %H:%M:%S'), stage='8_closeout',
           probe='SCM-PIT', position_A_um=X_A, load_nN=LOAD_NN, vac_ref_V=VAC_REF,
           bias_V=BIAS_V, x_walk_um=X_WALK.tolist(),
           preflight=dict(cr1_Hz=P.get('cr1_Hz'), cr2_Hz=P.get('cr2_Hz'),
                          cr3_Hz=P.get('cr3_Hz'),
                          k_N_per_m=C.K_LEVER_N_PER_M, f0_free_Hz=C.F0_FREE_HZ,
                          invols_by_x=P.get('invols_by_x')),
           part1_ref=None, part2_walk=None, part3_detection=None, freeair=None)

ref_series = walk_series = None


def pv(k, v):
    igor.Execute(f'PV("{k}", {v})')


def read_drive():
    try:
        return inst.a.get_gmv().get('DriveAmplitude')
    except Exception as e:
        say(f'    drive readback failed: {e}')
        return None


def set_drive(vac, tol=0.01):
    """Set DriveAmplitude and verify. One retry. Returns (ok, readback)."""
    pv('DriveAmplitude', vac); time.sleep(0.5)
    got = read_drive()
    ok = got is not None and abs(got - vac) <= max(tol * vac, 1e-4)
    say(f'    DriveAmplitude asked {vac:.4g} V  reads {got}  {"ok" if ok else "** MISMATCH"}')
    if not ok:
        say('    retrying drive-amplitude set once...')
        pv('DriveAmplitude', vac); time.sleep(0.5)
        got = read_drive()
        ok = got is not None and abs(got - vac) <= max(tol * vac, 1e-4)
        say(f'    DriveAmplitude now reads {got}  {"ok" if ok else "** STILL MISMATCHED"}')
    return ok, got


def cr1_peak(freq, Z, lo=230e3, hi=340e3):
    """Peak frequency / |Z| / phase in the CR1 window, with a crude peak test."""
    freq = np.asarray(freq, float)
    a = np.abs(np.asarray(Z))
    m = (freq >= lo) & (freq <= hi)
    if not m.any():
        return None
    fr, aa, zz = freq[m], a[m], np.asarray(Z)[m]
    k = int(np.argmax(aa))
    med = float(np.median(aa))
    return dict(f_Hz=float(fr[k]), amp=float(aa[k]),
                phase_deg=float(np.degrees(np.angle(zz[k]))),
                peak_over_median=float(aa[k] / med) if med > 0 else None,
                peak_ok=bool(aa[k] > 3 * med))


try:
    say('=' * 78)
    say('STAGE 8  CLOSE-OUT CALIBRATION  (SCM-PIT, DomainsB)')
    say(f'position A {X_A} um, {LOAD_NN:.0f} nN, V_ac {VAC_REF:.3f} V, bias {BIAS_V} V ({BIAS_ORDER})')
    say(f'walk: {N_WALK} positions {X_WALK[0]:.1f} -> {X_WALK[-1]:.1f} um')
    say(f'instrument before: x={inst.current_x} um, spot={inst.current_spot}, '
        f'load={inst.load_nN} nN, bias={inst.dc_bias_V} V')
    say('=' * 78)

    # ---------------------------------------------------------------- PART 1 --
    say('')
    say('--- PART 1: reference-condition repeat at position A (stage-1 recipe) ---')
    drive_before = read_drive()
    say(f'    DriveAmplitude on entry (stage 7 left this): {drive_before} V')
    ok_drive, drive_now = set_drive(VAC_REF)
    if not ok_drive:
        raise RuntimeError(f'could not set DriveAmplitude to {VAC_REF} V (reads {drive_now}); '
                           'refusing to run a "reference" comparison at an unknown drive level')
    out['drive_on_entry_V'] = drive_before
    out['drive_used_V'] = drive_now

    cond_ref = make_conditions(BIAS_V, [LOAD_NN], spots=[1, 2], bias_order=BIAS_ORDER)
    say(f'    {len(cond_ref)} conditions: ' + ', '.join(str(c) for c in cond_ref))
    t_p1 = time.time()
    ref_series = run_series(inst, [X_A], cond_ref, positions_um=[X_A],
                            checkpoint_path=CKPT_REF, resume=True, verbose=True)
    say(f'    PART 1 done in {(time.time() - t_p1) / 60:.1f} min -> {CKPT_REF}')

    F, Z = np.asarray(ref_series['freq_Hz'], float), np.asarray(ref_series['Z'])
    pk_ref = []
    for i, c in enumerate(cond_ref):
        pk = cr1_peak(F, Z[i, 0, :]) if Z.ndim == 3 else cr1_peak(F, Z[i])
        if pk:
            pk['condition'] = str(c)
            pk['bias_V'] = float(getattr(c, 'bias_V', np.nan))
            pk['spot'] = int(getattr(c, 'spot', -1))
            pk_ref.append(pk)
            say(f'    {str(c):>34s}  CR1 {pk["f_Hz"] / 1e3:7.2f} kHz  '
                f'|Z| {pk["amp"]:.6g}  phase {pk["phase_deg"]:+7.1f} deg  '
                f'{"" if pk["peak_ok"] else "** weak peak"}')
    out['part1_ref'] = dict(checkpoint=CKPT_REF, n_conditions=len(cond_ref), peaks=pk_ref,
                            minutes=(time.time() - t_p1) / 60)

    # ---------------------------------------------------------------- PART 2 --
    say('')
    say('--- PART 2: mode-shape walk repeat (stage-2a grid) ---')
    cond_walk = make_conditions([0.0], [LOAD_NN], spots=[1])
    t_p2 = time.time()
    walk_series = run_series(inst, X_WALK, cond_walk, positions_um=X_WALK,
                             checkpoint_path=CKPT_WALK, resume=True, verbose=True)
    say(f'    PART 2 done in {(time.time() - t_p2) / 60:.1f} min -> {CKPT_WALK}')

    Fw, Zw = np.asarray(walk_series['freq_Hz'], float), np.asarray(walk_series['Z'])
    xw = np.asarray(walk_series['x_um'], float)
    pk_walk = []
    for j, xx in enumerate(xw):
        zj = Zw[0, j, :] if Zw.ndim == 3 else Zw[j]
        pk = cr1_peak(Fw, zj)
        if pk:
            pk['x_um'] = float(xx)
            pk_walk.append(pk)
            say(f'    x {xx:7.1f} um  CR1 {pk["f_Hz"] / 1e3:7.2f} kHz  '
                f'|Z| {pk["amp"]:.6g}  phase {pk["phase_deg"]:+7.1f} deg  '
                f'{"" if pk["peak_ok"] else "** weak peak"}')
    out['part2_walk'] = dict(checkpoint=CKPT_WALK, n_positions=int(xw.size), peaks=pk_walk,
                             minutes=(time.time() - t_p2) / 60)

    # ---------------------------------------------------------------- PART 3 --
    say('')
    say('--- PART 3: final detection calibration at position A ---')
    t_p3 = time.time()
    _recal_was = getattr(inst, 'recalibrate_each', None)
    inst.recalibrate_each = True          # force a fresh AutoWedge + InvOLS
    label = inst._goto_and_prepare(X_A, capture_image=True)
    invols_final = inst._invols
    spring_final = inst._spring
    pf_invols = (P.get('invols_by_x') or {}).get(f'{X_A:.1f}')
    say(f'    final InvOLS at {X_A} um: {invols_final} m/V')
    say(f'    pre-flight InvOLS at the same x: {pf_invols} m/V')
    if invols_final and pf_invols:
        say(f'    ratio final/pre-flight: {invols_final / float(pf_invols):.3f}')
    say(f'    panel spring constant: {spring_final} N/m  (campaign {C.K_LEVER_N_PER_M} N/m)')
    out['part3_detection'] = dict(
        label=label, invols_final_m_per_V=invols_final,
        invols_preflight_m_per_V=(float(pf_invols) if pf_invols else None),
        invols_ratio=(invols_final / float(pf_invols) if (invols_final and pf_invols) else None),
        spring_panel_N_per_m=spring_final, spring_campaign_N_per_m=C.K_LEVER_N_PER_M,
        invols_rejected=list(getattr(inst.a, 'invols_rejected', [])),
        minutes=(time.time() - t_p3) / 60)

    # best-effort free-air tune -- see the module docstring; may be noise only
    say('')
    say('--- free-air tune (BEST EFFORT: electrical drive, tip out of contact) ---')
    try:
        inst.set_dc_bias(0.0)
        inst.a.withdraw(); time.sleep(2.0)
        set_drive(FREEAIR_DRIVE_V)
        tune = inst.a.tune_eigenmode(position_label='CLOSEOUT_FREEAIR',
                                     scan_index=900, save_tune_data=True)
        td = tune.get('tune_data')
        fa = dict(resonance_freq_Hz=tune.get('resonance_freq'), q_factor=tune.get('q_factor'),
                  tune_file=tune.get('tune_file'), drive_V=FREEAIR_DRIVE_V,
                  f0_free_preflight_Hz=C.F0_FREE_HZ, interpretable=False, note='')
        if td is not None:
            fr = np.asarray(td['frequency'], float)
            am = np.asarray(td['amplitude'], float)
            m = (fr > 50e3) & (fr < 80e3) & np.isfinite(am)
            if m.any():
                k = int(np.argmax(am[m]))
                f_pk = float(fr[m][k])
                floor = float(np.quantile(am[np.isfinite(am) & (fr > 100e3)], 0.2))
                snr_dB = float(20 * np.log10(am[m][k] / floor)) if floor > 0 else float('nan')
                fa.update(peak_50_80kHz_Hz=f_pk, peak_snr_dB=snr_dB)
                fa['interpretable'] = bool(snr_dB > 8.0)
                say(f'    50-80 kHz peak: {f_pk / 1e3:.2f} kHz at {snr_dB:.1f} dB '
                    f'(pre-flight free resonance {C.F0_FREE_HZ / 1e3:.2f} kHz)')
                if not fa['interpretable']:
                    fa['note'] = ('no mechanical drive off-surface; peak not above the floor -- '
                                  'as expected, do NOT read this as a cantilever measurement')
                    say('    -> below the 8 dB floor. Expected: there is no mechanical drive '
                        'off-surface. Ignore this number.')
                else:
                    fa['note'] = 'thermal/residual peak visible; bonus data point, not a calibration'
                    say('    -> visible. Bonus only; this is not a thermal k calibration.')
        out['freeair'] = fa
    except Exception as e:
        say(f'    free-air tune failed (harmless, it was best-effort): {type(e).__name__}: {e}')
        out['freeair'] = dict(failed=f'{type(e).__name__}: {e}')

    say('')
    say('=' * 78)
    say('CLOSE-OUT SUMMARY')
    if out['part1_ref']:
        z0 = [p for p in out['part1_ref']['peaks'] if abs(p['bias_V']) < 1e-9]
        for p in z0:
            say(f'  0 V, spot {p["spot"]}: CR1 {p["f_Hz"] / 1e3:.2f} kHz, '
                f'|Z| {p["amp"]:.6g}, phase {p["phase_deg"]:+.1f} deg')
    say(f'  InvOLS final {invols_final} m/V (pre-flight at x={X_A}: {pf_invols})')
    say(f'  panel k {spring_final} N/m')
    say('  NOTE: a proper thermal spring-constant re-cal is a manual Thermal-panel')
    say('        step and was deliberately NOT driven over COM. Worth 30 s by hand')
    say('        if the absolute d33 scale needs to be defended.')
    say('=' * 78)

except Exception as e:
    say(f'*** STAGE 8 FAILED: {type(e).__name__}: {e}')
    traceback.print_exc()
    out['error'] = f'{type(e).__name__}: {e}'
    raise
finally:
    try:
        with open(OUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=1, default=str)
        say(f'close-out result -> {OUT_JSON}')
    except Exception:
        traceback.print_exc()
    try:
        _rw = globals().get('_recal_was')
        if _rw is not None:
            inst.recalibrate_each = _rw          # leave instrument state as we found it
    except Exception as e:
        say(f'  recalibrate_each restore: {e}')
    try:
        set_drive(VAC_REF)          # leave the panel at the campaign's reference drive
    except Exception as e:
        say(f'  drive restore: {e}')
    C.finish(inst, t0, 'STAGE 8 CLOSE-OUT CALIBRATION')
    sys.stdout = _old; _tee.close()

closeout_res = out
