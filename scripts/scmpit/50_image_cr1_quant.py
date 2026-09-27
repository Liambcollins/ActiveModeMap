r"""Stage 5 -- quantitative PFM images on CR1 with the cantilever transfer known.

Frame: this morning's PFM_befre0000 (6 um, 256^2, 1 Hz, angle 90, both domains, spots
1/2 marked on it). Load 500 nN, drive 1.0 V, fixed drive frequency = CR1 measured at the
imaging spot just before the series (drift over 3.6 h today was 74 Hz rms; linewidth 1.5 kHz).

Two laser positions, chosen from stage 1 / the dense map:
  A) stage x = 154.9 um (148.8 from clamp): CR1 enhancement E = 200 = Q, |P|_CR1 near its
     maximum. Here -- and only here -- amplitude / Q happens to be right.
  B) stage x = 232.0 um (free end): E = 2.8. The "textbook" laser position; CR1 gives
     almost no gain because the pinned tip is the CR1 node.
Same conversion for both: d33 = A * InvOLS(x)/32 / (V_ac * E(x)), E from stage 1.

Biases: 0 V, and +0.77 V = V_cpd from stage 1 (nulls the electrostatic term). Predictions,
from stage 1 at CR1 (|b|/|P| = 0.19/V, arg(P/b) = +19 deg):
   V_dc = 0     : domain(spot 1) amplitude 0.865 |P|, domain(spot 2) 1.137 |P| -> ratio 1.31,
                  phase separation ~175 deg
   V_dc = +0.77 : both domains |P| -> ratio 1.00, separation 180 deg; d33 = 7.9 pm/V on both
A 0 V repeat at A brackets drift. A quasi-static reference (20 kHz, 0 V) at B matches this
morning's image and converts with no cantilever model at all.

~6 images x 4.3 min + moves/tunes ~ 35 min. Standing rules: finally -> bias 0, withdraw.
"""
import os, sys, time, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scmpit_common as C

inst = globals().get('inst'); igor = inst.a.igor
P = C.load_preflight(); C.check_inst(inst, P)

FRAME = dict(ScanSize=6e-06, XOffset=7.9308e-07, YOffset=-1.1606e-06, ScanAngle=90.0,
             ScanPoints=256, ScanLines=256, ScanRate=1.0016)        # PFM_befre0000.ibw, 11:45 today
LOAD_NN, DRIVE_V, V_CPD = 500.0, 1.0, 0.77
X_A, X_B = 154.9, 232.0                                             # stage coordinates
E_A, E_B = 200.2, 2.8                                               # stage-1 CR1 enhancement at A, B
F_QS = 20e3
BASE = 'QImg'
LOG = os.path.join(C.FILE_LOC, 'image_cr1_quant_log.txt')
RESULT = os.path.join(C.FILE_LOC, 'image_cr1_quant.json')
_tee = C.Tee(LOG); _old = sys.stdout; sys.stdout = _tee
say = C.stamp

def pv(k, v): igor.Execute(f'PV("{k}", {v})')

def verify(expected, tol=0.02, abstol=None):
    gmv, bad = inst.a.get_gmv(), []
    for k, want in expected.items():
        got = gmv.get(k)
        if got is None:
            say(f'    {k:<14} not in MasterVariables'); continue
        lim = abstol if abstol is not None else tol * max(abs(want), 1e-12)
        ok = abs(got - want) <= lim
        say(f'    {k:<14} asked {want:<12.6g} reads {got:<12.6g} {"ok" if ok else "** MISMATCH"}')
        if not ok: bad.append(k)
    return bad

def _scanning():
    return bool(igor.DataFolder(r'root:packages:MFP3D').Wave('OutWaves').GetTextWavePointValue(0, 0))

def grab(base, timeout_s=1200):
    inst.a.set_folder()
    igor.Execute(f'root:Packages:MFP3D:Main:Variables:BaseName = "{base}"'); igor.Execute('ARCheckSuffix()')
    pat = os.path.join(inst.a.file_loc, f'{base}*.ibw')
    before = {p: os.path.getmtime(p) for p in glob.glob(pat)}
    t0 = time.time()
    inst.a.ex('DownScan_0', 'MasterPanel'); time.sleep(10)
    while _scanning() and time.time() - t0 < timeout_s: time.sleep(2)
    if time.time() - t0 >= timeout_s: say('    WARNING: scan timeout')
    time.sleep(3)
    new = [p for p in glob.glob(pat) if p not in before or os.path.getmtime(p) > before[p]]
    if not new: raise RuntimeError(f'no new {base}*.ibw written')
    path = max(new, key=os.path.getmtime); last = -1
    for _ in range(20):
        s = os.path.getsize(path)
        if s == last and s > 0: break
        last = s; time.sleep(0.5)
    say(f'    -> {os.path.basename(path)} ({os.path.getsize(path)/1e6:.2f} MB, {(time.time()-t0)/60:.1f} min)')
    return path

out = dict(started=time.strftime('%Y-%m-%d %H:%M:%S'), frame=FRAME, load_nN=LOAD_NN, drive_V=DRIVE_V,
           v_cpd_V=V_CPD, positions={}, images=[])
t_start = time.time(); idx = 700
try:
    say('=' * 78); say('STAGE 5  QUANTITATIVE CR1 IMAGING  (frame of PFM_befre0000, 500 nN, 1 V)'); say('=' * 78)
    inst.set_dc_bias(0.0)
    say(f'GoToSpot(1) so the frame is where the spots were marked'); inst.goto_spot(1)
    for k, v in FRAME.items(): pv(k, v)
    pv('DriveAmplitude', DRIVE_V); time.sleep(1.5)
    bad = verify({k: FRAME[k] for k in ('ScanSize', 'ScanPoints', 'ScanLines', 'ScanRate', 'ScanAngle')})
    bad += verify({k: FRAME[k] for k in ('XOffset', 'YOffset')}, abstol=5e-8)
    hard = [k for k in bad if k in ('ScanSize', 'XOffset', 'YOffset', 'ScanAngle')]
    if hard: raise RuntimeError(f'scan frame did not take: {hard}')

    # Each position carries its OWN quasi-static reference, so the CR1 enhancement is
    # measured in situ (E = |A_cr1| / |A_qs| per position) rather than taken from stage 1.
    # That keeps the experiment valid even if the 1500 nN pass changed the tip.
    plan = [(X_A, E_A, 'A_qs20k_0V', 0.0, 'qs'), (X_A, E_A, 'A_cr1_0V', 0.0, 'cr1'),
            (X_A, E_A, 'A_cr1_Vcpd', V_CPD, 'cr1'), (X_A, E_A, 'A_cr1_0V_repeat', 0.0, 'cr1'),
            (X_B, E_B, 'B_qs20k_0V', 0.0, 'qs'), (X_B, E_B, 'B_cr1_0V', 0.0, 'cr1'),
            (X_B, E_B, 'B_cr1_Vcpd', V_CPD, 'cr1')]
    cur_x = None; f_cr1 = None
    for x, E, name, vdc, kind in plan:
        if x != cur_x:
            say(f'\n=== laser -> stage x = {x:.1f} um ({x - 6.1:.1f} um from clamp), AutoWedge + InvOLS, engage {LOAD_NN:.0f} nN ===')
            inst.a.withdraw(); time.sleep(1)
            inst._goto_and_prepare(x, capture_image=True)
            if inst._invols is None: raise RuntimeError('InvOLS None at imaging position')
            inst.set_dc_bias(0.0)
            sp = inst.set_load(LOAD_NN, reengage=True)
            t = inst.a.tune_eigenmode(position_label=f'IMGTUNE_X{x:05.1f}', scan_index=idx, save_tune_data=True); idx += 1
            td = t['tune_data']; fr = np.asarray(td['frequency'], float); am = np.asarray(td['amplitude'], float)
            m = (fr >= 270e3) & (fr <= 300e3); f_cr1 = float(fr[m][np.argmax(am[m])])
            say(f'  InvOLS {inst._invols:.4e} m/V, setpoint {sp:.3f} V, CR1 here {f_cr1/1e3:.3f} kHz  '
                f'(stage-1 E = {E:.1f}; conversion factor InvOLS/32/E = {inst._invols/32/E:.3e} m/V per V)')
            out['positions'][str(x)] = dict(invols=float(inst._invols), f_cr1_Hz=f_cr1, E=E, setpoint_V=float(sp))
            cur_x = x
        fdr = f_cr1 if kind == 'cr1' else F_QS
        say(f'\n--- {name}: V_dc = {vdc:+.2f} V, drive {fdr/1e3:.3f} kHz, {DRIVE_V} V ---')
        inst.set_dc_bias(vdc)
        pv('DriveFrequency', fdr); pv('DriveAmplitude', DRIVE_V); time.sleep(1.0)
        if verify({'DriveFrequency': fdr, 'DriveAmplitude': DRIVE_V}, tol=0.002): raise RuntimeError('drive did not take')
        path = grab(BASE)
        out['images'].append(dict(name=name, stage_x_um=x, x_true_um=x - 6.1, E=E, invols=float(inst._invols),
                                  bias_V=vdc, drive_Hz=fdr, drive_V=DRIVE_V, path=path, time=time.strftime('%H:%M:%S')))
        json.dump(out, open(RESULT, 'w'), indent=1)

    # ---- drive linearity: the 25 % gap between this morning's image (V_ac = 2 V) and the
    # stage-1 spectroscopy (V_ac = 1 V). Same spot, same 20 kHz drive, V_ac = 2 V.
    say(f'\n--- B_qs20k_0V_2Vac: drive linearity check, V_ac = 2.0 V ---')
    inst.set_dc_bias(0.0)
    pv('DriveFrequency', F_QS); pv('DriveAmplitude', 2.0); time.sleep(1.0)
    if verify({'DriveFrequency': F_QS, 'DriveAmplitude': 2.0}, tol=0.002):
        raise RuntimeError('drive did not take')
    path = grab(BASE)
    out['images'].append(dict(name='B_qs20k_0V_2Vac', stage_x_um=X_B, x_true_um=X_B - 6.1, E=E_B,
                              invols=float(inst._invols), bias_V=0.0, drive_Hz=F_QS, drive_V=2.0,
                              path=path, time=time.strftime('%H:%M:%S')))
    json.dump(out, open(RESULT, 'w'), indent=1)
    say('\nall images done')
except Exception as e:
    say(f'*** FAILED: {type(e).__name__}: {e}'); out['error'] = str(e)
    import traceback; traceback.print_exc(); raise
finally:
    C.finish(inst, t_start, 'STAGE 5 QUANT IMAGING')
    out['finished'] = time.strftime('%Y-%m-%d %H:%M:%S'); json.dump(out, open(RESULT, 'w'), indent=1, default=str)
    sys.stdout = _old; _tee.close()
