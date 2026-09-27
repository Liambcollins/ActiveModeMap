r"""R2 stage 5 -- quantitative PFM images on CR1. Same design as R1, three changes.

1. V_cpd is READ from `vcpd_current.json` (written by 11_vcpd_from_stage1.py from
   R2's own bias survey) instead of being hard-coded. R1's central lesson.
2. Position A is imaged at THREE biases (0, V_cpd/2, V_cpd) instead of two. R1
   extrapolated the true null from a two-point line and turned out to be 0.56 V
   off the imported value; three points bracket the crossing and make the
   linearity of the extrapolation checkable rather than assumed.
3. The CR1 search window is widened to 265-320 kHz. R1's 270-300 kHz window was
   set when CR1 was 285 kHz; it ended the day at 295.4 kHz, uncomfortably close
   to the edge.

Everything else is R1's stage 5 unchanged: same frame (PFM_befre0000, 6 um, 256^2,
1 Hz, angle 90, both domains, spots 1/2 marked on it), 500 nN, 1.0 V drive, fixed
drive frequency = CR1 measured at the imaging spot immediately before the series.

Two laser positions:
  A) stage x = 154.9 um (148.8 from clamp) -- away from the CR1 node, high and
     stable enhancement (R1: E = 198-226, still within 10 % of the morning value
     even after the tip changed).
  B) stage x = 232.0 um (free end) -- the CR1 node. R1 measured E there as 2.8 in
     the morning and 18-25 by evening; the close-out walk independently saw the
     free-end response grow x10.4. Whatever E is in R2, it is measured in situ
     from this position's own quasi-static frame, never imported.

Each position carries its own quasi-static (20 kHz) reference, so E = |A_cr1| /
|A_qs| is measured per position per domain and the analysis never needs a
cantilever model. The E_* constants below are priors for logging only.

9 frames x ~4.3 min + 2 moves/tunes ~ 45 min. Standing rules: finally -> bias 0,
withdraw.
"""
import os, sys, time, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scmpit_common as C

if not C.FILE_LOC.rstrip('\\/').endswith('_R2'):
    raise SystemExit(f'R2 guard: C.FILE_LOC is {C.FILE_LOC!r} -- run 00_setup_r2.py first')

inst = globals().get('inst'); igor = inst.a.igor
P = C.load_preflight(); C.check_inst(inst, P)

VCPD_JSON = os.path.join(C.FILE_LOC, 'vcpd_current.json')
if not os.path.exists(VCPD_JSON):
    raise SystemExit(f'{VCPD_JSON} missing -- 11_vcpd_from_stage1.py must run first')
_vc = json.load(open(VCPD_JSON, encoding='utf-8'))
V_CPD = float(_vc['v_cpd_V'])
if not (-5.0 < V_CPD < 5.0):
    raise SystemExit(f'V_cpd {V_CPD} from {VCPD_JSON} is not physical')

FRAME = dict(ScanSize=6e-06, XOffset=7.9308e-07, YOffset=-1.1606e-06, ScanAngle=90.0,
             ScanPoints=256, ScanLines=256, ScanRate=1.0016)        # PFM_befre0000.ibw
LOAD_NN, DRIVE_V = 500.0, 1.0
X_A, X_B = 154.9, 232.0                                             # stage coordinates
E_A, E_B = 200.0, 20.0                                              # priors, logging only
CR1_SEARCH = (265e3, 320e3)
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

V_HALF = round(V_CPD / 2.0, 3)
out = dict(started=time.strftime('%Y-%m-%d %H:%M:%S'), frame=FRAME, load_nN=LOAD_NN, drive_V=DRIVE_V,
           v_cpd_V=V_CPD, v_cpd_source=_vc.get('source'), v_half_V=V_HALF, positions={}, images=[])
t_start = time.time(); idx = 700
try:
    say('=' * 78)
    say(f'R2 STAGE 5  QUANTITATIVE CR1 IMAGING  (500 nN, {DRIVE_V} V, frame of PFM_befre0000)')
    say(f'V_cpd = {V_CPD:+.3f} V from {_vc.get("source")}; bias points 0, {V_HALF:+.3f}, {V_CPD:+.3f} V at A')
    say('=' * 78)
    inst.set_dc_bias(0.0)
    say('GoToSpot(1) so the frame is where the spots were marked'); inst.goto_spot(1)
    for k, v in FRAME.items(): pv(k, v)
    pv('DriveAmplitude', DRIVE_V); time.sleep(1.5)
    bad = verify({k: FRAME[k] for k in ('ScanSize', 'ScanPoints', 'ScanLines', 'ScanRate', 'ScanAngle')})
    bad += verify({k: FRAME[k] for k in ('XOffset', 'YOffset')}, abstol=5e-8)
    hard = [k for k in bad if k in ('ScanSize', 'XOffset', 'YOffset', 'ScanAngle')]
    if hard: raise RuntimeError(f'scan frame did not take: {hard}')

    plan = [(X_A, E_A, 'A_qs20k_0V',       0.0,     'qs'),
            (X_A, E_A, 'A_cr1_0V',         0.0,     'cr1'),
            (X_A, E_A, 'A_cr1_Vhalf',      V_HALF,  'cr1'),
            (X_A, E_A, 'A_cr1_Vcpd',       V_CPD,   'cr1'),
            (X_A, E_A, 'A_cr1_0V_repeat',  0.0,     'cr1'),
            (X_B, E_B, 'B_qs20k_0V',       0.0,     'qs'),
            (X_B, E_B, 'B_cr1_0V',         0.0,     'cr1'),
            (X_B, E_B, 'B_cr1_Vcpd',       V_CPD,   'cr1')]
    cur_x = None; f_cr1 = None
    for x, E, name, vdc, kind in plan:
        if x != cur_x:
            say(f'\n=== laser -> stage x = {x:.1f} um ({x - 6.1:.1f} um from clamp), '
                f'AutoWedge + InvOLS, engage {LOAD_NN:.0f} nN ===')
            inst.a.withdraw(); time.sleep(1)
            inst._goto_and_prepare(x, capture_image=True)
            if inst._invols is None: raise RuntimeError('InvOLS None at imaging position')
            inst.set_dc_bias(0.0)
            sp = inst.set_load(LOAD_NN, reengage=True)
            t = inst.a.tune_eigenmode(position_label=f'IMGTUNE_X{x:05.1f}', scan_index=idx,
                                      save_tune_data=True); idx += 1
            td = t['tune_data']; fr = np.asarray(td['frequency'], float)
            am = np.asarray(td['amplitude'], float)
            m = (fr >= CR1_SEARCH[0]) & (fr <= CR1_SEARCH[1])
            if not m.any(): raise RuntimeError(f'no tune points in {CR1_SEARCH} Hz')
            f_cr1 = float(fr[m][np.argmax(am[m])])
            snr = float(am[m].max() / max(np.median(am[m]), 1e-12))
            say(f'  InvOLS {inst._invols:.4e} m/V, setpoint {sp:.3f} V, '
                f'CR1 here {f_cr1/1e3:.3f} kHz (peak/median {snr:.1f})')
            if snr < 3.0:
                say('  ** WARNING: weak CR1 peak in the search window -- check the tune')
            out['positions'][str(x)] = dict(invols=float(inst._invols), f_cr1_Hz=f_cr1,
                                            E_prior=E, setpoint_V=float(sp), cr1_snr=snr)
            cur_x = x
        fdr = f_cr1 if kind == 'cr1' else F_QS
        say(f'\n--- {name}: V_dc = {vdc:+.3f} V, drive {fdr/1e3:.3f} kHz, {DRIVE_V} V ---')
        inst.set_dc_bias(vdc)
        pv('DriveFrequency', fdr); pv('DriveAmplitude', DRIVE_V); time.sleep(1.0)
        if verify({'DriveFrequency': fdr, 'DriveAmplitude': DRIVE_V}, tol=0.002):
            raise RuntimeError('drive did not take')
        path = grab(BASE)
        out['images'].append(dict(name=name, stage_x_um=x, x_true_um=x - 6.1, E_prior=E,
                                  invols=float(inst._invols), bias_V=vdc, drive_Hz=fdr,
                                  drive_V=DRIVE_V, kind=kind, path=path,
                                  time=time.strftime('%H:%M:%S')))
        json.dump(out, open(RESULT, 'w'), indent=1)

    # ---- drive linearity at B: same spot, same 20 kHz drive, V_ac = 2 V ----
    say(f'\n--- B_qs20k_0V_2Vac: drive linearity check, V_ac = 2.0 V ---')
    inst.set_dc_bias(0.0)
    pv('DriveFrequency', F_QS); pv('DriveAmplitude', 2.0); time.sleep(1.0)
    if verify({'DriveFrequency': F_QS, 'DriveAmplitude': 2.0}, tol=0.002):
        raise RuntimeError('drive did not take')
    path = grab(BASE)
    out['images'].append(dict(name='B_qs20k_0V_2Vac', stage_x_um=X_B, x_true_um=X_B - 6.1,
                              E_prior=E_B, invols=float(inst._invols), bias_V=0.0,
                              drive_Hz=F_QS, drive_V=2.0, kind='qs', path=path,
                              time=time.strftime('%H:%M:%S')))
    json.dump(out, open(RESULT, 'w'), indent=1)
    say('\nall images done')
except Exception as e:
    say(f'*** FAILED: {type(e).__name__}: {e}'); out['error'] = str(e)
    import traceback; traceback.print_exc(); raise
finally:
    try:
        pv('DriveAmplitude', DRIVE_V)
    except Exception:
        pass
    C.finish(inst, t_start, 'R2 STAGE 5 QUANT IMAGING')
    out['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    json.dump(out, open(RESULT, 'w'), indent=1, default=str)
    sys.stdout = _old; _tee.close()
