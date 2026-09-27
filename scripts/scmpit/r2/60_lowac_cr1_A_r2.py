r"""R2 stage 6 -- quantitative CR1 imaging at 30 mV drive, at R2's own measured null.

Identical in design to R1 stage 6. The only changes: V_cpd comes from
`vcpd_current.json` (by this point written by 55_vcpd_from_images.py from R2's
own stage-5 frames, minutes earlier, same tip, same drive regime) instead of a
hard-coded 1.33 V, and the CR1 search window is widened to 265-320 kHz.

What it tests, unchanged from R1:
  (a) that domain equalisation and 180 deg separation still hold in the
      small-signal regime, >30x below the 1 V used everywhere else -- and at a
      null derived entirely from 1 V data, so agreement here is a real
      cross-check rather than a tautology;
  (b) the drive-linearity series, extending the 1 V / 2 V pair from stage 5 down
      by a further factor of 33.

R1's result to compare against: domain ratio 1.623 (0 V) -> 0.980 at the
corrected null, phase separation 179.0-179.9 deg, and input-referred CR1 noise
rising only ~2.6x from 1 V to 30 mV while the quasi-static signal vanished below
the floor entirely.

3 images x ~4.4 min + one move/tune ~ 16 min. Standing rules: finally -> bias 0,
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
    raise SystemExit(f'{VCPD_JSON} missing -- the V_cpd chain did not run')
_vc = json.load(open(VCPD_JSON, encoding='utf-8'))
V_CPD_TRUE = float(_vc['v_cpd_V'])
if not (-5.0 < V_CPD_TRUE < 5.0):
    raise SystemExit(f'V_cpd {V_CPD_TRUE} from {VCPD_JSON} is not physical')

FRAME = dict(ScanSize=6e-06, XOffset=7.9308e-07, YOffset=-1.1606e-06, ScanAngle=90.0,
             ScanPoints=256, ScanLines=256, ScanRate=1.0016)        # same frame as stage 5
LOAD_NN = 500.0
DRIVE_V = 0.03                                                      # 30 mV
X_A = 154.9
CR1_SEARCH = (265e3, 320e3)
F_QS = 20e3
BASE = 'LoAC'
LOG = os.path.join(C.FILE_LOC, 'lowac_cr1_A_log.txt')
RESULT = os.path.join(C.FILE_LOC, 'lowac_cr1_A.json')
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
           v_cpd_true_V=V_CPD_TRUE, v_cpd_source=_vc.get('source'), x_a=X_A, images=[])
t_start = time.time(); idx = 800
try:
    say('=' * 78)
    say(f'R2 STAGE 6  CR1 IMAGING AT {DRIVE_V*1e3:.0f} mV DRIVE, V_cpd = {V_CPD_TRUE:+.3f} V')
    say(f'V_cpd source: {_vc.get("source")}')
    say('=' * 78)
    inst.set_dc_bias(0.0)
    inst.goto_spot(1)
    for k, v in FRAME.items(): pv(k, v)
    pv('DriveAmplitude', DRIVE_V); time.sleep(1.5)
    bad = verify({k: FRAME[k] for k in ('ScanSize', 'ScanPoints', 'ScanLines', 'ScanRate', 'ScanAngle')})
    bad += verify({k: FRAME[k] for k in ('XOffset', 'YOffset')}, abstol=5e-8)
    hard = [k for k in bad if k in ('ScanSize', 'XOffset', 'YOffset', 'ScanAngle')]
    if hard: raise RuntimeError(f'scan frame did not take: {hard}')

    say(f'\n=== laser -> stage x = {X_A:.1f} um, AutoWedge + InvOLS, engage {LOAD_NN:.0f} nN ===')
    inst.a.withdraw(); time.sleep(1)
    inst._goto_and_prepare(X_A, capture_image=True)
    if inst._invols is None: raise RuntimeError('InvOLS None at imaging position')
    inst.set_dc_bias(0.0)
    sp = inst.set_load(LOAD_NN, reengage=True)
    t = inst.a.tune_eigenmode(position_label=f'LOACTUNE_X{X_A:05.1f}', scan_index=idx,
                              save_tune_data=True); idx += 1
    td = t['tune_data']; fr = np.asarray(td['frequency'], float); am = np.asarray(td['amplitude'], float)
    m = (fr >= CR1_SEARCH[0]) & (fr <= CR1_SEARCH[1])
    if not m.any(): raise RuntimeError(f'no tune points in {CR1_SEARCH} Hz')
    f_cr1 = float(fr[m][np.argmax(am[m])])
    say(f'  InvOLS {inst._invols:.4e} m/V, setpoint {sp:.3f} V, CR1 here {f_cr1/1e3:.3f} kHz')
    out['invols'] = float(inst._invols); out['f_cr1_Hz'] = f_cr1

    plan = [('A_qs20k_0V_30mV', 0.0, F_QS), ('A_cr1_0V_30mV', 0.0, f_cr1),
            ('A_cr1_Vcpdtrue_30mV', V_CPD_TRUE, f_cr1)]
    for name, vdc, fdr in plan:
        say(f'\n--- {name}: V_dc = {vdc:+.3f} V, drive {fdr/1e3:.3f} kHz, {DRIVE_V*1e3:.0f} mV ---')
        inst.set_dc_bias(vdc)
        pv('DriveFrequency', fdr); pv('DriveAmplitude', DRIVE_V); time.sleep(1.0)
        if verify({'DriveFrequency': fdr, 'DriveAmplitude': DRIVE_V}, tol=0.01):
            raise RuntimeError('drive did not take')
        path = grab(BASE)
        out['images'].append(dict(name=name, bias_V=vdc, drive_Hz=fdr, drive_V=DRIVE_V,
                                  path=path, time=time.strftime('%H:%M:%S')))
        json.dump(out, open(RESULT, 'w'), indent=1)
    say('\nall low-drive images done')
except Exception as e:
    say(f'*** FAILED: {type(e).__name__}: {e}'); out['error'] = str(e)
    import traceback; traceback.print_exc(); raise
finally:
    try:
        pv('DriveAmplitude', 1.0)          # leave the panel at the campaign reference drive
    except Exception:
        pass
    C.finish(inst, t_start, 'R2 STAGE 6 LOW-AC CR1 IMAGING')
    out['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    json.dump(out, open(RESULT, 'w'), indent=1, default=str)
    sys.stdout = _old; _tee.close()
