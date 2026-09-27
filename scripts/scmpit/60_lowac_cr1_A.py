r"""Stage 6 -- quantitative CR1 imaging at very low drive (<50 mV), with the
corrected Vcpd from the stage-5 analysis.

Stage 5 (this afternoon) found phase separation is a clean 178-180 deg at every
bias/position (the domain structure itself is unambiguous), but amplitude
equalization at the imported Vcpd = +0.77 V (this morning's stage-1 spectroscopy,
BEFORE the load ladder) only got the domain ratio from ~1.65-1.69 (0 V) down to
~1.24 -- not to 1.0. Extrapolating each domain's amplitude linearly in bias from
the two measured points (0 V, +0.77 V), position A and position B *independently*
agree: the true null today is Vcpd = +1.33 V (both give 1.326-1.327 V). Consistent
with the tip change already flagged during the 750-1500 nN load-ladder passes.

This stage repeats the position-A quantitative CR1 measurement (E = 200.2, the
position away from the CR1 node, so its enhancement is trustworthy) at
V_ac = 30 mV instead of 1 V, at 0 V and at the corrected Vcpd = +1.33 V, plus its
own quasi-static (20 kHz) reference at 30 mV -- both to check that domain
equalization / 180 deg separation still hold at low drive (the linear, small-signal
PFM regime with no risk of drive-induced switching or electronics nonlinearity),
and to extend the drive-linearity series (this morning's image: 2 V; stage 1
spectroscopy: 1 V; stage 5: 1 V vs 2 V, agreed to <1%) down by >30x.

At Vac = 1 V, domain amplitudes at A were ~1.4-2.4 nm; at 30 mV expect ~45-80 pm,
still >>noise given last time's clean nm-scale signal.

~3 images x 4.4 min + one move/tune ~ 16 min. Standing rules: finally -> bias 0,
withdraw.
"""
import os, sys, time, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scmpit_common as C

inst = globals().get('inst'); igor = inst.a.igor
P = C.load_preflight(); C.check_inst(inst, P)

FRAME = dict(ScanSize=6e-06, XOffset=7.9308e-07, YOffset=-1.1606e-06, ScanAngle=90.0,
             ScanPoints=256, ScanLines=256, ScanRate=1.0016)        # same frame as stage 5
LOAD_NN = 500.0
DRIVE_V = 0.03                                                      # 30 mV, < 50 mV requested
X_A, E_A = 154.9, 200.2
V_CPD_TRUE = 1.33                                                   # stage-5 extrapolated null (was 0.77)
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
           v_cpd_true_V=V_CPD_TRUE, x_a=X_A, images=[])
t_start = time.time(); idx = 800
try:
    say('=' * 78); say('STAGE 6  QUANTITATIVE CR1 IMAGING AT 30 mV DRIVE, Vcpd CORRECTED TO +1.33 V'); say('=' * 78)
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
    t = inst.a.tune_eigenmode(position_label=f'LOACTUNE_X{X_A:05.1f}', scan_index=idx, save_tune_data=True); idx += 1
    td = t['tune_data']; fr = np.asarray(td['frequency'], float); am = np.asarray(td['amplitude'], float)
    m = (fr >= 270e3) & (fr <= 320e3); f_cr1 = float(fr[m][np.argmax(am[m])])
    say(f'  InvOLS {inst._invols:.4e} m/V, setpoint {sp:.3f} V, CR1 here {f_cr1/1e3:.3f} kHz')
    out['invols'] = float(inst._invols); out['f_cr1_Hz'] = f_cr1

    plan = [('A_qs20k_0V_30mV', 0.0, F_QS), ('A_cr1_0V_30mV', 0.0, f_cr1),
            ('A_cr1_Vcpdtrue_30mV', V_CPD_TRUE, f_cr1)]
    for name, vdc, fdr in plan:
        say(f'\n--- {name}: V_dc = {vdc:+.2f} V, drive {fdr/1e3:.3f} kHz, {DRIVE_V*1e3:.0f} mV ---')
        inst.set_dc_bias(vdc)
        pv('DriveFrequency', fdr); pv('DriveAmplitude', DRIVE_V); time.sleep(1.0)
        if verify({'DriveFrequency': fdr, 'DriveAmplitude': DRIVE_V}, tol=0.01): raise RuntimeError('drive did not take')
        path = grab(BASE)
        out['images'].append(dict(name=name, bias_V=vdc, drive_Hz=fdr, drive_V=DRIVE_V,
                                  path=path, time=time.strftime('%H:%M:%S')))
        json.dump(out, open(RESULT, 'w'), indent=1)
    say('\nall low-drive images done')
except Exception as e:
    say(f'*** FAILED: {type(e).__name__}: {e}'); out['error'] = str(e)
    import traceback; traceback.print_exc(); raise
finally:
    C.finish(inst, t_start, 'STAGE 6 LOW-AC CR1 IMAGING')
    out['finished'] = time.strftime('%Y-%m-%d %H:%M:%S'); json.dump(out, open(RESULT, 'w'), indent=1, default=str)
    sys.stdout = _old; _tee.close()
