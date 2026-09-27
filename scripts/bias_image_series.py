# =====================================================================================
# Bias-series PFM imaging on the 6 um frame of DomBias0021 — queued 2026-09-18
#
# WHY.  The two domains do not share a contact potential: fitting each domain's null
# separately gives +2.12 V and +0.85 V (CR1 band, stable to +/-0.04 V over all eight
# laser positions).  The "+1.51 V" reported earlier is their MEAN, so no single dc
# setpoint nulls both domains, and at the mean the residual electrostatic term is
# antisymmetric between the domains — it looks exactly like piezoresponse.
# The prediction this run tests, from spectroscopy at one point, over a whole image:
#
#     V_dc        amp ratio A:B     phase
#     0 V              3.0          ~5 deg   (domains nearly invisible in phase at CR1)
#     null_2           3.9          166 deg
#     mean             0.70         165 deg  (amplitudes cross over)
#     null_1           0.16         149 deg  (contrast INVERTED)
#
# Run order: laser to the free end -> 9-point bias sweep on EACH domain at the imaging
# load (pins both nulls at the contact the images are actually taken at) -> restore the
# DomBias0021 frame -> six images.
#
# Images 1-4 are the bias series at CR1 on a FIXED drive frequency (a fixed drive keeps
# the four amplitudes comparable; each image is preceded by a tune that is recorded but
# not applied, so resonance drift is measured rather than silently absorbed).
# Image 5 repeats 0 V as the drift control — if 1 and 5 disagree, the series is drift,
# not bias.  Image 6 is 30.0 kHz at 0 V, the geometry reference matching DomBias0021.
#
# Contact matches DomBias0021: 6 um, 256^2, 1.0 Hz, 1 V ac, 36.5 nN, XOffset/YOffset
# read from that file's note.  Nothing here calls inst.close().
# =====================================================================================
import os, time, glob, json
import numpy as np
from activemodemap.asylum import tune_to_complex

# ---- the frame, read from DomBias0021.ibw's own note --------------------------------
FRAME = dict(ScanSize=6e-06, XOffset=2.0035e-06, YOffset=4.3442e-07, ScanAngle=0.0,
             ScanPoints=256, ScanLines=256, ScanRate=1.0016)
IMG_LOAD_NN   = 36.5        # = DomBias0021's setpoint 0.49054 V x InvOLS 5.0248e-7 x k 0.14818
IMG_DRIVE_V   = 1.0         # DriveAmplitude, as in DomBias0021
LASER_X_UM    = 445.0       # free end: highest sensitivity, and where the bias survey was fit
REF_FREQ_HZ   = 30.0e3      # the geometry-reference frame, as in DomBias0021
SWEEP_V       = [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
NULL_FALLBACK = (2.12, 0.85)                 # (domain at spot 1, domain at spot 2), CR1 band
BASE          = 'BiasImg'
LOG           = r'D:\User Data\Liam\ActiveModeMap\DomainsBPPPCONTAU\bias_image_series_log.txt'
RESULT        = r'D:\User Data\Liam\ActiveModeMap\DomainsBPPPCONTAU\bias_image_series.json'
SPOT_1        = int(globals().get('SPOT_UP', 1))
SPOT_2        = int(globals().get('SPOT_DOWN', 2))

_lf = open(LOG, 'a', buffering=1)
def say(m=''):
    line = f'{time.strftime("%H:%M:%S")}  {m}'
    print(line, flush=True); _lf.write(line + '\n')

def pv(key, val):
    igor.Execute(f'PV("{key}", {val})')

def verify(expected, tol=0.02, abstol=None):
    """expected: {key: wanted}. Returns the keys the panel did not take."""
    gmv, bad = inst.a.get_gmv(), []
    for k, want in expected.items():
        got = gmv.get(k)
        if got is None:
            say(f'    {k:<14} not in MasterVariables — cannot verify'); continue
        lim = abstol if abstol is not None else tol * max(abs(want), 1e-12)
        ok = abs(got - want) <= lim
        say(f'    {k:<14} asked {want:<12.6g} reads {got:<12.6g} {"ok" if ok else "** MISMATCH"}')
        if not ok:
            bad.append(k)
    return bad

def _scanning():
    return bool(igor.DataFolder(r'root:packages:MFP3D').Wave('OutWaves').GetTextWavePointValue(0, 0))

def grab(base, timeout_s=1800):
    """Acquire one scan and return the file Igor ACTUALLY wrote (never a predicted name)."""
    inst.a.set_folder()
    igor.Execute(f'root:Packages:MFP3D:Main:Variables:BaseName = "{base}"')
    igor.Execute('ARCheckSuffix()')
    pat = os.path.join(inst.a.file_loc, f'{base}*.ibw')
    before = {p: os.path.getmtime(p) for p in glob.glob(pat)}
    t0 = time.time()
    inst.a.ex('DownScan_0', 'MasterPanel')
    time.sleep(10)
    while _scanning() and time.time() - t0 < timeout_s:
        time.sleep(2)
    if time.time() - t0 >= timeout_s:
        say('    WARNING: scan did not finish within the timeout')
    time.sleep(3)
    new = [p for p in glob.glob(pat) if p not in before or os.path.getmtime(p) > before[p]]
    if not new:
        raise RuntimeError(f'scan finished but no new {base}*.ibw in {inst.a.file_loc}')
    path = max(new, key=os.path.getmtime)
    last = -1
    for _ in range(20):                                   # wait for the write to settle
        s = os.path.getsize(path)
        if s == last and s > 0:
            break
        last = s; time.sleep(0.5)
    say(f'    -> {os.path.basename(path)} ({os.path.getsize(path)/1e6:.2f} MB, {(time.time()-t0)/60:.1f} min)')
    return path

def null_from_sweep(records, f_cr):
    """Complex least squares Z(V) = a + bV over a narrow CR1 band -> the null voltage."""
    V = np.array([r[0] for r in records], float)
    Zs = []
    for _, td in records:
        f, Z = tune_to_complex(td, band_Hz=(f_cr - 3.0e3, f_cr + 3.0e3))
        Zs.append(Z)
    n = min(z.size for z in Zs)
    Zm = np.array([z[:n] for z in Zs])                    # (nV, nf)
    a, b = np.linalg.pinv(np.c_[np.ones_like(V), V]) @ Zm
    w = np.abs(b)
    r = -(a * np.conj(b)).real / np.maximum(np.abs(b) ** 2, 1e-30)
    return float((r * w).sum() / w.sum()), float(np.abs(b).mean())

out = dict(started=time.strftime('%Y-%m-%d %H:%M:%S'), frame=FRAME, load_nN=IMG_LOAD_NN,
           sweep={}, images=[], nulls={})
say('=' * 78)
say(f'BIAS-SERIES IMAGING  (queued {time.strftime("%Y-%m-%d")}; runs after the dense map + sparse capture)')
say('=' * 78)
try:
    # ---- 1. laser to the free end, recalibrate -------------------------------------
    say(f'laser -> x = {LASER_X_UM:.0f} um, AutoWedge + InvOLS')
    inst._goto_and_prepare(LASER_X_UM, capture_image=False)
    if inst._invols is None:                    # only if recalibrate_each got turned off
        inst._invols = inst.a.measure_invols()
    if inst._spring is None:
        inst._spring = inst.a.get_gmv()['SpringConstant']
    say(f'  InvOLS {inst._invols:.4e} m/V, k {inst._spring:.5f} N/m')
    out['invols_m_per_V'] = float(inst._invols); out['spring_N_per_m'] = float(inst._spring)

    # ---- 2. pin BOTH nulls at the imaging contact -----------------------------------
    idx = 900
    for spot in (SPOT_1, SPOT_2):
        say(f'\nbias sweep on spot {spot}: {SWEEP_V} V at {IMG_LOAD_NN:.1f} nN')
        inst.goto_spot(spot)
        inst.set_load(IMG_LOAD_NN, reengage=True)
        recs, f_cr_spot = [], None
        for v in SWEEP_V:
            inst.set_dc_bias(v)
            tag = f'{"m" if v < 0 else "p"}{abs(v):.3f}V'.replace('.', 'p')
            t = inst.a.tune_eigenmode(position_label=f'PREIMG_S{spot}_DC{tag}',
                                      scan_index=idx, save_tune_data=True)
            idx += 1
            if t['tune_data'] is None:
                say(f'  {v:+.2f} V  tune returned nothing — skipped'); continue
            if f_cr_spot is None:
                f_cr_spot = t['resonance_freq']
            recs.append((v, t['tune_data']))
            say(f'  {v:+5.2f} V   f_res {t["resonance_freq"]/1e3:7.3f} kHz   Q {t["q_factor"]:.0f}')
        nv, bmag = null_from_sweep(recs, f_cr_spot)
        out['sweep'][str(spot)] = dict(f_cr_Hz=float(f_cr_spot), null_V=nv, b_mag=bmag,
                                       biases=[r[0] for r in recs])
        out['nulls'][str(spot)] = nv
        say(f'  spot {spot}: null at {nv:+.3f} V   (bias survey said '
            f'{NULL_FALLBACK[0] if spot == SPOT_1 else NULL_FALLBACK[1]:+.2f} V)')
    inst.set_dc_bias(0.0)

    n1 = out['nulls'].get(str(SPOT_1), NULL_FALLBACK[0])
    n2 = out['nulls'].get(str(SPOT_2), NULL_FALLBACK[1])
    if not (-2.0 < n1 < 6.0 and -2.0 < n2 < 6.0):
        say(f'  nulls {n1:+.2f}/{n2:+.2f} V are outside the plausible window — '
            f'falling back to the bias-survey values {NULL_FALLBACK}')
        n1, n2 = NULL_FALLBACK
    say(f'\nimaging setpoints: 0 V, {n2:+.2f} V (spot-2 null), {0.5*(n1+n2):+.2f} V (mean), '
        f'{n1:+.2f} V (spot-1 null), 0 V again')

    # ---- 3. back to the DomBias0021 frame -------------------------------------------
    say('\nrestoring the DomBias0021 scan frame')
    inst.a.withdraw(); time.sleep(1)
    for k, v in FRAME.items():
        pv(k, v)
    pv('DriveAmplitude', IMG_DRIVE_V)
    time.sleep(1.5)
    bad = verify({k: FRAME[k] for k in ('ScanSize', 'ScanPoints', 'ScanLines', 'ScanRate')},
                 tol=0.02)
    # offsets get an absolute tolerance: 2 % of a 0.43 um YOffset is meaninglessly tight
    bad += verify({k: FRAME[k] for k in ('XOffset', 'YOffset')}, abstol=5e-8)
    hard = [k for k in bad if k in ('ScanSize', 'XOffset', 'YOffset')]
    if hard:
        raise RuntimeError(f'scan frame did not take: {hard} — refusing to image the wrong area')
    inst.set_load(IMG_LOAD_NN, reengage=True)
    t0 = inst.a.tune_eigenmode(position_label='IMGTUNE_start', scan_index=idx, save_tune_data=True)
    idx += 1
    F_DRIVE = float(t0['resonance_freq'])
    say(f'  contact resonance here: {F_DRIVE/1e3:.3f} kHz, Q {t0["q_factor"]:.0f}  '
        f'-> fixed drive for images 1-5')
    out['f_drive_Hz'] = F_DRIVE

    # ---- 4. the images ---------------------------------------------------------------
    plan = [('cr1_0V',   0.0,              F_DRIVE),
            ('cr1_null2', n2,              F_DRIVE),
            ('cr1_mean',  0.5 * (n1 + n2), F_DRIVE),
            ('cr1_null1', n1,              F_DRIVE),
            ('cr1_0V_repeat', 0.0,         F_DRIVE),
            ('ref_30kHz_0V',  0.0,         REF_FREQ_HZ)]
    for name, vdc, fdr in plan:
        say(f'\n--- {name}:  V_dc = {vdc:+.3f} V,  drive {fdr/1e3:.3f} kHz ---')
        inst.set_dc_bias(vdc)
        t = inst.a.tune_eigenmode(position_label=f'IMGTUNE_{name}', scan_index=idx,
                                  save_tune_data=True)
        idx += 1
        say(f'    tune now: f_res {t["resonance_freq"]/1e3:.3f} kHz '
            f'(drift {(t["resonance_freq"]-F_DRIVE)/1e3:+.3f} kHz), Q {t["q_factor"]:.0f}')
        pv('DriveFrequency', fdr)
        pv('DriveAmplitude', IMG_DRIVE_V)
        time.sleep(1.0)
        if verify({'DriveFrequency': fdr, 'DriveAmplitude': IMG_DRIVE_V}, tol=0.002):
            raise RuntimeError('drive did not take — refusing to image on an unverified panel')
        path = grab(BASE)
        out['images'].append(dict(name=name, bias_V=vdc, drive_Hz=fdr, path=path,
                                  tune_f_res_Hz=float(t['resonance_freq']),
                                  tune_q=float(t['q_factor']),
                                  time=time.strftime('%H:%M:%S')))
        with open(RESULT, 'w') as fh:
            json.dump(out, fh, indent=1)
    say('\nall six images done')

except Exception as e:
    say(f'\n*** FAILED: {type(e).__name__}: {e}')
    out['error'] = f'{type(e).__name__}: {e}'
    import traceback; traceback.print_exc()
finally:
    try:
        inst.set_dc_bias(0.0)
    except Exception as e:
        say(f'  (bias reset failed: {e})')
    try:
        inst.a.withdraw()
    except Exception as e:
        say(f'  (withdraw failed: {e})')
    out['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    try:
        with open(RESULT, 'w') as fh:
            json.dump(out, fh, indent=1)
    except Exception:
        pass
    say(f'tip withdrawn, bias 0 V. results -> {RESULT}')
    say('=' * 78)
    _lf.close()

bias_image_out = out
