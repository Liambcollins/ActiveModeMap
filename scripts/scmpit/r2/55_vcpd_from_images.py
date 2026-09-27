r"""R2 stage 5b -- refine V_cpd from the stage-5 image series. Analysis only, no instrument.

R1 did this by hand after the fact and found the imported V_cpd was 0.56 V wrong.
Here it is part of the chain: read the CR1 frames at position A (and B if it has
two biases), take each domain's median amplitude, fit amplitude vs bias per
domain, and solve for the bias where the two domains cross. That crossing is the
electrostatic null -- the point where both domains report the same |d33| and the
phase separation is a clean 180 deg.

Domain split is by PHASE SIGN, not amplitude. R1 stage 7 found the domains'
amplitude ranking can invert as the tip evolves while the phase identity (set by
the fixed ferroelectric polarisation) does not. An amplitude-based split would
silently swap the domains; a phase-based one cannot.

The result overwrites `vcpd_current.json`, so stages 6 and 7 use a null measured
minutes earlier on the same tip in the same drive regime, rather than the
bias-survey value from two hours before.

If anything here fails, the previous `vcpd_current.json` is left untouched and
the downstream stages simply use the stage-1 value -- degraded, not broken.
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

RESULT = os.path.join(C.FILE_LOC, 'image_cr1_quant.json')
OUT = os.path.join(C.FILE_LOC, 'vcpd_current.json')
DETAIL = os.path.join(C.FILE_LOC, 'vcpd_from_images.json')
LOG = os.path.join(C.FILE_LOC, 'vcpd_from_images_log.txt')

AMP_CH, PHASE_CH = 1, 3        # AmplitudeRetrace, PhaseRetrace -- this setup's channel order

_tee = C.Tee(LOG); _old = sys.stdout; sys.stdout = _tee
say = C.stamp


def read_frame(path):
    from igor2.binarywave import load as ibw_load
    w = ibw_load(path)['wave']
    d = np.asarray(w['wData'], float)
    if d.ndim != 3 or d.shape[2] <= max(AMP_CH, PHASE_CH):
        raise ValueError(f'{os.path.basename(path)}: unexpected wave shape {d.shape}')
    amp, ph = d[:, :, AMP_CH], d[:, :, PHASE_CH]
    if not (np.nanmax(ph) - np.nanmin(ph) > 20.0):
        raise ValueError(f'{os.path.basename(path)}: channel {PHASE_CH} does not look like degrees '
                         f'(range {np.nanmin(ph):.1f}..{np.nanmax(ph):.1f})')
    return amp, ph


def domain_masks(phase):
    """Split on phase sign about the frame median -- polarisation, not brightness."""
    p = np.mod(phase - np.nanmedian(phase) + 180.0, 360.0) - 180.0
    m1, m2 = p >= 0, p < 0
    return m1, m2


try:
    say('=' * 78)
    say('R2 STAGE 5b  V_cpd FROM THE IMAGE SERIES')
    say('=' * 78)
    if not os.path.exists(RESULT):
        raise SystemExit(f'{RESULT} missing -- stage 5 must run first')
    res = json.load(open(RESULT, encoding='utf-8'))
    imgs = [im for im in res.get('images', []) if im.get('kind') == 'cr1']
    if not imgs:
        raise SystemExit('no CR1 frames recorded in the stage-5 result')

    by_pos = {}
    for im in imgs:
        by_pos.setdefault(im['stage_x_um'], []).append(im)

    # the highest-SNR frame at each position defines that position's domain mask,
    # and every frame there reuses it -- all frames share one XY raster.
    crossings, detail = {}, {}
    for x, frames in sorted(by_pos.items()):
        biases = sorted({round(f['bias_V'], 4) for f in frames})
        if len(biases) < 2:
            say(f'  x={x}: only {len(biases)} bias point(s) -- cannot extrapolate, skipped')
            continue
        ref = frames[0]
        _, ph_ref = read_frame(ref['path'])
        m1, m2 = domain_masks(ph_ref)
        say(f'  x={x}: mask from {os.path.basename(ref["path"])} -- '
            f'{m1.sum()} / {m2.sum()} px, biases {biases}')

        pts = {}
        for f in frames:
            b = round(f['bias_V'], 4)
            amp, ph = read_frame(f['path'])
            a1, a2 = float(np.nanmedian(amp[m1])), float(np.nanmedian(amp[m2]))
            sep = float(abs(np.nanmedian(ph[m1]) - np.nanmedian(ph[m2])))
            sep = min(sep, 360.0 - sep) if sep > 180.0 else sep
            pts.setdefault(b, []).append((a1, a2, sep))
            say(f'     {b:+7.3f} V  domain1 {a1*1e12:9.1f} pm   domain2 {a2*1e12:9.1f} pm   '
                f'ratio {a1/a2 if a2 else float("nan"):6.3f}   phase sep {sep:6.1f} deg')

        bs = np.array(sorted(pts), float)
        A1 = np.array([np.mean([p[0] for p in pts[b]]) for b in bs])
        A2 = np.array([np.mean([p[1] for p in pts[b]]) for b in bs])
        (s1, i1) = np.polyfit(bs, A1, 1)
        (s2, i2) = np.polyfit(bs, A2, 1)
        if abs(s1 - s2) < 1e-18:
            say(f'  x={x}: domain slopes are equal -- no crossing, skipped'); continue
        vx = float((i2 - i1) / (s1 - s2))
        crossings[x] = vx
        detail[str(x)] = dict(bias_V=bs.tolist(), amp1_m=A1.tolist(), amp2_m=A2.tolist(),
                              slope1=float(s1), slope2=float(s2), crossing_V=vx,
                              n_px=[int(m1.sum()), int(m2.sum())])
        say(f'  x={x}: slopes {s1*1e12:+.1f} / {s2*1e12:+.1f} pm/V  ->  crossing {vx:+.3f} V')

    if not crossings:
        raise SystemExit('no position yielded a crossing')

    vals = np.array(list(crossings.values()), float)
    v = float(np.mean(vals))
    say('')
    say(f'crossings: ' + ', '.join(f'x={k}: {val:+.3f} V' for k, val in crossings.items()))
    if vals.size > 1:
        say(f'spread across positions: {vals.max() - vals.min():+.3f} V '
            f'(R1 got 0.001 V between two positions with 70x different enhancement)')
    if not (-5.0 < v < 5.0):
        raise SystemExit(f'mean crossing {v} is not physical -- not written')

    prev = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {}
    payload = dict(v_cpd_V=v, source='R2_stage5_image_extrapolation',
                   crossings={str(k): float(val) for k, val in crossings.items()},
                   previous_v_cpd_V=prev.get('v_cpd_V'), previous_source=prev.get('source'),
                   written=time.strftime('%Y-%m-%d %H:%M:%S'))
    json.dump(payload, open(OUT, 'w', encoding='utf-8'), indent=1)
    json.dump(detail, open(DETAIL, 'w', encoding='utf-8'), indent=1)
    say('')
    say(f'V_cpd for stages 6 and 7: {v:+.3f} V '
        f'(was {prev.get("v_cpd_V")} from {prev.get("source")})  ->  {OUT}')
    say('=' * 78)
except Exception as e:
    say(f'*** FAILED: {type(e).__name__}: {e}')
    say(f'    {OUT} left untouched; stages 6 and 7 will use the stage-1 value')
    import traceback; traceback.print_exc()
    raise
finally:
    sys.stdout = _old; _tee.close()
