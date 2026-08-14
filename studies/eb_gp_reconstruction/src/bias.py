"""Bias-dependence analysis of the SCM-PIT series, rebuilt from the raw tunes.

Reproduces the 2026-08-10 finding and produces the numbers the two bias slides
need.  Method notes that matter:

* Sample each spectrum at ITS OWN resonance peak.  Mode A's f_res moves a few
  hundred Hz with bias; reading a fixed frequency bin on a Q ~ 200 line turns
  that shift into fake non-linearity (worth ~4x in the apparent residual).
* Fit Z_peak = a + b V in the COMPLEX plane.  A single force source scaling
  linearly with (V - V_cpd) traces a straight line through the origin; the
  perpendicular distance from the origin to the fitted line, |P_perp|, is the
  part no choice of bias can cancel, and that is the piezoresponse bound.
* V_cpd is where the line comes closest to the origin.
"""
import numpy as np

# Search band for mode A.  Deliberately TIGHT: the resonance moves only a few
# hundred Hz across +/-6 V, while a wide band lets the peak finder hop to mode B
# (or to the antiresonance shoulder) at the bias and position where mode A's
# amplitude collapses -- the artefact that made the saved Series_vs_condition.png
# look like the resonance jumps between 293 and 905 kHz.
BAND_A = (285e3, 302e3)


def load(path):
    d = np.load(path, allow_pickle=True)
    return d['x_um'].astype(float), d['freq_Hz'], d['Z'], d['conditions']


def peak_values(f, Z, band=BAND_A):
    """Z at each spectrum's own in-band peak, plus that peak's frequency."""
    m = (f >= band[0]) & (f <= band[1])
    fb, Zb = f[m], Z[..., m]
    j = np.argmax(np.abs(Zb), axis=-1)
    Zp = np.take_along_axis(Zb, j[..., None], axis=-1)[..., 0]
    return Zp, fb[j], fb, Zb


def fit_line(V, Zp):
    """Complex least squares Z = a + b V, with per-point residual scatter."""
    A = np.stack([np.ones_like(V), V], axis=1)
    coef, *_ = np.linalg.lstsq(A, Zp, rcond=None)
    a, b = coef
    res = Zp - A @ coef
    dof = max(len(V) - 2, 1)
    s2 = float(np.sum(np.abs(res) ** 2) / dof)          # per complex point
    cov = np.linalg.inv(A.conj().T @ A).real * s2
    return a, b, res, s2, cov


def perp(a, b):
    """Distance from the origin to the line a + bV, and the V that attains it."""
    v0 = -float(np.real(a * np.conj(b)) / (np.abs(b) ** 2))
    return float(np.abs(a + b * v0)), v0


def q_from_fwhm(fb, Ab):
    j = int(np.argmax(Ab))
    h = Ab[j] / np.sqrt(2)
    lo = j
    while lo > 0 and Ab[lo] > h:
        lo -= 1
    hi = j
    while hi < len(Ab) - 1 and Ab[hi] > h:
        hi += 1
    # linear interpolation onto the half-power crossings
    def cross(i0, i1):
        y0, y1 = Ab[i0], Ab[i1]
        return fb[i0] + (h - y0) / (y1 - y0) * (fb[i1] - fb[i0])
    flo = cross(lo, lo + 1) if lo + 1 <= j else fb[lo]
    fhi = cross(hi, hi - 1) if hi - 1 >= j else fb[hi]
    return float(fb[j] / (fhi - flo)), float(fb[j]), float(fhi - flo)


def analyse(path):
    x, f, Z, cond = load(path)
    import json
    V = np.array([c['bias_V'] for c in json.loads(str(cond))], float)
    Zp, fpk, fb, Zb = peak_values(f, Z)          # Zp (nbias, npos)
    out = dict(x=x, V=V, Zp=Zp, fpk=fpk, fb=fb, Zb=Zb)
    rows = []
    for i in range(len(x)):
        a, b, res, s2, cov = fit_line(V, Zp[:, i])
        P, v0 = perp(a, b)
        se = np.sqrt(s2 / len(V))                # sd of the perpendicular offset
        Q, f0, fw = q_from_fwhm(fb, np.abs(Zb[3, i]))   # V = 0 spectrum
        rows.append(dict(x=x[i], a=a, b=b, absb=abs(b), P=P, v_cpd=v0,
                         se=se, PoverSE=P / se, Q=Q, f0=f0, fwhm=fw,
                         lin_resid_pct=100 * np.sqrt(s2) / np.abs(Zp[:, i]).max()))
    out['rows'] = rows
    return out


if __name__ == '__main__':
    import sys, os as _os
    _D = _os.path.dirname(_os.path.abspath(__file__))
    while not _os.path.exists(_os.path.join(_D, 'config.py')):
        _D = _os.path.dirname(_D)
    sys.path.insert(0, _D)
    from config import BIAS_SERIES
    r = analyse(sys.argv[1] if len(sys.argv) > 1 else str(BIAS_SERIES))
    print(f"{'x um':>7} {'|b| /V':>10} {'|Pperp|':>10} {'P/SE':>7} "
          f"{'V_cpd':>7} {'Q':>6} {'f0 kHz':>9} {'lin%':>6}")
    for w in r['rows']:
        print(f"{w['x']:7.0f} {w['absb']:10.3e} {w['P']:10.3e} "
              f"{w['PoverSE']:7.1f} {w['v_cpd']:7.2f} {w['Q']:6.0f} "
              f"{w['f0']/1e3:9.2f} {w['lin_resid_pct']:6.2f}")
    bb = np.array([w['absb'] for w in r['rows']])
    PP = np.array([w['P'] for w in r['rows']])
    ok = bb > 0.1 * bb.max()          # exclude the mode-A null positions
    vc = np.array([w['v_cpd'] for w in r['rows']])[ok]
    print(f"\n|b| spans {bb.max()/bb.min():.0f}x ; |Pperp| spans "
          f"{PP.max()/PP.min():.0f}x")
    print(f"corr(log|Pperp|, log|b|) = "
          f"{np.corrcoef(np.log(PP), np.log(bb))[0,1]:+.2f}")
    print(f"V_cpd = {vc.mean():.2f} +/- {vc.std(ddof=1):.2f} V  "
          f"(from {ok.sum()} positions with usable mode-A sensitivity)")
