"""Smooth peak-frequency and Q evaluation for fmmpaper.eb_fit (import to patch). See xprobe_semgeom_blind_v2.py."""
import numpy as np
from fmmpaper import eb_fit
_peaks0 = eb_fit.peaks
def peaks_smooth(p, st, f_lo=None, f_hi=None, n=2500):
    fk = _peaks0(p, st, f_lo, f_hi, n)
    if fk is None: return None
    out = []
    for f in fk:
        ff = np.linspace(0.985 * f, 1.015 * f, 401)
        m, r = eb_fit._model(p, st, ff)
        i = np.argmin(np.abs(m.xi - 0.55)); a = np.abs(r["piezo"][:, i]); k = int(np.argmax(a))
        if 0 < k < a.size - 1:
            y0, y1, y2 = np.log(a[k - 1:k + 2]); den = y0 - 2 * y1 + y2
            d = 0.5 * (y0 - y2) / den if den != 0 else 0.0
            out.append(m.f_hz[k] + d * (m.f_hz[1] - m.f_hz[0]))
        else: out.append(m.f_hz[k])
    return np.sort(np.array(out))

def q_smooth(p, st, f1):
    for span in (0.04, 0.12):
        ff = np.linspace(f1 * (1 - span), f1 * (1 + span), 2401)
        m, r = eb_fit._model(p, st, ff)
        i = np.argmin(np.abs(m.xi - 0.55)); a = np.abs(r["piezo"][:, i]); j = int(np.argmax(a)); h = a[j] / np.sqrt(2)
        lo = j
        while lo > 0 and a[lo] > h: lo -= 1
        hi = j
        while hi < a.size - 1 and a[hi] > h: hi += 1
        if lo > 0 and hi < a.size - 1: break
    fl = np.interp(h, [a[lo], a[lo + 1]], [ff[lo], ff[lo + 1]]); fh = np.interp(h, [a[hi], a[hi - 1]], [ff[hi], ff[hi - 1]])
    return f1 / (fh - fl)

eb_fit.peaks = peaks_smooth; eb_fit.q_model = q_smooth
