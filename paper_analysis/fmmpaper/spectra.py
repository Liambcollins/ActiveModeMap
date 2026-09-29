"""Spectral feature extraction: peaks, Q, antiresonances, per-position mode profiles."""
from __future__ import annotations

import numpy as np


def _parab(f, v, j):
    if 0 < j < len(v) - 1:
        a, b, c = v[j - 1], v[j], v[j + 1]
        den = a - 2 * b + c
        if abs(den) > 1e-30:
            return f[j] + 0.5 * (a - c) / den * (f[j] - f[j - 1])
    return f[j]


def peak(freq, z, band):
    """Largest |z| peak inside band. Returns dict(f_Hz, amp, phase_deg, z, k, Q)."""
    m = (freq >= band[0]) & (freq <= band[1])
    fb = freq[m]; zb = z[..., m]; a = np.abs(zb)
    k = int(np.argmax(a))
    return dict(f_Hz=float(_parab(fb, a, k)), amp=float(a[k]),
                phase_deg=float(np.degrees(np.angle(zb[k]))), z=complex(zb[k]),
                k=int(np.flatnonzero(m)[k]), Q=q_factor(fb, a, k))


def q_factor(fb, a, k):
    """Q from the interpolated -3 dB (half-power) width around index k."""
    half = a[k] / np.sqrt(2); lo = k; hi = k
    while lo > 0 and a[lo] > half:
        lo -= 1
    while hi < a.size - 1 and a[hi] > half:
        hi += 1
    if lo == 0 or hi == a.size - 1 or hi <= lo:
        return float("nan")
    fl = np.interp(half, [a[lo], a[lo + 1]], [fb[lo], fb[lo + 1]])
    fh = np.interp(half, [a[hi], a[hi - 1]], [fb[hi], fb[hi - 1]])
    return float(_parab(fb, a, k) / (fh - fl)) if fh > fl else float("nan")


def peaks_along_x(freq, Zmap, band):
    """Per-position peak frequency, amplitude, complex value and Q. Zmap[pos, freq]."""
    rows = [peak(freq, Zmap[i], band) for i in range(Zmap.shape[0])]
    return {k: np.array([r[k] for r in rows]) for k in ("f_Hz", "amp", "phase_deg", "z", "Q", "k")}


def on_resonance_profile(freq, Zmap, band):
    """Complex response vs position at the map-wide resonance (median peak) frequency."""
    f0 = float(np.median(peaks_along_x(freq, Zmap, band)["f_Hz"]))
    j = int(np.argmin(np.abs(freq - f0)))
    return f0, Zmap[:, j]


def nodes_from_profile(x, amp, min_rel=0.5, window=3, merge_um=8.0):
    """Interior amplitude minima (nodes) of a mode profile. Same rule as r2an/s1_structure."""
    amp = np.asarray(amp); med = np.median(amp); nodes = []
    for i in range(2, amp.size - 2):
        seg = amp[max(0, i - window):i + window + 1]
        if amp[i] == seg.min() and amp[i] < min_rel * med:
            nodes.append(float(x[i]))
    merged = []
    for nd in nodes:
        if not merged or nd - merged[-1][-1] > merge_um:
            merged.append([nd])
        else:
            merged[-1].append(nd)
    return [float(np.mean(g)) for g in merged]


def dns_signed(x, v):
    """Detection null: sign change of the on-resonance complex response along x.

    ``v`` is complex (one value per position). Rotated onto its dominant axis first.
    Returns dict(dns, crossings, true_zero, depth). Ported from Manuscript_FMM/core.py.
    """
    v = np.asarray(v)
    ph = np.angle(np.sum(v * np.abs(v)))
    s = (v * np.exp(-1j * ph)).real
    cr = [x[i] + (x[i + 1] - x[i]) * s[i] / (s[i] - s[i + 1])
          for i in range(len(x) - 1) if s[i] * s[i + 1] < 0]
    depth = float(np.min(np.abs(v)) / np.max(np.abs(v)))
    return dict(dns=float(max(cr)) if cr else float("nan"),
                crossings=[float(c) for c in cr], true_zero=bool(cr), depth=depth)


def to_db(a, ref=None):
    a = np.abs(a)
    ref = np.nanmax(a) if ref is None else ref
    return 20 * np.log10(np.maximum(a, 1e-30) / ref)
