r"""Shared bits for the SCM-PIT campaign scripts (DomainsB_SCMPIT, 2026-09-20).

Every stage script imports this, reads the pre-flight result, and uses the same
Tee (with the close() the handoff asked for) and the same finish() block.
"""
import json
import os
import sys
import time

import numpy as np

REPO = r'C:\Users\Asylum User\Desktop\STAFF Software\Liam\ActiveModeMap'
FILE_LOC = r'D:\User Data\Liam\ActiveModeMap\DomainsB_SCMPIT'
BASE_FILENAME = 'SCMPIT'
PREFLIGHT_JSON = os.path.join(FILE_LOC, 'preflight_result.json')

K_LEVER_N_PER_M = 1.697          # thermal, TuneNCCal0000.ibw 2026-09-20 11:22
F0_FREE_HZ = 63.801e3
PROBE_L_UM_FALLBACK = 232.0      # earlier SCM-PIT (SCM_PIT/Dense_Grid_B); refined by pre-flight
HW_SIGN_TOWARD_FREE_END = +1     # DoLDMove(+x) goes toward the tip on this Cypher

if REPO not in sys.path:
    sys.path.insert(0, REPO)


class Tee:
    """Mirror stdout into a log file. Unlike the old runners this one CLOSES."""

    def __init__(self, path, out=None):
        self.f = open(path, 'a', buffering=1, encoding='utf-8')
        self.out = out if out is not None else sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()

    def close(self):
        try:
            self.f.flush()
            self.f.close()
        except Exception:
            pass


def stamp(m=''):
    print(f'{time.strftime("%H:%M:%S")}  {m}', flush=True)


def load_preflight(require=True):
    if not os.path.exists(PREFLIGHT_JSON):
        if require:
            raise SystemExit(f'pre-flight result missing: {PREFLIGHT_JSON} -- run 01_preflight.py first')
        return None
    with open(PREFLIGHT_JSON, 'r', encoding='utf-8') as f:
        p = json.load(f)
    for k in ('probe_L_um', 'x_lo_um', 'x_hi_um', 'cr1_Hz'):
        if p.get(k) is None:
            raise SystemExit(f'pre-flight result has no {k!r}; refusing to guess')
    return p


def cr1_band(p, lo_frac=0.80, hi_frac=1.25):
    """Analysis band around the measured CR1 (resonance + antiresonance above it)."""
    f1 = float(p['cr1_Hz'])
    return (lo_frac * f1, hi_frac * f1)


def check_inst(inst, p=None):
    """Refuse to run on an instrument whose bookkeeping does not match the campaign."""
    if inst is None:
        raise SystemExit("no `inst` in the namespace -- run 00_setup_inst.py first")
    k = getattr(inst, '_spring', None)
    if k and abs(k - K_LEVER_N_PER_M) / K_LEVER_N_PER_M > 0.10:
        print(f'  NOTE: panel spring constant {k:.4f} N/m vs campaign {K_LEVER_N_PER_M} N/m')
    if p is not None and inst.x_limits_um is not None:
        lo, hi = inst.x_limits_um
        if lo > p['x_lo_um'] + 1e-6 or hi < p['x_hi_um'] - 1e-6:
            print(f'  NOTE: inst.x_limits_um {inst.x_limits_um} narrower than pre-flight span '
                  f'({p["x_lo_um"]}, {p["x_hi_um"]}) -- widening to the measured span')
            inst.x_limits_um = (float(p['x_lo_um']), float(p['x_hi_um']))
            inst.span_um = float(p['probe_L_um'])


def finish(inst, t_start=None, tag=''):
    """The standing rule: bias to 0 V, withdraw, never inst.close()."""
    try:
        inst.set_dc_bias(0.0)
    except Exception as e:
        print('  bias reset:', e)
    try:
        inst.a.withdraw()
    except Exception as e:
        print('  withdraw:', e)
    extra = f' -- {(time.time() - t_start) / 60:.1f} min wall clock' if t_start else ''
    stamp(f'=== {tag} finished{extra} (tip withdrawn, bias 0 V) ===')


def find_peaks_simple(freq, amp, n=3, min_sep_Hz=60e3, floor_q=0.2, min_snr_dB=8.0,
                      f_min_Hz=50e3):
    """Top-n resonance peaks of |Z| without scipy: local maxima ranked by height,
    at least `min_sep_Hz` apart, at least `min_snr_dB` above the amplitude floor.
    Ignores everything below `f_min_Hz` (the DC / 1/f spike is not a resonance)."""
    freq = np.asarray(freq, float)
    amp = np.asarray(amp, float)
    good = np.isfinite(freq) & np.isfinite(amp) & (amp > 0) & (freq >= f_min_Hz)
    f, a = freq[good], amp[good]
    floor = np.quantile(a, floor_q)
    idx = np.argsort(a)[::-1]
    picked = []
    for i in idx:
        if 20 * np.log10(a[i] / floor) < min_snr_dB:
            break
        if 0 < i < a.size - 1 and not (a[i] >= a[i - 1] and a[i] >= a[i + 1]):
            continue
        if all(abs(f[i] - f[j]) >= min_sep_Hz for j in picked):
            picked.append(i)
        if len(picked) >= n:
            break
    picked.sort(key=lambda j: f[j])
    out = []
    for j in picked:
        # FWHM by half-max crossings around the peak
        half = a[j] / np.sqrt(2.0)
        lo = j
        while lo > 0 and a[lo] > half:
            lo -= 1
        hi = j
        while hi < a.size - 1 and a[hi] > half:
            hi += 1
        out.append(dict(f_Hz=float(f[j]), amp=float(a[j]),
                        snr_dB=float(20 * np.log10(a[j] / floor)),
                        fwhm_Hz=float(f[hi] - f[lo])))
    return out
