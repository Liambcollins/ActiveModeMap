"""Which arm wins across the FULL measured band, 25 kHz - 1.775 MHz?

The band-A study scored 270-450 kHz. That window contains one flexural mode and
its antiresonance; the measurement contains two contact modes (292.97 and
902.41 kHz), a third feature at 970.6 kHz, and ~1.5 MHz of everything else.
Scoring on the wide band is a different question and it has a different answer,
because the arms do not even have the same DOMAIN:

  GP only            defined at every measured bin -- it is a per-frequency
                     interpolator over position and knows nothing about f
  EB / FEM + GP       defined only where the library covers u = f/f_res.  The
  (band-A libraries)  band-A libraries span u = 0.55-2.05, i.e. 161-600 kHz at
                     the fitted f_res = 293 kHz.  Mode 2 is at u = 3.08 and is
                     simply OUTSIDE. predict_cplx CLIPS, so asking for it
                     returns the library edge -- plausible-looking and wrong.
  EB + GP (extended)  u = 0.55-3.45, i.e. 161-1010 kHz. Covers both modes.

So the honest comparison masks each arm to its own coverage and says so.
Scores are median |dB| error on HELD-OUT positions, region by region, plus the
measured SNR of each region so it is clear where the data itself carries
information.
"""
import sys, gc
import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_ROOT, 'config.py')):
    _ROOT = _os.path.dirname(_ROOT)
sys.path[:0] = [_ROOT, _os.path.join(_ROOT, 'src'),
                _os.path.join(_ROOT, 'figures')]
from config import (DENSE_GRID_A, FEM_LADDER, EB_REPO, EB_GEOMETRY, OUT, FIG,
                    add_eb_to_path)
add_eb_to_path()
import numpy as np, pandas as pd, pickle
import cplxrec as CR, physrec as PR
import activemodemap as amm

FDEC = 4
NS = [5, 6, 8, 12, 20, 30]
D = str(DENSE_GRID_A)
import os as _os
LIBS = {'eb_gp': str(OUT) + '/eblib_cplx.npz',
        'fem_gp': str(OUT) + '/femlib_isotropic_cplx.npz',
        'eb2_gp': str(OUT) + '/eblib_x_cplx.npz'}
ONLY = _os.environ.get('AMM_ONLY')

# ---------------------------------------------------------------- the wide data
dn = np.load(D + '/DenseReference.npz', allow_pickle=True)
x = dn['x_um'].astype(float)
dl = pd.read_csv(D + '/DenseReference_log.csv', parse_dates=['timestamp'])
dl['i'] = dl.tune_file.str.extract(r'_(\d{4})\.txt$')[0].astype(int)
ps = amm.fit_position_scale(dl[dl.i < 20], dl[dl.i >= 20], verbose=False)
x = ps.apply(x)
f = dn['freq_Hz'][::FDEC]
Z = dn['Z'][:, ::FDEC]
A = np.abs(Z)
print(f'{Z.shape[0]} positions x {Z.shape[1]} bins, {f[0]/1e3:.0f}-'
      f'{f[-1]/1e3:.0f} kHz  (decimated x{FDEC})', flush=True)

# ---------------------------------------------------------------- regions
# Fixed frequency edges, not data-driven, so the regions mean the same thing for
# every arm and every budget.
EDGES = [('below mode 1', 25e3, 265e3),
         ('mode 1 ridge', 285e3, 302e3),
         ('mode 1 anti-res', 302e3, 500e3),
         ('valley 1-2', 500e3, 700e3),
         ('mode 2 ridge', 880e3, 925e3),
         ('mode 2 flank', 700e3, 880e3),
         ('above mode 2', 925e3, 1775e3)]
REG = {name: (f >= lo) & (f < hi) for name, lo, hi in EDGES}
FLOOR = np.median(A[:, f > 1500e3])
print(f'\nnoise floor (median |Z| above 1.5 MHz) = {FLOOR:.2e}')
print(f"{'region':>16} {'bins':>5} {'median |Z|':>11} {'SNR':>7}")
for name, m in REG.items():
    med = float(np.median(A[:, m]))
    print(f'{name:>16} {m.sum():5d} {med:11.3e} {med/FLOOR:7.1f}')

SELS = {n: sorted(set(np.argmin(np.abs(x[:, None] -
        np.linspace(x.min(), x.max(), n)[None, :]), axis=0).tolist()))
        for n in NS}


SIG = np.median(A, axis=0) > 3.0 * np.median(A[:, f > 1500e3])


def score(o, sel, cov=None):
    """Median |dB| per region on held-out positions, masked to coverage.

    Reported twice: over ALL bins in scope, and over SIGNAL bins only (median
    |Z| above 3x the noise floor).  The all-bins number flatters any smooth
    interpolator, because ~85 % of the measured span is at the floor and
    'reconstructing' noise to 0.3 dB is not an achievement -- the +F in the dB
    ratio makes two noise realisations agree.  The signal-bin number is the one
    that means something.
    """
    held = np.array([i for i in range(len(x)) if i not in set(sel)])
    Ar = np.abs(o['Zrec'])[held]
    At = A[held]
    F = PR.noise_floor(A[sel])
    dbe = np.abs(20 * np.log10((Ar + F) / (At + F)))
    out = {}
    for name, m in REG.items():
        mm = m if cov is None else (m & cov)
        if mm.sum() == 0:
            out[name] = np.nan
            out[name + '_cov'] = 0.0
            continue
        out[name] = float(np.median(dbe[:, mm]))
        out[name + '_cov'] = float(mm.sum() / max(m.sum(), 1))
    inband = np.ones(len(f), bool) if cov is None else cov
    out['all'] = float(np.median(dbe[:, inband]))
    out['frac3dB'] = float(np.mean(dbe[:, inband] <= 3.0))
    out['frac6dB'] = float(np.mean(dbe[:, inband] <= 6.0))
    out['cov_frac'] = float(inband.mean())
    sg = inband & SIG
    out['sig'] = float(np.median(dbe[:, sg])) if sg.any() else np.nan
    out['sig_frac3dB'] = float(np.mean(dbe[:, sg] <= 3.0)) if sg.any() else np.nan
    out['sig_bins'] = int(sg.sum())
    Zr, Zt = o['Zrec'][held][:, inband], Z[held][:, inband]
    out['nrmse_cplx'] = float(100 * np.linalg.norm(Zr - Zt)
                              / np.linalg.norm(Zt))
    w = np.abs(Zt)
    out['phase_deg'] = float(np.degrees(
        np.sum(w * np.abs(np.angle(Zr * np.conj(Zt)))) / w.sum()))
    return out


rows = []
GPO = {}

# ---------------------------------------------------------------- GP only
for n in NS:
    sel = SELS[n]
    o = CR.rec_gp_cplx(x, sel, Z[sel], f)
    GPO[n] = o
    r = score(o, sel)
    r.update(rec='gp', n=n, ell=o['ell'], k1=None)
    rows.append(r)
    print(f"  gp     n={n:3d}  all {r['all']:5.2f} dB  signal {r['sig']:5.2f} dB "
          f"(<=3dB {100*r['sig_frac3dB']:4.1f} %)  m1 {r['mode 1 ridge']:5.2f}"
          f"  m2 {r['mode 2 ridge']:5.2f}", flush=True)

# ---------------------------------------------------------------- physics arms
import os
for key, lib in LIBS.items():
    if ONLY and key != ONLY:
        continue
    if not os.path.exists(lib):
        print(f'  !! {lib} missing, skipping {key}')
        continue
    print(CR.load(lib), flush=True)
    two = (key == 'eb2_gp')
    for n in NS:
        sel = SELS[n]
        if two:
            o = CR.rec_twoband_cplx(x, sel, Z[sel], f)
        else:
            # fit on the band the library covers, score over all of it
            o = CR.rec_phys_cplx(x, sel, Z[sel], f, loo=False,
                                 fit_band=(265e3, 500e3))
        th = o['theta'] or {}
        cov = CR.coverage(f, th['f_res'])
        r = score(o, sel, cov)
        r.update(rec=key, n=n, ell=o['ell'], k1=th.get('k1'),
                 f_res=th.get('f_res'))
        rows.append(r)
        # the same GP reconstruction, scored on the SAME mask -- without this
        # the physics arms are being judged on a quarter of the band and the GP
        # on all of it, which is not a comparison
        rg = score(GPO[n], sel, cov)
        rg.update(rec='gp@' + key, n=n, ell=GPO[n]['ell'], k1=None)
        rows.append(rg)
        print(f"  {key:6s} n={n:3d}  covers {100*r['cov_frac']:4.1f} %  "
              f"all {r['all']:5.2f} dB  signal {r['sig']:5.2f} dB "
              f"(<=3dB {100*r['sig_frac3dB']:4.1f} %)  m1 {r['mode 1 ridge']:5.2f}"
              f"  m2 {r['mode 2 ridge']:5.2f}  k1 "
              f"{th.get('k1', float('nan')):7.1f}"
              f"   | same mask, GP only: signal {rg['sig']:5.2f} dB "
              f"(<=3dB {100*rg['sig_frac3dB']:4.1f} %)", flush=True)
    CR._LIB.clear(); gc.collect()

pd.DataFrame(rows).to_csv(str(OUT) + '/wideband.csv', index=False)
pickle.dump(dict(x=x, f=f, Z=Z, REG=REG, EDGES=EDGES, FLOOR=FLOOR, SELS=SELS,
                 rows=rows), open(str(OUT) + '/wideband_meta.pkl', 'wb'))
print('\nwrote out/wideband.csv')
