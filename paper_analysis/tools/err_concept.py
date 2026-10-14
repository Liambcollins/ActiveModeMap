"""Supplementary Fig. 9 data (python tools/err_concept.py, ~1.5 min): held-out error vs N per band, with noise and SVD floors,
plus errors in derived quantities (resonance frequency, peak amplitude, node position)."""
import sys, time, json
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fmmpaper as F
from fmmpaper import ebfit, physrec, recon, semprior
PRIOR = sys.argv[1] if len(sys.argv) > 1 else "dense"

NS = [3, 4, 5, 6, 8, 10, 12, 16, 20]
X0 = -19.0
OUT = F.config.RESULTS_DIR / f"err_concept_soft{'_sem' if PRIOR == 'sem' else ''}.npz"

s = F.load("ppp_dense_1um"); xs, f, Z = s.x_um, s.freq_Hz, s.Z[0]; x = xs - X0
bands = dict(s.probe.bands_Hz); names = list(bands)
cal, FREE = (semprior.calibration("soft", X0) if PRIOR == "sem" else (ebfit.load_calibration("soft", "mass"), ebfit.FREE_DEFAULT))
M = {b: (f >= bands[b][0]) & (f <= bands[b][1]) for b in names}
Zb = {b: Z[:, M[b]] for b in names}; fbn = {b: f[M[b]] for b in names}
del Z
nr = lambda a, b: float(100 * np.linalg.norm(a - b) / np.linalg.norm(b))
interp = lambda ft, ff, A: (np.apply_along_axis(lambda r: np.interp(ft, ff, r), 1, A.real)
                            + 1j * np.apply_along_axis(lambda r: np.interp(ft, ff, r), 1, A.imag))
SM = 9  # bins (~210 Hz) smoothing for peak picking, applied to measured and reconstructed alike


def peak(A, ff):
    a = uniform_filter1d(np.abs(A), SM, axis=-1); i = np.argmax(a, -1)
    i = np.clip(i, 1, a.shape[-1] - 2); r = np.arange(a.shape[0])
    y0, y1, y2 = a[r, i - 1], a[r, i], a[r, i + 1]
    d = 0.5 * (y0 - y2) / np.where(np.abs(y0 - 2 * y1 + y2) > 0, y0 - 2 * y1 + y2, 1)
    return ff[i] + np.clip(d, -1, 1) * (ff[1] - ff[0]), y1


def linewidth(A, ff):
    a = uniform_filter1d(np.abs(A), SM, axis=-1); out = []
    for row in a:
        i = np.argmax(row); h = row[i] / np.sqrt(2)
        lo = i
        while lo > 0 and row[lo] > h: lo -= 1
        hi = i
        while hi < len(row) - 1 and row[hi] > h: hi += 1
        out.append(ff[hi] - ff[lo])
    return np.array(out)


def nodes(prof, xx, edge=5.0):
    p = uniform_filter1d(prof, 3)
    i = [k for k in range(1, len(p) - 1) if p[k] < p[k - 1] and p[k] <= p[k + 1] and p[k] < 0.35 * p.max()]
    return np.array([xx[k] for k in i if xx[0] + edge < xx[k] < xx[-1] - edge])


# measured references
ref = {}
for b in names:
    fp_, ap_ = peak(Zb[b], fbn[b]); lw = linewidth(Zb[b], fbn[b])
    fres = float(np.median(fp_)); j = int(np.argmin(np.abs(fbn[b] - fres)))
    prof = np.abs(uniform_filter1d(Zb[b].real, SM, axis=1) + 1j * uniform_filter1d(Zb[b].imag, SM, axis=1))[:, j]
    ref[b] = dict(fp=fp_, ap=ap_, lw=lw, j=j, nodes=nodes(prof, x))
    print(b, "f_res %.1f kHz, median linewidth %.0f Hz, nodes" % (fres / 1e3, np.median(lw)), np.round(ref[b]["nodes"], 1), flush=True)

# floors
floor = {}
for b in names:
    A = Zb[b]; mid = 0.5 * (A[:-2] + A[2:]); c = A[1:-1]
    noise = nr(mid, c) / np.sqrt(1.5)
    U, S_, Vh = np.linalg.svd(A, full_matrices=False)
    svd = {r: nr((U[:, :r] * S_[:r]) @ Vh[:r], A) for r in (3, 6)}
    fm, am = peak(mid, fbn[b])
    df = np.abs(fm - ref[b]["fp"][1:-1]) / ref[b]["lw"][1:-1] / np.sqrt(1.5)
    da = 100 * np.abs(am / ref[b]["ap"][1:-1] - 1) / np.sqrt(1.5)
    floor[b] = dict(noise=noise, svd3=svd[3], svd6=svd[6], df=float(np.median(df)), da=float(np.median(da)))
    print(b, floor[b], flush=True)

ARMS = ["EB", "EB+GP", "low-rank"]
res = {a: {k: np.full((len(NS), len(names)), np.nan) for k in ("nrmse", "df", "da", "dx")} for a in ARMS}
for iN, N in enumerate(NS):
    t0 = time.time(); sel = recon.select_equispaced(x, N); h = recon.held_out(len(x), sel)
    for ib, b in enumerate(names):
        # decimate the band exactly as in fig_recovery_vs_N (band_slice on the full-f array)
        fullf = f; Zfull = np.zeros((len(x), fullf.size), complex); Zfull[:, M[b]] = Zb[b]
        zb, fb = physrec.band_slice(fullf, Zfull, bands[b], n_max=250); del Zfull
        o = ebfit.rec_eb2(xs, sel, zb[sel], fb, cal, b, free=FREE)
        rec = {"EB": interp(fbn[b], fb, o["Zeb"]), "EB+GP": interp(fbn[b], fb, o["Zrec"]),
               "low-rank": recon.rec_lowrank(x, sel, Zb[b][sel], rank=min(6, N))["Zrec"]}
        for a in ARMS:
            R = rec[a]; res[a]["nrmse"][iN, ib] = nr(R[h], Zb[b][h])
            fp_, ap_ = peak(R[h], fbn[b])
            res[a]["df"][iN, ib] = np.median(np.abs(fp_ - ref[b]["fp"][h]) / ref[b]["lw"][h])
            res[a]["da"][iN, ib] = np.median(100 * np.abs(ap_ / ref[b]["ap"][h] - 1))
            nd = ref[b]["nodes"]
            if len(nd):
                prof = np.abs(uniform_filter1d(R.real, SM, axis=1) + 1j * uniform_filter1d(R.imag, SM, axis=1))[:, ref[b]["j"]]
                nr_ = nodes(prof, x)
                res[a]["dx"][iN, ib] = np.median([np.min(np.abs(nr_ - v)) if len(nr_) else 100.0 for v in nd])
        del rec
    print(f"N={N} {time.time()-t0:.0f}s  " + "  ".join(f"{a}: {np.round(res[a]['nrmse'][iN],1)}" for a in ARMS), flush=True)
    np.savez(OUT, NS=NS, names=names, floor=json.dumps(floor), **{f"{a}|{k}": v for a in ARMS for k, v in res[a].items()},
             ref_nodes=json.dumps({b: ref[b]["nodes"].tolist() for b in names}), done=iN)
print("done")
