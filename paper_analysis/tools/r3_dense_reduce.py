"""Reduce the R3 dense map (456 MB npz, 186 x 266656 complex) on the laptop: per position CR1/CR2 peaks and Q,
QS band mean, and decimated CR1/CR2 band slices. Output results/r3r4/r3_dense_reduced.npz (+ json summary)."""
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from fmmpaper import registry
from fmmpaper.spectra import peak
root = Path(sys.argv[1]) / "DomainsB_SCMPIT_R3"; pr = registry.PROBES["multi75g"]; t0 = time.time()
d = np.load(root / "75_dense_grid/dense_map_1um_350nN_0V_checkpoint.npz", allow_pickle=True)
x = np.asarray(d["x_um"], float); f = np.asarray(d["freq_Hz"], float); Z = d["Z"]
Z = Z[0] if Z.ndim == 3 else Z; print("loaded", Z.shape, Z.dtype, f"{time.time()-t0:.0f} s", str(d["phase_convention"]))
if str(d["phase_convention"]) != "model": Z = np.conj(Z)
qs = (f >= pr.qs_band_Hz[0]) & (f <= pr.qs_band_Hz[1]); zq = Z[:, qs].mean(1)
out = dict(x_label=x, zq=zq)
for b in ("CR1", "CR2"):
    lo, hi = pr.bands_Hz[b]; lo, hi = (360e3, 400e3) if b == "CR1" else (1120e3, 1180e3)
    m = (f >= lo) & (f <= hi); idx = np.flatnonzero(m)[::4]
    out[f"f_{b}"] = f[idx]; out[f"Z_{b}"] = Z[:, idx].astype(np.complex64)
    pk = [peak(f, Z[j], (lo, hi)) for j in range(len(x))]
    out[f"fpk_{b}"] = np.array([p["f_Hz"] for p in pk]); out[f"Q_{b}"] = np.array([p["Q"] for p in pk]); out[f"zpk_{b}"] = np.array([p["z"] for p in pk])
del Z
np.savez_compressed(HERE.parent / "results/r3r4/r3_dense_reduced.npz", **out)
print(f"done {time.time()-t0:.0f} s; CR1 {np.median(out['fpk_CR1'])/1e3:.2f} kHz (sd {np.std(out['fpk_CR1']):.0f} Hz), Q {np.nanmedian(out['Q_CR1']):.0f}; CR2 {np.median(out['fpk_CR2'])/1e3:.2f} kHz Q {np.nanmedian(out['Q_CR2']):.0f}")
