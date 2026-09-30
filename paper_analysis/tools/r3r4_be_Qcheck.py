"""Is the BEPyAE per-pixel Q the same Q as the tune/bracket Q? Fit the frame-averaged |spectrum| (per domain)
with a -3 dB width, and a complex SHO by least squares, and compare with the median pixel Q and with the tune/bracket Q."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from fmmpaper import beimg
from fmmpaper.spectra import q_factor
root = Path(sys.argv[1]); files = sys.argv[2:]
def sho(p, f):  # A*w0^2 / (w0^2 - w^2 + i w w0/Q) * e^{i phi}
    A, f0, Q, ph = p; return A * f0 ** 2 / (f0 ** 2 - f ** 2 + 1j * f * f0 / Q) * np.exp(1j * ph)
res = {}
for fn in files:
    b = beimg.load_be(root / fn, spectra=True); S = b["S"]; fb = b["f_bins"]; lab, _ = beimg.split_domains(b["phi"])
    out = {}
    for k in (0, 1):
        m = lab == k
        if m.sum() < 200: continue
        Sm = np.mean(S[m], 0)                                 # coherent average (one domain -> common phase)
        a = np.abs(Sm); i = int(a.argmax()); Qw = q_factor(fb, a, i)
        p0 = [a[i] / Qw, fb[i], Qw, np.angle(Sm[i]) + np.pi / 2]
        r = least_squares(lambda p: np.concatenate([(sho(p, fb) - Sm).real, (sho(p, fb) - Sm).imag]), p0, x_scale=[p0[0], 1e3, 50, 1])
        # incoherent: median of per-pixel |S| normalised to each pixel's max
        out[k] = dict(n=int(m.sum()), Q_width_avg=float(Qw), Q_sho_avg=float(r.x[2]), f0_kHz=float(r.x[1] / 1e3), A_sho_avg=float(abs(r.x[0])),
                      peak_avg=float(a.max()), Q_pix_med=float(np.median(b["Q"][m])), A_pix_med=float(np.median(b["A"][m])),
                      peak_pix_med=float(np.median((b["A"] * b["Q"])[m])))
    res[fn] = out; print(fn, json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in out.items()}))
json.dump(res, open(HERE.parent / "results/r3r4/be_Qcheck.json", "a"), indent=1)
