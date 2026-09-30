"""Sensitivity on resonance vs quasi-static (QS), with the same data used for d33.
A. AC drive series at SCM-PIT position A (R1 5 mV-2 V, R2 2 mV-2 V): two-domain P at one QS bin (30 kHz) and at the
   CR1 peak bin vs V_ac; |P| = sqrt((k V)^2 + n0^2) fit per channel -> noise floor, input-referred.
B. Survey spectra, every probe: per-bin noise (MAD about a local smooth) in the QS band and around CR1 -> the d33 that
   equals the noise in one frequency bin: QS sigma*conv, CR1 sigma*conv/E. Gain = E * sigma_QS / sigma_CR1.
C. Images at the same spot: pixel-to-pixel noise (MAD of neighbour differences / sqrt 2) in domain interiors, in pm/V
   (CR1 divided by the survey E; BE via A/M).
    python tools/xprobe_sensitivity.py <data root>
"""
import json, sys, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from scipy.ndimage import binary_erosion
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from fmmpaper import registry, domains, ebfit, beimg, forcecurve as fc, io as fio
from fmmpaper.spectra import peak
root = Path(sys.argv[1]); OUT = HERE.parent / "results/r3r4"; res = {}
pr = registry.PROBES["scmpitB"]
# ---------------- A. AC series
A = {}
for run, fold, fcf in (("R1", "DomainsB_SCMPIT/70_ac_series", "FSCMPIT0575.ibw"), ("R2", "DomainsB_SCMPIT_R2/70_ac_series", "FSCMPIT0710.ibw")):
    inv = fc.invols_fc(root / fold / fcf)["invols"]; conv = inv / 32 * 1e12
    files = sorted((root / fold).glob("ac_series_A_vac*_checkpoint.npz"), key=lambda p: float(re.search(r"vac(\d+\.\d+)mV", p.name)[1]))
    rows = []; kref = None
    for p in files:
        vac = float(re.search(r"vac(\d+\.\d+)mV", p.name)[1]) / 1e3
        d = np.load(p, allow_pickle=True); f = d["freq_Hz"]; Z = d["Z"][:, 0, :]; cond = json.loads(str(d["conditions"]))
        i1 = [i for i, c in enumerate(cond) if c["spot"] == 1 and abs(c["bias_V"]) < 1e-6][0]; i2 = [i for i, c in enumerate(cond) if c["spot"] == 2 and abs(c["bias_V"]) < 1e-6][0]
        P = (Z[i1] - Z[i2]) / 2
        if kref is None:
            dmax = np.load(files[-1], allow_pickle=True); Zm = dmax["Z"][:, 0, :]; kref = peak(f, Zm[i1], pr.bands_Hz["CR1"])["k"]
        jq = int(np.argmin(np.abs(f - 30e3))); qm = (f >= 15e3) & (f <= 45e3)
        rows.append(dict(vac=vac, P_qs_bin=abs(P[jq]), P_qs_band=abs(P[qm].mean()), P_cr=abs(P[kref]),
                         sig_qs=ebfit._sigma(P[qm]), sig_cr=ebfit._sigma(P[kref - 60:kref + 60])))
    t = pd.DataFrame(rows)
    hi = t.vac >= 0.5; E = float(np.median(t.P_cr[hi] / t.P_qs_band[hi]))
    # signal slope from V_ac >= 0.5 V; additive noise floor = per-bin noise at V_ac <= 30 mV; drive for SNR 10 per bin
    fits = {}
    for col, sig, g in (("P_qs_bin", "sig_qs", 1.0), ("P_cr", "sig_cr", E)):
        k = float(np.median(t[col][hi] / t.vac[hi])); n0 = float(np.median(t[sig][t.vac <= 0.03])); r_hi = float(np.median(t[sig][hi] / t[col][hi]))
        fits[col] = dict(k=k, n0=n0, d33=k * conv / g, noise_pmV_at_1V=n0 * conv / g, vac_snr10=10 * n0 / k, frac_noise_high_drive=r_hi)
    t["d33_qs_bin"] = t.P_qs_bin * conv / t.vac; t["d33_qs_band"] = t.P_qs_band * conv / t.vac; t["d33_cr"] = t.P_cr * conv / (E * t.vac)
    A[run] = dict(invols=inv, E=E, table=t.to_dict(orient="records"), fits=fits,
                  gain_bin=fits["P_qs_bin"]["noise_pmV_at_1V"] / fits["P_cr"]["noise_pmV_at_1V"])
    print(run, f"InvOLS {inv*1e9:.0f} nm/V, E {E:.0f}", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in fits.items()},
          f"low-drive gain (noise-equivalent d33, one bin) {A[run]['gain_bin']:.0f}x")
res["ac_series"] = A
# ---------------- B. survey noise-equivalent d33 per bin, all probes
X = pd.read_csv(OUT / "xprobe_table.csv"); rows = []
srcs = {"SCM-PIT R1 (500 nN)": ("scmpitB_r1_bias", None), "SCM-PIT R2 (500 nN)": ("scmpitB_r2_bias", None), "PPP-CONTAu (15 nN)": ("ppp_bias_15nN", None),
        "PPP-CONTAu (250 nN)": ("ppp_bias_250nN", None), "Multi75-G R3 (350 nN)": (None, root / "DomainsB_SCMPIT_R3/10_bias_survey/domains_bias_checkpoint_500nN_scmpit.npz")}
for ds, (key, path) in srcs.items():
    s = fio.load(key) if key else fio.load_series_file(path, registry.PROBES["multi75g"])
    d = domains.decompose_series(s); P = d["P"]; f = s.freq_Hz; qm = s.band("QS"); T = X[X.dataset == ds].set_index("x_um")
    iref = int(np.intersect1d(s.where(bias_V=0), s.where(spot=1))[0])
    for j, x in enumerate(s.x_um):
        pk = peak(f, s.Z[iref, j], s.probe.bands_Hz["CR1"]); k = pk["k"]; df = f[1] - f[0]; w = max(int(3 * pk["f_Hz"] / max(pk["Q"], 50) / df), 20)
        sq = ebfit._sigma(P[j, qm]); sc = ebfit._sigma(P[j, k - w:k + w])
        r = T.loc[x]; conv = r.invols_nm * 1e-9 / 32 * 1e12
        rows.append(dict(dataset=ds, x_um=x, xi=r.xi, used=r.used, E=r.E, noise_qs_pmV=sq * conv, noise_cr_pmV=sc * conv / r.E, gain=r.E * sq / sc,
                         d33=r.d33_qs_corr, snr_qs=r.d33_qs_corr / (sq * conv), snr_cr=r.d33_qs_corr / (sc * conv / r.E)))
B = pd.DataFrame(rows); B.to_csv(OUT / "xprobe_sensitivity_survey.csv", index=False)
print(B.groupby("dataset")[["noise_qs_pmV", "noise_cr_pmV", "gain", "E"]].median().round(4).to_string())
# ---------------- C. images
def pix_noise(path, scale, kind="qs", M=None, inv=None, lab=None):
    if kind == "be":
        b = beimg.load_be(path); amp = b["A"] / M * inv / 32 * 1e12; ph = b["phi"]
    else:
        q = beimg.load_qs(path); amp = q["amp_m"] * 1e12 / q["drive_V"] / scale; ph = np.radians(q["phase_deg"])
    if lab is None: lab, cen = beimg.split_domains(ph)
    out = []
    for k in (0, 1):
        m = binary_erosion(lab == k, iterations=4)
        if m.sum() < 500: continue
        dx = np.diff(amp, axis=1); mm = m[:, 1:] & m[:, :-1]
        out.append((float(np.median(amp[m])), float(1.4826 * np.median(np.abs(dx[mm] - np.median(dx[mm]))) / np.sqrt(2)), int(m.sum())))
    st = beimg.domain_stats(amp, ph, lab); contrast = st.get("contrast", np.nan)
    noise = float(np.sqrt(np.mean([o[1] ** 2 for o in out])))
    return dict(contrast=contrast, noise=noise, snr=contrast / noise, per_domain=out, lab=lab)
S2 = "DomainsB_SCMPIT_R2/50_quant_imaging"; S1 = "DomainsB_SCMPIT/50_quant_imaging"
C = [("SCM-PIT R2 A, 1 V", S2 + "/QImg0708.ibw", 1, S2 + "/QImg0711.ibw", 265.5),
     ("SCM-PIT R2 B, 1 V", S2 + "/QImg0714.ibw", 1, S2 + "/QImg0716.ibw", 37.5),
     ("SCM-PIT R1 A, 1 V", S1 + "/QImg0573.ibw", 1, S1 + "/QImg0575.ibw", 220.0),
     ("SCM-PIT R1 B, 1 V", S1 + "/QImg0578.ibw", 1, S1 + "/QImg0580.ibw", 24.2),
     ("SCM-PIT R1 A, 30 mV", "DomainsB_SCMPIT/60_lowac_30mV/LoAC0575.ibw", 1, "DomainsB_SCMPIT/60_lowac_30mV/LoAC0577.ibw", 220.0),
     ("SCM-PIT R2 A, 30 mV", "DomainsB_SCMPIT_R2/60_lowac_30mV/LoAC0710.ibw", 1, "DomainsB_SCMPIT_R2/60_lowac_30mV/LoAC0712.ibw", 265.5)]
rowsC = []
for lab, fq, _, fr, E in C:
    b = pix_noise(root / fr, E); a = pix_noise(root / fq, 1.0, lab=b["lab"])      # domain mask from the resonance frame (same raster)
    rowsC.append(dict(case=lab, qs_contrast=a["contrast"], qs_noise=a["noise"], qs_snr=a["snr"], res_kind="CR1 fixed f", res_contrast=b["contrast"], res_noise=b["noise"], res_snr=b["snr"], gain_noise=a["noise"] / b["noise"]))
r4 = json.load(open(OUT / "r4_frame_calibration.json")); M = r4["states"]["M350"]["M"]
for lab, fq, fb, inv in (("Multi75-G R4 700 nN (BE 8 ms)", "DomainsB_SCMPIT_R4/QImg0505.ibw", "DomainsB_SCMPIT_R4/52_image_be/BE_be_0V_700nN.hf5", 846.5e-9),):
    a = pix_noise(root / fq, 1.0); b = pix_noise(root / fb, 1.0, "be", M, inv)
    rowsC.append(dict(case=lab, qs_contrast=a["contrast"], qs_noise=a["noise"], qs_snr=a["snr"], res_kind="BE", res_contrast=b["contrast"], res_noise=b["noise"], res_snr=b["snr"], gain_noise=a["noise"] / b["noise"]))
r3 = json.load(open(OUT / "r3_frame_calibration.json")); M3 = r3["states"]["survey_E"]["M"]
a = pix_noise(root / "DomainsB_SCMPIT_R3/QImg0079.ibw", 1.0); b = pix_noise(root / "DomainsB_SCMPIT_R3/52_image_be/BE_A_be_Vcpd.hf5", 1.0, "be", M3, r3["invols_img"])
rowsC.append(dict(case="Multi75-G R3 350 nN (BE 8 ms)", qs_contrast=a["contrast"], qs_noise=a["noise"], qs_snr=a["snr"], res_kind="BE", res_contrast=b["contrast"], res_noise=b["noise"], res_snr=b["snr"], gain_noise=a["noise"] / b["noise"]))
Cq = pd.DataFrame(rowsC); Cq.to_csv(OUT / "xprobe_sensitivity_images.csv", index=False); pd.set_option("display.width", 250); print(Cq.round(3).to_string())
json.dump(dict(ac_series=A), open(OUT / "xprobe_sensitivity_ac.json", "w"), indent=1, default=float)
