"""Lever geometry WITHOUT a dense map: free-air walk + sparse contact capture (+ optional load ladder).

Step 1 (free air, no contact physics): the free frequency ladder f1:f2:f3 and the |mode shape| of
each free mode along the lever fix the clamp offset c, the lever length L (stage units), the tip
mass mu and rotary inertia J (and the unknown free-beam scale f0).
Step 2 (contact): the joint multi-band EB fit of the sparse two-domain survey (and any load-ladder
captures) with c, L, mu, J FROZEN at the free-air values; free: tip setback, tip height, cone
stiffness, plus per-capture k*/k, Q_c and per-band zeta, eps.
Step 3: compare with the dense-map calibration (results/lever_calibration_<probe>_<tag>.json):
geometry side by side, and E(x) on the lever interior from both calibrations fitted to the SAME
survey -- the number that says whether the dense map was necessary.

    python tools/calibrate_freeair.py --freeair R3/01_preflight/freeair_walk_checkpoint.npz \
        --survey R3/10_bias_survey/domains_bias_checkpoint_500nN_scmpit.npz \
        --preflight R3/01_preflight/preflight_result.json [--ladder a.npz b.npz] [--probe stiff]
    python tools/calibrate_freeair.py --selftest        # synthetic files from the calibrated lever, end to end

Free-air file: x_um (npos), freq_Hz (nf), Z complex (npos, nf) or amplitude (npos, nf); only |Z| is used.
Ladder files: single-condition (0 V, spot 1) checkpoints as written by run_series, one per load.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
from fmmpaper import jointeb as J, ebfit, physrec, registry, domains  # noqa: E402
from fmmpaper.io import load_series_file  # noqa: E402
import sim_calib_freeair as S  # noqa: E402  (Problem, make_je, contact_maps, free_air, fit, GEOM)

PROBE_KEY = {"stiff": "scmpitB", "soft": "pppcontau", "multi75g": "multi75g"}
FREE_MIN_HZ = 30e3


# ------------------------------------------------------------------ step 1: free air
def free_peaks(f, A, n=3, min_sep_frac=0.3):
    """Strongest n peaks of the free-end amplitude spectrum above FREE_MIN_HZ, ranked, then sorted in f."""
    m = f >= FREE_MIN_HZ; fi, ai = f[m], A[m]
    order = np.argsort(ai)[::-1]; picked = []
    for i in order:
        if 0 < i < ai.size - 1 and not (ai[i] >= ai[i - 1] and ai[i] >= ai[i + 1]): continue
        if all(abs(fi[i] - fi[j]) > min_sep_frac * max(fi[i], fi[j]) for j in picked): picked.append(i)
        if len(picked) == n: break
    out = []
    for i in picked:                               # sub-bin refinement: parabola through log|A| at the three bins around the max
        if 0 < i < ai.size - 1:
            y = np.log(ai[i - 1:i + 2]); d = (y[0] - y[2]) / (2 * (y[0] - 2 * y[1] + y[2])) if (y[0] - 2 * y[1] + y[2]) != 0 else 0.0
            out.append(fi[i] + d * (fi[i + 1] - fi[i]))
        else:
            out.append(fi[i])
    return np.sort(out)


def free_profiles(f, A, fpk, half=0.03):
    """Per mode: peak amplitude within +-half of the free-end peak at every position (the peak may shift)."""
    out = []
    for fp in fpk:
        m = (f >= (1 - half) * fp) & (f <= (1 + half) * fp)
        out.append(A[:, m].max(1))
    return np.array(out)                          # (nmode, npos)


def fit_free_air(x, fpk, prof, P_ref, f_w=5e-4, a_w=0.03):
    """c, L, mu, J, f0-scale from the ladder and the normalised shapes. Returns (P, fit_info)."""
    nm = len(fpk); prof_n = prof / np.sqrt(np.mean(prof ** 2, axis=1, keepdims=True))
    names = ["c_um", "L_um", "mu", "J", "f0s"]

    def unpack(v):
        P = dict(P_ref); P["c_um"], P["L_um"], P["mu"], P["J"], P["f0s"] = v[0], v[1], v[2], v[3], v[4]; return P, 1.0

    def resid(v):
        P, s = unpack(v); f, sh = S.free_air(P, x)
        f = f[:nm]; sh = sh[:nm]
        sh = sh / np.sqrt(np.mean(sh ** 2, axis=1, keepdims=True))
        return np.concatenate([(f - fpk) / (f_w * fpk), ((sh - prof_n) / a_w).ravel()])

    lo = [P_ref["c_um"] - 40, 0.7 * P_ref["L_um"], 0.0, 0.0, 0.7]; hi = [P_ref["c_um"] + 40, 1.3 * P_ref["L_um"], 0.2, 0.02, 1.3]
    best = None
    for c0 in (P_ref["c_um"] - 10, P_ref["c_um"], P_ref["c_um"] + 10):
        for L0 in (0.93 * P_ref["L_um"], P_ref["L_um"], 1.07 * P_ref["L_um"]):
            v0 = np.clip([c0, L0, 0.003, 1e-4, 1.0], np.array(lo) + 1e-9, np.array(hi) - 1e-9)
            r = least_squares(resid, v0, bounds=(lo, hi), x_scale="jac", max_nfev=300)
            if best is None or r.cost < best.cost: best = r
    P, s = unpack(best.x); f, sh = S.free_air(P, x)
    s = P["f0s"]
    return P, dict(f0_scale=float(s), chi2=float(2 * best.cost / resid(best.x).size), f_model_Hz=f[:nm].tolist(),
                   f_meas_Hz=fpk.tolist(), ratio_meas=(fpk / fpk[0]).tolist(), ratio_EB_uniform=[1.0, 6.267, 17.548][:nm],
                   shapes_model=(sh[:nm] / np.sqrt(np.mean(sh[:nm] ** 2, axis=1, keepdims=True))).tolist(), shapes_meas=prof_n.tolist())


# ------------------------------------------------------------------ step 2: contact
def survey_P(path, probe):
    s = load_series_file(path, probe)
    c = s.conditions
    if "spot" in c and c.spot.nunique() >= 2:
        d = domains.decompose_series(s, band=None); return s.x_um, d["freq_Hz"], d["P"], "two-domain P"
    i0 = int(np.argmin(np.abs(c.bias_V.values))) if "bias_V" in c else 0
    return s.x_um, s.freq_Hz, s.Z[i0], "single domain"


def capture_from(x, f, Z, bands):
    """Decimate the three CR bands to the model grid and estimate per-band noise."""
    fb, Zb, sig = [], [], []
    for b in ("CR1", "CR2", "CR3"):
        if b not in bands: continue
        zb, fbb = physrec.band_slice(f, Z, bands[b], n_max=600)   # fine enough for a Q~1000 CR3 line
        fb.append(fbb); Zb.append(zb); sig.append(max(ebfit._sigma(zb), 1e-3 * np.sqrt(np.mean(np.abs(zb) ** 2))))
    return dict(lf=1.0, fb=fb, Z=Zb, sigma=sig)


def fit_contact(x, caps, P_free, geom_free, starts=6, seed=0, free_data=None):
    r = np.random.default_rng(seed); best = None
    ref = P_free
    grid = [(ref["setback_um"], ref["tip_h_um"], ref["log_kcone"])] + [(sb, th, 2.9) for sb in (2.0, 4.0, 8.0, 14.0) for th in (4.0, 8.0, 15.0)]
    for st in range(max(starts, len(grid))):
        P0 = dict(P_free)
        sb, th, kc = grid[st] if st < len(grid) else (float(np.clip(r.uniform(1.5, 12), 1.1, 39)), float(r.uniform(4, 15)), r.uniform(2.0, 3.5))
        if "setback_um" in geom_free: P0["setback_um"] = sb
        if "tip_h_um" in geom_free: P0["tip_h_um"] = th
        if "log_kcone" in geom_free: P0["log_kcone"] = kc
        per0 = []
        for cap in caps:
            f1 = cap["fb"][0][np.argmax(np.abs(cap["Z"][0]).mean(0))]
            prob1 = S.Problem(x, [cap], None, [], P0)
            agrid = np.linspace(1.5, 4.8, 200)
            fg = np.array([S.eig(dict(P0, log_alpha=g), 1)[0] for g in agrid])
            la = float(agrid[np.argmin(np.abs(fg - f1))])
            per0.append((la, r.uniform(1.5, 2.3), np.array([3e-3, 1e-3, 5e-4])[:len(cap["fb"])], np.zeros(len(cap["fb"]))))
        prob = S.Problem(x, caps, free_data, geom_free, P0); prob.nb = len(caps[0]["fb"])
        try:
            Pf, per, chi2 = S.fit(prob, P0, per0, max_nfev=600)
        except Exception as e:
            print("   start", st, "failed:", e); continue
        print(f"   start {st}: chi2 {chi2:.2f}  setback {Pf['setback_um']:.2f}  tip_h {Pf['tip_h_um']:.2f}  kcone/k {10**Pf['log_kcone']:.0f}  "
              f"k*/k {10**per[0][0]:.0f}")
        if best is None or chi2 < best[2]: best = (Pf, per, chi2)
    return best


def E_interior(P, la, lq, zeta1, x):
    Pl = dict(P, log_alpha=la, log_Qc=lq); f1 = S.eig(Pl, 1)[0]
    je = J.JointEB(x, [J.Band("CR1", np.linspace(0.9 * f1, 1.1 * f1, 200), None), J.Band("QS", np.array([30e3]), None)], S.F0, n_modes=S.NM)
    (zp1, _), (zq, _) = je.band_maps(Pl, [zeta1, zeta1], x=x)
    return np.abs(zp1).max(1) / np.abs(zq)[:, 0]


# ------------------------------------------------------------------ main
def run(a):
    probe = registry.PROBES[PROBE_KEY[a.probe]]
    ref = ebfit.load_calibration(a.probe, a.cal); P_ref = dict(ref["P"])
    S.F0 = ref["f0_hz"]; J.SHARED["L_um"] = (P_ref["L_um"], 0.6 * P_ref["L_um"], 1.4 * P_ref["L_um"])
    out = dict(inputs=dict(freeair=a.freeair, survey=a.survey, ladder=a.ladder, preflight=a.preflight), reference=dict(P=P_ref, tag=a.cal))

    # ---- 1 free air
    d = np.load(a.freeair, allow_pickle=True)
    xf = np.asarray(d["x_um"], float); ff = np.asarray(d["freq_Hz"], float)
    A = np.abs(np.asarray(d["Z"])) if "Z" in d.files else np.asarray(d["amplitude"], float)
    o = np.argsort(xf); xf, A = xf[o], A[o]
    i_end = int(np.argmax(xf)); fpk = free_peaks(ff, A[i_end]); prof = free_profiles(ff, A, fpk)
    if len(fpk) < 3: raise SystemExit(f"need three free modes in the walk, found {np.round(fpk/1e3,1)} kHz -- raise the drive/DC bias or widen the tune window")
    print("free-air peaks at the free end (kHz):", np.round(fpk / 1e3, 2), " ratios", np.round(fpk / fpk[0], 3), "(uniform EB 1 : 6.267 : 17.548)")
    P_fa, info = fit_free_air(xf, fpk, prof, P_ref)
    print(f"free-air fit: c {P_fa['c_um']:.1f} um (ref {P_ref['c_um']:.1f}), L {P_fa['L_um']:.1f} um (ref {P_ref['L_um']:.1f}), "
          f"mu {P_fa['mu']:.4f} (ref {P_ref['mu']:.4f}), J {P_fa['J']:.2e} (ref {P_ref['J']:.2e}), f0 scale {info['f0_scale']:.4f}, chi2 {info['chi2']:.2f}")
    for j, (fm, fd) in enumerate(zip(info["f_model_Hz"], info["f_meas_Hz"])):
        print(f"   mode {j+1}: model {fm/1e3:9.3f} kHz  measured {fd/1e3:9.3f} kHz  ({100*(fm/fd-1):+.2f} %)")
    out["free_air"] = dict(P={k: P_fa[k] for k in ("c_um", "L_um", "mu", "J")}, **info, x_um=xf.tolist())

    # ---- 2 contact, with c/L/mu/J frozen from free air (c is allowed to float by +-3 um: laser-frame drift between walks)
    pf = json.load(open(a.preflight)) if a.preflight else None
    bands = ({f"CR{i}": (0.85 * pf[f"cr{i}_Hz"], 1.15 * pf[f"cr{i}_Hz"]) for i in (1, 2, 3) if pf and pf.get(f"cr{i}_Hz")} if pf else dict(probe.bands_Hz))
    xs, fs, Zs, src = survey_P(a.survey, probe)
    bands = {k: (v[0], min(v[1], fs.max())) for k, v in bands.items() if v[0] < 0.95 * fs.max()}   # CR3 may run into the 2 MHz window edge
    print(f"survey: {len(xs)} positions, {src}, bands " + ", ".join(f"{k} {v[0]/1e3:.0f}-{v[1]/1e3:.0f} kHz" for k, v in bands.items()))
    caps = [capture_from(xs, fs, Zs, bands)]
    for p in a.ladder or []:
        xl, fl, Zl, _ = survey_P(p, probe)
        if not np.allclose(np.sort(xl), np.sort(xs), atol=0.6): print("   ladder capture on different positions -- skipped:", p); continue
        caps.append(capture_from(xl[np.argsort(xl)], fl, Zl[np.argsort(xl)], bands))
    xs_o = np.argsort(xs); xs = xs[xs_o]; caps[0]["Z"] = [z[xs_o] for z in caps[0]["Z"]]
    # JOINT fit: the free-air residuals (ladder + shapes) enter alongside the contact spectra. A tip mass at the
    # very tip shifts every free mode by the same fraction, so free air alone cannot separate mu from the free-beam
    # scale f0s; the contact modes (tip pinned, insensitive to mu) do. Free-air positions must be the survey positions.
    if not np.allclose(np.sort(xf), np.sort(xs), atol=0.6): raise SystemExit("free-air walk and survey must use the same positions")
    prof_n = prof / np.sqrt(np.mean(prof ** 2, axis=1, keepdims=True))
    free_data = (fpk, prof_n[:, np.argsort(xf)])
    J.SHARED["c_um"] = (P_fa["c_um"], P_fa["c_um"] - 10, P_fa["c_um"] + 10)
    # mu and the free-beam scale f0s are degenerate in free air (a mass AT the tip shifts every free mode alike) and the
    # contact modes barely see mu, so mu is taken as a prior (--mu, default: the reference calibration) and f0s is then
    # set by the free f1. Freeing mu (--free_mu) opens a flat valley through setback/tip height/kcone (selftest: 5-10 % in E).
    P_fa["mu"] = a.mu if a.mu is not None else P_ref["mu"]
    geom_free = ["setback_um", "tip_h_um", "log_kcone", "c_um", "L_um", "J"] + (["mu"] if a.free_mu else []) + ([] if a.fix_f0s else ["f0s"])
    if getattr(a, "truth", None):     # selftest: chi2 floor at the truth
        prob_t = S.Problem(xs, caps, free_data, [], dict(a.truth)); prob_t.nb = len(caps[0]["fb"])
        per_t = [(a.truth["log_alpha"] + np.log10(c["lf"]), a.truth["log_Qc"], np.array([0.0028, 0.0028, 0.0028])[:prob_t.nb], np.full(prob_t.nb, -0.02)) for c in caps]
        v_t, _ = prob_t.pack(a.truth, per_t); prob_t._n = 1; rt = prob_t.residual(v_t)
        nfree = rt.size - sum(2 * z.size for z in caps[0]["Z"]) * len(caps)
        print(f"SELFTEST chi2 at the truth: total {np.mean(rt**2):.2f}, contact {np.mean(rt[:-nfree]**2):.2f}, free-air {np.mean(rt[-nfree:]**2):.2f}")
    P_start = dict(P_fa, f0s=info["f0_scale"])
    print(f"joint fit (free air + contact): {len(caps)} capture(s), free {geom_free}")
    Pn, per, chi2 = fit_contact(xs, caps, P_start, geom_free, starts=a.starts, free_data=free_data)
    out["contact_fit"] = dict(P={k: Pn[k] for k in S.GEOM + ["f0s"]}, chi2=chi2, per_capture=[dict(log_alpha=p[0], log_Qc=p[1], zeta=p[2].tolist(), eps=p[3].tolist()) for p in per])

    # ---- 3 the reference calibration fitted to the same survey (contact only), and E(x) from both
    J.SHARED["c_um"] = (P_ref["c_um"], P_ref["c_um"] - 15, P_ref["c_um"] + 15)
    Pr, per_r, chi2_r = fit_contact(xs, caps[:1], P_ref, ["c_um"], starts=1)
    x_eval = np.arange(xs.min() + 5, xs.max() - 5, 1.0)
    E_new = E_interior(Pn, per[0][0], per[0][1], per[0][2][0], x_eval); E_ref = E_interior(Pr, per_r[0][0], per_r[0][1], per_r[0][2][0], x_eval)
    dE = E_new / E_ref - 1
    if getattr(a, "truth", None):
        E_true = E_interior(a.truth, a.truth["log_alpha"], a.truth["log_Qc"], 0.0028, x_eval)
        print(f"SELFTEST: E(x) new calibration vs TRUTH: median {100*np.median(np.abs(E_new/E_true-1)):.1f} %, max {100*np.abs(E_new/E_true-1).max():.1f} %;"
              f"  dense-map reference vs TRUTH: median {100*np.median(np.abs(E_ref/E_true-1)):.1f} %, max {100*np.abs(E_ref/E_true-1).max():.1f} %")
    print("\nGEOMETRY            free-air + sparse      dense-map reference")
    for k in S.GEOM + ["f0s"]:
        print(f"  {k:12s} {Pn.get(k, 1.0):14.4g} {P_ref.get(k, 1.0):20.4g}")
    print(f"  chi2 (survey)  {chi2:14.2f} {chi2_r:20.2f}")
    print(f"E(x) interior, new vs reference calibration on the same survey: median {100*np.median(np.abs(dE)):.1f} %, max {100*np.abs(dE).max():.1f} %")
    out["compare"] = dict(x_eval=x_eval.tolist(), E_new=E_new.tolist(), E_ref=E_ref.tolist(), dE_median_pct=float(100 * np.median(np.abs(dE))),
                          dE_max_pct=float(100 * np.abs(dE).max()), chi2_new=chi2, chi2_ref=chi2_r, P_ref_refit={k: Pr[k] for k in S.GEOM})
    stem = Path(a.survey).stem
    outp = HERE.parent / "results" / f"calibrate_freeair_{stem}.json"; json.dump(out, open(outp, "w"), indent=1, default=float)
    print("wrote", outp)
    try:
        plot(out, HERE.parent / "figures" / f"calibrate_freeair_{stem}.png")
    except Exception as e:
        print("plot skipped:", e)
    return out


def plot(out, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fa = out["free_air"]; cmp_ = out["compare"]; x = np.array(fa["x_um"])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for j, (m, d) in enumerate(zip(fa["shapes_model"], fa["shapes_meas"])):
        ax[0].plot(x, d, "o", color=f"C{j}", label=f"mode {j+1} data"); ax[0].plot(x, m, "-", color=f"C{j}")
    ax[0].set(xlabel="stage x (um)", ylabel="|free mode shape| (norm.)", title="free-air walk"); ax[0].legend(fontsize=7)
    ax[1].bar(np.arange(3) - 0.2, fa["ratio_meas"][:3], 0.4, label="measured f_n/f_1"); ax[1].bar(np.arange(3) + 0.2, fa["ratio_EB_uniform"][:3], 0.4, label="uniform EB")
    ax[1].set(xticks=range(3), xticklabels=["1", "2", "3"], yscale="log", title="free ladder"); ax[1].legend(fontsize=7)
    xe = np.array(cmp_["x_eval"]); ax[2].plot(xe, cmp_["E_ref"], label="dense-map calibration"); ax[2].plot(xe, cmp_["E_new"], "--", label="free-air + sparse")
    ax[2].set(xlabel="stage x (um)", ylabel="E = |Z_CR1|/|Z_QS|", title=f"E(x): median diff {cmp_['dE_median_pct']:.1f} %, max {cmp_['dE_max_pct']:.1f} %"); ax[2].legend(fontsize=7)
    fig.tight_layout(); path.parent.mkdir(exist_ok=True); fig.savefig(path, dpi=150); print("wrote", path)


def selftest(a):
    """Synthesise free-air + survey (+ ladder) files from the calibrated lever and run the pipeline."""
    import tempfile
    ref = ebfit.load_calibration(a.probe, a.cal); P = dict(ref["P"]); P["J"] = 2e-4; P["mu"] = 0.004
    S.F0 = ref["f0_hz"]; S.MISMATCH = a.mismatch
    x = np.round(np.linspace(78, 232, 8), 1); tmp = Path(tempfile.mkdtemp())
    # free air: synthetic spectra = sum of Lorentzians at the free modes with |shape| amplitudes
    f_free, sh = S.free_air(P, x); ff = np.linspace(20e3, 2e6, 4000); A = np.zeros((x.size, ff.size))
    r = np.random.default_rng(3)
    for j in range(3):
        w = f_free[j] / 300.0; fj = f_free[j] * (1 + 5e-4 * r.standard_normal())
        amp = sh[j] * r.uniform(0.7, 1.3) * (1 + 0.03 * r.standard_normal(x.size))      # 3 % amplitude noise per position
        A += amp[:, None] / np.sqrt(1 + ((ff - fj) / w) ** 2)
    A += 1e-3 * r.standard_normal(A.shape) ** 2
    np.savez(tmp / "freeair.npz", x_um=x, freq_Hz=ff, amplitude=A)
    # survey and ladder: full spectra so the loader/band cutting is exercised
    def full_capture(lf, seed):
        fgrid = np.linspace(20e3, 2e6, 6000); Z = np.zeros((x.size, fgrid.size), complex)
        Pl = dict(P, log_alpha=P["log_alpha"] + np.log10(lf)); je = S.make_je(x, [fgrid]); je.bands = [J.Band("all", fgrid, None)]
        (zp, ze), = je.band_maps(Pl, [0.0028], x=x, dlf=None); Z = zp + EPS * ze
        s = 0.03 * np.sqrt(np.mean(np.abs(Z) ** 2)); Z += s * (r.standard_normal(Z.shape) + 1j * r.standard_normal(Z.shape)) / np.sqrt(2)
        return fgrid, Z
    EPS = -0.02
    fgrid, Z1 = full_capture(1.0, 1)
    cond = [dict(bias_V=b, load_nN=500.0, spot=sp) for sp in (1, 2) for b in (-3, 0, 3)]
    Zc = np.array([(Z1 if c["spot"] == 1 else -Z1) * (1 + 0.05 * c["bias_V"]) for c in cond])
    np.savez(tmp / "survey.npz", x_um=x, freq_Hz=fgrid, Z=Zc, conditions=json.dumps(cond), phase_convention="model")
    ladders = []
    for k, lf in enumerate((0.5, 2.0)):
        fg, Zl = full_capture(lf, 10 + k); p = tmp / f"ladder_{lf}.npz"
        np.savez(p, x_um=x, freq_Hz=fg, Z=Zl[None], conditions=json.dumps([dict(bias_V=0.0, load_nN=500 * lf, spot=1)]), phase_convention="model"); ladders.append(str(p))
    f_c = S.eig(P, 3); pf = tmp / "preflight.json"; json.dump({f"cr{i+1}_Hz": float(f_c[i]) for i in range(3)}, open(pf, "w"))
    print("SELFTEST truth:", {k: round(P[k], 4) for k in S.GEOM}, "(dense-map reference differs: J, mu)")
    a.truth = P
    a.freeair, a.survey, a.preflight, a.ladder = str(tmp / "freeair.npz"), str(tmp / "survey.npz"), str(pf), (ladders if a.use_ladder else None)
    run(a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeair"); ap.add_argument("--survey"); ap.add_argument("--preflight"); ap.add_argument("--ladder", nargs="*")
    ap.add_argument("--probe", default="stiff", choices=PROBE_KEY); ap.add_argument("--cal", default="mass"); ap.add_argument("--starts", type=int, default=6)
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--use_ladder", action="store_true"); ap.add_argument("--mismatch", type=float, default=0.0); ap.add_argument("--fix_f0s", action="store_true")
    ap.add_argument("--mu", type=float, default=None, help="tip-mass prior m_tip/m_beam (default: reference calibration)"); ap.add_argument("--free_mu", action="store_true")
    a = ap.parse_args()
    (selftest if a.selftest else run)(a)
