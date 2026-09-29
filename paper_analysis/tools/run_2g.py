"""Phase 2g driver: joint multi-mode EB fit with one shared geometry (PPP-CONTAu 1 um map).

    python tools/run_2g.py            # everything, resumes from results/_cache_05b.json
    python tools/run_2g.py ladder     # one part: stage1 | ladder | ind | lomo | smallN

Parts
-----
stage1  eigenfrequencies + mode-shape nodes of the shared-geometry model vs the dense map
        (frame free, and frame fixed at the 2a values c = 0, L = 445 um)
ladder  complex joint fits at N = 30 equispaced: M1f (shared geometry + free frame + per-mode
        zeta, eps, frequency correction), M0f (2a frame), M1f-1d (one zeta/eps), M1 (no
        frequency correction)
ind     single-band package fits (physrec) in the M1f frame: separates "frame" from "sharing"
lomo    leave-one-mode-out: geometry from 4 modes, held-out mode gets only zeta, eps, dlf, gain
smallN  N = 4..30 with starts built only from the design positions (their resonance
        frequencies and the InvOLS ruler at those positions), + a residual GP (joint EB+GP)
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fmmpaper as F  # noqa: E402
from fmmpaper import jointeb as J, physrec, recon, spectra  # noqa: E402

CACHE = F.config.RESULTS_DIR / "_cache_05b.json"
BANDS = ["CR1", "CR2", "CR3", "CR4", "CR5"]
F0_HZ = 13.649e3
EDGE_UM = 5.0
GEO = ("log_alpha", "log_kcone", "log_Qc", "setback_um", "tip_h_um")
FRAME = ("c_um", "L_um")


# ----------------------------------------------------------------------------- data
def setup():
    s = F.load("ppp_dense_1um")
    x = s.x_um
    bands = []
    for b in BANDS:
        z, f = physrec.band_slice(s.freq_Hz, s.Z[0], s.probe.bands_Hz[b], n_max=250)
        bands.append(J.Band(b, f, z))
    je = J.JointEB(x, bands, F0_HZ, n_modes=14)
    truth = {}
    for b in bands:
        pk = spectra.peaks_along_x(b.f_Hz, b.Z, (b.f_Hz[0], b.f_Hz[-1]))
        nd = spectra.nodes_from_profile(x, pk["amp"])
        truth[b.name] = dict(f_Hz=float(np.median(pk["f_Hz"])), Q=float(np.nanmedian(pk["Q"])),
                             nodes=[n for n in nd if x.min() + EDGE_UM <= n <= x.max() - EDGE_UM])
    return s, x, je, truth


def invols_log(ds_key="ppp_dense_1um"):
    """Per-position InvOLS from the acquisition log ('cal trigger ... InvOLS x m/V')."""
    from fmmpaper import io as _io
    p = _io.path_of(ds_key)
    log = p.with_name(p.name.replace("_checkpoint.npz", "_log.txt"))
    t = log.read_text(errors="replace")
    out = {}
    for blk in re.split(r"=== position \d+: x = ", t)[1:]:
        xv = float(blk.split()[0]); m = re.search(r"InvOLS ([\d.e+-]+) m/V", blk)
        if m:
            out[round(xv, 1)] = float(m.group(1))
    return out


def ruler(xs, inv):
    """Static-shape InvOLS ruler: 1/InvOLS ~ u^2 (3A - u), u = x - c (tip-loaded, displacement
    detection), linear beyond the contact. Returns (c, A, rms)."""
    xs = np.asarray(xs, float)
    y = np.array([1 / inv[round(v, 1)] for v in xs])

    def mdl(p, xx):
        lg, c, A = p
        u = np.clip(xx - c, 1e-3, None)
        return np.exp(lg) * np.where(u <= A, u ** 2 * (3 * A - u), 2 * A ** 3 + (u - A) * 3 * A ** 2)
    best = None
    for c0 in (-30.0, -10.0, 10.0):
        for A0 in (420.0, 470.0):
            r = least_squares(lambda p: np.log(y / mdl(p, xs)),
                              [np.log(y.mean() / (3 * A0 * 200 ** 2)), c0, A0],
                              bounds=([-np.inf, -60, 300], [np.inf, 95, 700]))
            if best is None or r.cost < best.cost:
                best = r
    return float(best.x[1]), float(best.x[2]), float(np.sqrt(2 * best.cost / len(xs)))


# ----------------------------------------------------------------------------- scoring
def score(je, x, fit, sel, band_idx=None):
    band_idx = fit["band_idx"] if band_idx is None else band_idx
    ho = recon.held_out(len(x), sel)
    saved = je.bands; je.bands = [saved[i] for i in band_idx]
    try:
        maps, _ = je.predict(fit["P"], fit["zeta"], fit["eps"], sel, x=x, dlf=fit.get("dlf"))
    finally:
        je.bands = saved
    rows = []
    for i, M in zip(band_idx, maps):
        rows.append(score_map(je.bands[i], x, M, ho))
    return rows, maps


def score_map(b, x, M, ho):
    D = b.Z
    e = float(np.sqrt(np.mean(np.abs(M[ho] - D[ho]) ** 2)) / np.sqrt(np.mean(np.abs(D[ho]) ** 2)))
    pm = spectra.peaks_along_x(b.f_Hz, M, (b.f_Hz[0], b.f_Hz[-1]))
    pd_ = spectra.peaks_along_x(b.f_Hz, D, (b.f_Hz[0], b.f_Hz[-1]))
    nm = spectra.nodes_from_profile(x, pm["amp"])
    nd = [n for n in spectra.nodes_from_profile(x, pd_["amp"]) if x.min() + EDGE_UM <= n <= x.max() - EDGE_UM]
    ne = float(np.mean([min(abs(t - q) for q in nm) for t in nd])) if (nm and nd) else (float("inf") if nd else float("nan"))
    df = float(100 * (np.median(pm["f_Hz"]) / np.median(pd_["f_Hz"]) - 1))
    return dict(band=b.name, nrmse=e, node_err_um=ne, df_pct=df)


def fit_record(fit, rows, **extra):
    return dict(P={k: float(v) for k, v in fit["P"].items()}, zeta=[float(v) for v in fit["zeta"]],
                eps=[float(v) for v in fit["eps"]], dlf=[float(v) for v in fit["dlf"]],
                cost=fit["cost"], band_idx=list(fit["band_idx"]), score=rows, **extra)


def as_fit(rec):
    return dict(P=rec["P"], zeta=np.array(rec["zeta"]), eps=np.array(rec["eps"]),
                dlf=np.array(rec["dlf"]), band_idx=rec["band_idx"])


# ----------------------------------------------------------------------------- stage 1
def stage1(je, truth, use=None):
    """use: band indices whose frequencies/nodes enter (default all). Model mode k <-> band k."""
    use = list(range(5)) if use is None else list(use)
    fm_all = np.array([truth[b]["f_Hz"] for b in BANDS])
    nodes_all = [truth[b]["nodes"] for b in BANDS]
    fm = fm_all[use]
    nodes = [nodes_all[k] for k in use]
    xf = np.linspace(95, 450, 1421)
    free = ["log_alpha", "log_kcone", "setback_um", "tip_h_um", "c_um", "L_um"]
    nres = len(use) + sum(len(n) for n in nodes)

    def eig(P):
        K, M, *_ = je._matrices(P)
        w2, V = np.linalg.eig(np.linalg.solve(M, K))
        o = np.argsort(w2.real)
        return np.sqrt(np.clip(w2.real[o], 0, None)) * je.fs, V[:, o].real

    def nodes_of(P, V, k):
        xi = np.clip((xf - P["c_um"]) / P["L_um"], 0, 1)
        _, ph, _ = J.beam_modes(xi, je.n)
        sh = V[:, k] @ ph
        i = np.where(np.sign(sh[:-1]) != np.sign(sh[1:]))[0]
        return xf[i] - sh[i] * (xf[i + 1] - xf[i]) / (sh[i + 1] - sh[i])

    def res(v, frame_fixed=None):
        P = {k: J.SHARED[k][0] for k in J.SHARED}
        P.update(dict(zip(free, v)))
        if frame_fixed:
            P.update(frame_fixed)
        if P["L_um"] < je.x.max() - P["c_um"] - 0.01:
            return np.full(nres, 5.0)
        f, V = eig(P)
        r = list(3 * np.log(f[use] / fm))
        for k, nd in zip(use, nodes):
            mod = nodes_of(P, V, k)
            r += [min([abs(t - q) for q in mod], default=100) / 50 for t in nd]
        return np.array(r)

    def summary(v, frame_fixed=None):
        P = {k: J.SHARED[k][0] for k in J.SHARED}; P.update(dict(zip(free, v)))
        if frame_fixed:
            P.update(frame_fixed)
        f, V = eig(P)
        ne = [min(abs(t - q) for q in nodes_of(P, V, k)) if len(nodes_of(P, V, k)) else float("inf")
              for k, nd in enumerate(nodes_all) for t in nd]
        return dict(P={k: float(P[k]) for k in free}, freq_err_pct=[float(v) for v in 100 * (f[:5] / fm_all - 1)],
                    node_err_um=[float(v) for v in ne],
                    model_nodes={BANDS[k]: [float(q) for q in nodes_of(P, V, k)] for k in range(5)})

    lo = [J.SHARED[k][1] for k in free]; hi = [J.SHARED[k][2] for k in free]
    runs = []
    for la in (3.0, 3.5):
        for lk in (1.0, 2.0):
            for sb in (8, 15):
                for c, L in ((0, 450), (-10, 465), (-30, 480), (10, 440)):
                    r = least_squares(res, [la, lk, sb, 15, c, L], bounds=(lo, hi), max_nfev=400,
                                      x_scale=[0.3, 0.3, 2, 2, 5, 5])
                    runs.append((float(r.cost), [float(q) for q in r.x]))
    runs.sort(key=lambda t: t[0])
    good = [v for c, v in runs if c < 2 * runs[0][0]]
    fixed = {"c_um": 0.0, "L_um": 445.0}
    bestf = None
    for la in (2.5, 3.0, 3.5, 4.0):
        for lk in (0.5, 1.5, 2.5):
            for sb in (5, 12, 20):
                for th in (8, 15, 22):
                    r = least_squares(lambda v: res(list(v) + [0.0, 445.0], fixed), [la, lk, sb, th],
                                      bounds=(lo[:4], hi[:4]), max_nfev=200)
                    if bestf is None or r.cost < bestf.cost:
                        bestf = r
    return dict(free_frame=dict(cost=runs[0][0], **summary(runs[0][1]), top=runs[:10],
                                spread={k: [min(g[i] for g in good), max(g[i] for g in good)]
                                        for i, k in enumerate(free)}),
                fixed_frame=dict(cost=float(bestf.cost), **summary(list(bestf.x) + [0.0, 445.0], fixed)))


def starts_from_stage1(je, st1, truth, k=3):
    fm = np.array([truth[b]["f_Hz"] for b in BANDS])
    z0 = list(np.log10(1 / (2 * np.array([truth[b]["Q"] for b in BANDS]))))
    free = ["log_alpha", "log_kcone", "setback_um", "tip_h_um", "c_um", "L_um"]
    out = []
    for c, v in st1["free_frame"]["top"][:k]:
        d = dict(zip(free, v))
        P = {kk: J.SHARED[kk][0] for kk in J.SHARED}; P.update(d)
        dl = list(np.clip(je.eig_freqs_Hz(P)[:5] / fm - 1, -0.039, 0.039))
        for qc in (2.0, 3.0):
            for e0 in (0.0, 1.0):
                out.append({**d, "log_Qc": qc, "log_zeta": z0, "eps": [e0] * 5, "dlf": dl})
    return out, z0, fm


# ----------------------------------------------------------------------------- parts
def part_ladder(je, x, truth, cache, n=30):
    sel = recon.select_equispaced(x, n)
    st, z0, fm = starts_from_stage1(je, cache["stage1"], truth)
    fx = cache["stage1"]["fixed_frame"]["P"]
    P = {k: J.SHARED[k][0] for k in J.SHARED}; P.update(fx)
    dl0 = list(np.clip(je.eig_freqs_Hz(P)[:5] / fm - 1, -0.039, 0.039))
    st0 = [{**{k: fx[k] for k in ("log_alpha", "log_kcone", "setback_um", "tip_h_um")},
            "log_Qc": qc, "log_zeta": z0, "eps": [e0] * 5, "dlf": dl0} for qc in (2.0, 3.0) for e0 in (0.0, 1.0)]
    specs = {
        "M1f": (J.Spec(free=GEO + FRAME, band_freq=True, name="shared geometry + free frame + per-mode zeta, eps, df"), st),
        "M0f": (J.Spec(free=GEO, fixed={"c_um": 0.0, "L_um": 445.0}, band_freq=True, name="shared geometry, 2a frame (c=0, L=445)"), st0),
        "M1f-1d": (J.Spec(free=GEO + FRAME, band_freq=True, shared_damping=True, name="as M1f, one zeta and one eps for all modes"), st),
        "M1": (J.Spec(free=GEO + FRAME, band_freq=False, name="as M1f, no per-mode frequency correction"), st[:4]),
    }
    out = cache.setdefault("ladder", {})
    for key, (spec, starts) in specs.items():
        if key in out:
            continue
        t = time.time()
        fit = je.fit(spec, sel, starts=starts, max_nfev=400)
        rows, _ = score(je, x, fit, sel)
        out[key] = fit_record(fit, rows, name=spec.name, N=n, sec=time.time() - t)
        save(cache); print(f"ladder {key}: {time.time() - t:.0f}s", [round(r["nrmse"], 3) for r in rows], flush=True)


def part_ind(je, x, truth, cache, n=30):
    sel = recon.select_equispaced(x, n); ho = recon.held_out(len(x), sel)
    P = cache["ladder"]["M1f"]["P"]
    geom = dict(physrec.GEOM["pppcontau"]); geom["L_um"] = P["L_um"]
    out = cache.setdefault("ind", {})
    for b in je.bands:
        if b.name in out:
            continue
        t = time.time()
        o = physrec.rec_eb(x - P["c_um"], sel, b.Z[sel], b.f_Hz, geom)
        out[b.name] = dict(**score_map(b, x, o["Zeb"], ho),
                           theta={k: float(v) for k, v in o["theta"].items()}, sec=time.time() - t)
        save(cache); print(f"ind {b.name}: {time.time() - t:.0f}s nrmse {out[b.name]['nrmse']:.3f}", flush=True)


def part_lomo(je, x, truth, cache, n=30):
    """Leave one mode out. The held-out mode enters nowhere: stage-1 starts use only the four
    training modes' frequencies and nodes, and the complex fit sees only their spectra."""
    sel = recon.select_equispaced(x, n)
    z0 = list(np.log10(1 / (2 * np.array([truth[b]["Q"] for b in BANDS]))))
    fm = np.array([truth[b]["f_Hz"] for b in BANDS])
    spec = J.Spec(free=GEO + FRAME, band_freq=True)
    free1 = ["log_alpha", "log_kcone", "setback_um", "tip_h_um", "c_um", "L_um"]
    out = cache.setdefault("lomo", {})
    for hold in range(5):
        name = BANDS[hold]
        if name in out:
            continue
        t = time.time(); train = [i for i in range(5) if i != hold]
        s1 = stage1(je, truth, use=train)
        st = []
        for c_, v in s1["free_frame"]["top"][:3]:
            d = dict(zip(free1, v)); P = {k: J.SHARED[k][0] for k in J.SHARED}; P.update(d)
            dl = list(np.clip(je.eig_freqs_Hz(P)[:5] / fm - 1, -0.039, 0.039))
            for qc in (2.0, 3.0):
                st.append({**d, "log_Qc": qc, "log_zeta": [z0[i] for i in train], "eps": [0.0] * 4,
                           "dlf": [dl[i] for i in train]})
        fit = je.fit(spec, sel, band_idx=train, starts=st, max_nfev=400)
        fixed = {k: fit["P"][k] for k in J.SHARED}
        P0 = dict(fixed)
        d0 = float(np.clip(je.eig_freqs_Hz(P0)[hold] / fm[hold] - 1, -0.039, 0.039))
        sth = [{"log_zeta": [z0[hold]], "eps": [e], "dlf": [d0 + dd]} for e in (0.0, 1.0) for dd in (0.0, -0.005, 0.005)]
        fh = je.fit(J.Spec(free=(), fixed=fixed, band_freq=True), sel, band_idx=[hold], starts=sth, max_nfev=200)
        rows, _ = score(je, x, fh, sel)
        out[name] = dict(score=rows[0], train_P={k: float(fit["P"][k]) for k in GEO + FRAME},
                         stage1_train=dict(P=s1["free_frame"]["P"], heldout_node_err_um=None),
                         eig_freq_err_pct=float(100 * (je.eig_freqs_Hz(P0)[hold] / fm[hold] - 1)),
                         dlf_heldout=float(fh["dlf"][0]), sec=time.time() - t)
        save(cache); print(f"lomo {name}: {time.time() - t:.0f}s", rows[0], out[name]["train_P"], flush=True)


def part_smallN(je, x, truth, cache, ns=(4, 5, 6, 8, 14, 30)):
    inv = invols_log()
    z0 = list(np.log10(1 / (2 * np.array([truth[b]["Q"] for b in BANDS]))))
    FS = ["log_alpha", "log_kcone", "setback_um", "tip_h_um"]
    spec = J.Spec(free=GEO + FRAME, band_freq=True)
    out = cache.setdefault("smallN", {})
    for n in ns:
        if str(n) in out:
            continue
        t = time.time()
        sel = recon.select_equispaced(x, n); ho = recon.held_out(len(x), sel)
        fm = np.array([np.median(spectra.peaks_along_x(b.f_Hz, b.Z[sel], (b.f_Hz[0], b.f_Hz[-1]))["f_Hz"]) for b in je.bands])
        c0, A0, _ = ruler(x[sel], inv)
        starts = []
        for L0 in (450.0, 462.0, 475.0):
            L0 = max(L0, x.max() - c0 + 0.1)

            def r_(v):
                P = {k: J.SHARED[k][0] for k in J.SHARED}; P.update(dict(zip(FS, v))); P["c_um"] = c0; P["L_um"] = L0
                return np.log(je.eig_freqs_Hz(P)[:5] / fm)
            best = None
            for la in (2.5, 3.0, 3.5, 4.0):
                for sb in (6, 12, 20):
                    rr = least_squares(r_, [la, 1.5, sb, 15], bounds=([J.SHARED[k][1] for k in FS], [J.SHARED[k][2] for k in FS]))
                    if best is None or rr.cost < best.cost:
                        best = rr
            P = {k: J.SHARED[k][0] for k in J.SHARED}; P.update(dict(zip(FS, best.x))); P["c_um"] = c0; P["L_um"] = L0
            dl = list(np.clip(je.eig_freqs_Hz(P)[:5] / fm - 1, -0.039, 0.039))
            starts += [{**{k: P[k] for k in FS}, "c_um": c0, "L_um": L0, "log_Qc": qc, "log_zeta": z0,
                        "eps": [0.0] * 5, "dlf": dl} for qc in (2.0, 3.0)]
        fit = je.fit(spec, sel, starts=starts, max_nfev=400)
        rows, maps = score(je, x, fit, sel)
        for r, M, b in zip(rows, maps, je.bands):     # joint EB + residual GP
            Mg = M + recon.rec_gp(x, sel, b.Z[sel] - M[sel])["Zrec"]
            g = score_map(b, x, Mg, ho)
            r["nrmse_gp"], r["node_err_gp"] = g["nrmse"], g["node_err_um"]
        out[str(n)] = fit_record(fit, rows, N=n, c_ruler=c0, sec=time.time() - t)
        save(cache); print(f"smallN {n}: {time.time() - t:.0f}s", [round(r["nrmse"], 3) for r in rows], flush=True)


# ----------------------------------------------------------------------------- driver
def save(cache):
    CACHE.write_text(json.dumps(cache, default=float))


def main(parts):
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    s, x, je, truth = setup()
    cache["truth"] = truth
    if "ruler_all" not in cache:
        inv = invols_log()
        c, A, rms = ruler(x, inv)
        cache["ruler_all"] = dict(c_um=c, A_um=A, rms=rms)
    if "stage1" in parts and "stage1" not in cache:
        t = time.time(); cache["stage1"] = stage1(je, truth); save(cache); print(f"stage1 {time.time() - t:.0f}s", flush=True)
    if "ladder" in parts:
        part_ladder(je, x, truth, cache)
    if "ind" in parts:
        part_ind(je, x, truth, cache)
    if "lomo" in parts:
        part_lomo(je, x, truth, cache)
    if "smallN" in parts:
        part_smallN(je, x, truth, cache)
    save(cache)
    return cache


if __name__ == "__main__":
    ALL = ["stage1", "ladder", "ind", "lomo", "smallN"]
    main([a for a in sys.argv[1:] if a in ALL] or ALL)
