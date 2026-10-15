"""Live view of a two-domain capture: the map filling in, E(x) and its convergence.

`LiveCapture` is the `on_position` callback for `run_series`. After every laser
position it re-runs the (fast) map-route analysis on what has been measured so far
and redraws four panels:

    a  |Z|(x, f) of the reference condition, reconstructed on the lever grid
       (sampled positions ticked at the top)
    b  quasi-static d_eff per position, raw and domain-potential corrected
       (flat after correction = the survey is behaving)
    c  E_map(x): reconstruction with its band, measured points, leave-one-out
       predictions, and the recommended imaging stop
    d  convergence: how much E(x) changed when the last position was added

It also implements `should_stop()` for an early finish when E(x) has stopped
changing (`capture.stop_when_stable`), and a manual stop through
`request_stop()` or a file called STOP in the campaign folder.
"""
from __future__ import annotations

import os
import time

import numpy as np

from ..lowrank import reconstruct_map, band_mask
from .analysis import analyse_survey

# fixed roles -> fixed colours (never cycled)
C_MEAS = "#1f4e79"      # measured points
C_REC = "#2a9d8f"       # reconstruction / corrected
C_RAW = "#9aa5b1"       # raw / uncorrected
C_LOO = "#e76f51"       # leave-one-out predictions
C_PICK = "#b5179e"      # recommended stop


class LiveCapture:
    def __init__(self, cfg, f_cr1_Hz=None, invols_getter=None, x_limits=None, stop_file=None,
                 figsize=(13, 8.5), show=True):
        self.cfg, self.f_cr1, self.get_invols = cfg, f_cr1_Hz, invols_getter
        self.x_limits = x_limits
        self.x_grid = (np.arange(x_limits[0], x_limits[1] + 0.5 * cfg.analysis.grid_step_um, cfg.analysis.grid_step_um)
                       if x_limits else None)
        self.stop_file = stop_file or cfg.path("STOP")
        self.history = []            # list of (n_positions, E_grid)
        self.rel_change = []         # median |dE/E| after each added position
        self.stable_count = 0
        self._stop = False
        self.last = None
        self.t0 = time.time()
        self.show = show
        self.fig = None
        self.figsize = figsize
        self._status_lines = []

    # ---- stop logic --------------------------------------------------------------------
    def request_stop(self):
        self._stop = True

    def should_stop(self):
        if self._stop or os.path.exists(self.stop_file):
            return True
        c = self.cfg.capture
        n = self.history[-1][0] if self.history else 0
        return bool(c.stop_when_stable and n >= c.min_positions and self.stable_count >= c.stable_over)

    # ---- the callback ------------------------------------------------------------------
    def __call__(self, series):
        return self.update(series)

    def update(self, series):
        n = int(np.asarray(series["x_um"]).size)
        res = None
        if n >= 3:
            inv = self.get_invols() if self.get_invols else None
            try:
                res = analyse_survey(series, self.cfg, f_cr1_Hz=self.f_cr1, invols_by_x=inv,
                                     x_limits=self.x_limits, x_grid=self.x_grid, verbose=False)
            except Exception as e:          # analysis may be ill-posed early on; keep acquiring
                self._status_lines = [f"analysis not yet possible with {n} positions ({type(e).__name__}: {e})"]
        self.last = res
        if res is not None:
            E = res["emap"]["E"]
            if self.history and self.history[-1][1].shape == E.shape:
                prev = self.history[-1][1]
                xg, xm = res["emap"]["x_grid"], res["emap"]["x_meas"]
                inside = (xg >= xm.min()) & (xg <= xm.max())       # judge only where positions exist
                with np.errstate(divide="ignore", invalid="ignore"):
                    rc = float(np.nanmedian(np.abs(E - prev)[inside] / np.abs(E)[inside]))
                self.rel_change.append(rc)
                self.stable_count = self.stable_count + 1 if rc <= self.cfg.capture.stable_rel else 0
            self.history.append((n, E))
        if self.show:
            self.draw(series, res)
        return res

    # ---- drawing -----------------------------------------------------------------------
    def draw(self, series, res):
        import matplotlib
        import matplotlib.pyplot as plt
        if self.fig is None:
            self.fig, axs = plt.subplots(2, 2, figsize=self.figsize)
            self.axs = axs.ravel()
        for ax in self.axs:
            ax.cla()
        a, b, c, d = self.axs
        cfg = self.cfg
        xs = np.asarray(series["x_um"], float); F = series["freq_Hz"]; Z = series["Z"]
        n = xs.size
        elapsed = (time.time() - self.t0) / 60

        # (a) reference-condition map
        ref = series.get("ref_index", 0)
        f_cr1 = self.f_cr1 or (res["f_cr1_Hz"] if res else float(np.mean(cfg.probe.cr_bands_Hz["cr1"])))
        band = (cfg.probe.qs_band_Hz[0], 1.35 * f_cr1)
        m = band_mask(F, band)
        idx = np.flatnonzero(m)
        if idx.size > 1500:
            idx = idx[:: int(np.ceil(idx.size / 1500))]
        Zr = Z[ref][:, idx]
        okp = np.isfinite(Zr).all(1)
        if okp.sum() >= 2:
            xg = np.arange(xs[okp].min(), xs[okp].max() + 0.5, 1.0)
            sel = [int(np.argmin(np.abs(xg - x))) for x in xs[okp]]
            rec = reconstruct_map(xg, sel, Zr[okp], min(cfg.analysis.rank, okp.sum()))
            A = np.abs(rec["Zrec"]); dB = 20 * np.log10(np.maximum(A, 1e-30) / A.max())
            a.imshow(dB.T, origin="lower", aspect="auto", cmap="viridis", vmin=-50, vmax=0,
                     extent=[xg.min(), xg.max(), F[idx].min() / 1e3, F[idx].max() / 1e3])
            a.plot(xs[okp], np.full(okp.sum(), F[idx].max() / 1e3 * 0.985), "v", color="white", ms=7)
        a.set_xlabel("laser position, stage x (um)"); a.set_ylabel("frequency (kHz)")
        a.set_title(f"a  |Z|(x, f) reconstructed from {n} position{'s' if n != 1 else ''} (dB)", loc="left", fontsize=10)

        if res is None:
            for ax, msg in ((b, "QS d_eff — needs 3 positions"), (c, "E_map(x) — needs 3 positions"),
                            (d, "convergence")):
                ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, color="0.4")
            self.fig.suptitle(f"two-domain capture: {n} positions, {elapsed:.1f} min   "
                              + " ".join(self._status_lines), fontsize=11)
            self._flush()
            return

        t, corr, em, rc = res["table"], res["corr"], res["emap"], res["recommend"]
        okc = t["complete"]
        inv = t["invols"]
        conv = np.where(np.isfinite(inv), inv, np.nan) / cfg.probe.amp_divisor / cfg.capture.drive_V * 1e12
        # (b) QS d_eff raw vs corrected
        b.plot(t["x_um"][okc], (np.abs(t["P_qs"]) * conv)[okc], "o", color=C_RAW, ms=7, label="raw")
        b.plot(t["x_um"][okc], res["deff_qs_pm_per_V"][okc], "o", color=C_REC, ms=7, label="domain-potential corrected")
        med = np.nanmedian(res["deff_qs_pm_per_V"])
        if np.isfinite(med):
            b.axhline(med, color=C_REC, lw=1, ls="--")
            b.text(0.02, 0.95, f"median {med:.2f} pm/V, sd {100 * np.nanstd(res['deff_qs_pm_per_V']) / med:.1f} %\n"
                               f"delta = {res['delta']['delta_V']:+.2f} V, V_cpd = {res['v_cpd_V']:+.2f} V",
                   transform=b.transAxes, va="top", fontsize=9)
        es = res.get("esbs")
        if es:
            b.axvline(es["x_esbs_um"], color=C_PICK, lw=1.2, ls=":")
            b.text(es["x_esbs_um"], 0.60, f"QS stop (blind spot) \nuncorrected error ~{es['raw_error_pct']:.1f} % ", color=C_PICK,
                   fontsize=8, va="top", ha="right" if es["x_esbs_um"] > 0.5 * (t["x_um"].min() + t["x_um"].max()) else "left",
                   transform=b.get_xaxis_transform())
        if not res["invols_known"]:
            b.text(0.98, 0.5, "InvOLS not known at every stop", transform=b.transAxes, ha="right", fontsize=8, color="0.4")
        b.set_xlabel("stage x (um)"); b.set_ylabel("quasi-static d_eff (pm/V)")
        b.set_title("b  quasi-static response along the lever (should be flat)", loc="left", fontsize=10)
        b.legend(loc="lower left", fontsize=8, frameon=False); b.set_ylim(bottom=0)

        # (c) E_map(x)
        c.fill_between(em["x_grid"], em["E_lo"], em["E_hi"], color=C_REC, alpha=0.15, lw=0)
        c.plot(em["x_grid"], em["E"], "-", color=C_REC, lw=2, label=f"reconstruction (rank {em['rank']})")
        c.plot(t["x_um"][okc], corr["E_meas"][okc], "o", color=C_MEAS, ms=8, label="measured")
        if np.isfinite(em["E_loo"]).any():
            c.plot(em["x_meas"], em["E_loo"], "o", mfc="none", mec=C_LOO, mew=2, ms=11, label="leave-one-out prediction")
        for k_, r in enumerate(rc["best"][: self.cfg.imaging.n_be_positions]):
            c.axvline(r["stage_um"], color=C_PICK, lw=1.2, ls="--")
            c.text(r["stage_um"], 0.97 - 0.07 * k_, f" BE stop: x={r['stage_um']:.0f}, E={r['E']:.0f}", color=C_PICK, va="top",
                   fontsize=9, transform=c.get_xaxis_transform())
        if es:
            c.plot([es["x_esbs_um"]], [np.interp(es["x_esbs_um"], em["x_grid"], em["E"])], "*", color=C_PICK, ms=14, mec="white",
                   label="QS stop (electrostatic blind spot)")
        Q = np.nanmedian(t["Q"])
        if np.isfinite(Q):
            c.axhline(Q, color="0.6", lw=1, ls=":"); c.text(em["x_grid"].min(), Q, f" Q = {Q:.0f}", va="bottom", fontsize=8, color="0.4")
        top = 1.6 * float(np.nanmax(np.concatenate([corr["E_meas"][okc], em["E"][(em["x_grid"] >= em["x_meas"].min()) & (em["x_grid"] <= em["x_meas"].max())]])))
        c.set_ylim(0, top if np.isfinite(top) and top > 0 else None)
        c.set_xlabel("stage x (um)"); c.set_ylabel("E_map = |P_CR1| / |P_QS|")
        c.set_title("c  resonance enhancement along the lever", loc="left", fontsize=10)
        c.legend(loc="upper right", fontsize=8, frameon=False)

        # (d) convergence
        if self.rel_change:
            ns = [h[0] for h in self.history[1:]]
            d.plot(ns, 100 * np.asarray(self.rel_change), "o-", color=C_MEAS)
            d.axhline(100 * cfg.capture.stable_rel, color=C_LOO, lw=1, ls="--")
            d.text(ns[0], 100 * cfg.capture.stable_rel, f" stable below {100 * cfg.capture.stable_rel:.0f} %",
                   va="bottom", fontsize=8, color=C_LOO)
            d.set_yscale("log")
        else:
            d.text(0.5, 0.5, "fills in during a live capture", ha="center", va="center", transform=d.transAxes, color="0.4")
        d.set_xlabel("positions measured"); d.set_ylabel("median change in E(x) (%)")
        d.set_title("d  how much the last position changed E(x)", loc="left", fontsize=10)
        ratios = em["loo_ratio"]
        rtxt = (f"E_meas/E_loo median {np.nanmedian(ratios):.2f} ({np.nanmin(ratios):.2f}-{np.nanmax(ratios):.2f})"
                if np.isfinite(ratios).any() else "")
        self.fig.suptitle(f"two-domain capture: {n} positions, {elapsed:.1f} min    {rtxt}"
                          + ("    STOP requested" if self.should_stop() else ""), fontsize=11)
        self._flush()

    def _flush(self):
        import matplotlib
        self.fig.tight_layout(rect=(0, 0, 1, 0.96))
        try:
            from IPython import get_ipython
            from IPython.display import display, clear_output
            ip = get_ipython()
        except Exception:
            ip = None
        backend = matplotlib.get_backend().lower()
        if ip is not None and ("inline" in backend or "agg" in backend):
            clear_output(wait=True)
            display(self.fig)
        else:
            self.fig.canvas.draw_idle()
            try:
                self.fig.canvas.flush_events()
            except Exception:
                pass

    def save(self, path):
        if self.fig is not None:
            self.fig.savefig(path, dpi=130)
        return path
