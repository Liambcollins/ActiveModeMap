"""Instrument abstraction for the online active-learning loop.

The loop is hardware-agnostic: it only needs an object with

    measure_at(x_um) -> (freq_Hz, Z_complex, meta)

that positions the detection spot at x (µm) and returns one complex spectrum.
Two implementations are provided:

  * VirtualInstrument — simulates the measurement from the physics forward model,
    so the notebook logic can be dry-run without the microscope.
  * (AsylumInstrument lives in `asylum.py`; it wraps the Igor/Asylum automation.)
"""

from __future__ import annotations

import numpy as np


class Instrument:
    """Base class. Subclass and implement measure_at()."""

    def measure_at(self, x_um):
        raise NotImplementedError

    def close(self):
        pass


class VirtualInstrument(Instrument):
    """Simulated instrument backed by the EB forward model.

    Returns complex spectra on a fixed frequency grid with realistic noise and
    a finite detection-spot footprint, matching the real instrument's interface
    so the same loop code runs in both settings.
    """

    def __init__(self, freq_grid_Hz, theta_true=None, noise_floor=0.08,
                 spot_fwhm_um=4.0, domain_sign=+1.0, rng=None,
                 geom=None):
        from .forward_model import EBForwardModel, ProbeGeometry
        self.rng = rng or np.random.default_rng(0)
        self.freq = np.asarray(freq_grid_Hz, float)
        geom = geom or ProbeGeometry()
        # build a model whose frequency grid matches the requested window
        fs = geom.f0_hz / (1.87510407 ** 2)             # Hz per scaled omega
        omega = self.freq / fs
        m = EBForwardModel(geom=geom, n_modes=14, nx=241,
                           nf=self.freq.size,
                           omega_lo=float(omega.min()),
                           omega_hi=float(omega.max()))
        self.model = m
        self.L_um = geom.L_um
        self.domain_sign = domain_sign
        if theta_true is None:
            theta_true = np.array([3.2, np.log10(2000 / 3), 1.4, -0.2, 0.05])
        self.theta_true = np.asarray(theta_true, float)
        resp = m.response(self.theta_true)
        self._map = m.measured(self.theta_true, domain_sign, resp)   # (nf, nx)
        self.scale_qs = np.abs(self._map[0]).max()
        self.sigma = noise_floor * self.scale_qs
        # detection-spot blur along position
        sig_um = spot_fwhm_um / 2.355
        self._sig_pts = sig_um / (self.L_um * (m.xi[1] - m.xi[0]))
        if self._sig_pts > 0.3:
            from scipy.ndimage import gaussian_filter1d
            self._map = (gaussian_filter1d(self._map.real, self._sig_pts, axis=1)
                         + 1j * gaussian_filter1d(self._map.imag, self._sig_pts, axis=1))
        # true D-NS (distance from free end) for reference
        self.dns_um_from_end = (1 - m.find_dns(resp)) * self.L_um
        self.n_measurements = 0

    def measure_at(self, x_um):
        """x_um measured from the base end of the accessible span (0..L)."""
        xi = np.clip(x_um / self.L_um, 0, 1)
        i = int(np.argmin(np.abs(self.model.xi - xi)))
        clean = self._map[:, i]
        noise = self.sigma * (self.rng.standard_normal(clean.size)
                              + 1j * self.rng.standard_normal(clean.size)) / np.sqrt(2)
        self.n_measurements += 1
        meta = {"xi": self.model.xi[i], "position_index": i}
        return self.freq.copy(), clean + noise, meta


# --------------------------------------------------------------------------- #
#  Live plotting helper                                                       #
# --------------------------------------------------------------------------- #
def plot_state(mm, rec, ax=None, title=None):
    """Draw the current reconstruction, measured positions, and D-NS estimate.
    `mm` is a LowRankModeMap, `rec` its latest reconstruct() output."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    x, f = mm.x_grid, mm.freq / 1e3
    ext = [x[0], x[-1], f[0], f[-1]]
    amp = np.abs(rec["Zrec"]).T
    im0 = ax[0].imshow(np.log10(amp + 1e-9), origin="lower", aspect="auto",
                       extent=ext, cmap="viridis")
    for xs in rec["x_sel"]:
        ax[0].axvline(xs, color="w", lw=0.7, alpha=0.8)
    if np.isfinite(rec["dns"]):
        ax[0].axvline(rec["dns"], color="#e34948", ls="--", lw=1.3)
    ax[0].set_title(title or f"reconstruction ({mm.n} positions)")
    ax[0].set_xlabel("position (µm)"); ax[0].set_ylabel("frequency (kHz)")
    im1 = ax[1].imshow((rec["std"] / (amp.max() + 1e-12)).T, origin="lower",
                       aspect="auto", extent=ext, cmap="magma")
    ax[1].set_title("uncertainty (1σ, rel.)")
    ax[1].set_xlabel("position (µm)"); ax[1].set_ylabel("frequency (kHz)")
    if rec.get("dns_samples") is not None and len(rec["dns_samples"]):
        ax[2].hist(rec["dns_samples"], bins=16, color="#2a78d6", alpha=0.75,
                   density=True)
        ax[2].axvline(rec["dns"], color="k", lw=1.2,
                      label=f"D-NS {rec['dns']:.1f}±{rec['dns_ci']/2:.1f} µm")
        ax[2].legend(frameon=False, fontsize=8)
    ax[2].set_title("D-NS posterior"); ax[2].set_xlabel("position (µm)")
    return ax
