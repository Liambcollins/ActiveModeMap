"""Virtual instrument: simulates 'park the detection laser at x, sweep frequency'.

Mirrors the experimental workflow of Figs. 3-4 in the manuscript: at each detection
position the instrument returns the combined (piezo + electrostatic) complex spectrum
for BOTH domain orientations of a PPLN-like sample (piezo sign flips, electrostatic
background does not) -- this is what makes the two pathways separable in practice.

Realism knobs:
- additive complex detector noise (constant displacement noise floor),
- finite laser-spot footprint: Gaussian blur of the response along x (~4 um FWHM),
- optional model mismatch: measurement generated from a *perturbed* physics model.
"""

from __future__ import annotations

import numpy as np
from .forward_model import EBForwardModel


class VirtualAFM:
    def __init__(self, model: EBForwardModel, theta_true: np.ndarray,
                 noise_floor: float = 0.08, spot_fwhm_um: float = 4.0,
                 rng: np.random.Generator | None = None,
                 mismatch_model: EBForwardModel | None = None):
        """noise_floor: detector noise sigma in units of the QUASISTATIC free-end
        signal (constant displacement noise floor; e.g. ~0.9 pm vs ~12 pm
        off-resonance signal in the manuscript -> ~0.08)."""
        self.model = model
        self.theta_true = np.asarray(theta_true, float)
        self.rng = rng or np.random.default_rng(0)
        self.noise_floor = noise_floor
        self.spot_fwhm_um = spot_fwhm_um
        gen = mismatch_model or model
        self._gen = gen

        resp = gen.response(self.theta_true)
        maps = {s: gen.measured(self.theta_true, s, resp) for s in (+1.0, -1.0)}
        # laser-spot footprint: Gaussian blur along x
        sig_um = spot_fwhm_um / 2.355
        sig_pts = sig_um / (gen.geom.L_um * (gen.xi[1] - gen.xi[0]))
        if sig_pts > 0.3:
            from scipy.ndimage import gaussian_filter1d
            maps = {s: gaussian_filter1d(m.real, sig_pts, axis=1)
                    + 1j * gaussian_filter1d(m.imag, sig_pts, axis=1)
                    for s, m in maps.items()}
        self.true_maps = maps
        self.scale = max(np.abs(maps[+1.0]).max(), np.abs(maps[-1.0]).max())
        # constant displacement noise floor, set relative to the quasistatic scale
        self.scale_qs = 0.5 * (np.abs(maps[+1.0][0, -1]) + np.abs(maps[-1.0][0, -1]))
        self.sigma = noise_floor * self.scale_qs
        # ground-truth spot positions (um from free end) from the generating model
        self.dns_um, self.desbs_um = gen.spots_um_from_end(self.theta_true)
        self.n_measurements = 0

    def measure(self, x_position: float) -> dict:
        """Frequency sweep at detection position x (scaled xi in [0,1]).

        Returns {'x': xi, 'plus': spectrum, 'minus': spectrum} (complex, len nf).
        """
        i = int(np.argmin(np.abs(self._gen.xi - x_position)))
        out = {"x": float(self._gen.xi[i])}
        for key, s in (("plus", +1.0), ("minus", -1.0)):
            clean = self.true_maps[s][:, i]
            add = self.sigma * (self.rng.standard_normal(clean.size)
                                + 1j * self.rng.standard_normal(clean.size)) / np.sqrt(2)
            gain = 1.0 + 0.02 * self.rng.standard_normal(clean.size)  # mult. noise
            out[key] = clean * gain + add
        self.n_measurements += 1
        return out
