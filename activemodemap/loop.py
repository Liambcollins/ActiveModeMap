"""Active-learning driver: measure -> infer -> acquire -> repeat, with metrics."""

from __future__ import annotations

import numpy as np
from .forward_model import EBForwardModel
from .virtual_afm import VirtualAFM
from .inference import PhysicsPosterior
from . import acquisition, baselines
from .hybrid import HybridSurrogate, acquire_hybrid


INIT_POSITIONS = [0.35, 0.70, 0.95]   # small identifiability seed design


def run_loop(afm: VirtualAFM, strategy: str, n_total: int = 12,
             rng: np.random.Generator | None = None,
             n_post_samples: int = 50) -> dict:
    """Run one AL session. strategy in {'variance','eig','spot','random',
    'equispaced','gp'}. Returns per-step metrics + final state."""
    rng = rng or np.random.default_rng(0)
    model = afm.model
    physics = strategy in acquisition.PHYSICS_STRATEGIES
    post = (PhysicsPosterior(model, sigma=afm.sigma, rng=rng,
                             spot_fwhm_um=afm.spot_fwhm_um)
            if physics or strategy in ("random", "equispaced", "hybrid")
            else None)
    gp = baselines.GPMapLearner(model, rng=rng) if strategy == "gp" else None
    surro = None

    data, xs = [], []
    # evaluate reconstruction on BOTH domain maps (stacked along frequency axis)
    true_map = np.concatenate([np.abs(afm.true_maps[+1.0]),
                               np.abs(afm.true_maps[-1.0])], axis=0)
    norm = true_map.max()
    hist = {"n": [], "rmse": [], "dns_err": [], "desbs_err": [],
            "dns_ci": [], "desbs_ci": [], "xs": xs}

    for step in range(n_total):
        # ---- choose next position
        if step < len(INIT_POSITIONS):
            x_next = INIT_POSITIONS[step]
        elif strategy == "random":
            x_next = baselines.next_random(rng, xs)
        elif strategy == "equispaced":
            x_next = baselines.next_equispaced(step - len(INIT_POSITIONS),
                                               n_total, xs)
        elif strategy == "gp":
            x_next = gp.acquire(xs)
        elif strategy == "hybrid":
            x_next = acquire_hybrid(post, surro, xs, step, rng)
        else:
            x_next = acquisition.PHYSICS_STRATEGIES[strategy](post, xs)

        meas = afm.measure(x_next)
        data.append(meas)
        xs.append(meas["x"])

        # ---- update surrogate + metrics
        if strategy == "gp":
            gp.add(meas, afm.scale)
            pred = gp.predict_map(afm.scale)
            rmse = np.sqrt(np.mean((pred - true_map) ** 2)) / norm
            dns_e = desbs_e = dns_ci = desbs_ci = np.nan
        else:
            post.fit(data)
            if strategy == "hybrid":
                surro = HybridSurrogate(post).update(data)
                plus, minus = surro.corrected_maps()
                mean_amp = np.concatenate([np.abs(plus), np.abs(minus)], axis=0)
                spots = surro.spot_posterior(n_post_samples, rng)
            else:
                mean_amp, _, spots = post.predictive_maps(n_post_samples)
            rmse = np.sqrt(np.mean((mean_amp - true_map) ** 2)) / norm
            dns = spots[:, 0][np.isfinite(spots[:, 0])]
            desbs = spots[:, 1][np.isfinite(spots[:, 1])]
            dns_e = abs(np.median(dns) - afm.dns_um) if dns.size else np.nan
            desbs_e = abs(np.median(desbs) - afm.desbs_um) if desbs.size else np.nan
            dns_ci = np.percentile(dns, 97.5) - np.percentile(dns, 2.5) if dns.size else np.nan
            desbs_ci = np.percentile(desbs, 97.5) - np.percentile(desbs, 2.5) if desbs.size else np.nan

        hist["n"].append(step + 1)
        hist["rmse"].append(float(rmse))
        hist["dns_err"].append(float(dns_e))
        hist["desbs_err"].append(float(desbs_e))
        hist["dns_ci"].append(float(dns_ci))
        hist["desbs_ci"].append(float(desbs_ci))

    hist["posterior"] = post
    hist["gp"] = gp
    return hist
