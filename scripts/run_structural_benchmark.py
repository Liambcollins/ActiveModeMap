"""Structural-mismatch benchmark: pure physics (EIG) vs hybrid (EB + GP
discrepancy + model-free spot extraction), over varied ground truths AND varied
structural perturbation strengths (each unknown to the inference model)."""

import time
import numpy as np

from activemodemap import EBForwardModel, VirtualAFM, run_loop
from activemodemap.hifi import HiFiForwardModel
from run_benchmark import sample_ground_truth

N_TRUTHS, N_TOTAL = 5, 10
model = EBForwardModel()
results = {}
t0 = time.time()
for t in range(N_TRUTHS):
    rng = np.random.default_rng(5000 + t)
    theta_true = sample_ground_truth(model, rng)
    hifi = HiFiForwardModel(krot_frac=rng.uniform(0.05, 0.2),
                            apex_load_frac=rng.uniform(0.15, 0.5),
                            damp_slope=rng.uniform(0.15, 0.5),
                            sens_ampl=rng.uniform(0.01, 0.05))
    for strat in ("eig", "hybrid"):
        rng_run = np.random.default_rng(rng.integers(1 << 31))
        afm = VirtualAFM(model, theta_true, rng=rng_run, mismatch_model=hifi)
        try:
            h = run_loop(afm, strat, n_total=N_TOTAL, rng=rng_run)
            rec = np.array([h["rmse"], h["dns_err"], h["desbs_err"],
                            h["dns_ci"], h["desbs_ci"]])
        except Exception as exc:
            print(f"FAILED {strat}/gt{t}: {exc}", flush=True)
            rec = np.full((5, N_TOTAL), np.nan)
        results.setdefault(strat, []).append(rec)
        print(f"[{time.time()-t0:6.1f}s] gt{t} {strat:7s} "
              f"final: dns={rec[1,-1]:.3f} desbs={rec[2,-1]:.3f} "
              f"(truth dns={afm.dns_um:.2f} desbs={afm.desbs_um:.2f})",
              flush=True)
np.savez("structural_results.npz",
         **{k: np.array(v) for k, v in results.items()})
print(f"done in {time.time()-t0:.0f}s")
