# Findings index

Read in this order if you are new to the study. Corrections are kept in place
rather than edited away — where a claim was withdrawn, the doc that withdrew it
says so.

| doc | what it settles |
|---|---|
| `gp-only-baseline.md` | why a GP-only arm had to exist: the low-rank baseline flattered the physics |
| `fitting-and-model-content.md` | what is actually fitted (k₁, f_res, g, gain), and the Q6 EB/FEM confound: EB has k₂ = 0 while the FEM arm is isotropic |
| `fem-arm-al-results.md` | the FEM arm's numbers, and that bare FEM's sub-µm node accuracy is a property of the mode shape, not of convergence |
| `higher-modes-and-k1.md` | **k₁ is not identifiable from band A.** The f₂/f₁ ratio is 7.3× the more sensitive channel |
| `model-variants-mode-ratio.md` | which EB variant satisfies the mode ratio *and* Hertz at once (k₂ = k₁ + a 100 N/m cone, k₁ ≈ 1310) |
| `dual-peak-fit-results.md` | the two-band fit: k₁ = 989 ± 15 N/m over n = 3–30, mode 2 to one bin, and the drive asymmetry the per-band gain measures |
| `wideband-arm-comparison.md` | across the full 25 kHz – 1.775 MHz span: physics wins the tail, the GP wins the coverage, medians tie |
| `complex-spectrum-recon-and-uncertainty.md` | amplitude *and* phase: both forward models fail phase by 41–42°; σ calibration vs n; bias-dependence recovery |
| `fem-runs-wanted.md` | the FEM exports that were requested and why (see `archive/fem/`) |

## Transfer functions and mode shapes vs raw data

Produced by `figures/d_tf_shapes.py` (`results/fig/d8_tf.png`,
`d9_shapes.png`), scored on **withheld** positions only, 265–1000 kHz so both
contact modes are in frame — which means the only physics arm that can appear is
the two-band one, since the band-A libraries stop at 600 kHz.

- Mode 1's shape has the detection node: **2 decades of amplitude in 3 µm**, and
  the phase reverses through it. EB two-band + GP tracks it; GP alone smooths it.
- **Mode 2 has no node anywhere in the measured span.** It arches instead, with
  its maximum at x ≈ 174 µm, falling 34× toward the clamp. This is why band B
  carries stiffness information and no shape ambiguity.
- Phase is plotted **wrapped**, with the sub-threshold spans shaded: unwrapping
  produced ±400° artefacts in the noise valley, where the measured phase is
  uniform random and no arm can or should match it.
