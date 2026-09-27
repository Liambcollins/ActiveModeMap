## Tables

**Table 1.** Measurement stages, wall clock. The overnight campaign (first nine) ran 22:56-05:28 for 392 min with zero failures; the last three (italic) are the load extension of §3.2, 79 min, run the following morning. Two analysis-only stages that derive $V_{cpd}$ from the preceding data are not listed.

| stage | design | wall clock (min) |
|---|---|---|
| Pre-flight | 17 positions, 10 µm steps, InvOLS + tune at each | 20.8 |
| Bias survey | 8 positions × 7 biases × 2 domains, 500 nN, 1 V | 45.5 |
| Wideband capture | 8 equispaced positions, 0 V, 500 nN | 9.6 |
| Wideband validation | 7 held-out midpoints | 8.2 |
| Quantitative imaging | 9 frames, 2 positions, 3 biases | 42.4 |
| Low-drive imaging | position A repeated at $V_{ac}$ = 30 mV | 14.7 |
| Drive series | 17 amplitudes, 2 mV – 2 V, both domains | 42.1 |
| Dense reference map | 161 positions at 1 µm | 192.4 |
| Close-out | reference condition, walk, final InvOLS | 16.2 |
| *Load ladder* | 5 loads × 8 positions, probe check after each | 52.7 |
| *Bias × load* | 3 loads × 7 biases × 2 domains, one position | 16.8 |
| *Post-load walk* | 8 positions, 0 V, 500 nN | 9.5 |

**Table 2.** Cantilever transfer function from the Run 2 bias survey: eight laser positions, 500 nN, $V_{ac}$ = 1 V, $\pm 9$ V in seven steps on both domains. $|P|$ is the decomposed piezoresponse in detector volts; $E = |P|_{CR1}/|P|_{qs}$.

| $x$ from clamp (µm) | InvOLS (m/V) | CR1 (kHz) | $Q$ | $|P|_{qs}$ (V) | $|P|_{CR1}$ (V) | $E$ | $V_{cpd}$ (V) | $|b|/|P|$ | $d_{33}$ (pm/V) |
|---|---|---|---|---|---|---|---|---|---|
| 65.9 | 7.971e-06 | 295.53 | 235 | 3.882e-05 | 2.020e-02 | 520.4 | +0.901 | 0.395 | 9.67 |
| 88.8 | 3.722e-06 | 295.39 | 235 | 8.220e-05 | 3.628e-02 | 441.3 | +1.013 | 0.346 | 9.56 |
| 111.6 | 2.202e-06 | 295.33 | 229 | 1.369e-04 | 4.608e-02 | 336.6 | +1.123 | 0.298 | 9.42 |
| 134.5 | 1.479e-06 | 295.50 | 235 | 1.906e-04 | 5.894e-02 | 309.3 | +1.085 | 0.250 | 8.81 |
| 157.3 | 1.069e-06 | 295.50 | 235 | 2.549e-04 | 5.965e-02 | 234.1 | +1.057 | 0.198 | 8.51 |
| 180.2 | 8.282e-07 | 295.56 | 241 | 3.159e-04 | 5.349e-02 | 169.3 | +1.145 | 0.146 | 8.18 |
| 203.0 | 6.661e-07 | 295.53 | 223 | 3.760e-04 | 3.567e-02 | 94.9 | +1.205 | 0.093 | 7.83 |
| 225.9 | 5.952e-07 | 295.29 | 229 | 4.230e-04 | 1.474e-02 | 34.9 | +1.685 | 0.042 | 7.87 |

Median $d_{33}$ 8.66 pm/V over a 15× range of enhancement; position calibration $x_0$ = +24.5 µm, 2.2 % rms.

**Table 3.** Low-rank wideband reconstruction, scored on seven never-visited positions over 100 kHz – 1.9 MHz. Both campaigns through identical code.

| rank | R2 NRMSE | R2 within 3 dB | R2 phase MAE | R2 CR3 band | R1 NRMSE | R1 CR3 band |
|---|---|---|---|---|---|---|
| 4 | 0.192 | 80 % | 11.3° | 0.245 | 0.275 | 0.393 |
| 5 | 0.193 | 93 % | 6.9° | 0.247 | 0.265 | 0.380 |
| 6 | 0.196 | 95 % | 6.8° | 0.251 | 0.271 | 0.378 |

**Table 4.** Quantitative imaging, Run 2. Every $d_{33}$ is amplitude ÷ ($V_{ac}$ × the in-situ enhancement measured for that domain); quasi-static frames need only ÷ $V_{ac}$.

| position | channel | bias (V) | $d_{33}$ domain 1 | $d_{33}$ domain 2 | amplitude ratio | phase separation |
|---|---|---|---|---|---|---|
| A | quasi-static 20 kHz | +0.000 | 6.50 | 11.31 | 0.575 | 179.4° |
| A | CR1 | +0.000 | 6.50 | 11.31 | 0.669 | 179.8° |
| A | CR1 | +0.529 | 7.00 | 9.92 | 0.821 | 179.7° |
| A | CR1 (null) | +1.057 | 8.08 | 9.30 | 1.011 | 179.0° |
| B | quasi-static 20 kHz | +0.000 | 7.23 | 8.75 | 0.826 | 179.4° |
| B | CR1 | +0.000 | 7.23 | 8.75 | 0.664 | 179.7° |
| B | CR1 (null) | +1.057 | 9.01 | 7.16 | 1.012 | 179.4° |

**Table 5.** Magnitude-detection floor, $|Z| = \sqrt{(kV_{ac})^2 + n_0^2}$, fitted to the 17-point drive series over three decades.

| series | $k$ (mV/V) | $n_0$ (µV) | signal = floor at | median residual |
|---|---|---|---|---|
| Run 2, domain 1 | 56.4 | 569 | 10.1 mV | 4.3 % |
| Run 2, domain 2 | 83.1 | 570 | 6.9 mV | 6.1 % |
| companion, domain 1 | 52.8 | 1041 | 19.7 mV | 2.1 % |
| companion, domain 2 | 86.2 | 417 | 4.8 mV | 2.8 % |

**Table 6.** Load ladder. $E$ is the median over the positions that returned a resonance; $k^*$ is inverted from CR1 with the blind-fit geometry of §3.4 held fixed, and is undetermined once the measurement reaches the model's stiff-contact asymptote (293.3 kHz).

| load (nN) | positions with CR1 | CR1 (kHz) | $Q$ | median $E$ | $k^*/k_{\mathrm{lever}}$ | $\mathrm{d}\ln f/\mathrm{d}\ln k^*$ |
|---|---|---|---|---|---|---|
| 50 | 6/8 | 291.01 ± 612 Hz | 117 | 115 | 642 | 0.0085 |
| 100 | 7/8 | 292.60 ± 216 Hz | 141 | 191 | 1992 | 0.0024 |
| 200 | 8/8 | 293.75 ± 78 Hz | 169 | 279 | saturated | < 0.001 |
| 350 | 8/8 | 294.59 ± 134 Hz | 211 | 329 | saturated | < 0.001 |
| 500 | 8/8 | 295.18 ± 190 Hz | 235 | 382 | saturated | < 0.001 |

**Table 7.** Bias series at three loads, working position (154.9 stage-µm), $\pm 9$ V in seven steps on both domains. The quasi-static decomposition passes the checks of §2.5 at every load; the CR1 decomposition fails all three (flip angle far from 180°, $|b_2/b_1|$ far from 1), so no CR1 quantity from this stage is quoted. The last column applies the InvOLS the probe checks measured, in place of the pre-flight curve.

| load (nN) | $V_{cpd}$ quasi-static (V) | $|P|_{qs}$ (V) | $|b|/|P|$ | flip angle | $|b_2/b_1|$ | CR1 fit | $d_{33}$ (pm/V) | $d_{33}$, measured InvOLS |
|---|---|---|---|---|---|---|---|---|
| 100 | +0.870 | 1.777e-04 | 0.274 | 182.3° | 0.981 | fails: 218°, 0.65 | 6.67 | 7.50 |
| 300 | +1.034 | 1.961e-04 | 0.252 | 181.3° | 0.997 | fails: 223°, 0.57 | 7.36 | 8.18 |
| 500 | +1.009 | 2.007e-04 | 0.244 | 181.3° | 1.004 | fails: 200°, 0.92 | 7.53 | 8.27 |

**Table 8.** InvOLS re-measured at the reference position (232 stage-µm) after each load step, against the pre-flight curve used throughout the reduction. The pre-flight was taken nine hours earlier.

| after load (nN) | measured InvOLS (m/V) | pre-flight (m/V) | ratio |
|---|---|---|---|
| 50 | 6.9810e-07 | 5.9519e-07 | 1.173 |
| 100 | 6.6975e-07 | 5.9519e-07 | 1.125 |
| 200 | 6.5865e-07 | 5.9519e-07 | 1.107 |
| 350 | 6.6282e-07 | 5.9519e-07 | 1.114 |
| 500 | 6.5378e-07 | 5.9519e-07 | 1.098 |

Mean 1.123, drifting 6.8 % across the series.
