## 4. Discussion

### 4.1 The enhancement is a transfer function, not a resonance property

The result in Figure 1c is the one with the widest practical reach. Dividing an on-resonance
amplitude by $Q$ is exact for a lumped oscillator driven and read at the same coordinate. A
cantilever in contact is driven at the tip and read at the laser spot, and the ratio of those two
transfer functions is a function of position that passes through the nodes of the mode. On this
probe it varies by 14.9× while $Q$ varies by 2 %. The two coincide at one position, near 157 µm
from the clamp, and there is nothing in a single-position measurement that would reveal how far
from that position one is sitting.

The practical consequences are concrete. First, an on-resonance $d_{33}$ quoted without an
independently determined enhancement carries an unbounded systematic error; on this lever the
error ranges from ×0.15 to ×2.2 depending only on laser placement. Second, the enhancement should
be measured rather than modelled whenever possible, and measuring it costs nothing extra: because
each swept spectrum spans 100 Hz to 2 MHz, the quasi-static and on-resonance responses come from
the *same* sweep at the *same* contact, so their ratio is available at every position already
visited. Third, laser position should be chosen deliberately. A position well away from a node
carries a stable enhancement — at 157 µm the value drifted only 13–14 % across a campaign that
altered the tip — whereas a node-adjacent position does not: at the free end the same probe's
enhancement changed by a factor of 7–12 between sessions, because the node's location is set by
the tip contact and the tip contact moves.

### 4.2 Exploration speed changes what is worth measuring

Two results bear on this. The low-rank reconstruction returns the complex wideband field at
never-visited positions from eight measured positions, at 0.19 NRMSE and 93 % within 3 dB — 18
minutes of measurement in place of 192. The reconstruction saturates at rank 4, so the field over
this span genuinely has only about four spatial degrees of freedom across 100 kHz – 1.9 MHz. Its
weakest band is CR3, where two interior nodes fall inside the measured interval; the general rule
that emerged is that reconstruction difficulty is set by how many interior nodes lie inside the
measured span, not by the probe type, and a design that is adequate for CR1 may be marginal two
modes up.

The load ladder is the second. Establishing that load does *not* matter for the recovered $d_{33}$
took 79 minutes and 90 spectra, and the answer it returned is worth more than the hour: a tenfold
load change moves the contact-resonance frequency by 1.4 % but doubles $Q$ and therefore raises the
enhancement by a factor of 2.5, while $V_{\mathrm{cpd}}$ and $d_{33}$ stay put. A workflow that sets load
once and treats the enhancement as a constant has the dependency exactly backwards. There is also
a negative result that only a ladder can produce: above ~100 nN this lever's CR1 sits at its
stiff-contact asymptote, so the frequency carries no contact-stiffness information at all, and no
improvement in frequency precision would change that. Reporting contact stiffness from CR1 on a
soft lever against a stiff substrate needs the saturation checked first, and checking it costs an
hour.

It is worth recording what did *not* work, because it is counter-intuitive. In earlier work on
this platform, information-maximising (active-learning) designs performed far worse than
equispaced ones for this task — the optimal design for estimating a few parameters clusters
measurements where the Jacobian is informative, whereas a low-rank spatial reconstruction needs
spatial spread. All designs reported here are therefore fixed and equispaced, with nothing
computed between positions.

### 4.3 Contact potential has a shelf life, and it is short

The contact potential is the single quantity that determined whether the imaging in this work was
quantitative or not. Measured twice within the session by different methods, it agreed to 30 mV
and produced domain equalisation to 1.1 %. Imported from four hours and one load excursion
earlier, it was wrong by 560 mV and the domains never equalised.

Two things make the in-session determination cheap enough to be routine. The bias survey that
determines it is also the measurement that separates the piezoelectric and electrostatic channels,
so it is not an extra experiment. And the three-point imaging extrapolation costs one additional
frame over the two-point version while converting the null from an extrapolation into a bracketed
crossing — which is what allowed the progression 0.669 → 0.821 → 1.011 to be verified as a bias
effect rather than drift, using a 0 V frame repeated at the end of the series that differed from
the first by under 2 %.

The agreement between the two laser positions deserves emphasis: 1.024 V at a position with
$E \approx 180$ and 1.031 V at a position with $E \approx 31$. These are different amplitude
scales, different noise levels, and a factor of six in mechanical gain, and they agree to 7 mV
because the crossing of two lines is invariant to the gain that scales them both.

### 4.4 What contact resonance is actually for

The conventional argument for contact resonance is that mechanical gain lifts a small signal above
a fixed detector noise floor. At $V_{ac} = 1$ V on this sample that argument fails: the
input-referred noise on resonance is 1.18 pm/V against 0.51 pm/V quasi-statically, and image SNR is
2.7 against 9.5. The pixel noise scales with the signal, so amplifying both changes nothing, and
the resonance channel adds a noise term the quasi-static channel does not have.

The argument succeeds at 30 mV, where the quasi-static signal falls below the floor entirely
(contrast 0.05 pm against 0.24 pm of noise) while the resonance channel still resolves the domains
and returns the same $d_{33}$ at the null. The correct statement is therefore not that contact
resonance improves signal-to-noise, but that it extends the usable drive range downwards by at
least a decade and a half — which matters for samples that cannot tolerate volts.

The extra noise term is identifiable. Between the two campaigns $Q_1$ rose from 179 to 222 and the
input-referred on-resonance noise rose from 0.49 to 1.18 pm/V while the quasi-static noise was
unchanged at 0.51–0.54 pm/V. A narrower resonance read at a fixed drive frequency converts small
excursions of the resonance frequency into amplitude noise, and the amplitude maps show it
directly as mottling absent from the quasi-static maps of the same frame (Figure 4). A higher $Q$
is not automatically better for fixed-frequency amplitude detection.

### 4.5 Limitations

**A cross-channel amplitude offset remains unresolved.** Converting the spectroscopy CR1 amplitude
to pm/V requires an enhancement, and applying the enhancement measured in the *imaging* channel to
the *spectroscopy* amplitudes over-states $d_{33}$ by 1.3–1.8× (the factor differs between the two
campaigns, so it is not a fixed instrument constant). The two quasi-static channels agree well —
imaging gives 6.50 and 11.31 pm/V at position A against a spectroscopy median of 8.66 pm/V along
the lever — so the discrepancy is specific to combining the two on-resonance channels. Figure 5a is
therefore presented normalised, and every absolute $d_{33}$ quoted in this work comes from a
quasi-static channel. Resolving this requires a deliberate cross-calibration in which the same
physical amplitude is read through both paths at the same instant.

**The per-domain enhancement is a convenience factor, not a mechanical gain.** $E$ is defined as
CR1(0 V)/quasi-static(0 V) for each domain separately, so it absorbs any domain-dependent
difference between the two channels at 0 V, where electrostatics are not nulled. That $E_1 \ne E_2$
(197 against 170 at position A) should not be read as the cantilever responding differently over
the two domains — it cannot know which domain it is over.

**The static-shape fit sits on its constraint boundary.** Fitting $1/\mathrm{InvOLS}(x)$ with a
clamped-beam profile gives $x_0 = +24.5$ µm and $L = 207.5$ µm, with $L$ pinned at
$\max(x) - x_0$; the data determine $x_0$ and $L$ follows. Applying the same fit to the companion
campaign's pre-flight curve gives $x_0 = +5.7$ µm, $L = 226.3$ µm, and restricting that fit to the
present campaign's narrower span changes it by 0.1 µm — so the ~19 µm difference between campaigns
is in the data, not an artefact of span. Since InvOLS is measured from force curves with the tip in
contact, a change in the effective contact point plausibly accounts for it, but this is not
established here.

**Scope.** All results come from one probe on one sample. The enhancement magnitudes, node
positions and contact-potential drift rates are specific to that combination; the methodological
findings — that $E \ne Q$, that $E$ is measurable from a single wideband sweep, that
$V_{\mathrm{cpd}}$ must be same-session, and that the low-drive rise is a detection floor — are not.

**The absolute $d_{33}$ scale carries an uncorrected InvOLS drift of about 10 %.** Every
$d_{33}$ in this work is referred to the pre-flight InvOLS curve. The probe checks in the load
extension re-measure InvOLS at the reference position and return 9.8-17.3 % higher values nine
hours later, drifting 6.8 % across a one-hour series (§3.2, Table 8). Relative comparisons - along
the lever, between channels, between the two campaigns, which all share the convention - are
unaffected, and the enhancement $E$ is a ratio and so is immune. But every absolute pm/V quoted
here should be read with a systematic of that size on top of the quoted scatter, and the remedy is
cheap: a force curve at a fixed reference position between stages, used by the reduction rather
than merely logged.

**The CR1 two-domain decomposition failed in the load extension.** At all three loads the
resonance-channel fit violates the consistency checks (§3.2), while the quasi-static fit at the
same loads passes cleanly. The most likely cause is that this one stage swept bias monotonically
rather than interleaved in sign. It is an argument for running the internal checks on every fit
and reporting them, rather than for distrusting the resonance channel generally - the same
decomposition passes at all eight positions of the bias survey - but it does mean the load
dependence of $E$ rests on the 0 V ladder alone, not on a decomposed piezoresponse.

**The model-based route works, but leans on one imported parameter.** The blind Euler–Bernoulli
fit of §3.4 predicts the withheld enhancement to 0.81–1.05× at seven interior positions and
recovers $d_{33} = 8.68$ pm/V (4.3 % s.d.) from the resonance channel using only the free-end
InvOLS, against 8.81 pm/V quasi-statically — the same agreement, on a second lever state, as the
companion campaign's 7.97 against 8.02 pm/V. The caveat is the beam length: the present dataset's
own static-shape fit hits the bound set by its measured span, so $L$ was imported from the
companion run's unconstrained fit of the same lever, and the quality of the blind prediction is
sensitive to that choice (§3.4). The import is of a *static* quantity and leaves the amplitude
prediction genuinely blind, but a future run should sample the laser position over a wider span so
that $L$ is determined in situ. It is also worth noting what the model does *not* fix: the
cross-channel offset above is unaffected, because it concerns two measurements of the same
resonance rather than the model of it.

---

## 5. Conclusions

1. **The contact-resonance enhancement is not $Q$.** Measured directly at eight positions on one
   lever, $E$ spans 34.9 to 520 while $Q$ stays at 233 ± 5. They coincide only near one position.
   An on-resonance $d_{33}$ divided by $Q$ is wrong by up to 2.2× on this probe, with no
   indication in the data.

2. **The enhancement can be measured in situ at no extra cost.** A swept spectrum covering
   100 Hz – 2 MHz contains the quasi-static and on-resonance response at the same position and the
   same contact; their ratio is the transfer function, model-free.

3. **Eight positions reconstruct the wideband field.** Low-rank truncation with spline-interpolated
   spatial factors predicts seven never-visited positions to 0.19 NRMSE, 93 % within 3 dB and 6.9°
   phase, saturating at rank 4 — 18 minutes standing in for 192.

4. **Load moves the damping, not the material number.** Over 50-500 nN the CR1 frequency changes
   1.4 % while $Q$ doubles and the enhancement rises x2.45 at every position along the lever;
   $V_{cpd}$ holds within 164 mV and $d_{33}$ within x1.13. Above ~100 nN the resonance frequency
   is at its stiff-contact asymptote and carries no stiffness information at any precision. The
   whole dimension cost 79 minutes.

5. **Contact potential must be same-session.** Two independent in-session determinations agreed to
   30 mV and two laser positions with 6× different enhancement agreed to 7 mV; a value imported
   from four hours earlier was 560 mV wrong and failed to equalise the domains.

6. **With a same-session null, on-resonance PFM is quantitative.** Domain amplitude ratios reach
   1.011 and 1.012 at two positions, phase separation is 177.9–180.0° throughout, and quasi-static
   $d_{33}$ along the lever is 7.83–9.67 pm/V (median 8.66, spread 1.24× over a 15× range of
   enhancement), reproducing to 6 % across two campaigns.

7. **There is no drive-amplitude nonlinearity.** Seventeen points from 2 mV to 2 V are flat to
   ±4 % above 50 mV, and the apparent rise below is reproduced to 2–6 % over three decades by a
   fixed additive detection floor of 570 µV.

8. **Contact resonance buys drive range, not signal-to-noise.** At 1 V it is the noisier channel;
   at 30 mV it is the only one that works. A higher $Q$ made the images noisier, not cleaner.

9. **A blind beam model supplies the transfer function when a sweep is not affordable.** An
   Euler–Bernoulli fit given only the three resonance frequencies, the static shape and $Q$ —
   no amplitudes — predicts the enhancement to 0.81–1.05× over seven positions and returns
   $d_{33} = 8.68$ pm/V from the resonance channel against 8.81 pm/V quasi-statically, a 1.5 %
   agreement, using no amplitude calibration beyond the free-end InvOLS. The same amplitudes
   divided by $Q$ vary by 6.4× along the lever.

---

## Data and code availability

Both campaigns are archived one experiment per folder, each containing the raw data, the
acquisition script exactly as it ran, the analysis scripts that produced the reported results, and
a `MANIFEST.md` giving the file count and type breakdown, the acquisition window in instrument
local time, the checkpoints and logs, and the role of every script. Run 1 (`DomainsB_SCMPIT`,
1255 files, 4.35 GB) has eleven experiment folders; Run 2 (`DomainsB_SCMPIT_R2`, 1235 files,
4.11 GB) has fifteen, the last three being the load extension of §3.2. Each campaign additionally
carries an `analysis/` folder holding the derived JSON summaries, the figures, and a snapshot of
every analysis script. Every number in this manuscript, including every table, is computed from
those summaries by the scripts included; no value is transcribed by hand.

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

---

## Appendix A — units convention

Two storage conventions coexist in this dataset and conflating them is the most common source of
error in analysing it.

| source | channel | stored as | conversion to metres |
|---|---|---|---|
| tune / spectroscopy checkpoints (`*_checkpoint.npz`) | complex $Z$ | detector volts | $\times\,\mathrm{InvOLS}/32$ |
| full-raster image files (`QImg*.ibw`, `LoAC*.ibw`) | `AmplitudeRetrace` | calibrated metres | none |

The factor 32 is the measured ratio between the deflection and amplitude channel sensitivities for
this probe (InvOLS $5.872\times10^{-7}$ m/V, AmpInvOLS $1.835\times10^{-8}$). Applying it to an
image file yields attometres; omitting it from a spectrum yields values $10^{7}$ too large. Both
errors occurred during this work and were caught by order-of-magnitude checks.

The complex spectrum is assembled as $Z = A\exp(-i\varphi)$ so that phase decreases through
resonance, matching a forward model solving $(K + i\omega C - \omega^2 M)q = F$.

## Appendix B — corrections to the earlier analysis

Two conclusions drawn from the companion campaign were overturned by the present dataset and are
recorded here because both were reasoning errors of a kind that is easy to repeat.

**A reported inversion of the domain amplitude ranking did not occur.** The companion campaign's
imaging analysis reported a domain ratio of 1.66 and its spectroscopy reported 0.61, and the
difference was written up as real, domain-dependent evolution of the contact. It was a labelling
artefact: $1/1.66 = 0.602$ against $0.613$ — the same measurement, reciprocated. The imaging labels
came from a clustering algorithm whose cluster order is arbitrary and happened to be inverted
relative to the hardware spot labels. A phase-sign cross-check appeared to confirm the labels but
used raw single-point 0 V phases, which the close-out later showed can differ by 50° between
measurements while the *fitted* domain flip angle holds at 182°. Re-analysed with labels anchored
to image geometry and cross-checked against the spectroscopy, the ratio is 0.604 in the companion
run and 0.669 here — an 11 % drift, not an inversion. **Anchor domain identity to a hardware
reference or to image geometry; never to a cluster index, a brightness ordering, or a single raw
phase sample.**

**An apparent 7.6 % free-air resonance shift was an uninformative measurement.** A best-effort tune
with the tip off the surface returned 58.93 kHz against a thermal $f_0$ of 63.801 kHz and was
flagged as possible tip mass loading. The identical step in the present campaign returned
69.81 kHz, 9 % *above* the thermal value. Three numbers spanning 58.9 / 63.8 / 69.8 kHz with no
consistent direction mean the electrically driven off-surface tune is not measuring the cantilever
resonance — there is no mechanical excitation with the tip retracted. The measurement should be
discarded rather than interpreted, and there is no evidence that the spring constant changed.
