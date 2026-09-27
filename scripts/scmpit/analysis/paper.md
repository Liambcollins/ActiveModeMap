# Recovering the cantilever transfer function in situ: quantitative on-resonance piezoresponse force microscopy with same-session contact-potential and $d_{33}$ calibration

**Liam Collins** — Center for Nanophase Materials Sciences, Oak Ridge National Laboratory

---

## Abstract

Contact-resonance piezoresponse force microscopy (CR-PFM) amplifies a weak electromechanical
signal by a factor that is routinely assumed to equal the resonance quality factor $Q$. We show by
direct measurement that this assumption fails by more than an order of magnitude over most of a
cantilever. Using an autonomously operated Cypher AFM in which the detection laser is stepped
along the lever under software control, we measure the contact-resonance enhancement
$E(x) = |P_{\mathrm{CR1}}(x)| / |P_{\mathrm{qs}}(x)|$ at eight positions on a single SCM-PIT probe
in contact with periodically poled lithium niobate, and find $E$ spanning **34.9 to 520** while $Q$
stays flat at **233 ± 5**. The two coincide only within a few micrometres of one particular
position. A 161-position map at 1 µm pitch resolves the first three contact-resonance mode shapes
and locates their nodes to ±1 µm, and a low-rank reconstruction from only **eight** measured
positions predicts the full 100 kHz – 1.9 MHz complex field at seven never-visited positions to
0.19 NRMSE and 93 % within 3 dB — 18 minutes of measurement standing in for 192.

With the transfer function known, the remaining obstacle to quantitative CR-PFM is the contact
potential. We show that $V_{\mathrm{cpd}}$ must be measured in the same session and the same tip
state: a value imported from four hours and one load excursion earlier was wrong by 560 mV and
left the two antiparallel domains unequalised, whereas two independent same-session determinations
— a $\pm 9$ V bias survey and a three-point extrapolation of the imaging amplitudes — agreed to
**30 mV**, and at that null the domain amplitude ratio reached **1.011** at one laser position and
**1.012** at another with 70× different enhancement. Quasi-static $d_{33}$ recovered along the
lever is 7.83–9.67 pm/V (median 8.66, spread 1.24×), reproducing to within 6 % across two
campaigns separated by a probe-altering load excursion. A 17-point drive series from 2 mV to 2 V
shows no amplitude nonlinearity; the apparent rise below 30 mV is quantitatively accounted for by
a fixed additive detection floor, $|Z| = \sqrt{(kV_{ac})^2 + n_0^2}$, fitted to within 2–6 % over
three decades. A five-point load ladder, 90 spectra in 79 minutes, shows that a tenfold change in load moves the
contact-resonance frequency by only **1.4 %** — it is already at the stiff-contact asymptote above
~100 nN — while $Q$ doubles and the enhancement rises **×2.5** at every position along the lever, and
$V_{\mathrm{cpd}}$ and $d_{33}$ stay put within 164 mV and ×1.13. We quantify when resonance
actually helps: at 1 V drive it does not, because
the pixel noise scales with the signal, but at 30 mV the quasi-static signal disappears below the
floor while the resonance channel remains usable. Finally, where a wideband sweep is not
affordable, we show the transfer function can be supplied by a model instead: an Euler–Bernoulli
fit given only the three resonance frequencies, the static laser-response shape and $Q$ — with
every amplitude withheld — predicts the measured enhancement to **0.81–1.05×** at seven positions
and converts the on-resonance amplitude to $d_{33} = 8.68$ pm/V against 8.81 pm/V quasi-statically,
a **1.5 %** agreement, using no amplitude calibration beyond the free-end InvOLS. The same
amplitudes divided by $Q$ vary by 6.4× along the lever.

---

## 1. Introduction

Piezoresponse force microscopy measures a surface displacement of order picometres per volt. The
displacement is small enough that most implementations drive the tip at or near a contact
resonance of the cantilever, where the mechanical response is amplified. That amplification is the
whole reason the technique works at low drive, and it is also the reason the technique is hard to
make quantitative: the measured amplitude is the product of the material property one wants and a
cantilever transfer function one usually does not know.

The standard shortcut is to divide the on-resonance amplitude by the quality factor $Q$. This is
exact only for a lumped harmonic oscillator whose displacement is read out at the point where the
force is applied. A real cantilever in contact is neither: the electromechanical drive acts at the
tip, the optical lever reads the slope of the beam at wherever the laser happens to sit, and the
ratio between the two is a function of position along the lever that passes through zeros at the
nodes of each mode. Whether $Q$ is a usable stand-in for that ratio is an empirical question, and
it is one that can be answered directly if the laser position can be controlled and the
quasi-static response measured at the same positions.

A second obstacle is electrostatic. The measured response contains a piezoelectric term that
reverses sign between antiparallel domains and an electrostatic term that does not. Separating
them requires a bias sweep, and nulling the electrostatic contribution requires the contact
potential difference $V_{\mathrm{cpd}}$. Contact potential is a property of the specific tip and
the specific contact, and both evolve. How quickly they evolve, and therefore how stale a
$V_{\mathrm{cpd}}$ value may be before it stops being usable, is rarely quantified.

This paper addresses three questions with one dataset acquired autonomously in a single
engage-to-withdraw session:

1. **Can the cantilever transfer function be recovered in situ, and is $Q$ an acceptable
   approximation to it?** (Section 3.1)
2. **How quickly can load and bias be explored, and what does that exploration reveal about the
   electrostatic and piezoelectric channels?** (Section 3.2)
3. **What does it take to make on-resonance PFM quantitative — to produce a map in pm/V in which
   the two domains of a poled ferroelectric report the same magnitude and opposite sign?**
   (Section 3.3)

All measurements reported here come from a single probe on a single sample, acquired without
operator intervention through a file-queue relay driving the microscope's scripting interface. The
dataset is one of a pair; the companion run, which included a 50–1500 nN load ladder, altered the
tip midway and is used here only where explicitly noted. Section 5 and Appendix B record what that
comparison taught us, including two conclusions from the earlier analysis that the present dataset
overturned.

---

## 2. Methods

### 2.1 Instrument, probe and sample

Measurements were made on an Asylum Research Cypher AFM under Igor Pro COM automation. The probe
was a Nanosensors SCM-PIT (Pt–Ir coated silicon), characterised in free air before any contact:
spring constant $k = 1.697$ N/m by thermal calibration, free resonance $f_0 = 63.801$ kHz, free-air
$Q_0 = 175.1$, and inverse optical lever sensitivity InvOLS $= 5.872 \times 10^{-7}$ m/V with the
laser at the free end. The ratio between the deflection and amplitude channel sensitivities was
measured for this probe as **÷32** and applied throughout.

The sample was periodically poled lithium niobate with two antiparallel domains visible in a single
6 µm frame. Two locations, one on each domain, were marked on an optical image and revisited
throughout by the microscope's spot-recall function; they are referred to as spot 1 and spot 2.
The standard contact load was 500 nN, giving a static deflection of 295 nm.

### 2.2 Autonomous acquisition

Acquisition scripts were executed inside the live Igor kernel through a file-queue relay: scripts
dropped into a queue directory are executed in the kernel's namespace, moved to a `done` or
`failed` directory, and a heartbeat file records state. This allows a multi-stage campaign to be
composed ahead of time and run unattended while analysis of earlier stages proceeds in parallel.
Every acquisition script resets the DC bias to zero and withdraws the tip in a `finally` block.

The campaign reported here ran 22:56–05:28 (6 h 32 min) as twelve queued stages with zero failures,
followed by a load extension described in Section 3.2. Stage durations are given in Table 1.

### 2.3 Measurement stages

**Pre-flight.** The laser was stepped from the free end toward the clamp in 10 µm increments,
recalibrating InvOLS and acquiring a 100 Hz – 2 MHz tune at each step, until the detection failed a
sanity test. This established the reachable span (72–232 stage-µm, 17 positions) and the
contact-resonance frequencies at 500 nN.

**Bias survey.** Eight positions spanning the reachable range × seven bias values
($-9$ to $+9$ V, interleaved in sign so that drift appears as scatter rather than as a bias slope)
× two domains, at 500 nN and $V_{ac} = 1$ V. Each spectrum is 58 816 complex points from 100 Hz to
2 MHz, so a single survey contains both the quasi-static and the on-resonance response at every
condition.

**Wideband capture and validation.** Eight equispaced positions measured in one pass with no
adaptive decision-making, then seven never-visited midpoints measured separately as a held-out
test set.

**Dense reference map.** 161 positions at 1 µm pitch, 0 V, 500 nN, spot 1 — 192 minutes.

**Quantitative imaging.** The same 6 µm frame imaged quasi-statically (20 kHz drive) and on CR1 at
two laser positions and three bias values, plus a repeat at 0 V as a drift check and a 1 V/2 V pair
for drive linearity.

**Low-drive imaging.** The position-A imaging repeated at $V_{ac} = 30$ mV.

**Drive series.** Seventeen drive amplitudes from 2 mV to 2 V at one position, both domains, at
0 V and at the measured null.

**Close-out.** The opening reference condition re-measured, plus a repeat of the mode-shape walk
and a final InvOLS, to bound the drift over the whole campaign.

### 2.4 Signal conventions and units

Raw tune and spectroscopy checkpoints store amplitude in detector volts and require
$\times \mathrm{InvOLS}/32$ to become metres. Full-raster image files store the amplitude channel
already in calibrated metres and require no further factor. Conflating the two produces results
wrong by seven orders of magnitude in either direction, and both errors were made and caught during
this work; the convention is stated explicitly here because it is the single most common source of
unit error in this kind of dataset.

The complex spectrum is built as $Z = A\exp(-i\varphi)$, placing the data in the same convention as
a forward model solving $(K + i\omega C - \omega^2 M)q = F$.

### 2.5 Two-domain decomposition

At each position and each spot the complex response is linear in DC bias,

$$Z_s(V) = a_s + b_s V,$$

with $s \in \{1,2\}$ the two domains. The piezoresponse and electrostatic terms separate as

$$P = \tfrac{1}{2}(a_1 - a_2), \qquad b = \tfrac{1}{2}(b_1 + b_2), \qquad
V_{\mathrm{cpd}} = -\,\frac{a_1 + a_2}{2b},$$

because the piezoresponse reverses sign between antiparallel domains and the electrostatic term
does not. Three internal consistency checks accompany every fit: the domain flip angle
$|\arg a_1 - \arg a_2|$ should be 180°, the electrostatic balance $|b_2/b_1|$ should be unity, and
the residual of the linear fit should be small. Values are reported with each result.

Quasi-static quantities are taken as the complex mean over the 15–45 kHz plateau, far below the
first contact resonance; on-resonance quantities are taken at the CR1 peak bin.

**Domain labelling.** Domain identity is anchored to image geometry — both campaigns imaged the
identical frame at identical offsets, so the same physical domain occupies the same pixels — and
cross-checked against the hardware spot labels. Labelling by cluster index or by which domain is
brighter is not stable across tip states and produced a spurious result in our earlier analysis
(Appendix B).

---

## 3. Results

### 3.1 Recovery of the cantilever transfer function

**The measured enhancement spans a factor of fifteen and is nothing like $Q$.**
Because every spectrum in the bias survey covers 100 Hz to 2 MHz, the quasi-static and
on-resonance piezoresponse are available at the same position, same contact, same instant. Their
ratio is the contact-resonance enhancement,

$$E(x) = \frac{|P_{\mathrm{CR1}}(x)|}{|P_{\mathrm{qs}}(x)|},$$

measured with no model and no assumption. Table 2 and Figure 1c give the result: $E$ falls
monotonically from **520 at 65.9 µm from the clamp to 34.9 at the free end**, a factor of 14.9,
while $Q$ extracted from the half-power width of the same peaks is flat at **233 ± 5**. The two
agree only near 157 µm. Dividing an on-resonance amplitude by $Q$ therefore over- or under-states
$d_{33}$ by up to 2.2× depending only on where the laser sits — an error that is invisible in the
data and carries no warning.

The physical origin is that the enhancement is a ratio of two different transfer functions — drive
at the tip, readout of beam slope at $x$ — and each has its own spatial dependence. $Q$ describes
the sharpness of the resonance, not the coupling between them.

**Mode shapes and nodes.** The 1 µm map (Figure 1a) resolves the first three contact resonances
along the beam. Node positions, taken as interior minima of $|Z|$, are **146.9 µm** (CR2) and
**110.9 and 178.9 µm** (CR3) from the clamp. CR1 falls monotonically toward the free end with no
interior minimum, consistent with a node at or beyond the tip for a pinned contact.

**The map is drift-free.** Over the 192 minutes of the dense map, CR1 varied by 150 Hz rms across
all 161 positions, CR2 by 1.3 kHz and CR3 by 1.3 kHz once positions inside a node are excluded
(Figure 1b). The reference against which everything else is scored is therefore stationary to
5 × 10$^{-4}$ in frequency.

**Static shape and the position axis.** The optical lever sensitivity itself carries the static
mode shape. Fitting $1/\mathrm{InvOLS}(x) \propto (x-x_0)^2\,[3L-(x-x_0)]$ to the eight survey
positions gives $x_0 = +24.5$ µm and $L = 207.5$ µm at 2.2 % rms (Figure 1d). Per-position InvOLS
values were interpolated from the 17-point pre-flight curve and cross-checked against the force
curve measured at each survey position; the two agree to 1–3 %.

**Eight positions are enough to reconstruct the whole field.** Singular-value truncation of the
8-position complex matrix, with the spatial factors interpolated by cubic spline, predicts the
complex response at seven never-visited positions over 100 kHz – 1.9 MHz with NRMSE 0.192 at rank 4,
saturating immediately (0.193 at rank 5, 0.196 at rank 6); 93 % of the spectrum falls within 3 dB
and the phase mean absolute error is 6.9° (Figure 1f, Table 3). The 18 minutes of measurement this
requires substitutes for the 192 minutes the dense map took. The reconstruction is worst in the CR3
band (NRMSE 0.245), where two interior nodes fall inside the measured interval.

### 3.2 Rapid exploration of load and bias

With the calibration machinery in place, load becomes cheap to explore. A five-point ladder at
50, 100, 200, 350 and 500 nN, each with a full 100 Hz - 2 MHz sweep at eight laser positions, took
**52.7 minutes**; a bias series of seven DC values on both domains at three loads took a further
**16.8 minutes**; and a post-load walk over the same eight positions took **9.5 minutes**. The
whole load dimension - 90 wideband spectra - cost **79 minutes** of unattended instrument time
(Figure 2, Table 1).

**The contact state is unchanged by the excursion.** A probe check after every load step returned
CR1 within $-0.21$ to $+0.03$ % of the pre-flight value, and the post-load walk gives a CR1 mean of
295.26 kHz across eight positions with a 136 Hz spread - **$-0.03$ %** against the dense map
measured four hours and one full load ladder earlier. Unlike the companion campaign, whose load
excursion demonstrably altered the tip (§3.3), this one did not, so the load series is directly
comparable with the rest of the dataset.

**The resonance frequency is saturated; the damping is not.** Over a tenfold change in load the
CR1 frequency moves from 291.01 to 295.18 kHz - **+1.43 %** - while $Q$ doubles, 117 to 235
(Figure 2a). The reason is visible when the measured frequencies are placed on the model's
$f(k^*)$ curve with the blind-fit geometry of §3.4 held fixed (Figure 2b): the curve flattens into
its stiff-contact (pinned) asymptote at 293.3 kHz, and the 200, 350 and 500 nN measurements all sit
*at or above* that asymptote. Only the two lightest loads retain any sensitivity, and even at
100 nN $\mathrm{d}\ln f / \mathrm{d}\ln k^*$ is **0.0024** - a 1 % frequency measurement would
constrain the contact stiffness to a factor of 50. On a lever this soft against a substrate this
stiff, contact-resonance frequency is not a contact-stiffness measurement above ~100 nN, at any
precision. The two loads that are informative give $k^*/k_{\mathrm{lever}} = 642$ (50 nN) and
1992 (100 nN), i.e. $k^* \approx 1.1$ and $3.4$ kN/m (Table 6).

**The enhancement moves with load, everywhere.** Over the six positions that returned a
resonance at every load, median $E$ rises **x2.45** from 50 to 500 nN, and the rise is present at
every one of them, ranging x1.9 to x2.8 (Figure 2c). The whole $E(x)$ curve translates upward
without appreciably changing shape. This is the practical consequence of the $Q$ doubling, and it
is the reason an enhancement measured at one load cannot be carried to another - a factor of two
to three, applied silently. (Taken over all positions available at each load the ratio is x3.3,
but that comparison is inflated: the two positions missing at 50 nN are the two highest-$E$
positions on the lever, so the like-for-like figure is the one quoted.)

**Three acquisitions at low load are unusable, and they say where the limit is.** At 50 nN the
two positions nearest the clamp returned no detectable contact resonance at all (band peak below
eight times the band median), and at 100 nN the nearest position still did; from 200 nN upward all
eight positions are recovered. These are also the positions of poorest optical-lever sensitivity,
where InvOLS is 8-13x larger than at the free end. Their *quasi-static* amplitude in the same
sweeps is anomalous too - 6-8x above the value the same position returns at every higher load,
against the smooth monotone trend every other position shows - so these three sweeps are
noise-dominated in both channels rather than merely lacking a resonance, and they are excluded
throughout. The honest statement is therefore narrower than it first appears: the low-load limit
observed here is a detection limit at the clamp-proximal positions, and this dataset does not
separate a genuinely weaker resonance from a poorer measurement of it.

**What load does not change is the material number.** Over a fivefold load range at the working
position, the quasi-static two-domain decomposition gives $V_{\mathrm{cpd}} = +0.870$, $+1.034$ and
$+1.009$ V at 100, 300 and 500 nN - a **164 mV** spread dominated by the lightest load - against
$+1.057$ V from the bias survey two hours earlier, and the electrostatic fraction $|b|/|P|$ falls
only from 0.274 to 0.244 (Figure 2d, Table 7). All three quasi-static fits pass the internal
checks of §2.5 (flip angle 181.3-182.3°, $|b_2/b_1|$ 0.98-1.00). Quasi-static $d_{33}$ is 6.67,
7.36 and 7.53 pm/V, a **x1.13** range. The residual load dependence is at the level of the
position-to-position spread already present in the bias survey (x1.24), so within this dataset load
is not a variable that has to be controlled to recover $d_{33}$ - which is what makes a five-point
ladder an adequate way to establish that, and worth the hour it costs.

**The CR1 decomposition fails here, and the checks catch it.** At all three loads the *resonance*
channel's two-domain fit violates the consistency conditions of §2.5: the domain flip angle is
199.8-222.7° rather than 180°, $|b_2/b_1|$ is 0.57-0.92 rather than unity, and the implied
$V_{\mathrm{cpd}}$ is negative at two of the three loads, against $+0.49$ to $+0.85$ V at all eight
positions of the bias survey. No $V_{\mathrm{cpd}}$ or $E$ from the CR1 channel of this stage is
quoted, and Table 7 marks them. One plausible cause is the bias order: this stage swept DC
monotonically from $-9$ to $+9$ V, whereas the bias survey interleaves the sign so that any
accumulating injected charge appears as scatter rather than as a slope. If so, the resonance
channel is the more sensitive detector of that failure, which is itself worth knowing; the
interleaved order should be used at every load in future.

**The absolute scale carries a calibration drift the probe checks measured and the reduction did
not use.** Every $d_{33}$ above uses the pre-flight InvOLS curve, for comparability with the bias
survey, which uses the same curve. But the probe check re-measures InvOLS at the reference position
after each load, and it returns values **+9.8 to +17.3 %** above that curve (Table 8), drifting monotonically
downward through the series - the pre-flight was taken nine hours earlier. Applying the measured
factor moves $d_{33}$ to 7.50, 8.18 and 8.27 pm/V (Figure 2d, grey against teal). The 500 nN value
is then **4 %** below the 8.62 pm/V obtained by interpolating the bias survey to this position,
instead of 13 %. Two consequences follow. First, the apparent load-extension-versus-survey
discrepancy is mostly an InvOLS calibration that aged, not a change in the sample or the tip - and
the stability checks that show the contact state unchanged are consistent with that, not in tension
with it. Second, because the bias survey uses the same aged curve, the absolute $d_{33}$ scale
of this work carries an estimated **10 %** systematic that the *relative* comparisons along the
lever and between channels do not. A force curve at the reference position between stages, and a
reduction that uses it, would remove this; it is the single cheapest improvement available to the
protocol.

### 3.3 Contact potential and $d_{33}$ for quantitative on-resonance PFM

**$V_{\mathrm{cpd}}$ must come from the same session.** Two independent determinations were made
within the campaign, an hour apart and by different methods. The $\pm 9$ V bias survey, using the
quasi-static plateau at the working position, gave $V_{\mathrm{cpd}} = +1.057$ V. Extrapolating the
two domains' imaging amplitudes to their crossing, using three bias points, gave **+1.024 V** at
position A and **+1.031 V** at the free end — positions whose enhancement differs by a factor of
seven, on entirely independent amplitude scales. The three agree within **30 mV**, and the two
positions agree within **7 mV** (Figure 3a-c).

By contrast, the companion campaign imported a $V_{\mathrm{cpd}}$ measured four hours earlier,
across a load excursion that altered the tip. That value was **560 mV** too low, and at it the two
domains did not equalise: the amplitude ratio moved only from 0.604 to 0.808 (Figure S1).

**Equalisation is achieved.** At the measured null the domain amplitude ratio reaches **1.011** at
position A and **1.012** at the free end (Table 4). Across the three bias points at position A the
ratio marches 0.669 → 0.821 → 1.011, and the 0 V frame repeated at the end of the 40-minute series
differs from the first by under 2 %, so the progression is bias and not drift. Domain phase
separation is **177.9–180.0°** in every frame at every bias (Figure 3d, Figure 4, Table 4).

**$d_{33}$ by independent routes.** Quasi-static spectroscopy along the lever gives 7.83–9.67 pm/V
with a median of **8.66 pm/V** and a spread of 1.24× over eight positions whose enhancement spans
15× (Figure 1e) — that is, the model-free channel returns the same material property regardless of
where the laser sits. Quasi-static imaging gives 6.50 and 11.31 pm/V for the two domains at
position A, and 7.23 and 8.75 pm/V at the free end. Across the two campaigns, separated by a
probe-altering load excursion, the quasi-static imaging values reproduce to within **6 %**.

**Drive linearity, and what the low-drive rise actually is.** Seventeen drive amplitudes from 2 mV
to 2 V (Figure 5a) show a flat $d_{33}$ above roughly 50 mV and an apparent rise below it. The rise
is not nonlinearity. Magnitude detection of a complex signal in additive noise returns
$\sqrt{(kV_{ac})^2 + n_0^2}$, which biases a weak amplitude high and never low. Fitting that
two-parameter form to each domain reproduces the entire three-decade range to within **2–6 %**
(Table 5), with a detection floor $n_0 = 570$ µV for both domains and response slopes agreeing
between campaigns to 7 %. Signal equals floor at 7–10 mV drive.

**When resonance helps, and when it does not.** At $V_{ac} = 1$ V the resonance channel gives *no*
SNR advantage: image SNR at position A is 2.7 on resonance against 9.5 quasi-statically, and the
input-referred noise is 1.18 pm/V against 0.51 pm/V. The pixel noise scales with the signal, so the
mechanical gain buys nothing. At 30 mV the situation inverts — the quasi-static frame falls below
the floor entirely (contrast 0.05 pm against 0.24 pm noise) while the resonance channel still
resolves the domains and returns $d_{33}$ values that converge on the same 6–7 pm/V at the null.
The case for contact resonance is therefore not lower fractional noise on a strong signal but a
usable signal where quasi-static PFM has none.

A corollary worth stating: a *higher* $Q$ made the images *noisier*. Between the two campaigns
$Q_1$ rose from 179 to 222 and the input-referred on-resonance noise at the same position rose from
0.49 to 1.18 pm/V, while the quasi-static noise was unchanged. A narrower resonance read at a fixed
drive frequency converts small excursions of the resonance into amplitude noise.

### 3.4 A blind Euler–Bernoulli fit converts an on-resonance amplitude to $d_{33}$

Sections 3.1 and 3.3 establish the enhancement $E(x)$ as a *measured* quantity and use the
quasi-static channel for every absolute number. The remaining question is whether the transfer
function can be supplied by a model instead, so that a single on-resonance amplitude — the only
thing a fast PFM image actually delivers — becomes a material property without a wideband sweep at
every pixel.

**The fit.** A two-pathway Euler–Bernoulli beam with an interior contact was fitted to the Run 2
lever with five free parameters: the contact stiffness ratio $k^*/k_{\mathrm{lever}}$, the tip-cone
stiffness ratio, the tip setback from the free end, the intrinsic modal damping $\zeta$, and the
tip height. The residual was given **only** the three contact-resonance frequencies from the
pre-flight tune (295.02, 912.95, 1831.27 kHz), the *shape* of $1/\mathrm{InvOLS}(x)$ with its
overall gain projected out, and the CR1 quality factor. Every amplitude, and the measured $E(x)$,
was withheld from the objective and used afterwards only to score the result. The free-end point
was excluded from the static-shape residual, as in the companion analysis.

**Fit quality.** The converged model reproduces the three resonances to $-1.8$, $+0.7$ and
$-0.1$ %, $Q$ to 0.1 % (235 against 235), and the static shape to 2.6 % rms, with
$k^*/k_{\mathrm{lever}} = 7.7 \times 10^2$, tip-cone ratio 1.8, tip setback 4.5 µm and
$\zeta = 2.6\times10^{-3}$.

**The blind prediction.** The predicted enhancement matches the withheld measurement to
**0.81–1.05×** at all seven interior positions, with 9 % scatter, over a range in which $E$ itself
varies by 15× (Figure 6a). A prediction of $E$ good to ten percent is obtained from data that
contains no amplitude information at all.

**$d_{33}$ from the resonance channel.** Dividing the CR1 amplitude by the predicted enhancement
and calibrating with the **free-end InvOLS alone** gives $d_{33} = 8.68$ pm/V with 4.3 % scatter
over the seven interior positions, against **8.81 pm/V** (7.6 % scatter) from the model-free
quasi-static channel with a per-position InvOLS — agreement to **1.5 %** (Figure 6b,c). Using the
per-position InvOLS with the model enhancement gives 8.31 pm/V and 3.5 % scatter.

**The $E = Q$ shortcut fails, and fails in a diagnosable way.** Dividing the same CR1 amplitudes by
$Q$ returns a median of 11.6 pm/V — 32 % high — but the more serious failure is the spatial one:
the result varies by **6.4×** along the lever, from 21.4 pm/V near the clamp to 1.2 pm/V at the
free end (Figure 6b). Because $Q$ is flat along the beam while $E$ is not (§3.1), the $E=Q$ result
simply traces the CR1 mode shape. A single-position measurement would report any value in that
range with no internal indication that anything is wrong.

**On the lever length.** Run 2's own static-shape fit drives the beam length to the lower bound
imposed by the measured span ($L \ge x_{\max} - x_0 = 207.5$ µm), because the laser positions
available in this run did not extend far enough past the clamp to constrain it. The fit above
therefore fixes $L$ at 225.9 µm, the value obtained from the unconstrained static-shape fit of the
companion run on the *same physical lever*; the Run 2 static shape is fitted almost as well by that
length (2.6 % rms against 2.2 %). Repeating the fit with $L$ at its lower bound degrades the blind
enhancement prediction to 0.80–1.84× and the resonance-channel $d_{33}$ to 8.93 pm/V with 28 %
scatter (Figure 6a, dotted). $L$ is thus the one parameter the present dataset cannot supply for
itself, and it is a prior from a static measurement, not from any amplitude — the prediction of
$E(x)$ remains blind with respect to every amplitude in either campaign. A run that samples the
laser position over a wider span, or that deliberately includes positions near the clamp, would
close this gap.

---

## 4. Discussion

*[To be completed.]*

## 5. Conclusions

*[To be completed.]*

---

## Data and code availability

*[To be completed.]*

## Appendix A — units convention

*[To be completed.]*

## Appendix B — corrections to the earlier analysis

*[To be completed.]*
