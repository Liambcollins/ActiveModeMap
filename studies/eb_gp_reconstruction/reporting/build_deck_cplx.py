#!/usr/bin/env python3
"""Standalone ORNL-branded deck: reconstructed complex spectra, bias-dependence
recovery, uncertainty.  Six arms: truth, GP only, EB, EB+GP, FEM, FEM+GP.

Built WITHOUT `ORNL template 1.pptm` -- that file is not inside either connected
folder, so the theme colours are set literally here from the values recorded in
`claude/deck-physics-informed-al-2026-08-11.md` (clrScheme "ORNL Color TEST").
To put these slides on the real template instead, re-run the study's
`reporting/build_deck.py` pattern with `Presentation('ornl.pptx')`.
"""
import os, sys
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

import os as _os
_ROOT = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_ROOT, 'config.py')):
    _ROOT = _os.path.dirname(_ROOT)
sys.path[:0] = [_ROOT, _os.path.join(_ROOT, 'src'),
                _os.path.join(_ROOT, 'figures')]
from config import EB_REPO, EB_GEOMETRY, FEM_LADDER, OUT, FIG

GREEN = RGBColor(0x00, 0x66, 0x2C)      # dk2, ORNL signature green
BLUE = RGBColor(0x00, 0x6B, 0xA6)       # accent2
TEAL = RGBColor(0x00, 0x45, 0x4D)       # accent1
ORANGE = RGBColor(0xA3, 0x5C, 0x00)     # accent5, darkened for contrast
MAGENTA = RGBColor(0xB5, 0x00, 0x93)    # accent6
INK = RGBColor(0x37, 0x3A, 0x36)        # dk1
MUT = RGBColor(0x6E, 0x71, 0x6D)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LT = RGBColor(0xF2, 0xF3, 0xF2)
FONT = 'Aptos'

SW, SH = Inches(13.333), Inches(7.5)
M = Inches(0.42)
FIGDIR = str(FIG)
OUTF = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                     'ActiveModeMap_spectra_bias_uncertainty.pptx')

prs = Presentation()
prs.slide_width, prs.slide_height = SW, SH
BLANK = prs.slide_layouts[6]


def slide():
    return prs.slides.add_slide(BLANK)


def box(s, l, t, w, h):
    tb = s.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    return tf


def para(tf, text, size=14, color=INK, bold=False, first=False, space=6,
         italic=False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    if not first:
        p.space_before = Pt(space)
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.color.rgb = color
    r.font.bold = bold
    r.font.italic = italic
    r.font.name = FONT
    return p


def rule(s, t, w=Inches(2.2), color=GREEN, h=Pt(3)):
    from pptx.enum.shapes import MSO_SHAPE
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, M, t, w, h)
    sh.fill.solid(); sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def header(s, kicker, title, sub=None):
    tf = box(s, M, Inches(0.30), SW - 2 * M, Inches(0.34))
    para(tf, kicker.upper(), 11.5, GREEN, bold=True, first=True)
    tf2 = box(s, M, Inches(0.60), SW - 2 * M, Inches(0.5))
    para(tf2, title, 23, INK, bold=True, first=True)
    y = Inches(1.04)
    if sub:
        tf3 = box(s, M, y, SW - 2 * M, Inches(0.46))
        para(tf3, sub, 12.5, MUT, first=True)
        y = Inches(1.44)
    return y


def figure(s, name, top, width=Inches(12.49)):
    from PIL import Image
    p = os.path.join(FIGDIR, name)
    w, h = Image.open(p).size
    height = Emu(int(width * h / w))
    s.shapes.add_picture(p, M, top, width=width, height=height)
    return top + height


def figure_fit(s, name, top, bottom=Inches(0.20), maxw=Inches(12.49)):
    """Largest picture that fits both the width margin and the space left."""
    from PIL import Image
    p = os.path.join(FIGDIR, name)
    w, h = Image.open(p).size
    avail_h = SH - top - bottom
    width = min(int(maxw), int(avail_h * w / h))
    height = int(width * h / w)
    left = int((SW - width) / 2)
    s.shapes.add_picture(p, Emu(left), top, width=Emu(width), height=Emu(height))
    return top + Emu(height)


def bullets(s, items, top, left=M, width=None, size=13.5, gap=8):
    width = width or (SW - 2 * M)
    tf = box(s, left, top, width, SH - top - Inches(0.2))
    for i, (lead, rest) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if i:
            p.space_before = Pt(gap)
        r = p.add_run(); r.text = lead + '  '
        r.font.size = Pt(size); r.font.bold = True
        r.font.color.rgb = GREEN; r.font.name = FONT
        r2 = p.add_run(); r2.text = rest
        r2.font.size = Pt(size); r2.font.color.rgb = INK; r2.font.name = FONT
    return tf


def logo(s):
    s.shapes.add_picture(str(_ROOT) + '/figures/ornl_logo.png',
                         SW - M - Inches(1.35), SH - Inches(0.62),
                         width=Inches(1.35))


def pageno(s, n):
    tf = box(s, SW - M - Inches(0.6), SH - Inches(0.60), Inches(0.5),
             Inches(0.3))
    p = para(tf, str(n), 10.5, MUT, first=True)
    p.alignment = PP_ALIGN.RIGHT


# =========================================================== 1. title =======
s = slide()
from pptx.enum.shapes import MSO_SHAPE
bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
                         Inches(0.30), SH)
bar.fill.solid(); bar.fill.fore_color.rgb = GREEN
bar.line.fill.background(); bar.shadow.inherit = False

tf = box(s, Inches(1.0), Inches(2.05), Inches(10.6), Inches(0.4))
para(tf, 'ACTIVE MODE MAP  ·  SCM-PIT  ·  MULTI75G', 13, GREEN, bold=True,
     first=True)
tf = box(s, Inches(1.0), Inches(2.52), Inches(11.0), Inches(1.9))
para(tf, 'Reconstructing the full complex spectrum', 40, INK, bold=True,
     first=True)
para(tf, 'amplitude, phase, bias dependence, and honest error bars',
     24, TEAL, space=10)
tf = box(s, Inches(1.0), Inches(4.60), Inches(10.4), Inches(1.3))
para(tf, 'Both forward models (two-segment Euler–Bernoulli and the '
         'geometry-faithful FEM), each bare and with a discrepancy GP, and a '
         'model-light GP baseline — reconstructing the mode-A contact '
         'resonance from 8 of 101 measured positions and now carrying the '
         'phase channel, not only log-amplitude.', 15, MUT, first=True)
tf = box(s, Inches(1.0), Inches(6.35), Inches(8.0), Inches(0.4))
para(tf, 'Dense_Grid_A · 101 positions × 412 frequencies · 14 August 2026',
     12.5, MUT, first=True)
logo(s)

# =========================================================== 2. amplitude ===
s = slide()
y = header(s, 'reconstructed spectrum  ·  1 of 2',
           'Amplitude: 8 of 101 positions reproduce the map to 2.9 %')
bullets(s, [
    ('Bare physics 6.3–6.4 %; with the GP either model reaches 2.9 %, the same '
     'as the GP alone.',
     'Both libraries were rebuilt here and reproduce the published '
     'log-amplitude arms exactly (EB 6.53 / 2.41, FEM 6.31 / 1.94, GP 3.41 %).'),
    ('It earns its place at the node.',
     'EB + GP and FEM + GP both land 0.15 µm from the true 224.09 µm; bare EB '
     'is 1.8 µm out, bare FEM 1.0 µm, and the GP alone never finds a crossing '
     'in span — 3 µm out at every budget from 4 to 30 positions.'),
    ('Do NOT read the fitted k₁ off this slide.',
     'EB fits 374 N/m here and FEM 1254, but neither is identified by band A: '
     'the mode-2 ratio puts EB at 1044 N/m, essentially Hertz. See slide 5.'),
], Inches(1.06), size=12.0, gap=6)
figure_fit(s, 'd1_amp.png', Inches(2.08))
pageno(s, 2); logo(s)

# =========================================================== 3. phase =======
s = slide()
y = header(s, 'reconstructed spectrum  ·  2 of 2',
           'Phase: both forward models get it wrong the same way')
bullets(s, [
    ('EB alone 42.4°, FEM alone 41.4° — 142 % and 144 % on the complex field, '
     'worse than predicting zero.',
     'One complex gain absorbs the detector’s reference phase, so what is left '
     'is model error. Two unrelated models, independently fitted and operating '
     'at k₁ values 3× apart, fail by the same 41–42° and take the '
     'antiresonance branch the same wrong way.'),
    ('Not a mis-set resonance either.',
     'Refitting f_res and damping against the complex residual only makes it '
     'worse — the discrepancy is structural.'),
], Inches(1.06), size=12.0, gap=6)
figure_fit(s, 'd2_phase.png', Inches(2.08))
pageno(s, 3); logo(s)

# =========================================================== 4. TF vs raw ===
s = slide()
y = header(s, 'reconstruction vs raw data  ·  1 of 2',
           'Transfer functions at withheld positions')
bullets(s, [
    ('Four positions none of the arms was shown, across 265–1000 kHz so both '
     'contact modes are in frame.',
     'Median |error| over all 93 withheld positions: 0.29 dB for EB two-band + '
     'GP and 0.30 dB for the GP alone, against 3.33 dB for the bare two-band '
     'physics — which misplaces the antiresonance and inverts the phase branch, '
     'visible as the pale blue line leaving the data between the two modes. '
     'Grey bands mark raw |Z| below 3× the noise floor, where the measured '
     'phase is not reliable and nothing should match it.'),
], Inches(1.06), size=12.0, gap=6)
figure_fit(s, 'd8_tf.png', Inches(1.78))
pageno(s, 4); logo(s)

# =========================================================== 5. shapes ======
s = slide()
y = header(s, 'reconstruction vs raw data  ·  2 of 2',
           'Resonance mode shapes, both flexural modes')
bullets(s, [
    ('Mode 1 collapses two decades into the last 3 µm; mode 2 arches instead, '
     'peaking at x = 174 µm with no node anywhere in the measured span.',
     'Peak-amplitude error over the withheld positions: mode 1 3.8 % and mode 2 '
     '6.4 % for EB two-band + GP, against 7.9 % / 6.2 % for the GP alone and '
     '67 % / 23 % for the bare physics. The right-hand panel is the '
     'consequence — mode 1 reverses 180° at its node and mode 2 never does, '
     'because it has none here.'),
], Inches(1.06), size=12.0, gap=6)
figure_fit(s, 'd9_shapes.png', Inches(2.02))
pageno(s, 5); logo(s)

# =========================================================== 4. dual peak ===
s = slide()
y = header(s, 'fitting both peaks',
           'k₁ is not identifiable from band A — the mode ratio identifies it')
bullets(s, [
    ('EB reaches the measured f₂/f₁ = 3.079 at k₁ = 1044 N/m — essentially the '
     'Hertz value.',
     'Over k₁ = 328 → 1291 N/m, f₁ moves 3.8 % while the ratio moves 28 %: the '
     'ratio is 7.3× the more sensitive channel. Band A carries the mode shape, '
     'band B carries the stiffness. FEM is the model that cannot get there — '
     'isotropic saturates at 2.942 over the whole ladder, frictionless at 2.66.'),
    ('Fit both windows with a gain per band and k₁ locks to 989 ± 15 N/m at '
     'every n from 3 to 30, with mode 2 inside one 437 Hz bin.',
     'Band A alone sits at 318–392 and drifts DOWN as n grows; one global gain '
     'swings 760–1580. The per-band gain ratio is −9.51 ± 1.24 dB, i.e. mode 2 '
     'couples 2.99× weaker — against 3.11× measured independently from the '
     'domain-flip data. The gain is measuring the drive asymmetry, not hiding '
     'it.'),
], Inches(1.06), size=11.5, gap=6)
figure_fit(s, 'd7_dualpeak.png', Inches(2.38))
pageno(s, 6); logo(s)

# =========================================================== 4. bias I ======
s = slide()
y = header(s, 'bias dependence  ·  1 of 2',
           'The bias response is large, and it is electrostatic',
           '7 biases (−6 … +6 V) × 7 positions, 1000 nN, 1 V AC electrical '
           'drive on the tip. Each spectrum is read at its OWN resonance peak.')
figure(s, 'd3_bias1.png', y, width=Inches(12.49))
bullets(s, [
    ('A 100× cancellation in the middle of the sweep, not at an end.',
     'Monotonic drift cannot do that, and the bias order was identical at every '
     'position — so the V-shape is the evidence that this is bias and not '
     'time.'),
    ('Z(V) is a straight line through the origin to 3.5–4.4 %.',
     'That is the signature of one force source scaling with (V − V_cpd). '
     'V_cpd = −1.86 ± 0.08 V, flat along the lever — a tip–sample property, '
     'reproduced at five independent engages.'),
    ('Q = 206 ± 4 at V = 0, and f_res moves less than 1 kHz across ±6 V.',
     'Still: randomise the condition order next time — this effect was big '
     'enough to be unambiguous; a 10 % effect would not be.'),
], Inches(5.62), size=12.5, gap=7, width=Inches(11.4))
pageno(s, 7); logo(s)

# =========================================================== 5. bias II =====
s = slide()
y = header(s, 'bias dependence  ·  2 of 2',
           'The mode map predicts the bias sensitivity',
           'Splitting Z(V) into the part that cancels (the electrostatic slope '
           'b) and the part that cannot (|P⊥|, the perpendicular distance from '
           'the origin to the fitted line).')
figure(s, 'd4_bias2.png', y, width=Inches(12.49))
bullets(s, [
    ('|∂Z/∂V| along the lever IS the mode-A shape.',
     'The 8-position FEM + GP reconstruction predicts the measured '
     'electrostatic slope to ≤ 4 % over 132–196 µm with a single free scale — '
     'the sparse map is a calibration for the bias experiment.'),
    ('The residual is instrumental, not tip–sample.',
     '|b| varies 43× along the beam; |P⊥| only 7×, and corr(log|P⊥|, log|b|) '
     '= −0.38. A piezoresponse is driven at the tip like the electrostatic '
     'force, so it must be detected through the same mode shape — it would '
     'have collapsed at the node too.'),
    ('No measurable piezoresponse: 0.077 ± 0.082 V equivalent DC offset, 1.3 % '
     'of a 6 V drive.',
     'The two positions at 224/225 µm look significant (P/SE ≈ 9–10) only '
     'because the detection has nulled |b| there.'),
], Inches(5.62), size=12.5, gap=7)
pageno(s, 8); logo(s)

# =========================================================== 6. uncertainty =
s = slide()
y = header(s, 'uncertainty from the data',
           'How much of the map do we actually believe?',
           'σ is the predictive sd of one real component of Z. Noise floor, '
           'gain, GP length scale and per-frequency variance all come from the '
           'revealed block; the withheld positions only ever score.')
figure(s, 'd5_uncert.png', y, width=Inches(12.49))
bullets(s, [
    ('σ ≈ |Z| exactly on the features worth measuring.',
     'The antiresonance trough and the node are where the signal is smallest '
     'and the reconstruction least constrained. The discrepancy GP carries ~4× '
     'more of σ there than the parameter fit does.'),
    ('Both physics arms start cautious and end over-confident.',
     'z runs 0.59 → 0.99 → 1.63 for FEM + GP and 0.48 → 0.77 → 1.42 for '
     'EB + GP over n = 5 → 12 → 30, while GP only sits at 0.4–0.6 throughout. '
     'At n = 4 no discrepancy GP is fitted and σ is wrong by 16× / 47×.'),
], Inches(6.00), size=12.0, gap=7, width=Inches(10.7))
pageno(s, 9); logo(s)

# =========================================================== 7. wide band ===
s = slide()
y = header(s, 'the full measured band',
           'Across 25 kHz – 1.775 MHz the arms do not share a domain')
bullets(s, [
    ('The physics arms cover 25 % of the measured band; the two-band EB library '
     '49 %; the GP 100 %.',
     'A band-A library spans u = f/f_res = 0.55–2.05, i.e. 161–600 kHz. Mode 2 '
     'sits at u = 3.08 and is simply outside it — and predict_cplx clips, so '
     'asking for it returns the library edge: plausible-looking and wrong. '
     'Every wide-band score here is masked to the arm’s own coverage.'),
], Inches(1.06), size=12.0, gap=6)
figure_fit(s, 'd6_wideband.png', Inches(1.72))
pageno(s, 10); logo(s)

# =========================================================== 8. verdict =====
s = slide()
y = header(s, 'the full measured band  ·  verdict',
           'Physics wins the tail; the GP wins the coverage')
ROWS = [('below mode 1', '0.17', '—', '—', '—'),
        ('mode 1 ridge', '0.12', '0.10', '0.10', '0.11'),
        ('mode 1 antiresonance', '0.31', '0.30', '0.28', '0.32'),
        ('valley, modes 1–2', '0.40', '0.53', '0.44', '0.52'),
        ('mode 2 flank', '0.25', '—', '—', '0.22'),
        ('mode 2 ridge', '0.30', '—', '—', '0.18'),
        ('above mode 2', '0.37', '—', '—', '—'),
        (None, None, None, None, None),
        ('signal bins within 3 dB', '97.6 %', '99.5 %', '99.5 %', '99.1 %'),
        ('coverage of the band', '100 %', '25 %', '25 %', '49 %')]
COLS = ['GP only', 'EB + GP', 'FEM + GP', 'EB + GP (2-band)']
CC = [MAGENTA, BLUE, GREEN, TEAL]
x0, w0, cw = M, Inches(2.55), Inches(1.62)
tf = box(s, x0, Inches(1.60), w0, Inches(0.3))
para(tf, 'median |error|  (dB), n = 8', 11, MUT, first=True, italic=True)
for j, (cn, cc) in enumerate(zip(COLS, CC)):
    tb = box(s, x0 + w0 + j * cw, Inches(1.54), cw, Inches(0.62))
    p = para(tb, cn, 11, cc, bold=True, first=True)
    p.alignment = PP_ALIGN.CENTER
yy = Inches(2.22)
for r in ROWS:
    if r[0] is None:
        yy = yy + Inches(0.14)
        continue
    strong = r[0].startswith('signal') or r[0].startswith('coverage')
    tb = box(s, x0, yy, w0, Inches(0.30))
    para(tb, r[0], 11.5, INK if strong else INK, bold=strong, first=True)
    for j in range(4):
        tb2 = box(s, x0 + w0 + j * cw, yy, cw, Inches(0.30))
        p = para(tb2, r[j + 1], 11.5, CC[j] if strong else INK, bold=strong,
                 first=True)
        p.alignment = PP_ALIGN.CENTER
    yy = yy + Inches(0.315)
bullets(s, [
    ('On the medians the arms are within a hair of each other; on the tail they '
     'are not.',
     '99.1–99.5 % of signal bins land within 3 dB for every physics arm against '
     '96.6–98.0 % for the GP, at every budget from 5 to 30 positions. The '
     'physics prior does not make the typical bin better — it removes the bad '
     'ones.'),
    ('GP alone from n = 5 is already close to its own ceiling.',
     'Median on signal bins 0.20 → 0.12 dB and mode 2 0.37 → 0.16 dB from '
     'n = 5 to 30, but the 3 dB fraction barely moves: 97.4 → 97.8 %. Past '
     'n ≈ 12 more positions buy the GP almost nothing, and it still never '
     'locates the node.'),
    ('The two-band EB arm reads both modes — and it pins k₁ to 984 ± 3 N/m at '
     'every n from 5 to 30.',
     'A 0.6 % spread, against 319–394 N/m from band A alone, with mode 2 to '
     '0.18 dB. It needed a 48-point coarse grid: at 16 points the fit settles '
     'in a k₁ ≈ 500 basin scoring 0.61 where k₁ = 1008 scores 0.35.'),
], Inches(5.72), size=11.5, gap=5, width=Inches(11.0))
pageno(s, 11); logo(s)

# =========================================================== 7. status ======
s = slide()
y = header(s, 'status', 'What is new, what is missing, what is next')
COLW = Inches(3.92)
cols = [
    ('New in this pass', GREEN, [
        'All three libraries re-stitched to KEEP the complex channel — the '
        'published builds discarded the phase (FEM at np.abs(), EB by storing '
        'log|z|). Each reproduces its published log-amplitude arm exactly.',
        'Six arms reconstruct amplitude and phase with a per-component '
        'predictive σ, split into a parameter term and a discrepancy term.',
        'Scoring extended from band A to the whole measured 25 kHz – 1.775 MHz, '
        'with each arm masked to its own library coverage.',
        'The bias series is re-analysed on its own resonance peak and tied to '
        'the sparse mode map for the first time.']),
    ('Three corrections', ORANGE, [
        'Reading a physical contact stiffness off a band-A fit. k₁ is not '
        'identifiable there in either model: EB fits 374 N/m but reaches the '
        'measured mode ratio at 1044. The older “only FEM recovers a physical '
        'k₁” claim was already overturned on 2026-08-12; slide 2 now says so.',
        'An earlier draft said the EB library could not be rebuilt because '
        'eb_models.py and geometry.py were missing. They are not — they live in '
        'EB-Solver-CResonance/fem_eb_comparison/py, not in that repo’s src/, '
        'which holds only the eb_cr_afm package.',
        'twoband.fit2g’s 16-point coarse grid in log k₁ is too sparse: at '
        'n = 5–8 it settles at k₁ ≈ 500 N/m, where k₁ = 1008 scores 0.35 '
        'against 0.61. With 48 points k₁ locks to 984 ± 3 N/m at every n. The '
        'published two-band stability at small n was the grid straddling the '
        'right basin, not robustness.']),
    ('Open', TEAL, [
        'Nothing physical exists above ~1 MHz: even the extended library stops '
        'at u = 3.45. Modes above 2 need a further extension, or the GP.',
        'The shared 41° phase error is the live physics question — two '
        'unrelated forward models failing identically points at the drive or '
        'detection model, not the beam.',
        'Re-derive the GP variance so σ stays calibrated past n = 12; '
        'randomise the condition order in the next bias series; and the GP’s '
        '2.2 % beyond-3 dB tail is still unexplained.']),
]
for i, (head, c, items) in enumerate(cols):
    left = M + i * Inches(4.16)
    rule(s, Inches(1.28), Inches(1.6), c)
    tf = box(s, left, Inches(1.48), COLW, Inches(0.4))
    para(tf, head, 16, c, bold=True, first=True)
    tf2 = box(s, left, Inches(1.98), COLW, Inches(4.5))
    for j, it in enumerate(items):
        para(tf2, '— ' + it, 11.5, INK, first=(j == 0), space=9)
# three separate rules, one per column
for i, (head, c, _) in enumerate(cols):
    from pptx.enum.shapes import MSO_SHAPE
    sh = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, M + i * Inches(4.16),
                            Inches(1.28), Inches(1.6), Pt(3))
    sh.fill.solid(); sh.fill.fore_color.rgb = c
    sh.line.fill.background(); sh.shadow.inherit = False
tf = box(s, M, Inches(6.62), Inches(11.2), Inches(0.6))
para(tf, 'Everything here is leakage-safe: each reconstructor sees only the '
         'revealed indices, their spectra and the frequency axis. Scripts: '
         'libraries/build_femlib_cplx.py, buildlib_cplx.py, buildlib_x_cplx.py, '
         'src/cplxrec.py, src/bias.py, src/run_wideband.py, figures/d_*.py.',
     11.0, MUT, first=True, italic=True)
pageno(s, 12); logo(s)

prs.save(OUTF)
print('wrote', OUTF, os.path.getsize(OUTF), 'bytes,', len(prs.slides.__iter__.__self__._sldIdLst), 'slides')
