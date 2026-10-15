"""Builds notebooks/08_smm_calibration.ipynb. Run from the repo root: python scripts/_build_smm_notebook.py"""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Sparse mode mapping (SMM) — calibrated contact-resonance PFM in one sitting

This notebook runs the **map route** of the fast-mode-mapping paper on the Cypher:

1. find the reachable span of the lever and the contact resonance (**pre-flight**),
2. record spectra at a few laser positions on two oppositely poled domains at several DC biases
   (**two-domain sparse capture**, live view),
3. separate piezoresponse from electrostatics, correct the domain potential, and reconstruct the
   resonance enhancement **E_map(x) = |P_CR1| / |P_QS|** along the lever (**analysis**),
4. choose a laser position where E is large and insensitive to drift (**imaging stop**),
5. move there, tune, and take quasi-static and CR1 frames that are converted to **d_eff in pm/V**.

**How to use it:** edit the *CONFIG* cell, then run the cells top to bottom. Each stage writes its result
into the campaign folder and can be re-run on its own. Every acquisition stage ends with the bias at 0 V
and the tip withdrawn, whether it finishes, fails or is interrupted.

**Before you start, in Igor (these stay manual):**
- approach, engage, and check that the contact resonance is visible in a tune;
- set the tune panel to a wide fixed window (2 MHz sweep width, 10 s) with no auto-recentre;
- take a PFM image of the two domains and **mark one spot on each domain** (spots 1 and 2 in the Force panel), leave that image window open;
- park the detection laser at the **free end** of the lever;
- know the spring constant (thermal) and the lever length.

**To stop:** `Kernel ▸ Interrupt` ends the current stage cleanly, or create an empty file called `STOP` in the
campaign folder to finish after the current position. **To rehearse without hardware:** set `VIRTUAL = True`
in CONFIG and point `REPLAY_CHECKPOINT` at an archived two-domain survey.
""")

code(r"""
# ---- 0. CONFIG  (edit this cell; nothing below hard-codes a number) ---------------------------------
import os, sys
REPO_DIR = os.path.abspath(os.path.join(os.getcwd(), '..'))        # the ActiveModeMap checkout
sys.path.insert(0, REPO_DIR)
from activemodemap.smm import SMMConfig, stages

VIRTUAL = False                                  # True = no hardware: replay an archived survey (safe rehearsal)
REPLAY_CHECKPOINT = r'D:\ActiveModeMap\DomainsB_SCMPIT_R2\10_bias_survey\domains_bias_checkpoint_500nN_scmpit.npz'
REPLAY_INVOLS     = r'D:\ActiveModeMap\DomainsB_SCMPIT_R2\analysis\s3_transfer.json'   # per-position InvOLS of that survey (a preflight_result.json also works, less exactly)

cfg = SMMConfig()

# -- where the data go -------------------------------------------------------------------------------
cfg.campaign.data_dir      = r'D:\User Data\Liam\ActiveModeMap\SMM_2026-10-15'   # new folder per campaign
cfg.campaign.base_filename = 'SMM'              # Igor base name for tunes and images
cfg.campaign.tag           = 'PPLN DomainsB, SCM-PIT, first SMM run'

# -- the probe -----------------------------------------------------------------------------------------
cfg.probe.name        = 'SCM-PIT'
cfg.probe.k_N_per_m   = 1.70                    # thermal spring constant
cfg.probe.length_um   = 232.0                   # stage x of the FREE END when the laser starts there (x = 0 is the clamp)
cfg.probe.hw_sign_toward_free_end = +1          # DoLDMove(+x) moves toward the tip on this Cypher
cfg.probe.cr_bands_Hz = {'cr1': (230e3, 340e3), 'cr2': (780e3, 1000e3), 'cr3': (1550e3, 1850e3)}   # priors; pre-flight refines
cfg.probe.qs_band_Hz  = (15e3, 45e3)            # quasi-static reference band (well below CR1, above 1/f)

# -- pre-flight: how far toward the base the laser can go --------------------------------------------
cfg.preflight.run        = True                 # False: skip the walk and use x_span_um
cfg.preflight.step_um    = 10.0
cfg.preflight.x_floor_um = 30.0                 # never closer to the clamp than this
cfg.preflight.min_snr_dB = 8.0                  # stop when CR1/CR2 drop below this above the floor
cfg.preflight.invols_bounds_m_per_V = (1e-8, 5e-5)
cfg.preflight.x_span_um  = None                 # e.g. (72.0, 232.0) when run = False

# -- the two-domain sparse capture ------------------------------------------------------------------
cfg.capture.load_nN     = 500.0
cfg.capture.drive_V     = 1.0                   # V_ac for every tune and frame
cfg.capture.bias_V      = [-9, -6, -3, 0, 3, 6, 9]
cfg.capture.bias_order  = 'interleaved'         # alternate signs so drift cannot look like a bias slope
cfg.capture.spots       = (1, 2)                # marked spots: domain 1, domain 2
cfg.capture.n_positions = 8                     # equispaced over the reachable span (8 is the paper's choice)
cfg.capture.positions_um = None                 # or an explicit list of stage positions, e.g. [232, 200, 170, 140, 110, 80]
cfg.capture.order       = 'coarse_to_fine'      # ends, middle, quarters... so an early stop still covers the lever
cfg.capture.stop_when_stable = False            # True: finish early once E(x) stops changing
cfg.capture.stable_rel  = 0.05                  #   ... by less than 5 % (median over the lever)
cfg.capture.stable_over = 2                     #   ... for two positions in a row
cfg.capture.min_positions = 6                   #   ... but never before this many

# -- analysis ---------------------------------------------------------------------------------------
cfg.analysis.cr_band_frac = (0.80, 1.25)        # CR1 analysis band relative to the measured CR1
cfg.analysis.track_peaks  = True                # read each bias condition at its own CR1 peak (paper convention)
cfg.analysis.fit_delta    = True                # domain-potential correction P = P_raw - delta*b, delta fitted in QS
cfg.analysis.rank         = 4                   # Chebyshev rank of the along-lever interpolation
cfg.analysis.grid_step_um = 1.0

# -- imaging: QS reference at the electrostatic blind spot, BE at interior stops -------------------
cfg.imaging.qs_x_um       = None                # None = blind spot from the analysis (b_QS/P_QS smallest)
cfg.imaging.qs_bias_plan  = ['0V']              # add 'Vcpd' for a frame at the contact potential
cfg.imaging.f_qs_Hz       = 20e3                # quasi-static drive
cfg.imaging.image_base    = 'QImg'
cfg.imaging.frame         = None                # None = the frame currently set in the MasterPanel; or a dict:
# cfg.imaging.frame = dict(ScanSize=6e-6, XOffset=7.93e-7, YOffset=-1.16e-6, ScanAngle=90.0, ScanPoints=256, ScanLines=256, ScanRate=1.0)
cfg.imaging.be_positions_um = None              # stage positions you choose, e.g. [150.0, 110.0]; None = recommended
cfg.imaging.n_be_positions  = 2                 # how many to recommend ...
cfg.imaging.be_min_spacing_um = 20.0            # ... at least this far apart
cfg.imaging.tol_pct       = 5.0                 # E may change by at most this over +/- drift_um
cfg.imaging.drift_um      = 2.0                 # laser drift budget
cfg.imaging.min_E_frac    = 0.5                 # and E >= half the maximum on the validated span
cfg.imaging.loo_tol       = 0.15                # only between positions whose leave-one-out E agrees to 15 %
cfg.imaging.be_bias_plan  = ['0V', 'Vcpd', '0V_repeat']
cfg.imaging.be_px         = 128
cfg.imaging.be_repeats    = 1
cfg.imaging.be_width_x_fwhm = 10.0              # BE band = 10 x the CR1 FWHM at the stop, clipped to be_width_Hz
cfg.imaging.be_width_Hz   = (20e3, 80e3)
cfg.imaging.be_acquire    = 'smm_be_adapter:be_frame'   # your BE call (see activemodemap/smm/be_adapter_template.py)
cfg.imaging.tune_after_each_stop = True

# -- hard limits (the stages refuse to run outside these) --------------------------------------------
cfg.limits.max_abs_bias_V = 10.0
cfg.limits.max_load_nN    = 1500.0
cfg.limits.max_drive_V    = 3.0

if VIRTUAL:
    cfg.campaign.virtual = True
    cfg.campaign.replay_checkpoint = REPLAY_CHECKPOINT
    cfg.campaign.replay_invols_json = REPLAY_INVOLS
    cfg.campaign.data_dir = cfg.campaign.data_dir.rstrip('\\/') + '_VIRTUAL'

cfg.check()
print(cfg.describe())
""")

md(r"""
## 1. Instrument
Connects to Igor and builds the instrument with the laser **at the free end**, or adopts an `inst` that already
exists in this kernel (never rebuild one blind: the laser position is tracked in software). Read the printed
MasterPanel values: spring constant, InvOLS, setpoint.
""")
code(r"""
inst = stages.connect(cfg, existing=globals().get('inst'))
""")

md(r"""
## 2. Pre-flight — reachable span and the contact resonances
Walks the laser from the free end toward the base in `step_um` steps (AutoWedge, InvOLS, engage, tune, withdraw
at each) until the detection fails. Writes the span, CR1/CR2/CR3 and InvOLS(x). About 1.2 min per step.
""")
code(r"""
pf = stages.preflight(inst, cfg)
""")

md(r"""
## 3. Two-domain sparse capture (live)
At each laser position: both spots, every bias, one tune each. The figure updates after every position —
**a** the reconstructed map filling in, **b** the quasi-static d_eff along the lever (flat after the domain-potential
correction = healthy survey), **c** E_map(x) with its leave-one-out check and the suggested imaging stop,
**d** how much the last position changed E(x). Interrupt the kernel or create `STOP` in the campaign folder to end
early; re-running this cell resumes from the checkpoint.
""")
code(r"""
%matplotlib inline
series = stages.capture(inst, cfg)          # e.g. stages.capture(inst, cfg, n_positions=6) for a one-off change
""")

md(r"""
## 4. Analysis — δ, V_cpd, E_map(x), d_eff
Pure numpy; safe to re-run as often as you like, also on an old checkpoint (`stages.analyse(cfg)` reads it from the
campaign folder). Check: the corrected quasi-static d_eff should be flat (sd of a few %), and `E_meas / E_loo`
should be close to 1 at the interior positions.
""")
code(r"""
res = stages.analyse(cfg)
""")

md(r"""
## 5. Imaging plan — QS at the electrostatic blind spot, BE at interior stops
The quasi-static reference frame goes where the QS electrostatic term b/P is smallest (on a stiff lever that is
the free end; the table says how far off an uncorrected QS image reads there). BE frames go to the stops you name in
`BE_POSITIONS` (stage µm), or to the recommended ones (largest validated E, flat over the drift budget, spaced).
Each BE stop carries E_map and whether the leave-one-out test validates that stretch of the lever.
""")
code(r"""
BE_POSITIONS = None               # e.g. [150.0, 110.0]  (None = recommended from E_map)
QS_POSITION  = None               # e.g. 232.0           (None = the electrostatic blind spot)
plan = stages.plan_imaging(cfg, be_positions_um=BE_POSITIONS, qs_x_um=QS_POSITION)
""")

md(r"""
## 6. Run the imaging plan
For every stop: laser → stop, AutoWedge + InvOLS, engage at the capture load, tune (CR1, FWHM), frames, tune-after,
withdraw. At the QS stop: single-frequency frames at `f_qs_Hz` (E = 1). At BE stops: one BE frame per bias-plan entry,
band centred on the CR1 just measured, width `be_width_x_fwhm` × FWHM, through the adapter named in
`cfg.imaging.be_acquire` (copy `activemodemap/smm/be_adapter_template.py` to `smm_be_adapter.py` and fill in your BE
call once). d_eff is printed per frame: QS from the amplitude channel, BE from the phase-coherent domain-mean spectra
and an SHO fit, divided by E_map(stop).
`stops=[0]` runs only the QS stop; `stops=[1]` only the first BE stop.
""")
code(r"""
frames = stages.run_imaging(inst, cfg)          # or stages.run_imaging(inst, cfg, stops=[0])
""")

md(r"""
## 7. Close-out
Bias 0 V, withdraw, a one-table summary of every frame's d_eff next to the survey value, and a manifest of everything
this campaign produced.
""")
code(r"""
manifest = stages.closeout(inst, cfg)
""")

md(r"""
## Expert corner
- **One-off overrides** without touching CONFIG: every stage accepts its section's fields as keywords, e.g.
  `stages.capture(inst, cfg, bias_V=[-6, -3, 0, 3, 6], n_positions=6)`, `stages.analyse(cfg, rank=3, track_peaks=False)`,
  `stages.pick_position(cfg, tol_pct=10)`.
- **Resume / redo**: `stages.capture(..., resume=False)` starts the survey over (delete or rename the checkpoint first);
  `stages.analyse(cfg)` and `stages.pick_position(cfg)` only read files and can be repeated.
- **Another stop later**: `stages.plan_imaging(cfg, be_positions_um=[200.0])` then `stages.run_imaging(inst, cfg, stops=[1])`.
- **Old data**: point `cfg.campaign.data_dir` at a finished campaign and run cells 4–5 only.
- **What is saved**: `smm_config.json`, `01_preflight/`, `10_survey/` (checkpoint `.npz`, `survey_records.json`,
  `capture_live.png`), `20_analysis/` (`emap_analysis.json`, `emap_maps.npz`, `emap_analysis.png`),
  `30_plan/imaging_plan.json`, `40_images/images.json` (+ the frames), `logs/`, `manifest.json`.
- **Conventions** (same as `paper_analysis/fmmpaper`): raw lock-in volts → metres = V·InvOLS/32; `P_raw = (Z₁ − Z₂)/2`;
  `P = P_raw − δ·b` with δ fitted on the displacement in the QS band assuming uniform d_eff; E read at the peak of each
  bias condition; E_map interpolated with a Chebyshev basis in x; leave-one-out at every measured position; QS blind spot =
  smallest |b_QS/P_QS| along the lever; BE d_eff = |P_SHO(f0)|·InvOLS/32/V_ac/E_map (SHO amplitude at f0, so ÷E not ÷M).
""")

nb["cells"] = cells
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks", "08_smm_calibration.ipynb")
nbf.write(nb, out)
print("wrote", out)
