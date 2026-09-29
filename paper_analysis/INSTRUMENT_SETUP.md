# Running the paper analysis on the instrument PC

The acquisition scripts (`scripts/scmpit/...`, `activemodemap.asylum`) stay in the Python 3.7 environment that talks
to Igor. The analysis (`paper_analysis/fmmpaper`) needs SciPy and Python ≥ 3.10, so it runs in a second, separate
environment. Nothing in it touches the instrument; it only reads the checkpoint `.npz` files the campaign scripts write.

## One-time setup (PowerShell, on the instrument PC)

```powershell
cd C:\...\ActiveModeMap            # the repo root (contains activemodemap\ and paper_analysis\)
git fetch
git checkout feat/position-scale-and-reanalysis
git pull

# a second Python (3.11 or 3.12) from python.org, then:
py -3.11 -m venv .venv-analysis
.\.venv-analysis\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r paper_analysis\requirements.txt      # numpy, scipy, pandas, matplotlib, igor2, jupyter
```

Tell the analysis where the data are. Either set two environment variables in the session

```powershell
$env:FMM_DATA_ROOT = "C:\path\to\ActiveModeMap\Data"    # the folder that contains DomainsB_SCMPIT_R2, DomainsBPPPCONTAU, ...
$env:FMM_AMM_REPO  = "C:\...\ActiveModeMap"             # the repo root (default: parent of paper_analysis, usually right)
```

or write them once into `paper_analysis\config.local.json`:

```json
{"data_root": "C:\\path\\to\\ActiveModeMap\\Data", "amm_repo": "C:\\...\\ActiveModeMap"}
```

Check: `python -c "import fmmpaper as F; print(F.config.describe())"` from inside `paper_analysis\`.

## During a campaign

1. Run the sparse capture as usual (two-domain bias survey, or the 0 V wideband capture).
2. Pick the imaging position from it (seconds):

   ```powershell
   cd paper_analysis
   python tools\pick_image_position.py "C:\...\10_bias_survey\domains_bias_checkpoint_500nN_scmpit.npz" --probe stiff
   ```

   It prints E(x) along the lever (stage coordinates and distance from the clamp), the value of E to use at any
   position, and a recommended position: large E and less than `--tol_pct` (default 3 %) change over ±`--drift_um`
   (default 2 µm) of laser drift. A two-domain capture is preferred; from a single-domain 0 V capture E carries the
   electrostatic channel and the script says so.
3. Image at that position. For BE-PFM the per-pixel SHO amplitude is the on-resonance amplitude; divide by E from
   step 2 (and by the amplitude InvOLS at that stop) for d33.

## Useful tools

- `tools\live_ebgp_d33.py` — reconstruct a live capture with the calibrated EB arm and score it on a validation capture.
- `tools\calibrate_lever.py <stiff|soft>` — the one-time lever calibration from a dense map (needed only for a new lever).
- `notebooks\X_reconstruction_explorer.ipynb` — interactive explorer (`jupyter lab` inside the venv).

The lever calibrations used by everything are `results\lever_calibration_stiff_mass.json` and `..._soft_mass.json`;
they are committed. A new lever needs a dense map and a run of `calibrate_lever.py`.
