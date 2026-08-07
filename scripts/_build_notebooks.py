"""Generate the ActiveModeMap notebooks as .ipynb files."""
import json, os

def md(*lines): return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}
def code(*lines): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": _src(lines)}
def _src(lines):
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]

def write(path, cells):
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python", "version": "3.x"}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(HERE, "notebooks")

# ===================================================================== #
# 01 — dry run against the virtual instrument                           #
# ===================================================================== #
c = []
c.append(md(
"# 01 · Active loop — dry run (no hardware)",
"",
"Runs the **exact same active-learning loop** used on the microscope, but against a",
"simulated `VirtualInstrument`. Use this to understand the workflow and sanity-check",
"parameters before spending microscope time — nothing here touches Igor.",
"",
"The loop: pick a detection position → 'measure' a complex tune there → update the",
"low-rank reconstruction → pick the next position by D-optimal design → stop when the",
"D-NS confidence interval is small enough."))
c.append(code(
"import sys, os", "sys.path.insert(0, os.path.abspath('..'))",
"import numpy as np", "import matplotlib.pyplot as plt",
"from activemodemap import VirtualInstrument, LowRankModeMap, plot_state"))
c.append(md("## Parameters"))
c.append(code(
"# common tune window (Hz) and accessible laser-position grid (um)",
"freq_grid = np.linspace(254e3, 449e3, 400)",
"L_um      = 225.0                      # cantilever length (for reporting)",
"x_grid    = np.arange(L_um - 40, L_um + 0.01, 1.0)   # last ~40 um near the tip",
"",
"RANK          = 5      # spatial basis dimension (map is low rank)",
"MAX_POSITIONS = 10",
"DNS_CI_TOL_UM = 1.0    # stop when D-NS 95% CI drops below this",
"SEEDS_UM      = [x_grid[0], x_grid[len(x_grid)//2], x_grid[-1]]"))
c.append(code(
"inst = VirtualInstrument(freq_grid_Hz=freq_grid, noise_floor=0.08,",
"                         spot_fwhm_um=4.0, rng=np.random.default_rng(1))",
"print(f'(hidden) true D-NS = {inst.dns_um_from_end:.2f} um from the free end')"))
c.append(md("## Run the loop (live plot each step)"))
c.append(code(
"mm = LowRankModeMap(x_grid, freq_grid, rank=RANK, seeds_um=SEEDS_UM,",
"                    dns_ci_tol_um=DNS_CI_TOL_UM)",
"rec = None",
"for step in range(MAX_POSITIONS):",
"    x = mm.next_position()",
"    f, Z, meta = inst.measure_at(x)          # <-- hardware call is swapped in here",
"    mm.add_measurement(x, f, Z)",
"    if mm.n >= mm.min_positions:",
"        rec = mm.reconstruct(nboot=150)",
"        dns_end = L_um - rec['dns']",
"        print(f'N={mm.n:2d}  x={x:6.1f} um  ->  D-NS = {dns_end:5.2f} um from end'",
"              f'  (95% CI {rec[\"dns_ci\"]:.2f} um)   converged={mm.converged()}')",
"        if mm.converged():",
"            print('\\nConverged.'); break"))
c.append(code(
"ax = plot_state(mm, rec)",
"plt.tight_layout(); plt.show()",
"print(f'positions used: {mm.n}   true D-NS {inst.dns_um_from_end:.2f} um   '",
"      f'recovered {L_um-rec[\"dns\"]:.2f} +/- {rec[\"dns_ci\"]/2:.2f} um')"))
write(os.path.join(NB, "01_active_loop_dryrun.ipynb"), c)

# ===================================================================== #
# 02 — reconstruct the dense example dataset (offline, real data)       #
# ===================================================================== #
c = []
c.append(md(
"# 02 · Reconstruct from the dense example data (offline)",
"",
"Loads the bundled 30-position dense sweep (`data/example_1000nN`, a Multi75E-G probe",
"on PPLN at 1000 nN, IDS) and shows that the **model-light low-rank reconstruction**",
"recovers the full mode shape — resonance and antiresonance branches, sharp — from a",
"handful of D-optimally chosen positions, matching the full sweep's D-NS. This is the",
"offline counterpart of the on-instrument loop."))
c.append(code(
"import sys, os, glob, re", "sys.path.insert(0, os.path.abspath('..'))",
"import numpy as np", "import matplotlib.pyplot as plt",
"from activemodemap.lowrank import (reconstruct_map, d_optimal_order,",
"                                   resonance_index, dns_from_map)"))
c.append(md("## Load the tune files into a position x frequency map"))
c.append(code(
"data_dir = os.path.join('..', 'data', 'example_1000nN')",
"def parse_tune(path):",
"    txt = open(path, encoding='latin-1').read()",
"    m = txt.split('AmpBEGIN')[1] if 'AmpBEGIN' in txt else txt",
"    m = re.split(r'END', m)[0]",
"    v = np.array([float(x) for x in m.split() if re.match(r'^-?[\\d.eE+-]+$', x)])",
"    n = (v.size // 3) * 3",
"    a = v[:n].reshape(-1, 3)",
"    return a[:, 0], a[:, 1], a[:, 2]   # freq, phase(deg), amp",
"files = sorted(glob.glob(os.path.join(data_dir, 'Tune_*.txt')),",
"               key=lambda s: int(re.search(r'_(\\d+)\\.txt', s).group(1)))",
"F, A, P = [], [], []",
"for fp in files:",
"    f, ph, a = parse_tune(fp); F.append(f); A.append(a); P.append(ph)",
"fg = F[0]",
"Z = np.array([np.interp(fg, f, a) for f, a in zip(F, A)]) * np.exp(",
"    1j * np.deg2rad(np.array([np.interp(fg, f, p) for f, p in zip(F, P)])))",
"npos = Z.shape[0]; x_grid = np.arange(npos, dtype=float)   # ~1 um steps",
"print(f'{npos} positions x {fg.size} frequencies, {fg[0]/1e3:.0f}-{fg[-1]/1e3:.0f} kHz')"))
c.append(md("## Reference D-NS from the full sweep, then reconstruct from few positions"))
c.append(code(
"true = np.abs(Z)",
"ires_full = resonance_index(fg, Z)",
"dns_ref = dns_from_map(x_grid, Z, fg, ires_full)",
"print(f'reference D-NS (all {npos} positions): {dns_ref:.2f}')",
"",
"RANK, K = 5, 6",
"sel = d_optimal_order(x_grid, RANK, n_select=K)",
"rec = reconstruct_map(x_grid, sel, Z[sel], RANK)",
"ires = resonance_index(fg, Z[sel])",
"dns = dns_from_map(x_grid, rec['Zrec'], fg, ires)",
"held = [i for i in range(npos) if i not in sel]",
"rmse = np.sqrt(np.mean((np.abs(rec['Zrec'][held]) - true[held])**2)) / true.max()",
"print(f'reconstruction from {K} positions {sorted(sel)}: '",
"      f'held-out RMSE {rmse*100:.1f}%, D-NS {dns:.2f}')"))
c.append(code(
"fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))",
"ext = [x_grid[0], x_grid[-1], fg[0]/1e3, fg[-1]/1e3]",
"vmin, vmax = np.log10(true+1e-6).min(), np.log10(true+1e-6).max()",
"ax[0].imshow(np.log10(true.T+1e-6), origin='lower', aspect='auto', extent=ext, cmap='viridis', vmin=vmin, vmax=vmax)",
"ax[0].axvline(dns_ref, color='#e34948', ls='--'); ax[0].set_title('measured (30 positions)')",
"ax[1].imshow(np.log10(np.abs(rec['Zrec']).T+1e-6), origin='lower', aspect='auto', extent=ext, cmap='viridis', vmin=vmin, vmax=vmax)",
"[ax[1].axvline(x_grid[s], color='w', lw=0.8) for s in sel]",
"ax[1].set_title(f'reconstructed from {K}')",
"ax[2].imshow((rec['std']/true.max()).T, origin='lower', aspect='auto', extent=ext, cmap='magma')",
"ax[2].set_title('uncertainty (1sigma)')",
"[a.set_xlabel('position (um)') for a in ax]; ax[0].set_ylabel('frequency (kHz)')",
"plt.tight_layout(); plt.show()"))
write(os.path.join(NB, "02_reconstruct_dense_data.ipynb"), c)

# ===================================================================== #
# 03 — run the full loop on the instrument                              #
# ===================================================================== #
c = []
c.append(md(
"# 03 · Run the active-learning mode-shape loop ON THE INSTRUMENT",
"",
"Drives an Asylum/Cypher system through Igor Pro to reconstruct the contact-resonance",
"mode shape from the **minimum number of laser positions**, choosing each position by",
"active learning and stopping when the displacement null spot (D-NS) is pinned.",
"",
"**Windows / Igor only.** Requires `win32com` and `igor2`, and the same environment",
"your dense-sweep notebook runs in.",
"",
"### Before you run — set up in Igor (once)",
"1. Approach and find the contact resonance at your chosen load; confirm the tip is well-behaved.",
"2. **Set a FIXED, WIDE tune range** that brackets the whole resonance **and** the",
"   antiresonance across the positions you will scan (e.g. the 254-449 kHz window used",
"   for the dense sweep). Do **not** use resonance tracking / auto-recenter — every",
"   position must be tuned over the *same* frequency window.",
"3. Park the detection laser at the **start** of the span you want to map (positions",
"   below are absolute micrometers measured from here, increasing toward the tip).",
"4. Make sure the save folder exists and the base filename is set as usual."))
c.append(md("## 1 · Imports and connect to Igor"))
c.append(code(
"import sys, os", "sys.path.insert(0, os.path.abspath('..'))",
"import numpy as np", "import matplotlib.pyplot as plt", "import win32com.client",
"from activemodemap.asylum import AFMLaserSweepAutomation, AsylumInstrument",
"from activemodemap import LowRankModeMap, plot_state",
"",
"igor = win32com.client.Dispatch('IgorPro.Application')",
"print('Connected to Igor Pro')"))
c.append(md("## 2 · Experiment parameters — edit these"))
c.append(code(
"# --- file locations (same convention as the dense-sweep notebook) ---",
"file_loc      = r'D:\\User Data\\MARTI\\2026\\ActiveModeMap\\demo'   # must exist",
"base_filename = 'AMap'",
"log_filename  = os.path.join(file_loc, 'AMap_log')",
"",
"# --- contact / drive ---",
"load_nN   = 1000      # applied load (nN)",
"dc_bias_V = 0.0       # DC bias on Output.A (single-domain D-NS map)",
"tune_center_Hz = 350000    # for logging only; the actual window is whatever you set in Igor",
"",
"# --- accessible laser-position grid (ABSOLUTE um from the start position) ---",
"span_um   = 30.0      # total span to map, starting at the current laser position",
"step_um   = 1.0       # candidate grid spacing (the loop visits a subset of these)",
"x_grid    = np.arange(0.0, span_um + 1e-6, step_um)",
"",
"# --- active-learning settings ---",
"RANK          = 5       # spatial basis dimension",
"MAX_POSITIONS = 10      # hard cap on positions actually measured",
"MIN_POSITIONS = 6       # do not stop before this many",
"DNS_CI_TOL_UM = 1.0     # stop once the D-NS 95% CI is below this",
"RECALIBRATE_EACH = True # AutoWedge + InvOLS at every position (safer; set False to reuse first)",
"",
"os.makedirs(file_loc, exist_ok=True)"))
c.append(md("## 3 · Build the automation, the instrument adapter, and the loop manager"))
c.append(code(
"automation = AFMLaserSweepAutomation(igor, file_loc, base_filename, log_filename)",
"automation.eigenmode_center_freq = tune_center_Hz",
"automation.autowedge_pause = 15.0",
"automation.invols_bounds = (4e-8, 10e-7)",
"",
"inst = AsylumInstrument(automation, load_nN=load_nN, dc_bias_V=dc_bias_V,",
"                        x_start_um=0.0, recalibrate_each=RECALIBRATE_EACH)",
"",
"mm = LowRankModeMap(x_grid, freq_grid=None, rank=RANK,",
"                    seeds_um=[x_grid[0], x_grid[len(x_grid)//2], x_grid[-1]],",
"                    min_positions=MIN_POSITIONS, dns_ci_tol_um=DNS_CI_TOL_UM)",
"print('Ready. Grid positions:', x_grid)"))
c.append(md(
"## 4 · Run the active-learning loop",
"",
"Each iteration: pick the next position, drive the instrument (withdraw → move laser →",
"optical image → AutoWedge+InvOLS → engage → tune → read the complex spectrum →",
"withdraw), update the reconstruction, and decide whether to continue. Interrupt the",
"cell at any time; whatever has been measured is retained in `mm` and `inst.records`."))
c.append(code(
"rec = None",
"for step in range(MAX_POSITIONS):",
"    x = mm.next_position()",
"    print(f'\\n=== position {mm.n+1}/{MAX_POSITIONS}:  x = {x:.1f} um ===')",
"    freq, Z, meta = inst.measure_at(x)",
"    mm.add_measurement(x, freq, Z)",
"    if meta.get('resonance_freq_Hz'):",
"        print(f\"    f_res = {meta['resonance_freq_Hz']/1e3:.2f} kHz, \"",
"              f\"InvOLS = {meta['invols_m_per_V']:.2e} m/V\")",
"    if mm.n >= mm.min_positions:",
"        rec = mm.reconstruct(nboot=150)",
"        print(f'    D-NS estimate: {rec[\"dns\"]:.2f} um  (95% CI {rec[\"dns_ci\"]:.2f} um)')",
"        try:",
"            from IPython.display import clear_output; clear_output(wait=True)",
"        except Exception: pass",
"        ax = plot_state(mm, rec, title=f'{mm.n} positions'); plt.tight_layout(); plt.show()",
"        if mm.converged():",
"            print('\\nConverged — D-NS confidence interval below tolerance.'); break",
"inst.close()   # withdraw + zero DC bias",
"print('\\nLoop done. Positions measured:', mm.n)"))
c.append(md("## 5 · Final reconstruction and save"))
c.append(code(
"rec = mm.reconstruct(nboot=400)",
"x, f = mm.x_grid, mm.freq/1e3",
"ext = [x[0], x[-1], f[0], f[-1]]",
"fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))",
"amp = np.abs(rec['Zrec']).T",
"ax[0].imshow(np.log10(amp+1e-12), origin='lower', aspect='auto', extent=ext, cmap='viridis')",
"[ax[0].axvline(xs, color='w', lw=0.8) for xs in rec['x_sel']]",
"ax[0].axvline(rec['dns'], color='#e34948', ls='--')",
"ax[0].set_title(f'reconstructed mode shape ({mm.n} positions)')",
"ax[1].imshow((rec['std']/(amp.max()+1e-12)).T, origin='lower', aspect='auto', extent=ext, cmap='magma')",
"ax[1].set_title('uncertainty (1sigma, rel.)')",
"ax[2].hist(rec['dns_samples'], bins=16, color='#2a78d6', alpha=0.75, density=True)",
"ax[2].axvline(rec['dns'], color='k')",
"ax[2].set_title(f\"D-NS = {rec['dns']:.2f} +/- {rec['dns_ci']/2:.2f} um\")",
"[a.set_xlabel('position (um)') for a in ax[:2]]; ax[0].set_ylabel('frequency (kHz)')",
"plt.tight_layout()",
"fig.savefig(os.path.join(file_loc, 'ActiveModeMap_result.png'), dpi=150, bbox_inches='tight')",
"",
"# save the reconstruction, the raw measured spectra, and the per-position log",
"sel = rec['sel_idx']",
"np.savez(os.path.join(file_loc, 'ActiveModeMap_result.npz'),",
"         x_grid=mm.x_grid, freq=mm.freq, Zrec=rec['Zrec'], std=rec['std'],",
"         measured_x=mm.x_grid[sel],",
"         measured_Z=np.array([mm._measured[i] for i in sel]),",
"         dns=rec['dns'], dns_ci=rec['dns_ci'])",
"log_path = inst.save_log()",
"print('saved:  ActiveModeMap_result.png / .npz  and', os.path.basename(log_path))",
"print(f'D-NS = {rec[\"dns\"]:.2f} +/- {rec[\"dns_ci\"]/2:.2f} um  from {mm.n} positions')"))
c.append(md(
"## Notes and extensions",
"",
"- **D-ESBS (electrostatic blind spot).** This notebook maps the D-NS from a single-",
"  domain tune series. To also locate the D-ESBS, measure both PPLN domain orientations",
"  at each position (as in the dense-sweep protocol) and separate the piezoresponse and",
"  electrostatic channels via their difference and sum; the same low-rank reconstruction",
"  then applies to each channel.",
"- **Load / bias series.** Wrap this loop in an outer loop over loads or biases, re-",
"  seeding `mm` each time (the D-NS migrates with load).",
"- **Physics-informed option.** For a well-characterized probe you can instead drive the",
"  loop with the physics-informed reconstruction (`activemodemap.loop`), which needs",
"  fewer positions and returns contact stiffness and the drive ratio, at the cost of",
"  sensitivity to model error. See the paper's Methods and the simulation notebooks."))
write(os.path.join(NB, "03_run_on_instrument.ipynb"), c)
print("done")
