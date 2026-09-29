from nbbuild import SETUP

CELLS = [
("md", """
# Figure 4 · Fast sweeps: mode-shape dynamics vs bias and load (soft probe)

Every (load, V_dc, domain) of the two-domain bias surveys is an 8-position capture, reconstructed with the calibrated EB arm over the whole band. Maps vs V_dc, spectra vs V_dc at one position, CR1 and CR3 mode shapes vs V_dc, the fitted electrostatic weight ε(V_dc) on both domains, and the contact stiffness at 15 vs 250 nN. Compute ~25 min (cached in `results/fig4_sweeps.npz`). Source: `tools/fig4_sweeps.py`.

Rerun this notebook to regenerate the figure in `figures/`; the figure module is imported from `tools/`.
"""),
("code", SETUP),
("code", """
sys.path.insert(0, str(pathlib.Path.cwd().parent / "tools"))
import importlib, fig4_sweeps as M
importlib.reload(M)
out = M.load_cache() if M.CACHE.exists() else (lambda o: (M.save(o), o)[1])(M.compute())
M.plot(out, F.config.FIG_DIR / 'Fig4_sweeps.png')
"""),
("code", """
from IPython.display import Image, display
display(Image(filename=str(F.config.FIG_DIR / "Fig4_sweeps.png"), width=900))
"""),
]
