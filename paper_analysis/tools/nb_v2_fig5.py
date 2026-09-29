from nbbuild import SETUP

CELLS = [
("md", """
# Figure 5 · Quantitative d33 across two domains with E from the mode map (stiff probe)

Two-domain decomposition, E predicted at never-visited positions from the two-domain P map, d33 along the lever, and the CR1 images at positions A and B calibrated with E_FMM(x) against the quasi-static images and spectroscopy. Needs `results/live_ebgp_d33_stiff_r2.json` (from `tools/live_ebgp_d33.py`). Source: `tools/fig5_d33.py`.

Rerun this notebook to regenerate the figure in `figures/`; the figure module is imported from `tools/`.
"""),
("code", SETUP),
("code", """
sys.path.insert(0, str(pathlib.Path.cwd().parent / "tools"))
import importlib, fig5_d33 as M
importlib.reload(M)
M.main()
"""),
("code", """
from IPython.display import Image, display
display(Image(filename=str(F.config.FIG_DIR / "Fig5_d33.png"), width=900))
"""),
]
