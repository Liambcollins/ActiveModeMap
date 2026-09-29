from nbbuild import SETUP

CELLS = [
("md", """
# Figure 2 · Sparse reconstruction vs N (soft probe)

EB, EB+GP and low-rank from N = 4, 6, 9, 12 equispaced positions: error strips per band (absolute error as % of the band RMS), recovered spectra with the EB+GP posterior band, recovered mode shapes. Set `M.PROBE = 'stiff'` for the SI version. Source: `tools/fig_recovery_vs_N.py`.

Rerun this notebook to regenerate the figure in `figures/`; the figure module is imported from `tools/`.
"""),
("code", SETUP),
("code", """
sys.path.insert(0, str(pathlib.Path.cwd().parent / "tools"))
import importlib, fig_recovery_vs_N as M
importlib.reload(M)
M.PROBE = 'soft'
out = M.load() if M.CACHE().exists() else (lambda o: (M.save(o), o)[1])(M.compute())
M.plot(out, F.config.FIG_DIR / 'fig_recovery_vs_N_soft_linear.png', err_scale='linear')
"""),
("code", """
from IPython.display import Image, display
display(Image(filename=str(F.config.FIG_DIR / "fig_recovery_vs_N_soft_linear.png"), width=900))
"""),
]
