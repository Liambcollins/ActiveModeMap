from nbbuild import SETUP

CELLS = [
("md", """
# Figure 3 · Method comparison: error vs N, floors, sampling, what transfers

Error vs N for EB / EB+GP / GP / low-rank on one band and on all bands with the three floors; the stiff probe with a data-only library basis from an earlier dense map; sampling strategies (equispaced, 50 random, node-avoiding); the inferred contact stiffness vs N; the live-capture transfer test. The compute step takes ~25 min (cached in `results/fig3_methods.json`). Source: `tools/fig3_methods.py`.

Rerun this notebook to regenerate the figure in `figures/`; the figure module is imported from `tools/`.
"""),
("code", SETUP),
("code", """
sys.path.insert(0, str(pathlib.Path.cwd().parent / "tools"))
import importlib, fig3_methods as M
importlib.reload(M)
import json
out = json.load(open(M.CACHE)) if M.CACHE.exists() else json.loads(json.dumps(M.compute(), default=float))
M.plot(out, F.config.FIG_DIR / 'Fig3_methods.png')
"""),
("code", """
from IPython.display import Image, display
display(Image(filename=str(F.config.FIG_DIR / "Fig3_methods.png"), width=900))
"""),
]
