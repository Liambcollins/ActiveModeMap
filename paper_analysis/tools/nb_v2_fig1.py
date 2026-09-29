from nbbuild import SETUP

CELLS = [
("md", """
# Figure 1 · Concept and the dense ground truth (soft probe)

Schematic, the two dense maps (soft probe with five modes as the main map), one conventional sweep, fixed-frequency profiles and the acquisition time. Source: `tools/fig1_concept.py`.

Rerun this notebook to regenerate the figure in `figures/`; the figure module is imported from `tools/`.
"""),
("code", SETUP),
("code", """
sys.path.insert(0, str(pathlib.Path.cwd().parent / "tools"))
import importlib, fig1_concept as M
importlib.reload(M)
M.plot(F.config.FIG_DIR / 'Fig1_concept.png')
"""),
("code", """
from IPython.display import Image, display
display(Image(filename=str(F.config.FIG_DIR / "Fig1_concept.png"), width=900))
"""),
]
