"""Tiny helper: build .ipynb files from a list of ('md'|'code', source) cells.

Notebooks are authored in tools/nb_*.py so they diff cleanly in git; run
``python tools/build_notebooks.py`` to regenerate notebooks/*.ipynb.
"""
import nbformat as nbf


def build(cells, path):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for kind, src in cells:
        src = src.strip("\n")
        nb.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    nbf.write(nb, path)
    return path


SETUP = '''\
%matplotlib inline
import sys, pathlib
# make the paper package importable when running from notebooks/
sys.path.insert(0, str(pathlib.Path.cwd().parent))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import fmmpaper as F
from fmmpaper import io, axis, spectra, domains, recon, plotting as fp, results
fp.setup()
print(F.config.describe())'''
