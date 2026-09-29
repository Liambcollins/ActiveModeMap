"""Regenerate notebooks/*.ipynb from the tools/nb_*.py sources.

    python tools/build_notebooks.py            # build only
    python tools/build_notebooks.py --run      # build and execute (writes outputs)
    python tools/build_notebooks.py --run --only=03a   # execute one notebook

Edit the notebook *sources* here, or edit the .ipynb directly and stop using this
script for that notebook -- either works, but pick one per notebook.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from nbbuild import build  # noqa: E402
import nb_00_overview, nb_02_sparse, nb_02a_ppp_per_mode, nb_03a_generality, nb_04_live, nb_06_transfer, nb_si_invols, nb_si_background, nb_05_contact_state, nb_05b_joint_geometry, nb_fig1, nb_fig2, nb_fig3, nb_fig4, nb_fig4_stiff, nb_fig5, nb_fig6, nb_fig7, nb_si_stiff_repeat, nb_si_load, nb_explorer, nb_stubs, nb_v2_fig1, nb_v2_fig2, nb_v2_fig3, nb_v2_fig4, nb_v2_fig5  # noqa: E402

NB = pathlib.Path(__file__).resolve().parent.parent / "notebooks"
NB.mkdir(exist_ok=True)

WORKING = {
    "00_data_overview": nb_00_overview.CELLS,
    "02_fig2-3_sparse_reconstruction_gridB": nb_02_sparse.CELLS,
    "02a_fig3_ppp_per_mode_physics_vs_data": nb_02a_ppp_per_mode.CELLS,
    "03a_fig3_generality_scmpitB_dense": nb_03a_generality.CELLS,
    "04_fig4_live_fmm": nb_04_live.CELLS,
    "05a_fig5_contact_state_frame_drift": nb_05_contact_state.CELLS,
    "05b_fig5_joint_geometry_all_modes": nb_05b_joint_geometry.CELLS,
    "06_fig6_transfer_function_d33": nb_06_transfer.CELLS,
    "V2_F1_concept": nb_v2_fig1.CELLS,
    "V2_F2_reconstruction_vs_N": nb_v2_fig2.CELLS,
    "V2_F3_methods": nb_v2_fig3.CELLS,
    "V2_F4_sweeps_bias_load": nb_v2_fig4.CELLS,
    "V2_F5_d33_two_domains": nb_v2_fig5.CELLS,
    "F1_fig1_why_and_how_fast": nb_fig1.CELLS,
    "F2_fig2_sparse_reconstruction": nb_fig2.CELLS,
    "F3_fig3_how_many_and_where": nb_fig3.CELLS,
    "F4_fig4_live_fmm": nb_fig4.CELLS,
    "F4s_fig4_live_fmm_stiff_only": nb_fig4_stiff.CELLS,
    "F5_fig5_one_geometry": nb_fig5.CELLS,
    "F6_fig6_E_not_Q_d33": nb_fig6.CELLS,
    "F7_fig7_electrostatic_channel": nb_fig7.CELLS,
    "SIa_invols_drift": nb_si_invols.CELLS,
    "SIb_additive_background": nb_si_background.CELLS,
    "SIc_stiff_repeat_and_gridB": nb_si_stiff_repeat.CELLS,
    "SId_load_series_A": nb_si_load.CELLS,
    "X_reconstruction_explorer": nb_explorer.CELLS,
}


def main(run=False, only=None):
    # --only rebuilds (and runs) just the matching notebook, so other notebooks keep their outputs
    items = {k: c for k, c in WORKING.items() if not only or k.startswith(only)}
    paths = [build(c, NB / f"{k}.ipynb") for k, c in items.items()]
    if not only:
        for k, (title, plan, keys) in nb_stubs.STUBS.items():
            build(nb_stubs.cells_for(title, plan, keys), NB / f"{k}.ipynb")
    if run:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
        for p in paths:
            nb = nbformat.read(p, as_version=4)
            ExecutePreprocessor(timeout=3600, kernel_name="python3").preprocess(
                nb, {"metadata": {"path": str(NB)}})
            nbformat.write(nb, p)
            print("executed", p.name)


if __name__ == "__main__":
    only = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")), None)
    main(run="--run" in sys.argv, only=only)
