"""Demo + benchmark figures for the AL mode-shape-mapping study."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from activemodemap import EBForwardModel, VirtualAFM, run_loop

# categorical palette (fixed order) + shared style
C = {"eig": "#2a78d6", "spot": "#eb6834", "variance": "#1baf7a",
     "random": "#eda100", "equispaced": "#e87ba4", "gp": "#4a3aa7"}
LBL = {"eig": "EIG (D-optimal)", "spot": "Spot-targeted",
       "variance": "Integrated variance", "random": "Random",
       "equispaced": "Equispaced", "gp": "Pure GP (no physics)"}
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.6,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


# ---------------------------------------------------------------- demo figure
def demo_figure():
    rng = np.random.default_rng(42)
    m = EBForwardModel()
    theta_true = np.array([3.2, np.log10(2000 / 3), 1.4, -0.2, 0.05])
    afm = VirtualAFM(m, theta_true, rng=rng)
    h = run_loop(afm, "eig", n_total=8, rng=rng)
    post = h["posterior"]
    mean_amp, std_amp, spots = post.predictive_maps(80)

    L = m.geom.L_um
    x_um, f_khz = m.xi * L, m.f_hz / 1e3
    nf = m.omega.size
    true_map = np.abs(afm.true_maps[+1.0])
    ext = [x_um[0], x_um[-1], f_khz[0], f_khz[-1]]

    fig, axs = plt.subplots(2, 2, figsize=(9.2, 6.6))
    for ax, mp, title in [(axs[0, 0], true_map, "Ground truth  $|z(x,f)|$"),
                          (axs[0, 1], mean_amp[:nf], "Posterior-mean reconstruction")]:
        im = ax.imshow(np.log10(mp + 1e-3), origin="lower", aspect="auto",
                       extent=ext, cmap="viridis")
        ax.grid(False)
        ax.set_xlabel("detection position (µm)")
        ax.set_ylabel("frequency (kHz)")
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, label="log$_{10}$ amplitude", shrink=0.85)
    for x in h["xs"]:
        axs[0, 1].axvline(x * L, color="w", lw=0.7, alpha=0.85)
    axs[0, 1].text(0.02, 0.965, f"{len(h['xs'])} positions (white)",
                   transform=axs[0, 1].transAxes, color="w", fontsize=8,
                   va="top")

    err_map = np.abs(mean_amp[:nf] - true_map) / true_map.max()
    im = axs[1, 0].imshow(np.log10(err_map + 1e-6), origin="lower",
                          aspect="auto", extent=ext, cmap="magma",
                          vmin=-5, vmax=-1)
    axs[1, 0].grid(False)
    axs[1, 0].set_xlabel("detection position (µm)")
    axs[1, 0].set_ylabel("frequency (kHz)")
    axs[1, 0].set_title("Reconstruction error (log$_{10}$, rel.)", fontsize=10)
    plt.colorbar(im, ax=axs[1, 0], shrink=0.85)

    ax = axs[1, 1]
    n = h["n"]
    ax.plot(n, h["dns_err"], "-o", ms=4, color="#2a78d6", label="D-NS error")
    ax.plot(n, h["desbs_err"], "-o", ms=4, color="#eb6834", label="D-ESBS error")
    ax.fill_between(n, np.array(h["dns_ci"]) / 2, color="#2a78d6", alpha=0.14,
                    label="D-NS 95% CI half-width")
    ax.axhline(4.0, color="0.4", lw=0.8, ls="--")
    ax.text(n[-1], 4.25, "laser spot size", ha="right", fontsize=8, color="0.35")
    ax.set_yscale("log")
    ax.set_xlabel("number of measured positions")
    ax.set_ylabel("spot-position error (µm)")
    ax.set_title(f"Null/blind-spot convergence  "
                 f"(truth: D-NS {afm.dns_um:.1f}, D-ESBS {afm.desbs_um:.1f} µm)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Physics-informed active learning demo — EIG acquisition, "
                 "well-specified case", y=1.005, fontsize=11)
    fig.tight_layout()
    fig.savefig("figures/demo_reconstruction.png")
    plt.close(fig)
    print("demo figure done")


# ----------------------------------------------------------- benchmark figure
def benchmark_figure():
    d = np.load("benchmark_results.npz", allow_pickle=True)
    strategies = list(d["strategies"])
    scenarios = list(d["scenarios"])
    titles = {"well_specified": "Well-specified",
              "high_noise": "High noise (4×)",
              "mismatched": "Model mismatch (perturbed geometry)"}
    metrics = [(0, "map RMSE (rel.)", "log"),
               (1, "D-NS position error (µm)", "log"),
               (2, "D-ESBS position error (µm)", "log")]

    fig, axs = plt.subplots(len(metrics), len(scenarios),
                            figsize=(11.5, 8.4), sharex=True)
    for j, scen in enumerate(scenarios):
        for i, (mi, mlabel, scale) in enumerate(metrics):
            ax = axs[i, j]
            for st in strategies:
                key = f"{scen}|{st}"
                if key not in d:
                    continue
                arr = d[key][:, mi, :]          # (n_truths, n_steps)
                if np.all(np.isnan(arr)):
                    continue
                med = np.nanmedian(arr, axis=0)
                q1 = np.nanpercentile(arr, 25, axis=0)
                q3 = np.nanpercentile(arr, 75, axis=0)
                n = np.arange(1, arr.shape[1] + 1)
                ax.plot(n, med, "-", color=C[st], lw=1.6, label=LBL[st])
                ax.fill_between(n, q1, q3, color=C[st], alpha=0.10)
            ax.set_yscale(scale)
            if i == 0:
                ax.set_title(titles.get(scen, scen), fontsize=10.5)
            if i == len(metrics) - 1:
                ax.set_xlabel("number of measured positions")
            if j == 0:
                ax.set_ylabel(mlabel)
            if mi in (1, 2):
                ax.axhline(4.0, color="0.45", lw=0.8, ls="--")
                ax.axhline(1.0, color="0.7", lw=0.7, ls=":")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, loc="lower center",
               bbox_to_anchor=(0.5, -0.025), frameon=False, fontsize=9)
    fig.suptitle("Active-learning benchmark: median (IQR band) over ground "
                 "truths — dashed line = 4 µm laser spot, dotted = 1 µm target",
                 y=1.005, fontsize=11)
    fig.tight_layout()
    fig.savefig("figures/benchmark_curves.png")
    plt.close(fig)
    print("benchmark figure done")


if __name__ == "__main__":
    import sys
    if "demo" in sys.argv or len(sys.argv) == 1:
        demo_figure()
    if "bench" in sys.argv:
        benchmark_figure()
