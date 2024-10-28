"""
various visualization for diffusion analysis

"""

import numpy as np
from matplotlib import style


def plot_ensemble_MSD(df, ax, coefs, D, D_std, npts):
    ax.errorbar(
        df["lag"],
        df["mean"],
        yerr=df["std"] / np.sqrt(df["count"]),
        fmt="o",
        mec="k",
        mew=1.25,
        mfc="w",
        ecolor="#b5b5b5",
        elinewidth=1.2,
        capsize=3,
    )
    ax.plot(
        df["lag"][:npts],
        coefs[0] * df["lag"][:npts] + coefs[1],
        "k--",
        zorder=float("inf"),
        lw=2,
    )
    ax.set_title(f"D = {D:.3f} +/- {D_std:.3f} $\\mu m^2/s$")
    ax.set_ylabel("MSD")
    ax.set_xlabel("$\\tau$, seconds")

    # also show trajectory counts as secondary y-axis
    with style.context("ggplot"):
        ax_2 = ax.twinx()

        ax_2.plot(
            df["lag"],
            df["count"],
            drawstyle="steps-mid",
            c="#8095ab",
            zorder=-10,
        )
        ax_2.tick_params(axis="y", labelcolor="#8095ab", colors="#8095ab")
        ax_2.set_ylabel("# tracks", color="#8095ab")
        ax_2.grid(False)
