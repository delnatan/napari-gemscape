from pathlib import Path

import imageio
import matplotlib.patheffects as path_effects
import numpy as np
from magicgui import magic_factory
from matplotlib.patches import Circle
from napari import Viewer


@magic_factory(mp4_output_path={"widget_type": "FileEdit", "mode": "w"})
def record_timelapse_movie(
    viewer: Viewer,
    t_start: int,
    t_end: int,
    mp4_output_path: Path,
    fps: int = 5,
    movie_title: str = "movie title",
):
    """record movie as you see it in the napari canvas
    This function assumes that you have a 2D timelapse and that the first
    dimension is the time axis.

    """
    frames = []

    for t in range(t_start, t_end + 1):
        viewer.dims.current_step = (t, 0, 0)
        frame = viewer.screenshot(flash=False)
        frames.append(frame)

    writer = imageio.get_writer(
        mp4_output_path,
        fps=fps,
        quality=9,
        format=".mp4",
        output_params=["-metadata", f"title={movie_title}"],
    )

    for frame in frames:
        # ensure divisibility by 2
        M, N = frame.shape[:2]
        M = M // 2 * 2
        N = N // 2 * 2
        writer.append_data(frame[:M, :N, :3])

    writer.close()


def print_dict_keys(d, level=0):
    """convenient function to print out nested dictionary values"""
    for key, value in d.items():
        if isinstance(value, dict):
            print("|   " * level + "|-- " + str(key) + " (dict)")
            print_dict_keys(value, level + 1)
        else:
            if isinstance(value, (int, float, str, bool)):
                print(
                    "|   " * level
                    + "|-- "
                    + str(key)
                    + " ("
                    + type(value).__name__
                    + "): "
                    + str(value)
                )
            else:
                print(
                    "|   " * level
                    + "|-- "
                    + str(key)
                    + " ("
                    + type(value).__name__
                    + ")"
                )


def convert_numpy_types(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def hairball_plot(ax, tracks, nth=2, shell_diams=[0.1, 0.3], dxy=0.065):
    df = tracks[["particle", "frame", "x", "y"]].copy()
    df[["dx", "dy"]] = df.groupby("particle")[["x", "y"]].diff()
    df["xo"] = df.groupby("particle")["dx"].cumsum()
    df["yo"] = df.groupby("particle")["dy"].cumsum()

    particle_ids = df["particle"].unique()[::nth]
    thinned = df[df["particle"].isin(particle_ids)]

    for pid in thinned["particle"].unique():
        _df = thinned[thinned["particle"] == pid]
        ax.plot(
            _df["xo"].values * dxy,
            _df["yo"].values * dxy,
            "-",
            lw=0.9,
            alpha=0.25,
            color="#444444",
            zorder=1,
        )

    for diam in shell_diams:
        c = Circle(
            (0, 0),
            diam,
            linestyle="dashed",
            color="black",
            fill=False,
            linewidth=1.2,
        )
        _t = ax.text(
            diam,
            diam,
            f"{diam:.2f} $\\mu m$",
            color="white",
            ha="center",
            fontsize=8,
        )
        _t.set_path_effects(
            [
                path_effects.Stroke(linewidth=1.25, foreground="black"),
                path_effects.Normal(),
            ]
        )

        ax.add_artist(c)

    ax.set_ylim(-shell_diams[-1] * 1.75, shell_diams[-1] * 1.75)
    ax.set_xlim(-shell_diams[-1] * 1.75, shell_diams[-1] * 1.75)

    ax.set_aspect("equal")
