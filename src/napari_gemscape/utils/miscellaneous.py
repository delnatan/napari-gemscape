from pathlib import Path

import imageio
from napari import Viewer


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
