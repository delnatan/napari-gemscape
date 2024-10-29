"""

This file contains functions that take napari Layer objects or data (as a
pandas DataFrame) and does the processing. They return quantities that can be
used in a GEM analysis pipeline.


"""

import pandas as pd
import trackpy as tp
from napari.layers import Points, Tracks

from napari_gemscape.core.utils import (
    compute_msd,
    compute_track_quantities,
    compute_track_stats,
    fit_msds,
)

tp.quiet(suppress=True)


def link_trajectory(
    points=None,
    minimum_track_length=3,
    maximum_displacement=4.2,
    alpha_cutoff=0.05,
    drift_corr_smooth=5,
):
    """link trajectory given napari 'Points' layer

    filtering steps are also carried out for 1) minimum track length 2) motion
    classification
    """
    if points is None:
        return

    if isinstance(points, Points):
        df = pd.DataFrame(points.data, columns=["frame", "y", "x"])
        df = df.merge(points.features)
    elif isinstance(points, pd.DataFrame):
        df = points
    else:
        return None

    # link trajectories
    df = tp.link(df, maximum_displacement)

    if drift_corr_smooth > -1:
        drift = tp.compute_drift(df, smoothing=drift_corr_smooth)
        df = tp.subtract_drift(df, drift).reset_index(drop=True)
    else:
        drift = None

    # filter out 'short' particle tracks
    tdf = df.groupby("particle").filter(
        lambda x: len(x) >= minimum_track_length
    )

    # sort by particle and frame (important for computing track stats)
    tdf = tdf.sort_values(
        by=["particle", "frame"], ascending=True
    ).reset_index(drop=True)

    # compute track stats (motion classification included)
    track_stats = (
        tdf.groupby("particle", group_keys=True)
        .apply(compute_track_quantities)
        .reset_index()
    )

    track_stats.drop(
        columns=[c for c in track_stats.columns if c.startswith("level")],
        inplace=True,
    )

    tdf = tdf.merge(track_stats)

    # compute per-trajectory stats (Rg in pixel unit and motion classification)
    particle_stats = (
        tdf.groupby("particle", group_keys=True)
        .apply(compute_track_stats, p_mobile_alpha=alpha_cutoff)
        .reset_index()
    )

    # 'transform' particle stats back to trajectories
    tdf = tdf.merge(particle_stats, on="particle")

    # figure out which ones are moving
    frac_mobile = particle_stats["motion"].eq("mobile").mean()

    return {"frac_mobile": frac_mobile, "tracks_df": tdf, "drift": drift}


def fit_msd(
    tracks: None,
    dxy=0.065,
    dt=0.01,
    n_pts_to_fit=3,
    separate_immobile=True,
):
    """fit MSDs given napari `Tracks` layer"""
    if tracks is None:
        return None

    if isinstance(tracks, Tracks):
        df = tracks.features
    elif isinstance(tracks, pd.DataFrame):
        df = tracks
    else:
        return None

    results = {}

    if separate_immobile:
        results["stationary_analysis"] = analyze_msd(
            df, dxy, dt, n_pts_to_fit=n_pts_to_fit, motion_type="stationary"
        )
        results["mobile_analysis"] = analyze_msd(
            df, dxy, dt, n_pts_to_fit=n_pts_to_fit, motion_type="mobile"
        )
    else:
        results["all_analysis"] = analyze_msd(
            df, dxy, dt, n_pts_to_fit=n_pts_to_fit
        )

    return results


def analyze_msd(df, dxy, dt, n_pts_to_fit=3, motion_type=None):
    """Compute MSD and effective diffusion constants for a given particle
    subset.
    """
    if motion_type:
        df = df[df["motion"] == motion_type]

    if df.empty:
        return {
            "msd_ta": None,
            "msd_ens": None,
            "D_eff": None,
            "D_eff_std": None,
            "coefs": None,
        }

    # compute time-averaged MSD per particle
    msd_ta = (
        df.groupby("particle", group_keys=True)
        .apply(compute_msd, dxy, dt)
        .reset_index(level=0)
    )

    # compute ensemble MSD
    msd_ens = (
        msd_ta.groupby("lag")["MSD"]
        .agg(["mean", "std", "count"])
        .reset_index(level=0)
    )

    (D_eff, D_eff_sd), (_, _), coefs = fit_msds(
        msd_ens["lag"].values[:n_pts_to_fit],
        msd_ens["mean"].values[:n_pts_to_fit],
        msd_ens["std"].values[:n_pts_to_fit],
    )

    return {
        "msd_ta": msd_ta,
        "msd_ens": msd_ens,
        "D_eff": D_eff,
        "D_eff_sd": D_eff_sd,
        "coefs": coefs,
    }
