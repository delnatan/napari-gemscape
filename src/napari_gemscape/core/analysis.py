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
    minimum_track_length=2,
    maximum_displacement=4,
    prob_mobile_cutoff=0.5,
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

    # filter out 'short' particle tracks
    tdf = df.groupby("particle").filter(
        lambda x: len(x) >= minimum_track_length
    )

    # sort by particle and frame (important for computing track stats)
    tdf = tdf.sort_values(
        by=["particle", "frame"], ascending=True
    ).reset_index(drop=True)

    # compute track stats: length, sigma, step-wise Rayleigh 1-CDF
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
        .apply(compute_track_stats)
        .reset_index()
    )

    # 'transform' particle stats back to trajectories
    tdf = tdf.merge(particle_stats, on="particle")

    # figure out which ones are moving
    frac_mobile = particle_stats["motion"].eq("mobile").mean()

    return {"frac_mobile": frac_mobile, "tracks_df": tdf}


def fit_msd(
    tracks: None, dxy=0.065, dt=0.01, n_pts_to_fit=3, drift_corr_smooth=-1
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

    if drift_corr_smooth > -1:
        drift = tp.compute_drift(df, smoothing=drift_corr_smooth)
        xdrift = drift["x"].values
        ydrift = drift["y"].values
        df = tp.subtract_drift(df, drift).reset_index(drop=True)
    else:
        xdrift = None
        ydrift = None

    motion_groups = df.groupby("motion")

    if "mobile" in motion_groups.groups:
        mdf = motion_groups.get_group("mobile")
        m_msd = (
            mdf.groupby("particle", group_keys=True)
            .apply(compute_msd, dxy, dt)
            .reset_index(level=0)
        )
        m_msd_ens = (
            m_msd.groupby("lag")["MSD"]
            .agg(["mean", "std", "count"])
            .reset_index(level=0)
        )
        (m_D_eff, m_D_eff_sd), (_, _), mcoefs = fit_msds(
            m_msd_ens["lag"].values[:n_pts_to_fit],
            m_msd_ens["mean"].values[:n_pts_to_fit],
            m_msd_ens["std"].values[:n_pts_to_fit],
        )
    else:
        m_msd = None
        m_msd_ens = None
        m_D_eff, m_D_eff_sd, mcoefs = None, None, None

    if "stationary" in motion_groups.groups:
        sdf = motion_groups.get_group("stationary")
        s_msd = (
            sdf.groupby("particle", group_keys=True)
            .apply(compute_msd, dxy, dt)
            .reset_index(level=0)
        )
        s_msd_ens = (
            s_msd.groupby("lag")["MSD"]
            .agg(["mean", "std", "count"])
            .reset_index(level=0)
        )
        (s_D_eff, s_D_eff_sd), (_, _), scoefs = fit_msds(
            s_msd_ens["lag"].values[:n_pts_to_fit],
            s_msd_ens["mean"].values[:n_pts_to_fit],
            s_msd_ens["std"].values[:n_pts_to_fit],
        )
    else:
        s_msd = None
        s_msd_ens = None
        s_D_eff, s_D_eff_sd, scoefs = None, None, None

    return {
        "m_msd_ta": m_msd,
        "s_msd_ta": s_msd,
        "m_msd_ens": m_msd_ens,
        "s_msd_ens": s_msd_ens,
        "m_D": [m_D_eff, m_D_eff_sd],
        "s_D": [s_D_eff, s_D_eff_sd],
        "mcoefs": mcoefs,
        "scoefs": scoefs,
        "xdrift": xdrift,
        "ydrift": ydrift,
    }
