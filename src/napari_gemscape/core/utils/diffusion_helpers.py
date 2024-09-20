import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import jnp_zeros


def compute_track_quantities(sdf: pd.DataFrame):
    """computes step quantities
    length, sigma, and CDF from Rayleigh distribution

    this function is meant to be used in `df.apply()`
    """
    step_length = (sdf["x"].diff() ** 2 + sdf["y"].diff() ** 2) ** 0.5
    step_sigma = (sdf["xy_std"] ** 2 + sdf["xy_std"].shift(1) ** 2) ** 0.5
    rcdfval = 1 - np.exp((-(step_length**2)) / (2 * step_sigma**2))

    return pd.DataFrame(
        {
            "frame": sdf["frame"],
            "step_length": step_length,
            "step_sigma": step_sigma,
            "prob_mobile_step": rcdfval,
        }
    )


def compute_track_stats(sdf: pd.DataFrame, prob_mobile_cutoff: float = 0.5):
    """computes radius of gyration, track length, and other things
    that may be used to filter and/or classify a trajectory

    Args:
    sdf (DataFrame.groupby): with columns 'x', 'y', 'frame', and 'particle'

    Usage:
        tracks.groupby("particle").apply(compute_track_stats).reset_index()

    """
    mu_x, mu_y = sdf[["x", "y"]].mean()
    dists_square = (sdf["x"] - mu_x) ** 2 + (sdf["y"] - mu_y) ** 2
    Rg = np.sqrt(dists_square.mean())
    npts = len(sdf)
    # currently, a simple average of the probability is used
    prob_mobile = sdf["prob_mobile_step"].mean()
    movement_class = (
        "mobile" if prob_mobile > prob_mobile_cutoff else "stationary"
    )

    return pd.Series(
        {
            "Rg": Rg,
            "prob_mobile": prob_mobile,
            "track_length": npts,
            "motion": movement_class,
        }
    )


def compute_R(illumination_time: float, frame_interval: float):
    """
    Compute the value of R (blur-motion coefficient) based on the given
    illumination time and frame interval.

    s(t)
    |
    |................<-- illumination time
    |               |
    |               |
    |               |
    |_______________|_______.______ time
                            ^frame interval

    This calculation involves a cumulative shutter function S(t) from a shutter
    function s(t)

    s(t) = 1 (illumination ON)
    s(t) = 0 (illumination OFF)

    before calculating the cumulative shutter function, s(t) must be normalized
    so that ∫s(t)dt = 1

    the cumulative shutter function is
    S(t) = ∫ s(t) dt

    R = 1/T ∫ S(t)⋅(1 - S(t)) dt

    Where T is the frame interval (t goes from 0 to T)

    Parameters:
    - illumination_time (float): The time for which the signal is 1.
    - frame_interval (float): The total time over which the integral is
    computed.

    Returns:
    - float: The computed value of R.
    """

    def R_integrand(t, illumination_time, frame_interval):
        # Compute the integral of the signal s_t over [0, frame_interval]
        Z1, _ = quad(lambda t: 1, 0, illumination_time)
        Z2, _ = quad(lambda t: 0, illumination_time, frame_interval)
        Z = Z1 + Z2

        # Compute the integral of the signal s_t over [0, t]
        if t <= illumination_time:
            S_t = quad(lambda t: 1, 0, t)[0] / Z
        else:
            S_t = (
                quad(lambda t: 1, 0, illumination_time)[0]
                + quad(lambda t: 0, illumination_time, t)[0]
            ) / Z

        return S_t * (1 - S_t)

    # Compute the integral of R_integrand over [0, frame_interval]
    R, _ = quad(
        R_integrand,
        0,
        frame_interval,
        args=(illumination_time, frame_interval),
    )

    return np.round(R / frame_interval, 4)


def compute_msd(sdf: pd.DataFrame, dxy: float, dt: float):
    """compute MSD for single trajectory

    Args:
    sdf (DataFrame): with columns 'x','y','frame' and single 'track_id'.

    Returns:
    DataFrame with 'lag', 'MSD', 'stdMSD', 'n'

    here 'n' contains the number of points used in averaging distances

    Usage:
    tracks.groupby("particle").apply(compute_msd).reset_index(level=0)

    """
    nframes = sdf.shape[0]

    if nframes > 1:
        lags = np.arange(1, nframes)
        nlags = len(lags)
        msdarr = np.zeros(nlags)
        stdarr = np.zeros(nlags)
        npts = np.zeros(nlags, dtype=int)
        frames = sdf["frame"].to_numpy()
        frames_set = set(frames)
        xc = sdf["x"].to_numpy() * dxy
        yc = sdf["y"].to_numpy() * dxy

        for i, lag in enumerate(lags):
            # ensure only correct lag time is used
            # find frames that have the current lag length
            frames_lag_end = frames + lag
            valid_end_frames = np.array(list(frames_set & set(frames_lag_end)))
            valid_start_frames = valid_end_frames - lag
            s1 = np.where(np.isin(frames, valid_start_frames))[0]
            s2 = np.where(np.isin(frames, valid_end_frames))[0]
            # only take distances that correspond to the correct lag
            sqdist = np.square(xc[s2] - xc[s1]) + np.square(yc[s2] - yc[s1])
            msdarr[i] = np.mean(sqdist)
            stdarr[i] = np.std(sqdist)
            npts[i] = len(sqdist)

        return pd.DataFrame(
            {"lag": lags * dt, "MSD": msdarr, "stdMSD": stdarr, "n": npts}
        )
    else:
        return None


def fit_msds(x, y, s, ndim=2):
    """do weighted linear regression

    Since 'y' can be computed from displacements in arbitrary
    dimensions, the 'ndim' parameter needs to be specified. By
    default it is 2.

    Args:
        x (ndarray): independent variables, x
        y (ndarray): observed data, y
        s (ndarray): standard deviation of y
        ndim (int): dimensionality of data, default 2.

    """
    X = np.column_stack([x, np.ones_like(x)])
    A = X.T @ (X / s[:, None])
    b = X.T @ (y / s)
    # coefs[0], slope
    # coefs[1], y-intercept
    coefs = np.linalg.solve(A, b)

    # compute covariances
    y_pred = X @ coefs
    residuals = y - y_pred
    var_residuals = np.sum(residuals**2) / (len(y) - 2)
    iXtX = np.linalg.inv(X.T @ X)
    cov_mat = var_residuals * iXtX
    coef_variances = np.diag(cov_mat)

    # compute diffusion coefficients
    D = coefs[0] / (2 * ndim)
    loc_error = coefs[1]

    # get standard deviations for fit coefficients
    D_std = np.sqrt(coef_variances[0] / (2 * ndim))
    loc_error_std = np.sqrt(coef_variances[1])

    return (D, D_std), (loc_error, loc_error_std), coefs


def constrained_diffusion(Rc, D, t):
    """numerical expression for diffusion in a circle

    Arguments
    =========
    Rc (float)   : radius of confinement
    D (float)    : diffusion coefficient
    t (np.array) : array of time lags
    """

    tau = Rc**2 / D
    m = 5
    MSD = np.zeros_like(t)

    # bessel function derivative first-order, m zeros
    alpha = jnp_zeros(1, m)
    a2 = alpha * alpha

    exp_coeff = -a2[:, None] / tau  # shape (5, 1)
    denom = a2 * (a2 - 1)  # shape (5,)

    sum_terms = np.sum(
        np.exp(exp_coeff * t) * (1 / denom[:, None]), axis=0
    )  # shape (size(t),)

    MSD = Rc**2 * (1 - 8 * sum_terms)

    return MSD


def combine_mean_stdev(group):
    """
    Compute combined mean and standard deviation for grouped data.

    This function calculates the combined mean, standard deviation, and total count
    for a pandas GroupBy object containing pre-computed means, standard deviations,
    and counts for subgroups.

    Parameters:
    ----------
    group : pandas.core.groupby.GroupBy
        A GroupBy object with columns:
        - 'count': number of samples in each subgroup
        - 'mean': mean value of each subgroup
        - 'std': standard deviation of each subgroup

    Returns:
    -------
    pandas.Series
        A Series containing:
        - 'mean': combined mean of all subgroups
        - 'std': combined standard deviation of all subgroups
        - 'count': total count of all samples

    Notes:
    -----
    This function uses the following formulas for combining means and variances:
    - Combined mean: Σ(n_i * x_i) / N
    - Combined variance: (Σ(n_i * s_i^2) + Σ(n_i * x_i^2) - N * x_c^2) / (N - 1)
    where n_i is the count, x_i is the mean, and s_i is the standard deviation of each subgroup,
    N is the total count, and x_c is the combined mean.

    The function assumes that the input standard deviations are sample standard deviations
    (i.e., calculated with N-1 in the denominator).

    Example:
    -------
    >>> df = pd.DataFrame({
    ...     'group': ['A', 'A', 'B', 'B'],
    ...     'count': [10, 15, 20, 25],
    ...     'mean': [5, 7, 6, 8],
    ...     'std': [2, 3, 2.5, 3.5]
    ... })
    >>> result = df.groupby('group').apply(combine_mean_stdev)
    """
    total_count = group["count"].sum()

    # we need to 'undo' the averaging
    x = group["count"] * group["mean"]
    xx = group["std"] ** 2 * (group["count"] - 1) + x**2 / group["count"]

    combined_mean = x.sum() / total_count
    combined_variance = (xx.sum() - x.sum() ** 2 / total_count) / (
        total_count - 1
    )
    combined_stdev = np.sqrt(combined_variance)
    return pd.Series(
        {"mean": combined_mean, "std": combined_stdev, "count": total_count}
    )


def negloglikfn(pars, x, y_obs, y_std):
    Rc, D = pars
    y = constrained_diffusion(Rc, D, x)
    return np.sum(((y - y_obs) / (y_std)) ** 2)


def fit_constrained_diffusion(x, y, sd, Rc0=0.2, D0=0.1):
    p0 = (Rc0, D0)
    optres = minimize(
        negloglikfn,
        p0,
        args=(x, y, sd),
        bounds=((1e-4, 1.25), (1e-4, 1.25)),
    )
    return optres.x[0], optres.x[1]


def batch_fit_constrained_model(df, group_name="source_file", max_lag=0.4):
    dpars = {group_name: [], "Rc": [], "D": [], "success": []}

    # fit for every worm (1 worm in every timelapse)
    for image_id, edf in df.groupby(group_name):
        x_obs = edf["lag"].values
        mask = x_obs <= max_lag
        y_obs = edf["mean"].values[mask]
        y_std = edf["std"].values[mask]
        p0 = np.array((np.sqrt(y_obs.max()), 0.1))

        optres = minimize(
            negloglikfn,
            p0,
            args=(x_obs[mask], y_obs, y_std),
            bounds=((1e-4, 1.2), (1e-4, 1.2)),
        )

        dpars[group_name].append(image_id)
        dpars["Rc"].append(optres.x[0])
        dpars["D"].append(optres.x[1])
        dpars["success"].append(optres.success)

    return pd.DataFrame(dpars)
