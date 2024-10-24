import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import jnp_zeros
from scipy.stats import chi2


def compute_track_quantities(sdf: pd.DataFrame):
    """computes step quantities

    step_length (magnitude), step_sigma (combined uncertainties),
    and p-value of observing step_length given uncertainties

    this function is meant to be used in `df.apply()`
    """
    dx = sdf["x"].diff()  # displacements in x
    dy = sdf["y"].diff()  # displacements in y
    step_length = (dx**2 + dy**2) ** 0.5
    step_sigma = (sdf["xy_std"] ** 2 + sdf["xy_std"].shift(1) ** 2) ** 0.5
    p_value = np.exp(-(step_length**2) / (2 * step_sigma**2))

    return pd.DataFrame(
        {
            "frame": sdf["frame"],
            "dx": dx,
            "dy": dy,
            "step_length": step_length,
            "step_sigma": step_sigma,
            "p_value": p_value,
        }
    )


def compute_track_stats(sdf: pd.DataFrame, p_mobile_alpha: float = 0.05):
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
    log_pvals = np.log(sdf["p_value"]).sum()
    motion_chi2 = -2 * log_pvals
    ndf = 2 * sdf["p_value"].size
    p_combined = chi2.sf(motion_chi2, ndf)

    movement_class = "mobile" if p_combined < p_mobile_alpha else "stationary"

    return pd.Series(
        {
            "Rg": Rg,
            "log_p_values": log_pvals,
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


def fit_imsd(imsd_df, npts=3, nremoved=2):
    """function to fit iMSD to get 'alpha' and diffusion coefficients

    Note: using default parameteres, it's good idea to use at least
    tracks with >5 coordinates.

    Usage:
        # compute iMSD from tracks, 'data'
        imsd = (
            data.groupby("particle")
            .apply(compute_msd, dxy=0.065, dt=0.010, include_groups=False)
            .reset_index()
        )

        particle_fits = (
            imsd.groupby("particle")
            .apply(fit_imsd, include_groups=False)
            .reset_index()
        )

    """
    x = imsd_df["lag"].values[:-nremoved]
    y = imsd_df["MSD"].values[:-nremoved]
    logx = np.log(x)
    logy = np.log(y)
    alpha, logD = np.polyfit(logx, logy, 1)
    D_anomalous = np.exp(logD)
    slope_3, intercept_3 = np.polyfit(x[:npts], y[:npts], 1)
    D_conventional = slope_3 / 2.0

    return pd.Series(
        {
            "alpha": alpha,
            "D_general": D_anomalous,
            "D_conventional": D_conventional,
        }
    )


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


def negloglikfn_bg(pars, x, y_obs):
    Rc, D, bg = pars
    y = constrained_diffusion(Rc, D, x) + bg
    return np.sum((y - y_obs) ** 2)


def negloglikfn(pars, x, y_obs):
    Rc, D = pars
    y = constrained_diffusion(Rc, D, x)
    return np.sum((y - y_obs) ** 2)


def fit_constrained_diffusion(x, y, Rc0=0.2, D0=0.1, bg0=1e-3, fit_bg=False):
    if fit_bg:
        p0 = (Rc0, D0, bg0)
        optfun = negloglikfn_bg
        parbounds = ((1e-4, 1.25), (1e-4, 1.25), (1e-9, 0.1))
    else:
        p0 = (Rc0, D0)
        optfun = negloglikfn
        parbounds = ((1e-4, 1.25), (1e-4, 1.25))

    optres = minimize(
        optfun,
        p0,
        args=(x, y),
        bounds=parbounds,
        method="L-BFGS-B",
    )
    return optres.x


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


def compute_vacf(particle, max_delta=10):
    """function to compute velocity autocorrelation

    See Steph Weber's 2012 paper on 'Analytical tools to distinguish
    the effect of localization error, confinement, and medium elasticity
    on the velocity autocorrelation function'.

    particles with not enough coordinates are omitted as delta is increased.

    Args:
        particle (pandas.DataFrame): a dataframe grouped by 'particle'.
        max_delta (int): maximum number of 'delta' time resolution.

    Returns:
       Dataframe with 'lag', 'delta', 'normalized ACF'

    Usage:
        vacfs = data.groupby("particle").apply(compute_vacf)

        vacfs is a dataframe with particle as index and columns:
        'tau', 'delta', 'normalized ACF'


    """
    output = []

    N = len(particle)

    delta_max = min(max_delta, N - 2)

    for delta in range(1, delta_max + 1):
        vx = particle["x"].diff(periods=delta).dropna() / delta
        vy = particle["y"].diff(periods=delta).dropna() / delta
        nvelo = len(vx)
        if nvelo < 2:
            continue
        # minimum number for computing full autocorrelation is twice n
        ntwice = 2 * nvelo
        # compute 'optimal' fft size
        npow2 = 2 ** int(np.ceil(np.log2(ntwice)))
        Vx = np.fft.rfft(vx, n=npow2)
        Vy = np.fft.rfft(vy, n=npow2)
        acf_x = np.fft.irfft(Vx * np.conj(Vx), n=ntwice)[:nvelo]
        acf_y = np.fft.irfft(Vy * np.conj(Vy), n=ntwice)[:nvelo]
        acf = acf_x + acf_y
        acf = acf / acf[0]  # normalize so max(acf) = 1 at tau = 0

        df_delta = pd.DataFrame(
            {
                "delta": delta,
                "tau": np.arange(len(acf)),
                "normalized ACF": acf,
            }
        )

        output.append(df_delta)

    if output:
        vacf_df = pd.concat(output, ignore_index=True)
        return vacf_df
    else:
        return None
