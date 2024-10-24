import numpy as np
import pandas as pd
from numpy.fft import irfft2, rfft2
from scipy.optimize import curve_fit
from tqdm.auto import tqdm, trange


def gaussian_2d(coords, A, x0, y0, sigma):
    (x, y) = coords
    exponent = ((x - x0) ** 2 + (y - y0) ** 2) / sigma**2
    return A * np.exp(-exponent / 2.0).ravel()


def stics(
    imgser: np.ndarray, upper_tau_limit: int, subtract_mean: bool = False
) -> np.ndarray:
    """
    Calculates the spatio-temporal image correlation function given a 3D image
    series.

    Parameters
    ----------
    imgser : np.ndarray
        3D array of image series with shape (t, y, x), where:
            - t: Number of time frames
            - y: Number of rows (height)
            - x: Number of columns (width)
    upper_tau_limit : int
        The maximum lag time (tau) to compute correlations for.

    Returns
    -------
    timecorr : np.ndarray
        3D array of time correlation functions with shape (y, x,
        upper_tau_limit), where:
            - y: Number of rows (height)
            - x: Number of columns (width)
            - tau: Lag time indices from 0 to upper_tau_limit-1
    """
    # Validate input dimensions
    if imgser.ndim != 3:
        raise ValueError(
            f"imgser must be a 3D array, but has shape {imgser.shape}"
        )

    Nt, Ny, Nx = imgser.shape

    # remove mean from image series
    img_means = imgser.mean(axis=(1, 2))

    if subtract_mean:
        mean_removed_img = imgser - img_means[:, None, None]
    else:
        mean_removed_img = imgser

    if upper_tau_limit < 1:
        raise ValueError("upper_tau_limit must be at least 1")

    # Adjust upper_tau_limit if it exceeds the temporal dimension
    if upper_tau_limit > Nt:
        print(
            f"upper_tau_limit ({upper_tau_limit}) exceeds the number of time"
            f" frames ({Nt}). "
            f"Setting upper_tau_limit to {Nt}."
        )
        upper_tau_limit = Nt

    # Preallocate the timecorr array
    # Shape: (y, x, upper_tau_limit)
    timecorr = np.zeros((upper_tau_limit, Ny, Nx), dtype=np.float64)

    # Compute FFT2 for all images over spatial dimensions (y, x)
    # Shape of fft_imgser: (t, y, x), complex numbers
    fft_imgser = rfft2(mean_removed_img, axes=(1, 2))

    # Iterate over each tau using tqdm for the progress bar
    for tau in tqdm(
        range(upper_tau_limit), desc="Calculating time correlation functions"
    ):
        if tau >= Nt:
            break  # No more pairs available for this tau

        num_pairs = Nt - tau
        if num_pairs <= 0:
            # No valid pairs for this tau
            continue

        # Compute the cross-spectrum for all valid pairs at this tau
        # Shape of cross_spectrum: (t - tau, y, x)
        # Broadcasting is handled automatically
        cross_spectrum = fft_imgser[:num_pairs, :, :] * np.conj(
            fft_imgser[tau : tau + num_pairs, :, :]
        )

        mean_product = (
            img_means[:num_pairs] * img_means[tau : tau + num_pairs]
        ).mean()

        # Compute the inverse FFT to get the cross-correlation
        # Take the real part since the input is real
        # Shape of cross_corr: (t - tau, y, x)
        cross_corr = irfft2(cross_spectrum, s=(Ny, Nx), axes=(1, 2))
        mean_cross_corr = cross_corr.mean(axis=0) / mean_product

        # Store the result in the timecorr array
        timecorr[tau, :, :] = mean_cross_corr

    return timecorr


def extract_around_origin(xcorr: np.ndarray, w: int = 10):
    Nlags, Ny, Nx = xcorr.shape

    # take take the smallest dimension for N
    N = min(Ny, Nx)
    Nhalf = N // 2
    # for region size, use the smaller size
    w = min(Nhalf - 1, w)
    indices = np.r_[: w + 1, -w:0]
    yi, xi = np.ix_(indices, indices)

    return xcorr[:, yi, xi], yi, xi


def fit_gaussian_to_xcorr(xcorr: np.ndarray, w: int = 10):
    Nlags, Ny, Nx = xcorr.shape

    # extract subregion from xcorr
    g, yi, xi = extract_around_origin(xcorr, w=w)

    # setup Gaussian coordinates
    Y, X = np.meshgrid(yi, xi, indexing="ij")

    # do gaussian fit to STICS images
    fit_results = {"lag": [], "amplitude": [], "x0": [], "y0": [], "sigma": []}

    for lag in trange(1, Nlags + 1):
        data = g[lag - 1].ravel()
        init_guess = (data[0], 0.0, 0.0, 1.4)
        popt, pcov = curve_fit(gaussian_2d, (X, Y), data, p0=init_guess)
        fit_results["lag"].append(lag)
        fit_results["amplitude"].append(popt[0])
        fit_results["x0"].append(popt[1])
        fit_results["y0"].append(popt[2])
        fit_results["sigma"].append(popt[3])

    return pd.DataFrame(fit_results)
