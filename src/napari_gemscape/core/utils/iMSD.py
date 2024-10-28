import numpy as np
import pandas as pd
from numpy.fft import irfftn, rfftn
from scipy.optimize import curve_fit
from tqdm.auto import trange


def gaussian_2d(coords, A, x0, y0, var):
    (x, y) = coords
    exponent = ((x - x0) ** 2 + (y - y0) ** 2) / var
    return A * np.exp(-exponent / 2.0).ravel()


def compute_spatiotemporal_correlation(
    image_stack, max_tau, subtract_mean=False
):
    """
    Compute the spatiotemporal correlation function g(ξ, η, τ) using FFT.

    Parameters:
        image_stack (ndarray): 3D array with dimensions (T, H, W), where T is
                               time, and H, W are the spatial dimensions of the
                               images.
        max_tau (int): Maximum temporal lag τ to consider.

    Returns:
        ndarray: 4D array containing g(ξ, η, τ) values with dimensions
        (max_tau, H, W).

    """
    T, H, W = image_stack.shape

    # Mean intensity for each time frame
    mean_intensity = np.mean(image_stack, axis=(1, 2), keepdims=True)

    if subtract_mean:
        # Subtract mean to make the calculation zero-mean (optional but can help
        # with stability)
        zero_mean_stack = image_stack - mean_intensity

        # Fourier transform of the entire stack
        fft_stack = rfftn(zero_mean_stack, axes=(1, 2))
    else:
        fft_stack = rfftn(image_stack, axes=(1, 2))

    # Allocate space for the correlation function
    g = np.zeros((max_tau, H, W))

    for tau in range(1, max_tau + 1):
        # Shift the stack by tau frames to compute correlation
        shifted_fft = np.roll(fft_stack, -tau, axis=0)[: T - tau]

        # Compute cross-correlation using inverse FFT of the product
        cross_corr = irfftn(
            fft_stack[: T - tau] * np.conj(shifted_fft),
            s=(H, W),
            axes=(1, 2),
        )

        # Average over time to get <I(x, y, t) * I(x+ξ, y+η, t+τ)>
        avg_cross_corr = np.mean(cross_corr, axis=0)

        # Compute the denominator (mean intensity squared, averaged over time)
        avg_intensity_sq = np.mean(mean_intensity[: T - tau] ** 2, axis=0)

        # Compute g(ξ, η, τ)
        g[tau - 1] = (avg_cross_corr / avg_intensity_sq) - 1

    return g


def extract_around_origin(xcorr: np.ndarray, w: int = 10, shift_origin=True):
    Nlags, Ny, Nx = xcorr.shape

    # take take the smallest dimension for N
    N = min(Ny, Nx)
    Nhalf = N // 2
    # for region size, use the smaller size
    w = min(Nhalf - 1, w)

    indices = np.r_[: w + 1, -w:0]

    if shift_origin:
        indices = np.roll(indices, w)

    yi, xi = np.ix_(indices, indices)

    return xcorr[:, yi, xi], yi, xi


def fit_gaussian_to_xcorr(
    xcorr: np.ndarray, w: int = 10, init_var: float = 2.0
):
    """
    Fits 2D Gaussian to cross correlated image series

    Parameters:
    - xcorr, np.ndarray: the output from running STICS
    - w, int: window 'radius' to crop around origin
    - init_var, float: initial guess for variance

    Returns:
    pandas.DataFrame with fit results

    """
    Nlags, Ny, Nx = xcorr.shape

    # extract subregion from xcorr
    g, yi, xi = extract_around_origin(xcorr, w=w)

    # setup Gaussian coordinates
    Y, X = np.meshgrid(yi, xi, indexing="ij")

    # do gaussian fit to STICS images
    fit_results = {
        "lag": [],
        "amplitude": [],
        "x0": [],
        "y0": [],
        "variance": [],
    }

    for lag in trange(1, Nlags + 1):
        data = g[lag - 1].ravel()
        init_guess = (data[0], 0.0, 0.0, init_var)
        try:
            popt, pcov = curve_fit(
                gaussian_2d,
                (X, Y),
                data,
                p0=init_guess,
                bounds=((-np.inf, -w, -w, 0.0), (np.inf, w, w, np.inf)),
            )
            fit_results["lag"].append(lag)
            fit_results["amplitude"].append(popt[0])
            fit_results["x0"].append(popt[1])
            fit_results["y0"].append(popt[2])
            fit_results["variance"].append(popt[3])

        except RuntimeError:
            continue

    return pd.DataFrame(fit_results)


def create_tiled_image(arr: np.ndarray, n_cols: int) -> np.ndarray:
    """
    Tile a 3D NumPy array of shape (n_images, height, width) into a single 2D
    image.

    Parameters:
    - arr: np.ndarray with shape (n_images, height, width)
    - n_cols: Number of columns in the tiled image

    Returns:
    - Tiled 2D NumPy array
    """
    n_images, height, width = arr.shape
    n_rows = (
        n_images + n_cols - 1
    ) // n_cols  # Ceiling division to get number of rows

    # Calculate the total number of tiles needed and pad if necessary
    total_tiles = n_rows * n_cols
    pad_width = total_tiles - n_images
    if pad_width > 0:
        padding = np.zeros((pad_width, height, width), dtype=arr.dtype)
        arr = np.concatenate([arr, padding], axis=0)

    # Reshape and transpose to arrange tiles in grid
    tiled = arr.reshape(n_rows, n_cols, height, width)
    tiled = tiled.transpose(0, 2, 1, 3).reshape(
        n_rows * height, n_cols * width
    )

    return tiled
