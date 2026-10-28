"""
Inverse-variance weighted mean of velocities from separate fits.
"""

from __future__ import annotations

import numpy as np


def inverse_variance_mean(velocities, sigmas, mask=None):
    """
    Weighted mean of several velocity measurements, pixel by pixel.

    Each measurement is weighted by one over its variance, and the mean has
    an uncertainty of (sum of 1 / sigma^2)^(-1/2). See e.g. Bevington and
    Robinson, Data Reduction and Error Analysis for the Physical Sciences.

    Parameters
    ----------
    velocities, sigmas : array_like
        Velocities and their one-sigma uncertainties in km/s, shape
        (n_lines, ...).
    mask : array_like of bool, optional
        Measurements to use, same shape. Default: every finite measurement
        with a positive uncertainty.

    Returns
    -------
    velocity, sigma : np.ndarray
        Mean and its uncertainty, shape (...). NaN where no measurement is
        used.
    """
    v = np.asarray(velocities, dtype=float)
    s = np.asarray(sigmas, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        use = np.isfinite(v) & np.isfinite(s) & (s > 0)
        if mask is not None:
            use &= np.asarray(mask, dtype=bool)
        w = np.where(use, 1.0 / np.where(use, s, 1.0) ** 2, 0.0)
        total = w.sum(axis=0)
        mean = np.where(total > 0, (w * np.where(use, v, 0.0)).sum(axis=0) / total, np.nan)
        sigma = np.where(total > 0, total ** -0.5, np.nan)
    return mean, sigma
