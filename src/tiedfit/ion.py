"""
The whole method for one ion: calibrate the rest wavelengths, then fit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .calibrate import Calibration, rest_offsets, shift_rest
from .combine import inverse_variance_mean
from .fit import FitResult, fit_single, fit_tied


@dataclass
class IonFit:
    """
    Single-line and tied fits of the lines of one ion.

    Attributes
    ----------
    singles : list of FitResult
        Single-line fit of each window.
    tied : FitResult
        Tied fit of all windows.
    calibration : Calibration or None
        Rest-wavelength offsets applied before the fits, or None if the
        rest wavelengths were not calibrated.
    windows : list of Window
        Windows as fitted, with any calibration applied to their lines.
    """

    singles: list
    tied: FitResult
    calibration: Calibration | None
    windows: list

    def inverse_variance(self, max_chi2=10.0, max_sigma=200.0):
        """
        Inverse-variance weighted mean of the single-line velocities.

        In each pixel, only the lines whose fits pass the quality cuts of
        :meth:`FitResult.good` are used.

        Returns
        -------
        velocity, sigma : np.ndarray
        """
        v = np.array([r.velocity for r in self.singles])
        s = np.array([r.sigma_velocity for r in self.singles])
        good = np.array([r.good(max_chi2, max_sigma) for r in self.singles])
        return inverse_variance_mean(v, s, good)

    def best_line(self, max_chi2=10.0, max_sigma=200.0):
        """Index of the line with the smallest median uncertainty over the good pixels."""
        med = []
        for r in self.singles:
            g = r.good(max_chi2, max_sigma)
            med.append(np.median(r.sigma_velocity[g]) if g.any() else np.inf)
        return int(np.argmin(med))


def fit_ion(windows, calibrate=True, options=None, n_jobs=1, max_sigma=5.0, min_pixels=200):
    """
    Fit every line of one ion alone and together.

    Each window is first fitted alone. With `calibrate`, the rest
    wavelengths are then calibrated against each other with
    :func:`~tiedfit.calibrate.rest_offsets`, and the single-line fits are
    repeated with the corrected rest wavelengths. Last comes the tied fit.

    The calibration takes its medians over all the pixels given, so pass
    a whole raster rather than a small region where possible. A line that
    cannot be calibrated keeps its rest wavelength, with a warning.

    Parameters
    ----------
    windows : list of Window
        Windows of one ion with the same spatial shape.
    calibrate : bool
        Calibrate the rest wavelengths first. Default True.
    options : FitOptions, optional
    n_jobs : int
        Number of processes. -1 uses every CPU this process may run on.
        Default 1. With more than one, call the fit from under
        `if __name__ == "__main__":` in a script.
    max_sigma, min_pixels
        Passed to :func:`~tiedfit.calibrate.rest_offsets`.

    Returns
    -------
    IonFit
    """
    windows = list(windows)
    singles = [fit_single(w, options, n_jobs) for w in windows]
    calibration = None
    if calibrate:
        calibration = rest_offsets(singles, max_sigma, min_pixels)
        missing = [w.target.wavelength for w, ok in zip(windows, calibration.calibrated) if not ok]
        if missing:
            warnings.warn(
                f"rest wavelengths of {missing} not calibrated: fewer than {min_pixels} pixels "
                f"measured to better than {max_sigma} km/s; their atomic-data rest wavelengths "
                "are used", stacklevel=2)
        if np.any(calibration.offsets != 0):
            windows = shift_rest(windows, calibration.offsets)
            singles = [fit_single(w, options, n_jobs) for w in windows]
    tied = fit_tied(windows, options, n_jobs)
    return IonFit(singles, tied, calibration, windows)
