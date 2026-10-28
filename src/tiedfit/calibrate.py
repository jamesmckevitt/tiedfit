"""
Relative rest wavelengths of the lines of one ion, measured from the data.

A tied fit needs the rest wavelengths of its lines to agree to better than
the velocity precision, which atomic data often do not. The offset of each
line from a reference line is measured as the median difference of their
single-line velocities over the pixels where both are well measured. Both
lines come from the same exposures, so any drift of the instrument's
wavelength scale is common to them and cancels.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .lines import Window


@dataclass
class Calibration:
    """
    Rest-wavelength offsets of the lines of one ion.

    Attributes
    ----------
    offsets : np.ndarray
        Velocity in km/s of each line relative to the reference line, as
        measured with the rest wavelengths used in the fits. Zero for the
        reference and for lines with too few pixels.
    reference : int
        Index of the reference line, or -1 if no line had enough pixels.
    n_pixels : np.ndarray
        Number of pixels each offset is the median over.
    calibrated : np.ndarray of bool
        Lines whose offset was measured, including the reference.
    """

    offsets: np.ndarray
    reference: int
    n_pixels: np.ndarray
    calibrated: np.ndarray


def rest_offsets(singles, max_sigma=5.0, min_pixels=200):
    """
    Offset of each line's rest wavelength from that of a reference line.

    The reference is the line with the smallest median velocity uncertainty
    over the pixels where it is measured to better than `max_sigma`. The
    offset of each other line is the median of its velocity minus the
    reference's, over the pixels where both are measured to better than
    `max_sigma`.

    Parameters
    ----------
    singles : list of FitResult
        Single-line fits of each line, same spatial shape.
    max_sigma : float
        Velocity uncertainty in km/s a pixel must be better than. Default 5.
    min_pixels : int
        Fewest pixels for a median; lines with fewer are left uncalibrated.
        Default 200.

    Returns
    -------
    Calibration
    """
    n = len(singles)
    ok = []
    for r in singles:
        s = r.sigma_velocity
        with np.errstate(invalid="ignore"):
            ok.append(np.isfinite(r.velocity) & np.isfinite(s) & (s > 0) & (s < max_sigma))
    median_sigma = [float(np.median(r.sigma_velocity[m])) if m.sum() >= min_pixels else np.inf
                    for r, m in zip(singles, ok)]
    offsets = np.zeros(n)
    n_pixels = np.zeros(n, dtype=int)
    calibrated = np.zeros(n, dtype=bool)
    if not np.isfinite(median_sigma).any():
        return Calibration(offsets, -1, n_pixels, calibrated)

    ref = int(np.argmin(median_sigma))
    n_pixels[ref] = int(ok[ref].sum())
    calibrated[ref] = True
    for i in range(n):
        if i == ref:
            continue
        both = ok[i] & ok[ref]
        n_pixels[i] = int(both.sum())
        if n_pixels[i] < min_pixels:
            continue
        offsets[i] = float(np.median(singles[i].velocity[both] - singles[ref].velocity[both]))
        calibrated[i] = True
    return Calibration(offsets, ref, n_pixels, calibrated)


def shift_rest(windows, offsets):
    """
    Windows with the rest wavelengths of the target ion's lines shifted.

    Every line of the target's ion in window i is moved by `offsets[i]`.
    Blends of other ions are left alone, as they fit their own shift.

    Parameters
    ----------
    windows : list of Window
    offsets : array_like
        Velocity in km/s by which to move each window's lines, e.g.
        `Calibration.offsets`.

    Returns
    -------
    list of Window
    """
    out = []
    for w, off in zip(windows, offsets):
        lines = [line.shifted(off) if line.ion == w.ion else line for line in w.lines]
        out.append(Window(w.wavelength, w.data, w.uncertainty, lines))
    return out


def zero_along_slit(velocity, axis=0, min_pixels=50):
    """
    Velocities less their median along the slit at each slit position.

    Hinode/EIS and similar instruments carry no absolute wavelength
    reference, so velocities are often measured relative to the median
    along the slit in each exposure.

    Parameters
    ----------
    velocity : array_like
        Velocity map in km/s.
    axis : int
        Axis along the slit. Default 0, the y axis of an EIS raster.
    min_pixels : int
        Fewest finite pixels for a median; positions with fewer become NaN.
        Default 50.

    Returns
    -------
    velocity, medians : np.ndarray
        Corrected velocities, and the median removed at each position.
    """
    v = np.asarray(velocity, dtype=float)
    n = np.sum(np.isfinite(v), axis=axis, keepdims=True)
    with np.errstate(invalid="ignore"):
        med = np.where(n >= min_pixels, _nanmedian(v, axis), np.nan)
    return v - med, np.squeeze(med, axis=axis)


def _nanmedian(v, axis):
    """Median of the finite values along `axis`, NaN where there are none."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(v, axis=axis, keepdims=True)
