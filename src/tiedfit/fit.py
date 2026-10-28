"""
Single-line and tied fits of spectral windows, pixel by pixel.

Both fits are weighted non-linear least squares
(:func:`scipy.optimize.curve_fit` with the data uncertainties as absolute
weights). The velocity is a parameter of the fit, so its uncertainty is the
square root of its diagonal element of the covariance matrix. A single-line
fit is a tied fit with one window.
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit

from .lines import C_KMS, check_lines
from .model import _evaluate, _resolve, layout, shifts, widths, window_model

FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


@dataclass(frozen=True)
class FitOptions:
    """
    Settings of the fits.

    Parameters
    ----------
    max_velocity : float
        Largest velocity in km/s the fit may return. A fit that ends within
        2% of this limit is rejected, as it has usually locked onto a blend.
        Default 80 km/s, about 0.05 Angstrom at 195 Angstrom.
    instrument_fwhm : float
        Instrumental FWHM in spectral pixels, used for the starting width.
        Default 3.0, as for Hinode/EIS.
    width_bounds : tuple of float
        Lower and upper limits of the Gaussian widths (sigma) in spectral
        pixels, both fitted and fixed. A line's fixed `width` outside them
        raises a ValueError. Default (0.5, 8.0).
    max_blend_shift : float
        Largest shift in Angstrom of a blend of another ion from its rest
        wavelength. Default 0.05.
    max_evaluations : int
        Largest number of model evaluations in one fit. Default 2000.

    Raises
    ------
    ValueError
        If a limit is not positive, or `width_bounds` is not increasing.
    """

    max_velocity: float = 80.0
    instrument_fwhm: float = 3.0
    width_bounds: tuple = (0.5, 8.0)
    max_blend_shift: float = 0.05
    max_evaluations: int = 2000

    def __post_init__(self):
        lo, hi = self.width_bounds
        if not 0 < lo < hi:
            raise ValueError(f"`width_bounds` must be two increasing positive numbers, not {self.width_bounds}")
        for name in ("max_velocity", "instrument_fwhm", "max_blend_shift", "max_evaluations"):
            if not getattr(self, name) > 0:
                raise ValueError(f"`{name}` must be positive, not {getattr(self, name)}")


@dataclass
class FitResult:
    """
    Results of a single-line or tied fit for every pixel.

    Attributes
    ----------
    velocity : np.ndarray
        Fitted velocity in km/s, NaN where the fit failed. Shape of the
        windows' spatial axes.
    sigma_velocity : np.ndarray
        One-sigma uncertainty of `velocity` in km/s.
    chi2 : np.ndarray
        Reduced chi-squared of the fit.
    params : list of np.ndarray
        Parameters of each window (see :mod:`tiedfit.model`), each of shape
        (..., n_parameters).
    lines : list of list of Line
        Lines of each window, as fitted.
    """

    velocity: np.ndarray
    sigma_velocity: np.ndarray
    chi2: np.ndarray
    params: list
    lines: list

    def good(self, max_chi2=10.0, max_sigma=200.0):
        """
        Pixels whose fit passes the quality cuts.

        Parameters
        ----------
        max_chi2 : float
            Largest reduced chi-squared. Default 10.
        max_sigma : float
            Largest velocity uncertainty in km/s. Default 200.

        Returns
        -------
        np.ndarray of bool
            True where the fit succeeded and passes both cuts.
        """
        s = self.sigma_velocity
        with np.errstate(invalid="ignore"):
            return (np.isfinite(self.velocity) & np.isfinite(s) & (s > 0)
                    & (s <= max_sigma) & (self.chi2 <= max_chi2))

    def intensity(self, window=0, line=0):
        """
        Integrated intensity of a line: amplitude times width times sqrt(2 pi).

        In the units of the data times Angstrom.
        """
        w, amp = _resolve(self.lines[window], self.params[window])
        return amp[..., line] * w[..., line] * np.sqrt(2.0 * np.pi)

    def width(self, window=0, line=0):
        """Gaussian width (sigma) of a line in Angstrom; by default the window's."""
        return widths(self.lines[window], self.params[window])[..., line]

    def shift(self, window, line):
        """
        Fitted wavelength shift in Angstrom of a blend of another ion.

        Parameters
        ----------
        window : int
            Window number.
        line : int
            Line number in the window's list of lines.

        Raises
        ------
        ValueError
            If the line is of the target's ion; such lines move with the
            velocity and have no shift of their own.
        """
        lines = self.lines[window]
        if lines[line].ion == lines[0].ion:
            raise ValueError(f"line {line} is of the target's ion and has no shift of its own")
        return shifts(self.lines[window], self.params[window])[..., line]

    def model(self, window, index, wavelength, only=None):
        """
        Fitted model of one window at one pixel.

        Parameters
        ----------
        window : int
            Window number.
        index : tuple of int
            Pixel, e.g. (y, x).
        wavelength : array_like
            Wavelengths in Angstrom.
        only : int, optional
            Background plus this line alone.
        """
        return window_model(self.lines[window], wavelength, self.velocity[index],
                            self.params[window][index], only=only)


def _pixel_size(wavelength):
    """Spectral pixel size in Angstrom: median spacing of the finite wavelengths."""
    x = np.asarray(wavelength, dtype=float)
    d = np.abs(np.diff(x, axis=-1))
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if d.size else np.nan


def _width_bounds(wavelength, options):
    """`options.width_bounds` converted from spectral pixels to Angstrom."""
    pixel = _pixel_size(wavelength)
    return options.width_bounds[0] * pixel, options.width_bounds[1] * pixel


def _starting_point(x, y, e, lines, lay, options):
    """Starting parameters and bounds of one window, as arrays."""
    lo_w, hi_w = _width_bounds(x, options)
    width = max(options.instrument_fwhm * FWHM_TO_SIGMA, 1.0) * _pixel_size(x)
    width = float(np.clip(width, lo_w * 1.001, hi_w * 0.999))
    background = float(np.median(y))
    floor = float(np.median(e))
    p0 = np.zeros(lay.n_parameters)
    lo = np.zeros(lay.n_parameters)
    hi = np.zeros(lay.n_parameters)
    p0[:2], lo[:2], hi[:2] = [width, background], [lo_w, -np.inf], [hi_w, np.inf]
    for line, k in zip(lines, lay.amplitude):
        if k is None:
            continue
        near = np.abs(x - line.wavelength) < 2.5 * width
        peak = y[near].max() if near.any() else y.max()
        p0[k], lo[k], hi[k] = max(float(peak) - background, floor), 0.0, np.inf
    for k in lay.shift:
        if k is not None:
            p0[k], lo[k], hi[k] = 0.0, -options.max_blend_shift, options.max_blend_shift
    for k in lay.width:
        if k:
            p0[k], lo[k], hi[k] = width, lo_w, hi_w
    return p0, lo, hi


def fit_pixel(xs, ys, es, lines_list, options=None):
    """
    Tied fit of one pixel's windows with one shared velocity.

    Parameters
    ----------
    xs, ys, es : list of np.ndarray
        Wavelength in Angstrom, data and uncertainty of each window, each 1-D.
    lines_list : list of list of Line
        Lines of each window, target first.
    options : FitOptions, optional

    Spectral pixels whose data or uncertainty is not finite, or whose
    uncertainty is not positive, are left out of the fit.

    Raises
    ------
    ValueError
        If a list of lines cannot be fitted (see :func:`~tiedfit.lines.check_lines`)
        or a line's fixed width is outside `options.width_bounds`.

    Returns
    -------
    velocity, sigma_velocity, chi2 : float
        NaN if the fit failed.
    params : list of np.ndarray
        Parameters of each window, NaN if the fit failed.
    """
    options = options or FitOptions()
    for xi, lines in zip(xs, lines_list):
        check_lines(lines)
        _check_fixed_widths(lines, xi, options)
    return _fit_pixel(xs, ys, es, lines_list, [layout(lines) for lines in lines_list], options)


def _fit_pixel(xs, ys, es, lines_list, layouts, options):
    """`fit_pixel` with the lines already checked and their layouts worked out."""
    sizes = [lay.n_parameters for lay in layouts]
    failed = (np.nan, np.nan, np.nan, [np.full(n, np.nan) for n in sizes])

    # Spectral pixels without a finite value and a positive uncertainty
    # are left out, and a window needs more pixels than parameters.
    kept = []
    for xi, yi, ei, n in zip(xs, ys, es, sizes):
        with np.errstate(invalid="ignore"):
            use = np.isfinite(xi) & np.isfinite(yi) & np.isfinite(ei) & (ei > 0)
        if use.sum() <= n:
            return failed
        kept.append((xi[use], yi[use], ei[use]))
    xs, ys, es = zip(*kept)
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    e = np.concatenate(es)
    # At least one degree of freedom, counting the velocity.
    if y.size <= 1 + sum(sizes):
        return failed

    v_max = options.max_velocity
    parts = [_starting_point(xi, yi, ei, lines, lay, options)
             for xi, yi, ei, lines, lay in zip(xs, ys, es, lines_list, layouts)]
    if not all(np.isfinite(part[0][0]) for part in parts):
        return failed
    p0 = np.concatenate([[0.0]] + [part[0] for part in parts])
    lo = np.concatenate([[-v_max]] + [part[1] for part in parts])
    hi = np.concatenate([[v_max]] + [part[2] for part in parts])
    starts = list(1 + np.cumsum([0] + sizes[:-1]))
    edges = np.cumsum([0] + [len(xi) for xi in xs])

    def model(_x, *p):
        out = np.empty_like(_x)
        for i, lines in enumerate(lines_list):
            s, n = starts[i], sizes[i]
            out[edges[i]:edges[i + 1]] = _evaluate(
                lines, layouts[i], _x[edges[i]:edges[i + 1]], p[0], p[s:s + n])
        return out

    try:
        with warnings.catch_warnings():
            # A covariance that cannot be estimated comes back as inf, which
            # FitResult.good rejects.
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, pcov = curve_fit(model, x, y, p0=p0, sigma=e, absolute_sigma=True,
                                   bounds=(lo, hi), maxfev=options.max_evaluations)
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return failed
    if not np.all(np.isfinite(popt)) or abs(popt[0]) >= 0.98 * v_max:
        return failed

    chi2 = float(np.sum(((y - model(x, *popt)) / e) ** 2) / max(y.size - len(popt), 1))
    params = [np.asarray(popt[s:s + n], float) for s, n in zip(starts, sizes)]
    return float(popt[0]), float(np.sqrt(max(pcov[0, 0], 0.0))), chi2, params


def _fit_chunk(task):
    """Fit a block of pixels; the unit of work given to each process."""
    xs, ys, es, lines_list, options = task
    n_pix = ys[0].shape[0]
    v = np.full(n_pix, np.nan)
    sv = np.full(n_pix, np.nan)
    chi2 = np.full(n_pix, np.nan)
    layouts = [layout(lines) for lines in lines_list]
    params = [np.full((n_pix, lay.n_parameters), np.nan) for lay in layouts]
    for k in range(n_pix):
        out = _fit_pixel([x[k] for x in xs], [y[k] for y in ys], [e[k] for e in es],
                         lines_list, layouts, options)
        v[k], sv[k], chi2[k] = out[:3]
        for i, p in enumerate(out[3]):
            params[i][k] = p
    return v, sv, chi2, params


def _check_fixed_widths(lines, wavelength, options):
    """Raise ValueError if a fixed line width is outside `options.width_bounds`."""
    fixed = [line for line in lines if line.width not in (None, "free")]
    if not fixed:
        return
    lo, hi = _width_bounds(wavelength, options)
    if not np.isfinite(lo):
        return
    for line in fixed:
        if not lo <= line.width <= hi:
            raise ValueError(
                f"fixed width {line.width} Angstrom of the line at {line.wavelength} is outside "
                f"{lo:.4g}-{hi:.4g} Angstrom ({options.width_bounds} pixels); widths are in Angstrom")


def _n_workers(n_jobs):
    """Number of processes for `n_jobs`: a positive number, or -1 for every CPU this process may use."""
    if n_jobs is None:
        return 1
    if n_jobs == -1:
        try:
            return len(os.sched_getaffinity(0))
        except AttributeError:
            return os.cpu_count() or 1
    if int(n_jobs) != n_jobs or n_jobs < 1:
        raise ValueError(f"`n_jobs` must be a positive integer or -1, not {n_jobs}")
    return int(n_jobs)


def fit_tied(windows, options=None, n_jobs=1):
    """
    Fit several windows of one ion with one shared velocity in every pixel.

    Each window keeps its own width, background and amplitudes. Lines of
    the target ion in every window move with the shared velocity.

    Parameters
    ----------
    windows : list of Window
        Windows of the same ion with the same spatial shape.
    options : FitOptions, optional
    n_jobs : int
        Number of processes. -1 uses every CPU this process may run on.
        Default 1. With more than one, call the fit from under
        `if __name__ == "__main__":` in a script.

    Returns
    -------
    FitResult

    Raises
    ------
    ValueError
        If the windows differ in spatial shape or target ion, a list of
        lines cannot be fitted (see :func:`~tiedfit.lines.check_lines`), or
        a line's fixed width is outside `options.width_bounds`.
    """
    windows = list(windows)
    options = options or FitOptions()
    shape = windows[0].shape
    for w in windows[1:]:
        if w.shape != shape:
            raise ValueError("windows differ in spatial shape")
        if w.ion != windows[0].ion:
            raise ValueError(f"windows are of different ions: {windows[0].ion}, {w.ion}")
    lines_list = [w.lines for w in windows]
    for w in windows:
        check_lines(w.lines)
        _check_fixed_widths(w.lines, w.wavelength, options)

    n_pix = int(np.prod(shape, dtype=int))
    xs = [np.broadcast_to(w.wavelength, w.data.shape).reshape(n_pix, -1) for w in windows]
    ys = [w.data.reshape(n_pix, -1) for w in windows]
    es = [w.uncertainty.reshape(n_pix, -1) for w in windows]

    workers = _n_workers(n_jobs)
    n_chunks = max(1, min(n_pix, 4 * workers))
    bounds = np.linspace(0, n_pix, n_chunks + 1).astype(int)
    tasks = [([x[a:b] for x in xs], [y[a:b] for y in ys], [e[a:b] for e in es], lines_list, options)
             for a, b in zip(bounds[:-1], bounds[1:])]
    if workers == 1:
        parts = [_fit_chunk(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_fit_chunk, tasks))

    v = np.concatenate([p[0] for p in parts]).reshape(shape)
    sv = np.concatenate([p[1] for p in parts]).reshape(shape)
    chi2 = np.concatenate([p[2] for p in parts]).reshape(shape)
    params = [np.concatenate([p[3][i] for p in parts]).reshape(shape + (-1,))
              for i in range(len(windows))]
    return FitResult(v, sv, chi2, params, lines_list)


def fit_single(window, options=None, n_jobs=1):
    """
    Fit one window on its own in every pixel.

    Parameters
    ----------
    window : Window
    options : FitOptions, optional
    n_jobs : int
        Number of processes. -1 uses every CPU this process may run on.
        Default 1. With more than one, call the fit from under
        `if __name__ == "__main__":` in a script.

    Returns
    -------
    FitResult
    """
    return fit_tied([window], options=options, n_jobs=n_jobs)
