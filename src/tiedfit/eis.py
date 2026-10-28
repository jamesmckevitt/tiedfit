"""
Spectral windows from Hinode/EIS level-1 files, read with EISPAC.

Needs the optional dependency `eispac` (`pip install tiedfit[eis]`). The
data are read in photon counts with EISPAC's uncertainties, which combine
the Poisson noise of the counts with the read noise.
"""

from __future__ import annotations

import numpy as np

from .lines import Line, Window

#: Hinode/EIS instrumental FWHM in Angstrom, about three detector pixels
#: (Ugarte-Urra 2010, EIS Software Note 7).
EIS_FWHM = 0.067


def read_window(filename, lines, half_width=0.25, min_pixels=8):
    """
    One spectral window of an EIS level-1 file as a :class:`~tiedfit.Window`.

    If several windows of the file contain the target line, the one with
    the most spectrum on both sides of it is read. The window is cropped to
    `half_width` either side of the target line, so that strong unrelated
    lines near its edges do not enter the fit. Every listed line must lie
    within the cropped range.
    Missing data (marked by EISPAC with values of -100 or less) are set to
    NaN and left out of the fit.

    Parameters
    ----------
    filename : str or path
        Level-1 HDF5 data file (`eis_YYYYMMDD_HHMMSS.data.h5`), with its
        header file next to it.
    lines : list of Line
        Lines to fit, target first. The target picks the EIS window.
    half_width : float
        Half-width of the fitted range in Angstrom. Default 0.25.
    min_pixels : int
        Fewest spectral pixels the cropped window must have. Default 8.

    Returns
    -------
    Window
        Data and uncertainty in photons, shape (n_y, n_x, n_wavelength),
        with the wavelength of every pixel after EISPAC's corrections.

    Raises
    ------
    ValueError
        If no window of the file contains the target line, the cropped
        window has fewer than `min_pixels` spectral pixels, or a listed line
        lies outside it.
    """
    import eispac

    target = lines[0].wavelength
    info = eispac.read_wininfo(str(filename))
    if info is None:
        raise ValueError(f"cannot read the window information of {filename}")
    margin = np.minimum(target - info.wvl_min, info.wvl_max - target)
    if not np.any(margin >= 0):
        raise ValueError(f"no window of {filename} contains {target} Angstrom")
    cube = eispac.read_cube(str(filename), window=int(info.iwin[np.argmax(margin)]),
                            apply_radcal=False)
    wave = np.asarray(cube.wavelength, dtype=float)
    data = np.asarray(cube.data, dtype=float)
    err = np.asarray(cube.uncertainty.array, dtype=float)
    bad = (data <= -100) | (err <= 0) | ~np.isfinite(err)
    if cube.mask is not None and np.ndim(cube.mask):
        bad |= np.asarray(cube.mask, dtype=bool)
    data = np.where(bad, np.nan, data)
    err = np.where(bad, np.nan, err)

    keep = np.abs(np.nanmedian(wave, axis=(0, 1)) - target) <= half_width
    if keep.sum() < min_pixels:
        raise ValueError(f"only {keep.sum()} spectral pixels within {half_width} Angstrom of "
                         f"{target}, fewer than {min_pixels}; increase `half_width`")
    wave, data, err = wave[..., keep], data[..., keep], err[..., keep]
    lo, hi = np.nanmin(wave), np.nanmax(wave)
    outside = [line.wavelength for line in lines if not lo <= line.wavelength <= hi]
    if outside:
        raise ValueError(f"lines {outside} lie outside the fitted range {lo:.3f}-{hi:.3f} "
                         f"Angstrom of the window at {target}; increase `half_width`")
    return Window(wave, data, err, lines)


def read_windows(filename, line_lists, half_width=0.25, min_pixels=8):
    """
    Several windows of one EIS file, e.g. every line of an ion.

    Parameters
    ----------
    filename : str or path
    line_lists : list of list of Line
        Lines of each window, target first.
    half_width, min_pixels
        As for :func:`read_window`.

    Returns
    -------
    list of Window
    """
    return [read_window(filename, lines, half_width, min_pixels) for lines in line_lists]


def target_only(ion, wavelengths):
    """Line lists with one target line per window and no blends."""
    return [[Line(float(w), ion)] for w in wavelengths]
