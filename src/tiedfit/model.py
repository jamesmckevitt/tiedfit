"""
Line profile model of one spectral window.

Each line is a Gaussian on a constant background. Lines share the width
fitted to the window unless they have a `width` of their own. The
parameters of a window are, in order::

    [width, background, amplitudes of the free lines..., blend shifts...,
     free widths...]

with widths in Angstrom, one amplitude for each line without a `ratio`,
one wavelength shift in Angstrom for each line of another ion, and one
width for each line with `width="free"`. The velocity is kept apart, as
the tied fit shares it between windows. :func:`layout` gives the position
of each of these.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .lines import C_KMS


@dataclass(frozen=True)
class Layout:
    """
    Where each line's parameters sit in a window's parameter vector.

    Attributes
    ----------
    n_parameters : int
        Length of the vector, not counting the velocity.
    amplitude : tuple
        For each line, the index of its free amplitude, or None if it has a
        `ratio`.
    shift : tuple
        For each line, the index of its blend shift, or None for lines of
        the target's ion.
    width : tuple
        For each line, the index of its width (0 for the window's width),
        or None if the width is fixed.
    """

    n_parameters: int
    amplitude: tuple
    shift: tuple
    width: tuple


def layout(lines):
    """Parameter layout of a window with these lines, see :class:`Layout`."""
    n_amp = sum(1 for line in lines if line.ratio is None)
    n_shift = sum(1 for line in lines[1:] if line.ion != lines[0].ion)
    n_width = sum(1 for line in lines if line.width == "free")
    amp, shift, width = [], [], []
    i_amp, i_shift, i_width = 2, 2 + n_amp, 2 + n_amp + n_shift
    for line in lines:
        if line.ratio is None:
            amp.append(i_amp)
            i_amp += 1
        else:
            amp.append(None)
        if line.ion != lines[0].ion:
            shift.append(i_shift)
            i_shift += 1
        else:
            shift.append(None)
        if line.width is None:
            width.append(0)
        elif line.width == "free":
            width.append(i_width)
            i_width += 1
        else:
            width.append(None)
    return Layout(2 + n_amp + n_shift + n_width, tuple(amp), tuple(shift), tuple(width))


def _width_and_amplitude(line, j, lay, p):
    """
    Width and amplitude of line `j` from window parameters `p`.

    `p` holds the parameters along its last axis, with any leading axes.
    """
    window_width = p[..., 0]
    k = lay.width[j]
    width = window_width if k == 0 else (p[..., k] if k is not None else float(line.width))
    k = lay.amplitude[j]
    if k is not None:
        amp = p[..., k]
    else:
        # The ratio is of integrated intensities, amplitude times width.
        amp = line.ratio * p[..., lay.amplitude[0]] * window_width / width
    return width, amp


def _resolve(lines, params):
    """Widths and amplitudes of every line, each shape (..., n_lines), NaN where the fit failed."""
    p = np.asarray(params, dtype=float)
    lay = layout(lines)
    pairs = [_width_and_amplitude(line, j, lay, p) for j, line in enumerate(lines)]
    w = np.stack([np.broadcast_to(a, p.shape[:-1]) for a, _ in pairs], axis=-1)
    amp = np.stack([np.broadcast_to(b, p.shape[:-1]) for _, b in pairs], axis=-1)
    bad = _failed(p)[..., None]
    return np.where(bad, np.nan, w), np.where(bad, np.nan, amp)


def _failed(p):
    """Pixels whose fit failed: any parameter not finite."""
    return ~np.all(np.isfinite(p), axis=-1)


def widths(lines, params):
    """
    Gaussian width (sigma) of every line in Angstrom.

    Parameters
    ----------
    lines : list of Line
    params : array_like
        Window parameters along the last axis.

    Returns
    -------
    np.ndarray
        Widths, shape (..., n_lines). NaN where the fit failed, also for
        lines with a fixed width.
    """
    return _resolve(lines, params)[0]


def amplitudes(lines, params):
    """
    Peak amplitude of every line.

    A line with a `ratio` has an integrated intensity (amplitude times
    width) of `ratio` times the target's, so its amplitude is scaled by the
    ratio of the two widths when they differ.

    Parameters
    ----------
    lines : list of Line
        Target first.
    params : array_like
        Window parameters along the last axis.

    Returns
    -------
    np.ndarray
        Amplitudes, shape (..., n_lines), NaN where the fit failed.
    """
    return _resolve(lines, params)[1]


def shifts(lines, params):
    """
    Wavelength shift of every line in Angstrom from its rest wavelength.

    NaN for lines of the target's ion, which move with the velocity instead,
    and where the fit failed.
    """
    p = np.asarray(params, dtype=float)
    out = np.stack([p[..., k] if k is not None else np.full(p.shape[:-1], np.nan)
                    for k in layout(lines).shift], axis=-1)
    return np.where(_failed(p)[..., None], np.nan, out)


def window_model(lines, wavelength, velocity, params, only=None):
    """
    Model spectrum of a window.

    Parameters
    ----------
    lines : list of Line
        Target first.
    wavelength : array_like
        Wavelengths in Angstrom at which to evaluate the model.
    velocity : float
        Doppler velocity of the target's ion in km/s, positive away from
        the observer (redshift).
    params : array_like
        Window parameters, see the module docstring.
    only : int, optional
        If given, the background plus line number `only` alone.

    Returns
    -------
    np.ndarray
        Model, same shape as `wavelength`.
    """
    return _evaluate(lines, layout(lines), np.asarray(wavelength, dtype=float), velocity,
                     params, only)


def _evaluate(lines, lay, x, velocity, params, only=None):
    """`window_model` with the layout already worked out, for the fit's inner loop."""
    p = np.asarray(params, dtype=float)
    y = np.full(x.shape, float(p[1]))
    for j, line in enumerate(lines):
        if only is not None and j != only:
            continue
        width, amp = _width_and_amplitude(line, j, lay, p)
        k = lay.shift[j]
        if k is None:
            centre = line.wavelength * (1.0 + velocity / C_KMS)
        else:
            centre = line.wavelength + params[k]
        y = y + amp * np.exp(-0.5 * ((x - centre) / width) ** 2)
    return y
