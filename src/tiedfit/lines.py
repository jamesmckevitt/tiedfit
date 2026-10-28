"""
Emission lines and the spectral windows that hold them.

A window is a stretch of spectrum around one target line, with the data,
their uncertainties and the lines to fit in it. The first line of a window
is its target. Other lines of the target's ion move with the target's
Doppler shift. Lines of other ions are blends, and each gets its own
small wavelength shift.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numbers

import numpy as np

C_KMS = 299792.458

_NUMERALS = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
             (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def _to_roman(n):
    out = ""
    for value, numeral in _NUMERALS:
        while n >= value:
            out += numeral
            n -= value
    return out


def _from_roman(s):
    values = dict((numeral, value) for value, numeral in _NUMERALS if len(numeral) == 1)
    total = 0
    for i, c in enumerate(s):
        v = values[c]
        total += -v if i + 1 < len(s) and values[s[i + 1]] > v else v
    return total


def ion_name(name):
    """
    Ion name in spectroscopic notation, e.g. "Fe XII".

    Accepts "Fe XII", "Fe 12" or "fe_12".

    Raises
    ------
    ValueError
        If `name` is not an element and a charge stage.
    """
    parts = name.replace("_", " ").split()
    if len(parts) != 2 or not parts[0].isalpha():
        raise ValueError(f"not an ion name: {name!r}")
    element, stage = parts
    if not (stage.isdigit() or set(stage.upper()) <= set("IVXLCDM")):
        raise ValueError(f"not an ion name: {name!r}")
    element = element.capitalize()
    if stage.isdigit():
        stage = _to_roman(int(stage))
    return f"{element} {stage.upper()}"


@dataclass(frozen=True)
class Line:
    """
    One emission line.

    Parameters
    ----------
    wavelength : float
        Rest wavelength in Angstrom.
    ion : str
        Ion that emits the line, e.g. "Fe XII" or "Fe 12"; names of this
        form are converted to "Fe XII". Lines with the same `ion` as a
        window's target share its velocity.
    ratio : float, optional
        Integrated intensity as a fixed fraction of the target's. Only for
        lines of the target's ion; for example an unresolved line whose
        ratio to the target is known from atomic data. Default: the
        amplitude is free.
    strength : float, optional
        Expected intensity relative to the target, used only to decide
        which of two unresolved lines to keep in :func:`merge_unresolved`.
    width : float or "free", optional
        Gaussian width (sigma) of the line. Default None: the line has the
        width fitted to the window, which is the target's width. A number
        fixes the width, in Angstrom. The fit's `width_bounds`, which are
        in spectral pixels, are converted to Angstrom with the window's
        pixel size, and a fixed width must lie within them. "free" fits a
        width for this line alone, within the same bounds. Not allowed for
        the target.
    """

    wavelength: float
    ion: str
    ratio: float | None = None
    strength: float | None = None
    width: float | str | None = None

    def __post_init__(self):
        try:
            object.__setattr__(self, "ion", ion_name(self.ion))
        except ValueError:
            pass
        w = self.width
        if w is None or w == "free":
            return
        if (isinstance(w, bool) or not isinstance(w, numbers.Real)
                or not np.isfinite(w) or w <= 0):
            raise ValueError(f"`width` must be None, \"free\" or a positive number, not {w!r}")

    def shifted(self, velocity):
        """Same line with its rest wavelength moved by `velocity` in km/s."""
        return replace(self, wavelength=self.wavelength * (1.0 + velocity / C_KMS))


@dataclass
class Window:
    """
    A spectral window: data around a target line and the lines to fit.

    Parameters
    ----------
    wavelength : array_like
        Wavelength of each spectral pixel in Angstrom, shape (n_wavelength,)
        or the same shape as `data` if it changes from pixel to pixel.
    data : array_like
        Spectra, shape (..., n_wavelength). The leading axes are spatial or
        temporal (e.g. (n_y, n_x) for a raster) and may be absent.
    uncertainty : array_like
        One-sigma uncertainty of `data`, same shape and units.
    lines : list of Line
        Lines to fit, target first.

    Raises
    ------
    ValueError
        If the shapes do not match, there are no lines, a line with a
        `ratio` is not of the target's ion, or the target has a `ratio` or
        a `width`.
    """

    wavelength: np.ndarray
    data: np.ndarray
    uncertainty: np.ndarray
    lines: list = field(default_factory=list)

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=float)
        self.uncertainty = np.asarray(self.uncertainty, dtype=float)
        self.wavelength = np.asarray(self.wavelength, dtype=float)
        self.lines = list(self.lines)
        if self.data.ndim < 1:
            raise ValueError("`data` needs a wavelength axis")
        if self.uncertainty.shape != self.data.shape:
            raise ValueError("`uncertainty` must have the shape of `data`")
        if self.wavelength.shape[-1] != self.data.shape[-1]:
            raise ValueError("`wavelength` and `data` differ in length along the last axis")
        if self.wavelength.ndim != 1 and self.wavelength.shape != self.data.shape:
            raise ValueError("`wavelength` must have shape (n_wavelength,) or the shape of `data`")
        check_lines(self.lines)

    @property
    def target(self):
        """Target line."""
        return self.lines[0]

    @property
    def ion(self):
        """Ion of the target line."""
        return self.lines[0].ion

    @property
    def shape(self):
        """Shape of the spatial or temporal axes."""
        return self.data.shape[:-1]

    def with_lines(self, lines):
        """Same data with a different list of lines."""
        return Window(self.wavelength, self.data, self.uncertainty, lines)


def check_lines(lines):
    """
    Raise ValueError if a window's list of lines cannot be fitted.

    The list needs at least one line. The target, which comes first, cannot
    have a `ratio` or a `width`, and only lines of the target's ion can
    have a `ratio`.
    """
    if not lines:
        raise ValueError("a window needs at least one line")
    target = lines[0]
    if target.ratio is not None:
        raise ValueError("the target line cannot have a `ratio`")
    if target.width is not None:
        raise ValueError("the target line cannot have its own `width`; it has the window's")
    for line in lines[1:]:
        if line.ratio is not None and line.ion != target.ion:
            raise ValueError(f"{line.ion} {line.wavelength} has a `ratio` but is not of the target's ion")


def merge_unresolved(lines, min_separation):
    """
    Lines left after merging those the instrument cannot resolve.

    A blend of another ion closer than `min_separation` to the target is
    dropped, so its light goes into the target's component. Of any other
    lines closer than `min_separation` to each other, only the strongest
    (by `strength`, missing counts as zero) is kept. Fitting two components
    that cannot be told apart makes the fit degenerate. Lines with a
    `ratio` are always kept, as their intensity is tied to the target's.

    Parameters
    ----------
    lines : list of Line
        Target first.
    min_separation : float
        Separation in Angstrom below which two lines are not resolved. Half
        the instrumental FWHM is a reasonable choice.

    Returns
    -------
    list of Line
        Target first, then the kept lines from strongest to weakest.
    """
    target = lines[0]
    kept = [target]
    others = sorted(lines[1:], key=lambda line: -(line.strength or 0.0))
    for line in others:
        if line.ratio is not None:
            kept.append(line)
            continue
        if line.ion != target.ion and abs(line.wavelength - target.wavelength) < min_separation:
            continue
        if any(abs(line.wavelength - k.wavelength) < min_separation for k in kept):
            continue
        kept.append(line)
    return kept
