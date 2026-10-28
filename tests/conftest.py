"""Synthetic spectra for the tests."""

import numpy as np
import pytest

from tiedfit import Line, Window
from tiedfit.lines import C_KMS

PIXEL = 0.0223  # Angstrom, as Hinode/EIS


def spectra(rng, lines, amplitudes, velocity, n, width=0.03, background=5.0,
            shifts=None, half_width=0.25, widths=None):
    """
    Poisson spectra of `n` pixels around `lines[0]`.

    Lines of the target's ion are moved by `velocity` in km/s, others by
    `shifts` in Angstrom. All lines have Gaussian width `width` unless
    `widths` gives one per line. The uncertainty is the square root of the counts,
    at least 1.
    """
    centre = lines[0].wavelength
    x = np.arange(centre - half_width, centre + half_width + 1e-9, PIXEL)
    model = np.full(x.shape, background)
    shifts = iter(shifts or [])
    widths = widths or [width] * len(lines)
    for line, a, wd in zip(lines, amplitudes, widths):
        if line.ion == lines[0].ion:
            c = line.wavelength * (1 + velocity / C_KMS)
        else:
            c = line.wavelength + next(shifts)
        model = model + a * np.exp(-0.5 * ((x - c) / wd) ** 2)
    counts = rng.poisson(np.broadcast_to(model, (n, x.size))).astype(float)
    return Window(x, counts, np.sqrt(np.maximum(counts, 1.0)), lines)


@pytest.fixture
def rng():
    return np.random.default_rng(1)


@pytest.fixture
def fe12():
    """Three Fe XII lines with peak counts from bright to faint."""
    return [([Line(195.119, "Fe XII")], 400.0),
            ([Line(193.509, "Fe XII")], 250.0),
            ([Line(192.394, "Fe XII")], 120.0)]
