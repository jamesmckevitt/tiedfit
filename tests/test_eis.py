"""Reading Hinode/EIS files; runs only if TIEDFIT_EIS_FILE names a level-1 data file."""

import os

import numpy as np
import pytest

pytest.importorskip("eispac")
FILE = os.environ.get("TIEDFIT_EIS_FILE")
pytestmark = pytest.mark.skipif(not FILE, reason="set TIEDFIT_EIS_FILE to an EIS level-1 file")


def test_read_window_crops_around_the_target():
    from tiedfit import Line
    from tiedfit.eis import read_window

    w = read_window(FILE, [Line(195.119, "Fe XII")], half_width=0.25)
    assert w.data.ndim == 3 and w.wavelength.shape == w.data.shape
    centre = np.nanmedian(w.wavelength, axis=(0, 1))
    assert np.all(np.abs(centre - 195.119) <= 0.25)
    assert np.all(np.isnan(w.data) == np.isnan(w.uncertainty))


def test_line_outside_the_cropped_range_is_refused():
    from tiedfit import Line
    from tiedfit.eis import read_window

    lines = [Line(186.887, "Fe XII"), Line(186.598, "Fe VIII")]
    with pytest.raises(ValueError, match="half_width"):
        read_window(FILE, lines, half_width=0.25)
    w = read_window(FILE, lines, half_width=0.35)
    assert np.nanmin(w.wavelength) < 186.598


def test_too_narrow_a_window_is_refused():
    from tiedfit import Line
    from tiedfit.eis import read_window

    with pytest.raises(ValueError, match="half_width"):
        read_window(FILE, [Line(195.119, "Fe XII")], half_width=0.05)
