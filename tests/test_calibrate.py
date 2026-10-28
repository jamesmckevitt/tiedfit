"""Rest-wavelength calibration, the along-slit zero point and line merging."""

import numpy as np
import pytest

from tiedfit import (Line, fit_ion, inverse_variance_mean, merge_unresolved, rest_offsets,
                     fit_single, zero_along_slit)

from conftest import spectra


def test_rest_offsets_recover_errors_in_the_line_list(rng, fe12):
    # The data have the true rest wavelengths; the line list is off by
    # +3 km/s for 193.509 and -2 km/s for 192.394.
    true = [spectra(rng, lines, [a], velocity=0.0, n=1500) for lines, a in fe12]
    wrong = [true[0],
             true[1].with_lines([true[1].target.shifted(-3.0)]),
             true[2].with_lines([true[2].target.shifted(2.0)])]
    cal = rest_offsets([fit_single(w) for w in wrong], min_pixels=200)
    assert cal.reference == 0
    assert cal.calibrated.all()
    np.testing.assert_allclose(cal.offsets, [0.0, 3.0, -2.0], atol=0.2)


def test_fit_ion_calibrates_before_the_tied_fit(rng, fe12):
    true = [spectra(rng, lines, [a], velocity=0.0, n=1500) for lines, a in fe12]
    wrong = [true[0],
             true[1].with_lines([true[1].target.shifted(-3.0)]),
             true[2].with_lines([true[2].target.shifted(2.0)])]
    plain = fit_ion(wrong, calibrate=False)
    cal = fit_ion(wrong)
    # Without calibration the tied velocity is pulled off the reference
    # line's zero; with it, it sits on it.
    assert abs(np.median(plain.tied.velocity)) > 0.5
    assert abs(np.median(cal.tied.velocity)) < 0.2
    assert np.median(cal.tied.chi2) < np.median(plain.tied.chi2)
    assert cal.best_line() == 0
    v, s = cal.inverse_variance()
    assert np.all(np.isfinite(s[cal.tied.good()]))


def test_zero_along_slit():
    v = np.array([[1.0, 5.0], [3.0, np.nan], [2.0, 7.0]])
    out, med = zero_along_slit(v, axis=0, min_pixels=2)
    np.testing.assert_allclose(med, [2.0, 6.0])
    np.testing.assert_allclose(out[:, 0], [-1.0, 1.0, 0.0])
    out, med = zero_along_slit(v, axis=0, min_pixels=3)
    assert np.isnan(med[1]) and np.all(np.isnan(out[:, 1]))


def test_inverse_variance_mean():
    v, s = inverse_variance_mean([[1.0], [3.0]], [[1.0], [1.0]])
    np.testing.assert_allclose(v, [2.0])
    np.testing.assert_allclose(s, [1 / np.sqrt(2)])
    v, s = inverse_variance_mean([[1.0], [3.0]], [[1.0], [1.0]], mask=[[True], [False]])
    np.testing.assert_allclose(v, [1.0])
    np.testing.assert_allclose(s, [1.0])


def test_merge_unresolved():
    lines = [Line(186.887, "Fe XII"),
             Line(186.854, "Fe XII", strength=0.28),   # 0.033 from the target
             Line(186.870, "S XI", strength=0.01),     # other ion, unresolved
             Line(186.598, "Fe VIII", strength=0.28),
             Line(186.610, "Ca XIV", strength=0.05)]   # unresolved from Fe VIII
    kept = merge_unresolved(lines, 0.035)
    assert [line.wavelength for line in kept] == [186.887, 186.598]


def test_merge_keeps_lines_with_a_ratio():
    lines = [Line(186.887, "Fe XII"), Line(186.854, "Fe XII", ratio=0.28, strength=0.28)]
    assert merge_unresolved(lines, 0.035) == lines


def test_fit_ion_warns_when_a_line_is_not_calibrated(rng, fe12):
    # 100 pixels is fewer than the 200 a median needs.
    windows = [spectra(rng, lines, [a], velocity=0.0, n=100) for lines, a in fe12]
    with pytest.warns(UserWarning, match="not calibrated"):
        fit_ion(windows)
