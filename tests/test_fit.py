"""Single-line and tied fits of synthetic spectra with known velocities."""

import numpy as np
import pytest

from tiedfit import FitOptions, Line, fit_pixel, fit_single, fit_tied, inverse_variance_mean, window_model
from tiedfit.lines import C_KMS
from tiedfit.model import layout

from conftest import spectra

N = 400


def test_single_fit_recovers_velocity_and_its_uncertainty(rng):
    w = spectra(rng, [Line(195.119, "Fe XII")], [200.0], velocity=12.0, n=N)
    r = fit_single(w)
    ok = r.good()
    assert ok.mean() > 0.99
    v = r.velocity[ok]
    # The mean is unbiased to within 3 standard errors.
    assert abs(v.mean() - 12.0) < 3 * v.std() / np.sqrt(ok.sum())
    # The formal uncertainty matches the scatter to within 10%, about three
    # times the standard error of a standard deviation from 400 values.
    assert abs(np.median(r.sigma_velocity[ok]) / v.std() - 1) < 0.1


def test_tied_fit_matches_inverse_variance_mean_for_bright_lines(rng, fe12):
    windows = [spectra(rng, lines, [a], velocity=-8.0, n=N) for lines, a in fe12]
    singles = [fit_single(w) for w in windows]
    tied = fit_tied(windows)
    v = np.array([s.velocity for s in singles])
    s = np.array([s.sigma_velocity for s in singles])
    _, s_ivw = inverse_variance_mean(v, s)
    ok = tied.good() & np.isfinite(s_ivw)
    # Every line is well measured alone, so tying gains nothing over the
    # weighted mean; the two agree to within a few percent.
    assert abs(np.median(tied.sigma_velocity[ok]) / np.median(s_ivw[ok]) - 1) < 0.03
    # And it beats the best single line.
    assert np.median(tied.sigma_velocity[ok]) < 0.8 * np.median(singles[0].sigma_velocity[ok])
    vt = tied.velocity[ok]
    assert abs(vt.mean() + 8.0) < 3 * vt.std() / np.sqrt(ok.sum())


def test_line_with_fixed_ratio(rng):
    lines = [Line(186.887, "Fe XII"), Line(186.854, "Fe XII", ratio=0.3)]
    w = spectra(rng, lines, [150.0, 45.0], velocity=5.0, n=N)
    assert layout(lines).n_parameters == 3
    r = fit_single(w)
    ok = r.good()
    v = r.velocity[ok]
    assert abs(v.mean() - 5.0) < 3 * v.std() / np.sqrt(ok.sum())
    # The fixed line takes 30% of the target's amplitude.
    assert np.allclose(r.intensity(line=1)[ok], 0.3 * r.intensity(line=0)[ok])


def test_blend_of_another_ion_gets_its_own_shift(rng):
    lines = [Line(195.119, "Fe XII"), Line(195.25, "Fe X")]
    w = spectra(rng, lines, [300.0, 150.0], velocity=10.0, n=N, shifts=[0.01])
    r = fit_single(w)
    ok = r.good()
    shift = r.shift(0, 1)[ok]
    assert abs(np.median(shift) - 0.01) < 0.002
    v = r.velocity[ok]
    assert abs(v.mean() - 10.0) < 3 * v.std() / np.sqrt(ok.sum())


def test_processes_give_the_same_result(rng, fe12):
    windows = [spectra(rng, lines, [a], velocity=3.0, n=60) for lines, a in fe12]
    one = fit_tied(windows, n_jobs=1)
    two = fit_tied(windows, n_jobs=2)
    np.testing.assert_array_equal(one.velocity, two.velocity)
    np.testing.assert_array_equal(one.sigma_velocity, two.sigma_velocity)


def test_missing_spectral_pixels_are_left_out(rng):
    w = spectra(rng, [Line(195.119, "Fe XII")], [200.0], velocity=0.0, n=3)
    w.data[0, :3] = np.nan
    w.data[1, :] = np.nan
    r = fit_single(w)
    assert np.isfinite(r.velocity[0])
    assert np.isnan(r.velocity[1])
    assert np.isfinite(r.velocity[2])


def test_fit_at_the_velocity_limit_is_rejected(rng):
    w = spectra(rng, [Line(195.119, "Fe XII")], [300.0], velocity=40.0, n=20)
    r = fit_single(w, FitOptions(max_velocity=20.0))
    assert np.all(np.isnan(r.velocity))


def test_raster_shape_is_kept(rng):
    w = spectra(rng, [Line(195.119, "Fe XII")], [200.0], velocity=0.0, n=12)
    w.data = w.data.reshape(3, 4, -1)
    w.uncertainty = w.uncertainty.reshape(3, 4, -1)
    r = fit_single(w)
    assert r.velocity.shape == (3, 4)
    assert r.params[0].shape == (3, 4, 3)


def test_blend_with_its_own_fixed_width(rng):
    # A narrow cool blend next to a broad target: with the blend's width
    # fixed at its true value the target's width and velocity are recovered.
    lines = [Line(195.119, "Fe XII"), Line(195.25, "Fe VIII", width=0.02)]
    w = spectra(rng, lines, [300.0, 200.0], velocity=6.0, n=N, shifts=[0.0], widths=[0.035, 0.02])
    r = fit_single(w)
    ok = r.good()
    assert abs(np.median(r.width()[ok]) - 0.035) < 0.001
    assert np.all(r.width(line=1)[ok] == 0.02)
    v = r.velocity[ok]
    assert abs(v.mean() - 6.0) < 3 * v.std() / np.sqrt(ok.sum())


def test_blend_with_a_free_width(rng):
    lines = [Line(195.119, "Fe XII"), Line(195.25, "Fe VIII", width="free")]
    w = spectra(rng, lines, [300.0, 300.0], velocity=-4.0, n=N, shifts=[0.0], widths=[0.035, 0.022])
    r = fit_single(w)
    ok = r.good()
    assert abs(np.median(r.width()[ok]) - 0.035) < 0.001
    assert abs(np.median(r.width(line=1)[ok]) - 0.022) < 0.001
    assert r.params[0].shape[-1] == 6  # width, background, 2 amplitudes, shift, blend width
    v = r.velocity[ok]
    assert abs(v.mean() + 4.0) < 3 * v.std() / np.sqrt(ok.sum())


def test_target_cannot_have_its_own_width():
    from tiedfit import Window
    with pytest.raises(ValueError):
        Window(np.arange(5.0), np.ones(5), np.ones(5), [Line(2.0, "Fe XII", width=0.1)])
    with pytest.raises(ValueError):
        Line(195.0, "Fe XII", width=-1.0)
    with pytest.raises(ValueError):
        Line(195.0, "Fe XII", width="fixed")


def test_ratio_is_an_intensity_ratio_when_widths_differ(rng):
    # The fixed line is narrower than the target; its intensity, not its
    # amplitude, is 30% of the target's.
    lines = [Line(195.119, "Fe XII"), Line(195.30, "Fe XII", ratio=0.3, width=0.02)]
    w = spectra(rng, lines, [200.0, 200.0 * 0.3 * 0.035 / 0.02], velocity=2.0, n=N,
                widths=[0.035, 0.02])
    r = fit_single(w)
    ok = r.good()
    # Both intensities, and the fit's chi-squared, must match the input.
    true = 200.0 * 0.035 * np.sqrt(2 * np.pi)
    assert abs(np.median(r.intensity(line=0)[ok]) / true - 1) < 0.02
    assert abs(np.median(r.intensity(line=1)[ok]) / (0.3 * true) - 1) < 0.02
    assert np.median(r.chi2[ok]) < 1.3
    v = r.velocity[ok]
    assert abs(v.mean() - 2.0) < 3 * v.std() / np.sqrt(ok.sum())
    assert abs(np.median(r.width()[ok]) - 0.035) < 0.001


def test_failed_pixels_have_no_widths_or_shifts(rng):
    lines = [Line(195.119, "Fe XII"), Line(195.25, "Fe X", width=0.02)]
    w = spectra(rng, lines, [200.0, 100.0], velocity=0.0, n=2, shifts=[0.0])
    w.data[1] = np.nan
    r = fit_single(w)
    assert np.isfinite(r.width(line=1)[0]) and np.isnan(r.width(line=1)[1])
    assert np.isnan(r.shift(0, 1)[1]) and np.isnan(r.intensity(line=1)[1])
    with pytest.raises(ValueError):
        r.shift(0, 0)


def test_fixed_width_outside_the_bounds_is_refused(rng):
    # 1.3 Angstrom is about 58 EIS pixels, beyond the 8-pixel upper bound;
    # probably meant as pixels.
    lines = [Line(195.119, "Fe XII"), Line(195.25, "Fe X", width=1.3)]
    w = spectra(rng, lines[:1], [200.0], velocity=0.0, n=2).with_lines(lines)
    with pytest.raises(ValueError, match="Angstrom"):
        fit_single(w)
    with pytest.raises(ValueError, match="Angstrom"):
        fit_pixel([w.wavelength], [w.data[0]], [w.uncertainty[0]], [lines])
    with pytest.raises(ValueError):
        Line(195.0, "Fe XII", width=True)
    with pytest.raises(ValueError):
        Line(195.0, "Fe XII", width=[0.02])


def test_tied_fit_with_free_widths_in_two_windows(rng):
    # Free-width blends in two windows check that each window's parameters
    # are read from the right place in the joint parameter vector.
    a = [Line(195.119, "Fe XII"), Line(195.26, "Fe VIII", width="free")]
    b = [Line(193.509, "Fe XII"), Line(193.37, "Fe VIII", width="free")]
    wa = spectra(rng, a, [300.0, 150.0], velocity=5.0, n=N, shifts=[0.005], widths=[0.035, 0.022])
    wb = spectra(rng, b, [200.0, 150.0], velocity=5.0, n=N, shifts=[-0.005], widths=[0.035, 0.025])
    r = fit_tied([wa, wb])
    ok = r.good()
    v = r.velocity[ok]
    assert abs(v.mean() - 5.0) < 3 * v.std() / np.sqrt(ok.sum())
    assert abs(np.median(r.width(0, 1)[ok]) - 0.022) < 0.001
    assert abs(np.median(r.width(1, 1)[ok]) - 0.025) < 0.001
    assert abs(np.median(r.shift(0, 1)[ok]) - 0.005) < 0.001
    assert abs(np.median(r.shift(1, 1)[ok]) + 0.005) < 0.001


def test_reported_widths_and_intensities_rebuild_the_fitted_model(rng):
    # The model the fit uses and the widths and intensities FitResult
    # reports must agree, for shared, fixed and free widths and a ratio.
    lines = [Line(195.119, "Fe XII"), Line(195.30, "Fe XII", ratio=0.3, width=0.02),
             Line(194.95, "Fe VIII", width="free")]
    w = spectra(rng, lines, [200.0, 105.0, 120.0], velocity=3.0, n=20, shifts=[0.0],
                widths=[0.035, 0.02, 0.025])
    r = fit_single(w)
    k = int(np.flatnonzero(r.good())[0])
    x = w.wavelength
    rebuilt = np.full(x.shape, r.params[0][k, 1])
    doppler = 1 + r.velocity[k] / C_KMS
    centres = [195.119 * doppler, 195.30 * doppler, 194.95 + r.shift(0, 2)[k]]
    for j, c in enumerate(centres):
        sig = r.width(0, j)[k]
        amp = r.intensity(0, j)[k] / (sig * np.sqrt(2 * np.pi))
        rebuilt += amp * np.exp(-0.5 * ((x - c) / sig) ** 2)
    np.testing.assert_allclose(r.model(0, k, x), rebuilt, rtol=1e-10)
    np.testing.assert_allclose(window_model(lines, x, r.velocity[k], r.params[0][k]), rebuilt,
                               rtol=1e-10)


def test_decreasing_wavelengths(rng):
    w = spectra(rng, [Line(195.119, "Fe XII")], [200.0], velocity=7.0, n=N)
    w.wavelength = w.wavelength[::-1].copy()
    w.data = w.data[:, ::-1].copy()
    w.uncertainty = w.uncertainty[:, ::-1].copy()
    r = fit_single(w)
    ok = r.good()
    assert ok.mean() > 0.99
    v = r.velocity[ok]
    assert abs(v.mean() - 7.0) < 3 * v.std() / np.sqrt(ok.sum())


def test_fit_pixel_checks_its_lines(rng):
    w = spectra(rng, [Line(195.119, "Fe XII")], [200.0], velocity=0.0, n=1)
    for lines in ([Line(195.119, "Fe XII", width=0.03)],
                  [Line(195.119, "Fe XII", ratio=0.3)],
                  [Line(195.119, "Fe XII"), Line(195.3, "Fe X", ratio=0.3)]):
        with pytest.raises(ValueError):
            fit_pixel([w.wavelength], [w.data[0]], [w.uncertainty[0]], [lines])


def test_pixel_without_wavelengths_fails_alone(rng):
    # One pixel's wavelengths are missing; with a fixed-width blend in the
    # window that pixel fails and the others are fitted.
    lines = [Line(195.119, "Fe XII"), Line(195.25, "Fe X", width=0.02)]
    w = spectra(rng, lines, [200.0, 100.0], velocity=0.0, n=4, shifts=[0.0], widths=[0.03, 0.02])
    w.wavelength = np.broadcast_to(w.wavelength, w.data.shape).copy()
    w.wavelength[2] = np.nan
    r = fit_single(w)
    assert np.isnan(r.velocity[2])
    assert np.all(np.isfinite(r.velocity[[0, 1, 3]]))


def test_failed_fits_report_nan_everywhere():
    from tiedfit.model import amplitudes, shifts, widths
    lines = [Line(195.119, "Fe XII"), Line(195.25, "Fe X", width=0.02),
             Line(195.0, "Fe XII", ratio=0.2)]
    p = np.array([[0.03, 5.0, 100.0, 50.0, 0.001], [np.nan] * 5])  # width, background, 2 amplitudes, shift
    for f in (amplitudes, shifts, widths):
        out = f(lines, p)
        assert np.all(np.isnan(out[1]))
    assert np.all(np.isfinite(widths(lines, p)[0]))


def test_lines_changed_after_building_the_window_are_checked(rng):
    from dataclasses import replace
    w = spectra(rng, [Line(195.119, "Fe XII")], [200.0], velocity=0.0, n=2)
    w.lines[0] = replace(w.lines[0], ratio=0.3)
    with pytest.raises(ValueError):
        fit_single(w)


def test_unhashable_ratio_is_accepted(rng):
    lines = [Line(186.887, "Fe XII"), Line(186.854, "Fe XII", ratio=np.array(0.3))]
    w = spectra(rng, lines, [150.0, 45.0], velocity=0.0, n=5)
    assert np.isfinite(fit_single(w).velocity).all()


def test_options_are_checked():
    for bad in (dict(max_blend_shift=0.0), dict(width_bounds=(2.0, 2.0)), dict(max_velocity=-1.0),
                dict(width_bounds=(0.0, 3.0))):
        with pytest.raises(ValueError):
            FitOptions(**bad)


def test_fit_without_degrees_of_freedom_fails(rng):
    # Target only: width, background, amplitude and velocity. Four spectral
    # pixels leave no degree of freedom and the fit fails; five are enough.
    w = spectra(rng, [Line(195.119, "Fe XII")], [200.0], velocity=0.0, n=2)
    centre = int(np.argmin(np.abs(w.wavelength - 195.119)))
    for row, n in ((0, 4), (1, 5)):
        lost = np.ones(w.wavelength.size, bool)
        lost[centre - 2:centre - 2 + n] = False
        w.data[row, lost] = np.nan
    r = fit_single(w)
    assert np.isnan(r.velocity[0]) and np.isnan(r.chi2[0])
    assert np.isfinite(r.velocity[1])


def test_n_jobs_is_checked(rng):
    w = spectra(rng, [Line(195.119, "Fe XII")], [200.0], velocity=0.0, n=4)
    with pytest.raises(ValueError):
        fit_single(w, n_jobs=0)
    with pytest.raises(ValueError):
        fit_single(w, n_jobs=-2)
    assert np.isfinite(fit_single(w, n_jobs=-1).velocity).all()


def test_ion_names_in_either_notation_are_the_same_ion():
    assert Line(195.0, "Fe 12").ion == "Fe XII"
    assert Line(195.0, "fe_12").ion == "Fe XII"
    assert Line(195.0, "blend").ion == "blend"
    lines = [Line(195.119, "Fe XII"), Line(195.18, "Fe 12")]
    assert layout(lines).shift == (None, None)


def test_wavelength_must_not_broadcast_along_the_wrong_axis():
    from tiedfit import Window
    data = np.ones((4, 4, 10))
    with pytest.raises(ValueError):
        Window(np.ones((4, 10)), data, data, [Line(195.0, "Fe XII")])
    Window(np.arange(10.0), data, data, [Line(195.0, "Fe XII")])
    Window(np.ones((4, 4, 10)), data, data, [Line(195.0, "Fe XII")])
