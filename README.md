# tiedfit

Doppler velocities from several emission lines of one ion, fitted together.

All the lines of one ion are formed at practically the same temperature, so they share one Doppler shift. Fitting them with one velocity measures that shift more precisely than the best line alone.

## Installation

```
pip install tiedfit
```

Optional extras: `tiedfit[eis]` adds [EISPAC](https://github.com/USNavalResearchLaboratory/eispac) for reading Hinode/EIS files, and `tiedfit[chianti]` adds [fiasco](https://fiasco.readthedocs.io) for line lists from CHIANTI. fiasco needs a copy of the CHIANTI database; see its documentation.

## The method

Each spectral window holds a target line and, where needed, nearby lines that blend with it. Each line is a Gaussian on a constant background. By default the lines in a window share one fitted width. Lines of different ions have different thermal widths, but in instruments such as Hinode/EIS the instrumental width is the larger part of the observed width, so one width per window usually describes the lines well enough to fit the target's position. Where it does not, a line other than the target can be given a fixed width, `Line(..., width=0.02)` in Angstrom, or a width fitted for it alone, `Line(..., width="free")`. A free width is best kept for blends that are clearly separated from the target, as a close one can trade width with the target's line.

- In a single-line fit, each window is fitted alone with its own velocity.
- In a tied fit, the windows of one ion are fitted at once with one velocity. Each window keeps its own width, background and line amplitudes.
- Lines of the target's ion, in any window, move with the shared velocity. Blends of other ions move independently, so each gets its own wavelength shift. The shift is limited to `max_blend_shift` (0.05 Angstrom by default) so that a blend stays near its own rest wavelength and cannot move onto the target line, where the two would no longer be distinguishable.
- A line of the target's ion can have its intensity fixed as a ratio of the target's, for example a line too close to the target to be fitted on its own.

The fit is weighted least squares ([`scipy.optimize.curve_fit`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)) with the data uncertainties as weights. The velocity is a parameter of the fit, so its uncertainty comes from the covariance matrix.

A tied fit needs the rest wavelengths of its lines to agree to better than the precision of the velocity. In practice the measured line positions can be offset from one another, by uncertainties in the atomic data, by unresolved blends that move a line's apparent centre, or by errors in the instrument's wavelength calibration across the detector. `tiedfit` therefore measures these offsets from the data. It fits each line alone and takes the line with the smallest median uncertainty as the reference. The offset of each other line is the median difference between its velocity and the reference's, over the pixels where both are measured to better than 5 km/s. It then moves each line's rest wavelength by its offset and repeats the fits.

The velocities are relative to the reference line's rest wavelength. For instruments without an absolute wavelength reference, `zero_along_slit` subtracts the median velocity along the slit at each slit position.

## Example: Hinode/EIS

```python
from tiedfit import Line, fit_ion
from tiedfit.eis import read_windows

# Five Fe XII lines, with the Fe VIII line that blends with 186.887
fe12 = [
    [Line(186.887, "Fe XII"), Line(186.598, "Fe VIII")],
    [Line(192.394, "Fe XII")],
    [Line(193.509, "Fe XII")],
    [Line(195.119, "Fe XII")],
    [Line(196.640, "Fe XII")],
]

if __name__ == "__main__":
    windows = read_windows("eis_20071211_102542.data.h5", fe12, half_width=0.35)
    result = fit_ion(windows, n_jobs=-1)

    result.calibration.offsets     # rest-wavelength offsets in km/s
    result.tied.velocity           # velocity map in km/s, shape (n_y, n_x)
    result.tied.sigma_velocity     # its uncertainty
    good = result.tied.good()      # reduced chi-squared <= 10, uncertainty <= 200 km/s
```

`n_jobs=-1` fits the pixels in parallel on every CPU the script may use. The extra Python processes import the script, so in a script the fit must sit under `if __name__ == "__main__":` as here. This is not needed in a notebook or with the default `n_jobs=1`.

`read_windows` crops each window to `half_width` (in Angstrom, default 0.25) either side of its target, and every listed line must lie inside that range. Fe VIII 186.598 is 0.29 Angstrom from Fe XII 186.887, hence `half_width=0.35` here.

`result.singles` holds the single-line fits, and `result.inverse_variance()` gives the inverse-variance weighted mean of the single-line velocities for comparison.

## Example: other instruments

Build a `Window` for each line from arrays. The data can have any number of leading axes, for example (n_y, n_x) for a raster or (n_t, n_y) for a sit-and-stare series. The wavelength can be one array of shape (n_wavelength,) for every pixel, or have the same shape as the data.

```python
from tiedfit import FitOptions, Line, Window, fit_ion

windows = [
    Window(wavelength_1, counts_1, uncertainty_1, [Line(195.119, "Fe XII")]),
    Window(wavelength_2, counts_2, uncertainty_2, [Line(193.509, "Fe XII")]),
]
options = FitOptions(instrument_fwhm=2.0, max_velocity=100.0)
result = fit_ion(windows, options=options)
```

Wavelengths are in Angstrom. The data can be in any units, but the uncertainties must be in the same units and should include every source of noise, as they set both the weights and the velocity uncertainty. Spectral pixels whose data or uncertainty are not finite are left out of the fit.

`FitOptions` sets the instrumental FWHM in pixels (for the starting width), the limits of the line width, the largest velocity and the largest blend shift.

To fit without the calibration, or with your own rest wavelengths, call `fit_single` and `fit_tied` directly.

## Line lists from CHIANTI

`tiedfit.chianti.window_lines` lists the lines of an ion and of possible blends near a target line, with their intensities relative to the target from a DEM:

```python
from tiedfit import merge_unresolved
from tiedfit.chianti import read_dem, window_lines

log_t, dem = read_dem("quiet_sun", log_t_min=5.5)
lines = window_lines("Fe XII", 186.887, 0.35, log_t, dem, min_strength=0.05)
lines = merge_unresolved(lines, min_separation=0.035)
```

`merge_unresolved` removes lines the instrument cannot separate from a stronger one: blends of other ions within `min_separation` of the target, and the weaker of any two other lines closer than that. Lines with a `ratio` are kept. Half the instrumental FWHM is a reasonable `min_separation`. To keep a close line of the target's ion with its intensity fixed relative to the target instead, give it a ratio, e.g. `dataclasses.replace(line, ratio=line.strength)`, before merging. Use the same `half_width` here as when reading the data.

By default every ion in CHIANTI is searched for blends. Pass `blend_ions` to search only some.

## Citation

If you use `tiedfit` in published work, please cite McKevitt and Matthews, "A multi-line fitting technique to improve the measurement precision of scanning-slit solar EUV spectrographs" (in preparation), and the code: [doi:10.5281/zenodo.23728143](https://doi.org/10.5281/zenodo.23728143).

## Licence

MIT. See `LICENSE`.
