"""
Line lists and blends predicted with CHIANTI, through fiasco.

Needs the optional dependencies `fiasco` and `astropy`
(`pip install tiedfit[chianti]`) and a copy of the CHIANTI database set up
for fiasco. Intensities are in photons, from the contribution functions
weighted by a differential emission measure (DEM), or at one temperature.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .lines import Line, _from_roman, ion_name

def _fiasco_name(name):
    element, stage = ion_name(name).split()
    return f"{element} {_from_roman(stage)}"


def read_dem(name, log_t_min=None):
    """
    A DEM from the CHIANTI database or a file in the same format.

    Parameters
    ----------
    name : str or path
        CHIANTI DEM name such as "quiet_sun" or "active_region", or the path
        of a file with columns log10 T and log10 DEM, ended by a line
        starting with -1.
    log_t_min : float, optional
        Set the DEM to zero below this log10 T. The CHIANTI DEMs rise
        steeply in the chromosphere, where the optically thin intensities
        are not valid. Default: no cut.

    Returns
    -------
    log_t, dem : np.ndarray
        log10 T in K and the DEM in cm^-5 K^-1.

    Raises
    ------
    FileNotFoundError
        If no such DEM is found.
    """
    import fiasco

    root = Path(fiasco.defaults["ascii_dbase_root"]) / "dem"
    candidates = [Path(name), Path(f"{name}.dem"), root / f"{name}.dem", root / name]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        found = sorted(p.stem for p in root.rglob("*.dem")) if root.is_dir() else []
        raise FileNotFoundError(f"no DEM {name!r}; CHIANTI has {found}")
    log_t, log_dem = [], []
    for row in path.read_text().splitlines():
        parts = row.split()
        if not parts or parts[0].startswith("%"):
            continue
        if parts[0].startswith("-1"):
            break
        try:
            log_t.append(float(parts[0]))
            log_dem.append(float(parts[1]))
        except (ValueError, IndexError):
            continue
    log_t = np.array(log_t)
    dem = 10.0 ** np.array(log_dem)
    if log_t_min is not None:
        dem = np.where(log_t >= log_t_min, dem, 0.0)
    return log_t, dem


def line_intensities(ion, wavelength_range, log_t, dem=None, density=1e9, **ion_kwargs):
    """
    Wavelengths and intensities in photons of an ion's lines in a range.

    Parameters
    ----------
    ion : str
        Ion, e.g. "Fe XII".
    wavelength_range : tuple of float
        Shortest and longest wavelength in Angstrom.
    log_t : float or array_like
        log10 T in K: one temperature, or the temperature grid of `dem`.
    dem : array_like, optional
        DEM in cm^-5 K^-1 on `log_t`. Default: the intensities at the one
        temperature `log_t`.
    density : float
        Electron density in cm^-3. Default 1e9.
    **ion_kwargs
        Passed to :class:`fiasco.Ion`, e.g. `abundance`.

    Raises
    ------
    ValueError
        If `dem` is not given and `log_t` holds more than one temperature.

    Returns
    -------
    wavelength, intensity : np.ndarray
        Wavelength in Angstrom and intensity in photons (arbitrary units,
        the same for every ion with the same DEM), sorted by wavelength.
        Empty if the ion has no lines in the range.
    """
    import astropy.units as u
    import fiasco

    log_t = np.atleast_1d(np.asarray(log_t, dtype=float))
    if dem is None and log_t.size != 1:
        raise ValueError("give one temperature, or a DEM on the `log_t` grid")
    temperature = 10.0 ** log_t * u.K
    obj = fiasco.Ion(_fiasco_name(ion), temperature, **ion_kwargs)
    transitions = obj.transitions
    bound = transitions.is_bound_bound
    wave = np.abs(transitions.wavelength[bound].to_value(u.AA))
    lo, hi = wavelength_range
    inside = (wave >= lo) & (wave <= hi)
    if not inside.any():
        return np.array([]), np.array([])

    g = obj.contribution_function(density * u.cm ** -3)[:, 0, :][:, inside]
    g = g.to_value(u.erg * u.cm ** 3 / u.s)
    if dem is None:
        energy = g[0]
    else:
        dem = np.asarray(dem, dtype=float)
        # Width in K of each temperature bin of the log grid.
        step = np.gradient(log_t) if log_t.size > 1 else np.array([0.1])
        dt = 10.0 ** log_t * (10.0 ** (step / 2) - 10.0 ** (-step / 2))
        weight = dem * dt
        used = weight > 0
        g_used = g[used]
        if not np.all(np.isfinite(g_used)):
            raise ValueError(f"CHIANTI gives no contribution function for {ion} over part of "
                             "the DEM's temperatures; cut the DEM with `log_t_min`")
        energy = np.sum(g_used * weight[used][:, None], axis=0)
    hc = 1.98644586e-8  # erg Angstrom
    photons = energy * wave[inside] / hc
    order = np.argsort(wave[inside])
    return wave[inside][order], photons[order]


def window_lines(ion, wavelength, half_width, log_t, dem=None, density=1e9,
                 min_strength=0.05, blend_ions=None, **ion_kwargs):
    """
    Lines to fit in a window around a target line, with their strengths.

    Lists the lines of the target's ion and of other ions within
    `half_width` of the target that are at least `min_strength` times as
    bright as it. Each line's `strength` is its intensity relative to the
    target, so the result can go to :func:`~tiedfit.merge_unresolved`. To
    fix the intensity of a line of the target's ion relative to the
    target, give it a `ratio`, e.g.
    `dataclasses.replace(line, ratio=line.strength)`.

    By default every ion in CHIANTI is searched for blends; pass
    `blend_ions` to search fewer.

    Parameters
    ----------
    ion : str
        Target ion, e.g. "Fe XII".
    wavelength : float
        Target wavelength in Angstrom. The nearest line of the ion in CHIANTI
        is taken as the target.
    half_width : float
        Half-width of the window in Angstrom.
    log_t, dem, density, **ion_kwargs
        As for :func:`line_intensities`.
    min_strength : float
        Faintest line to list, relative to the target. Default 0.05.
    blend_ions : list of str, optional
        Ions to search for blends. Default: every ion in CHIANTI. Pass an
        empty list for none.

    Returns
    -------
    list of Line
        Target first, at `wavelength`, then the other lines from brightest
        to faintest.

    Raises
    ------
    ValueError
        If the ion has no line within 0.02 Angstrom of `wavelength`.
    """
    import fiasco

    name = ion_name(ion)
    window = (wavelength - half_width, wavelength + half_width)
    wave, inten = line_intensities(name, window, log_t, dem, density, **ion_kwargs)
    if wave.size == 0 or np.min(np.abs(wave - wavelength)) > 0.02:
        raise ValueError(f"{name} has no line in CHIANTI within 0.02 Angstrom of {wavelength}")
    k = int(np.argmin(np.abs(wave - wavelength)))
    target = inten[k]

    found = []
    for j, (w, i) in enumerate(zip(wave, inten)):
        if j != k and i >= min_strength * target:
            found.append(Line(float(w), name, strength=float(i / target)))

    if blend_ions is None:
        blend_ions = [str(n) for n in fiasco.list_ions()]
    for other in blend_ions:
        try:
            other = ion_name(other)
            if other == name:
                continue
            w_o, i_o = line_intensities(other, window, log_t, dem, density, **ion_kwargs)
        except Exception:
            # Ions without the data for a contribution function are skipped.
            continue
        for w, i in zip(w_o, i_o):
            if i >= min_strength * target:
                found.append(Line(float(w), other, strength=float(i / target)))

    found.sort(key=lambda line: -line.strength)
    return [Line(float(wavelength), name, strength=1.0)] + found
