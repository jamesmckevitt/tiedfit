"""Line lists from CHIANTI; skipped without fiasco and its database."""

import numpy as np
import pytest

chianti = pytest.importorskip("tiedfit.chianti")
fiasco = pytest.importorskip("fiasco")


def _have_database():
    from pathlib import Path
    return Path(fiasco.defaults["hdf5_dbase_root"]).exists()


pytestmark = pytest.mark.skipif(not _have_database(), reason="no CHIANTI database for fiasco")


def test_ion_name():
    assert chianti.ion_name("fe_12") == "Fe XII"
    assert chianti.ion_name("Fe 12") == "Fe XII"
    assert chianti.ion_name("Fe XII") == "Fe XII"


def test_read_dem():
    log_t, dem = chianti.read_dem("quiet_sun", log_t_min=5.5)
    assert log_t.size == dem.size > 10
    assert np.all(dem[log_t < 5.5] == 0)


def test_window_lines_finds_the_unresolved_fe12_line():
    log_t, dem = chianti.read_dem("quiet_sun", log_t_min=5.5)
    lines = chianti.window_lines("Fe XII", 186.887, 0.3, log_t, dem, blend_ions=["Fe VIII"])
    assert lines[0].wavelength == 186.887 and lines[0].ion == "Fe XII"
    near = [line for line in lines if abs(line.wavelength - 186.854) < 0.002]
    assert len(near) == 1 and near[0].ion == "Fe XII"
    # The 186.854/186.887 ratio depends on density; at 1e9 cm^-3 it is
    # a few tenths.
    assert 0.1 < near[0].strength < 0.6
    assert any(line.ion == "Fe VIII" for line in lines)


def test_ion_names_of_highly_charged_ions():
    assert chianti.ion_name("Zn 31") == "Zn XXXI"
    assert chianti._fiasco_name("Zn XXXI") == "Zn 31"
    assert chianti._fiasco_name("Fe XIV") == "Fe 14"
    assert all(chianti._fiasco_name(chianti.ion_name(f"Fe {n}")) == f"Fe {n}" for n in range(1, 27))


def test_dem_reaching_below_the_ionisation_tables():
    # A flat DEM from log T 3.5, below the CHIANTI ionisation equilibrium
    # tables, is refused unless cut; once cut, the Fe XII lines are found.
    log_t = np.arange(3.5, 7.01, 0.1)
    dem = np.full(log_t.size, 1e21)
    with pytest.raises(ValueError, match="log_t_min"):
        chianti.window_lines("Fe XII", 186.887, 0.1, log_t, dem, blend_ions=[])
    cut = np.where(log_t >= 5.5, dem, 0.0)
    lines = chianti.window_lines("Fe XII", 186.887, 0.1, log_t, cut, blend_ions=[])
    assert any(abs(line.wavelength - 186.854) < 0.002 for line in lines)
