"""
Doppler velocities from several lines of one ion fitted together.

All the lines of one ion share a Doppler shift, so fitting them with one
velocity measures it more precisely than any one line alone.
"""

from .calibrate import Calibration, rest_offsets, shift_rest, zero_along_slit
from .combine import inverse_variance_mean
from .fit import FitOptions, FitResult, fit_pixel, fit_single, fit_tied
from .ion import IonFit, fit_ion
from .lines import Line, Window, merge_unresolved
from .model import window_model

__version__ = "0.1.0"

__all__ = [
    "Calibration", "FitOptions", "FitResult", "IonFit", "Line", "Window",
    "fit_ion", "fit_pixel", "fit_single", "fit_tied", "inverse_variance_mean",
    "merge_unresolved", "rest_offsets", "shift_rest", "window_model", "zero_along_slit",
]
