"""A collection of utilities for creating and working with light curves."""

from .lc_data_slice import LCDataSlice
from .lc_data_io import LCDataIO
from .apply_correction import\
    apply_parallel_correction,\
    apply_reconstructive_correction_transit,\
    save_correction_statistics,\
    load_correction_statistics
