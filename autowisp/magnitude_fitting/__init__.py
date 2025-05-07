"""Modules implementing magnitude fitting."""

from autowisp.magnitude_fitting.linear import LinearMagnitudeFit
from autowisp.magnitude_fitting.master_photref_collector_zarr import (
    MasterPhotrefCollector,
)
from autowisp.magnitude_fitting.iterative_refit import (
    iterative_refit,
    single_iteration,
)
from autowisp.magnitude_fitting.util import get_master_photref
