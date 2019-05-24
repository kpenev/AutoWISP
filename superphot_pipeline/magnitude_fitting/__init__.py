"""Modules implementing magnitude fitting."""

from superphot_pipeline.magnitude_fitting.linear import LinearMagnitudeFit
from superphot_pipeline.magnitude_fitting.master_photref_collector import\
    MasterPhotrefCollector
from superphot_pipeline.magnitude_fitting.util import\
    iterative_refit,\
    read_master_catalogue,\
    get_master_photref,\
    get_single_photref
