"""A general purpose SuperPhot based photometry pipeline."""

from superphot_pipeline.processor import Processor
from superphot_pipeline.data_reduction_file import DataReductionFile
from superphot_pipeline.magnitude_fitting import\
    LinearMagnitudeFit,\
    MasterPhotrefCollector,\
    get_single_photref,\
    read_master_catalogue
