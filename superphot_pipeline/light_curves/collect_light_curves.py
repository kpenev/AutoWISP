"""Functions for creating light curves from DR files."""

from superphot_pipeline.hat.file_parsers import parse_fname_keywords
from superphot_pipeline import DataReductionFile
from .lc_data_io import LCDataIO

def collect_light_curves(dr_filenames, configuration, **path_substitutions):
    """
    Add the data from a collection of DR files to LCs, creating LCs if needed.

    Args:
        dr_filenames([str]):    The filenames of the data reduction files to add
            to LCs.

        configuration:    Object with attributes configuring the LC collection
            procedure.

        path_substitutions:    Any substitutions to resolve paths within DR and
            LC files to data to read/write (e.g. versions of various
            componenents).

    Returns:
        [(src ID part 1, src ID part 2, ...)];
            The sources for which new lightcurves were created.
    """

    with DataReductionFile(dr_filenames[0], 'r') as first_dr:
        data_io = LCDataIO.create(configuration,
                                  first_dr.parse_hat_source_id,
                                  parse_fname_keywords,
                                  **path_substitutions)

    config_skipped = list(map(data_io.read, enumerate(dr_filenames)))

    data_io.prepare_for_writing([entry[0] for entry in config_skipped])
    data_io.print_organized_configurations()
