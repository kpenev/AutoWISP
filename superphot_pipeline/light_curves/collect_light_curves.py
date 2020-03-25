"""Functions for creating light curves from DR files."""

from superphot_pipeline.hat.file_parsers import parse_fname_keywords
from superphot_pipeline import DataReductionFile
from .lc_data_io import LCDataIO

def collect_light_curves(dr_filenames,
                         configuration,
                         dr_fname_parser=parse_fname_keywords,
                         optional_header=None,
                         **path_substitutions):
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

        dr_fname_parser:    See same name argument to LCDataIO::create().

        optional_header:    See same name argument to LCDataIO::create().

    Returns:
        [(src ID part 1, src ID part 2, ...)];
            The sources for which new lightcurves were created.
    """

    with DataReductionFile(dr_filenames[0], 'r') as first_dr:
        data_io = LCDataIO.create(configuration,
                                  first_dr.parse_hat_source_id,
                                  dr_fname_parser,
                                  optional_header=optional_header,
                                  **path_substitutions)
    frame_chunk = data_io.max_dimension_size['frame']
    sources_lc_fnames = [(source_id, configuration.lc_fname_pattern % source_id)
                         for source_id in data_io.source_destinations.keys()]

    num_processed = 0
    while num_processed < len(dr_filenames):
        stop_processing = min(len(dr_filenames), num_processed + frame_chunk)
        data_io.prepare_for_reading()
        config_skipped = list(
            map(
                data_io.read,
                enumerate(dr_filenames[num_processed: stop_processing])
            )
        )

        data_io.prepare_for_writing([entry[0] for entry in config_skipped])
        data_io.print_organized_configurations()

        for write_arg in sources_lc_fnames:
            data_io.write(write_arg)

        num_processed = stop_processing
