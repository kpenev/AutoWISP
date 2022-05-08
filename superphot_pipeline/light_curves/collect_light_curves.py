"""Functions for creating light curves from DR files."""

from string import Formatter
import os.path
import os

from superphot_pipeline.hat.file_parsers import parse_fname_keywords
from superphot_pipeline.hat.header_util import get_jd as get_hat_jd
from superphot_pipeline import DataReductionFile
from .lc_data_io import LCDataIO

class DecodingStringFormatter(Formatter):
    """Add one more conversion type: ``'d'`` that calls decode on the arg."""

    def convert_field(self, value, conversion):
        """If conversion is ``'d'`` -> ``value.decode()`` else pass to parent"""

        if conversion == 'd':
            return value.decode()
        return super().convert_field(value, conversion)

#This is simple enough
#pylint: disable=too-many-locals
def collect_light_curves(dr_filenames,
                         configuration,
                         get_jd=get_hat_jd,
                         *,
                         dr_fname_parser=parse_fname_keywords,
                         optional_header=None,
                         observatory=None,
                         **path_substitutions):
    """
    Add the data from a collection of DR files to LCs, creating LCs if needed.

    Args:
        dr_filenames([str]):    The filenames of the data reduction files to add
            to LCs.

        configuration:    Object with attributes configuring the LC collection
            procedure.

        get_jd:    A function that should return the JD of the middle of the
            exposure for a given FITS header. By default HAT-style headers are
            assumed, which contain a JD keyword which needs to be corrected by
            adding `2.4e6`.

        path_substitutions:    Any substitutions to resolve paths within DR and
            LC files to data to read/write (e.g. versions of various
            componenents).

        dr_fname_parser:    See same name argument to LCDataIO::create().

        optional_header:    See same name argument to LCDataIO::create().

    Returns:
        [(src ID part 1, src ID part 2, ...)];
            The sources for which new lightcurves were created.
    """

    srcid_formatter = DecodingStringFormatter()
    with DataReductionFile(dr_filenames[0], 'r') as first_dr:
        data_io = LCDataIO.create(configuration,
                                  first_dr.parse_hat_source_id,
                                  dr_fname_parser,
                                  get_jd=get_jd,
                                  optional_header=optional_header,
                                  observatory=observatory,
                                  **path_substitutions)
    frame_chunk = data_io.max_dimension_size['frame']
    sources_lc_fnames = [
        (
            source_id,
            srcid_formatter.format(
                configuration.lc_fname,
                source_id
            )
        )
        for source_id in data_io.source_destinations.keys()
    ]

    for dirname in {
            os.path.abspath(os.path.dirname(lc_fname))
            for _, lc_fname in sources_lc_fnames
    }:
        if not os.path.exists(dirname):
            os.makedirs(dirname)

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
#pylint: enable=too-many-locals
