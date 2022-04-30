#!/usr/bin/env python3

"""Detect stars within calibrated image(s)."""

from superphot_pipeline.processing_steps.manual_util import get_cmdline_parser
from superphot_pipeline.image_utilities import get_fits_fnames
from superphot_pipeline.fits_utilities import get_primary_header
from superphot_pipeline import SourceFinder, DataReductionFile

def parse_command_line():
    """Return the parsed command line arguments."""

    parser = get_cmdline_parser(__doc__)
    parser.add_argument(
        'calibrated_images',
        nargs='+',
        help='A combination of individual images and image directories to '
        'process. Directories are not searched recursively.'
    )
    parser.add_argument(
        '--srcextract-only-if',
        default='True',
        help='Expression involving the header of the input images that '
        'evaluates to True/False if a particular image from the specified '
        'image collection should/should not be processed.'
    )
    parser.add_argument(
        '--srcfind-tool',
        choices=['fistar', 'hatphot'],
        default='fistar',
        help='The source extractor to use.'
    )
    parser.add_argument(
        '--brightness-threshold',
        type=float,
        default=1000,
        help='The minimum brightness to require of extracted sources.'
    )
    parser.add_argument(
        '--filter-sources',
        default='True',
        help='A condition involving the output columns from source extraction '
        'to impose on the list of extracted sources (sources that fail are '
        'discarded).'
    )
    parser.add_argument(
        '--data-reduction-fname',
        default='DR/{RAWFNAME}.h5',
        help='Format string to generate the filename(s) of the data reduction '
        'files where extracted sources are saved. Replacement fields can be '
        'anything from the header of the calibrated image.'
    )
    parser.add_argument(
        '--srcextract-version',
        default=0,
        help='If you wish to have multiple source extractions performed on the '
        'same frame specify different version numbers here so they can reside '
        'side-by-side in the data reduction file.'
    )
    return parser.parse_args()

def find_stars(image_collection, configuration):
    """Extract sources from all input images and save them to DR files."""

    srcextract_version = configuration.pop('srcextract_version')
    DataReductionFile.fname_template = configuration.pop('data_reduction_fname')
    find_stars_in_image = SourceFinder(**configuration)
    for image_fname in image_collection:
        fits_header=get_primary_header(image_fname)
        with DataReductionFile(header=fits_header, mode='a') as dr_file:
            dr_file.add_frame_header(fits_header)
            dr_file.add_sources(
                find_stars_in_image(image_fname),
                'srcextract.sources',
                'srcextract_column_name',
                srcextract_version=srcextract_version
            )


if __name__ == '__main__':
    cmdline_config = vars(parse_command_line())
    del cmdline_config['config_file']
    cmdline_config['tool'] = cmdline_config.pop('srcfind_tool')
    find_stars(
        get_fits_fnames(
            cmdline_config.pop('calibrated_images'),
            cmdline_config.pop('srcextract_only_if')
        ),
        cmdline_config
    )
