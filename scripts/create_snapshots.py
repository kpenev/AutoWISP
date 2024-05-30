#!/usr/bin/env python3

"""Generate snapshots from a collection of FITS files for quick review."""

import logging

from command_line_util import get_default_frame_processing_cmdline

from autowisp.file_utilities import find_fits_fnames
from autowisp.image_utilities import create_snapshot

def parse_configuration(default_config_files=('create_snapshots.cfg',),
                        default_snapshot_pattern='%(FITS_ROOT)s.jpg'):
    """Return the configuration to use for splitting by channel."""

    parser = get_default_frame_processing_cmdline(__doc__,
                                                  default_config_files,
                                                  default_snapshot_pattern)

    return parser.parse_args()

def create_all_snapshots(configuration):
    """Create the snapshots per the specified configuraion (from cmdline)."""

    logging.basicConfig(level=getattr(logging, configuration.log_level))
    for image_fname in find_fits_fnames(configuration.images):
        create_snapshot(image_fname,
                        configuration.outfname_pattern,
                        overwrite=configuration.allow_overwrite,
                        skip_existing=configuration.resume)

if __name__ == '__main__':
    create_all_snapshots(parse_configuration())
