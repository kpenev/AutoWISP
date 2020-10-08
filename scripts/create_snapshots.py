#!/usr/bin/env python3

"""Generate snapshots from a collection of FITS files for quick review."""

import logging

from configargparse import ArgumentParser, DefaultsFormatter

from superphot_pipeline.image_utilities import\
    create_snapshot,\
    fits_image_generator

def parse_configuration(default_config_files=('create_snapshots.cfg',),
                        default_snapshot_pattern='%(FITS_ROOT)s.jpg'):
    """Return the configuration to use for splitting by channel."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=default_config_files,
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False
    )

    parser.add_argument(
        '--config', '-c',
        is_config_file=True,
        help='Config file to use instead of default.'
    )

    parser.add_argument(
        'images',
        nargs='+',
        help='The images to create snapshots of. Should include either fits '
        'images and/or directories. In the latter case, all files with `.fits` '
        'in their filename in the specified directory are included '
        '(sub-directories are not searched).'
    )
    parser.add_argument(
        '--outfname-pattern',
        default=default_snapshot_pattern,
        help='A %%-substitution pattern involving FITS header keywords, '
        'augmented by FITS_ROOT (name of FITS file with path and extension '
        'removed) that expands to the filename for storing the by channel '
        'images. Note that the FILTERS keyword is replaced by the particular '
        'channel being saved.'
    )
    parser.add_argument(
        '--allow-overwrite', '--overwrite', '-f',
        action='store_true',
        help='If images exist and this argument is not passed, an excetpion is '
        'thrown.'
    )
    parser.add_argument(
        '--log-level',
        choices=('DEBUG', 'INFO', 'WARNING', 'ERROR'),
        default='INFO',
        help='Set the verbosity of logging output.'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Pass this option to skip creating by channel files that already '
        'exist. If only some of the channel files for a given input image '
        'exist, those are overwritten if allowed, or an error is raised if not.'
    )

    return parser.parse_args()

def create_all_snapshots(configuration):
    """Create the snapshots per the specified configuraion (from cmdline)."""

    logging.basicConfig(level=getattr(logging, configuration.log_level))
    for image_fname in fits_image_generator(configuration.images):
        create_snapshot(image_fname,
                        configuration.outfname_pattern,
                        overwrite=configuration.allow_overwrite,
                        skip_existing=configuration.resume)

if __name__ == '__main__':
    create_all_snapshots(parse_configuration())
