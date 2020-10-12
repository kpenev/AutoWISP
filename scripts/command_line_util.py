"""Shared set-up of command line arguments for mulitple scripts."""

from configargparse import ArgumentParser, DefaultsFormatter

def get_default_frame_processing_cmdline(description,
                                         default_config_files,
                                         default_outfname_pattern):
    """
    Define common command line arguments for processing collections of frames.

    Args:
        description(str):    A description of the scrsipt whose command line
            parsing is being set up.

        default_config_files(list):    A list of filenames of config files to
            parse by default.

    Returns:
        ArgumentParser:
            A command line/config file parser pre-populated with options common
            to all scripts that process collections of FITS frames.
    """

    parser = ArgumentParser(
        description=description,
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
        help='The images to process. Should include either fits '
        'images and/or directories. In the latter case, all files with `.fits` '
        'in their filename in the specified directory are included '
        '(sub-directories are not searched).'
    )

    parser.add_argument(
        '--allow-overwrite', '--overwrite', '-f',
        action='store_true',
        help='If images exist and this argument is not passed, an excetpion is '
        'thrown.'
    )
    parser.add_argument(
        '--prevent-dir-creation',
        dest='allow_dir_creation',
        action='store_false',
        help='Prohibit the script from creating directories needed to store the'
        ' output files. By default it is allowed.'
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
    parser.add_argument(
        '--outfname-pattern',
        default=default_outfname_pattern,
        help='A %%-substitution pattern involving FITS header keywords, '
        'augmented by FITS_ROOT (name of FITS file with path and extension '
        'removed) that expands to the filename for storing the by channel '
        'images.'
    )

    return parser
