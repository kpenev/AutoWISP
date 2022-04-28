#!/usr/bin/env python3

"""Calibrate image(s)."""

import re

from configargparse import ArgumentParser, DefaultsFormatter, Action
from asteval import Interpreter
from astropy.io import fits

from superphot_pipeline.image_utilities import fits_image_generator
from superphot_pipeline.image_calibration import Calibrator, overscan_methods


def parse_area_str(area_str):
    """Parse a string formatted as <xmin>,<xmax>,<ymin>,<ymax> to dict."""

    return dict(
        zip(
            ['xmin', 'xmax', 'ymin', 'ymax'],
            [int(s) for s in area_str.split(',')]
        )
    )


#Interface defined by argparse
#pylint: disable=too-few-public-methods
class ParseChannelsAction(Action):
    """Parse chanel command line arguments as described in the help."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Parse all channels into a dictionary as required by Calibrator."""

        result = dict()
        rex = re.compile(r'(?P<name>\w+)\('
                         r'(?P<x_start>\d+)(,(?P<x_step>\d+))?;'
                         r'(?P<y_start>\d+)(,(?P<y_step>\d+))?\)')
        channel_str = re.sub(r'\s+', '', ''.join(values))
        while channel_str:
            parsed = rex.match(channel_str)
            if not parsed or parsed.span()[0] != 0:
                raise ValueError('Malformatted channel specification: '
                                 +
                                 repr(channel_str))
            channel_str = channel_str[parsed.span()[1]:]

            parsed = parsed.groupdict()
            for direction in 'xy':
                if parsed[direction + '_step'] is None:
                    parsed[direction + '_step'] = 2
            result[parsed['name']] = tuple(
                slice(
                    int(parsed[direction + '_start']),
                    None,
                    int(parsed[direction + '_step'])
                )
                for direction in 'xy'
            )
        setattr(namespace, self.dest, result)


class ParseOverscanAction(Action):
    """Parse overscan command line argument per the help."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Parse the --overscans argument to dict as required by Calibrator."""

        try:
            method, areas_str = values[0].split(':')
            setattr(
                namespace,
                self.dest,
                dict(
                    method=getattr(overscan_methods, method.title()),
                    areas=[parse_area_str(area)
                           for area in areas_str.split(';')]
                )
            )
        except Exception as orig_exception:
            raise ValueError(
                'Malformatted overscan specification: '
                +
                repr(values)
            ) from orig_exception
#pylint: enable=too-few-public-methods


def parse_command_line():
    """Return the parsed command line arguments."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=[],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=True
    )
    parser.add_argument(
        '--config-file', '-c',
        is_config_file=True,
        help='Specify a configuration file in liu of using command line '
        'options. Any option can still be overriden on the command line.'
    )

    parser.add_argument(
        'image_collection',
        nargs='+',
        help='A combination of individual images and image directories to '
        'process. Directories are not searched recursively.'
    )
    parser.add_argument(
        '--calibrate-only-if',
        default='True',
        help='Expression involving the header of the input images that '
        'evaluates to True/False if a particular image from the specified '
        'image collection should/should not be processed.'
    )
    parser.add_argument(
        '--raw-hdu',
        type=int,
        default=None,
        help='Which hdu of the raw frames contains the image to calibrate. If '
        'not specified, the first HDU with non-zero NAXIS value is used.'
    )
    parser.add_argument(
        '--split-channels',
        default=False,
        nargs='+',
        action=ParseChannelsAction,
        help='Allows processing images from color detectors which have pixels '
        'sensitive to different colors staggered on the image. Each entry '
        'specifies a sub-set of pixels that together form an image of just one '
        'color. If at least one channel is specified, any pixels not assigned '
        'to a channel are ignored. The format of each entry is: '
        '<channel name>(<x start>[,<x step>=2]];<y start>[,<y step>=2])'
        '[<channel name>(<x start>[,<x step>=2]];<y start>[,<y step>=2]).'
    )
    parser.add_argument(
        '--saturation-threshold',
        type=float,
        default=None,
        help='A value above which a pixel is considered saturated. If '
        r'uspecified, the value is set to 80%% of the highest pixel value in '
        'EACH image.'
    )
    parser.add_argument(
        '--overscans',
        action=ParseOverscanAction,
        default=dict(areas=None, method=None),
        help='Overscan correction to apply. The format is: '
        '<method>:<xmin1>,<xmax1>,<ymin1>,<ymax1>'
        '[;<xmin2>,<xmax2>,<ymin2>,<ymax2>;...].'
    )
    parser.add_argument(
        '--image-area',
        default=None,
        type=parse_area_str,
        help='The area of the image to process (the rest is discarded). The '
        'format is <xmin>,<xmax>,<ymin>,<ymax>'
    )
    parser.add_argument(
        '--gain',
        default=None,
        type=float,
        help='The gain to assume for the input image (electrons/ADU). If not '
        'specified, it must be defined in the header as GAIN keyword.'
    )
    for master in ['bias', 'dark', 'flat']:
        parser.add_argument(
            '--master-' + master,
            default=None,
            help='The master ' + master + ' to apply. No ' + master +
            ' correction is applied of not specified.'
        )
    parser.add_argument(
        '--master-mask',
        default=None,
        nargs='+',
        help='Mask(s) to apply, indicating pixel quality. All pixels are '
        'considered "good" if no mask is specified.'
    )
    parser.add_argument(
        '--calibrated-fname',
        default='CAL/{BASE_FNAME}.fits.fz',
        help='Format string to generate the filenames for saving the '
        'calibrated images. Replacement fields can be anything from the header '
        'of the generated image (including {CLRCHNL} - name of channel if '
        'channel splitting is done), and {BASE_FNAME} - the name of the '
        'corresponding input FITS file without directories or `.fits` and '
        '`.fits.fz` extension.'
    )
    return parser.parse_args()


def calibrate(image_collection, configuration):
    """Calibrate the images from the specified collection."""

    image_condition = configuration.pop('calibrate_only_if')
    calibrate_image = Calibrator(**configuration)
    for image_fname in fits_image_generator(image_collection):
        if image_condition != 'True':
            with fits.open(image_fname, 'readonly') as image:
                for hdu in image:
                    if hdu['NAXIS'] != 0:
                        evaluate = Interpreter().symtable.update(hdu.header)
                        break
                if not evaluate(image_condition):
                    continue
        calibrate_image(image_fname)


if __name__ == '__main__':
    cmdline_config = vars(parse_command_line())
    del cmdline_config['config_file']
    calibrate(cmdline_config.pop('image_collection'), cmdline_config)
