#!/usr/bin/env python3

"""A script for visually exploring the results of a PSF/PRF fit."""

from argparse import ArgumentParser

from superphot import PiecewiseBicubicPSFMap
from superphot.utils import explore_prf

def parse_command_line():
    """Parse command line to attributes of an object."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        'dr_fname',
        help='The filename of the data reduction file containing the PSF/PRF '
        'map to explore.'
    )

    return explore_prf.parse_command_line(parser)

if __name__ == '__main__':
    cmdline_args = parse_command_line()
