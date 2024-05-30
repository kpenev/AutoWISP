#!/usr/bin/env python3

"""Generate XML showing the currently defined structure of DR and LC files."""

from argparse import ArgumentParser
from tempfile import TemporaryDirectory

from lxml import etree

from autowisp import DataReductionFile, LightCurveFile

#TODO: fix output of root attributes.

def parse_command_line():
    """Return the command line options as attributes of an object."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '--data-reduction-output', '--dr',
        default='data_reduction_structure.xml',
        help='The filename to use to save the currently defined structure for '
        'data reduction files. Default: %(default)s.'
    )
    parser.add_argument(
        '--light-curve-output', '--lc',
        default='light_curve_structure.xml',
        help='The filename to use to save the currently defined structure for '
        'light curve files. Default: %(default)s.'
    )
    return parser.parse_args()

def output_structure(hdf5_file, output_fname):
    """Create an XML file showing the structure in the given HDF5 file."""

    root_element = hdf5_file.layout_to_xml()
    root_element.addprevious(
        etree.ProcessingInstruction(
            'xml-stylesheet',
            'type="text/xsl" href="hdf5_file_structure.xsl"'
        )
    )
    etree.ElementTree(element=root_element).write(output_fname,
                                                  pretty_print=True,
                                                  xml_declaration=True,
                                                  encoding='utf-8')

if __name__ == '__main__':
    cmdline_args = parse_command_line()
    with TemporaryDirectory() as tmp_dir:
        output_structure(
            DataReductionFile(tmp_dir + '/dr.h5', 'w'),
            cmdline_args.data_reduction_output
        )
        output_structure(
            LightCurveFile(tmp_dir + '/lc.h5', 'w'),
            cmdline_args.light_curve_output
        )
