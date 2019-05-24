#!/usr/bin/env python3
"""Create plots showing the magnitude fitting process."""

from argparse import ArgumentParser
from os.path import splitext

from matplotlib import pyplot
import scipy

from superphot_pipeline.magnitude_fitting import\
    read_master_catalogue,\
    get_single_photref,\
    get_master_photref
from superphot_pipeline import DataReductionFile

def parse_command_line():
    """Return the command line options as attributes of an object."""

    parser = ArgumentParser(description=__doc__)

    parser.add_argument(
        'dr_fname',
        help='The filename of the data reduction file to plot the magnitude '
        'fitting of.'
    )
    parser.add_argument(
        'photref',
        help='The filename of the single or master photometric reference used '
        'in the fitting. HDDF5 files are assumed to be single references and '
        'FITS files are treated masters.'
    )
    parser.add_argument(
        'catalogue',
        help='The filename of the catalogue file used during magnitude fitting.'
    )

    return parser.parse_args()

def plot_fit_step(photref, catalogue, source_data, iteration=0):
    """
    Show a single magnitude fitting iteration.

    Args:
        photref(dict):    A properly parsed photometric reference from either
            get_single_photref() or get_master_photref().

        catalogue(dict):    A properly parsed catalogue returned by
            read_master_catalogue().

        source_data(numpy structured array):    The information about the
            sources in the data reduction file which was subjected to magnitude
            fitting, as returned by DataReductionFile.get_source_data().

    Returns:
        None
    """

    plot_x = []
    ref_y = []
    raw_y = []
    fit_y = []
    for source in source_data:
        source_id = tuple(source['ID'])
        if source_id in catalogue and source_id in photref:
            plot_x.append(catalogue[source_id]['R'])
            ref_y.append(float(photref[source_id]['mag']))
            raw_y.append(float(source['mag'][iteration]))
            fit_y.append(float(source['mag'][iteration+1]))

    pyplot.plot(plot_x, scipy.array(raw_y) - scipy.array(ref_y), '.r')
    pyplot.plot(plot_x, scipy.array(fit_y) - scipy.array(ref_y), '.g')
    pyplot.plot(plot_x, scipy.array(fit_y) - scipy.array(raw_y), '.b')
    pyplot.show()

def create_all_plots(cmdline_args):
    """Crete the plots per the command line."""

    path_substitutions = dict(shapefit_version=0,
                              srcproj_version=0,
                              apphot_version=0,
                              background_version=0,
                              magfit_version=0)

    fitted_dr = DataReductionFile(cmdline_args.dr_fname, 'r')
    source_data = fitted_dr.get_source_data(string_source_ids=False,
                                            **path_substitutions)
    fitted_dr.close()

    catalogue = read_master_catalogue(cmdline_args.catalogue,
                                      fitted_dr.parse_hat_source_id)
    photref_split = splitext(cmdline_args.photref)
    while True:
        print('Checking extension: ' + repr(photref_split[1]))
        if photref_split[1].lower() == '.fits':
            photref = get_master_photref(cmdline_args.photref)
            break
        if photref_split[1].lower() in ['.h5', '.hdf5']:
            photref_dr = DataReductionFile(cmdline_args.photref, 'r')
            photref = get_single_photref(photref_dr, **path_substitutions)
            photref_dr.close()
            break
        photref_split = splitext(photref_split[0])
        assert photref_split[1]

    plot_fit_step(photref, catalogue, source_data)

if __name__ == '__main__':
    create_all_plots(parse_command_line())
