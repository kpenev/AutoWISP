#!/usr/bin/env python3

"""Demonstrate the usage of EPD."""

from glob import glob
from os.path import join as join_paths, dirname, splitext, basename
import logging

from superphot_pipeline import parallel_epd, save_epd_statistics
from superphot_pipeline.magnitude_fitting import read_master_catalogue
from superphot_pipeline import DataReductionFile

def parse_lc_fname(fname):
    """Return the source ID as 3-tuple of ints for the given LC."""

    return tuple(int(component)
                 for component in splitext(basename(fname))[0].split('-'))

def add_catalogue_info(lc_fnames, catalogue_fname, magnitute_column, result):
    """Fill the catalogue information fields in result."""

    with DataReductionFile() as mem_dr:
        catalogue = read_master_catalogue(catalogue_fname,
                                          mem_dr.parse_hat_source_id)

    for lc_ind, fname in enumerate(lc_fnames):
        source_id = parse_lc_fname(fname)
        result[lc_ind]['ID'] = source_id
        cat_info = catalogue[source_id]
        result[lc_ind]['mag'] = cat_info[magnitute_column]
        result[lc_ind]['xi'] = cat_info['xi']
        result[lc_ind]['eta'] = cat_info['eta']

def do_epd():
    """Do the EPD without polluting the main scope."""

    data_dir = join_paths(dirname(__file__),
                          'test_data',
                          '10-20170306')
    lc_fnames = glob(join_paths(data_dir, 'lcs', '*.hdf5'))
    epd_result = parallel_epd(
        lc_fnames,
        5,
        used_variables=dict(
            x=('srcproj.x', dict()),
            y=('srcproj.y', dict()),
            S=('srcextract.psf_map.eval',
               dict(srcextract_psf_param='S')),
            D=('srcextract.psf_map.eval',
               dict(srcextract_psf_param='D')),
            K=('srcextract.psf_map.eval',
               dict(srcextract_psf_param='K')),
            bg=('bg.value', dict())
        ),
        fit_points_filter_expression=None,
        fit_terms_expression='O2{x}',
        fit_datasets=(
            [
                (
                    'shapefit.magfit.magnitude',
                    dict(magfit_iteration=5),
                    'shapefit.epd.magnitude'
                )
            ]
            +
            [
                (
                    'apphot.magfit.magnitude',
                    dict(magfit_iteration=5, aperture_index=ap_ind),
                    'apphot.epd.magnitude'
                )
                for ap_ind in range(39)
            ]
        ),
        error_avg='nanmedian',
        rej_level=5,
        max_rej_iter=20
    )
    add_catalogue_info(lc_fnames,
                       join_paths(data_dir, 'lc_dump_catalogue.ucac4'),
                       'R',
                       epd_result)


    statistics_fname = join_paths(data_dir, 'lcs', 'epd_statistics.txt')

    save_epd_statistics(epd_result, statistics_fname)

    logging.info('Generated statistics file: %s.' % repr(statistics_fname))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    do_epd()
