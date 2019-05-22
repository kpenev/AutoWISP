"""Functions for creating and working with photometric renfereneces."""

import numpy
from numpy.lib import recfunctions
from astropy.io import fits

def get_single_photref(data_reduction_file, **path_substitutions):
    """Create a photometric reference out of the raw photometry in a DR file."""

    source_data = data_reduction_file.get_source_data(
        magfit_iterations=[0],
        shape_map_variables=False,
        string_source_ids=False,
        **path_substitutions
    )
    return {tuple(source['ID']): dict(x=source['x'],
                                      y=source['y'],
                                      mag=source['mag'],
                                      mag_err=source['mag_err'])
            for source in source_data}

def get_master_photref(photref_fname):
    """Read a FITS photometric reference created by MasterPhotrefCollector."""

    result = dict()
    with fits.open(photref_fname, 'readonly') as photref_fits:
        num_photometries = len(photref_fits) - 1
        for phot_ind, phot_reference in enumerate(photref_fits[1:]):
            for source_index, source_id in enumerate(
                    zip(phot_reference.data['IDprefix'].astype('int'),
                        phot_reference.data['IDfield'],
                        phot_reference.data['IDsource'])
            ):
                if source_id not in result:
                    result[source_id] = dict(
                        mag=numpy.full(num_photometries,
                                       numpy.nan,
                                       numpy.float64),
                        mag_err=numpy.full(num_photometries,
                                           numpy.nan,
                                           numpy.float64)
                    )
                result[source_id]['mag'][phot_ind] = phot_reference.data[
                    'magnitude'
                ][
                    source_index
                ]
                result[source_id]['mag_err'][phot_ind] = phot_reference.data[
                    'mediandev'
                ][
                    source_index
                ]
    return result

def read_master_catalogue(fname, source_id_parser):
    """Return the catalogue info in the given file formatted for magfitting."""

    data = numpy.genfromtxt(fname, dtype=None, names=True, deletechars='')
    data.dtype.names = [colname.split('[', 1)[0]
                        for colname in data.dtype.names]
    catalogue_sources = data['ID']
    data = recfunctions.drop_fields(data, 'ID', usemask=False)
    return {source_id_parser(source_id): source_data
            for source_id, source_data in zip(catalogue_sources, data)}
