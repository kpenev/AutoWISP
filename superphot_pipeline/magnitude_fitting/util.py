"""Functions for creating and working with photometric renfereneces."""

import numpy
from numpy.lib import recfunctions

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

def read_master_catalogue(fname, source_id_parser):
    """Return the catalogue info in the given file formatted for magfitting."""

    data = numpy.genfromtxt(fname, dtype=None, names=True, deletechars='')
    data.dtype.names = [colname.split('[', 1)[0]
                        for colname in data.dtype.names]
    catalogue_sources = data['ID']
    data = recfunctions.drop_fields(data, 'ID', usemask=False)
    return {source_id_parser(source_id): source_data
            for source_id, source_data in zip(catalogue_sources, data)}
