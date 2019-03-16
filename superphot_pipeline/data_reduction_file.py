"""Define a class for worknig with data reduction files."""

from ctypes import c_char_p, c_uint, c_double, c_ushort, c_int

import numpy

from superphot_pipeline.database.hdf5_file_structure import\
    HDF5FileDatabaseStructure

git_id = '$Id$'

#Out of my control (most ancestors come from h5py module).
#pylint: disable=too-many-ancestors
class DataReductionFile(HDF5FileDatabaseStructure):
    """
    Interface for working with the pipeline data reduction (DR) files.

    Attributes:
        _key_io_tree_to_dr (dict):    A dictionary specifying the correspondence
            between the keys used in SuperPhotIOTree to store quantities and the
            element key in the DR file.

        _dtype_dr_to_io_tree (dict):    A dictionary specifying the
            correspondence between data types for entries in DR files and data
            types in SuperPhotIOTree.
    """

    _key_io_tree_to_dr = {
        'bg.model': 'bg.cfg.model',
        'psffit.min_bg_pix': 'shapefit.cfg.src.min_bg_pix',
        'psffit.gain': 'shapefit.cfg.gain',
        'psffit.magnitude_1adu': 'shapefit.cfg.magnitude_1adu',
        'psffit.grid': 'shapefit.cfg.psf.bicubic.grid',
        'psffit.initial_aperture': 'shapefit.cfg.psf.bicubic.initial_aperture',
        'psffit.max_abs_amplitude_change':
        'shapefit.cfg.psf.bicubic.max_abs_amplitude_change',
        'psffit.max_rel_amplitude_change':
        'shapefit.cfg.psf.bicubic.max_rel_amplitude_change',
        'psffit.pixrej': 'shapefit.cfg.psf.bicubic.pixrej',
        'psffit.smoothing': 'shapefit.cfg.psf.bicubic.smoothing',
        'psffit.max_chi2': 'shapefit.cfg.psf.max-chi2',
        'psffit.max_iterations': 'shapefit.cfg.psf.max_iterations',
        'psffit.min_convergence_rate': 'shapefit.cfg.psf.min_convergence_rate',
        'psffit.model': 'shapefit.cfg.psf.model',
        'psffit.terms': 'shapefit.cfg.psf.terms',
        'psffit.srcpix_cover_bicubic_grid':
        'shapefit.cfg.src.cover_bicubic_grid',
        'psffit.srcpix_max_aperture': 'shapefit.cfg.src.max_aperture',
        #TODO: fix tree entry name to psffit.src_max_count
        'psffit.srcpix_max_count': 'shapefit.cfg.src.max_count',
        'psffit.srcpix_min_pix': 'shapefit.cfg.src.min_pix',
        'psffit.srcpix_max_pix': 'shapefit.cfg.src.max_pix',
        'psffit.srcpix_max_sat_frac': 'shapefit.cfg.src.max_sat_frac',
        'psffit.srcpix_min_signal_to_noise':
        'shapefit.cfg.src.min_signal_to_noise'
    }

    _dtype_dr_to_io_tree = {
        numpy.string_: str,
        numpy.uint: c_uint,
        numpy.int: c_int,
        numpy.float64: c_double
    }

    @classmethod
    def _get_root_tag_name(cls):
        """The name of the root tag in the layout configuration."""

        return 'DataReduction'

    def __init__(self, *args, **kwargs):
        """See HDF5File for description of arguments."""

        super().__init__('data_reduction', *args, **kwargs)

    def add_star_shape_fit(self, shape_fit_result_tree):
        """
        Add the results of a star shape fit to the DR file.

        Args:
            shape_fit_result_tree(superphot.SuperPhotIOTree):    The return
                value of a successful call of superphot.FitStarShape.fit().

        Returns:
            None
        """

        def parse_grid_str(grid_str):
            """Parse the grid string entry from the SuperPhotIOTree."""

            return numpy.array(
                [
                    [float(v) for v in sub_grid.split(',')]
                    for sub_grid in grid_str.split(';')
                ]
            )

        for element_type in ['dataset', 'attribute', 'link']:
            print(element_type
                  +
                  ':\n\t'
                  +
                  '\n\t'.join(self._elements[element_type]))
        print('DR quantities:\n')
        for quantity_name in shape_fit_result_tree.defined_quantity_names():
            print('\t' + quantity_name, end='')

            dr_key = self._key_io_tree_to_dr.get(quantity_name, quantity_name)

            found = False
            for element_type in ['dataset', 'attribute', 'link']:
                if dr_key in self._elements[element_type]:
                    if quantity_name == 'psffit.grid':
                        #TODO: implement grid and cover_grid parsing
                        dtype=str
                        value = parse_grid_str(
                            shape_fit_result_tree.get(quantity_name, dtype)
                        )
                    elif quantity_name == 'psffit.srcpix_cover_bicubic_grid':
                        value = (
                            shape_fit_result_tree.get(
                                quantity_name,
                                dtype
                            ).lower()
                            ==
                            'true'
                        )
                    else:
                        dtype = self._dtype_dr_to_io_tree[self.get_dtype(dr_key)]
                        value = shape_fit_result_tree.get(quantity_name, dtype)
                    #TODO: add automatic detection for versions
                    self.add_attribute(dr_key,
                                       value,
                                       if_exists='error',
                                       background_version=0,
                                       shapefit_version=0)
                    print(' (' + repr(dtype) + ' ' + element_type + '): ',
                          end='')
                    found = True
                    break
            if not found:
                dtype=str
                value = shape_fit_result_tree.get(quantity_name, dtype)
                print(' (ignored): ', end='')
            print(repr(value))

#pylint: enable=too-many-ancestors

if __name__ == '__main__':

    dr_file = DataReductionFile('test.hdf5', 'a')

    from lxml import etree
    from superphot import FitStarShape
    from superphot.superphot_io_tree import SuperPhotIOTree

    root_element = dr_file.layout_to_xml()
    root_element.addprevious(
        etree.ProcessingInstruction(
            'xml-stylesheet',
            'type="text/xsl" href="hdf5_file_structure.xsl"'
        )
    )
    etree.ElementTree(element=root_element).write('example_structure.xml',
                                                  pretty_print=True,
                                                  xml_declaration=True,
                                                  encoding='utf-8')

    fitprf = FitStarShape(mode='prf',
                          shape_terms='{1}',
                          grid=[-1.0, 0.0, 1.0],
                          initial_aperture=2.0,
                          smoothing=None,
                          min_convergence_rate=0.0)

    #Debugging code
    #pylint: disable=protected-access
    tree = SuperPhotIOTree(fitprf._library_configuration)
    dr_file.add_star_shape_fit(tree)
