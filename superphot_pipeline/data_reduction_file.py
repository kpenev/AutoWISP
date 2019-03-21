"""Define a class for worknig with data reduction files."""

from ctypes import c_uint, c_double, c_int, c_ubyte
import re

import numpy

from superphot import SmoothDependence
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

        _hat_id_prefixes (numpy.array):    A list of the currently recognized
            HAT-ID prefixes, with the correct data type ready for adding as a
            dataset.
    """

    _key_io_tree_to_dr = {
        'projsrc.x': 'srcproj.x',
        'projsrc.y': 'srcproj.y',
        'bg.model': 'bg.cfg.model',
        'bg.value': 'bg.values',
        'bg.error': 'bg.errors',
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
        'shapefit.cfg.src.min_signal_to_noise',
        'psffit.mag': 'shapefit.magnitudes',
        'psffit.mag_err': 'shapefit.magnitude_errors',
        'psffit.chi2': 'shapefit.chi2',
        'psffit.sigtonoise': 'shapefit.signal_to_noise',
        'psffit.npix': 'shapefit.num_pixels',
        'psffit.quality': 'shapefit.quality_flag',
        'psffit.psfmap': 'shapefit.map_coef',
    }

    _dtype_dr_to_io_tree = {
        numpy.string_: str,
        numpy.uint: c_uint,
        numpy.uint8: c_ubyte,
        numpy.int: c_int,
        numpy.float64: c_double
    }

    @classmethod
    def _get_root_tag_name(cls):
        """The name of the root tag in the layout configuration."""

        return 'DataReduction'

    def _parse_hat_source_id(self, source_id):
        """Return the prefix ID, field numberand source number."""

        print('Parsing source ID:' + repr(source_id))
        if isinstance(source_id, bytes):
            c_style_end = source_id.find(b'\0')
            if c_style_end >= 0:
                source_id = source_id[:c_style_end].decode()
            else:
                source_id = source_id.decode()
        print('Re-formatted to ' + repr(source_id))
        prefix_str, field_str, source_str = source_id.split('-')
        print('Prefix: %s, field: %s, source: %s'
              %
              (prefix_str, field_str, source_str))
        return (
            numpy.where(self._hat_id_prefixes
                        ==
                        prefix_str.encode('ascii'))[0][0],
            int(field_str),
            int(source_str)
        )

    @staticmethod
    def _parse_grid_str(grid_str):
        """Parse the grid string entry from the SuperPhotIOTree."""

        result = [
            numpy.array([float(v) for v in sub_grid.split(',')])
            for sub_grid in grid_str.split(';')
        ]
        if len(result) == 1:
            return [result, result]

        assert len(result) == 2
        return result

    def _add_shapefit_sources(self,
                              shape_fit_result_tree,
                              num_sources,
                              image_index,
                              fit_variables,
                              **path_substitutions):
        """
        Add the sources used for shape fitting as projected sources.

        Args:
            shape_fit_result_tree:    See same name argument to
                add_star_shape_fit().

            num_sources:    See same name argument to add_star_shape_fit().

            image_index:    See same name argument to add_star_shape_fit().

            fit_variables:    See same name argument to add_star_shape_fit().

            path_substitutions:    Values to substitute in the path to the
                dataset (usually versions of various components).

        Returns:
            None
        """

        def add_fit_variables():
            """Add datasets for all source variables used in the fit."""

            print('READING VARIABLES, expected size: %dx%d'
                  %
                  (len(fit_variables), num_sources))
            variables = shape_fit_result_tree.get_psfmap_variables(
                image_index,
                len(fit_variables),
                num_sources
            )
            print('FINISHED READING VARIABLES')
            print('VARABLES SHAPE: ' + repr(variables.shape))
            for var_name, var_values in zip(fit_variables, variables):
                print(var_name + ': ' + repr(var_values))

                dataset_data = numpy.array(
                    var_values,
                    dtype=self.get_dtype('srcproj.' + var_name)
                )
                print('\tAdding dataset: ' + repr(dataset_data))
                self.add_dataset(
                    'srcproj.' + var_name,
                    dataset_data,
                    if_exists='error',
                    **path_substitutions
                )

        def add_source_ids():
            """Add the datasets containing the source IDs."""

            print('Getting source IDs at: '
                  +
                  'projsrc.srcid.name.' + str(image_index))
            source_ids = shape_fit_result_tree.get(
                'projsrc.srcid.name.' + str(image_index),
                numpy.dtype('S100'),
                shape=num_sources
            )

            id_data = {
                id_part: numpy.empty(
                    (len(source_ids),),
                    dtype=self.get_dtype('srcproj.hat_id_' + id_part)
                )
                for id_part in ['prefix', 'field', 'source']
            }
            for source_index, source_id in enumerate(source_ids):
                (
                    id_data['prefix'][source_index],
                    id_data['field'][source_index],
                    id_data['source'][source_index]
                ) = self._parse_hat_source_id(source_id)
            for id_part in ['prefix', 'field', 'source']:
                self.add_dataset(
                    'srcproj.hat_id_' + id_part,
                    id_data[id_part],
                    if_exists='error',
                    **path_substitutions
                )

            self.add_attribute(
                'srcproj.recognized_hat_id_prefixes',
                self._hat_id_prefixes,
                if_exists='error',
                **path_substitutions
            )

        add_source_ids()
        add_fit_variables()

    def _add_shapefit_map(self,
                          shape_fit_result_tree,
                          **path_substitutions):
        """
        Add the coefficients defining the PSF/PRF map to DR file.

        Args:
            shape_fit_result_tree:    See same name argument to
                add_star_shape_fit().

            path_substitutions:    See same name argument to
                _add_shapefit_sources().

        Returns:
            None
        """

        grid = self._parse_grid_str(
            shape_fit_result_tree.get('psffit.grid', str)
        )
        for direction, splits in zip(['x', 'y'], grid):
            self.add_attribute('shapefit.cfg.psf.bicubic.grid.' + direction,
                               splits,
                               if_exists='error',
                               **path_substitutions)
        num_terms = len(
            SmoothDependence.expand_expression(
                shape_fit_result_tree.get('psffit.terms', str)
            )
        )
        coefficients = shape_fit_result_tree.get(
            'psffit.psfmap',
            shape=(4,
                   grid[0].size - 2,
                   grid[1].size - 2,
                   num_terms)
        )
        self.add_dataset('shapefit.map_coef',
                         coefficients,
                         if_exists='error',
                         background_version=0,
                         shapefit_version=0)

    def __init__(self, *args, **kwargs):
        """See HDF5File for description of arguments."""

        super().__init__('data_reduction', *args, **kwargs)

        self._hat_id_prefixes = numpy.array(
            ['HAT', 'UCAC4'],
            dtype=self.get_dtype('srcproj.recognized_hat_id_prefixes')
        )

    def add_star_shape_fit(self,
                           shape_fit_result_tree,
                           num_sources,
                           image_index=0,
                           fit_variables=('x', 'y')):
        """
        Add the results of a star shape fit to the DR file.

        Args:
            shape_fit_result_tree(superphot.SuperPhotIOTree):    The return
                value of a successful call of superphot.FitStarShape.fit().

            num_sources (int):    The number of surces used in the fit (used to
                determine the expected size of datasets).

            image_index (int):    The index of the image whose DR file is being
                filled within the input list of images passed to PSF/PRF
                fitting.

            fit_variables (iterable):    The variables that were used in the
                fit in the order in which they appear in the tree.

        Returns:
            None
        """

        for element_type in ['dataset', 'attribute', 'link']:
            print(element_type
                  +
                  ':\n\t'
                  +
                  '\n\t'.join(self._elements[element_type]))
        print('DR quantities:')

        self._add_shapefit_map(shape_fit_result_tree,
                               background_version=0,
                               shapefit_version=0)
        self._add_shapefit_sources(
            shape_fit_result_tree=shape_fit_result_tree,
            num_sources=num_sources,
            image_index=image_index,
            fit_variables=fit_variables,
            background_version=0,
            shapefit_version=0,
            srcproj_version=0
        )

        indexed_rex = re.compile(r'.*\.(?P<image_index_str>[0-9]+)$')
        for quantity_name in shape_fit_result_tree.defined_quantity_names():

            print('\t' + quantity_name)

            indexed_match = indexed_rex.fullmatch(quantity_name)
            if indexed_match:
                if int(indexed_match['image_index_str']) == image_index:
                    key_quantity = quantity_name[
                        :
                        indexed_match.start('image_index_str')-1
                    ]
                    print('\t\t-> ' + key_quantity)
                else:
                    print('\t\tSkipping')
                    continue
            else:
                key_quantity = quantity_name

            dr_key = self._key_io_tree_to_dr.get(key_quantity, key_quantity)

            found = False
            for element_type in ['dataset', 'attribute', 'link']:
                if (
                        dr_key in self._elements[element_type]
                        and
                        key_quantity not in ['psffit.variables',
                                             'psffit.grid',
                                             'psffit.psfmap']
                        and
                        not key_quantity.startswith('projsrc.')
                ):
                    if quantity_name == 'psffit.srcpix_cover_bicubic_grid':
                        dtype = str
                        print('\t\tGetting '
                              +
                              repr(dtype)
                              +
                              ' (cover grid) value')
                        value = (
                            shape_fit_result_tree.get(
                                quantity_name,
                                dtype
                            ).lower()
                            ==
                            'true'
                        )
                    else:
                        dtype = (
                            self._dtype_dr_to_io_tree[self.get_dtype(dr_key)]
                        )
                        print('\t\tGetting ' + repr(dtype) + ' value(s)')
                        value = shape_fit_result_tree.get(
                            quantity_name,
                            dtype,
                            shape=(num_sources
                                   if element_type == 'dataset' else
                                   None)
                        )
                    #TODO: add automatic detection for versions
                    print('\t\t(' + repr(dtype) + ' ' + element_type + '): ')
                    found = True
                    getattr(self, 'add_' + element_type)(dr_key,
                                                         value,
                                                         if_exists='error',
                                                         background_version=0,
                                                         shapefit_version=0)
                    break
            if found:
                print('\t\t' + repr(value))
            else:
                dtype = str
                print('\t\t(ignored): ')

#pylint: enable=too-many-ancestors

if __name__ == '__main__':

    dr_file = DataReductionFile('test.hdf5', 'a')

    from lxml import etree
    #pylint: disable=ungrouped-imports
    from superphot import FitStarShape, SuperPhotIOTree, SubPixPhot
    #pylint: enable=ungrouped-imports

#    root_element = dr_file.layout_to_xml()
#    root_element.addprevious(
#        etree.ProcessingInstruction(
#            'xml-stylesheet',
#            'type="text/xsl" href="hdf5_file_structure.xsl"'
#        )
#    )
#    etree.ElementTree(element=root_element).write('example_structure.xml',
#                                                  pretty_print=True,
#                                                  xml_declaration=True,
#                                                  encoding='utf-8')

    fitprf = FitStarShape(mode='prf',
                          shape_terms='{1}',
                          grid=[-1.0, 0.0, 1.0],
                          initial_aperture=2.0,
                          smoothing=None,
                          min_convergence_rate=0.0)
    subpixphot = SubPixPhot()

    #Debugging code
    #pylint: disable=protected-access
    tree = SuperPhotIOTree(subpixphot._library_configuration)
    #dr_file.add_star_shape_fit(tree, 0)

    test_sources = numpy.empty(10, dtype=[('id', 'S100'),
                                          ('x', numpy.float64),
                                          ('enabled', numpy.bool),
                                          ('y', numpy.float64),
                                          ('bg', numpy.float64),
                                          ('bg_err', numpy.float64)])
    for i in range(10):
        test_sources[i]['id'] = 'HAT-%03d-%07d' % (i, i)
        test_sources[i]['x'] = (10.0 * numpy.pi) * (i % 4)
        test_sources[i]['y'] = (10.0 * numpy.pi) * (i / 4)
        test_sources[i]['bg'] = 10.0 + 0.01 * i
        test_sources[i]['bg_err'] = 1.0 + 0.2 * i
    print('Source data: ' + repr(test_sources))
    map_coefficients = numpy.ones((4, 1, 1, 10), dtype=numpy.float64)
    print('Test sources: ' + repr(test_sources))
    tree.set_aperture_photometry_inputs(
        source_data=test_sources,
        star_shape_grid=[[-1.0, 0.0, 1.0], [-1.5, 0.0, 1.0]],
        star_shape_map_terms='O3{x, y}',
        star_shape_map_coefficients=map_coefficients
    )
    dr_file.add_star_shape_fit(tree, 10)
