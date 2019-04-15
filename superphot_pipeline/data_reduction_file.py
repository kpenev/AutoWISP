"""Define a class for worknig with data reduction files."""

from ctypes import c_uint, c_double, c_int, c_ubyte
import re

import numpy
import h5py

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
        'apphot.const_error': 'apphot.cfg.error_floor',
        'apphot.aperture': 'apphot.cfg.aperture',
        'apphot.gain': 'apphot.cfg.gain',
        'apphot.magnitude-1adu': 'apphot.cfg.magnitude_1adu'
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
                if var_name == 'enabled':
                    dataset_data = (
                        var_values != 0
                    ).astype(
                        self.get_dtype('srcproj.' + var_name)
                    )
                else:
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

    def _get_shapefit_sources(self, **path_substitutions):
        """
        Read the sources used for shape fitting from this DR file.

        Args:
            path_substitutions:    See get_aperture_photometry_inputs().

        Returns:
            numpy.array(dtype=[('ID', 'S#'),\
                               ('x', numpy.float64),\
                               ('y', numpy.float64),\
                               ('bg', numpy.float64),\
                               ('bg_err', numpy.float64),\
                               ('bg_npix', numpy.float64),\
                               ('mag', numpy.float64),\
                               ('mag_err', numpy.float64),
                               ...]):
                The source data used as input for shape fitting. It is
                guaranteed to contain at least the fields listed in the return
                type, and all other variables avialable for the projected
                sources.

            float:
                The magnitude that corresponds to a flux of 1ADU from shape
                fitting photometry.
        """

        def get_source_ids():
            """
            Return the IDs of the projected sources found in the file.

            Args:
                None

            Returns:
                [str]:
                    List of the source ID strings.
            """

            hat_id_prefixes = self.get_attribute(
                'srcproj.recognized_hat_id_prefixes',
                **path_substitutions
            )
            id_data = tuple(
                self.get_dataset(
                    'srcproj.hat_id_' + id_part,
                    **path_substitutions
                )
                for id_part in ('prefix', 'field', 'source')
            )
            for data_ind in 1, 2:
                assert len(id_data[data_ind]) == len(id_data[0])

            return [
                b'%s-%03d-%07d' % (hat_id_prefixes[prefix_ind],
                                   field,
                                   source)
                for prefix_ind, field, source in zip(*id_data)
            ]

        def list_shape_map_var_names():
            """
            Return a list of the names of all variables shape map can depend on.

            Args:
                None

            Returns:
                [str]:
                    The names of all source projection variables to check for.
            """

            return list(
                filter(
                    lambda pipeline_key: (
                        pipeline_key.startswith('srcproj.')
                        and
                        (not pipeline_key.startswith('srcproj.hat_id_'))
                    ),
                    self._defined_elements['dataset']
                )
            )

        def get_fit_variables(num_sources):
            """Return a dictionary containing all stored map variables."""

            found_data = dict()
            shape_map_var_names = list_shape_map_var_names()
            print('Shape map variables: ' +  repr(shape_map_var_names))
            for var_name in shape_map_var_names:
                try:
                    found_data[var_name] = self.get_dataset(
                        var_name,
                        expected_shape=(num_sources,),
                        **path_substitutions
                    )
                except IOError:
                    pass
            return found_data

        def add_measurements(num_sources, destination):
            """Add bacgkround and shape fit photometry to destination."""

            for destination_key, dataset_key in (
                    ('bg', 'bg.values'),
                    ('bg_err', 'bg.errors'),
                    ('bg_npix', 'bg.npix'),
                    ('mag', 'shapefit.magnitudes'),
                    ('mag_err', 'shapefit.magnitude_errors')
            ):
                destination[destination_key] = self.get_dataset(
                    dataset_key,
                    expected_shape=(num_sources,),
                    **path_substitutions
                )

        source_ids = get_source_ids()
        num_sources = len(source_ids)
        found_source_variables = get_fit_variables(num_sources)
        add_measurements(num_sources, found_source_variables)
        result = numpy.empty(
            (num_sources,),
            dtype=(
                [('ID', 'S100')]
                +
                [
                    (
                        var_name,
                        numpy.bool if var_name == 'enabled' else numpy.float64
                    )
                    for var_name in found_source_variables
                ]
            )
        )
        result['ID'] = source_ids
        for var_name in found_source_variables:
            result[var_name] = found_source_variables[var_name]
        return (
            result,
            self.get_attribute('shapefit.cfg.magnitude_1adu',
                               **path_substitutions)
        )

    def _get_shapefit_map_grid(self, **path_substitutions):
        """Return the grid used to represent star shape from this DR file."""

        return numpy.array([
            self.get_attribute('shapefit.cfg.psf.bicubic.grid.' + direction,
                               **path_substitutions)
            for direction in ('x', 'y')
        ])

    def _get_shapefit_map(self, **path_substitutions):
        """
        Read the map of the shapes of point sources from this DR file.

        Args:
            path_substitutions:    See get_aperture_photometry_inputs().

        Returns:
            2-D numpy.array(dtype=numpy.float64):
                The grid used to represent source shapes.

            str:
                An expression defining the terms in the star shape map.

            4-D numpy.array(dtype=numpy.float64):
                The coefficients of the shape map. See the C++ documentation for
                more details of the layout.
        """

        return (
            self._get_shapefit_map_grid(**path_substitutions),
            self.get_attribute('shapefit.cfg.psf.terms',
                               **path_substitutions).decode(),
            self.get_dataset('shapefit.map_coef', **path_substitutions)
        )

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
                         **path_substitutions)

    def _auto_add_tree_quantities(self,
                                  result_tree,
                                  num_sources,
                                  skip_quantities,
                                  image_index=0,
                                  **path_substitutions):
        """
        Best guess for how to add tree quantities to DR file.

        Args:
            result_tree(SuperPhotIOTree):    The tree to extract quantities to
                add.

            num_sources(int):    The number of sources (assumed to be ththe
                length of all datasets).

            skip_quantities(compiled rex matcher):    Quantities matching this
                regular expression will not be added to the DR file by this
                function.

            image_index(int):    For quantities which are split by image, only
                the values associated to this image index will be added.

        Returns:
            None
        """

        indexed_rex = re.compile(r'.*\.(?P<image_index_str>[0-9]+)$')
        for quantity_name in result_tree.defined_quantity_names():

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

            for element_type in ['dataset', 'attribute', 'link']:
                if (
                        dr_key in self._elements[element_type]
                        and
                        skip_quantities.match(key_quantity) is None
                ):
                    dtype = (
                        self._dtype_dr_to_io_tree[self.get_dtype(dr_key)]
                    )
                    print('\t\tGetting ' + repr(dtype) + ' value(s)')
                    value = result_tree.get(
                        quantity_name,
                        dtype,
                        shape=(num_sources
                               if element_type == 'dataset' else
                               None)
                    )
                    #TODO: add automatic detection for versions
                    print('\t\t(' + repr(dtype) + ' ' + element_type + '): ')
                    getattr(self, 'add_' + element_type)(dr_key,
                                                         value,
                                                         if_exists='error',
                                                         **path_substitutions)
                    break

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
        print('Added shape fit sources')
        self.add_attribute(
            self._key_io_tree_to_dr['psffit.srcpix_cover_bicubic_grid'],
            (
                shape_fit_result_tree.get(
                    'psffit.srcpix_cover_bicubic_grid',
                    str
                ).lower()
                ==
                'true'
            ),
            if_exists='error',
            shapefit_version=0
        )
        print('Added cover grid attribute.')
        self._auto_add_tree_quantities(
            result_tree=shape_fit_result_tree,
            num_sources=num_sources,
            skip_quantities=re.compile(
                '|'.join([r'^psffit\.variables$',
                          r'^psffit\.grid$',
                          r'^psffit\.psfmap$',
                          r'^psffit.srcpix_cover_bicubic_grid$',
                          r'^projsrc\.'])
            ),
            image_index=image_index,
            background_version=0,
            shapefit_version=0
        )

    def fill_aperture_photometry_input_tree(self,
                                            tree,
                                            shapefit_version=0,
                                            srcproj_version=0,
                                            background_version=0):
        """
        Fill a SuperPhotIOTree with shape fit info for aperture photometry.

        Args:
            shapefit_version:    The version of the star shape fit results
                stored in the file to use when initializing the tree.

            srcproj_version:    The version of the projected sources to use for
                aperture photometry.

        Returns:
            int:
                The number of sources added to the tree.
        """

        def list_source_variable_datasets(**substitutions):
            """
            List all variables that must be included in the source data.

            Args:
                None

            Returns:
                [2-tuples]:
                    Each tuple identifies the name of a variable that may
                    participate in the PSF map and its associated dataset path
                    in self.
            """

            result = []
            dset_key_var_name = {'shapefit.magnitudes': 'mag',
                                 'shapefit.magnitude_errors': 'mag_err',
                                 'bg.values': 'bg',
                                 'bg.errors': 'bg_err',
                                 'bg.npix': 'bg_npix'}
            var_name_parser = re.compile(r'srcproj.(?P<var_name>\w+)')
            for dataset_key in self._elements['dataset']:
                if dataset_key.startswith('srcproj.hat_id_'):
                    continue
                parsed_var_name = var_name_parser.fullmatch(dataset_key)
                if (
                        parsed_var_name is None
                        and
                        dataset_key not in dset_key_var_name
                ):
                    continue

                dataset_path = (self._file_structure[dataset_key].abspath
                                %
                                substitutions)
                if dataset_path in self:
                    var_name = (parsed_var_name.group('var_name')
                                if parsed_var_name is not None else
                                dset_key_var_name[dataset_key])

                    result.append((var_name, dataset_path))
            return result

        def get_source_data(**substitutions):
            """See SuperPhotIOTree.set_aperture_photometry_inputs() argument."""

            fit_variable_datasets = list_source_variable_datasets(
                **substitutions
            )
            sources_shape = self[fit_variable_datasets[0][1]].shape
            assert len(sources_shape) == 1

            source_data = numpy.empty(
                shape=sources_shape,
                dtype=(
                    [('id', 'S20')]
                    +
                    [
                        (
                            var_path[0],
                            (
                                numpy.uint if var_path[0] == 'bg_npix'
                                else numpy.float64
                            )
                        )
                        for var_path in fit_variable_datasets
                    ]
                )
            )
            for var_name, dset_path in fit_variable_datasets:
                assert self[dset_path].shape == sources_shape
                source_data[var_name] = self[dset_path]

            prefix_dset = self[
                self._file_structure['srcproj.hat_id_prefix'].abspath
                %
                substitutions
            ]
            translate_prefix = dict(
                (value, prefix)
                for prefix, value in h5py.check_dtype(
                    enum=prefix_dset.dtype
                ).items()
            )
            for source_ind, (id_prefix, id_field, id_source) in enumerate(
                    zip(
                        prefix_dset,
                        self[
                            self._file_structure[
                                'srcproj.hat_id_field'
                            ].abspath
                            %
                            substitutions
                        ],
                        self[
                            self._file_structure[
                                'srcproj.hat_id_source'
                            ].abspath
                            %
                            substitutions
                        ]
                    )
            ):
                source_data['id'][source_ind] = '%s-%03d-%07d' % (
                    translate_prefix[id_prefix],
                    id_field,
                    id_source
                )
            return source_data

        def get_star_shape_grid(**substitutions):
            """Return the grid used for representing the star shape."""

            return numpy.array([
                self.get_attribute('shapefit.cfg.psf.bicubic.grid.x',
                                   **substitutions),
                self.get_attribute('shapefit.cfg.psf.bicubic.grid.y',
                                   **substitutions)
            ])

        source_data = get_source_data(
            shapefit_version=shapefit_version,
            srcproj_version=srcproj_version,
            background_version=background_version
        )
        print('Source data:\n' + repr(source_data))
        magnitude_1adu = self.get_attribute(
            'shapefit.cfg.magnitude_1adu',
            shapefit_version=shapefit_version
        )
        tree.set_aperture_photometry_inputs(
            source_data=source_data,
            star_shape_grid=get_star_shape_grid(
                shapefit_version=shapefit_version
            ),
            star_shape_map_terms=self.get_attribute(
                'shapefit.cfg.psf.terms',
                shapefit_version=shapefit_version
            ),
            star_shape_map_coefficients=self.get_dataset(
                'shapefit.map_coef',
                shapefit_version=shapefit_version
            ),
            magnitude_1adu=magnitude_1adu
        )
        return source_data.size

    def add_aperture_photometry(self,
                                apphot_result_tree,
                                num_sources,
                                num_apertures):
        """
        Add the results of aperture photometry to the DR file.

        Args:
            apphot_result_tree:(superphot.SuperPhotIOTree):    The tree which
                was passed to the :class:superphot.SubPixPhot instance which did
                the aperture photometry (i.e. where the results were added).

            num_sources(int):    The number of sources for which aperture
                photometry was done. The same as the number of sources the star
                shape fitting which was used by the aperture photometry was
                performed on for the photometered image.

        Returns:
            None
        """

        for aperture_index, aperture in enumerate(
                apphot_result_tree.get('apphot.aperture',
                                       c_double,
                                       shape=(num_apertures,))
        ):
            self.add_attribute('apphot.cfg.aperture',
                               aperture,
                               if_exists='error',
                               apphot_version=0,
                               aperture_index=aperture_index)

        self._auto_add_tree_quantities(
            result_tree=apphot_result_tree,
            num_sources=num_sources,
            skip_quantities=re.compile(r'(?!apphot\.)|^apphot.aperture$'),
            apphot_version=0
        )

    def get_aperture_photometry_inputs(self,
                                       **path_substitutions):
        """
        Return all required information for aperture photometry from PSF fit DR.

        Args:
            path_substitutions:    Values to substitute in the paths to the
                datasets and attributes containing shape fit informaiton
                (usually versions of various components).

        Returns:
            dict:
                All parameters required by
                SuperPhotIOTree.set_aperture_photometry_inputs() directly
                passable to that method using **.
        """

        result = dict()
        result['source_data'], result['magnitude_1adu'] = (
            self._get_shapefit_sources(**path_substitutions)
        )
        (
            result['star_shape_grid'],
            result['star_shape_map_terms'],
            result['star_shape_map_coefficients']
        ) = self._get_shapefit_map(**path_substitutions)
        return result

#pylint: enable=too-many-ancestors

def mock_shape_fit():
    """Return a SuperPhotIOTree containing entries as if shape fit was done."""

    #pylint: disable=ungrouped-imports
    from superphot import FitStarShape, SuperPhotIOTree
    #pylint: enable=ungrouped-imports

    fitprf = FitStarShape(mode='prf',
                          shape_terms='O2{x, y}',
                          grid=[[-1.0, 0.0, 1.0], [-1.5, 0.0, 1.0]],
                          initial_aperture=2.0,
                          smoothing=None,
                          min_convergence_rate=0.0)

    #pylint: disable=protected-access
    tree = SuperPhotIOTree(fitprf._library_configuration)
    #pylint: enable=protected-access

    test_sources = numpy.empty(10, dtype=[('id', 'S100'),
                                          ('x', numpy.float64),
                                          ('enabled', numpy.bool),
                                          ('y', numpy.float64),
                                          ('mag', numpy.float64),
                                          ('mag_err', numpy.float64),
                                          ('bg', numpy.float64),
                                          ('bg_err', numpy.float64),
                                          ('bg_npix', numpy.uint)])
    for i in range(10):
        test_sources[i]['id'] = 'HAT-%03d-%07d' % (i, i)
        test_sources[i]['x'] = (10.0 * numpy.pi) * (i % 4)
        test_sources[i]['y'] = (10.0 * numpy.pi) * (i / 4)
        test_sources[i]['bg'] = 10.0 + 0.01 * i
        test_sources[i]['bg_err'] = 1.0 + 0.2 * i
        test_sources[i]['mag'] = numpy.pi - i
        test_sources[i]['mag_err'] = 0.01 * i
        test_sources[i]['bg_npix'] = 10 * i
    print('Source data: ' + repr(test_sources))
    map_coefficients = numpy.ones((4, 1, 1, 6), dtype=numpy.float64)
    print('Test sources: ' + repr(test_sources))
    tree.set_aperture_photometry_inputs(
        source_data=test_sources,
        star_shape_grid=[[-1.0, 0.0, 1.0], [-1.5, 0.0, 1.0]],
        star_shape_map_terms='O2{x, y}',
        star_shape_map_coefficients=map_coefficients,
        magnitude_1adu=10.0
    )
    return tree

def debug():
    """Some debugging code executed when module is run as script."""

    dr_file = DataReductionFile('test.hdf5', 'r')
    apphot_inputs = dr_file.get_aperture_photometry_inputs(background_version=0,
                                                           srcproj_version=0,
                                                           shapefit_version=0)
    for key, value in apphot_inputs.items():
        print(key + ': ' + repr(value))
    dr_file.close()
    exit(0)

    #from lxml import etree
    #pylint: disable=ungrouped-imports
    #pylint: disable=unused-import
    from superphot._initialize_library import superphot_library
    from ctypes import c_void_p, c_char_p
    #pylint: enable=ungrouped-imports
    #pylint: enable=unused-import


    dr_file = DataReductionFile('test.hdf5', 'w')
    dr_file.add_star_shape_fit(mock_shape_fit(), 10)
    dr_file.close()
    exit(0)

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

if __name__ == '__main__':

    debug()
