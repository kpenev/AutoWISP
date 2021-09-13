"""Define a class for worknig with data reduction files."""

#pylint: disable=too-many-lines

from ctypes import c_uint, c_double, c_int, c_ubyte
import re

import numpy
import h5py

from superphot import SmoothDependence
from superphot_pipeline.hat.file_parsers import parse_anmatch_transformation

from .post_process import DataReductionPostProcess

git_id = '$Id$'

#TODO: Add missed attributes: bg.cfg.annulus, bg.cfg.zero.

#Out of my control (most ancestors come from h5py module).
#pylint: disable=too-many-ancestors
class DataReductionFile(DataReductionPostProcess):
    """
    Interface for working with the pipeline data reduction (DR) files.

    Attributes:
        _product(str):    The pipeline key of the HDF5 product. In this case:
            `'data_reduction'`

        _key_io_tree_to_dr (dict):    A dictionary specifying the correspondence
            between the keys used in SuperPhotIOTree to store quantities and the
            element key in the DR file.

        _dtype_dr_to_io_tree (dict):    A dictionary specifying the
            correspondence between data types for entries in DR files and data
            types in SuperPhotIOTree.
    """

    @classmethod
    def _product(cls):
        return 'data_reduction'

    _key_io_tree_to_dr = {
        'projsrc.x': 'srcproj.x',
        'projsrc.y': 'srcproj.y',
        'bg.model': 'bg.cfg.model',
        'bg.value': 'bg.value',
        'bg.error': 'bg.error',
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
        'psffit.mag': 'shapefit.magnitude',
        'psffit.mag_err': 'shapefit.magnitude_error',
        'psffit.chi2': 'shapefit.chi2',
        'psffit.sigtonoise': 'shapefit.signal_to_noise',
        'psffit.npix': 'shapefit.num_pixels',
        'psffit.quality': 'shapefit.quality_flag',
        'psffit.psfmap': 'shapefit.map_coef',
        'apphot.const_error': 'apphot.cfg.error_floor',
        'apphot.aperture': 'apphot.cfg.aperture',
        'apphot.gain': 'apphot.cfg.gain',
        'apphot.magnitude-1adu': 'apphot.cfg.magnitude_1adu',
        'apphot.mag': 'apphot.magnitude',
        'apphot.mag_err': 'apphot.magnitude_error',
        'apphot.quality': 'apphot.quality_flag'
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

            variables = shape_fit_result_tree.get_psfmap_variables(
                image_index,
                len(fit_variables),
                num_sources
            )
            for var_name, var_values in zip(fit_variables, variables):
                if var_name == 'enabled':
                    dataset_data = (
                        var_values != 0
                    ).astype(
                        self.get_dtype('srcproj.' + var_name)
                    )
                else:
                    dataset_data = numpy.array(
                        var_values,
                        dtype=self.get_dtype('srcproj.' + var_name)
                    )
                self.add_dataset(
                    'srcproj.' + var_name,
                    dataset_data,
                    if_exists='error',
                    **path_substitutions
                )

        def add_source_ids():
            """Add the datasets containing the source IDs."""

            source_ids = shape_fit_result_tree.get(
                'projsrc.srcid.name.' + str(image_index),
                numpy.dtype('S100'),
                shape=num_sources
            )

            id_data = self.parse_hat_source_id(source_ids)

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

    def _get_shapefit_map_grid(self, **path_substitutions):
        """Return the grid used to represent star shape from this DR file."""

        return numpy.array([
            self.get_attribute('shapefit.cfg.psf.bicubic.grid.x',
                               **path_substitutions),
            self.get_attribute('shapefit.cfg.psf.bicubic.grid.y',
                               **path_substitutions)
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
                               **path_substitutions),
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
        apphot_indexed_rex = re.compile(r'|apphot\..*\.'
                                        r'(?P<image_index_str>[0-9]+)\.'
                                        r'(?P<ap_index_str>[0-9]+)$')
        for quantity_name in result_tree.defined_quantity_names():
            indexed_match = apphot_indexed_rex.fullmatch(quantity_name)
            if indexed_match:
                path_substitutions['aperture_index'] = int(
                    indexed_match['ap_index_str']
                )
            else:
                path_substitutions.pop('aperture_index', 0)
                indexed_match = indexed_rex.fullmatch(quantity_name)

            if indexed_match:
                if int(indexed_match['image_index_str']) == image_index:
                    key_quantity = quantity_name[
                        :
                        indexed_match.start('image_index_str')-1
                    ]
                else:
                    continue
            else:
                key_quantity = quantity_name

            dr_key = self._key_io_tree_to_dr.get(key_quantity, key_quantity)

            for element_type in ['dataset', 'attribute', 'link']:
                if (
                        dr_key in self.elements[element_type]
                        and
                        skip_quantities.match(key_quantity) is None
                ):
                    value = result_tree.get(
                        quantity_name,
                        self._dtype_dr_to_io_tree[self.get_dtype(dr_key)],
                        shape=(num_sources
                               if element_type == 'dataset' else
                               None)
                    )
                    #TODO: add automatic detection for versions
                    getattr(self, 'add_' + element_type)(dr_key,
                                                         value,
                                                         if_exists='error',
                                                         **path_substitutions)
                    break

    @classmethod
    def get_fname_from_header(cls, header):
        """Return the filename of the DR file for the given header."""

        #TODO: implement filename template from DB ofter DB has been designed.
        #pylint: disable=no-member
        return cls.fname_template % header
        #pylint: enable=no-member

    def get_dataset_creation_args(self, dataset_key, **path_substitutions):
        """See HDF5File.get_dataset_creation_args(), but handle srcextract."""

        result = super().get_dataset_creation_args(dataset_key)

        if dataset_key == 'srcextract.sources':
            column = path_substitutions['srcextract_column_name']
            if column in ['ID', 'NumberPixels']:
                result['compression'] = 'gzip'
                result['compression_opts'] = 9
            else:
                del result['compression']
                result['scaleoffset'] = 3
        elif dataset_key == 'catalogue.columns':
            column = path_substitutions['catalogue_column_name']
            if column in ['hat_id_prefix',
                          'hat_id_field',
                          'hat_id_source',
                          'objtype',
                          'doublestar',
                          'sigRA',
                          'sigDec',
                          'phqual',
                          'magsrcflag']:
                result['compression'] = 'gzip'
                result['compression_opts'] = 9
                result['shuffle'] = True
            elif column in ['RA', 'Dec']:
                del result['compression']
                result['scaleoffset'] = 7
            elif column in ['xi', 'eta']:
                del result['compression']
                result['scaleoffset'] = 6
            elif column in ['ucacmag',
                            'J', 'H', 'K',
                            'B', 'V', 'R', 'I',
                            'u', 'g', 'r', 'i', 'z']:
                del result['compression']
                result['scaleoffset'] = 3
            elif column in ['dist',
                            'epochRA', 'epochDec',
                            'sigucacmag',
                            'errJ', 'errH', 'errK']:
                del result['compression']
                result['scaleoffset'] = 2
            else:
                del result['compression']
                result['scaleoffset'] = 1

        return result

    def __init__(self, *args, **kwargs):
        """Open or create a data reduction file.

        Args:
            See HDF5File.__init__() for description of arguments, however
            instead of fname, a DataReductionFile can be specified by the header
            of the frame it corresponds to (or at least a dict-like object
            defining the header keywords required by the DR filename template).
        """

        if 'header' in kwargs:
            kwargs['fname'] = self.get_fname_from_header(kwargs['header'])
            del kwargs['header']

        super().__init__(*args, **kwargs)

        self._hat_id_prefixes = numpy.array(
            ['HAT', 'UCAC4'],
            dtype=self.get_dtype('srcproj.recognized_hat_id_prefixes')
        )

    def get_dtype(self, element_key):
        """Return numpy data type for the element with by the given key."""

        result = super().get_dtype(element_key)

        if element_key.endswith('.hat_id_prefix'):
            return h5py.special_dtype(
                enum=(
                    result,
                    dict((prefix, value)
                         for value, prefix in enumerate(self._hat_id_prefixes))
                )
            )

        return result

    def parse_hat_source_id(self, source_id):
        """Return the prefix ID, field number, and source number."""

        if isinstance(source_id, numpy.ndarray):
            id_data = {
                id_part: numpy.empty(
                    (len(source_id),),
                    dtype=self.get_dtype('srcproj.hat_id_' + id_part)
                )
                for id_part in ['prefix', 'field', 'source']
            }

            for source_index, this_id in enumerate(source_id):
                (
                    id_data['prefix'][source_index],
                    id_data['field'][source_index],
                    id_data['source'][source_index]
                ) = self.parse_hat_source_id(this_id)
            return id_data

        if isinstance(source_id, bytes):
            c_style_end = source_id.find(b'\0')
            if c_style_end >= 0:
                source_id = source_id[:c_style_end].decode()
            else:
                source_id = source_id.decode()
        prefix_str, field_str, source_str = source_id.split('-')
        return (
            numpy.where(self._hat_id_prefixes
                        ==
                        prefix_str.encode('ascii'))[0][0],
            int(field_str),
            int(source_str)
        )

    def get_hat_source_id_str(self, source_id):
        """Return the string representation of 3-integer HAT-id."""

        return (self._hat_id_prefixes[source_id[0]].decode()
                +
                '-%03d-%07d' % tuple(source_id[1:]))

    def get_source_count(self, **path_substitutions):
        """
        Return the number of sources for the given tool versions.

        Args:
            path_substitutions:    Values to substitute in the paths to the
                datasets and attributes containing shape fit informaiton
                (usually versions of various components).

        Returns:
            int:
                The number of projected sources in the databasets reached by the
                given substitutions.
        """

        return self[
            self._file_structure['srcproj.hat_id_prefix'].abspath
            %
            path_substitutions
        ].len()

    def add_frame_header(self, header, **substitutions):
        """Add the header of the corresponding FITS frame to DR file."""

        self.write_fitsheader_to_dataset('fitsheader', header, **substitutions)

    def get_frame_header(self, **substitutions):
        """Return the header of the corresponding FITS frame."""

        return self.read_fitsheader_from_dataset('fitsheader', **substitutions)

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
        self._auto_add_tree_quantities(
            result_tree=shape_fit_result_tree,
            num_sources=num_sources,
            skip_quantities=re.compile(
                '|'.join([r'^psffit\.variables$',
                          r'^psffit\.grid$',
                          r'^psffit\.psfmap$',
                          r'^psffit.srcpix_cover_bicubic_grid$',
                          r'^projsrc\.',
                          r'^apphot\.'])
            ),
            image_index=image_index,
            background_version=0,
            shapefit_version=0
        )

    def get_aperture_photometry_inputs(self, **path_substitutions):
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

        def get_shape_map_variable_names(source_data):
            """Identify and return the variable names directly as c_char_p."""

            result = []
            for var_name in source_data.dtype.names:
                if (
                        var_name == 'enabled'
                        or
                        (
                            source_data[var_name].dtype.kind == 'f'
                            and
                            var_name not in ['bg', 'bg_err', 'bg_npix',
                                             'flux', 'flux_err',
                                             'mag', 'mag_err']
                        )
                ):
                    result.append(var_name)
            return result

        result = dict()
        result['source_data'] = self.get_source_data(
            magfit_iterations=[0],
            shapefit=True,
            apphot=False,
            shape_map_variables=True,
            string_source_ids=True,
            **path_substitutions
        )
        result['magnitude_1adu'] = self.get_attribute(
            'shapefit.cfg.magnitude_1adu',
            **path_substitutions
        )
        (
            result['star_shape_grid'],
            result['star_shape_map_terms'],
            result['star_shape_map_coefficients']
        ) = self._get_shapefit_map(**path_substitutions)
        result['star_shape_map_varnames'] = get_shape_map_variable_names(
            result['source_data']
        )
        return result

    def fill_aperture_photometry_input_tree(self,
                                            tree,
                                            shapefit_version=0,
                                            srcproj_version=0,
                                            background_version=0):
        """
        Fill a SuperPhotIOTree with shape fit info for aperture photometry.

        Args:
            tree(superphot.SuperPhotIOTree):    The tree to fill.

            shapefit_version(int):    The version of the star shape fit results
                stored in the file to use when initializing the tree.

            srcproj_version(int):    The version of the projected sources to
                assume was used for shape fitting, and to use for aperture
                photometry.

            background_vesrion(int):    The version of the background extraction
                to assume was used for shape fitting, and to use for aperture
                photometry.

        Returns:
            int:
                The number of sources added to the tree.
        """

        aperture_photometry_inputs = self.get_aperture_photometry_inputs(
            shapefit_version=shapefit_version,
            srcproj_version=srcproj_version,
            background_version=background_version
        )
        tree.set_aperture_photometry_inputs(**aperture_photometry_inputs)
        return (
            aperture_photometry_inputs['source_data'].size,
            aperture_photometry_inputs['star_shape_map_varnames']
        )

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

            num_apertures(int):    The number of apertures for which photometry
                was extracted.

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

    def get_num_apertures(self, **path_substitutions):
        """Return the number of apertures used for aperture photometry."""

        num_apertures = 0
        while True:
            try:
                self._check_for_dataset('apphot.magnitude',
                                        aperture_index=num_apertures,
                                        **path_substitutions)
                num_apertures += 1
            except IOError:
                return num_apertures

        assert False

    def get_num_magfit_iterations(self, **path_substitutions):
        """
        Return how many magnitude fitting iterations are in the file.

        Args:
            path_substitutions:    See get_source_count().

        Returns:
            int:
                The number of magnitude fitting iterations performed on the
                set of photometry measurements identified by the
                path_substitutions argument.
        """

        path_substitutions['aperture_index'] = 0
        path_substitutions['magfit_iteration'] = 0
        for photometry_mode in ['shapefit', 'apphot']:
            try:
                self._check_for_dataset(
                    photometry_mode + '.magfit.magnitude',
                    **path_substitutions
                )
            except IOError:
                continue

            while True:
                path_substitutions['magfit_iteration'] += 1
                try:
                    self._check_for_dataset(
                        photometry_mode + '.magfit.magnitude',
                        **path_substitutions
                    )
                except IOError:
                    break

        return path_substitutions['magfit_iteration']

    def has_shape_fit(self, **path_substitutions):
        """True iff shape fitting photometry exists for path_substitutions."""

        try:
            self._check_for_dataset('shapefit.magnitude',
                                    **path_substitutions)
            return True
        except IOError:
            return False

    def get_shape_map_variables(self, num_sources, **path_substitutions):
        """Return a dictionary containing all stored map variables."""

        found_data = dict()
        shape_map_var_names = list(
            filter(
                lambda pipeline_key: (
                    pipeline_key.startswith('srcproj.')
                    and
                    (not pipeline_key.startswith('srcproj.hat_id_'))
                ),
                self.elements['dataset']
            )
        )

        for var_name in shape_map_var_names:
            try:
                found_data[var_name[len('srcproj.'):]] = self.get_dataset(
                    var_name,
                    expected_shape=(num_sources,),
                    **path_substitutions
                )
            except IOError:
                print('Dataset ' + var_name + ' not found!')
        return found_data

    #Could not think of a reasonable way to simplify further.
    #pylint: disable=too-many-locals
    #pylint: disable=too-many-statements
    def get_source_data(self,
                        *,
                        magfit_iterations='all',
                        shape_fit=True,
                        apphot=True,
                        shape_map_variables=True,
                        string_source_ids=True,
                        background=True,
                        position=True,
                        **path_substitutions):
        """
        Extract available photometry from the data reduction file.

        Args:
            magfit_iterations(iterable):    The set of magnitude fitting
                iterations to include in the result. ``0`` is the raw photometry
                (i.e. no magnitude fitting), 1 is  single reference frame fit, 1
                is the first re-fit etc. Use ``'all'`` to get all iterations.
                Negative numbers have the same interpretation as python list
                indices. For example ``-1`` is the final iteration.

            shape_fit(bool):    Should the result include shape fit photometry
                measurements.

            apphot(bool):    Should the result include aperture photometry
                measurements.

            shape_map_variables(bool):    Should the result include the
                variables on which the star shape map depents?

            string_source_ids(bool):    Should source IDs be formatted as
                strings (True) or a set of integers (False)?

            background(bool):    Should the result include information about the
                background behind the sources?

            position(bool):    Should the result include the (x, y) positions of
                the sources.

            path_substitutions:    See get_source_count().

        Returns:
            numpy structured array:
                The photometry information in the current data reduction file.
                The fields are:

                    * ID: an array of sources IDs in the given DR file. Either a
                      string (if string_source_ids) or (an array of) integer(s).
                      For HAT IDs, each non-string entry is 3 integers (prefix,
                      field, source).

                    * x (numpy.float64): The x coordinates of the sources

                    * y (numpy.float64): The y coordinates of the sources

                    * bg (numpy.float64): The background estimates for the
                      sources

                    * bg_err (numpy.float64): Error estimate for 'bg'

                    * bg_npix (numpy.uint): The number of pixel background
                      extraction was based on.

                    * mag (2-D numpy.float64 array): measured magnitudes. The
                      first dimension is the index within the
                      ``magfit_iterations`` argument and the second index
                      iterates over photometry, starting with shape fitting (if
                      the ``shape_fit`` argument is True),
                      followed by the aperture photometry measurement for each
                      aperture (if the ``apphot`` argument is True).

                    * mag_err (numpy.float64): Error estimate for ``mag``. Same
                      shape and order.

                    * phot_flag: The quality flag for the photometry. Same
                      shape and order as ``mag``.

                    * <map variable> (numpy.float64): one entry for each
                      variable shape map depends on. The name of the field is
                      exactly the variable name. Only included if the
                      ``shape_map_variables`` argument is True.
        """

        def initialize_result(num_sources, num_apertures, shape_map_var_names):
            """Return empty result structure with the correct shape & dtype."""

            dtype = [
                (('ID',) + (('S15',) if string_source_ids else (numpy.int, 3)))
            ]
            if background:
                dtype.extend([('bg', numpy.float64),
                              ('bg_err', numpy.float64),
                              ('bg_npix', numpy.uint)])

            num_photometries = 1 if shape_fit else 0
            if apphot:
                num_photometries += num_apertures

            magnitude_shape = (len(magfit_iterations), num_photometries)

            if num_photometries > 0:
                dtype.extend([
                    ('mag', numpy.float64, magnitude_shape),
                    ('mag_err', numpy.float64, magnitude_shape),
                    ('phot_flag', numpy.uint, magnitude_shape)
                ])

            dtype.extend([
                (
                    var_name,
                    numpy.bool if var_name == 'enabled' else numpy.float64
                )
                for var_name in shape_map_var_names
            ])

            return numpy.empty(
                shape=(num_sources,),
                dtype=dtype
            )

        def fill_source_ids(result):
            """
            Add the IDs of the projected sources found in the file to result.

            Args:
                result:    The array created by initialize_result() to fill with
                    source IDs.

            Returns:
                [str]:
                    List of the source ID strings.
            """

            hat_id_prefixes = self.get_attribute(
                'srcproj.recognized_hat_id_prefixes',
                **path_substitutions
            )
            if string_source_ids:
                id_data = tuple(
                    self.get_dataset(
                        'srcproj.hat_id_' + id_part,
                        **path_substitutions
                    )
                    for id_part in ('prefix', 'field', 'source')
                )
                for data_ind in 0, 1, 2:
                    assert len(id_data[data_ind]) == result.size

                for source_index in range(result.size):
                    result['ID'][source_index] = (
                        b'%s-%03d-%07d'
                        %
                        (
                            hat_id_prefixes[id_data[0][source_index]],
                            id_data[1][source_index],
                            id_data[2][source_index]
                        )
                    )
            else:
                for component_index, id_part in enumerate(('prefix',
                                                           'field',
                                                           'source')):
                    result['ID'][:, component_index] = self.get_dataset(
                        'srcproj.hat_id_' + id_part,
                        **path_substitutions
                    )

        def normalize_magfit_iterations():
            """Make sure ``magfit_iterations`` is a list of positive indices."""

            if magfit_iterations != 'all' and min(magfit_iterations) >= 0:
                return magfit_iterations

            all_magfit_indices = numpy.array(
                [0]
                +
                list(
                    range(
                        1,
                        self.get_num_magfit_iterations(**path_substitutions) + 1
                    )
                )
            )

            if magfit_iterations == 'all':
                return all_magfit_indices

            return all_magfit_indices[magfit_iterations]

        def fill_background(result):
            """Fill the background entries in the result."""

            for result_key, dataset_key in (('bg', 'bg.value'),
                                            ('bg_err', 'bg.error'),
                                            ('bg_npix', 'bg.npix')):
                result[result_key] = self.get_dataset(
                    dataset_key,
                    expected_shape=result.shape,
                    **path_substitutions
                )

        def fill_photometry(result):
            """Fill the photomtric measurements entries in result."""

            for result_key, dataset_key_tail in (
                    ('mag', 'magnitude'),
                    ('mag_err', 'magnitude_error'),
                    ('phot_flag', 'quality_flag')
            ):
                for iter_index, magfit_iter in enumerate(magfit_iterations):
                    if magfit_iter == 0 or result_key != 'mag':
                        dataset_key_middle = ''
                    else:
                        dataset_key_middle = 'magfit.'
                    photometry_index = 0
                    path_substitutions['magfit_iteration'] = magfit_iter - 1
                    if shape_fit:
                        result[
                            result_key
                        ][
                            :,
                            iter_index,
                            photometry_index
                        ] = self.get_dataset(
                            (
                                'shapefit.'
                                +
                                dataset_key_middle
                                +
                                dataset_key_tail
                            ),
                            expected_shape=result.shape,
                            **path_substitutions
                        )
                        photometry_index += 1
                    if apphot:
                        num_apertures = (result[result_key].shape[2]
                                         -
                                         photometry_index)
                        for aperture_index in range(num_apertures):
                            result[
                                result_key
                            ][
                                :,
                                iter_index,
                                photometry_index
                            ] = self.get_dataset(
                                (
                                    'apphot.'
                                    +
                                    dataset_key_middle
                                    +
                                    dataset_key_tail
                                ),
                                expected_shape=result.shape,
                                aperture_index=aperture_index,
                                **path_substitutions
                            )
                            photometry_index += 1

        shape_fit = shape_fit and self.has_shape_fit(**path_substitutions)
        magfit_iterations = normalize_magfit_iterations()
        num_sources = self.get_source_count(**path_substitutions)
        if shape_map_variables:
            shape_map_var_data = self.get_shape_map_variables(
                num_sources,
                **path_substitutions
            )
        else:
            shape_map_var_data = {
                coordinate: self.get_dataset('srcproj.' + coordinate,
                                             expected_shape=(num_sources,),
                                             **path_substitutions)
                for coordinate in (['x', 'y'] if position else [])
            }

        result = initialize_result(
            num_sources,
            self.get_num_apertures(**path_substitutions) if apphot else None,
            shape_map_var_data.keys()
        )

        fill_source_ids(result)

        for var_name in shape_map_var_data:
            result[var_name] = shape_map_var_data[var_name]

        if background:
            fill_background(result)

        fill_photometry(result)
        return result

    def get_source_ids(self, string_source_ids=True, **path_substitutions):
        """Return the IDs of the sources in the given DR file.


        Args:
            string_source_ids:    Should source IDs be formatted as strings
                (True) or a set of integers (False)?

            path_substitutions:    See get_source_count().

        Returns:
            numpy.array:
                See ID field of result in get_source_data().
        """

        return self.get_source_data(string_source_ids=string_source_ids,
                                    magfit_iterations=[],
                                    shape_fit=False,
                                    apphot=False,
                                    shape_map_variables=False,
                                    background=False,
                                    position=False,
                                    **path_substitutions)['ID']

    def add_magnitude_fitting(self,
                              *,
                              fitted_magnitudes,
                              fit_statistics,
                              magfit_configuration,
                              missing_indices,
                              **path_substitutions):
        """
        Add a magnitude fitting iteration to the DR file.

        Args:
            fitted_magnitudes(numpy.array):   The differential photometry
                corrected magnitudes of the sources.

            fit_statistics(dict):    Summary statistics about how the fit went.
                It should define at least the following keys:
                ``initial_src_count``, ``final_src_count``, and ``residual``.

            magfit_configuration:    The configuration structure with which
                magnitude fitting was performed.

            missing_indices:    A list of indices within the file of sources
                for which no entries are included in fitted_magnitudes.

        Returns:
            None
        """

        def pad_missing_magnitudes():
            """Return fitted magnitudes with nans added at missing_indices."""


            if not missing_indices:
                return fitted_magnitudes

            fitted_magnitudes_shape = list(fitted_magnitudes.shape)
            fitted_magnitudes_shape[0] += len(missing_indices)
            padded_fitted_magnitudes = numpy.empty(
                shape=fitted_magnitudes_shape,
                dtype=fitted_magnitudes.dtype
            )
            padded_fitted_magnitudes[missing_indices] = numpy.nan
            padded_fitted_magnitudes[
                [
                    ind not in missing_indices
                    for ind in range(fitted_magnitudes_shape[0])
                ]
            ] = fitted_magnitudes
            return padded_fitted_magnitudes

        def add_magfit_datasets(fitted_magnitudes,
                                include_shape_fit):
            """Create the datasets holding the newly fitted magnitudes."""

            num_apertures = fitted_magnitudes.shape[1]
            apphot_start = 0
            if include_shape_fit:
                num_apertures -= 1
                apphot_start = 1
                self.add_dataset('shapefit.magfit.magnitude',
                                 fitted_magnitudes[:, 0],
                                 if_exists='error',
                                 **path_substitutions)
            for aperture_index in range(num_apertures):
                self.add_dataset('apphot.magfit.magnitude',
                                 fitted_magnitudes[
                                     :,
                                     aperture_index + apphot_start
                                 ],
                                 if_exists='error',
                                 aperture_index=aperture_index,
                                 **path_substitutions)

        def add_attributes(include_shape_fit):
            """Add attributes with the magfit configuration."""

            for phot_index in range(fitted_magnitudes.shape[1]):

                phot_method = (
                    'shapefit' if include_shape_fit and phot_index == 0
                    else 'apphot'
                )

                if phot_method == 'apphot':
                    path_substitutions['aperture_index'] = (
                        path_substitutions.get('aperture_index', -1)
                        +
                        1
                    )


                if path_substitutions['magfit_iteration'] == 0:

                    self.add_attribute(
                        phot_method + '.magfitcfg.correction_type',
                        b'linear',
                        if_exists='error',
                        **path_substitutions
                    )

                    for pipeline_key_end, config_attribute in [
                            ('correction', 'correction_parametrization'),
                            ('require', 'fit_source_condition')
                    ]:
                        self.add_attribute(
                            phot_method + '.magfitcfg.' + pipeline_key_end,
                            getattr(magfit_configuration, config_attribute),
                            if_exists='error',
                            **path_substitutions
                        )

                    for config_param in ['noise_offset',
                                         'max_mag_err',
                                         'rej_level',
                                         'max_rej_iter',
                                         'error_avg']:
                        self.add_attribute(
                            phot_method + '.magfitcfg.' + config_param,
                            getattr(magfit_configuration, config_param),
                            if_exists='error',
                            **path_substitutions
                        )

                for pipeline_key_end, statistics_key in [
                        ('num_input_src', 'initial_src_count'),
                        ('num_fit_src', 'final_src_count'),
                        ('fit_residual', 'residual')
                ]:
                    self.add_attribute(
                        phot_method + '.magfit.' + pipeline_key_end,
                        fit_statistics[statistics_key][phot_index],
                        if_exists='error',
                        **path_substitutions
                    )

        path_substitutions['magfit_iteration'] = self.get_num_magfit_iterations(
            **path_substitutions
        )
        include_shape_fit = self.has_shape_fit(**path_substitutions)
        add_magfit_datasets(pad_missing_magnitudes(),
                            include_shape_fit)

        add_attributes(include_shape_fit)

    def add_hat_astrometry(self, filenames, configuration, **path_substitutions):
        """
        Add astrometry derived by fistar, and anmatch to the DR file.

        Args:
            filanemes(dict):    The files containing the astrometry results.
                Should have the following keys: `'fistar'`, `'trans'`,
                `'match'`, `'catalogue'`.

            configuration:    An object with attributes containing the
                configuraiton of how astormetry was performed.

            path_substitutions:    See get_source_count()

        Returns:
            None
        """

        def add_sources(data,
                        dataset_key,
                        column_substitution_name,
                        parse_ids=False):
            """
            Creates datasets out of the fields in an array of sources.

            Args:
                data(structured numpy.array):    The data about the sources to
                    add.

                dataset_key(str):    The pipeline key for the dataset to add.

                column_substitution_name(str):    The %-subsittution variable to
                    distinguish between the column in the array.

                parse_ids(bool):    Should self.parse_hat_source_id() be used to
                    translate string IDs to datasets to insert?

            Returns:
                None
            """

            for column_name in data.dtype.names:
                if parse_ids and column_name == 'ID':
                    id_data = self.parse_hat_source_id(data['ID'])
                    for id_part in ['prefix', 'field', 'source']:
                        self.add_dataset(
                            dataset_key=dataset_key,
                            data=id_data[id_part],
                            **{column_substitution_name: 'hat_id_' + id_part},
                            **path_substitutions
                        )
                else:
                    self.add_dataset(dataset_key=dataset_key,
                                     data=data[column_name],
                                     **{column_substitution_name: column_name},
                                     **path_substitutions)


        def add_match(extracted_sources, catalogue_sources):
            """Create dset of the matched indices from catalogue & extracted."""

            num_cat_columns = len(catalogue_sources.dtype.names)
            match_ids = numpy.genfromtxt(filenames['match'],
                                         dtype=None,
                                         names=['cat_id', 'extracted_id'],
                                         usecols=(0, num_cat_columns))
            extracted_sorter = numpy.argsort(extracted_sources['ID'])
            catalogue_sorter = numpy.argsort(catalogue_sources['ID'])
            match = numpy.empty([match_ids.size, 2], dtype=int)
            match[:, 0] = catalogue_sorter[
                numpy.searchsorted(catalogue_sources['ID'],
                                   match_ids['cat_id'],
                                   sorter=catalogue_sorter)
            ]
            match[:, 1] = extracted_sorter[
                numpy.searchsorted(extracted_sources['ID'],
                                   match_ids['extracted_id'],
                                   sorter=extracted_sorter)
            ]
            self.add_dataset(dataset_key='skytoframe.matched',
                             data=match,
                             **path_substitutions)

        def add_trans():
            """Create dsets/attrs describing the sky to frame transformation."""

            transformation, info = parse_anmatch_transformation(
                filenames['trans']
            )
            self.add_dataset(
                dataset_key='skytoframe.coefficients',
                data=numpy.stack((transformation['dxfit'],
                                  transformation['dyfit'])),
                **path_substitutions
            )
            for entry in ['type', 'order', 'offset', 'scale']:
                self.add_attribute(
                    attribute_key='skytoframe.' + entry,
                    attribute_value=transformation[entry],
                    **path_substitutions
                )
            for entry in ['residual', 'unitarity']:
                self.add_attribute(
                    attribute_key='skytoframe.' + entry,
                    attribute_value=info[entry],
                    **path_substitutions
                )
            self.add_attribute(
                attribute_key='skytoframe.sky_center',
                attribute_value=numpy.array([info['2mass']['RA'],
                                             info['2mass']['DEC']]),
                **path_substitutions
            )

        def add_configuration():
            """Add the information about the configuration used."""

            for component, config_attribute in [
                    ('srcextract', 'binning'),
                    ('catalogue', 'name'),
                    ('catalogue', 'epoch'),
                    ('catalogue', 'filter'),
                    ('catalogue', 'fov'),
                    ('catalogue', 'orientation'),
                    ('skytoframe', 'srcextract_filter'),
                    ('skytoframe', 'sky_preprojection'),
                    ('skytoframe', 'max_match_distance'),
                    ('skytoframe', 'frame_center'),
                    ('skytoframe', 'weights_expression')
            ]:
                if component == 'catalogue':
                    value = getattr(configuration,
                                    'astrom_catalogue_' + config_attribute)
                else:
                    value = getattr(configuration,
                                    component + '_' + config_attribute)
                self.add_attribute(
                    component + '.cfg.' + config_attribute,
                    value,
                    **path_substitutions
                )

        extracted_sources = numpy.genfromtxt(
            filenames['fistar'],
            names=['ID', 'x', 'y', 'Background', 'Amplitude', 'S', 'D', 'K',
                   'FWHM', 'Ellipticity', 'PositionAngle', 'Flux',
                   'SignalToNoise', 'NumberPixels'],
            dtype=None
        )
        catalogue_sources = numpy.genfromtxt(filenames['catalogue'],
                                             dtype=None,
                                             names=True,
                                             deletechars='')
        catalogue_sources.dtype.names = [
            name.split('[', 1)[0] for name in catalogue_sources.dtype.names
        ]

        add_sources(extracted_sources,
                    'srcextract.sources',
                    'srcextract_column_name')
        add_sources(catalogue_sources,
                    'catalogue.columns',
                    'catalogue_column_name',
                    parse_ids=True)
        add_match(extracted_sources, catalogue_sources)
        add_trans()
        add_configuration()

    #pylint: enable=too-many-locals
    #pylint: enable=too-many-statements

#pylint: enable=too-many-ancestors
