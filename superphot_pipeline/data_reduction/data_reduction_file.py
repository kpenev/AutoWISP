"""Define a class for worknig with data reduction files."""

#pylint: disable=too-many-lines

import string
from functools import partial
import logging

import numpy
import h5py
import pandas

from superphot_pipeline.hat.file_parsers import parse_anmatch_transformation
from superphot_pipeline.miscellaneous import RECOGNIZED_HAT_ID_PREFIXES

from superphot_pipeline.database.hdf5_file_structure import\
    HDF5FileDatabaseStructure

git_id = '$Id$'

#TODO: Add missed attributes: bg.cfg.annulus, bg.cfg.zero.

#The class has to satisfy many needs, hence many public methods.
#pylint: disable=too-many-public-methods

#Out of my control (most ancestors come from h5py module).
#pylint: disable=too-many-ancestors
class DataReductionFile(HDF5FileDatabaseStructure):
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


    @classmethod
    def _get_root_tag_name(cls):
        """The name of the root tag in the layout configuration."""

        return 'DataReduction'

    def _prepare_source_iter(self,
                             dataset_key,
                             column_substitution_name,
                             **path_substitutions):
        """
        Return required head and tail of paths identifying source collection.

        Args:
            See `get_sources()`.

        Returns:
            str:    The path to the parent group containing all source columns.

            str:    The string that must be in the beginning of each path for it
                to be considered part of the source collection.

            str:    The string that must be in the end of each path for it to be
                considered part of the source collection.
        """

        path_substitutions[column_substitution_name] = '{column}'
        print('Parsing source path: '
              +
              repr(self._file_structure[dataset_key].abspath))
        parsed_path = string.Formatter().parse(
            self._file_structure[dataset_key].abspath
            %
            path_substitutions
        )
        pre_column, verify, _, _ = next(parsed_path)
        print(f'Pre_column: {pre_column}, verify: {verify}')
        assert verify == 'column'
        try:
            name_tail = next(parsed_path)
            for i in range(1, 4):
                assert name_tail[i] is None
            name_tail = name_tail[0]
            try:
                next(parsed_path)
                assert False
            except  StopIteration:
                pass
        except StopIteration:
            name_tail = ''
        parent, name_head = pre_column.rsplit('/', 1)
        return parent, name_head, name_tail


    @classmethod
    def get_fname_from_header(cls, header):
        """Return the filename of the DR file for the given header."""

        #TODO: implement filename template from DB ofter DB has been designed.
        #pylint: disable=no-member
        return cls.fname_template.format(**dict(header))
        #pylint: enable=no-member

    def get_dataset_creation_args(self, dataset_key, **path_substitutions):
        """See HDF5File.get_dataset_creation_args(), but handle srcextract."""

        result = super().get_dataset_creation_args(dataset_key,
                                                   **path_substitutions)

        if dataset_key == 'srcextract.sources':
            column = path_substitutions['srcextract_column_name']
            if column.lower() in ['id', 'numberpixels', 'npix', 'nsatpix']:
                result['compression'] = 'gzip'
                result['compression_opts'] = 9
            else:
                del result['compression']
                result['scaleoffset'] = 3

        return result


    def add_sources(self,
                    data,
                    dataset_key,
                    column_substitution_name,
                    *,
                    parse_ids=False,
                    ascii_columns=(),
                    **path_substitutions):
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

            string_columns([str]):    A list of column names to convert to ascii
                strings before saving.

        Returns:
            None
        """

        def iter_data():
            """Iterate over (column name, values) of the input data."""

            if hasattr(data, 'dtype'):
                for column_name in data.dtype.names:
                    yield column_name, data[column_name]
            else:
                yield data.index.name, data.index.array
                for column_name, series in data.items():
                    yield column_name, series.array

        for column_name, column_data in iter_data():
            if column_name in ascii_columns or column_data.dtype.kind in 'SUO':
                column_data = column_data.astype('string_')
            if parse_ids and column_name == 'ID':
                id_data = self.parse_hat_source_id(column_data)
                for id_part in ['prefix', 'field', 'source']:
                    self.add_dataset(
                        dataset_key=dataset_key,
                        data=id_data[id_part],
                        **{column_substitution_name: 'hat_id_' + id_part},
                        **path_substitutions
                    )
            else:
                logging.getLogger(__name__).debug(
                    'Saving %s dataset of type: %s',
                    repr(column_name),
                    repr(column_data.dtype)
                )
                self.add_dataset(
                    dataset_key=dataset_key,
                    data=column_data,
                    **{column_substitution_name: column_name.replace('/','')},
                    **path_substitutions
                )


    def delete_sources(self,
                       dataset_key,
                       column_substitution_name,
                       **path_substitutions):
        """Delete all columns of a given source collection."""

        parent, name_head, name_tail = self._prepare_source_iter(
            dataset_key,
            column_substitution_name,
            **path_substitutions
        )
        self[parent].visit(
            partial(self.delete_columns, parent, name_head, name_tail)
        )


    def get_sources(self,
                    dataset_key,
                    column_substitution_name,
                    **path_substitutions):
        """
        Return a collection of sources previously stored in the DR file.

        Args:
            dataset_key(str):    The pipeline key for the dataset to return.

            column_substitution_name(str):    The %-subsittution variable to
                distinguish between the column in the array.

        Returns:
            dict:
                The keys are the columns of the sources stored and the values
                are 1-D numpy arrays containing the data.
        """

        parent, name_head, name_tail = self._prepare_source_iter(
            dataset_key,
            column_substitution_name,
            **path_substitutions
        )
        result = pandas.DataFrame()
        print(
            f'Collecting columns frcom {self.filename} under {parent}, '
            f'starting with {name_head} and ending with {name_tail}'
        )
        self[parent].visititems(
            partial(self.collect_columns, result, name_head, name_tail)
        )
        column_names = [colname.lower() for colname in result.columns]
        for id_colname in ['id', 'source_id']:
            if id_colname in column_names:
                result.set_index(id_colname, inplace=True)
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
            RECOGNIZED_HAT_ID_PREFIXES,
            dtype=self.get_dtype('srcproj.recognized_hat_id_prefixes')
        )

    def get_dtype(self, element_key):
        """Return numpy data type for the element with by the given key."""

        if element_key.endswith('.hat_id_prefix'):
            return h5py.special_dtype(
                enum=(
                    numpy.ubyte,
                    dict((prefix, value)
                         for value, prefix in enumerate(self._hat_id_prefixes))
                )
            )

        result = super().get_dtype(element_key)

        return result

    def parse_hat_source_id(self, source_id):
        """Return the prefix ID, field number, and source number."""

        if hasattr(source_id, 'dtype') and source_id.shape:
            id_data = {
                id_part: numpy.empty(
                    (len(source_id),),
                    dtype=id_dtype
                )
                for id_part, id_dtype in [
                    ('prefix', self.get_dtype('.hat_id_prefix')),
                    ('field', numpy.uint16),
                    ('source', numpy.uint32)
                ]
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

        path_substitutions['srcproj_column_name'] = 'hat_id_prefix'
        return self[
            self._file_structure['srcproj.columns'].abspath
            %
            path_substitutions
        ].len()

    def add_frame_header(self, header, **substitutions):
        """Add the header of the corresponding FITS frame to DR file."""

        self.write_fitsheader_to_dataset('fitsheader', header, **substitutions)

    def get_frame_header(self, **substitutions):
        """Return the header of the corresponding FITS frame."""

        return self.read_fitsheader_from_dataset('fitsheader', **substitutions)


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

    def has_shape_fit(self, accept_zeropsf=True, **path_substitutions):
        """True iff shape fitting photometry exists for path_substitutions."""

        try:
            self._check_for_dataset('shapefit.magnitude',
                                    **path_substitutions)
            return (
                accept_zeropsf
                or
                min(
                    self.get_attribute(
                        'shapefit.cfg.psf.bicubic.grid.x',
                        **path_substitutions
                    ).size,
                    self.get_attribute(
                        'shapefit.cfg.psf.bicubic.grid.y',
                        **path_substitutions
                    ).size
                ) > 2
            )
        except IOError:
            return False

    #Could not think of a reasonable way to simplify further.
    #pylint: disable=too-many-locals
    #pylint: disable=too-many-statements
    def get_source_data(self,
                        *,
                        magfit_iterations='all',
                        shape_fit=True,
                        apphot=True,
                        string_source_ids=True,
                        all_numeric_source_ids=False,
                        background=True,
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
                measurements. If ``True`` and no shape fit is present, still
                excludes shape fit columns.

            apphot(bool):    Should the result include aperture photometry
                measurements.

            string_source_ids(bool):    Should source IDs be formatted as
                strings (True) or a set of integers (False)?

            background(bool):    Should the result include information about the
                background behind the sources?

            path_substitutions:    See get_source_count().

        Returns:
            pandas.Dataframe:
                The photometry information in the current data reduction file.
                The columns always included are:

                    * ID(set as index): an array of sources IDs in the given DR
                      file. Either a string (if string_source_ids) or 1- or
                      3-column composite index depending on ID type.

                    * <catalogue quantity> (dtype as needed): one entry for each
                      catalogue column.

                    * x (numpy.float64): The x coordinates of the sources

                    * y (numpy.float64): The y coordinates of the sources

               The following columns are included if the corresponding input
               argument is set to True:

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
        """

        def assemble_hat_id(prefix, field, source):

            return f'{prefix.decode()}-{field:03d}-{source:07d}'.encode('ascii')

        def initialize_result():
            """Create the part of the result always included."""

            result = self.get_sources('srcproj.columns',
                                      'srcproj_column_name',
                                      **path_substitutions)
            logging.getLogger(__name__).debug(
                'Initial source data columns: %s',
                repr(result)
            )
            hat_id_components = ['hat_id_prefix',
                                 'hat_id_field',
                                 'hat_id_source']
            if string_source_ids:
                if result.index.name == 'source_id':
                    result['ID'] = numpy.vectorize(
                        lambda i: str(i).encode('ascii')
                    )(result.index)
                else:
                    result['ID'] = numpy.vectorize(
                        assemble_hat_id
                    )(
                        *[result[comp] for comp in hat_id_components]
                    )
                    for id_component in hat_id_components:
                        del result[id_component]
                result.set_index('ID', inplace=True)
            elif set(hat_id_components) < set(result.columns):
                if all_numeric_source_ids:
                    result.insert(0,
                                  'hat_id_prefnum',
                                  len(self._hat_id_prefixes))
                    for new_id, old_id in enumerate(self._hat_id_prefixes):
                        result.loc[
                            result['hat_id_prefix'] == old_id,
                            'hat_id_prefnum'
                        ] = new_id
                    assert (result['hat_id_prefnum'].max()
                            <
                            len(self._hat_id_prefixes))
                    hat_id_components[0] = 'hat_id_prefnum'
                result.set_index(hat_id_components, inplace=True)

            logging.getLogger(__name__).debug(
                'Source data after formatting ID: %s',
                repr(result)
            )

            return result


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
                for magfit_iter in magfit_iterations:
                    if magfit_iter == 0 or result_key != 'mag':
                        dataset_key_middle = ''
                    else:
                        dataset_key_middle = 'magfit.'
                    path_substitutions['magfit_iteration'] = magfit_iter - 1
                    column_tail = f'_mfit{magfit_iter:03d}'
                    if shape_fit:
                        result[
                            'shapefit_'
                            +
                            result_key
                            +
                            column_tail
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
                    if apphot:
                        num_apertures = self.get_num_apertures(
                            **path_substitutions
                        )
                        for aperture_index in range(num_apertures):
                            result[
                                f'ap{aperture_index:03d}_'
                                +
                                result_key
                                +
                                column_tail
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

        shape_fit = shape_fit and self.has_shape_fit(**path_substitutions)
        magfit_iterations = normalize_magfit_iterations()
        result = initialize_result()

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
        include_shape_fit = self.has_shape_fit(accept_zeropsf=False,
                                               **path_substitutions)
        add_magfit_datasets(pad_missing_magnitudes(),
                            include_shape_fit)

        add_attributes(include_shape_fit)

    def add_hat_astrometry(self,
                           filenames,
                           configuration,
                           **path_substitutions):
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

        self.add_sources(extracted_sources,
                         'srcextract.sources',
                         'srcextract_column_name',
                         **path_substitutions)
        self.add_sources(catalogue_sources,
                         'catalogue.columns',
                         'catalogue_column_name',
                         parse_ids=True,
                         **path_substitutions)
        add_match(extracted_sources, catalogue_sources)
        add_trans()
        add_configuration()


    def get_matched_sources(self, **path_substitutions):
        """Get combined catalogue and extracted matched sources."""

        match = self.get_dataset(
            dataset_key='skytoframe.matched',
            **path_substitutions
        )

        catalogue = self.get_sources(
            'catalogue.columns',
            'catalogue_column_name',
            **path_substitutions
        ).iloc[match[:, 0]].reset_index()
        extracted_sources = self.get_sources(
            'srcextract.sources',
            'srcextract_column_name',
            **path_substitutions
        ).iloc[match[:, 1]].reset_index()
        return pandas.concat(
            [catalogue, extracted_sources],
            axis=1
        )


    def save_source_extracted_psf_map(self,
                                      *,
                                      fit_results,
                                      fit_terms_expression,
                                      weights_expression,
                                      error_avg,
                                      rej_level,
                                      max_rej_iter,
                                      **path_substitutions):
        """Create the datasets and attributes holding the fit results."""

        psf_parameters = fit_results['coefficients'].keys()
        self.add_dataset('srcextract.psf_map',
                         numpy.stack(fit_results['coefficients'][param_name]
                                     for param_name in psf_parameters),
                         **path_substitutions)

        for param_key, param_value in [
                (
                    'cfg.psf_params',
                    numpy.array([name.encode('ascii') for name
                                 in psf_parameters])
                ),
                ('cfg.terms', fit_terms_expression.encode('ascii')),
                (
                    'cfg.weights',
                    (
                        b'none' if weights_expression is None
                        else weights_expression.encode('ascii')
                    )
                ),
                ('cfg.error_avg', error_avg.encode('ascii')),
                ('cfg.rej_level', rej_level),
                ('cfg.max_rej_iter', max_rej_iter),
                (
                    'residual',
                    numpy.array([
                        fit_results['fit_res2'][param_name]**0.5
                        for param_name in psf_parameters
                    ])
                ),
                (
                    'num_fit_src',
                    numpy.array([
                        fit_results['num_fit_src'][param_name]
                        for param_name in psf_parameters
                    ])
                )
        ]:
            self.add_attribute('srcextract.psf_map.' + param_key,
                               param_value,
                               **path_substitutions)
    #pylint: enable=too-many-locals
    #pylint: enable=too-many-statements

#pylint: enable=too-many-ancestors
#pylint: enable=too-many-public-methods
