"""Define a class for collecting LC data from DR files."""

import re
from multiprocessing import Value
from traceback import format_exception
import sys
import itertools

import h5py
import numpy
from numpy.lib.recfunctions import merge_arrays as merge_structured_arrays
from astropy import units
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

from superphot_pipeline.light_curves import LCDataSlice, LightCurveFile
from superphot_pipeline.magnitude_fitting import read_master_catalogue

class LCDataReader:
    """
    A callable class which gathers a slice of LC data from frames/DR files.

    Suitable for multiprocessing.

    Two instances of this class should never be created simultaneously!

    Attributes:
        %(something)s_config:    Attributes containing the configurations for
            the various components for which configuration was defined in the
            processed frames and values which are in turn dictionaries indexed
            by the configurations found for this component as frozenset suitable
            for directly passing to LightCurve.add_configuration and values
            giving the number assigned to those configurations in the
            LCDataSlice structure that was filled.

        lc_data_slice (LCDataSlice):    An object which will be filled with
            data as it is being read from data reduction files.


        dataset_dimensions (dict):    Identifiers for the dimensions of
            datasets. The keys are the pipeline keys of lightcurve datasets and
            the values are tuples containing some of: `'frame'`,
            `'source'`, `'aperture_index'`, `'magfit_iteration'`,
            `'srcextract_psf_param'` identifying what the entries in the dataset
            depend on.

        header_datasets (dict):    The set of datasets which contain FITS header
            keywords. The index is the pipeline key identifying the dataset, and
            the value is the corresponding header keyword.

        max_dimension_size (dict):    A dictionary with keys the various
            dimensions on which datasets can depend (see dataset_dimensions) and
            entries the maximum size for each dimension.

        source_destinations(dict):    Key: (field, source). Value: the offset of
            the given source in the LCDataSlice arrays that are per-source.

        config_components(dict):    Keys: all configuration components that can
            be defined. Values: 2-tuple:

                * a tuple of the keywords required to resolve the path to the
                  configuration index dataset for this component for which all
                  values found must be dumped.

                * a set of the datasets belonging to this component

        [_ra_dec]:    Optional. Only exists if one of BJD, Z or HA quantities
            are evaluated per source instead of using the values for the frame
            center.

        _config:    The configuration of how to perform the LC dumping.

        _path_substitutions:    See path_substitutions argument to create().

    """

    dataset_dimensions = dict()
    header_datasets = dict()
    config_components = dict()
    max_dimension_size = dict()
    _catalogue = dict()
    _ra_dec = []
    _path_substitutions = dict()

    @classmethod
    def _classify_datasets(cls, lc_example, ignore_splits):
        """
        Set dataset_dimensions, header_datasets, & config_components attributes.

        Args:
            lc_example(LightCurveFile):    A fully configured instance of
                LightCurveFile to serve as an example of the structure of
                lightcurve files to expect.

            ignore_splits:    See path_substitutions argument of create().

        """

        def organize_datasets():
            """Set dataset_dimensions and header_datasets attributes."""

            config_datasets = dict()
            config_components = dict()

            substitution_rex = re.compile(r'.*?%[(](?P<substitution>.*?)[)]')
            key_rex = dict(
                config=re.compile(
                    '|'.join([
                        r'_cfg_version$',
                        r'.software_versions$',
                        r'^srcextract\.binning$',
                        r'\.cfg\.|\.magfitcfg\.',
                        r'^srcextract\.psf_map\.cfg\.'
                    ])
                ),
                perframe=re.compile(
                    '|'.join([
                        r'skytoframe\.(sky_center|residual|unitarity)$',
                        r'^shapefit\.global_chi2$',
                        r'magfit\.(num_input_src|num_fit_src|fit_residual)$',
                        r'^fitsheader\.(?!cfg\.)',
                        r'\.cfg_index$',
                        r'^srcextract.psf_map.(residual|num_fit_src)$',
                    ])
                ),
                persource=re.compile(
                    '|'.join([
                        r'^srcextract\.psf_map.eval',
                        r'^srcproj\.([xy]|enabled)$',
                        r'^bg\.(value|error|npix)$',
                        r'^shapefit\.(chi2|num_pixels|signal_to_noise)$',
                        r'\.(magnitude|magnitude_error|quality_flag)$',
                        r'skypos\..*'
                    ])
                )
            )

            for lc_quantity in lc_example.elements['dataset']:
                split_quantity = lc_quantity.split('.')
                if split_quantity[0] == 'fitsheader':
                    cls.header_datasets[lc_quantity] = split_quantity[-1]

                path_template = lc_example.get_element_path(lc_quantity)
                parent = path_template.rsplit('/', 1)[0]
                dimensions = substitution_rex.findall(path_template)
                for split in ignore_splits:
                    try:
                        dimensions.remove(split)
                    except ValueError:
                        pass
                found_match = False
                for key_type, type_rex in key_rex.items():
                    if type_rex.search(lc_quantity):
                        assert not found_match
                        found_match = True
                        if key_type == 'config':
                            if parent not in config_datasets:
                                config_datasets[parent] = set()
                            config_datasets[parent].add(lc_quantity)
                        else:
                            dimensions.append('frame')
                        if key_type == 'persource':
                            dimensions.append('source')
                assert found_match
                cls.dataset_dimensions[lc_quantity] = tuple(
                    sorted(
                        set(dimensions)
                    )
                )
                if lc_quantity.endswith('.cfg_index'):
                    config_group = parent + '/Configuration'
                    assert config_group not in config_components
                    config_components[config_group] = lc_quantity[:-len('.cfg_index')]
            return config_datasets, config_components

        def organize_config(config_datasets, config_components):
            """Fill the :attr:`config_components` attribute."""

            for parent, component in config_components.items():
                component_dsets = config_datasets[parent]
                dimensions = cls.dataset_dimensions[next(iter(component_dsets))]
                for dset in component_dsets:
                    assert cls.dataset_dimensions[dset] == dimensions
                assert (cls.dataset_dimensions[component + '.cfg_index']
                        ==
                        dimensions + ('frame',))
                cls.config_components[component] = (
                    dimensions,
                    component_dsets
                )

        organize_config(*organize_datasets())

    @classmethod
    def create(cls,
               config,
               source_id_parser,
               source_list=None,
               **path_substitutions):
        """
        Configure the class for use in multiprocessing LC collection.

        Args:
            config:    The configuratoin of how to generate the lightcurves as
                returned by the rawlc_config function or as parsed from the
                command line. In the latter case, the configuration should
                contain the extra attributes:

                    - max_apertures: The maximum number of photometriec
                      apertures in any input frame.

                    - max_magfit_iterations: The maximum number of magnitude
                      fitting iterations in any input frame.

                    - catalogue_fname: The filename of a catalogue file
                      containing at least RA and Dec.

                    - srcextract_psf_params: List of the parameters describing
                      PSF shapes of the extracted sources.

                    - memblocksize: The maximum amount of memory (in bytes) to
                      allocate for temporaririly storing source information
                      before dumping to LC.

            source_id_parser:    A callable that can convert string source IDs
                to the corresponding tuple of integers source IDs.

            source_list:    A list that includes all sources for which
                lightcurves will be generated. Sources should be formatted
                as (field, source) tuples of two integers. Sources not in this
                list are ignored. If None, the sources in the catalogue are used
                instead.

            path_substitutions:    Path substitutions to be kept fixed during
                the entire lightcurve dumping process. Used to resolve paths
                both within the input data reduction files and with the
                generated lightcurves.

        Returns:
            None
        """

        if hasattr(cls, 'lc_data_slice'):
            del cls.lc_data_slice

        cls._path_substitutions = path_substitutions

        cls._catalogue = read_master_catalogue(config.catalogue_fname,
                                               source_id_parser)

        if source_list is None:
            source_list = list(cls._catalogue.keys())

        no_light_curve = LightCurveFile()

        cls.source_destinations = {source: index
                                   for index, source in enumerate(source_list)}
        cls.max_dimension_size = dict(
            source=len(source_list),
            aperture_index=config.max_apertures,
            magfit_iteration=config.max_magfit_iterations,
            srcextract_psf_param=len(config.srcextract_psf_params)
        )

        cls._classify_datasets(no_light_curve, path_substitutions.keys())

        cls.max_dimension_size['frame'] = LCDataSlice.configure(
            get_dtype=no_light_curve.get_dtype,
            dataset_dimensions=cls.dataset_dimensions,
            max_dimension_size=cls.max_dimension_size,
            max_mem=config.memblocksize
        )

        cls.lc_data_slice = Value(LCDataSlice, lock=False)
        cls._config = config
        if config.persrc.bjd or config.persrc.ha or config.persrc.z:
            cls._ra_dec = [
                (cls._catalogue[src]['ra'], cls._catalogue[src]['dec'])
                for src in source_list
            ]

        result = LCDataReader()
        return result

    #Handling the many branches is exactly the point.
    #pylint: disable=too-many-branches
    @staticmethod
    def _config_to_lc_format(lc_quantity, lc_dtype, value):
        """Return value as it would be read from the LC."""

        try:
            if value is None:
                if lc_dtype.kind == 'i':
                    result = np.iinfo(lc_dtype).min
                elif lc_dtype.kind == 'u':
                    result = np.iinfo(lc_dtype).max
                elif lc_dtype.kind == 'f':
                    result = 'NaN'
                elif h5py.check_dtype(vlen=lc_dtype) is str:
                    result = ''
                else:
                    assert False
            elif isinstance(value, str) and value == 'NaN':
                return value
            else:
                vlen_type = h5py.check_dtype(vlen=lc_dtype)
                if vlen_type is str:
                    result = str(value)
                elif vlen_type is None:
                    result = lc_dtype.type(value)
                else:
                    result = HashableArray(
                        np.array(value, dtype=vlen_type)
                    )
            return result
        except:
            raise Exception(
                "".join(format_exception(*sys.exc_info()))
                +
                '\nWhile converting to LC type: %s=%s'
                %
                (lc_quantity, repr(value))
            )
    #pylint: enable=too-many-branches

    @classmethod
    def _get_dimensions_iterator(cls, dimensions):
        """
        Return iterator over all possible values for a set of dimensions.

        Args:
            dimensions(str or iterable):    Either the name of dataset for which
                to iterate over all dimensions other than `'frame'` and
                `'source'` or a tuple contaning all dimensions to iterate over.

        Returns:
            iterator:
                Covering all possible combinatinos of values for each dimensions
                identified by `dimensions`.
        """

        if isinstance(dimensions, str):
            dimensions = filter(
                lambda dim: dim not in ['frame', 'source'],
                cls.dataset_dimensions[dimensions]
            )

        return itertools.product(
            *(range(cls.max_dimension_size[dim]) for dim in dimensions)
        )

    def _get_configurations(self,
                            data_reduction,
                            source_extracted_psf_map_order,
                            frame_header,
                            get_lc_dtype,
                            **path_substitutions):
        """
        Extract all configurations from the given data reduction file.

        Args:
            data_reduction(DataReductionFile):    An opened (at least for
                reading) data reduction file for the frame being processed.

            source_extracted_psf_map_order(int):    The maximum polynomial order
                used for smoothing the source extraction PSF map.

            frame_header(dict-like):    A pyfits header or the equivalent
                dictionary for the frame being processed.

            get_lc_dtype(callable):    Should return the data type of a dataset
                within light curves.

            path_substitutions:    See __call__().
        Returns:
            dict:
                The keys are the various components for which configuration is
                defined in the given data reduction file/frame. The values are
                lists of 2-tuples:

                    * a dictionary of the substitutions required to
                      resolve the path to the configuration index dataset

                    * a frozen set of 2-tuples:

                        - the pipeline key of the configuration quantity

                        - The value of the configuration parameter.

                The result is suitable for directly passing to
                LightCurveFile.add_configuration().
        """

        def get_component_config(component_dsets, substitutions):
            """
            Return the configuration for a component/substitution combo.

            Args:
                component_dsets(iterable of strings):    The datasets which make
                    up the configuration component.

                substitutions(dict):    Substitutions required to fully resolve
                    the light curve paths of the datasets.

            Returns:
                frozenset or None:
                    The frozen set part of the result of the parent function if
                    at least one of the datasets had an entry for the given
                    substitutions. Otherwise None.
            """

            found_config = False
            config_list = []
            for dset_key in component_dsets:
                dset_dtype = get_lc_dtype(dset_key)
                try:
                    if dset_key == 'srcextract.psf_map.order':
                        value = source_extracted_psf_map_order
                    elif dset_key in self.header_datasets:
                        assert not substitutions
                        value = frame_header[
                            self.header_datasets[dset_key]
                        ]
                    else:
                        value = data_reduction.get_attribute(
                            dset_key,
                            **substitutions,
                            **path_substitutions
                        )
                    found_config = True
                except KeyError:
                    value = None

                config_list.append(
                    (
                        dset_key,
                        self._config_to_lc_format(dset_key, dset_dtype, value)
                    )
                )

            return frozenset(config_list) if found_config else None

        result = dict()
        for component, (dimensions, component_dsets) in self.config_components:
            result[component] = []
            for dim_values in self._get_dimensions_iterator(dimensions):
                substitutions = dict(zip(dimensions, dim_values))

                configuration = get_component_config(component_dsets,
                                                     substitutions)
                if configuration is not None:
                    result[component].append(substitutions, configuration)
        return result

    @classmethod
    def _get_slice_field(cls, pipeline_key):
        """Return the field in the data slice containing the given quantity."""

        return getattr(cls.lc_data_slice, pipeline_key.replace('.', '_'))


    #Split internally into sub-functions for readability
    #pylint: disable=too-many-locals
    #pylint: disable=too-many-statements
    def _add_to_data_slice(self,
                           *,
                           data_reduction,
                           frame_header,
                           frame_index,
                           lc_example,
                           **path_substitutions):
        """
        Add the information from a single frame to the LCDataSlice.

        Args:
            data_reduction(DataReductionFile):    An opened (at least for
                reading) data reduction file for the frame being processed.

            frame_header(dict-like):    A pyfits header or the equivalent
                dictionary for the frame being processed.

            frame_index(int):    The index at which to place this frame's data
                  in the LCDataSlice object being filled.

            lc_example(LightCurveFile):    See _classify_datasets().

            path_substitutions:    See __call__().

        Returns:
            []:
                The of sources IDs (as tuples of integers) between (exclusive)
                self.last_source_below and self.first_source_above found in at
                least some of the frames which were not included in the
                LCDataSlice.
        """

        def fill_header_keywords():
            """Fill the non-config datasets equal to header keywords."""

            for lc_quantity, hdr_quantity in self.header_datasets.items():
                if 'frame' in self.dataset_dimensions[lc_quantity]:
                    assert len(self.dataset_dimensions[lc_quantity]) == 1
                    self._get_slice_field(lc_quantity)[frame_index] = (
                        frame_header[hdr_quantity]
                    )

        def get_dset_default(quantity):
            """Return a value to fill undefined entries datasets with."""

            creation_args = lc_example.get_dataset_creation_args(
                quantity
            )
            default_value = creation_args.get('fillvalue')
            if default_value is None:
                assert h5py.check_dtype(vlen=creation_args.dtype) is str
                default_value = ''
            return default_value

        def set_field_entry(quantity, value, dim_values, source_index=None):
            """
            Set the correct index within the field for the specified entry.

            Args:
                quantity(str):    The pipeline key of the dataset being filled.

                value:    The value to set to the field. Must have the correct
                    type.

                dim_values(tuple(int,...)):    The value for each dimension of
                    the dataset.

                source_index(int or None):    For source dependent datasets
                    only, the index of the source in the data_slice.

            Returns:
                None
            """

            dimensions = self.dataset_dimensions[quantity]
            if dimensions[-1] == 'source':
                assert dimensions[-2] == 'frame'
                assert source_index is not None
                dimensions = dimensions[:-2] + ('source', 'frame')
                dim_values += (source_index, frame_index)
            else:
                assert dimensions[-1] == 'frame'
                dim_values += (frame_index,)

            index = 0
            for coord, dimension in zip(dim_values, dimensions):
                index = index * self.max_dimension_size[dimension] + coord

            self._get_slice_field(quantity)[index] = value

        def fill_from_attribute(quantity):
            """Fill the value of a quantity equal to a DR attribute."""

            for dim_values in self._get_dimensions_iterator(quantity):
                set_field_entry(
                    quantity,
                    data_reduction.get_attribute(
                        quantity,
                        default_value=get_dset_default(quantity),
                        **path_substitutions,
                        **dict(
                            zip(self.dataset_dimensions[quantity], dim_values)
                        )
                    ),
                    dim_values
                )

        def fill_from_dataset(quantity, data_slice_source_indices):
            """
            Fill the value of a quantity equal to a DR dataset.

            Args:
                quantity(str):    The pipeline key of the dataset to fill.

                data_slice_source_indices:    See fill_direct_from_dr().

            Returns:
                None
            """

            num_sources = len(data_slice_source_indices)
            for dim_values in self._get_dimensions_iterator(quantity):
                dataset = data_reduction.get_dataset(
                    quantity,
                    expected_shape=(num_sources,),
                    default_value=get_dset_default(quantity),
                    **path_substitutions,
                    **dict(
                        zip(self.dataset_dimensions[quantity], dim_values)
                    )
                )

                for phot_src_ind, data_slice_src_ind in enumerate(
                        data_slice_source_indices
                ):
                    if data_slice_src_ind is None:
                        continue
                    set_field_entry(quantity,
                                    dataset[phot_src_ind],
                                    dim_values,
                                    data_slice_src_ind)

        def fill_direct_from_dr(data_slice_source_indices):
            """
            Fill all quantities coming directly from DR files.

            Args:
                data_slice_source_indices(iterable of int):    The indices
                    within data_slice of the sources in the DR currently being
                    processed

            Returns:
                None
            """

            non_header_or_config = (set(self.dataset_dimensions.keys())
                                    -
                                    set(self.header_datasets.keys()))
            for config_quantity in self.config_components.values():
                non_header_or_config -= config_quantity[1]

            for quantity in non_header_or_config:
                if 'source' not in self.dataset_dimensions[quantity]:
                    assert quantity not in data_reduction.elements['dataset']
                    assert quantity not in data_reduction.elements['link']
                    if quantity in data_reduction.elements['attribute']:
                        fill_from_attribute(quantity)
                else:
                    assert quantity not in data_reduction.elements['attribute']
                    if (
                            quantity in data_reduction.elements['dataset']
                            or
                            quantity in data_reduction.elements['link']
                    ):
                        fill_from_dataset(quantity,
                                          data_slice_source_indices)

        def fill_sky_position_datasets(data_slice_source_indices):
            """Fill all datasets with pipeline key prefix `'skypos'`."""

            num_sources = len(data_slice_source_indices)

            #False positive, pylint does not see units attributes
            #pylint: disable=no-member
            location = EarthLocation(lat=frame_header['SITELAT'] * units.deg,
                                     lon=frame_header['SITELONG'] * units.deg,
                                     height=frame_header['SITEALT'] * units.m)

            obs_time = Time(2.4e6 + frame_header['JD'],
                            format='jd',
                            location=location)

            source_coords = SkyCoord(
                ra=self._ra_dec[0] * units.deg,
                dec=self._ra_dec[1] * units.deg,
                frame='icrs'
            )

            data = dict()

            alt_az = source_coords.transform_to(
                AltAz(obstime=obs_time, location=location)
            )
            data['a180'] = 180.0  + alt_az.az.to(units.deg).value
            data['a180'][data['a180'] > 180.0] -= 360
            data['zenith_distance'] = 90.0 - alt_az.alt.to(units.deg).value
            #pylint: enable=no-member

            data['BJD'] = (obs_time.tdb
                           +
                           obs_time.light_travel_time(source_coords))
            data['hour_angle'] = (obs_time.sidereal_time('apparent')
                                  -
                                  self._ra_dec[0] / 15.0)
            data['per_source'] = numpy.ones((num_sources,), dtype=numpy.bool)

            for source_index in range(num_sources):
                for quantity, values in data.items():
                    set_field_entry('skypos.' + quantity,
                                    values[source_index],
                                    source_index)

        def merge_with_catalogue_information(source_data):
            """Extend source_data with catalogue information."""

            num_sources = len(source_data)
            cat_data = numpy.empty(
                num_sources,
                dtype=next(iter(self._catalogue.values())).dtype
            )
            for source_index, source_id in enumerate(source_data['ID']):
                cat_data[source_index] = self._catalogue[source_id]

            return merge_structured_arrays(source_data, cat_data)

        def fill_srcextract_psf_map(source_data,
                                    data_slice_source_indices):
            """Fill all datasets containing the source exatrction PSF map."""

            psf_map = data_reduction.get_source_extracted_psf_map(
                **path_substitutions
            )
            psf_param_values = psf_map(source_data)

            assert(set(psf_param_values.dtype.names)
                   ==
                   set(self._config.srcextract_psf_params))

            for param_index, param_name in enumerate(
                    self._config.srcextract_psf_params
            ):
                for data_slice_src_ind, value in zip(
                        data_slice_source_indices,
                        psf_param_values[param_name]
                ):
                    if data_slice_src_ind is None:
                        continue
                    set_field_entry('srcextract.psf_map.eval',
                                    value,
                                    (param_index,),
                                    data_slice_src_ind)

        fill_header_keywords()
        source_data = data_reduction.get_source_data(
            string_source_ids=False,
            shape_fit=False,
            apphot=False,
            shape_map_variables=False,
            background=False,
            **path_substitutions
        )
        source_data = merge_with_catalogue_information(source_data)

        data_slice_source_indices = [self.source_destinations.get(src_id)
                                     for src_id in source_data['ID']]
        fill_direct_from_dr(data_slice_source_indices)
        fill_sky_position_datasets(data_slice_source_indices)
        fill_srcextract_psf_map(source_data, data_slice_source_indices)

        skipped_sources = []
        for src, slice_ind in zip(source_data['ID'],
                                  data_slice_source_indices):
            if slice_ind is None:
                skipped_sources.append(src)

        return skipped_sources
    #pylint: enable=too-many-locals
    #pylint: enable=too-many-statements

    def __call__(self, frame):
        """
        Add single frame's information to configurations and the LCDataSlice.

        Args:
            - frame: A 3-tulpe containing the following:

                - The filename of the DR file to read.

                - The index at which to place this frame's data in the
                  LCDataSlice object being filled.

            path_substitutions:    Any %-substitution arguments required to
                fully resolve the data being dumped to light curves for which
                not all values are being dumped within a DR file, and
                respectively the destination for the data in the LC file. The
                keys should be exactly the ignore_splits arguments passed to
                create().


        Returns: None.
        """

        def fix_header(frame_header):
            """Patch problems with FITS headers and adds filename info."""

            if 'COMPOSIT' not in frame_header:
                frame_header['COMPOSIT'] = 1
            if 'FILVER' not in frame_header:
                frame_header['FILVER'] = 0
            if 'MNTSTATE' not in frame_header:
                frame_header['MNTSTATE'] = 'unkwnown'
            #This is actually intended to fail if MNTSTAT is not bool type
            #pylint: disable=singleton-comparison
            elif frame_header['MNTSTATE'] == True:
                frame_header['MNTSTATE'] = 'T'
            #pylint: enable=singleton-comparison

        try:
            lc_example = LightCurveFile()
            dr_fname, frame_index = frame
            with DataReduction(dr_fname, 'r') as data_reduction:
                frame_header = dict(data_reduction.get_fits_header().iteritems())
                fix_header(frame_header)
                configurations = self._get_configurations(data_reduction,
                                                          psf_map['order'],
                                                          frame_header,
                                                          lc_example.get_dtype)

                skipped_sources = self._add_to_data_slice(
                    data_reduction=data_reduction,
                    frame_header=frame_header,
                    frame_index=frame_index,
                    lc_example=lc_example,
                    **path_substitutions
                )
            return configurations, skipped_sources
        except:
            raise Exception(
                "".join(format_exception(*sys.exc_info()))
                +
                '\nWhile processing: ' + repr(frame)
            )
