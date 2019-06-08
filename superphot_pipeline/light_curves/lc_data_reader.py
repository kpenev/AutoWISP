"""Define a class for collecting LC data from DR files."""

import re
from multiprocessing import Value
from traceback import format_exception
import sys
import itertools

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
            `'source'`, `'aperture_index'`, `'magfit_iteration'` identifying
            what the entries in the dataset depend on.

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
                  configuration index dataset for this component.

                * a set of the datasets belonging to this component

        [_ra_dec]:    Optional. Only exists if one of BJD, Z or HA quantities
            are evaluated per source instead of using the values for the frame
            center.

        _config:    The configuration of how to perform the LC dumping.

    """

    dataset_dimensions = dict()
    header_datasets = dict()
    config_components = dict()
    max_dimension_size = dict()

    @classmethod
    def _classify_datasets(cls, lc_example):
        """
        Set dataset_dimensions, header_datasets, & config_components attributes.

        Args:
            lc_example(LightCurveFile):    A fully configured instance of
                LightCurveFile to serve as an example of the structure of
                lightcurve files to expect.
        """

        config_datasets = set()
        config_components = dict()

        substitution_rex = re.compile(r'.*?%[(](?P<substitution>.*?)[)]')
        key_rex = dict(
            config='|'.join([
                r'_cfg_version$',
                r'.software_versions$',
                r'sdkmap\.order$',
                r'^srcextract\.binning$',
                r'\.cfg\.|\.magfitcfg\.',
                r'^srcextract\.sdkmap\.scale$'
            ]),
            perframe='|'.join([
                r'skytoframe\.(sky_center|residual|unitarity)$',
                r'^shapefit\.global_chi2$',
                r'magfit\.(num_input_src|num_fit_src|fit_residual)$',
                r'^fitsheader\.(?!cfg\.)',
                r'\.cfg_index$'
            ]),
            persource='|'.join([
                r'^srcextract\.sdkmap\.[sdk]',
                r'^srcproj\.([xy]|enabled)$',
                r'^bg\.(value|error|npix)$',
                r'^shapefit\.(chi2|num_pixels|signal_to_noise)$',
                r'\.(magnitude|magnitude_error|quality_flag$'
            ])
        )

        for lc_quantity in lc_example.elements['dataset']:
            split_quantity = lc_quantity.split('.')
            if split_quantity[0] == 'fitsheader':
                cls.header_datasets[lc_quantity] = split_quantity[-1]

            path_template = lc_example.get_element_path(lc_quantity)
            parent = path_template.rsplit('/', 1)[0]
            dimensions = substitution_rex.findall(path_template)
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
            cls.dataset_dimensions[lc_quantity] = tuple(sorted(set(dimensions)))
            if lc_quantity.endswith('.cfg_index'):
                assert parent not in config_components
                config_components[parent] = lc_quantity[:-len('.cfg_index')]

        for parent, component in config_components:
            component_dsets = config_datasets[parent]
            dimensions = cls.dataset_dimensions[next(iter(dsets))]
            for dset in component_dsets:
                assert cls.dataset_dimensions[dset] == dimensions
            assert (cls.dataset_dimensions[component + '.cfg_index']
                    ==
                    dimensions)
            cls.config_components[component] = (
                dimensions,
                dsets
            )

    @classmethod
    def create(cls,
               config,
               source_list,
               source_id_parser,
               *,
               source_range=(None, None)):
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

                    - catalogue_columns: List of the columns in the catalogue
                      file. The two catalogue attributes are only necessary if
                      BJD, HA or Z are being evaluated on a per-source basis.

            source_list:    A list that includes all sources for which
                lightcurves will be generated. Sources should be formatted
                as (field, source) tuples of two integers. Sources not in this
                list are ignored.

            source_id_parser:    A callable that can convert string source IDs
                to the corresponding tuple of integers source IDs.

            The following arguments are never used if config contains the
            extra arguments for command line configuration, but must be
            supplied if database configuration is used.

            - project_id:    The project for which lightcurves are being
                created.

            - sphotref_id:    The single photometric reference ID.

            - db:    A calibration database object.
            - track_skipped_sources:    Should objects keep track of sources for
                which no lightcurves were created because they were not in
                source_list?

        Returns:
            None
        """

        if hasattr(cls, 'lc_data_slice'):
            del cls.lc_data_slice

        no_light_curve = LightCurveFile()

        cls.source_destinations = {source: index
                                   for index, source in enumerate(source_list)}
        cls.max_dimension_size = dict(
            source=len(source_list),
            aperture_index=config.max_apertures,
            magfit_iteration=config.max_magfit_iterations
        )

        cls._classify_datasets(no_light_curve)

        cls.max_dimension_size['frame'] = LCDataSlice.configure(
            get_dtype=no_light_curve.get_dtype,
            dataset_dimensions=cls.dataset_dimensions,
            max_dimension_size=cls.max_dimension_size,
            max_mem=config.memblocksize
        )

        cls.lc_data_slice = Value(LCDataSlice, lock=False)
        cls._config = config
        if config.persrc.bjd or config.persrc.ha or config.persrc.z:
            catalogue = read_master_catalogue(config.catalogue.fname,
                                              source_id_parser)
            cls._ra_dec = [(catalogue[src]['ra'], catalogue[src]['dec'])
                           for src in source_list]
        else:
            cls._ra_dec = None

        cls.last_source_below, cls.first_source_above = source_range

        result = LCDataReader()
        result.reset()
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

    def _get_configurations(self, data_reduction, frame_header, get_lc_dtype):
        """
        Extract all configurations from the given data reduction file.

        Args:
            data_reduction(DataReductionFile):    An opened (at least for
                reading) data reduction file for the frame being processed.

            frame_header(dict-like):    A pyfits header or the equivalent
                dictionary for the frame being processed.

            get_lc_dtype(callable):    Should return the data type of a dataset
                within light curves.

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
                    if dset_key in self.header_datasets:
                        assert not substitutions
                        value = frame_header[
                            self.header_datasets[dset_key]
                        ]
                    else:
                        value = data_reduction.get_attribute(
                            dset_key,
                            **substitutions
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
            for dim_values in itertools.product(
                    *(
                        range(self.max_dimension_size[dim])
                        for dim in dimensions
                    )
            ):
                substitutions = dict(zip(dimensions, dim_values))

                configuration = get_component_config(component_dsets,
                                                     substitutions)
                if configuration is not None:
                    result[component].append(substitutions, configuration)

    def __add_to_data_slice(self,
                            data_reduction,
                            frame_header,
                            source_extracted_psf_map,
                            frame_center,
                            data_slice_index) :
        """
        Add the information from a single frame to the LCDataSlice.

        Args:
            - data_reduction: An opened (at least for reading) data reduction
                              file for the frame being processed.
            - frame_header: A pyfits header or the equivalent dictionary for
                            the frame being processed.
            - source_extracted_psf_map: See __call__
            - frame_center: Dictionary with the following:
                - jd: The JD when the frame was observed.
                - bjd: BJD for the frame center.
                - zenith_distance: Zenith distance of the frame center.
                - hour_angle: Hour angle of the frame center.
                - ra: RA of the frame center.
            - data_slice_index: See __call__

        Returns: The list of sources between (exclusive)
                 self.last_source_below and self.first_source_above found in
                 at least some of the frames which were not includedj in the
                 LCDataSlice.
        """

        def add_header_keywords() :
            """Fill the entries directly equal to header keywords."""

            for lc_quantity, hdr_quantity \
                    in \
                    self.__header_datasets.iteritems() :
                getattr(
                    self.lc_data_slice, lc_quantity.replace('.', '_')
                )[data_slice_index]=frame_header[hdr_quantity]

        def add_global_magfit_quantities(num_apertures, has_psffit) :
            """Fill the global (per frame) magfit quantities."""

            enabled_mode_chars=[]
            if self.require_single_magfit : enabled_mode_chars.append('s')
            if self.require_master_magfit : enabled_mode_chars.append('m')
            for mode_char in enabled_mode_chars :
                ap_ind = 0
                phot_mode = 'psffit' if has_psffit else 'apphot'
                while ap_ind < num_apertures :
                    getattr(
                        self.lc_data_slice,
                        phot_mode + '_' + mode_char + 'pr_fit_residual'
                    )[ap_ind][data_slice_index] = (
                        data_reduction.get_attribute(
                            phot_mode + '.' + mode_char + 'prmagfit_residual',
                            ap_ind = ap_ind,
                            default_value = (
                                np.nan if (
                                    mode_char == 's'
                                    and
                                    self.require_single_magfit == 'optional'
                                ) else None
                            )
                        )
                    )
                    getattr(
                        self.lc_data_slice,
                        phot_mode + '_' + mode_char + 'pr_fit_nsources'
                    )[ap_ind][data_slice_index]=(
                        data_reduction.get_attribute(
                            phot_mode + '.' + mode_char + 'prmagfit_fit_src',
                            ap_ind=ap_ind,
                            default_value = (
                                0 if (
                                    mode_char == 's'
                                    and
                                    self.require_single_magfit == 'optional'
                                ) else None
                            )
                        )
                    )
                    if phot_mode == 'psffit':
                        phot_mode = 'apphot'
                    else:
                        ap_ind += 1

        add_header_keywords()
        photometry = data_reduction.get_photometry(
            raw=True,
            single_fit=self.require_single_magfit,
            master_fit=self.require_master_magfit
        )
        has_psffit = (photometry['photometry'][0]['type'] == 'psffit')
        num_apertures = len(photometry['photometry']) - (1 if has_psffit else 0)
        assert(num_apertures<=self.__max_apertures)
        add_global_magfit_quantities(num_apertures, has_psffit)
        if self.__config.persrc.bjd :
            source_bjds=bjd(self.__ra_dec, frame_center['jd'])
        else : source_bjds=ConstList(frame_center['bjd'])
        psf_param=dict()
        for param_name in source_extracted_psf_map.keys() :
            psf_param[param_name]=source_extracted_psf_map[param_name](
                photometry['x'],
                photometry['y']
            )
        skipped_sources = set()
        for phot_source_ind, source in enumerate(photometry['sources']) :
            if source not in self.source_destinations :
                if (
                        (
                            self.last_source_below is None
                            or
                            source > self.last_source_below
                        )
                        and
                        (
                            self.first_source_above is None
                            or
                            source < self.first_source_above
                        )

                ) :
                    skipped_sources.add(source)
                continue
            lc_source_ind=self.source_destinations[source]
            x=photometry['x'][phot_source_ind]
            y=photometry['y'][phot_source_ind]
            self.lc_data_slice.astrometry_x\
                    [lc_source_ind][data_slice_index]=x
            self.lc_data_slice.astrometry_y\
                    [lc_source_ind][data_slice_index]=y
            self.lc_data_slice.bg_value[lc_source_ind][data_slice_index]=(
                photometry['bg'][phot_source_ind]
            )
            self.lc_data_slice.bg_error[lc_source_ind][data_slice_index]=(
                photometry['bg err'][phot_source_ind]
            )
            self.lc_data_slice.astrometry_bjd\
                    [lc_source_ind][data_slice_index]=(
                        source_bjds[lc_source_ind]
                    )
            if self.__ra_dec is not None :
                ra, dec=self.__ra_dec[lc_source_ind]
            if self.__config.persrc.ha :
                hour_angle=(frame_center['hour_angle']
                            +
                            (frame_center['ra']-ra)/15.0)%24
                if hour_angle>12.0 : hour_angle-=24.0
            else : hour_angle=frame_center['hour_angle']
            self.lc_data_slice.astrometry_hour_angle\
                    [lc_source_ind][data_slice_index]=(
                        hour_angle
                    )
            if self.__config.persrc.z :
                zenith_distance=(
                    90.0
                    -
                    hrz_from_equ(ra,
                                 dec,
                                 frame_header['SITELAT'],
                                 frame_header['SITELONG'],
                                 frame_center['jd'])[1]
                )
            else : zenith_distance=frame_center['zenith_distance']
            self.lc_data_slice.astrometry_zenith_distance\
                    [lc_source_ind][data_slice_index]=(
                        zenith_distance
                    )
            for psf_param_name in psf_param.keys() :
                getattr(
                    self.lc_data_slice,
                    'srcfind_' + psf_param_name
                )[lc_source_ind][data_slice_index]=(
                    psf_param[psf_param_name][phot_source_ind]
                )
            magnitudes_to_transfer=[('mag', 'mag'),
                                    ('mag err', 'mag_err')]
            if self.require_single_magfit :
                magnitudes_to_transfer.append(
                    ('single fit mag', 'spr_fit_mag')
                )
            if self.require_master_magfit :
                magnitudes_to_transfer.append(
                    ('master fit mag', 'mpr_fit_mag')
                )
            for ap_ind in range(num_apertures + (1 if has_psffit else 0)):
                for source, dest in magnitudes_to_transfer :
                    if (
                        source == 'single fit mag'
                        and
                        self.require_single_magfit == 'optional'
                        and
                        source not in photometry['photometry'][ap_ind]
                    ) :
                        value = np.nan
                    else : value = photometry['photometry'][ap_ind][source]\
                            [phot_source_ind]
                    getattr(
                        self.lc_data_slice,
                        'apphot_'+dest
                    )[lc_source_ind][ap_ind][data_slice_index] = value
                self.lc_data_slice.apphot_quality\
                        [lc_source_ind][ap_ind][data_slice_index]=int(
                            photometry['photometry'][ap_ind]['status flag']\
                                      [phot_source_ind]
                        )
        return skipped_sources

    def reset(self) :
        """
        Fill background in LC slice with nans for detecting undefined values.
        """

        np.frombuffer(self.lc_data_slice.bg_value, self.__bg_dtype)[:]=np.nan

    def fistar_sdk_map(self, data_reduction) :
        """
        Return dictionary of functions evaluating the fistar PSF map in DR.

        Args:
            - data_reduction: An open DataReduction instance.
        Returns: A dictionary with keys s, d and k, each of which is a
                 function of (x,y) which evaluates to the corresponding PSF
                 parameter.
        """

        if not self.require_srcfind_psfmap :
            return dict([
                (v, lambda x, y: np.nan*x) for v in ['s', 'd', 'k']
            ])
        x_scale, y_scale=data_reduction.get_attribute('srcfind.psfmap_scale')
        x_offset, y_offset=data_reduction.get_attribute(
            'srcfind.psfmap_offset'
        )
        sdk_coef=data_reduction.get_single_dataset('srcfind.psfmap_coef')
        return dict(zip(
            ['s', 'd', 'k'],
            [make_polynomial_function(x_offset, y_offset,
                                      x_scale, y_scale,
                                      coef)
             for coef in sdk_coef]
        ))

    def __call__(self, frame) :
        """
        Add single frame's information to configurations and the LCDataSlice.

        Args:
            - frame: A 3-tulpe containing the following:
                - record: A tuple identifying the frame to process,
                          containing either:
                          (filename, JD, BJD, zenith distance, hour angle,
                           right ascention)
                           or
                          (station_id, fnum, cmpos, night, JD, BJD,
                           zenith distance, hour angle, right ascention).
                - data_slice_index: The index at which to place this frame's
                                    data in the LCDataSlice object being
                                    filled.

        Returns: None.
        """

        def fix_header(frame_header) :
            """Patch problems with FITS headers and adds filename info."""

            if direct_dr_fname :
                try :
                    stid, frame_header['CMPOS'], frame_header['FNUM']=(
                        parse_frame_name(direct_dr_fname)
                    )
                except : pass
            else :
                frame_header['FNUM']=record[1]
                frame_header['CMPOS']=record[2]
                frame_header['NIGHT']=record[3]
            if 'COMPOSIT' not in frame_header : frame_header['COMPOSIT']=1
            if 'FILVER' not in frame_header : frame_header['FILVER']=0
            if 'MNTSTATE' not in frame_header :
                frame_header['MNTSTATE']='unkwnown'
            elif frame_header['MNTSTATE']==True :
                frame_header['MNTSTATE']='T'

        try :
            record, data_slice_index=frame
            if len(record)==6 :
                data_reduction=DataReduction(record[0], 'r')
                direct_dr_fname=record[0]
            else :
                assert(len(record)==9)
                direct_dr_fname=False
                data_reduction=DataReduction.from_image_vars(
                    stid=record[0],
                    fnum=record[1],
                    cmpos=record[2],
                    image_type=self.image_type,
                    night=record[3],
                    project_id=self.project_id,
                    mode='r'
                )
            frame_center=dict(zip(
                ('jd', 'bjd', 'zenith_distance', 'hour_angle', 'ra'),
                record[-5:]
            ))
            frame_header=dict(data_reduction.get_fits_header().iteritems())
            fix_header(frame_header)
            source_extracted_psf_map=self.fistar_sdk_map(data_reduction)
            configurations=self.__get_configurations(data_reduction,
                                                     frame_header)
            skipped_sources = self.__add_to_data_slice(
                data_reduction,
                frame_header,
                source_extracted_psf_map,
                frame_center,
                data_slice_index
            )
            data_reduction.close()
            return configurations, skipped_sources
        except : raise Exception(
            "".join(format_exception(*sys.exc_info()))
            +
            '\nWhile processing: ' + repr(frame)
        )


