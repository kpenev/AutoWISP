#pylint: disable=too-many-lines
"""Define function to add defaults to all light curve structure tables."""

#TODO: Figure out proper structure with multiple versions of all components.

#TODO: Add configuration index datasets under:
#   - Background
#   - SkyToFrameTransformation

import re

import numpy

#Pylint false positive due to quirky imports.
#pylint: disable=no-name-in-module
from superphot_pipeline.database.data_model import\
    HDF5Product,\
    HDF5StructureVersion,\
    HDF5Attribute,\
    HDF5DataSet
#pylint: enable=no-name-in-module

from superphot_pipeline.database.initialize_data_reduction_structure import\
    _default_paths as _dr_default_paths

_version_rex = re.compile(r'/Version%\([a-zA-Z_]*\)[0-9]*d')

_default_paths = dict(
    sdk_map='/SourceExtractionSDKMap',
    catalogue='/SkyToFrameTransformation',
    magfit='/MagnitudeFitting'
)

def _get_structure_version_id(db_session, product='data_reduction'):
    return db_session.query(
        HDF5StructureVersion.hdf5_structure_version_id
    ).filter(
        HDF5StructureVersion.hdf5_product_id == HDF5Product.hdf5_product_id
    ).filter(
        HDF5Product.pipeline_key == product
    ).one()[0]

def _get_source_extraction_datasets():
    """Create the default datasets for source extraction data."""

    sdk_map_key_start = 'srcextract.sdkmap.'

    def get_configuration_datasets():
        """Return the datasets containing the config. for source exatraction."""

        config_path_start = _default_paths['sdk_map'] + '/Configuration/'

        return [
            HDF5DataSet(
                pipeline_key=(sdk_map_key_start
                              +
                              'srcextract_cfg_version'),
                abspath=config_path_start + 'ConfigurationVersion',
                dtype='numpy.uint',
                replace_nonfinite=repr(numpy.iinfo('u4').max),
                description='The configuration version ID from the database '
                'for the source extraction used for this frame.'
            ),
            HDF5DataSet(
                pipeline_key=sdk_map_key_start + 'software_versions',
                abspath=config_path_start + 'SoftwareVersions',
                dtype="'S100'",
                description='An Nx2 array of strings consisting of software '
                'elements and their versions used for source extraction.',
            ),
            HDF5DataSet(
                pipeline_key=sdk_map_key_start + 'order',
                abspath=config_path_start + 'SpatialOrder',
                dtype='numpy.uint',
                replace_nonfinite=repr(numpy.iinfo('u4').max),
                description='The maximum total order of the spatial terms '
                'included in the smoothed PSF map from source extraction.',
            )
        ]

    def get_per_source_datasets():
        """Return datasets containing source extraction data for each source."""

        return [
            HDF5DataSet(
                pipeline_key=sdk_map_key_start + psf_parameter.lower(),
                abspath=_default_paths['sdk_map'] + '/' + psf_parameter,
                dtype='numpy.float64',
                scaleoffset=4,
                replace_nonfinite=repr(numpy.finfo('f4').min),
                description='The values of the %s parameter for elliptical '
                'gaussion PSF for each source.' % psf_parameter,
            )
            for psf_parameter in 'SDK'
        ]

    return get_configuration_datasets() + get_per_source_datasets()

def _get_catalogue_attributes():
    """Create the attributes for catalogue information for this source."""

    key_start = 'catalogue.'
    parent = '/'

    return [
        HDF5Attribute(
            pipeline_key=key_start + 'name',
            parent=parent,
            name='Catalogue',
            dtype='numpy.string_',
            description='The catalogue from which this source information was '
            'queried.'
        ),
        HDF5Attribute(
            pipeline_key=key_start + 'epoch',
            parent=parent,
            name='CatalogueEpoch',
            dtype='numpy.string_',
            description='The epoch (JD) up to which catalogue positions were '
            'corrected.'
        ),
        HDF5Attribute(
            pipeline_key=key_start + 'information',
            parent=parent,
            name='Catalogue_%(catalogue_column_name)s',
            dtype='manual',
            replace_nonfinite=repr(numpy.finfo('f4').min),
            description='A single catalogue value for this source.'
        )
    ]

def _get_catalogue_datasets():
    """Return the datasets describing source projection."""

    key_start = 'catalogue.cfg.'

    def get_configuration_datasets():
        """Create datasets for config. of sky-tosframe and source projection."""

        config_path_start = _default_paths['catalogue'] + '/Configuration/'

        return [
            HDF5DataSet(
                pipeline_key=key_start + 'name',
                abspath=config_path_start + 'CatalogueName',
                dtype='numpy.string_',
                description='The catalogue used for astrometry.'
            ),
            HDF5DataSet(
                pipeline_key=key_start + 'filter',
                abspath=config_path_start + 'CatalogueFilter',
                dtype='numpy.string_',
                description='Any filtering applied to catalogue sources before '
                'matching.'
            )
        ]

    return get_configuration_datasets() + get_per_source_datasets()

def _get_frame_datasets():
    """Return all datasets containing FITS header keywords."""

    def get_per_frame_datasets():
        """Return the datasets of header keywords with one entry per LC pt."""

        result = []
        for keyword, dset_name, dtype, default, description in [
                (
                    'FNUM',
                    'FrameNumber',
                    'numpy.uint',
                    repr(numpy.iinfo('u4').max),
                    'The number of the frame corresponding to this datapoint in'
                    ' the light curve.'
                ),
                (
                    'FOCUS',
                    'FocusSetting',
                    'numpy.float64',
                    repr(numpy.finfo('f4').min),
                    'The focus setting of the telescope for this observation.'
                ),
                (
                    'WIND',
                    'WindSpeed',
                    'numpy.float64',
                    repr(numpy.finfo('f4').min),
                    'The wind speed in m/s'
                ),
                (
                    'WINDDIR',
                    'WindDirection',
                    'numpy.float64',
                    repr(numpy.finfo('f4').min),
                    'Wind direction [degrees] reported for this observation.'
                ),
                (
                    'AIRPRESS',
                    'AirPressure',
                    'numpy.float64',
                    repr(numpy.finfo('f4').min),
                    'Air pressure [Pa] for this observation.'
                ),
                (
                    'AIRTEMP',
                    'AirTemperature',
                    'numpy.float64',
                    repr(numpy.finfo('f4').min),
                    'Air temperature [C]'
                ),
                (
                    'HUMIDITY',
                    'Humidity',
                    'numpy.float64',
                    repr(numpy.finfo('f4').min),
                    'Relative humidity [%]'
                ),
                (
                    'DEWPT',
                    'DewPoint',
                    'numpy.float64',
                    repr(numpy.finfo('f4').min),
                    'Dew point [C]'
                ),
                (
                    'SUNDIST',
                    'SunDistance',
                    'numpy.float64',
                    repr(numpy.finfo('f4').min),
                    'Distance from Sun [deg] (frame center)'
                ),
                (
                    'SUNELEV',
                    'SunElevation',
                    'numpy.float64',
                    repr(numpy.finfo('f4').min),
                    'Elevation of Sun [deg]'
                ),
                (
                    'MOONDIST',
                    'MoonDistance',
                    'numpy.float64',
                    repr(numpy.finfo('f4').min),
                    'Distance from Moon [deg] (frame center)'
                ),
                (
                    'MOONPH',
                    'MoonPhase',
                    'numpy.float64',
                    repr(numpy.finfo('f4').min),
                    'Phase of Moon'
                ),
                (
                    'MOONELEV',
                    'MoonElevation',
                    'numpy.float64',
                    repr(numpy.finfo('f4').min),
                    'Elevation of Moon [deg]'
                )
        ]:
            args = dict(
                pipeline_key='fitsheader.' + keyword.lower(),
                abspath='/FrameInformation/' + dset_name,
                dtype=dtype,
                replace_nonfinite=default
                description=description
            )
            if dtype == 'numpy.float64':
                args['scaleoffset'] = 3
            else:
                args['compression'] = 'gzip'
                args['compression_options'] = '9'

            result.append(HDF5DataSet(**args))

        return result

    def get_config_datasets():
        """Return the datasets of header keywords treated as configuration."""

        result = []
        for keyword, dset_name, dtype, scaleoffset, description in [
                (
                    'STID',
                    'StationID',
                    'numpy.uint',
                    None,
                    'ID of station that took this observation'
                ),
                (
                    'CMPOS',
                    'CameraPosition',
                    'numpy.uint',
                    None,
                    'ID of the position of the camera on a multi-telescope '
                    'mount'
                ),
                (
                    'COLOR',
                    'ColorChannel',
                    'numpy.string_',
                    None,
                    'The color of the channel contributing this datapoint.'
                ),
                (
                    'SITEID',
                    'SiteID',
                    'numpy.uint',
                    None,
                    'ID of the site where this observation took place.'
                ),
                (
                    'SITELAT',
                    'SiteLatitude',
                    'numpy.float64',
                    6,
                    'Observing site latitude [deg].'
                ),
                (
                    'SITELONG',
                    'SiteLongitude',
                    'numpy.float64',
                    6,
                    'Observing site longitude [deg].'
                ),
                (
                    'SITEALT',
                    'SiteALtitude',
                    'numpy.float64',
                    3,
                    'Observing site altitude above sea level [m].'
                ),
                (
                    'MTID',
                    'MointID',
                    'numpy.uint',
                    None,
                    'ID of the mount used for this observing session.'
                ),
                (
                    'MTVER',
                    'MointVersion',
                    'numpy.uint',
                    None,
                    'Version of the mount used for this observing session.'
                ),
                (
                    'CMID',
                    'CameraID',
                    'numpy.uint',
                    None,
                    'ID of the camera used for this observing session.'
                ),
                (
                    'CMVER',
                    'CameraVersion',
                    'numpy.uint',
                    None,
                    'Version of the camera used for this observing session,'
                ),
                (
                    'TELID',
                    'TelescopeID',
                    'numpy.uint',
                    None,
                    'ID of the telescopes used for this observing session.'
                ),
                (
                    'TELVER',
                    'TelescopeVersion',
                    'numpy.uint',
                    None,
                    'Version of the telescopes used for this observing session.'
                ),
                (
                    'MNTSTATE',
                    'PSFBroadeningPattern',
                    'numpy.string_',
                    None,
                    'The PSF broadening pattern followed during exposure.'
                ),
                (
                    'PROJID',
                    'ProjectID',
                    'numpy.uint',
                    None,
                    'ID of the project this observing session is part of.'
                ),
                (
                    'NRACA',
                    'TargetedRA',
                    'numpy.float64',
                    3,
                    'Nominal RA of midexpo [hr] (averaged field center)'
                ),
                (
                    'NDECCA',
                    'TargetedDec',
                    'numpy.float64',
                    3,
                    'Nominal Dec of midexpo [hr] (averaged field center)'
                ),
        ]:
            args = dict(
                pipeline_key='fitsheader.cfg.' + keyword.lower(),
                abspath='/FrameInformation/Configuration/' + dset_name,
                dtype=dtype,
                description=description
            )
            if scaleoffset is None:
                args['compression'] = 'gzip'
                args['compression_options'] = '9'
            else:
                args['scaleoffset'] = scaleoffset

            if dtype == 'numpy.float64':
                args['replace_nonfinite'] = repr(numpy.finfo('f4').min)

            result.append(HDF5DataSet(**args))

        return result

    return get_per_frame_datasets() + get_config_datasets()

def transform_dr_to_lc_path(dr_path):
    """Return the path in a light curve file corresponding to a DR file path."""

    result = re.sub(_version_rex, '', dr_path)
    for dr_string, lc_string in [
            ('/FittedMagnitudes', _default_paths['magfit']),
            ('/ProjectedSources', '/ProjectedPosition'),
            ('/SourceExtractionPSFMap', '/SourceExtractionPSF'),
            ('/ProjectedToFrameMap', ''),
            ('/SourceExtraction/SDKMap', '/SourceExtraction'),
            ('/SourceExtraction', _default_paths['sdk_map']),
            ('/CatalogueSources', _default_paths['astrometry'])
    ]:
        result = re.sub(dr_string, lc_string, result)

    return result

def _get_data_reduction_attribute_datasets(db_session):
    """Return all datasets from attributes in data reduction files."""

    dr_structure_version_id = _get_structure_version_id(db_session)
    result = []
    for pipeline_key, scaleoffset, is_config in [
            ('skytoframe.sky_center', 5, False),
            ('skytoframe.residual', 2, False),
            ('skytoframe.unitarity', 5, False),
            ('shapefit.global_chi2', 2, False),
            ('shapefit.magfit.num_input_src', None, False),
            ('shapefit.magfit.num_fit_src', None, False),
            ('shapefit.magfit.fit_residual', 2, False),
            ('apphot.magfit.num_input_src', None, False),
            ('apphot.magfit.num_fit_src', None, False),
            ('apphot.magfit.fit_residual', 2, False),
            ('srcextract.binning', None, True),
            ('srcextract.sdkmap.scale', 3, True),
            ('srcextract.sdkmap.offset', 3, True),
            ('skytoframe.cfg.srcextract_filter', None, True),
            ('skytoframe.cfg.sky_preprojection', None, True),
            ('skytoframe.cfg.frame_center', 3, True),
            ('skytoframe.cfg.max_match_distance', 3, True),
            ('skytoframe.cfg.weights_expression', None, True),
            ('bg.cfg.zero', None, True),
            ('bg.cfg.model', None, True),
            ('bg.cfg.annulus', None, True),
            ('shapefit.cfg.gain', 3, True),
            ('shapefit.cfg.magnitude_1adu', 5, True),
            ('shapefit.cfg.psf.model', None, True),
            ('shapefit.cfg.psf.terms', None, True),
            ('shapefit.cfg.psf.max-chi2', 3, True),
            ('shapefit.cfg.psf.min_convergence_rate', 3, True),
            ('shapefit.cfg.psf.max_iterations', None, True),
            ('shapefit.cfg.psf.ignore_dropped', None, True),
            ('shapefit.cfg.src.cover_bicubic_grid', None, True),
            ('shapefit.cfg.src.min_signal_to_noise', 3, True),
            ('shapefit.cfg.src.max_aperture', 3, True),
            ('shapefit.cfg.src.max_sat_frac', 3, True),
            ('shapefit.cfg.src.min_pix', None, True),
            ('shapefit.cfg.src.max_pix', None, True),
            ('shapefit.cfg.src.min_bg_pix', None, True),
            ('shapefit.cfg.src.max_count', None, True),
            ('shapefit.cfg.psf.bicubic.grid.x', None, True),
            ('shapefit.cfg.psf.bicubic.grid.y', None, True),
            ('shapefit.cfg.psf.bicubic.pixrej', 3, True),
            ('shapefit.cfg.psf.bicubic.initial_aperture', 3, True),
            ('shapefit.cfg.psf.bicubic.max_rel_amplitude_change', 3, True),
            ('shapefit.cfg.psf.bicubic.smoothing', 3, True),
            ('shapefit.magfitcfg.correction_type', None, True),
            ('shapefit.magfitcfg.correction', None, True),
            ('shapefit.magfitcfg.require', None, True),
            ('shapefit.magfitcfg.max_src', None, True),
            ('shapefit.magfitcfg.noise_offset', 3, True),
            ('shapefit.magfitcfg.max_mag_err', 3, True),
            ('shapefit.magfitcfg.rej_level', 3, True),
            ('shapefit.magfitcfg.max_rej_iter', None, True),
            ('shapefit.magfitcfg.error_avg', None, True),
            ('shapefit.magfitcfg.count_weight_power', 3, True),
            ('apphot.cfg.error_floor', 3, True),
            ('apphot.cfg.gain', 3, True),
            ('apphot.cfg.magnitude_1adu', 5, True),
            ('apphot.magfitcfg.correction_type', None, True),
            ('apphot.magfitcfg.correction', None, True),
            ('apphot.magfitcfg.require', None, True),
            ('apphot.magfitcfg.max_src', None, True),
            ('apphot.magfitcfg.noise_offset', 3, True),
            ('apphot.magfitcfg.max_mag_err', 3, True),
            ('apphot.magfitcfg.rej_level', 3, True),
            ('apphot.magfitcfg.max_rej_iter', None, True),
            ('apphot.magfitcfg.error_avg', None, True),
            ('apphot.magfitcfg.count_weight_power', 3, True)
    ]:
        dr_attribute = db_session.query(HDF5Attribute).filter_by(
            hdf5_structure_version_id=dr_structure_version_id,
            pipeline_key=pipeline_key
        ).one()
        lc_path = (
            transform_dr_to_lc_path(dr_attribute.parent)
            +
            ('/Configuration' if is_config else '')
            +
            '/'
            +
            dr_attribute.name
        )
        args = dict(
            pipeline_key=pipeline_key,
            abspath=lc_path,
            dtype=dr_attribute.dtype,
            description=dr_attribute.description
        )
        if scaleoffset is None:
            args['compression'] = 'gzip'
            args['compression_options'] = '9'
        else:
            args['scaleoffset'] = scaleoffset

        result.append(HDF5DataSet(**args))

    return result

def _get_data_reduction_dataset_datasets(db_session):
    """Return datasets built from entries for the source in DR datasets."""

    result = []

    dr_structure_version_id = _get_structure_version_id(db_session)
    for pipeline_key in ['srcproj.x',
                         'srcproj.y',
                         'srcproj.enabled',
                         'bg.value',
                         'bg.error',
                         'bg.npix',
                         'shapefit.num_pixels',
                         'shapefit.signal_to_noise',
                         'shapefit.magnitude',
                         'shapefit.magnitude_error',
                         'shapefit.quality_flag',
                         'shapefit.chi2',
                         'shapefit.magfit.magnitude',
                         'apphot.magnitude',
                         'apphot.magnitude_error',
                         'apphot.quality_flag',
                         'apphot.magfit.magnitude']:
        dr_dataset = db_session.query(HDF5DataSet).filter_by(
            hdf5_structure_version_id=dr_structure_version_id,
            pipeline_key=pipeline_key
        ).one()

        lc_path = transform_dr_to_lc_path(dr_dataset.abspath)
        if pipeline_key.endswith('magfit.magnitude'):
            lc_path += '/Magnitude'

        result.append(
            HDF5DataSet(
                pipeline_key=pipeline_key,
                abspath=lc_path,
                dtype=dr_dataset.dtype,
                compression=dr_dataset.compression,
                compression_options=dr_dataset.compression_options,
                scaleoffset=dr_dataset.scaleoffset,
                shuffle=dr_dataset.shuffle,
                replace_nonfinite=dr_dataset.replace_nonfinite,
                description=dr_dataset.description
            )
        )

    return result

def _get_configuration_index_datasets(db_session):
    """Return a list of datasets of indicies within configuration datasets."""

    result = []

    storage_options = dict(
        dtype='numpy.uint',
        compression='gzip',
        compression_options='9',
        description=('The index within the configuration datasets containing '
                     'the configuration used for this light curve point.')
    )
    key_tail = '.cfg_index'
    path_tail = '/ConfigurationIndex'

    dr_structure_version_id = _get_structure_version_id(db_session)
    for photometry_method in ['shapefit', 'apphot']:
        magfit_dataset = db_session.query(HDF5DataSet).filter_by(
            hdf5_structure_version_id=dr_structure_version_id,
            pipeline_key=photometry_method + '.magfit.magnitude'
        ).one()

        magfit_path = transform_dr_to_lc_path(
            magfit_dataset.abspath
        ).rsplit(
            '/',
            1
        )[0]

        result.extend([
            HDF5DataSet(
                pipeline_key=photometry_method + key_tail,
                abspath=('/'
                         +
                         magfit_path.strip('/').split('/', 1)[0]
                         +
                         path_tail),
                **storage_options
            ),
            HDF5DataSet(
                pipeline_key=(magfit_dataset.pipeline_key.rsplit('.', 1)[0]
                              +
                              key_tail),
                abspath=magfit_path + path_tail,
                **storage_options
            )
        ])

    result.extend([
        HDF5DataSet(
            pipeline_key='srcextract.sdkmap' + key_tail,
            abspath=_default_paths['sdk_map'] + path_tail,
            **storage_options
        ),
        HDF5DataSet(
            pipeline_key='astrometry' + key_tail,
            abspath=_default_paths['astrometry'] + path_tail,
            **storage_options
        ),
        HDF5DataSet(
            pipeline_key='fitsheader' + key_tail,
            abspath='/FrameInformation' + path_tail,
            **storage_options
        ),
        HDF5DataSet(
            pipeline_key='skytoframe' + key_tail,
            abspath=(_dr_default_paths['skytoframe']['root'].rsplit('/', 1)[0]
                     +
                     path_tail),
            **storage_options
        ),
        HDF5DataSet(
            pipeline_key='bg' + key_tail,
            abspath=(_dr_default_paths['background'].rsplit('/', 1)[0]
                     +
                     path_tail),
            **storage_options
        )
    ])

    return result

def _get_attributes(db_session):
    """Return a list of all attributes of light cuves."""

    aperture_size_attribute = db_session.query(HDF5Attribute).filter_by(
        hdf5_structure_version_id=_get_structure_version_id(db_session),
        pipeline_key='apphot.cfg.aperture'
    ).one()

    return (
        _get_catalogue_attributes()
        +
        [
            HDF5Attribute(
                pipeline_key='apphot.cfg.aperture',
                parent=transform_dr_to_lc_path(aperture_size_attribute.parent),
                name=aperture_size_attribute.name,
                dtype=aperture_size_attribute.dtype,
                description=aperture_size_attribute.description
            )
        ]
    )


def _get_datasets(db_session):
    """Return a list of all datasets in light curves."""

    return (
        _get_source_extraction_datasets()
        +
        _get_astrometry_datasets()
        +
        _get_frame_datasets()
        +
        _get_data_reduction_attribute_datasets(db_session)
        +
        _get_data_reduction_dataset_datasets(db_session)
        +
        _get_configuration_index_datasets(db_session)
    )

def get_default_light_curve_structure(db_session):
    """Create default configuration for the layout of light curve files."""

    default_structure = HDF5Product(
        pipeline_key='light_curve',
        description=('Contains all per-source processing information.')
    )
    default_structure.structure_versions = [
        HDF5StructureVersion(version=0)
    ]
    default_structure.structure_versions[0].attributes = (
        _get_attributes(db_session)
    )
    default_structure.structure_versions[0].datasets = (
        _get_datasets(db_session)
    )
    return default_structure
