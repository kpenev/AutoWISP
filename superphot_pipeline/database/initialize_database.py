#!/usr/bin/env python3

"""Create all datbase tables and define default configuration."""

from superphot_pipeline.database.interface import db_engine, db_session_scope
from superphot_pipeline.database.data_model.base import DataModelBase

#Pylint false positive due to quirky imports.
#pylint: disable=no-name-in-module
from superphot_pipeline.database.data_model import\
    HDF5Product,\
    HDF5StructureVersion,\
    HDF5Attribute
#pylint: enable=no-name-in-module

_default_paths = dict(
    shapefit='/ShapeFit/Version%(shape_fit_version)03d',
    apphot='/AperturePhotometry/Version%(apphot_version)03d',
    srcextract=dict(
        root='/SourceExtraction/Version%(srcextract_version)03d',
        sources='/Sources',
        fistar_sdk_map='/FiStarSDKMap'
    ),
    skytoframe=dict(
        root='/SkyToFrameTransformation/Version%(skytoframe_version)03d',
        coefficients='/ProjectedToFrameMap'
    ),
    srcproj='/ProjectedSources/Version$(srcproj_version)03d'
)

def get_magfit_attributes(photometry_mode, is_master):
    """
    Return a set of magnitude fitting attributes for a single photometry.

    Args:
        photometry_mode(str):    The method by which the raw magnitudes used for
            magnitude fitting were extracted.

        is_master (bool):    Should names be for single (False) or master (True)
            magnitude fitting.

    Returns:
        [HDF5Attribute]:
            The attributes describing magnitude fitting.
    """

    pipeline_key_start = ('m' if is_master else 's') + 'prmagfit.'
    if photometry_mode.lower() in ['psffit', 'prffit', 'shapefit']:
        nest_under = _default_paths['shapefit']
        pipeline_key_start = 'shapefit.' + pipeline_key_start
    elif photometry_mode.lower() == 'apphot':
        nest_under = _default_paths['apphot']
        pipeline_key_start = 'apphot.' + pipeline_key_start
    else:
        raise ValueError('Unrecognized photometry mode: '
                         +
                         repr(photometry_mode))
    nest_under += (
        ('Master' if is_master else 'Single')
        +
        'ReferenceFittedMagnitude'
    )

    result = [
        HDF5Attribute(
            pipeline_key=pipeline_key_start + 'num_input_src',
            parent=nest_under,
            name='NumberInputSources',
            dtype="'i'",
            description='The number of sources magnitude fitting was applied '
            'to.'
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + 'num_fit_src',
            parent=nest_under,
            name='NumberFitSources',
            dtype="'i'",
            description='The number of unrejected sources used in the last '
            'iteration of this magintude fit.'
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + 'fit_residual',
            parent=nest_under,
            name='FitResidual',
            dtype="'f8'",
            description='The RMS residual from the single refence magnitude '
            'fit.'
        )
    ]
    pipeline_key_start = pipeline_key_start[:-1] + 'cfg.'
    return result + [
        HDF5Attribute(
            pipeline_key=pipeline_key_start + 'correction_type',
            parent=nest_under,
            name='CorrectionType',
            dtype='numpy.string_',
            description='The type of function being fitted for now the '
            'supported types are: linear (nonlinear and spline in the future).'
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + 'correction',
            parent=nest_under,
            name='CorrectionExpression',
            dtype='numpy.string_',
            description='The actual parametric expression for the magnitude '
            'correction.'
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + 'filter',
            parent=nest_under,
            name='SourceFilter',
            dtype='numpy.string_',
            description='Any condition imposed on the sources used to derive '
            'the correction function parameters.'
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + 'max_src',
            parent=nest_under,
            name='MaxSources',
            dtype="'i'",
            description='The maximum number of sources to use in the fit.'
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + 'noise_offset',
            parent=nest_under,
            name='ExtraNoiseLevel',
            dtype="'f8'",
            description='A constant added to the magnitude error before using '
            'in the fit.'
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + 'max_mag_err',
            parent=nest_under,
            name='MaxMagnitudeError',
            dtype="'f8'",
            description='Sources with estimated magnitude error larger than '
            'this are not used in the fit.'
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + 'rej_level',
            parent=nest_under,
            name='RejectionLevel',
            dtype="'f8'",
            description='Sources rej_level time average error away from the '
            'best fit are rejected and the fit is repeated.'
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + 'max_rej_iter',
            parent=nest_under,
            name='MaxRejectionIterations',
            dtype="'i'",
            description='Stop rejecting outlier sources after this number of '
            'rejection/refitting cycles.'
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + 'error_avg',
            parent=nest_under,
            name='ErrorAveraging',
            dtype="numpy.string_",
            description='How to calculate the scale for rejecting sources.'
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + 'count_weight_power',
            parent=nest_under,
            name='NumberMeasurementsWeightingPower',
            dtype="'f8'",
            description='The number of observations for a star/max number of '
            'observations raised to this power is multiplied by the error based'
            ' weight when doing the magnitude fit.'
        )
    ]

#Silence complaint about too long a name.
#pylint: disable=invalid-name
def get_background_data_reduction_attributes():
    """
    Create default attributes in data reduction files describing BG extraction.
    """

    return [
        HDF5Attribute(
            pipeline_key='bg.cfg.zero',
            parent='/Background',
            name='BackgroudIsZero',
            dtype="'?'",
            description='Assume that the background has already been subtracted'
            ' from the input image?'
        ),
        HDF5Attribute(
            pipeline_key='bg.cfg.model',
            parent='/Background',
            name='Model',
            dtype='numpy.string_',
            description='How was the backgroun modelled.'
        ),
        HDF5Attribute(
            pipeline_key='bg.cfg.annulus',
            parent='/Background',
            name='Annulus',
            dtype="'f8'",
            description='The inner and outer radius of the annulus centered '
            'around each source used to estimate the background and its error.'
        ),
        HDF5Attribute(
            pipeline_key='bg.cfg.min_pix',
            parent='/Background',
            name='MinPixels',
            dtype="'i'",
            description='The minimum number of pixels required to estimate a '
            'reliable value and error for the background.'
        ),
        HDF5Attribute(
            pipeline_key='bg.sofware_versions',
            parent='/Background',
            name='SoftwareVersions',
            dtype="'S100'",
            description='An Nx2 array of strings consisting of '
            'software elements and their versions used for estimating the '
            'backgrund for each source.'
        )
    ]

def get_shapefit_data_reduction_attributes():
    """
    Create the default attribute in data reduction files describing PSF fitting.

    Args:
        None

    Returns:
        [HDF5Attribute]:
            All attributes related to PSF/PRF fitting to include in data
            reduction files.
    """

    def get_config_attributes():
        """Create the attributes specifying the shape fitting configuration."""

        return [
            HDF5Attribute(
                pipeline_key='shapefit.cfg.magnitude_1adu',
                parent=_default_paths['shapefit'],
                name='Magnitude1ADU',
                dtype="'f8'",
                description='The magnitude that corresponds to a flux of '
                '1ADU on the input image.'
            ),
            HDF5Attribute(
                pipeline_key='shapefit.cfg.psf.model',
                parent=_default_paths['shapefit'],
                name='Model',
                dtype='numpy.string_',
                description='The model used to represent the PSF/PRF.'
            ),
            HDF5Attribute(
                pipeline_key='shapefit.cfg.psf.terms',
                parent=_default_paths['shapefit'],
                name='Terms',
                dtype='numpy.string_',
                description='The terms the PSF/PRF is allowed to depend '
                'on. See SuperPhot documentation for full description.'
            ),
            HDF5Attribute(
                pipeline_key='shapefit.cfg.psf.max-chi2',
                parent=_default_paths['shapefit'],
                name='MaxReducedChiSquared',
                dtype="'f8'",
                description='The value of the reduced chi squared above '
                'which sources are excluded from the fit.'
            ),
            HDF5Attribute(
                pipeline_key='shapefit.cfg.psf.min_convergence_rate',
                parent=_default_paths['shapefit'],
                name='MinimumConvergenceRate',
                dtype="'f8'",
                description='The minimum rate of convergence required '
                'before stopping iterations.'
            ),
            HDF5Attribute(
                pipeline_key='shapefit.cfg.psf.max_iterations',
                parent=_default_paths['shapefit'],
                name='MaxIterations',
                dtype="'i'",
                description='The maximum number of shape/amplitude '
                'fitting iterations allowed during PSF/PRF fitting.'
            ),
            HDF5Attribute(
                pipeline_key='shapefit.cfg.psf.ignore_dropped',
                parent=_default_paths['shapefit'],
                name='DiscardDroppedSources',
                dtype="'?'",
                description='If True, sources dropped during source '
                'selection will not have their amplitudes fit for. '
                'Instead their shape fit fluxes/magnitudes and associated'
                ' errors will all be NaN.'
            ),
            HDF5Attribute(
                pipeline_key='shapefit.cfg.src.cover_bicubic_grid',
                parent=_default_paths['shapefit'],
                name='CoverGridWithPixels',
                dtype="'?'",
                description='For bicubic PSF fits, If true all pixels '
                'that at least partially overlap with the grid are '
                'assigned to the corresponding source.'
            ),
            HDF5Attribute(
                pipeline_key='shapefit.cfg.src.min_signal_to_noise',
                parent=_default_paths['shapefit'],
                name='SourcePixelMinSignalToNoise',
                dtype="'f8'",
                description='How far above the background (in units of '
                'RMS) should pixels be to still be considered part of a '
                'source.'
            ),
            HDF5Attribute(
                pipeline_key='shapefit.cfg.src.max_aperture',
                parent=_default_paths['shapefit'],
                name='SourceMaxAperture',
                dtype="'f8'",
                description='If this option has a positive value, pixels '
                'are assigned to sources in circular apertures (the '
                'smallest such that all pixels that pass the signal to '
                'noise cut are still assigned to the source).'
            ),
            HDF5Attribute(
                pipeline_key='shapefit.cfg.src.max_sat_frac',
                parent=_default_paths['shapefit'],
                name='SourceMaxSaturatedFraction',
                dtype="'f8'",
                description='If more than this fraction of the pixels '
                'assigned to a source are saturated, the source is '
                'excluded from the fit.'
            ),
            HDF5Attribute(
                pipeline_key='shapefit.cfg.src.min_pix',
                parent=_default_paths['shapefit'],
                name='SourceMinPixels',
                dtype="'i'",
                description='The minimum number of pixels that must be '
                'assigned to a source in order to include the source is '
                'the shapefit.'
            ),
            HDF5Attribute(
                pipeline_key='shapefit.cfg.src.max_pix',
                parent=_default_paths['shapefit'],
                name='SourceMaxPixels',
                dtype="'i'",
                description='The maximum number of pixels that must be '
                'assigned to a source in order to include the source is '
                'the shapefit.'
            ),
            HDF5Attribute(
                pipeline_key='shapefit.cfg.src.max_count',
                parent=_default_paths['shapefit'],
                name='MaxSources',
                dtype="'i'",
                description='The maximum number of sources to include in '
                'the fit for the PSF shape.'
            ),
            HDF5Attribute(
                pipeline_key='shapefit.cfg.psf.bicubic.grid',
                parent=_default_paths['shapefit'],
                name='Grid',
                dtype="'f8'",
                description='The x and y boundaries of the grid on which '
                'the PSF map is defined.'
            ),
            HDF5Attribute(
                pipeline_key='shapefit.cfg.psf.bicubic.pixrej',
                parent=_default_paths['shapefit'],
                name='PixelRejectionThreshold',
                dtype="'f8'",
                description='Pixels with fitting residuals (normalized by'
                ' the standard deviation) bigger than this value are '
                'excluded from the fit.'
            ),
            HDF5Attribute(
                pipeline_key='shapefit.cfg.psf.bicubic.initial_aperture',
                parent=_default_paths['shapefit'],
                name='InitialAmplitudeAperture',
                dtype="'f8'",
                description='This aperture is used to derive an initial '
                'guess for the amplitudes of sources.'
            ),
            HDF5Attribute(
                pipeline_key='shapefit.cfg.psf.bicubic.max_abs_amplitude_change',
                parent=_default_paths['shapefit'],
                name='MaxAbsoluteAmplitudeChange',
                dtype="'f8'",
                description='The absolute root of sum squares tolerance '
                'of the source amplitude changes in order to declare the '
                'piecewise bicubic PSF fitting converged.'
            ),
            HDF5Attribute(
                pipeline_key='shapefit.cfg.psf.bicubic.max_rel_amplitude_change',
                parent=_default_paths['shapefit'],
                name='MaxRelativeAmplitudeChange',
                dtype="'f8'",
                description='The relative root of sum squares tolerance of the '
                'source amplitude changes in order to declare the piecewise '
                'bicubic PSF fittingiiii converged.'
            ),
            HDF5Attribute(
                pipeline_key='shapefit.cfg.psf.bicubic.smoothing',
                parent=_default_paths['shapefit'],
                name='BicubicSmoothing',
                dtype="'f8'",
                description='The amount of smoothing used during PSF '
                'fitting.'
            )
        ]


    return (
        get_config_attributes()
        +
        [
            HDF5Attribute(
                pipeline_key='shapefit.sofware_versions',
                parent=_default_paths['shapefit'],
                name='SoftwareVersions',
                dtype="'S100'",
                description='An Nx2 array of strings consisting of '
                'software elements and their versions usef during PSF/PRF'
                ' fitting.'
            ),
            HDF5Attribute(
                pipeline_key='shapefit.global_chi2',
                parent=_default_paths['shapefit'],
                name='GlobalReducedChi2',
                dtype="'f8'",
                description='The overall reduced chi squared of the '
                'PSF/PRF fit.'
            )
        ]
        +
        get_magfit_attributes('shapefit', False)
        +
        get_magfit_attributes('shapefit', True)
    )

def get_apphot_data_reduction_attributes():
    """
    Create default data reduction attributes describing aperture photometry.
    """

    return (
        [
            HDF5Attribute(
                pipeline_key='apphot.sofware_versions',
                parent=_default_paths['apphot'],
                name='SoftwareVersions',
                dtype="'S100'",
                description='An Nx2 array of strings consisting of '
                'software elements and their versions used for aperture '
                'photometry.'
            ),
            HDF5Attribute(
                pipeline_key='apphot.cfg.error_floor',
                parent=_default_paths['apphot'],
                name='ErrorFloor',
                dtype="'f8'",
                description='A value to add to the error estimate of pixels '
                '(intended to represent things like readout noise, truncation '
                'noise etc.).'
            ),
            HDF5Attribute(
                pipeline_key='apphot.cfg.aperture',
                parent=(_default_paths['apphot']
                        +
                        '/Aperture%(aperture_index)03d'),
                name='Aperture',
                dtype="'f8'",
                description='The size of the aperture used for aperture '
                'photometry.'
            )
        ]
        +
        get_magfit_attributes('apphot', False)
        +
        get_magfit_attributes('apphot', True)
    )

def get_default_source_extraction_attributes():
    """Create default data reduction attributes describing source extraction."""

    map_parent = (_default_paths['srcextract']['root']
                  +
                  _default_paths['srcextract']['fistar_sdk_map'])
    fistar_attributes = [
        HDF5Attribute(
            pipeline_key='srcextract.fistar.cmdline',
            parent=_default_paths['srcextract']['root'],
            name='FiStarCommandLine',
            dtype='numpy.string_',
            description='The command line with which fistar was invoked.'
        ),
        HDF5Attribute(
            pipeline_key='srcextract.fistar.sdkmap.scale',
            parent=map_parent,
            name='Scale',
            dtype="'f8'",
            description='The scaling to apply to source positinos after '
            'offsetting and before substituting in the PSF map. A pair of '
            'values giving the x and y scalings.'
        ),
        HDF5Attribute(
            pipeline_key='srcextract.fistar.sdkmap.offset',
            parent=map_parent,
            name='Offset',
            dtype="'f8'",
            description='The offsets to apply to source positions before '
            'scaling and substitutiting in the PSF map. A pair of values giving'
            ' the x and y offsets.'
        )
    ]

    return (
        [
            HDF5Attribute(
                pipeline_key='srcextract.software_versions',
                parent=_default_paths['srcextract']['root'],
                name='SoftwareVersions',
                dtype="'S100'",
                description='An Nx2 array of strings consisting of '
                'software elements and their versions used for source '
                'extraction.'
            ),
            HDF5Attribute(
                pipeline_key='srcextract.binning',
                parent=_default_paths['srcextract']['root'],
                name='ImageBinFactor',
                dtype="'i'",
                description='Two values, giving the factors by which the input '
                'image was binned in the x and y directions respectively '
                'before passing to the source extractor. Useful for way out of '
                'focus images.'
            ),
            HDF5Attribute(
                pipeline_key='srcextract.columns',
                parent=(_default_paths['srcextract']['root']
                        +
                        _default_paths['srcextract']['sources']),
                name='Columns',
                dtype="'S100'",
                description='A list of the source extraction columns stored.'
            )
        ]
        +
        fistar_attributes
    )

def get_catalogue_query_attributes(pipeline_component, all_config):
    """
    Create default data reduction attributes describing catalogue queries.

    Args:
        pipeline_component:    What pipeline step is using this catalogue query.
            This is the first portion of the pipeline key assigned to the
            attributes (without the trailing dot).

        all_config:    If true, all newly created attributes are defined as
            configrations. The case where all_config is False, is only useful
            for deriving the astrometry, where some of the parameters of the
            query are derived.
    """

    parent = _default_paths[pipeline_component]
    if isinstance(parent, dict):
        parent = parent['root']

    assert isinstance(parent, str)

    config_key_start = pipeline_component + '.cfg.'
    nonconfig_key_start = (config_key_start if all_config
                           else pipeline_component + '.')

    return [
        HDF5Attribute(
            pipeline_key=config_key_start + 'catalogue',
            parent=parent,
            name='Catalogue',
            dtype='numpy.string_',
            description='The catalogue to query.'
        ),
        HDF5Attribute(
            pipeline_key=config_key_start + 'catalogue_epoch',
            parent=parent,
            name='CatalogueEpoch',
            dtype="'f8'",
            description='The epoch (JD) up to which source positions were '
            'corrected when used.'
        ),
        HDF5Attribute(
            pipeline_key=config_key_start + 'catalogue_filter',
            parent=parent,
            name='CatalogueFilter',
            dtype='numpy.string_',
            description='Any filtering applied to the catalogue sources, in '
            'addition to the field selection and brightness range, before '
            'using them.'
        ),
        HDF5Attribute(
            pipeline_key=nonconfig_key_start + 'catalogue_size',
            parent=parent,
            name='CatalogueQuerySize',
            dtype="'f8'",
            description='The width and height of the field queried from the'
            ' catalogue.'
        ),
        HDF5Attribute(
            pipeline_key=nonconfig_key_start + 'catalogue_orientation',
            parent=parent,
            name='CatalogueQueryOrientation',
            dtype="'f8'",
            description='The angle from the north direction and the '
            'positive y direction at the frame center.'
        ),
        HDF5Attribute(
            pipeline_key=nonconfig_key_start + 'brightness_expression',
            parent=parent,
            name='CatalogueBrightnessExpression',
            dtype='numpy.string_',
            description='An expression involving catalogue columns used as '
            'the brightness estimate (in magnitudes) for catalogue sources.'
        ),
        HDF5Attribute(
            pipeline_key=nonconfig_key_start + 'brightness_range',
            parent=parent,
            name='CatalogueBrightnessRange',
            dtype="'f8'",
            description='The minimum and maximum brightness magnitude for '
            'catalogue sources used for finding the pre-projected to frame '
            'transformation.'
        )
    ]

def get_default_skytoframe_attributes():
    """Create default data reduction attributes describing the astrometry."""

    parent = _default_paths['skytoframe']['root']
    config_attributes = [
        HDF5Attribute(
            pipeline_key='skytoframe.software_versions',
            parent=parent,
            name='SoftwareVersions',
            dtype="'S100'",
            description='An Nx2 array of strings consisting of '
            'software elements and their versions used for deriving the sky to '
            'frame transformation.'
        ),
        HDF5Attribute(
            pipeline_key='skytoframe.cfg.srcextract_filter',
            parent=parent,
            name='ExtractedSourcesFilter',
            dtype='numpy.string_',
            description='Any filtering applied to the extracted sources before '
            'using them to derive the pre-projected to frame transformation.'
        ),
        HDF5Attribute(
            pipeline_key='skytoframe.cfg.sky_preprojection',
            parent=parent,
            name='SkyPreProjection',
            dtype='numpy.string_',
            description='he pre-projection aronud the central coordinates used '
            'for the sources when deriving the pre-shrunk sky to frame '
            'transformation (\'arc\', \'tan\', ...).'
        ),
        HDF5Attribute(
            pipeline_key='skytoframe.cfg.frame_center',
            parent=parent,
            name='FrameCenter',
            dtype="'f8'",
            description='The frame coordinates around which the pre-projected '
            'to frame transformation is defined.'
        ),
        HDF5Attribute(
            pipeline_key='skytoframe.cfg.max_match_distance',
            parent=parent,
            name='MaxMatchDistance',
            dtype="'f8'",
            description='The maximum distance (in pixels) between extracted and'
            'projected source positions in ordet to still consider the sources '
            'matched.'
        ),
        HDF5Attribute(
            pipeline_key='skytoframe.cfg.weights_expression',
            parent=parent,
            name='WeightsExpression',
            dtype='numpy.string_',
            description='An expression involving catalogue and/or source '
            'extraction columns for the weights to use for various sources '
            'when deriving the pre-projected to frame transformation.'
        )
    ]
    return (
        config_attributes
        +
        get_catalogue_query_attributes('skytoframe', False)
        +
        [
            HDF5Attribute(
                pipeline_key='skytoframe.sky_center',
                parent=parent,
                name='CenterSkyCoordinates',
                dtype="'f8'",
                description='The (RA, Dec) coordinates corresponding to the '
                'frame center, around which the sky pre-projection is '
                'performed.'
            ),
            HDF5Attribute(
                pipeline_key='skytoframe.residual',
                parent=parent + _default_paths['skytoframe']['coefficients'],
                name='WeightedResidual',
                dtype="'f8'",
                description='The weighted residual of the best-fit '
                'pre-projected to sky transformation.'
            ),
            HDF5Attribute(
                pipeline_key='skytoframe.unitarity',
                parent=parent + _default_paths['skytoframe']['coefficients'],
                name='Unitarity',
                dtype="'f8'",
                description='The unitarity of the best-fit pre-projected to '
                'frame transformation..'
            )
        ]
    )

def get_default_source_projection_attributes():
    """Create default data reduction attributes describing source projection."""

    return (
        [
            HDF5Attribute(
                pipeline_key='srcproj.software_versions',
                parent=_default_paths['srcproj'],
                name='SoftwareVersions',
                dtype="'S100'",
                description='An Nx2 array of strings consisting of '
                'software elements and their versions used for projecting '
                'catalogue sources to the frame.'
            )
        ]
        +
        get_catalogue_query_attributes('srcproj', True)
    )

def get_default_data_reduction_attributes():
    """Create the default database attributes in data reduction HDF5 files."""

    return (
        get_default_source_extraction_attributes()
        +
        get_default_skytoframe_attributes()
        +
        get_default_source_projection_attributes()
        +
        get_background_data_reduction_attributes()
        +
        get_shapefit_data_reduction_attributes()
        +
        get_apphot_data_reduction_attributes()
    )

def get_default_data_reduction_structure():
    """Add the default configuration for the layout of data reduction files."""

    default_structure = HDF5Product(
        pipeline_key='data_reduction',
        description=('Contains all per-frame processing information and '
                     'products except calibrated image/error/mask.')
    )
    default_structure.structure_versions = [
        HDF5StructureVersion(version=0)
    ]
    default_structure.structure_versions[0].attributes = (
        get_default_data_reduction_attributes()
    )
    return default_structure
#pylint: enable=invalid-name

def add_default_hdf5_structures():
    """Add a default HDF5 structure to the database."""

    with db_session_scope() as db_session:
        db_session.add(get_default_data_reduction_structure())

def create_all_tables():
    """Create all database tables currently defined."""

    for table in DataModelBase.metadata.sorted_tables:
        if not db_engine.has_table(table.name):
            table.create(db_engine)


if __name__ == '__main__':
    create_all_tables()
    add_default_hdf5_structures()
