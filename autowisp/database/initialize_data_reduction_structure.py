# pylint: disable=too-many-lines
"""Define function to add defaults to all HDF5 structure tables."""

import numpy

# Pylint false positive due to quirky imports.
# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    HDF5Product,
    HDF5StructureVersion,
    HDF5Attribute,
    HDF5DataSet,
    HDF5Link,
)

# pylint: enable=no-name-in-module

_default_paths = {
    "srcextract": {
        "root": "/SourceExtraction/Version%(srcextract_version)03d",
        "sources": "/Sources",
        "psf_map": "/PSFMap",
    },
    "catalogue": "/CatalogueSources/Version%(catalogue_version)03d",
    "skytoframe": {
        "root": "/SkyToFrameTransformation/Version%(skytoframe_version)03d",
        "coefficients": "/ProjectedToFrameMap",
        "matched": "/MatchedSources",
    },
    "srcproj": "/ProjectedSources/Version%(srcproj_version)03d",
    "background": "/Background/Version%(background_version)03d",
    "shapefit": "/ShapeFit/Version%(shapefit_version)03d",
    "apphot": {
        "root": "/AperturePhotometry/Version%(apphot_version)03d",
        "apsplit": "/Aperture%(aperture_index)03d",
    },
    "subpixmap": "/SubPixelMap/Version%(subpixmap_version)03d",
}

_default_nonfinite = repr(numpy.finfo("f4").min / 2)


def _get_source_extraction_attributes():
    """Create default data reduction attributes describing source extraction."""

    map_parent = (
        _default_paths["srcextract"]["root"]
        + _default_paths["srcextract"]["psf_map"]
    )
    fistar_attributes = [
        HDF5Attribute(
            pipeline_key="srcextract.fistar.cmdline",
            parent=_default_paths["srcextract"]["root"],
            name="FiStarCommandLine",
            dtype="numpy.string_",
            description="The command line with which fistar was invoked.",
        ),
        HDF5Attribute(
            pipeline_key="srcextract.psf_map.cfg.psf_params",
            parent=map_parent,
            name="Parameters",
            dtype="manual",
            description="The parameters for which smoothing was done.",
        ),
        HDF5Attribute(
            pipeline_key="srcextract.psf_map.cfg.terms",
            parent=map_parent,
            name="Terms",
            dtype="numpy.string_",
            description="An expression expanding to the terms to include when "
            "smoothing the fistar PSF parameters.",
        ),
        HDF5Attribute(
            pipeline_key="srcextract.psf_map.cfg.weights",
            parent=map_parent,
            name="Weights",
            dtype="numpy.string_",
            description="An expression that evaluates to the weight for each "
            "source in the smoothing fit.",
        ),
        HDF5Attribute(
            pipeline_key="srcextract.psf_map.cfg.error_avg",
            parent=map_parent,
            name="ErrorAveraging",
            dtype="numpy.string_",
            description="How are the residuals after the fit averaged to "
            "detect outlier sources?",
        ),
        HDF5Attribute(
            pipeline_key="srcextract.psf_map.cfg.rej_level",
            parent=map_parent,
            name="RejectionLevel",
            dtype="numpy.float64",
            description="Sources are outliers if their residual "
            "from the best fit is bigger than RejectionLevel * average "
            "residuals.",
        ),
        HDF5Attribute(
            pipeline_key="srcextract.psf_map.cfg.max_rej_iter",
            parent=map_parent,
            name="MaxRejectionIterations",
            dtype="numpy.uint",
            description="Maximum of this many outlier rejection/re-fitting "
            "iterations are performed.",
        ),
        HDF5Attribute(
            pipeline_key="srcextract.psf_map.residual",
            parent=map_parent,
            name="RMSResidual",
            dtype="numpy.float64",
            description="Root of average square residuals of the fit for each "
            "PSF parameter.",
        ),
        HDF5Attribute(
            pipeline_key="srcextract.psf_map.num_fit_src",
            parent=map_parent,
            name="NumberFitSources",
            dtype="numpy.uint",
            description="The number of non-rejected sources used in the "
            "accepted fit for each PSF parameter.",
        ),
    ]

    return [
        HDF5Attribute(
            pipeline_key="srcextract.software_versions",
            parent=_default_paths["srcextract"]["root"],
            name="SoftwareVersions",
            dtype="numpy.string_",
            description="An Nx2 array of strings consisting of "
            "software elements and their versions used for source "
            "extraction.",
        ),
        HDF5Attribute(
            pipeline_key="srcextract.psf_map.software_versions",
            parent=_default_paths["srcextract"]["root"],
            name="SoftwareVersions",
            dtype="numpy.string_",
            description="An Nx2 array of strings consisting of "
            "software elements and their versions used for source "
            "extraction.",
        ),
        HDF5Attribute(
            pipeline_key="srcextract.cfg.binning",
            parent=_default_paths["srcextract"]["root"],
            name="ImageBinFactor",
            dtype="numpy.uint",
            description="Two values, giving the factors by which the input "
            "image was binned in the x and y directions respectively "
            "before passing to the source extractor. Useful for way out of "
            "focus images.",
        ),
    ] + fistar_attributes


def _get_source_extraction_datasets():
    """Create default data reduction datesets for source extraction."""

    return [
        HDF5DataSet(
            pipeline_key="srcextract.sources",
            abspath=(
                _default_paths["srcextract"]["root"]
                + _default_paths["srcextract"]["sources"]
                + "/%(srcextract_column_name)s"
            ),
            dtype="manual",
            compression="manual",
            compression_options="manual",
            description="A single quantity derived for each extracted source "
            "during source extraction.",
        ),
        HDF5DataSet(
            pipeline_key="srcextract.psf_map",
            abspath=(
                _default_paths["srcextract"]["root"]
                + _default_paths["srcextract"]["psf_map"]
            ),
            dtype="numpy.float64",
            description="The coefficients of the map giving the smoothed PSF "
            "shape parameters.",
        ),
    ]


def _get_catalogue_attributes():
    """Create default data reduction attributes describing catalogue queries."""

    parent = _default_paths["catalogue"]
    pipeline_key_start = "catalogue.cfg."

    return [
        HDF5Attribute(
            pipeline_key=pipeline_key_start + "name",
            parent=parent,
            name="Name",
            dtype="numpy.string_",
            description="The catalogue to query.",
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + "epoch",
            parent=parent,
            name="Epoch",
            dtype="numpy.float64",
            description="The epoch (JD) up to which source positions were "
            "corrected when used.",
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + "filter",
            parent=parent,
            name="Filter",
            dtype="numpy.string_",
            description="Any filtering applied to the catalogue sources, in "
            "addition to the field selection and brightness range, before "
            "using them.",
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + "fov",
            parent=parent,
            name="QuerySize",
            dtype="numpy.float64",
            description="The width and height of the field queried from the"
            " catalogue.",
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + "orientation",
            parent=parent,
            name="QueryOrientation",
            dtype="numpy.float64",
            description="The minimum and maximum brightness magnitude for "
            "catalogue sources used for finding the pre-projected to frame "
            "transformation.",
        ),
    ]


def _get_catalogue_datasets():
    """Create default data reduction datasets for catalogue_queries."""

    return [
        HDF5DataSet(
            pipeline_key="catalogue.columns",
            abspath=_default_paths["catalogue"] + "/%(catalogue_column_name)s",
            dtype="manual",
            compression="manual",
            compression_options="manual",
            description="A single catalogue column.",
        )
    ]


def _get_skytoframe_attributes():
    """Create default data reduction attributes describing the astrometry."""

    parent = _default_paths["skytoframe"]["root"]
    config_attributes = [
        HDF5Attribute(
            pipeline_key="skytoframe.software_versions",
            parent=parent,
            name="SoftwareVersions",
            dtype="numpy.string_",
            description="An Nx2 array of strings consisting of "
            "software elements and their versions used for deriving the sky to "
            "frame transformation.",
        ),
        HDF5Attribute(
            pipeline_key="skytoframe.cfg.srcextract_filter",
            parent=parent,
            name="ExtractedSourcesFilter",
            dtype="numpy.string_",
            description="Any filtering applied to the extracted sources before "
            "using them to derive the pre-projected to frame transformation.",
        ),
        HDF5Attribute(
            pipeline_key="skytoframe.cfg.sky_preprojection",
            parent=parent,
            name="SkyPreProjection",
            dtype="numpy.string_",
            description="The pre-projection aronud the central coordinates used"
            " for the sources when deriving the pre-shrunk sky to frame "
            "transformation ('arc', 'tan', ...).",
        ),
        HDF5Attribute(
            pipeline_key="skytoframe.cfg.frame_center",
            parent=parent,
            name="FrameCenter",
            dtype="numpy.float64",
            description="The frame coordinates around which the pre-projected "
            "to frame transformation is defined.",
        ),
        HDF5Attribute(
            pipeline_key="skytoframe.cfg.max_match_distance",
            parent=parent,
            name="MaxMatchDistance",
            dtype="numpy.float64",
            description="The maximum distance (in pixels) between extracted and"
            "projected source positions in ordet to still consider the sources "
            "matched.",
        ),
        HDF5Attribute(
            pipeline_key="skytoframe.cfg.weights_expression",
            parent=parent,
            name="WeightsExpression",
            dtype="numpy.string_",
            description="An expression involving catalogue and/or source "
            "extraction columns for the weights to use for various sources "
            "when deriving the pre-projected to frame transformation.",
        ),
    ]
    return config_attributes + [
        HDF5Attribute(
            pipeline_key="skytoframe.sky_center",
            parent=parent,
            name="CenterSkyCoordinates",
            dtype="numpy.float64",
            description="The (RA, Dec) coordinates corresponding to the "
            "frame center, around which the sky pre-projection is "
            "performed.",
        ),
        HDF5Attribute(
            pipeline_key="skytoframe.residual",
            parent=parent + _default_paths["skytoframe"]["coefficients"],
            name="WeightedResidual",
            dtype="numpy.float64",
            description="The weighted residual of the best-fit "
            "pre-projected to sky transformation.",
        ),
        HDF5Attribute(
            pipeline_key="skytoframe.unitarity",
            parent=parent + _default_paths["skytoframe"]["coefficients"],
            name="Unitarity",
            dtype="numpy.float64",
            description="The unitarity of the best-fit pre-projected to "
            "frame transformation..",
        ),
        HDF5Attribute(
            pipeline_key="skytoframe.type",
            parent=parent + _default_paths["skytoframe"]["coefficients"],
            name="Type",
            dtype="numpy.string_",
            description="The type of transformation describing the "
            "pre-projected to frame transformation.",
        ),
        HDF5Attribute(
            pipeline_key="skytoframe.terms",
            parent=parent + _default_paths["skytoframe"]["coefficients"],
            name="Terms",
            dtype="numpy.string_",
            description="The terms in the pre-projected to frame "
            "transformation.",
        ),
    ]


def _get_sky_to_frame_datasets():
    """Create default data reduction sky to frame transformation data sets."""

    return [
        HDF5DataSet(
            pipeline_key="skytoframe.matched",
            abspath=(
                _default_paths["skytoframe"]["root"]
                + _default_paths["skytoframe"]["matched"]
            ),
            dtype="numpy.uint",
            scaleoffset=0,
            description="The indices within the catalogue source and the "
            "extracted sources defining pairs of matched source.",
        ),
        HDF5DataSet(
            pipeline_key="skytoframe.coefficients",
            abspath=(
                _default_paths["skytoframe"]["root"]
                + _default_paths["skytoframe"]["coefficients"]
            ),
            dtype="numpy.float64",
            description="The coefficients defining the transformation from "
            "pre-projected sky coordinates to frame coordinates.",
        ),
    ]


def _get_link(used_component, user_component, description):
    """
    Return link to one version of used_component within user_component group.

    Args:
        used_component:    The pipeline name of the component being used.

        user_component:    The pipeline component using used_component.

        description:    A description of the link.

    Returns:
        autowisp.database.data_model.HDF5Link:
            A properly constructed link signifying that user_component was
            derived based on a particular version of used_component.
    """

    used_root = _default_paths[used_component]
    if isinstance(used_root, dict):
        used_root = used_root["root"]

    used_group_name, version_suffix = used_root.rsplit("/", 1)

    assert version_suffix == "Version%(" + used_component + "_version)03d"
    assert used_group_name[0] == "/"

    link_path = _default_paths[user_component]
    if isinstance(link_path, dict):
        link_path = link_path["root"]
    link_path += used_group_name

    return HDF5Link(
        pipeline_key=user_component + "." + used_component,
        abspath=link_path,
        target=used_root,
        description=description,
    )


def _get_sky_to_frame_links():
    """Create default data reduction links for sky to frame transformations."""

    return [
        _get_link(
            "catalogue",
            "skytoframe",
            description="The version of the catalogue used for deriving "
            "this sky to frame transformation.",
        ),
        _get_link(
            "srcextract",
            "skytoframe",
            description="The version of the extracted sources used for "
            "deriving this sky to frame transformation.",
        ),
    ]


def _get_source_projection_attributes():
    """Create default data reduction attributes describing source projection."""

    root_path = _default_paths["srcproj"]

    return [
        HDF5Attribute(
            pipeline_key="srcproj.software_versions",
            parent=root_path,
            name="SoftwareVersions",
            dtype="'S100'",
            description="An Nx2 array of strings consisting of "
            "software elements and their versions used for projecting "
            "catalogue sources to the frame.",
        ),
        HDF5Attribute(
            pipeline_key="srcproj.recognized_hat_id_prefixes",
            parent=(root_path),
            name="RecognizedHATIDPrefixes",
            dtype="'S100'",
            description="A list of all possible prefixes to source HAT-IDs.",
        ),
    ]


def _get_source_projection_datasets():
    """Create default projected sources data reduction data sets."""

    return [
        HDF5DataSet(
            pipeline_key="srcproj.columns",
            abspath=(_default_paths["srcproj"] + "/%(srcproj_column_name)s"),
            dtype="manual",
            compression="manual",
            compression_options="manual",
            description="A single column from the projected sources used for "
            "photometry.",
        )
    ]


def _get_source_projection_links():
    """Create default data reduction links for projected sources."""

    return [
        _get_link(
            "catalogue",
            "srcproj",
            description="The catalgue sources which were projected.",
        ),
        _get_link(
            "skytoframe",
            "srcproj",
            description="The sky to frame transformation used to project these "
            "sources.",
        ),
    ]


def _get_background_attributes():
    """
    Create default attributes in data reduction files describing BG extraction.
    """

    return [
        HDF5Attribute(
            pipeline_key="bg.cfg.zero",
            parent=_default_paths["background"],
            name="BackgroudIsZero",
            dtype="numpy.bool_",
            description="Assume that the background has already been subtracted"
            " from the input image?",
        ),
        HDF5Attribute(
            pipeline_key="bg.cfg.model",
            parent=_default_paths["background"],
            name="Model",
            dtype="numpy.string_",
            description="How was the backgroun modelled.",
        ),
        HDF5Attribute(
            pipeline_key="bg.cfg.annulus",
            parent=_default_paths["background"],
            name="Annulus",
            dtype="numpy.float64",
            description="The inner and outer radius of the annulus centered "
            "around each source used to estimate the background and its error.",
        ),
        HDF5Attribute(
            pipeline_key="bg.sofware_versions",
            parent=_default_paths["background"],
            name="SoftwareVersions",
            dtype="numpy.string_",
            description="An Nx2 array of strings consisting of "
            "software elements and their versions used for estimating the "
            "backgrund for each source.",
        ),
    ]


def _get_background_datasets():
    """Create default data reduction data sets of background measurements."""

    return [
        HDF5DataSet(
            pipeline_key="bg.value",
            abspath=_default_paths["background"] + "/Value",
            dtype="numpy.float64",
            scaleoffset=3,
            replace_nonfinite=_default_nonfinite,
            description="The best estimate of the background under each "
            "projected source.",
        ),
        HDF5DataSet(
            pipeline_key="bg.error",
            abspath=_default_paths["background"] + "/Error",
            dtype="numpy.float64",
            scaleoffset=3,
            replace_nonfinite=_default_nonfinite,
            description="An error estimate of the background under each "
            "projected source.",
        ),
        HDF5DataSet(
            pipeline_key="bg.npix",
            abspath=_default_paths["background"] + "/NumberPixels",
            dtype="numpy.uint",
            scaleoffset=0,
            description="The number of pixels the background value and error "
            "estimates are based on.",
        ),
    ]


def _get_background_links():
    """Create default data reduction links for sky to background extraction."""

    return [
        _get_link(
            "srcproj",
            "background",
            description="The soures for which background was measured.",
        )
    ]


def _get_magfit_key_and_path(photometry_mode):
    """
    Return start of pipeline key and path for magfit datasets/attributes.

    Args:
        photometry_mode(str):    The method by which the raw magnitudes used for
            magnitude fitting were extracted.

    Returns:
        str, str:
            The start of the pipeline_key and the absolute path of the dataset
            in the HDF5 file under whihc to place the fitted magnitudes and
            associated attributes.
    """

    pipeline_key_start = "magfit."
    if photometry_mode.lower() in ["psffit", "prffit", "shapefit"]:
        dset_path = _default_paths["shapefit"]
        pipeline_key_start = "shapefit." + pipeline_key_start
    elif photometry_mode.lower() == "apphot":
        dset_path = (
            _default_paths["apphot"]["root"]
            + _default_paths["apphot"]["apsplit"]
        )
        pipeline_key_start = "apphot." + pipeline_key_start
    else:
        raise ValueError(
            "Unrecognized photometry mode: " + repr(photometry_mode)
        )
    dset_path += "/" + "FittedMagnitudes/Version%(magfit_version)03d"

    return pipeline_key_start, dset_path, "/Iteration%(magfit_iteration)03d"


def _get_magfit_attributes(photometry_mode):
    """
    Return a set of magnitude fitting attributes for a single photometry.

    Args:
        photometry_mode(str):    The method by which the raw magnitudes used for
            magnitude fitting were extracted.

    Returns:
        [autowisp.database.data_model.HDF5Attribute]:
            The attributes describing magnitude fitting.
    """

    pipeline_key_start, dset_path, iter_split = _get_magfit_key_and_path(
        photometry_mode
    )
    result = [
        HDF5Attribute(
            pipeline_key=pipeline_key_start + "num_input_src",
            parent=dset_path + iter_split,
            name="NumberInputSources",
            dtype="numpy.uint",
            description="The number of sources magnitude fitting was applied "
            "to.",
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + "num_fit_src",
            parent=dset_path + iter_split,
            name="NumberFitSources",
            dtype="numpy.uint",
            description="The number of unrejected sources used in the last "
            "iteration of this magintude fit.",
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + "fit_residual",
            parent=dset_path + iter_split,
            name="FitResidual",
            dtype="numpy.float64",
            description="The RMS residual from the single refence magnitude "
            "fit.",
        ),
    ]
    pipeline_key_start = pipeline_key_start[:-1] + ".cfg."
    return result + [
        HDF5Attribute(
            pipeline_key=pipeline_key_start + "single_photref",
            parent=dset_path,
            name="SinglePhotometricReference",
            dtype="numpy.string_",
            description="The name of the DR file used as single photometric "
            "reference to initiate magnitude fitting iterations.",
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + "correction_type",
            parent=dset_path,
            name="CorrectionType",
            dtype="numpy.string_",
            description="The type of function being fitted for now the "
            "supported types are: linear (nonlinear and spline in the future).",
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + "correction",
            parent=dset_path,
            name="CorrectionExpression",
            dtype="numpy.string_",
            description="The actual parametric expression for the magnitude "
            "correction.",
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + "require",
            parent=dset_path,
            name="SourceFilter",
            dtype="numpy.string_",
            description="Any condition imposed on the sources used to derive "
            "the correction function parameters.",
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + "max_src",
            parent=dset_path,
            name="MaxSources",
            dtype="numpy.uint",
            description="The maximum number of sources to use in the fit.",
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + "noise_offset",
            parent=dset_path,
            name="ExtraNoiseLevel",
            dtype="numpy.float64",
            description="A constant added to the magnitude error before using "
            "in the fit.",
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + "max_mag_err",
            parent=dset_path,
            name="MaxMagnitudeError",
            dtype="numpy.float64",
            description="Sources with estimated magnitude error larger than "
            "this are not used in the fit.",
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + "rej_level",
            parent=dset_path,
            name="RejectionLevel",
            dtype="numpy.float64",
            description="Sources rej_level time average error away from the "
            "best fit are rejected and the fit is repeated.",
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + "max_rej_iter",
            parent=dset_path,
            name="MaxRejectionIterations",
            dtype="numpy.uint",
            description="Stop rejecting outlier sources after this number of "
            "rejection/refitting cycles.",
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + "error_avg",
            parent=dset_path,
            name="ErrorAveraging",
            dtype="numpy.string_",
            description="How to calculate the scale for rejecting sources.",
        ),
        HDF5Attribute(
            pipeline_key=pipeline_key_start + "count_weight_power",
            parent=dset_path,
            name="NumberMeasurementsWeightingPower",
            dtype="numpy.float64",
            description="The number of observations for a star/max number of "
            "observations raised to this power is multiplied by the error based"
            " weight when doing the magnitude fit.",
        ),
    ]


def _get_magfit_datasets(photometry_mode):
    """
    Return a set of magnitude fitting data sets for a single photometry.

    Args:
        photometry_mode(str):    The method by which the raw magnitudes used for
            magnitude fitting were extracted.

    Returns:
        [autowisp.database.data_model.HDF5DataSet]:
            The datasets containing the magnitude fitting results.
    """

    pipeline_key_start, dset_path, iter_split = _get_magfit_key_and_path(
        photometry_mode
    )

    return [
        HDF5DataSet(
            pipeline_key=pipeline_key_start + "magnitude",
            abspath=dset_path + iter_split,
            dtype="numpy.float64",
            scaleoffset=5,
            replace_nonfinite=_default_nonfinite,
            description=(
                f"The fitted {photometry_mode} photometry magnitudes."
            ),
        )
    ]


def _get_shapefit_attributes():
    """
    Create the default attribute in data reduction files describing PSF fitting.

    Args:
        None

    Returns:
        [autowisp.database.data_model.HDF5Attribute]:
            All attributes related to PSF/PRF fitting to include in data
            reduction files.
    """

    parent_path = _default_paths["shapefit"]

    def get_config_attributes():
        """Create the attributes specifying the shape fitting configuration."""

        return [
            HDF5Attribute(
                pipeline_key="shapefit.cfg.gain",
                parent=parent_path,
                name="Gain",
                dtype="numpy.float64",
                description="The gain (electrons per ADU) assumed for the "
                "input image.",
            ),
            HDF5Attribute(
                pipeline_key="shapefit.cfg.magnitude_1adu",
                parent=parent_path,
                name="Magnitude1ADU",
                dtype="numpy.float64",
                description="The magnitude that corresponds to a flux of "
                "1ADU on the input image.",
            ),
            HDF5Attribute(
                pipeline_key="shapefit.cfg.psf.model",
                parent=parent_path,
                name="Model",
                dtype="numpy.string_",
                description="The model used to represent the PSF/PRF.",
            ),
            HDF5Attribute(
                pipeline_key="shapefit.cfg.psf.terms",
                parent=parent_path,
                name="Terms",
                dtype="numpy.string_",
                description="The terms the PSF/PRF is allowed to depend "
                "on. See AstroWISP documentation for full description.",
            ),
            HDF5Attribute(
                pipeline_key="shapefit.cfg.psf.max-chi2",
                parent=parent_path,
                name="MaxReducedChiSquared",
                dtype="numpy.float64",
                description="The value of the reduced chi squared above "
                "which sources are excluded from the fit.",
            ),
            HDF5Attribute(
                pipeline_key="shapefit.cfg.psf.min_convergence_rate",
                parent=parent_path,
                name="MinimumConvergenceRate",
                dtype="numpy.float64",
                description="The minimum rate of convergence required "
                "before stopping iterations.",
            ),
            HDF5Attribute(
                pipeline_key="shapefit.cfg.psf.max_iterations",
                parent=parent_path,
                name="MaxIterations",
                dtype="numpy.int32",
                description="The maximum number of shape/amplitude "
                "fitting iterations allowed during PSF/PRF fitting.",
            ),
            HDF5Attribute(
                pipeline_key="shapefit.cfg.psf.ignore_dropped",
                parent=parent_path,
                name="DiscardDroppedSources",
                dtype="numpy.bool_",
                description="If True, sources dropped during source "
                "selection will not have their amplitudes fit for. "
                "Instead their shape fit fluxes/magnitudes and associated"
                " errors will all be NaN.",
            ),
            HDF5Attribute(
                pipeline_key="shapefit.cfg.src.cover_bicubic_grid",
                parent=parent_path,
                name="CoverGridWithPixels",
                dtype="numpy.bool_",
                description="For bicubic PSF fits, If true all pixels "
                "that at least partially overlap with the grid are "
                "assigned to the corresponding source.",
            ),
            HDF5Attribute(
                pipeline_key="shapefit.cfg.src.min_signal_to_noise",
                parent=parent_path,
                name="SourcePixelMinSignalToNoise",
                dtype="numpy.float64",
                description="How far above the background (in units of "
                "RMS) should pixels be to still be considered part of a "
                "source.",
            ),
            HDF5Attribute(
                pipeline_key="shapefit.cfg.src.max_aperture",
                parent=parent_path,
                name="SourceMaxAperture",
                dtype="numpy.float64",
                description="If this option has a positive value, pixels "
                "are assigned to sources in circular apertures (the "
                "smallest such that all pixels that pass the signal to "
                "noise cut are still assigned to the source).",
            ),
            HDF5Attribute(
                pipeline_key="shapefit.cfg.src.max_sat_frac",
                parent=parent_path,
                name="SourceMaxSaturatedFraction",
                dtype="numpy.float64",
                description="If more than this fraction of the pixels "
                "assigned to a source are saturated, the source is "
                "excluded from the fit.",
            ),
            HDF5Attribute(
                pipeline_key="shapefit.cfg.src.min_pix",
                parent=parent_path,
                name="SourceMinPixels",
                dtype="numpy.uint",
                description="The minimum number of pixels that must be "
                "assigned to a source in order to include the source is "
                "the shapefit.",
            ),
            HDF5Attribute(
                pipeline_key="shapefit.cfg.src.max_pix",
                parent=parent_path,
                name="SourceMaxPixels",
                dtype="numpy.uint",
                description="The maximum number of pixels that must be "
                "assigned to a source in order to include the source is "
                "the shapefit.",
            ),
            HDF5Attribute(
                pipeline_key="shapefit.cfg.src.min_bg_pix",
                parent=parent_path,
                name="SourceMinBackgroundPixels",
                dtype="numpy.uint",
                description="The minimum number of backrgound pixels required "
                "to consider the estimates for the background value and error "
                "reliable for the source.",
            ),
            HDF5Attribute(
                pipeline_key="shapefit.cfg.src.max_count",
                parent=parent_path,
                name="MaxSources",
                dtype="numpy.uint",
                description="The maximum number of sources to include in "
                "the fit for the PSF shape.",
            ),
            HDF5Attribute(
                pipeline_key="shapefit.cfg.psf.bicubic.grid.x",
                parent=parent_path,
                name="GridXBoundaries",
                dtype="numpy.float64",
                description="The x boundaries of the grid on which "
                "the PSF map is defined.",
            ),
            HDF5Attribute(
                pipeline_key="shapefit.cfg.psf.bicubic.grid.y",
                parent=parent_path,
                name="GridYBoundaries",
                dtype="numpy.float64",
                description="The y boundaries of the grid on which "
                "the PSF map is defined.",
            ),
            HDF5Attribute(
                pipeline_key="shapefit.cfg.psf.bicubic.pixrej",
                parent=parent_path,
                name="PixelRejectionThreshold",
                dtype="numpy.float64",
                description="Pixels with fitting residuals (normalized by"
                " the standard deviation) bigger than this value are "
                "excluded from the fit.",
            ),
            HDF5Attribute(
                pipeline_key="shapefit.cfg.psf.bicubic.initial_aperture",
                parent=parent_path,
                name="InitialAmplitudeAperture",
                dtype="numpy.float64",
                description="This aperture is used to derive an initial "
                "guess for the amplitudes of sources.",
            ),
            HDF5Attribute(
                pipeline_key=(
                    "shapefit.cfg.psf.bicubic." "max_abs_amplitude_change"
                ),
                parent=parent_path,
                name="MaxAbsoluteAmplitudeChange",
                dtype="numpy.float64",
                description="The absolute root of sum squares tolerance "
                "of the source amplitude changes in order to declare the "
                "piecewise bicubic PSF fitting converged.",
            ),
            HDF5Attribute(
                pipeline_key=(
                    "shapefit.cfg.psf.bicubic." "max_rel_amplitude_change"
                ),
                parent=parent_path,
                name="MaxRelativeAmplitudeChange",
                dtype="numpy.float64",
                description="The relative root of sum squares tolerance of the "
                "source amplitude changes in order to declare the piecewise "
                "bicubic PSF fittingiiii converged.",
            ),
            HDF5Attribute(
                pipeline_key="shapefit.cfg.psf.bicubic.smoothing",
                parent=parent_path,
                name="BicubicSmoothing",
                dtype="numpy.float64",
                description="The amount of smoothing used during PSF "
                "fitting.",
            ),
        ]

    return (
        get_config_attributes()
        + [
            HDF5Attribute(
                pipeline_key="shapefit.sofware_versions",
                parent=parent_path,
                name="SoftwareVersions",
                dtype="numpy.string_",
                description="An Nx2 array of strings consisting of "
                "software elements and their versions usef during PSF/PRF"
                " fitting.",
            ),
            HDF5Attribute(
                pipeline_key="shapefit.global_chi2",
                parent=parent_path,
                name="GlobalReducedChi2",
                dtype="numpy.float64",
                description="The overall reduced chi squared of the "
                "PSF/PRF fit.",
            ),
        ]
        + _get_magfit_attributes("shapefit")
    )


def _get_shapefit_datasets():
    """Create the datasets to contain shape fitting results."""

    root_path = _default_paths["shapefit"]
    return [
        HDF5DataSet(
            pipeline_key="shapefit.map_coef",
            abspath=root_path + "/MapCoefficients",
            dtype="numpy.float64",
            description="The coefficients of the derived PSF/PRF map.",
        ),
        HDF5DataSet(
            pipeline_key="shapefit.num_pixels",
            abspath=root_path + "/NumberPixels",
            dtype="numpy.uint",
            scaleoffset=0,
            replace_nonfinite=0,
            description="The number of pixels for each source on which PSF "
            "fitting was performed.",
        ),
        HDF5DataSet(
            pipeline_key="shapefit.signal_to_noise",
            abspath=root_path + "/SignalToNoise",
            dtype="numpy.float64",
            scaleoffset=3,
            replace_nonfinite=_default_nonfinite,
            description="The total signal to noise of all the pixels "
            "assigned to the source for PSF fitting.",
        ),
        HDF5DataSet(
            pipeline_key="shapefit.magnitude",
            abspath=root_path + "/Magnitude",
            dtype="numpy.float64",
            scaleoffset=5,
            replace_nonfinite=_default_nonfinite,
            description="The PSF/PRF fitting raw magnitudes of the "
            "projected sources.",
        ),
        HDF5DataSet(
            pipeline_key="shapefit.magnitude_error",
            abspath=root_path + "/MagnitudeError",
            dtype="numpy.float64",
            scaleoffset=5,
            replace_nonfinite=_default_nonfinite,
            description="Error estimates for the PSF/PRF fitting "
            "magnitudes.",
        ),
        HDF5DataSet(
            pipeline_key="shapefit.quality_flag",
            abspath=root_path + "/QualityFlag",
            dtype="numpy.uint",
            compression="gzip",
            compression_options="9",
            scaleoffset=0,
            shuffle=True,
            replace_nonfinite=255,
            description="Quality flags for the PSF fitting of the projected"
            " sources.",
        ),
        HDF5DataSet(
            pipeline_key="shapefit.chi2",
            abspath=root_path + "/ChiSquared",
            dtype="numpy.float64",
            scaleoffset=2,
            replace_nonfinite=_default_nonfinite,
            description="The reduced chi-squared values for PSF fitting "
            "for the corresponding source.",
        ),
    ] + _get_magfit_datasets("shapefit")


def _get_shapefit_links():
    """Create default data reduction links for sky to background extraction."""

    return [
        _get_link(
            "subpixmap",
            "shapefit",
            description="The sub-pixel sensitivity map assumed for this PSF "
            "fit.",
        ),
        _get_link(
            "background",
            "shapefit",
            description="The background measurement used for this PSF/PRF fit. "
            "Also contains the projected sources.",
        ),
    ]


def _get_apphot_attributes():
    """
    Create default data reduction attributes describing aperture photometry.
    """

    root_path = _default_paths["apphot"]["root"]
    apsplit_path = (
        _default_paths["apphot"]["root"] + _default_paths["apphot"]["apsplit"]
    )
    return [
        HDF5Attribute(
            pipeline_key="apphot.sofware_versions",
            parent=root_path,
            name="SoftwareVersions",
            dtype="numpy.string_",
            description="An Nx2 array of strings consisting of "
            "software elements and their versions used for aperture "
            "photometry.",
        ),
        HDF5Attribute(
            pipeline_key="apphot.cfg.error_floor",
            parent=root_path,
            name="ErrorFloor",
            dtype="numpy.float64",
            description="A value to add to the error estimate of pixels "
            "(intended to represent things like readout noise, truncation "
            "noise etc.).",
        ),
        HDF5Attribute(
            pipeline_key="apphot.cfg.aperture",
            parent=apsplit_path,
            name="Aperture",
            dtype="numpy.float64",
            description="The size of the aperture used for aperture "
            "photometry.",
        ),
        HDF5Attribute(
            pipeline_key="apphot.cfg.gain",
            parent=root_path,
            name="Gain",
            dtype="numpy.float64",
            description="The gain (electrons per ADU) assumed for the "
            "input image.",
        ),
        HDF5Attribute(
            pipeline_key="apphot.cfg.magnitude_1adu",
            parent=root_path,
            name="Magnitude1ADU",
            dtype="numpy.float64",
            description="The magnitude that corresponds to a flux of "
            "1ADU on the input image.",
        ),
    ] + _get_magfit_attributes("apphot")


def _get_apphot_datasets():
    """Create the datasets to contain shape aperture photometry results."""

    abspath_start = (
        _default_paths["apphot"]["root"] + _default_paths["apphot"]["apsplit"]
    )
    return [
        HDF5DataSet(
            pipeline_key="apphot.magnitude",
            abspath=abspath_start + "/Magnitude",
            dtype="numpy.float64",
            scaleoffset=5,
            replace_nonfinite=_default_nonfinite,
            description="The aperture photometry raw magnitudes of the "
            "projected sources.",
        ),
        HDF5DataSet(
            pipeline_key="apphot.magnitude_error",
            abspath=abspath_start + "/MagnitudeError",
            dtype="numpy.float64",
            scaleoffset=5,
            replace_nonfinite=_default_nonfinite,
            description="Error estimates for the aperture photometry "
            "magnitudes.",
        ),
        HDF5DataSet(
            pipeline_key="apphot.quality_flag",
            abspath=abspath_start + "/QualityFlag",
            dtype="numpy.uint",
            compression="gzip",
            compression_options="9",
            scaleoffset=0,
            shuffle=True,
            replace_nonfinite=255,
            description="Quality flags for the aperture photometry of the "
            "projected sources.",
        ),
    ] + _get_magfit_datasets("apphot")


def _get_apphot_links():
    """Create default data reduction links for sky to background extraction."""

    return [
        _get_link(
            "subpixmap",
            "apphot",
            description="The sub-pixel sensitivity map assumed for this "
            "aperture photometry.",
        ),
        _get_link(
            "shapefit",
            "apphot",
            description="The PSF/PRF fit use for this aperture photometry. "
            "Also contains the background measurements and projected sources.",
        ),
    ]


def _get_attributes():
    """Create the default database attributes in data reduction HDF5 files."""

    return (
        [
            HDF5Attribute(
                pipeline_key="repack",
                parent="/",
                name="Repack",
                dtype="numpy.string_",
                description="A list of the datasets deleted from the file since"
                " the last re-packing. If not empty, indicates that the file "
                "size can be decreased by re-packing.",
            )
        ]
        + _get_source_extraction_attributes()
        + _get_catalogue_attributes()
        + _get_skytoframe_attributes()
        + _get_source_projection_attributes()
        + _get_background_attributes()
        + _get_shapefit_attributes()
        + _get_apphot_attributes()
    )


def _get_datasets():
    """Create the default database datasets in data reduction HDF5 files."""

    return (
        _get_source_extraction_datasets()
        + _get_catalogue_datasets()
        + _get_sky_to_frame_datasets()
        + _get_source_projection_datasets()
        + _get_background_datasets()
        + _get_shapefit_datasets()
        + _get_apphot_datasets()
        + [
            HDF5DataSet(
                pipeline_key="fitsheader",
                abspath="/FITSHeader",
                dtype="'i1'",
                description="A binary dump of the header of the calibrated "
                "frames corresponding to this DR file.",
            ),
            HDF5DataSet(
                pipeline_key="subpixmap",
                abspath=_default_paths["subpixmap"],
                dtype="numpy.float64",
                description="The sub-pixel sensitivity map.",
            ),
        ]
    )


def _get_links():
    """Create the default database links in data reduction HDF5 files."""

    return (
        _get_sky_to_frame_links()
        + _get_source_projection_links()
        + _get_background_links()
        + _get_shapefit_links()
        + _get_apphot_links()
    )


# Silence complaint about too long a name.
# pylint: disable=invalid-name
def get_default_data_reduction_structure():
    """Add the default configuration for the layout of data reduction files."""

    default_structure = HDF5Product(
        pipeline_key="data_reduction",
        description=(
            "Contains all per-frame processing information and "
            "products except calibrated image/error/mask."
        ),
    )
    default_structure.structure_versions = [HDF5StructureVersion(version=0)]
    default_structure.structure_versions[0].attributes = _get_attributes()
    default_structure.structure_versions[0].datasets = _get_datasets()
    default_structure.structure_versions[0].links = _get_links()
    return default_structure


# pylint: enable=invalid-name
