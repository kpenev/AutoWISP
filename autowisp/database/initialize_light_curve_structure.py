# pylint: disable=too-many-lines
"""Define function to add defaults to all light curve structure tables."""

# TODO: Figure out proper structure with multiple versions of all components.

import re

import numpy

# Pylint false positive due to quirky imports.
# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    HDF5Product,
    HDF5StructureVersion,
    HDF5Attribute,
    HDF5DataSet,
)

# pylint: enable=no-name-in-module

from autowisp.database.initialize_data_reduction_structure import (
    _default_paths as _dr_default_paths,
)

_version_rex = re.compile(r"/Version%\([a-zA-Z_]*\)[0-9]*d")

_default_paths = {
    "srcextract_psf_map": "/SourceExtraction/PSFMap",
    "catalogue": "/SkyToFrameTransformation",
    "magfit": "/MagnitudeFitting",
    "sky_position": "/SkyPosition",
}

_default_nonfinite = repr(numpy.finfo("f4").min / 2)


def _get_structure_version_id(db_session, product="data_reduction"):
    return (
        db_session.query(HDF5StructureVersion.id)
        .filter(HDF5StructureVersion.hdf5_product_id == HDF5Product.id)
        .filter(HDF5Product.pipeline_key == product)
        .one()[0]
    )


def _get_source_extraction_datasets():
    """Create the default datasets for source extraction data."""

    psf_map_key_start = "srcextract.psf_map."

    def get_configuration_datasets():
        """Return the datasets containing the config. for source exatraction."""

        config_path_start = (
            _default_paths["srcextract_psf_map"] + "/Configuration/"
        )

        return [
            HDF5DataSet(
                pipeline_key=psf_map_key_start + "software_versions",
                abspath=config_path_start + "SoftwareVersions",
                dtype="numpy.string_",
                description="An Nx2 array of strings consisting of software "
                "elements and their versions used for source extraction.",
            )
        ]

    def get_per_source_datasets():
        """Return datasets containing source extraction data for each source."""

        return [
            HDF5DataSet(
                pipeline_key=psf_map_key_start + "eval",
                abspath=(
                    _default_paths["srcextract_psf_map"]
                    + "/%(srcextract_psf_param)s/Value"
                ),
                dtype="numpy.float64",
                scaleoffset=4,
                replace_nonfinite=_default_nonfinite,
                description="The values of the psf parameters for source "
                "extraction based PSF for each source.",
            )
        ]

    return get_configuration_datasets() + get_per_source_datasets()


def _get_catalogue_attributes():
    """Create the attributes for catalogue information for this source."""

    key_start = "catalogue."
    parent = "/"

    return [
        HDF5Attribute(
            pipeline_key=key_start + "name",
            parent=parent,
            name="Catalogue",
            dtype="numpy.string_",
            description="The catalogue from which this source information was "
            "queried.",
        ),
        HDF5Attribute(
            pipeline_key=key_start + "epoch",
            parent=parent,
            name="CatalogueEpoch",
            dtype="numpy.string_",
            description="The epoch (JD) up to which catalogue positions were "
            "corrected.",
        ),
        HDF5Attribute(
            pipeline_key=key_start + "information",
            parent=parent,
            name="Catalogue_%(catalogue_column_name)s",
            dtype="manual",
            description="A single catalogue value for this source.",
        ),
    ]


def _get_frame_datasets():
    """Return all datasets containing FITS header keywords."""

    def get_per_frame_datasets():
        """Return the datasets of header keywords with one entry per LC pt."""

        result = []
        for keyword, dset_name, dtype, default, description in [
            (
                "FNUM",
                "FrameNumber",
                "numpy.uint",
                repr(numpy.iinfo("u4").max),
                "The number of the frame corresponding to this datapoint in"
                " the light curve.",
            ),
            (
                "RAWFNAME",
                "RawFileName",
                "numpy.string_",
                None,
                "The filename of the RAW image that contributed this "
                "datapoint in the light curve.",
            ),
            (
                "FOCUS",
                "FocusSetting",
                "numpy.float64",
                _default_nonfinite,
                "The focus setting of the telescope for this observation.",
            ),
            (
                "WIND",
                "WindSpeed",
                "numpy.float64",
                _default_nonfinite,
                "The wind speed in m/s",
            ),
            (
                "WINDDIR",
                "WindDirection",
                "numpy.float64",
                _default_nonfinite,
                "Wind direction [degrees] reported for this observation.",
            ),
            (
                "AIRPRESS",
                "AirPressure",
                "numpy.float64",
                _default_nonfinite,
                "Air pressure [Pa] for this observation.",
            ),
            (
                "AIRTEMP",
                "AirTemperature",
                "numpy.float64",
                _default_nonfinite,
                "Air temperature [C]",
            ),
            (
                "HUMIDITY",
                "Humidity",
                "numpy.float64",
                _default_nonfinite,
                "Relative humidity [%]",
            ),
            (
                "DEWPT",
                "DewPoint",
                "numpy.float64",
                _default_nonfinite,
                "Dew point [C]",
            ),
            (
                "SUNDIST",
                "SunDistance",
                "numpy.float64",
                _default_nonfinite,
                "Distance from Sun [deg] (frame center)",
            ),
            (
                "SUNELEV",
                "SunElevation",
                "numpy.float64",
                _default_nonfinite,
                "Elevation of Sun [deg]",
            ),
            (
                "MOONDIST",
                "MoonDistance",
                "numpy.float64",
                _default_nonfinite,
                "Distance from Moon [deg] (frame center)",
            ),
            (
                "MOONPH",
                "MoonPhase",
                "numpy.float64",
                _default_nonfinite,
                "Phase of Moon",
            ),
            (
                "MOONELEV",
                "MoonElevation",
                "numpy.float64",
                _default_nonfinite,
                "Elevation of Moon [deg]",
            ),
        ]:
            args = {
                "pipeline_key": "fitsheader." + keyword.lower(),
                "abspath": "/FrameInformation/" + dset_name,
                "dtype": dtype,
                "replace_nonfinite": default,
                "description": description,
            }
            if dtype == "numpy.float64":
                args["scaleoffset"] = 3
            else:
                args["compression"] = "gzip"
                args["compression_options"] = "9"

            result.append(HDF5DataSet(**args))

        return result

    def get_config_datasets():
        """Return the datasets of header keywords treated as configuration."""

        result = []
        for keyword, dset_name, dtype, scaleoffset, description in [
            (
                "STID",
                "StationID",
                "numpy.uint",
                None,
                "ID of station that took this observation.",
            ),
            (
                "CLRCHNL",
                "ColorChannel",
                "numpy.string_",
                None,
                "The color of the channel contributing this datapoint.",
            ),
            (
                "SITEID",
                "SiteID",
                "numpy.uint",
                None,
                "ID of the site where this observation took place.",
            ),
            (
                "SITELAT",
                "SiteLatitude",
                "numpy.float64",
                6,
                "Observing site latitude [deg].",
            ),
            (
                "SITELONG",
                "SiteLongitude",
                "numpy.float64",
                6,
                "Observing site longitude [deg].",
            ),
            (
                "SITEALT",
                "SiteALtitude",
                "numpy.float64",
                3,
                "Observing site altitude above sea level [m].",
            ),
            (
                "MTID",
                "MointID",
                "numpy.uint",
                None,
                "ID of the mount used for this observing session.",
            ),
            (
                "MTVER",
                "MointVersion",
                "numpy.uint",
                None,
                "Version of the mount used for this observing session.",
            ),
            (
                "CMID",
                "CameraID",
                "numpy.uint",
                None,
                "ID of the camera used for this observing session.",
            ),
            (
                "CMVER",
                "CameraVersion",
                "numpy.uint",
                None,
                "Version of the camera used for this observing session,",
            ),
            (
                "TELID",
                "TelescopeID",
                "numpy.uint",
                None,
                "ID of the telescopes used for this observing session.",
            ),
            (
                "TELVER",
                "TelescopeVersion",
                "numpy.uint",
                None,
                "Version of the telescopes used for this observing session.",
            ),
            (
                "MNTSTATE",
                "PSFBroadeningPattern",
                "numpy.string_",
                None,
                "The PSF broadening pattern followed during exposure.",
            ),
            (
                "PROJID",
                "ProjectID",
                "numpy.uint",
                None,
                "ID of the project this observing session is part of.",
            ),
            (
                "NRACA",
                "TargetedRA",
                "numpy.float64",
                3,
                "Nominal RA of midexpo [hr] (averaged field center)",
            ),
            (
                "NDECCA",
                "TargetedDec",
                "numpy.float64",
                3,
                "Nominal Dec of midexpo [hr] (averaged field center)",
            ),
        ]:
            args = {
                "pipeline_key": "fitsheader.cfg." + keyword.lower(),
                "abspath": "/FrameInformation/Configuration/" + dset_name,
                "dtype": dtype,
                "description": description,
            }
            if scaleoffset is None:
                args["compression"] = "gzip"
                args["compression_options"] = "9"
            else:
                args["scaleoffset"] = scaleoffset

            if dtype == "numpy.float64":
                args["replace_nonfinite"] = _default_nonfinite

            result.append(HDF5DataSet(**args))

        return result

    return get_per_frame_datasets() + get_config_datasets()


def transform_dr_to_lc_path(pipeline_key, dr_path):
    """Return the path in a light curve file corresponding to a DR file path."""

    if pipeline_key == "srcextract.cfg.binning":
        return "/SourceExtraction/PSFMap"

    result = re.sub(_version_rex, "", dr_path)

    for dr_string, lc_string in [
        ("/FittedMagnitudes", _default_paths["magfit"]),
        ("/ProjectedSources", "/ProjectedPosition"),
        ("/SourceExtraction/.*/PSFMap", _default_paths["srcextract_psf_map"]),
        ("/ProjectedToFrameMap", ""),
        ("/CatalogueSources", "/SkyToFrameTransformation"),
        (
            r"/MagnitudeFitting/Iteration%\(magfit_iteration\)03d",
            "/MagnitudeFitting",
        ),
    ]:
        result = re.sub(dr_string, lc_string, result)

    if pipeline_key in [
        "srcextract.psf_map.residual",
        "srcextract.psf_map.num_fit_src",
    ]:
        result += "/%(srcextract_psf_param)s"

    return result


def _get_data_reduction_attribute_datasets(db_session):
    """Return all datasets from attributes in data reduction files."""

    dr_structure_version_id = _get_structure_version_id(db_session)
    result = []
    for pipeline_key, scaleoffset, is_config in [
        ("skytoframe.sky_center", 5, False),
        ("skytoframe.residual", 2, False),
        ("skytoframe.unitarity", 5, False),
        ("shapefit.global_chi2", 2, False),
        ("shapefit.magfit.num_input_src", None, False),
        ("shapefit.magfit.num_fit_src", None, False),
        ("shapefit.magfit.fit_residual", 2, False),
        ("apphot.magfit.num_input_src", None, False),
        ("apphot.magfit.num_fit_src", None, False),
        ("apphot.magfit.fit_residual", 2, False),
        ("srcextract.cfg.binning", None, True),
        ("srcextract.psf_map.cfg.psf_params", None, True),
        ("srcextract.psf_map.cfg.terms", None, True),
        ("srcextract.psf_map.cfg.weights", None, True),
        ("srcextract.psf_map.cfg.error_avg", None, True),
        ("srcextract.psf_map.cfg.rej_level", None, True),
        ("srcextract.psf_map.cfg.max_rej_iter", None, True),
        ("srcextract.psf_map.residual", 3, False),
        ("srcextract.psf_map.num_fit_src", None, False),
        ("skytoframe.cfg.srcextract_filter", None, True),
        ("skytoframe.cfg.sky_preprojection", None, True),
        ("skytoframe.cfg.frame_center", 3, True),
        ("skytoframe.cfg.max_match_distance", 3, True),
        ("skytoframe.cfg.weights_expression", None, True),
        ("bg.cfg.zero", None, True),
        ("bg.cfg.model", None, True),
        ("bg.cfg.annulus", None, True),
        ("shapefit.cfg.gain", 3, True),
        ("shapefit.cfg.magnitude_1adu", 5, True),
        ("shapefit.cfg.psf.model", None, True),
        ("shapefit.cfg.psf.terms", None, True),
        ("shapefit.cfg.psf.max-chi2", 3, True),
        ("shapefit.cfg.psf.min_convergence_rate", 3, True),
        ("shapefit.cfg.psf.max_iterations", None, True),
        ("shapefit.cfg.psf.ignore_dropped", None, True),
        ("shapefit.cfg.src.cover_bicubic_grid", None, True),
        ("shapefit.cfg.src.min_signal_to_noise", 3, True),
        ("shapefit.cfg.src.max_aperture", 3, True),
        ("shapefit.cfg.src.max_sat_frac", 3, True),
        ("shapefit.cfg.src.min_pix", None, True),
        ("shapefit.cfg.src.max_pix", None, True),
        ("shapefit.cfg.src.min_bg_pix", None, True),
        ("shapefit.cfg.src.max_count", None, True),
        ("shapefit.cfg.psf.bicubic.grid.x", None, True),
        ("shapefit.cfg.psf.bicubic.grid.y", None, True),
        ("shapefit.cfg.psf.bicubic.pixrej", 3, True),
        ("shapefit.cfg.psf.bicubic.initial_aperture", 3, True),
        ("shapefit.cfg.psf.bicubic.max_rel_amplitude_change", 3, True),
        ("shapefit.cfg.psf.bicubic.smoothing", 3, True),
        ("shapefit.magfit.cfg.single_photref", None, True),
        ("shapefit.magfit.cfg.correction_type", None, True),
        ("shapefit.magfit.cfg.correction", None, True),
        ("shapefit.magfit.cfg.require", None, True),
        ("shapefit.magfit.cfg.max_src", None, True),
        ("shapefit.magfit.cfg.noise_offset", 3, True),
        ("shapefit.magfit.cfg.max_mag_err", 3, True),
        ("shapefit.magfit.cfg.rej_level", 3, True),
        ("shapefit.magfit.cfg.max_rej_iter", None, True),
        ("shapefit.magfit.cfg.error_avg", None, True),
        ("shapefit.magfit.cfg.count_weight_power", 3, True),
        ("apphot.cfg.error_floor", 3, True),
        ("apphot.cfg.gain", 3, True),
        ("apphot.cfg.magnitude_1adu", 5, True),
        ("apphot.magfit.cfg.single_photref", None, True),
        ("apphot.magfit.cfg.correction_type", None, True),
        ("apphot.magfit.cfg.correction", None, True),
        ("apphot.magfit.cfg.require", None, True),
        ("apphot.magfit.cfg.max_src", None, True),
        ("apphot.magfit.cfg.noise_offset", 3, True),
        ("apphot.magfit.cfg.max_mag_err", 3, True),
        ("apphot.magfit.cfg.rej_level", 3, True),
        ("apphot.magfit.cfg.max_rej_iter", None, True),
        ("apphot.magfit.cfg.error_avg", None, True),
        ("apphot.magfit.cfg.count_weight_power", 3, True),
        ("catalogue.cfg.orientation", None, False),
        ("catalogue.cfg.filter", None, True),
        ("catalogue.cfg.name", None, True),
        ("catalogue.cfg.fov", None, False),
        ("catalogue.cfg.epoch", None, False),
    ]:
        dr_attribute = (
            db_session.query(HDF5Attribute)
            .filter_by(
                hdf5_structure_version_id=dr_structure_version_id,
                pipeline_key=pipeline_key,
            )
            .one()
        )

        lc_path = (
            transform_dr_to_lc_path(pipeline_key, dr_attribute.parent)
            + ("/Configuration" if is_config else "")
            + "/"
            + dr_attribute.name
        )

        args = {
            "pipeline_key": pipeline_key,
            "abspath": lc_path,
            "dtype": dr_attribute.dtype,
            "description": dr_attribute.description,
        }
        if scaleoffset is None:
            args["compression"] = "gzip"
            args["compression_options"] = "9"
        else:
            args["scaleoffset"] = scaleoffset
            if dr_attribute.dtype == "numpy.float64":
                args["replace_nonfinite"] = _default_nonfinite

        result.append(HDF5DataSet(**args))

    return result


def _get_data_reduction_dataset_datasets(db_session):
    """Return datasets built from entries for the source in DR datasets."""

    result = []
    magfit_datasets = []

    dr_structure_version_id = _get_structure_version_id(db_session)
    for pipeline_key in [
        "srcproj.columns",
        "bg.value",
        "bg.error",
        "bg.npix",
        "shapefit.num_pixels",
        "shapefit.signal_to_noise",
        "shapefit.magnitude",
        "shapefit.magnitude_error",
        "shapefit.quality_flag",
        "shapefit.chi2",
        "shapefit.magfit.magnitude",
        "apphot.magnitude",
        "apphot.magnitude_error",
        "apphot.quality_flag",
        "apphot.magfit.magnitude",
    ]:
        dr_dataset = (
            db_session.query(HDF5DataSet)
            .filter_by(
                hdf5_structure_version_id=dr_structure_version_id,
                pipeline_key=pipeline_key,
            )
            .one()
        )

        lc_path = transform_dr_to_lc_path(pipeline_key, dr_dataset.abspath)
        if pipeline_key.endswith("magfit.magnitude"):
            lc_path += "/Magnitude"

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
                description=dr_dataset.description,
            )
        )
        if pipeline_key.endswith("magfit.magnitude"):
            magfit_datasets.append(result[-1])

    return result, magfit_datasets


def _get_sky_position_datasets():
    """Return datasets describing when and where on the sky the source is."""

    path_start = _default_paths["sky_position"] + "/"

    return [
        HDF5DataSet(
            pipeline_key="skypos.per_source",
            abspath=path_start + "PerSource",
            dtype="numpy.bool_",
            description="Were sky position quantities calcualated individually "
            "for each source, as opposed to assuming the values for the frame "
            "center apply to all sources in the frame.",
        ),
        HDF5DataSet(
            pipeline_key="skypos.BJD",
            abspath=path_start + "BJD",
            dtype="numpy.float64",
            scaleoffset=8,
            description="The barycentric Julian Date of this light curve data "
            "point.",
        ),
        HDF5DataSet(
            pipeline_key="skypos.hour_angle",
            abspath=path_start + "HourAngle",
            dtype="numpy.float64",
            scaleoffset=6,
            description="The hour angle of the source for this light curve data"
            " point.",
        ),
        HDF5DataSet(
            pipeline_key="skypos.a180",
            abspath=path_start + "Azimuth180",
            dtype="numpy.float64",
            scaleoffset=6,
            description="The azimuth angle of the source for this light curve "
            "data point in degrees in the range (-180, 180].",
        ),
        HDF5DataSet(
            pipeline_key="skypos.zenith_distance",
            abspath=path_start + "ZenithDistance",
            dtype="numpy.float64",
            scaleoffset=6,
            description="The zenith distance of the source for this light curve"
            " data point in degrees.",
        ),
    ]


def _get_detrended_datasets(magfit_datasets, mode="epd"):
    """
    Create the default datasets for storing detrending results.

    Args:
        magfit_datasets(dict):    The light curve datasets containing the
            magnitude fitted magnitudes indexed by the photometry method.
    """

    result = []
    for magfit_dset in magfit_datasets:
        print(f"Magfit dataset: {magfit_dset!s}")
        magfit_tail = "/MagnitudeFitting/Magnitude"
        assert magfit_dset.abspath.endswith(magfit_tail)
        root_path = (
            magfit_dset.abspath[: -len(magfit_tail)] + "/" + mode.upper() + "/"
        )
        detrend_key = magfit_dset.pipeline_key.replace(
            ".magfit.", "." + mode.lower() + "."
        )
        property_key_prefix = detrend_key.rsplit(".", 1)[0]
        config_key_prefix = property_key_prefix + ".cfg."
        cfg_path = root_path + "FitProperties/"

        result.extend(
            [
                HDF5DataSet(
                    pipeline_key=detrend_key,
                    abspath=(root_path + "Magnitude"),
                    dtype=magfit_dset.dtype,
                    scaleoffset=magfit_dset.scaleoffset,
                    compression=magfit_dset.compression,
                    compression_options=magfit_dset.compression_options,
                    replace_nonfinite=magfit_dset.replace_nonfinite,
                    description=f"The {mode} corrected magnitude fitted magnitudes.",
                ),
                HDF5DataSet(
                    pipeline_key=(property_key_prefix + ".fit_residual"),
                    abspath=(cfg_path + "FitResidual"),
                    dtype=magfit_dset.dtype,
                    scaleoffset=3,
                    replace_nonfinite=_default_nonfinite,
                    description=(
                        "The residual of the last iteration of the iterative "
                        f"rejction {mode} fit."
                    ),
                ),
                HDF5DataSet(
                    pipeline_key=(property_key_prefix + ".num_fit_points"),
                    abspath=(cfg_path + "NumberFitPoints"),
                    dtype="numpy.uint",
                    compression="gzip",
                    compression_options="9",
                    description=(
                        "The number of points used in the last iteration of the "
                        f"iterative rejction {mode} fit."
                    ),
                ),
                HDF5DataSet(
                    pipeline_key=config_key_prefix + "error_avg",
                    abspath=(cfg_path + "ErrorAveraging"),
                    dtype="numpy.string_",
                    compression="gzip",
                    compression_options="9",
                    description="How to calculate the scale for rejecting sources.",
                ),
                HDF5DataSet(
                    pipeline_key=config_key_prefix + "rej_level",
                    abspath=(cfg_path + "RejectionLevel"),
                    dtype="numpy.float64",
                    compression="gzip",
                    compression_options="9",
                    description="Points rej_level times average error away from the"
                    " best fit are rejected and the fit is repeated.",
                ),
                HDF5DataSet(
                    pipeline_key=config_key_prefix + "max_rej_iter",
                    abspath=(cfg_path + "MaxRejectionIterations"),
                    dtype="numpy.uint",
                    compression="gzip",
                    compression_options="9",
                    description="Stop rejecting outlier points after this number "
                    "of rejection/refitting cycles.",
                ),
                HDF5DataSet(
                    pipeline_key=(property_key_prefix + ".cfg_index"),
                    abspath=(root_path + "FitPropertiesIndex"),
                    dtype="numpy.uint",
                    compression="gzip",
                    compression_options="9",
                    description=(
                        f"The index within the datasets containing {mode} fit "
                        "properties applicable to each data point."
                    ),
                ),
            ]
        )
        if mode == "epd":
            result.extend(
                [
                    HDF5DataSet(
                        pipeline_key=config_key_prefix + "variables",
                        abspath=(cfg_path + "Variables"),
                        dtype="numpy.string_",
                        compression="gzip",
                        compression_options="9",
                        description="The list of variables and the datasets they "
                        "correspond to used in the fitting.",
                    ),
                    HDF5DataSet(
                        pipeline_key=config_key_prefix + "fit_terms",
                        abspath=(cfg_path + "CorrectionExpression"),
                        dtype="numpy.string_",
                        compression="gzip",
                        compression_options="9",
                        description="The expression that expands to the terms to "
                        "include in the EPD fit.",
                    ),
                    HDF5DataSet(
                        pipeline_key=config_key_prefix + "fit_filter",
                        abspath=(cfg_path + "Filter"),
                        dtype="numpy.string_",
                        compression="gzip",
                        compression_options="9",
                        description="Filtering applied to select points to which to"
                        " apply the correction.",
                    ),
                    HDF5DataSet(
                        pipeline_key=config_key_prefix + "fit_weights",
                        abspath=(cfg_path + "WeightsExpression"),
                        dtype="numpy.string_",
                        compression="gzip",
                        compression_options="9",
                        description=(
                            "The expression that expands to the weights used for "
                            f"each point in the {mode} fit."
                        ),
                    ),
                ]
            )
        elif mode == "tfa":
            result.extend(
                [
                    HDF5DataSet(
                        pipeline_key=config_key_prefix + "saturation_magnitude",
                        abspath=(cfg_path + "SaturationMagnitude"),
                        dtype="numpy.float64",
                        compression="gzip",
                        compression_options="9",
                        description="The magnitude below which sources are "
                        "considered saturated and hence are excused from the rms vs"
                        " magnitude fit.",
                    ),
                    HDF5DataSet(
                        pipeline_key=config_key_prefix
                        + "mag_rms_dependence_order",
                        abspath=(cfg_path + "MagnitudeRMSDependenceOrder"),
                        dtype="numpy.uint",
                        compression="gzip",
                        compression_options="9",
                        description="The polynomial order of the dependence to fit "
                        "for RMS (after EPD) vs magnitude, when identifying quiet "
                        "stars.",
                    ),
                    HDF5DataSet(
                        pipeline_key=(
                            config_key_prefix + "mag_rms_outlier_threshold"
                        ),
                        abspath=(cfg_path + "MagRMSOutlierThreshold"),
                        dtype="numpy.float64",
                        compression="gzip",
                        compression_options="9",
                        description="Stars are not allowed to be in the template if"
                        " their RMS is more than this many sigma away from the "
                        "mag-rms fit. This is also the threshold used for rejecting"
                        " outliers when doing the iterative fit for the rms as a "
                        "function of magnutude.",
                    ),
                    HDF5DataSet(
                        pipeline_key=config_key_prefix + "mag_rms_max_rej_iter",
                        abspath=(cfg_path + "MaxMagRMSRejectionIterations"),
                        dtype="numpy.uint",
                        compression="gzip",
                        compression_options="9",
                        description="The maximum number of rejection fit iterations"
                        " to do when deriving the rms(mag) dependence.",
                    ),
                    HDF5DataSet(
                        pipeline_key=config_key_prefix + "max_rms",
                        abspath=(cfg_path + "MaxRMS"),
                        dtype="numpy.float64",
                        compression="gzip",
                        compression_options="9",
                        description="Stars are allowed to be in the template only "
                        "if their RMS is no larger than this.",
                    ),
                    HDF5DataSet(
                        pipeline_key=config_key_prefix + "faint_mag_limit",
                        abspath=(cfg_path + "FaintMagnitudeLimit"),
                        dtype="numpy.float64",
                        compression="gzip",
                        compression_options="9",
                        description="Stars fainter than this cannot be template "
                        "stars.",
                    ),
                    HDF5DataSet(
                        pipeline_key=(
                            config_key_prefix + "min_observations_quantile"
                        ),
                        abspath=(cfg_path + "MinimumObservationsQuantile"),
                        dtype="numpy.float64",
                        compression="gzip",
                        compression_options="9",
                        description="The minimum number of observations required of"
                        " template stars is the smaller of this quantile among the "
                        "input collection of stars and that determined by "
                        "MinimumObservationsFraction.",
                    ),
                    HDF5DataSet(
                        pipeline_key=(
                            config_key_prefix + "min_observations_fraction"
                        ),
                        abspath=(cfg_path + "MinimumObservationsFraction"),
                        dtype="numpy.float64",
                        compression="gzip",
                        compression_options="9",
                        description="The minimum number of observations required of"
                        " template stars is the smaller this fraction of the "
                        "longest lightcurve and that determined by "
                        "MinimumObservationsQuantile",
                    ),
                    HDF5DataSet(
                        pipeline_key=config_key_prefix + "num_templates",
                        abspath=(cfg_path + "NumberTemplates"),
                        dtype="numpy.uint",
                        compression="gzip",
                        compression_options="9",
                        description="The maximum number of template stars to use.",
                    ),
                    HDF5DataSet(
                        pipeline_key=(config_key_prefix + "variables"),
                        abspath=(cfg_path + "PointsFilterVariables"),
                        dtype="numpy.string_",
                        compression="gzip",
                        compression_options="9",
                        description="The variables to use for selecting which "
                        "points from a LC can be part of a template or can "
                        "participate in the de-trending fit.",
                    ),
                    HDF5DataSet(
                        pipeline_key=(
                            config_key_prefix + "fit_points_filter_expression"
                        ),
                        abspath=(cfg_path + "PointsFilterExpression"),
                        dtype="numpy.string_",
                        compression="gzip",
                        compression_options="9",
                        description="The expression defining which points from a LC"
                        " can be part of a template or can participate in the "
                        "de-trending fit.",
                    ),
                ]
            )

    return result


def _get_configuration_index_datasets(db_session):
    """Return a list of datasets of indicies within configuration datasets."""

    result = []

    storage_options = {
        "dtype": "numpy.uint",
        "compression": "gzip",
        "compression_options": "9",
        "description": (
            "The index within the configuration datasets containing "
            "the configuration used for this light curve point."
        ),
    }
    key_tail = ".cfg_index"
    path_tail = "/ConfigurationIndex"

    dr_structure_version_id = _get_structure_version_id(db_session)
    for photometry_method in ["shapefit", "apphot"]:
        magfit_dataset = (
            db_session.query(HDF5DataSet)
            .filter_by(
                hdf5_structure_version_id=dr_structure_version_id,
                pipeline_key=photometry_method + ".magfit.magnitude",
            )
            .one()
        )

        magfit_path = transform_dr_to_lc_path(
            magfit_dataset.pipeline_key, magfit_dataset.abspath
        )

        result.extend(
            [
                HDF5DataSet(
                    pipeline_key=photometry_method + key_tail,
                    abspath=(
                        "/"
                        + magfit_path.strip("/").split("/", 1)[0]
                        + path_tail
                    ),
                    **storage_options,
                ),
                HDF5DataSet(
                    pipeline_key=(
                        magfit_dataset.pipeline_key.rsplit(".", 1)[0] + key_tail
                    ),
                    abspath=magfit_path + path_tail,
                    **storage_options,
                ),
            ]
        )

    result.extend(
        [
            HDF5DataSet(
                pipeline_key="srcextract.psf_map" + key_tail,
                abspath=_default_paths["srcextract_psf_map"] + path_tail,
                **storage_options,
            ),
            HDF5DataSet(
                pipeline_key="fitsheader" + key_tail,
                abspath="/FrameInformation" + path_tail,
                **storage_options,
            ),
            HDF5DataSet(
                pipeline_key="skytoframe" + key_tail,
                abspath=(
                    _dr_default_paths["skytoframe"]["root"].rsplit("/", 1)[0]
                    + path_tail
                ),
                **storage_options,
            ),
            HDF5DataSet(
                pipeline_key="bg" + key_tail,
                abspath=(
                    _dr_default_paths["background"].rsplit("/", 1)[0]
                    + path_tail
                ),
                **storage_options,
            ),
        ]
    )

    return result


def _get_attributes(db_session):
    """Return a list of all attributes of light cuves."""

    aperture_size_attribute = (
        db_session.query(HDF5Attribute)
        .filter_by(
            hdf5_structure_version_id=_get_structure_version_id(db_session),
            pipeline_key="apphot.cfg.aperture",
        )
        .one()
    )

    return _get_catalogue_attributes() + [
        HDF5Attribute(
            pipeline_key="confirmed_lc_length",
            parent="/",
            name="LightCurveLength",
            dtype="numpy.uint",
            description="How many data points are currently present in the "
            "lightcurve.",
        ),
        HDF5Attribute(
            pipeline_key="apphot.cfg.aperture",
            parent=transform_dr_to_lc_path(
                "apphot.cfg.aperture", aperture_size_attribute.parent
            ),
            name=aperture_size_attribute.name,
            dtype=aperture_size_attribute.dtype,
            description=aperture_size_attribute.description,
        ),
    ]


def _get_datasets(db_session):
    """Return a list of all datasets in light curves."""

    (dr_dataset_datasets, magfit_datasets) = (
        _get_data_reduction_dataset_datasets(db_session)
    )
    return (
        _get_source_extraction_datasets()
        + _get_frame_datasets()
        + _get_data_reduction_attribute_datasets(db_session)
        + dr_dataset_datasets
        + _get_configuration_index_datasets(db_session)
        + _get_sky_position_datasets()
        + _get_detrended_datasets(magfit_datasets, "epd")
        + _get_detrended_datasets(magfit_datasets, "tfa")
    )


def get_default_light_curve_structure(db_session):
    """Create default configuration for the layout of light curve files."""

    default_structure = HDF5Product(
        pipeline_key="light_curve",
        description=("Contains all per-source processing information."),
    )
    default_structure.structure_versions = [HDF5StructureVersion(version=0)]
    default_structure.structure_versions[0].attributes = _get_attributes(
        db_session
    )
    default_structure.structure_versions[0].datasets = _get_datasets(db_session)
    return default_structure
