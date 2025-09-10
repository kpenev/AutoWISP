"""Define default steps and masters for the data processing pipeline."""

master_info = {
    "zero": {
        "must_match": frozenset(("CAMERAID", "CLRCHNL")),
        "config_name": "master-bias",
        "created_by": ("stack_to_master", "zero"),
        "split_by": frozenset(("OBSSSNID",)),
        "used_by": [
            ("calibrate", "dark", False),
            ("calibrate", "flat", False),
            ("calibrate", "object", False),
        ],
        "description": "An estimate of the zero level of a camera.",
    },
    "dark": {
        "must_match": frozenset(("CAMERAID", "CLRCHNL")),
        "config_name": "master-dark",
        "created_by": ("stack_to_master", "dark"),
        "split_by": frozenset(("OBSSSNID",)),
        "used_by": [
            ("calibrate", "flat", False),
            ("calibrate", "object", False),
        ],
        "description": "An estimate of the dark current of a camera.",
    },
    "highflat": {
        "must_match": frozenset(("CAMERAID", "TELSCPID", "CLRCHNL")),
        "config_name": "master-flat",
        "created_by": ("stack_to_master_flat", "flat"),
        "split_by": frozenset(("OBSSSNID",)),
        "used_by": [("calibrate", "object", False)],
        "description": "An estimate of the relative sensitivity of image "
        "pixels to light from infinity entering the telescope. Constructed from"
        " flat frames with high (but not saturated) light.",
    },
    "lowflat": {
        "must_match": frozenset(("CAMERAID", "TELSCPID", "CLRCHNL")),
        "config_name": "low-flat-master-fname",
        "created_by": ("stack_to_master_flat", "flat"),
        "split_by": frozenset(("OBSSSNID",)),
        "used_by": [],
        "description": "An estimate of the relative sensitivity of image "
        "pixels to light from infinity entering the telescope. Constructed from"
        " flat frames with low light.",
    },
    "single_photref": {
        "must_match": frozenset(("TARGETID", "CLRCHNL", "EXPTIME")),
        "config_name": "single-photref-dr-fname",
        "created_by": None,
        "split_by": frozenset(),
        "used_by": [
            ("fit_magnitudes", "object", False),
            ("create_lightcurves", "object", False),
            ("epd", "object", False),
            ("generate_epd_statistics", "object", False),
            ("tfa", "object", False),
            ("generate_tfa_statistics", "object", False),
        ],
        "description": "The reference image to use to start magnitude "
        "fitting. Subsequently replaced by average of the corrected "
        "brightnes of each star.",
    },
    "master_photref": {
        "must_match": frozenset(("TARGETID", "CLRCHNL", "EXPTIME")),
        "config_name": "master-photref-dr-fname",
        "created_by": ("fit_magnitudes", "object"),
        "split_by": frozenset(),
        "used_by": [("fit_magnitudes", "object", True)],
        "description": "The master photometric reference to use for magnitude "
        "fitting if available.",
    },
    "magfit_stat": {
        "must_match": frozenset(("TARGETID", "CLRCHNL", "EXPTIME")),
        "config_name": "magfit-stat-fname",
        "created_by": ("fit_magnitudes", "object"),
        "split_by": frozenset(),
        "used_by": [],
        "description": "The statistics file generated during magnitude "
        "fitting.",
    },
    "magfit_catalog": {
        "must_match": frozenset(("TARGETID", "CLRCHNL", "EXPTIME")),
        "config_name": "magfit-catalog-fname",
        "created_by": ("fit_magnitudes", "object"),
        "split_by": frozenset(),
        "used_by": [],
        "description": "The catalog file generated during magnitude fitting.",
    },
    "lightcurve_catalog": {
        "must_match": frozenset(("TARGETID", "CLRCHNL", "EXPTIME")),
        "config_name": "detrending-catalog",
        "created_by": ("create_lightcurves", "object"),
        "split_by": frozenset(),
        "used_by": [
            ("epd", "object", False),
            ("generate_epd_statistics", "object", False),
            ("tfa", "object", False),
            ("generate_tfa_statistics", "object", False),
        ],
        "description": "The catalog file generated for collecting lightcurves.",
    },
    "epd_stat": {
        "must_match": frozenset(("TARGETID", "CLRCHNL", "EXPTIME")),
        "config_name": "epd-statistics-fname",
        "created_by": ("generate_epd_statistics", "object"),
        "split_by": frozenset(),
        "used_by": [("tfa", "object", False)],
        "description": "The statistics file showing the performance after EPD.",
    },
    "tfa_stat": {
        "must_match": frozenset(("TARGETID", "CLRCHNL", "EXPTIME")),
        "config_name": "tfa-statistics-fname",
        "created_by": ("generate_tfa_statistics", "object"),
        "split_by": frozenset(),
        "used_by": [],
        "description": "The statistics file showing the performance after TFA.",
    },
}


step_dependencies = [
    ("add_images_to_db", None, []),
    ("calibrate", "zero", []),
    ("stack_to_master", "zero", [("calibrate", "zero")]),
    ("calibrate", "dark", [("stack_to_master", "zero")]),
    ("stack_to_master", "dark", [("calibrate", "dark")]),
    (
        "calibrate",
        "flat",
        [("stack_to_master", "zero"), ("stack_to_master", "dark")],
    ),
    ("stack_to_master_flat", "flat", [("calibrate", "flat")]),
    (
        "calibrate",
        "object",
        [
            ("stack_to_master", "zero"),
            ("stack_to_master", "dark"),
            ("stack_to_master_flat", "flat"),
        ],
    ),
    ("find_stars", "object", [("calibrate", "object")]),
    ("solve_astrometry", "object", [("find_stars", "object")]),
    (
        "fit_star_shape",
        "object",
        [("solve_astrometry", "object"), ("calibrate", "object")],
    ),
    (
        "measure_aperture_photometry",
        "object",
        [("fit_star_shape", "object"), ("calibrate", "object")],
    ),
    (
        "fit_source_extracted_psf_map",
        "object",
        [("find_stars", "object"), ("solve_astrometry", "object")],
    ),
    (
        "fit_magnitudes",
        "object",
        [
            ("solve_astrometry", "object"),
            ("fit_star_shape", "object"),
            ("measure_aperture_photometry", "object"),
            ("fit_source_extracted_psf_map", "object"),
        ],
    ),
    (
        "create_lightcurves",
        "object",
        [
            ("solve_astrometry", "object"),
            ("fit_star_shape", "object"),
            ("measure_aperture_photometry", "object"),
            ("fit_magnitudes", "object"),
            ("fit_source_extracted_psf_map", "object"),
        ],
    ),
    (
        "calculate_photref_merit",
        "object",
        [
            ("calibrate", "object"),
            ("find_stars", "object"),
            ("solve_astrometry", "object"),
            ("fit_star_shape", "object"),
            ("fit_source_extracted_psf_map", "object"),
        ],
    ),
    (
        "epd",
        "object",
        [("create_lightcurves", "object"), ("fit_magnitudes", "object")],
    ),
    (
        "generate_epd_statistics",
        "object",
        [("create_lightcurves", "object"), ("epd", "object")],
    ),
    (
        "tfa",
        "object",
        [
            ("epd", "object"),
            ("generate_epd_statistics", "object"),
            ("create_lightcurves", "object"),
            ("fit_magnitudes", "object"),
        ],
    ),
    (
        "generate_tfa_statistics",
        "object",
        [("create_lightcurves", "object"), ("tfa", "object")],
    ),
]



