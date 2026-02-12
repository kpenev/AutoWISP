"""Query Gaia DR3 with just polygon and mag limit to make catalog test cases"""

from os import path

from astropy import units as u

from autowisp.catalog import gaia, write_query_to_file


def generate_test_data(num_stars=1000):
    """Create test data files for testing catalog queries."""

    for test_i, fov in enumerate(
        [
            {"ra": 25.0, "dec": 25.0, "width": 10.0, "height": 10.0},
            {"ra": 145.0, "dec": -45.0, "width": 5.0, "height": 20.0},
            {"ra": 0.0, "dec": 25.0, "width": 20.0, "height": 10.0},
            {"ra": 5.5, "dec": 25.0, "width": 10.0, "height": 20.0},
            {"ra": 355.5, "dec": 25.0, "width": 10.0, "height": 20.0},
            {"ra_corner": 0.0, "dec": 0.0, "width": 10.0, "height": 20.0},
            {"ra": 100.0, "dec": 80.0, "width": 10.0, "height": 10.0},
            {"ra": 100.0, "dec": -80.0, "width": 10.0, "height": 10.0},
            {"ra": 20.0, "dec": 80.0, "width": 10.0, "height": 10.0},
            {"ra": 20.0, "dec": -80.0, "width": 10.0, "height": 10.0},
            {"ra": 0.0, "dec": 90.0, "width": 10.0, "height": 10.0},
            {"ra": 0.0, "dec": -90.0, "width": 10.0, "height": 10.0},
            {"ra": 45.0, "dec": 90.0, "width": 10.0, "height": 10.0},
            {"ra": 90.0, "dec": -90.0, "width": 10.0, "height": 10.0},
            {"ra": 25.0, "dec": 0.0, "width": 10.0, "height": 10.0},
            {"ra": 0.0, "dec": 0.0, "width": 20.0, "height": 10.0},
            {"ra": 5.01, "dec": 9.0, "width": 10.0, "height": 10.0},
        ]
    ):
        cat_fname = f"catalog_tests/test_catalog_{test_i:02d}.fits"
        if path.exists(cat_fname):
            continue
        if "ra_corner" in fov:
            ra_corner = fov.pop("ra_corner")
            corners = gaia.estimate_fov_corners(
                ra=ra_corner * u.deg,
                dec=fov["dec"] * u.deg,
                width=fov["width"] * u.deg,
                height=fov["height"] * u.deg,
            )
            # Corners 0,1 are left side (xi = -width/2).
            # Compute signed RA offset handling 0/360 wrap.
            left_offsets = corners["RA"][:2] - ra_corner
            left_offsets[left_offsets > 180] -= 360
            left_offsets[left_offsets < -180] += 360
            fov["ra"] = ra_corner - min(left_offsets)
        fov_with_units = {key: value * u.deg for key, value in fov.items()}
        corners = gaia.estimate_fov_corners(**fov_with_units)
        fov_with_units["epoch"] = 2100.0 * u.yr
        query_str = (
            f"SELECT TOP {num_stars + 1}\n    "
            + "\n    ".join(
                [
                    "EPOCH_PROP_POS(ra, dec, parallax, pmra, pmdec, "
                    "radial_velocity, ref_epoch, "
                    f"{fov_with_units['epoch'].to_value('yr')}) AS propagated,",
                    "source_id,",
                    "ra,",
                    "dec,",
                    "pmra,",
                    "pmdec,",
                    "phot_g_n_obs,",
                    "phot_g_mean_mag,",
                    "phot_g_mean_flux,",
                    "phot_g_mean_flux_error,",
                    "phot_bp_n_obs,",
                    "phot_bp_mean_mag,",
                    "phot_bp_mean_flux,",
                    "phot_bp_mean_flux_error,",
                    "phot_rp_n_obs,",
                    "phot_rp_mean_mag,",
                    "phot_rp_mean_flux,",
                    "phot_rp_mean_flux_error,",
                    "phot_proc_mode,",
                    "phot_bp_rp_excess_factor,",
                    "(phot_g_mean_mag) AS magnitude",
                ]
            )
            + "\nFROM gaiadr3.gaia_source WHERE"
            + "\n    phot_g_mean_mag < 15"
            + "\n    AND 1 = CONTAINS("
            + "\n        POINT(ra, dec),"
            + "\n        POLYGON(\n            "
            + "\n            ".join(
                [
                    f"{corners[0]['RA']}, {corners[0]['Dec']}, ",
                    f"{corners[1]['RA']}, {corners[1]['Dec']}, ",
                    f"{corners[2]['RA']}, {corners[2]['Dec']}, ",
                    f"{corners[3]['RA']}, {corners[3]['Dec']}",
                ]
            )
            + "\n        )"
            + "\n    )"
            + "\nORDER BY phot_g_mean_mag ASC"
        )
        print(query_str)
        stars = gaia.get_result(query_str, ["RA", "Dec"], True)
        print(stars.columns)
        mag_limit = 0.5 * (
            stars["phot_g_mean_mag"][num_stars]
            + stars["phot_g_mean_mag"][num_stars - 1]
        )
        print(f"Mag limit: {mag_limit}")
        write_query_to_file(
            stars[:-1],
            cat_fname,
            True,
            **fov_with_units,
            magnitude_limit=mag_limit,
            magnitude_expression="phot_g_mean_mag",
        )


if __name__ == "__main__":
    # conf.timeout = 600
    generate_test_data()
