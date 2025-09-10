#!/usr/bin/env python3

"""Find the appropriate astrometry-catalog-max-magnitude based on the
flux threshold"""

import contextlib
import pandas as pd
import numpy as np
from configargparse import ArgumentParser, DefaultsFormatter

from astropy.coordinates import SkyCoord
from astropy import units
from asteval import Interpreter
from scipy.spatial import cKDTree

from autowisp.data_reduction.data_reduction_file import DataReductionFile
from autowisp.astrometry import astrometry
from autowisp.processing_steps.solve_astrometry import (
    add_anet_cmdline_args,
    prepare_configuration,
)
from autowisp.catalog import create_catalog_file, read_catalog_file


def parse_command_line():
    """Return the parsed command line arguments."""

    parser = ArgumentParser(
        description="Match astrometry catalog with Gaia DR3",
        formatter_class=DefaultsFormatter,
    )
    parser.add_argument(
        "--dr-paths",
        "-d",
        nargs="+",
        type=str,
        required=True,
        help="Path to the DR source extract file(s)",
    )
    parser.add_argument(
        "--flux-threshold",
        "-f",
        type=float,
        required=True,
        help="Flux threshold for matching",
    )
    parser.add_argument(
        "--image-margin",
        "-m",
        type=float,
        default=1.1,
        help="Image margin for finding the magnitude of the faintest star"
        "of the brightest stars that fits in the image. It is a safety"
        "margin to account for: some extracted sources are fake, and not all"
        "real sources are extracted."
        "(default: 1.1)",
    )

    add_anet_cmdline_args(parser)

    return parser.parse_args()


def read_dr_file(dr_path, cmdline_args):
    """
    Read DR file and solve the initial astrometry for it.

    Args:
        dr_path (str): Path to the data reduction file.
        cmdline_args (dict or Namespace): Command line arguments or a
        dictionary of configuration parameters.

    Returns:
        field_corr (np.ndarray): Initial correlation data from astrometry.net.
        dr_df (pd.DataFrame): DataFrame containing the extracted sources from
        the DR file.
    """

    if not isinstance(cmdline_args, dict):
        config = vars(cmdline_args)
    else:
        config = cmdline_args

    web_lock = contextlib.nullcontext()

    with DataReductionFile(dr_path, "r") as dr_file:

        dr_header = dr_file.get_frame_header()
        config = prepare_configuration(config, dr_header)
        # we just needed the frame-fov-estimate to be converted from str

        fov_estimate = max(*config["frame_fov_estimate"]).to_value("deg")

        config["tweak_order_range"] = config["tweak_order"]
        config["fov_range"] = (
            fov_estimate / config["image_scale_factor"],
            fov_estimate * config["image_scale_factor"],
        )

        srcextract_version = 0
        dr_df = dr_file.get_sources(
            "srcextract.sources",
            "srcextract_column_name",
            srcextract_version=srcextract_version,
        )

        xy_extracted = dr_df[["x", "y"]].values
        xy_extracted_struct = np.zeros(
            xy_extracted.shape[0], dtype=[("x", "f8"), ("y", "f8")]
        )
        xy_extracted_struct["x"] = xy_extracted[:, 0]
        xy_extracted_struct["y"] = xy_extracted[:, 1]

        field_corr, _ = astrometry.get_initial_corr(  # _: tweak order
            dr_file=dr_file,
            xy_extracted=xy_extracted_struct,
            config=config,
            header=None,
            web_lock=web_lock,
        )
        print(field_corr)

    return field_corr, dr_df


def match_sources(field_corr, dr_df):
    """
    Match sources from the corr file with the DR file using a KDTree.

    Args:
        field_corr (np.ndarray): Initial correlation data from astrometry.net.
        dr_df (pd.DataFrame): DataFrame containing the extracted sources from
        the DR file.

    Returns:
        corr_df (pd.DataFrame): DataFrame containing the matched sources.
        n_extracted (int): Number of extracted sources.
    """

    # convert field_corr with shape (n,) to pandas DataFrame
    # field_corr is a structured np array, before converting it to DataFrame:
    field_corr_native = field_corr.astype(field_corr.dtype.newbyteorder("="))
    corr_df = pd.DataFrame(
        field_corr_native,
        columns=["field_x", "field_y", "field_ra", "field_dec"],
    )

    # Applying kdtree for fast nearest neighbor search
    dr_tree = cKDTree(dr_df[["x", "y"]].values)
    corr_query = corr_df[["field_x", "field_y"]].values
    _, indices = dr_tree.query(corr_query)

    # Add matched info to corr_df
    corr_df["matched_dr_ix"] = indices
    corr_df["matched_dr_x"] = dr_df.iloc[indices]["x"].values
    corr_df["matched_dr_y"] = dr_df.iloc[indices]["y"].values
    corr_df["matched_dr_flux"] = dr_df.iloc[indices]["flux"].values

    corr_matched = corr_df.reset_index(drop=True)

    # To use the second method of approximating the magnitude limit
    n_extracted = dr_df.shape[0]

    print(
        f"\nOut of {n_extracted} extracted sources, "
        f"corr_matched is found as: \n {corr_matched} "
    )

    return corr_matched, n_extracted


# pylint: disable=too-many-locals
def query_gaia(
    corr_matched,
    n_extracted,
    image_scale_factor,
    frame_fov_estimate=None,
    image_margin=1.1
):
    """
    Query Gaia DR3 catalog for sources matching the extracted sources.

    Args:
        corr_matched (pd.DataFrame): DataFrame containing the matched
            sources.
        n_extracted (int): Number of extracted sources from the DR file.
        image_scale_factor (float): A safety factor originally to account for
            solving astrometry in a range of fov/factor to fov*factor.
        frame_fov_estimate (tuple, optional): Estimated field of view of the
            frame.

    Returns:
        corr_matched_sorted (pd.DataFrame): DataFrame containing the matched
            sources sorted by magnitude with Gaia phot_g_mean_mag.
    """

    # estimate the frame center from astrometry.net correlation file
    ra_center = np.mean(corr_matched["field_ra"])
    dec_center = np.mean(corr_matched["field_dec"])

    height = Interpreter({'units': units})(frame_fov_estimate[0])
    width = Interpreter({'units': units})(frame_fov_estimate[1])

    create_catalog_file(
        "cat.fits",
        overwrite=True,
        ra=ra_center * units.deg,
        dec=dec_center * units.deg,
        width=width,
        height=height,
        magnitude_expression="phot_g_mean_mag",
        epoch=2025 * units.yr,
        magnitude_limit=20.0,  # Needed for WISPGaia.query_brightness_limited()
        columns=["source_id", "ra", "dec", "phot_g_mean_mag"],
        order_by="magnitude",
        order_dir="ASC",
        max_objects=int(n_extracted * image_scale_factor**2),
    )

    gaia_results = read_catalog_file("cat.fits").reset_index(drop=True)

    print(f"Catalog created with {len(gaia_results)} sources.")

    def make_coords(df, ra_col, dec_col):
        return SkyCoord(
            ra=df[ra_col].values * units.deg, dec=df[dec_col].values * units.deg
        )

    src_coords = make_coords(corr_matched, "field_ra", "field_dec")
    gaia_coords = make_coords(gaia_results, "RA", "Dec")

    idx, d2d, _ = src_coords.match_to_catalog_sky(gaia_coords)

    # Set a matching threshold (Need a smaller one?)
    match_mask = d2d < 5.0 * units.arcsec
    print(f"Matched {np.sum(match_mask)} sources out of {len(corr_matched)}")

    g_mag_10_percent = round(
        gaia_results.iloc[int(n_extracted * image_margin)]["phot_g_mean_mag"], 2
    )
    print(
        f"Gaia G mag of the faintest star among {n_extracted} * {image_margin} "
        f"brightest sources in the FOV: {g_mag_10_percent} "
    )

    # Assign Gaia G magnitude to sources
    corr_matched["phot_g_mean_mag"] = np.nan
    corr_matched.loc[match_mask, "phot_g_mean_mag"] = gaia_results[
        "phot_g_mean_mag"
    ].iloc[idx[match_mask]]

    corr_matched_sorted = corr_matched.sort_values(
        by="phot_g_mean_mag", ascending=False
    ).reset_index(drop=True)

    return corr_matched_sorted

def main():
    """Main function to run the astrometry matching process."""

    cmdline_args = parse_command_line()
    dr_paths = cmdline_args.dr_paths
    flux_threshold = cmdline_args.flux_threshold
    image_scale_factor = cmdline_args.image_scale_factor

    mags = []
    for dr_path in dr_paths:
        corr_df, dr_df = read_dr_file(dr_path, cmdline_args)
        corr_matched, n_extracted = match_sources(corr_df, dr_df)
        corr_matched_sorted = query_gaia(
            corr_matched,
            n_extracted,
            image_scale_factor,
            cmdline_args.frame_fov_estimate,
            cmdline_args.image_margin,
        )
        corr_matched_sorted["cum_median"] = (
            (
                corr_matched_sorted["phot_g_mean_mag"]
                + 2.5 * np.log10(corr_matched_sorted["matched_dr_flux"])
            )
            .expanding()
            .median()
        )
        y0 = corr_matched_sorted["cum_median"].iloc[3]  # top 3 faintest sources
        mags.append(-2.5 * np.log10(flux_threshold) + y0 + 0.2)

    print(f"Suggested max_mag_astrometry based on "
          f"matching flux and mag ~ {np.array(mags).mean():.2f}")


if __name__ == "__main__":
    main()
