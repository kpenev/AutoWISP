"""Interface for downloading TESS lightcurves."""

from glob import glob
from os import path
import re

import numpy
from astroquery.mast import Observations

# from astroquery.exceptions import InvalidQueryError
from astropy.io import fits

# import lightkurve


def get_tess_lightcurve(tic, sector, provenance="SPOC"):
    """Return the lightcurve using astroquery interface."""

    def format_result(fits_list):
        """Format the result appropriately."""

        print("FITS list: " + repr(fits_list))
        result = {}
        for fits_path, fits_sector in fits_list:
            print(f"Opening: {fits_path!r}")
            with fits.open(fits_path, "readonly") as fits_f:
                lightcurve = fits_f[1].data[:]
            result[fits_sector] = lightcurve

        if sector == "all":
            print(f"Returning {len(result):d} LCs")
            return result
        return result[sector]

    # Too ugly
    # pylint: disable=consider-using-f-string
    sector_tic_fname_part = "s{sector}-{tic:016d}".format(
        sector=("*" if sector == "all" else f"{sector:04d}".format()), tic=tic
    )
    # pylint: enable=consider-using-f-string

    if provenance == "QLP":
        fits_list = glob(
            path.join(
                "mastDownload",
                "HLSP",
                f"hlsp_qlp_tess_ffi_{sector_tic_fname_part}_tess_v01_llc",
                f"hlsp_qlp_tess_ffi_{sector_tic_fname_part}_tess_v01_llc.fits",
            )
        )
        parse_sector_rex = re.compile(
            ".*hlsp_qlp_tess_ffi_s(?P<sector>[0-9]*)-[0-9]*_tess_v01_llc"
        )
    else:
        fits_list = glob(
            path.join(
                "mastDownload",
                "TESS",
                f"tess*-{sector_tic_fname_part}-*-s",
                f"tess*-{sector_tic_fname_part}-*-s_lc.fits",
            )
        )
        parse_sector_rex = re.compile(
            ".*tess.*-s(?P<sector>[0-9]*)-[0-9]*-.*-s"
        )

    print("Fits list: " + repr(fits_list))
    assert not fits_list or sector == "all" or len(fits_list) == 1

    print("Fits path: " + repr(fits_list))
    if fits_list:
        fits_list = [
            (fits_fname, int(parse_sector_rex.match(fits_fname)["sector"]))
            for fits_fname in fits_list
        ]
    else:
        # False positive
        # pylint: disable=no-member
        objects = Observations.query_object("TIC" + str(tic), radius=0.001)
        # pylint: enable=no-member

        print(f"\tFound {len(objects)} objects:\n{objects!r}")

        selection = numpy.logical_and(
            objects["obs_collection"]
            == ("TESS" if provenance == "SPOC" else "HLSP"),
            objects["dataproduct_type"] == "timeseries",
        )
        selection = numpy.logical_and(selection, objects["project"] == "TESS")
        selection = numpy.logical_and(
            selection, objects["provenance_name"] == provenance
        )

        print(
            "\tAvailable sectors: "
            + repr(objects["sequence_number"][selection])
        )

        if sector != "all":
            selection = numpy.logical_and(
                selection, objects["sequence_number"] == sector
            )
            print(
                f"\tSelected {selection.sum():d} objects from sectors:\n"
                + repr(objects[selection]["sequence_number"])
            )

        # False positive
        # pylint: disable=bad-string-format-type
        print(f"\tSelected {selection.sum():d}/{len(objects):d} objects")
        # pylint: enable=bad-string-format-type

        fits_list = []
        for get_object in objects[selection]:
            # False positive
            # pylint: disable=no-member
            products = Observations.get_product_list(get_object)
            # pylint: enable=no-member

            print("\tFound %d products:" + repr(products))

            product_selection = numpy.logical_and(
                products["productType"] == "SCIENCE",
                products["description"]
                == ("Light curves" if provenance == "SPOC" else "FITS"),
            )
            if product_selection.sum() == 0:
                continue
            if provenance == "SPOC" and sector != "all":
                if product_selection.sum() != 1:
                    print(
                        "Ambiguous products:\n"
                        + repr(products[product_selection])
                    )
                assert product_selection.sum() == 1
            download = Observations.download_products(
                products[product_selection]
            )
            fits_list.append(
                (download["Local Path"], get_object["sequence_number"])
            )

    return format_result(fits_list)
