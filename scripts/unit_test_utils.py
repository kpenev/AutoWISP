"""Tools for creating unit tests for the pipeline."""

from astropy.io import fits


def get_frame_center(input_fname, output_fname, width=500, height=500):
    """Cut out the central portion of a FITS file and save to new FITS file."""

    with fits.open(input_fname, "readonly") as in_file:
        for hdu in in_file:
            if hdu.header["NAXIS"] == 0:
                continue
            center = (
                hdu.header["NAXIS1"] // 2,
                hdu.header["NAXIS2"] // 2,
            )
            y_slice = slice(
                center[1] - height // 2, center[1] + height - height // 2
            )
            x_slice = slice(
                center[0] - width // 2, center[0] + width - width // 2
            )
            fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.CompImageHDU(
                        hdu.data[y_slice, x_slice],
                        hdu.header,
                        quantize_level=0.5,
                    ),
                ]
            ).writeto(output_fname, overwrite=False)
            break


def parse_command_line():
    """Parse command line arguments for unit tests."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Unit test utilities for the pipeline."
    )
    parser.add_argument(
        "--get-frame-center",
        nargs="+",
        type=str,
        default=None,
        metavar="[input_fname, output_fname, [width, [height]]",
        help="Create a new image containing the central portion of an existing "
        "image. If the width and height are omitted, they default to 500 pixels"
        "each.",
    )

    return parser.parse_args()


def main(config):
    """Carry out the tasks specified by the command line arguments."""

    if config.get_frame_center is not None:
        get_frame_center(
            config.get_frame_center[0],
            config.get_frame_center[1],
            (
                int(config.get_frame_center[2])
                if len(config.get_frame_center) > 2
                else 500
            ),
            (
                int(config.get_frame_center[3])
                if len(config.get_frame_center) > 3
                else 500
            ),
        )


if __name__ == "__main__":
    main(parse_command_line())
