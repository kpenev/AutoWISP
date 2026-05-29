#!/usr/bin/env python3
"""Write a FITS file containing per-HDU pixel differences of two input files."""

import argparse
import sys

from astropy.io import fits
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file1", help="First FITS file")
    parser.add_argument(
        "file2", help="Second FITS file (subtracted from file1)"
    )
    parser.add_argument("output", help="Output FITS file with file1 - file2")
    return parser.parse_args()


def data_hdus(hdulist):
    """Return HDUs carrying image data, skipping the dummy primary HDU that
    compressed FITS files prepend."""

    if (
        len(hdulist) > 1
        and isinstance(hdulist[0], fits.PrimaryHDU)
        and hdulist[0].data is None
    ):
        return list(hdulist[1:])
    return list(hdulist)


def main():
    args = parse_args()

    with fits.open(args.file1) as hdus1, fits.open(args.file2) as hdus2:
        data1 = data_hdus(hdus1)
        data2 = data_hdus(hdus2)

        if len(data1) != len(data2):
            sys.exit(
                f"HDU count mismatch: {args.file1} has {len(data1)} data HDU(s), "
                f"{args.file2} has {len(data2)}"
            )

        out = fits.HDUList()
        for idx, (h1, h2) in enumerate(zip(data1, data2)):
            if h1.data is None and h2.data is None:
                out.append(
                    fits.PrimaryHDU(header=h1.header)
                    if idx == 0
                    else fits.ImageHDU(header=h1.header)
                )
                continue

            if h1.data is None or h2.data is None:
                sys.exit(f"HDU {idx}: one file has no data, the other does")

            if h1.data.shape != h2.data.shape:
                sys.exit(
                    f"HDU {idx}: shape mismatch {h1.data.shape} vs "
                    f"{h2.data.shape}"
                )

            diff = h1.data.astype(np.float64) - h2.data.astype(np.float64)
            cls = fits.PrimaryHDU if idx == 0 else fits.ImageHDU
            out.append(cls(data=diff, header=h1.header))

            imagetyp = h1.header.get(
                "IMAGETYP", h2.header.get("IMAGETYP", "<none>")
            )
            print(
                f"HDU {idx} IMAGETYP={imagetyp}: "
                f"min={diff.min():.6g} max={diff.max():.6g}"
            )

        out.writeto(args.output, overwrite=True)


if __name__ == "__main__":
    main()
