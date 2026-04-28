#!/usr/bin/env python3

"""Print FITS header keywords present in every file given on the command line."""

import sys
from astropy.io import fits


def get_keywords(fname):
    """Return a dict of keyword -> value for all HDUs in a FITS file."""
    result = {}
    with fits.open(fname, memmap=False) as hdul:
        for hdu in hdul:
            for k, v in hdu.header.items():
                if k:
                    result[k] = v
    return result


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} FILE [FILE ...]", file=sys.stderr)
        sys.exit(1)

    fnames = sys.argv[1:]
    per_file = []
    for fname in fnames:
        try:
            per_file.append(get_keywords(fname))
        except Exception as exc:
            print(f"{fname}: {exc}", file=sys.stderr)
            sys.exit(1)

    common = set(per_file[0])
    for kws in per_file[1:]:
        common &= set(kws)

    n = len(per_file)
    sample_indices = sorted({0, n // 2, n - 1})

    for keyword in sorted(common):
        values = [kws[keyword] for kws in per_file]
        if all(v == values[0] for v in values[1:]):
            print(f"{keyword} = {values[0]}")
            continue

        counts = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1

        print(f"{keyword}:")
        if len(counts) <= 5:
            for v, c in sorted(counts.items(), key=lambda x: (-x[1], str(x[0]))):
                print(f"    {v!r} ({c} image{'s' if c != 1 else ''})")
        else:
            for i in sample_indices:
                print(f"    [{fnames[i]}] {values[i]!r}")


if __name__ == "__main__":
    main()
