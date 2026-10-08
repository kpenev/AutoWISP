"""List every HDF5 dataset or attribute that differs between two directories.

Usage: python compare_h5.py <generated dir> <expected dir> [path regex ...]

Paths matching any of the given regexes are left out (e.g. the step's own
outputs, to confirm nothing else differs). Paths are printed with the
aperture/iteration numbers collapsed, so the summary stays short.
"""

import re
import sys
from collections import Counter
from glob import glob
from os import path

import h5py
import numpy


def same(value1, value2):
    """True iff the two values are identical (NaN equal to NaN)."""

    value1, value2 = numpy.asarray(value1), numpy.asarray(value2)
    if value1.shape != value2.shape:
        return False
    if value1.dtype.kind == "O":
        return all(
            same(entry1, entry2)
            for entry1, entry2 in zip(value1.ravel(), value2.ravel())
        )
    if value1.dtype.kind == "f":
        return numpy.array_equal(value1, value2, equal_nan=True)
    return numpy.array_equal(value1, value2)


def collect(h5_file):
    """Return {path: object} for every group and dataset in the file."""

    result = {"/": h5_file}
    h5_file.visititems(lambda name, obj: result.__setitem__("/" + name, obj))
    return result


def compare_files(generated_fname, expected_fname, ignore, differences):
    """Add the differences between two HDF5 files to ``differences``."""

    with (
        h5py.File(generated_fname, "r") as generated,
        h5py.File(expected_fname, "r") as expected,
    ):
        generated_objects = collect(generated)
        expected_objects = collect(expected)
        for name in set(generated_objects) | set(expected_objects):
            if any(rex.search(name) for rex in ignore):
                continue
            short = re.sub(r"\d{3}", "NNN", name)
            if name not in expected_objects:
                differences["only generated: " + short] += 1
                continue
            if name not in generated_objects:
                differences["only expected: " + short] += 1
                continue
            gen_obj, exp_obj = generated_objects[name], expected_objects[name]
            if isinstance(gen_obj, h5py.Dataset) and not same(
                gen_obj[()], exp_obj[()]
            ):
                differences["data: " + short] += 1
            for attr in set(gen_obj.attrs) | set(exp_obj.attrs):
                if attr not in gen_obj.attrs or attr not in exp_obj.attrs:
                    differences[f"attr missing: {short}@{attr}"] += 1
                elif not same(gen_obj.attrs[attr], exp_obj.attrs[attr]):
                    differences[f"attr: {short}@{attr}"] += 1


def main():
    """Print a summary of the differences."""

    generated_dir, expected_dir = sys.argv[1:3]
    ignore = [re.compile(rex) for rex in sys.argv[3:]]
    differences = Counter()
    generated_fnames = sorted(glob(path.join(generated_dir, "*.h5")))
    for generated_fname in generated_fnames:
        compare_files(
            generated_fname,
            path.join(expected_dir, path.basename(generated_fname)),
            ignore,
            differences,
        )
    print(
        f"{len(generated_fnames)} files compared; {len(differences)} kinds "
        "of difference"
    )
    for what, count in sorted(differences.items()):
        print(f"  {count:5d}  {what}")


if __name__ == "__main__":
    main()
