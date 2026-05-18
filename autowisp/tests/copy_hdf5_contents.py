"""Copy selected datasets and attributes between matching HDF5 files in two directories.

Usage:
    python autowisp/tests/copy_hdf5_contents.py source_dir dest_dir spec_file [options]

The spec file lists items to copy, one per line:

    # comments and blank lines are ignored
    dataset /Group/Subgroup/DatasetName
    attribute /Group/Subgroup AttributeName
"""

import sys
import logging
from argparse import ArgumentParser
from pathlib import Path

import h5py


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spec file parsing
# ---------------------------------------------------------------------------

class CopySpec:
    """Parsed list of datasets and attributes to copy."""

    def __init__(self):
        self.datasets = []    # list of str HDF5 paths
        self.attributes = []  # list of (obj_path: str, attr_name: str)

    @classmethod
    def from_file(cls, spec_path):
        spec = cls()
        with open(spec_path, 'r', encoding='utf-8') as fobj:
            for lineno, raw in enumerate(fobj, 1):
                line = raw.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(None, 2)
                kind = parts[0].lower()
                if kind == 'dataset':
                    if len(parts) != 2:
                        raise ValueError(
                            f"{spec_path}:{lineno}: 'dataset' expects exactly one path"
                        )
                    spec.datasets.append(parts[1])
                elif kind == 'attribute':
                    if len(parts) != 3:
                        raise ValueError(
                            f"{spec_path}:{lineno}: 'attribute' expects object_path "
                            f"and attribute_name"
                        )
                    spec.attributes.append((parts[1], parts[2]))
                else:
                    raise ValueError(
                        f"{spec_path}:{lineno}: unknown keyword '{parts[0]}'; "
                        f"expected 'dataset' or 'attribute'"
                    )
        return spec


# ---------------------------------------------------------------------------
# Per-item copy helpers
# ---------------------------------------------------------------------------

def _copy_dataset(src, dst, dataset_path, on_conflict, missing_source, dry_run):
    label = f"dataset '{dataset_path}' in '{Path(dst.filename).name}'"
    if dataset_path not in src:
        if missing_source == 'error':
            raise KeyError(f"Source missing {label}")
        log.warning("Source missing %s — skipping", label)
        return
    if dataset_path in dst:
        if on_conflict == 'skip':
            log.info("Already exists %s — skipping", label)
            return
        if on_conflict == 'error':
            raise KeyError(f"Destination already has {label}")
        # overwrite: remove first so copy() can create it fresh
        log.info("Overwriting %s", label)
        if not dry_run:
            del dst[dataset_path]
    else:
        log.info("Copying %s", label)
    if not dry_run:
        # Ensure parent groups exist in destination
        parent = dataset_path.rsplit('/', 1)[0]
        if parent and parent not in dst:
            dst.require_group(parent)
        src.copy(dataset_path, dst, name=dataset_path)


def _copy_attribute(src, dst, obj_path, attr_name, on_conflict, missing_source, dry_run):
    label = f"attribute '{attr_name}' on '{obj_path}' in '{Path(dst.filename).name}'"
    if obj_path not in src or attr_name not in src[obj_path].attrs:
        if missing_source == 'error':
            raise KeyError(f"Source missing {label}")
        log.warning("Source missing %s — skipping", label)
        return
    if obj_path in dst and attr_name in dst[obj_path].attrs:
        if on_conflict == 'skip':
            log.info("Already exists %s — skipping", label)
            return
        if on_conflict == 'error':
            raise KeyError(f"Destination already has {label}")
        log.info("Overwriting %s", label)
    else:
        log.info("Copying %s", label)
    if not dry_run:
        dst.require_group(obj_path) if obj_path not in dst else None
        dst[obj_path].attrs[attr_name] = src[obj_path].attrs[attr_name]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        'source_dir',
        help='Directory containing the source HDF5 files.'
    )
    parser.add_argument(
        'dest_dir',
        help='Directory containing the destination HDF5 files (must already exist).'
    )
    parser.add_argument(
        'spec_file',
        help='Text file listing datasets and attributes to copy.'
    )
    parser.add_argument(
        '--on-conflict',
        choices=['skip', 'overwrite', 'error'],
        default='error',
        help='Action when an item already exists in the destination file. '
             'Default: %(default)s.'
    )
    parser.add_argument(
        '--missing-source',
        choices=['skip', 'error'],
        default='skip',
        help='Action when an item is absent from the source file. '
             'Default: %(default)s.'
    )
    parser.add_argument(
        '--pattern',
        default='*.h5',
        help='Glob pattern for HDF5 filenames. Default: %(default)s.'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be copied without writing anything.'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging.'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format='%(levelname)s: %(message)s'
    )

    spec = CopySpec.from_file(args.spec_file)
    source_dir = Path(args.source_dir)
    dest_dir = Path(args.dest_dir)

    src_files = sorted(source_dir.glob(args.pattern))
    if not src_files:
        log.warning("No files matching '%s' found in %s", args.pattern, source_dir)

    errors = []
    for src_path in src_files:
        dst_path = dest_dir / src_path.name
        if not dst_path.exists():
            log.warning("No matching destination for '%s' — skipping", src_path.name)
            continue
        if args.dry_run:
            print(f"[dry-run] Would process: {src_path.name}")
        try:
            with h5py.File(src_path, 'r') as src, \
                 h5py.File(dst_path, 'r' if args.dry_run else 'a') as dst:
                for dataset_path in spec.datasets:
                    _copy_dataset(
                        src, dst, dataset_path,
                        args.on_conflict, args.missing_source, args.dry_run
                    )
                for obj_path, attr_name in spec.attributes:
                    _copy_attribute(
                        src, dst, obj_path, attr_name,
                        args.on_conflict, args.missing_source, args.dry_run
                    )
        except (KeyError, OSError) as exc:
            errors.append(f"{src_path.name}: {exc}")

    if errors:
        for msg in errors:
            print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
