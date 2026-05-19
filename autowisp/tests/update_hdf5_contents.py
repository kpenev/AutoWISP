"""Bring HDF5 files in one directory closer to matching files in another.

Usage:
    python update_hdf5_contents.py source_dir dest_dir spec_file [options]

The spec file lists items to apply, one per line. Each line starts with ``+``
to copy from source to destination or ``-`` to delete from destination:

    # comments and blank lines are ignored
    + dataset /Group/Subgroup/DatasetName
    - dataset /Group/StaleDataset
    + attribute /Group/Subgroup AttributeName
    - attribute /Group/Subgroup StaleAttribute

The ``-`` form only touches the destination; ``source_dir`` is not consulted.
Operations are applied in spec-file order, so a ``-`` followed by a ``+`` for
the same path replaces it cleanly even with ``--on-conflict=error``.

Path and attribute-name tokens are interpreted as Python ``re`` regular
expressions, anchored end-to-end (i.e. matched with :func:`re.fullmatch`).
For ``+`` operations the regexes are resolved against the source file; for
``-`` operations they are resolved against the destination file. A line that
matches zero items is treated like a missing source / destination (governed
by ``--missing-source`` / ``--missing-dest``). Note that HDF5 paths starting
with a literal ``.`` will be matched by the regex metacharacter ``.``; escape
special characters with ``\\`` if a strict literal match is required.
"""

import re
import sys
import logging
from argparse import ArgumentParser
from pathlib import Path

import h5py


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spec file parsing
# ---------------------------------------------------------------------------


class UpdateSpec:  # pylint: disable=too-few-public-methods
    """Parsed ordered list of HDF5 operations (copy and/or delete)."""

    def __init__(self):
        """Create an empty spec with no operations."""

        # Each entry is one of:
        #   ('+', 'dataset',   dataset_path)
        #   ('-', 'dataset',   dataset_path)
        #   ('+', 'attribute', obj_path, attr_name)
        #   ('-', 'attribute', obj_path, attr_name)
        self.operations = []

    @classmethod
    def from_file(cls, spec_path):
        """Parse ``spec_path`` and return the populated :class:`UpdateSpec`.

        Each non-blank, non-comment line must be either
        ``<action> dataset <hdf5_path>`` or
        ``<action> attribute <hdf5_path> <attr_name>``, where ``<action>`` is
        ``+`` (copy from source) or ``-`` (delete from destination). Raises
        :class:`ValueError` on malformed lines.
        """

        spec = cls()
        with open(spec_path, "r", encoding="utf-8") as fobj:
            for lineno, raw in enumerate(fobj, 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(None, 3)
                action = parts[0]
                if action not in ("+", "-"):
                    raise ValueError(
                        f"{spec_path}:{lineno}: line must start with '+' "
                        f"(copy) or '-' (delete); got '{parts[0]}'"
                    )
                if len(parts) < 3:
                    raise ValueError(
                        f"{spec_path}:{lineno}: expected '<+/-> <kind> <path> "
                        f"[<attr_name>]'"
                    )
                kind = parts[1].lower()
                if kind == "dataset":
                    if len(parts) != 3:
                        raise ValueError(
                            f"{spec_path}:{lineno}: 'dataset' expects exactly "
                            "one path"
                        )
                    spec.operations.append((action, "dataset", parts[2]))
                elif kind == "attribute":
                    if len(parts) != 4:
                        raise ValueError(
                            f"{spec_path}:{lineno}: 'attribute' expects "
                            "object_path and attribute_name"
                        )
                    spec.operations.append(
                        (action, "attribute", parts[2], parts[3])
                    )
                else:
                    raise ValueError(
                        f"{spec_path}:{lineno}: unknown kind '{parts[1]}'; "
                        f"expected 'dataset' or 'attribute'"
                    )
        return spec


# ---------------------------------------------------------------------------
# Regex resolution against an HDF5 file
# ---------------------------------------------------------------------------


def _all_dataset_paths(h5_root):
    """Return every dataset path in ``h5_root`` (absolute, ``/...``)."""

    paths = []

    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            paths.append("/" + name)

    h5_root.visititems(_visit)
    return paths


def _all_object_paths(h5_root):
    """Return every object path in ``h5_root`` (groups + datasets) and ``/``."""

    paths = ["/"]

    def _visit(name, _obj):
        paths.append("/" + name)

    h5_root.visititems(_visit)
    return paths


def _resolve_dataset_paths(h5_root, pattern):
    """Return ``(dataset_path,)`` tuples in ``h5_root`` matching the regex.

    Tuples (rather than bare strings) are returned so callers can unpack the
    result uniformly with the ``(obj_path, attr_name)`` pairs returned by
    :func:`_resolve_attribute_targets`. Matching uses :func:`re.fullmatch`.
    """

    compiled = re.compile(pattern)
    return [(p,) for p in _all_dataset_paths(h5_root) if compiled.fullmatch(p)]


def _resolve_attribute_targets(h5_root, obj_pattern, attr_pattern):
    """Return ``(obj_path, attr_name)`` pairs in ``h5_root`` matching regexes.

    Both patterns are matched with :func:`re.fullmatch`.
    """

    obj_re = re.compile(obj_pattern)
    attr_re = re.compile(attr_pattern)
    targets = []
    for obj_path in _all_object_paths(h5_root):
        if not obj_re.fullmatch(obj_path):
            continue
        for attr_name in h5_root[obj_path].attrs.keys():
            if attr_re.fullmatch(attr_name):
                targets.append((obj_path, attr_name))
    return targets


# ---------------------------------------------------------------------------
# Per-item copy / delete helpers
# ---------------------------------------------------------------------------


def _copy_dataset(  # pylint: disable=too-many-arguments
    src,
    dst,
    dataset_path,
    *,
    on_conflict="error",
    missing_source="error",
    dry_run=False,
):
    """Copy a single dataset from ``src`` to ``dst``.

    ``on_conflict`` (``'skip'``/``'overwrite'``/``'error'``) controls behavior
    when ``dataset_path`` already exists in ``dst``; ``missing_source``
    (``'skip'``/``'error'``) controls behavior when it is absent from ``src``.
    If ``dry_run`` is true, log the intended action without modifying ``dst``.
    Missing parent groups in the destination are created on demand.
    """

    label = f"dataset '{dataset_path}' in '{Path(dst.filename).name}'"
    if dataset_path not in src:
        if missing_source == "error":
            raise KeyError(f"Source missing {label}")
        log.warning("Source missing %s — skipping", label)
        return
    if dataset_path in dst:
        if on_conflict == "skip":
            log.info("Already exists %s — skipping", label)
            return
        if on_conflict == "error":
            raise KeyError(f"Destination already has {label}")
        # overwrite: remove first so copy() can create it fresh
        log.info("Overwriting %s", label)
        if not dry_run:
            del dst[dataset_path]
    else:
        log.info("Copying %s", label)
    if not dry_run:
        # Ensure parent groups exist in destination
        parent = dataset_path.rsplit("/", 1)[0]
        if parent and parent not in dst:
            dst.require_group(parent)
        src.copy(dataset_path, dst, name=dataset_path)


def _copy_attribute(  # pylint: disable=too-many-arguments
    src,
    dst,
    obj_path,
    attr_name,
    *,
    on_conflict="error",
    missing_source="error",
    dry_run=False,
):
    """Copy an attribute ``attr_name`` on ``obj_path`` from ``src`` to ``dst``.

    Semantics for ``on_conflict``, ``missing_source``, and ``dry_run`` match
    :func:`_copy_dataset`. The destination object is created as a group if it
    does not yet exist.
    """

    label = (
        f"attribute '{attr_name}' on '{obj_path}' in "
        f"'{Path(dst.filename).name}'"
    )
    if obj_path not in src or attr_name not in src[obj_path].attrs:
        if missing_source == "error":
            raise KeyError(f"Source missing {label}")
        log.warning("Source missing %s — skipping", label)
        return
    if obj_path in dst and attr_name in dst[obj_path].attrs:
        if on_conflict == "skip":
            log.info("Already exists %s — skipping", label)
            return
        if on_conflict == "error":
            raise KeyError(f"Destination already has {label}")
        log.info("Overwriting %s", label)
    else:
        log.info("Copying %s", label)
    if not dry_run:
        if obj_path not in dst:
            dst.require_group(obj_path)
        dst[obj_path].attrs[attr_name] = src[obj_path].attrs[attr_name]


def _delete_dataset(dst, dataset_path, *, missing_dest="skip", dry_run=False):
    """Delete ``dataset_path`` from ``dst``.

    ``missing_dest`` (``'skip'``/``'error'``) controls behavior when the
    dataset is absent. If ``dry_run`` is true, log the intended action without
    modifying ``dst``.
    """

    label = f"dataset '{dataset_path}' in '{Path(dst.filename).name}'"
    if dataset_path not in dst:
        if missing_dest == "error":
            raise KeyError(f"Destination missing {label}")
        log.warning("Destination missing %s — nothing to delete", label)
        return
    log.info("Deleting %s", label)
    if not dry_run:
        del dst[dataset_path]


def _delete_attribute(
    dst, obj_path, attr_name, *, missing_dest="skip", dry_run=False
):
    """Delete attribute ``attr_name`` on ``obj_path`` in ``dst``.

    Semantics for ``missing_dest`` and ``dry_run`` match
    :func:`_delete_dataset`.
    """

    label = (
        f"attribute '{attr_name}' on '{obj_path}' in "
        f"'{Path(dst.filename).name}'"
    )
    if obj_path not in dst or attr_name not in dst[obj_path].attrs:
        if missing_dest == "error":
            raise KeyError(f"Destination missing {label}")
        log.warning("Destination missing %s — nothing to delete", label)
        return
    log.info("Deleting %s", label)
    if not dry_run:
        del dst[obj_path].attrs[attr_name]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    """Return the parsed command-line arguments for the update tool."""

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_dir",
        help="Directory containing the source HDF5 files. Only consulted for "
        "'+' (copy) operations.",
    )
    parser.add_argument(
        "dest_dir",
        help="Directory containing the destination HDF5 files (must already "
        "exist).",
    )
    parser.add_argument(
        "spec_file",
        help="Text file listing '+' (copy) and '-' (delete) operations.",
    )
    parser.add_argument(
        "--on-conflict",
        choices=["skip", "overwrite", "error"],
        default="error",
        help="Action when an item already exists in the destination file for "
        "a '+' (copy) operation. Default: %(default)s.",
    )
    parser.add_argument(
        "--missing-source",
        choices=["skip", "error"],
        default="error",
        help="Action when an item is absent from the source file for a '+' "
        "(copy) operation. Default: %(default)s.",
    )
    parser.add_argument(
        "--missing-dest",
        choices=["skip", "error"],
        default="error",
        help="Action when an item to delete is absent from the destination "
        "file for a '-' (delete) operation. Default: %(default)s.",
    )
    parser.add_argument(
        "--pattern",
        default="*.h5",
        help="Glob pattern for HDF5 filenames. Default: %(default)s.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without writing anything.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging."
    )
    return parser.parse_args()


def _report_empty_match(label, missing_policy):
    """Raise or warn that a regex matched zero items, per ``missing_policy``."""

    if missing_policy == "error":
        raise KeyError(f"No {label}")
    log.warning("No %s — skipping", label)


def _apply_operations(spec, src, dst, args):
    """Apply every operation in ``spec`` to the (``src``, ``dst``) pair.

    The path / attribute-name tokens in each op are treated as Python regular
    expressions and resolved against ``src`` for ``+`` ops and ``dst`` for
    ``-`` ops. If a regex matches zero items, ``--missing-source`` /
    ``--missing-dest`` controls whether that is an error or a warning.
    """

    for op in spec.operations:
        action, kind = op[0], op[1]
        resolver = (
            _resolve_dataset_paths
            if kind == "dataset"
            else _resolve_attribute_targets
        )
        targets = resolver(src if action == "+" else dst, *op[2:])
        if not targets:
            _report_empty_match(
                f"{'source' if action == '+' else 'destination'} "
                f"{kind}s matching r'{op[2]}'"
                + (f" / r'{op[3]}'" if kind == "attribute" else "")
                + "in "
                + (
                    Path(src.filename).name
                    if action == "+"
                    else Path(dst.filename).name
                ),
                args.missing_source if action == "+" else args.missing_dest,
            )
            continue
        if action == "+":
            copy = _copy_dataset if kind == "dataset" else _copy_attribute
            for tgt in targets:
                copy(
                    src,
                    dst,
                    *tgt,
                    on_conflict=args.on_conflict,
                    missing_source=args.missing_source,
                    dry_run=args.dry_run,
                )
        else:  # action == "-"
            delete = _delete_dataset if kind == "dataset" else _delete_attribute
            for tgt in targets:
                delete(
                    dst,
                    *tgt,
                    missing_dest=args.missing_dest,
                    dry_run=args.dry_run,
                )

def main():
    """Entry point: apply the spec-listed operations to matching HDF5 files.

    Pairs files in ``source_dir`` with same-named files in ``dest_dir`` (via
    ``--pattern``), then applies every operation in the spec -- in file order
    -- to each pair. ``+`` lines copy from source to destination; ``-`` lines
    delete from destination only. Errors are collected and reported at the
    end; the process exits non-zero if any pair failed.
    """

    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    spec = UpdateSpec.from_file(args.spec_file)
    source_dir = Path(args.source_dir)
    dest_dir = Path(args.dest_dir)

    src_files = sorted(source_dir.glob(args.pattern))
    if not src_files:
        log.warning(
            "No files matching '%s' found in %s", args.pattern, source_dir
        )

    errors = []
    for src_path in src_files:
        dst_path = dest_dir / src_path.name
        if not dst_path.exists():
            log.warning(
                "No matching destination for '%s' — skipping", src_path.name
            )
            continue
        if args.dry_run:
            print(f"[dry-run] Would process: {src_path.name}")
        try:
            with h5py.File(src_path, "r") as src, h5py.File(
                dst_path, "r" if args.dry_run else "a"
            ) as dst:
                _apply_operations(spec, src, dst, args)
        except (KeyError, OSError) as exc:
            errors.append(f"{src_path.name}: {exc}")

    if errors:
        for msg in errors:
            print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
