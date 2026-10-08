#!/usr/bin/env python3

"""Compare two AutoWISP configuration trees exported by the browser UI.

The exports carry ``id``, ``parentId``, ``level`` and ``relationship``
fields that only encode where a node landed in that particular dump, so
two configurations that differ merely in the order their parameters were
written out produce a completely useless ``diff``. This compares what the
nodes mean instead: parameters are matched by name at each level of the
tree and only their values (and, optionally, their descriptions) are
reported.
"""

import argparse
import json
import sys

# Fields describing the position of a node in the dump rather than its
# content.
_IGNORED_FIELDS = ("id", "parentId", "level", "relationship")


def parse_command_line(args=None):
    """Return the parsed command line arguments."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("first", help="The first configuration to compare.")
    parser.add_argument("second", help="The second configuration to compare.")
    parser.add_argument(
        "--descriptions",
        action="store_true",
        help="Also report parameters whose description differs. Those come "
        "from the command line help of the pipeline, so they flag "
        "configurations exported by different AutoWISP versions rather "
        "than differently configured pipelines.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=100,
        help="Truncate reported values/descriptions to this many characters "
        "(0 disables truncation). Default: %(default)s",
    )
    return parser.parse_args(args)


def flatten(node, path=()):
    """Map the path of each node to its values and description.

    Args:
        node:    The node to flatten, as read from the exported JSON.

        path:    The names of the ancestors of `node`, most distant first.

    Returns:
        dict:    Keys are ``(<ancestor name>, ..., <node name>)`` tuples,
            values are ``(<description>, <list of value children>)``. Nodes
            of type ``value`` are folded into their parent instead of
            getting an entry of their own.
    """

    result = {}
    values = []
    for child in node.get("children", []):
        if child["type"] == "value":
            values.append(child["name"])
        else:
            result.update(flatten(child, path + (node["name"],)))
    result[path + (node["name"],)] = (node.get("description", ""), values)
    return result


def _format(value, width):
    """Return `value` as a string, truncated to `width` if requested."""

    if isinstance(value, list):
        text = ", ".join(
            "<unset>" if entry is None else str(entry) for entry in value
        )
    else:
        text = str(value)
    if 0 < width < len(text):
        # Truncated mid-word rather than with textwrap.shorten(), since
        # values like filename patterns are a single very long "word"
        # which shorten() would replace entirely by its placeholder.
        return text[: width - 4] + " ..."
    return text


def report(first, second, names, args):
    """Print how the two flattened configurations differ."""

    only_first = sorted(set(first) - set(second))
    only_second = sorted(set(second) - set(first))
    shared = sorted(set(first) & set(second))
    changed = [key for key in shared if first[key][1] != second[key][1]]
    described = [key for key in shared if first[key][0] != second[key][0]]

    for label, keys, source in [
        (f"ONLY IN {names[0]}", only_first, first),
        (f"ONLY IN {names[1]}", only_second, second),
    ]:
        print(f"\n{label} ({len(keys)})")
        for key in keys:
            value = _format(source[key][1], args.width)
            print(f"    {'.'.join(key[1:])} = {value}")

    print(f"\nDIFFERENT VALUES ({len(changed)})")
    for key in changed:
        print(f"    {'.'.join(key[1:])}")
        for name, source in zip(names, (first, second)):
            print(f"        {name}: {_format(source[key][1], args.width)}")

    if args.descriptions:
        print(f"\nDIFFERENT DESCRIPTIONS ({len(described)})")
        for key in described:
            print(f"    {'.'.join(key[1:])}")
            for name, source in zip(names, (first, second)):
                print(f"        {name}: {_format(source[key][0], args.width)}")
    else:
        print(
            f"\n{len(described)} parameter(s) differ only in their description"
            " (pass --descriptions to list them)."
        )

    return bool(only_first or only_second or changed)


def main(args):
    """Avoid polluting the global namespace."""

    configurations = []
    for fname in (args.first, args.second):
        with open(fname, encoding="utf-8") as config_file:
            configurations.append(flatten(json.load(config_file)))

    names = []
    for fname in (args.first, args.second):
        name = fname.rsplit("/", 1)[-1]
        names.append(name[:-5] if name.endswith(".json") else name)

    return report(configurations[0], configurations[1], names, args)


if __name__ == "__main__":
    sys.exit(1 if main(parse_command_line()) else 0)
