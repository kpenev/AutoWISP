"""Write ``autowisp/_version.py`` at build time (invoked by meson).

Records the version reported by
:func:`autowisp.miscellaneous.get_code_version_str` so that installed copies
of the package - which are not git repositories - can report the exact code
version they were built from without needing git at run time.

Usage (see ``autowisp/meson.build``)::

    python _write_version.py <source_root> <output_version_file>
"""

import sys
from pathlib import Path

# Make ``import autowisp`` resolve against the source tree being built.
sys.path.insert(0, sys.argv[1])

# pylint: disable=wrong-import-position
from autowisp.miscellaneous import get_code_version_str

# pylint: enable=wrong-import-position

Path(sys.argv[2]).write_text(
    '"""Auto-generated at build time; do not edit."""\n\n'
    f'code_version = "{get_code_version_str()}"\n',
    encoding="utf-8",
)
