"""Clone the database of a project being added, as the interface asked.

Run in the background by the browser interface, as ``python -m
autowisp.clone_existing_project <record>``, with the file recording the
addition (see :mod:`autowisp.database.pending_projects`), which is where it
reports how the clone went. Its output, a failure's traceback included, goes
to the log the interface launched it with.
"""

import logging
import sys

from autowisp.database.pending_projects import run_clone

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(asctime)s %(name)s: %(message)s",
    )
    run_clone(sys.argv[1])
