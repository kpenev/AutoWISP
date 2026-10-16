"""Open sessions on a database, each within a transaction.

Imports only SQLAlchemy, so that every database module can use it:
:mod:`autowisp.database.interface` imports :mod:`autowisp.database.migrate`,
which imports :mod:`autowisp.database.experiments`, so a helper in either of
the first two could not be used by the last.
"""

from contextlib import contextmanager

from sqlalchemy.orm import Session


@contextmanager
def start_engine_session(engine, expire_on_commit=True):
    """
    Yield a session on the engine's database, within a transaction.

    The transaction is committed on leaving the block, and rolled back if it
    raises.

    Args:
        engine:    The engine of the database.

        expire_on_commit(bool):    Whether the objects loaded in the session
            expire with the commit, so that reading one after the block
            raises. Only a caller handing them out of the block should pass
            False.

    Yields:
        Session:    The session.
    """

    with (
        Session(engine, expire_on_commit=expire_on_commit) as db_session,
        db_session.begin(),
    ):
        yield db_session
