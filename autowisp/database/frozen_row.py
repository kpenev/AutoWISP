"""A dependency-free, picklable snapshot of a database row.

This module intentionally imports nothing from SQLAlchemy (or anything
else heavy), so that :class:`FrozenRow` can be referenced from the
exception hierarchy without dragging the ORM into it. The companion
:func:`autowisp.database.interface.snapshot_row` -- which *does* need
SQLAlchemy -- builds a :class:`FrozenRow` from a live ORM instance.
"""

from dataclasses import dataclass

git_id = "$Id$"


@dataclass(frozen=True)
class FrozenRow:
    """Immutable, picklable snapshot of an ORM row's column values.

    Holds the column values of a SQLAlchemy row detached from any
    session, so it stays usable after the session that produced it is
    closed and survives pickling across process/host boundaries (where a
    live ORM instance would not). Column values are reached by attribute
    (``snapshot.host``) or via :attr:`columns`.

    Built from a live instance with
    :func:`autowisp.database.interface.snapshot_row` (parent side, where
    the row exists) or directly from a dict (a worker, which only has
    primitives threaded through its config).

    This is a *general* utility -- not specific to errors. Any place that
    needs a durable, picklable copy of a row (related-artifact context,
    provenance, caching a row past its session) can use it.

    Attributes:
        table(str):    Name of the source table, for display/debugging.

        columns(dict):    ``{column_key: value}`` for every mapped
            column captured. Treated as read-only.
    """

    table: str
    columns: dict

    def __getattr__(self, name):
        # Only reached when normal attribute lookup fails. Guard the real
        # dataclass fields explicitly so that an access during unpickling
        # (before ``__dict__`` is populated) cannot recurse forever.
        if name in ("table", "columns"):
            raise AttributeError(name)
        try:
            return self.columns[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
