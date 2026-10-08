"""Shared base class for the browser interface's database models."""

from django.db import models


# A model base contributes fields rather than behaviour.
# pylint: disable=too-few-public-methods
class BuiModelBase(models.Model):
    """
    Base for browser-interface models, recording when each row changed.

    Every model in the browser-interface database should derive from this,
    so that a row can always be placed in time when something needs
    unpicking after the fact.  It is the counterpart of the project
    database's :class:`autowisp.database.data_model.base.DataModelSubBase`,
    which carries the same information in a column named ``timestamp``.

    ``auto_now`` alone would not be enough to keep ``modified`` honest.  It
    fires on ``Model.save()`` and ``bulk_create()``, but not on
    ``QuerySet.update()``, ``bulk_update()`` or raw SQL -- and
    ``bulk_update`` is worse than a simple omission, since it reads the
    in-memory value rather than calling ``pre_save`` and so writes back
    whatever was loaded, preserving a stale timestamp.  A database trigger
    covers what ``auto_now`` cannot; see
    :mod:`autowisp.browser_interface.core.timestamp_triggers`.  Both are
    kept: the trigger is the guarantee, ``auto_now`` is what leaves the
    in-memory object correct after a ``save()`` without re-reading it.
    """

    created = models.DateTimeField(
        auto_now_add=True,
        help_text="When the record was created",
    )
    modified = models.DateTimeField(
        auto_now=True,
        help_text="When the record was last changed",
    )

    class Meta:
        """Contribute the fields to subclasses without creating a table."""

        abstract = True


# pylint: enable=too-few-public-methods
