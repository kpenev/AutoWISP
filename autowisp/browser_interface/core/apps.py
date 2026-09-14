"""Application configuration for the shared browser-interface app."""

from django.apps import AppConfig
from django.db.models.signals import post_migrate, pre_migrate

from core.timestamp_triggers import (
    drop_modified_triggers,
    install_modified_triggers,
)


class CoreConfig(AppConfig):
    """Machinery shared by every browser-interface app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "core"

    def ready(self):
        """Arrange for the modified-timestamp triggers to be maintained.

        Dropped before migrations and recreated after, because SQLite
        refuses to alter a column a trigger names.
        """

        pre_migrate.connect(
            drop_modified_triggers,
            dispatch_uid="core.drop_modified_triggers",
        )
        post_migrate.connect(
            install_modified_triggers,
            dispatch_uid="core.install_modified_triggers",
        )
