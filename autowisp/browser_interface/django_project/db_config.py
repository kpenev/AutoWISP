"""Resolve which database the browser interface should use.

By default this is an SQLite file in the user data directory, which is what
every existing installation has and what a fresh one gets with no
configuration at all.  Pointing it at a server instead means giving a URL,
spelled the way the project database spells its own so the two read alike::

    mysql+pymysql://user:password@host:3306/autowisp_bui
    mariadb+pymysql://user:password@host:3306/autowisp_bui

taken from ``$AUTOWISP_BUI_DB_URL`` or, failing that, a ``bui_db.url`` file
in the user data directory -- the same two-step the project database uses,
except global rather than per project, since the browser interface has no
project home to hang a URL off.

**One database per host.**  A server-hosted database is for keeping the
installation's state somewhere backed up and administered centrally, not
for sharing between hosts.  Sharing would need more than a URL: ``Project``
records a local filesystem path, so a project list written on one host is
only meaningful on another if the project directories are mounted
identically, and schema upgrades would stop being the single-process event
that lets the browser interface migrate itself on launch.
"""

from os import environ, path

from sqlalchemy.engine import make_url

#: Environment variable holding the URL, for deployments that would rather
#: not leave a password on disk.  Takes precedence over the file.
url_env_var = "AUTOWISP_BUI_DB_URL"

#: File in the user data directory holding the URL, when the environment
#: does not.  Named to match the project database's ``autowisp_db.url``.
url_fname = "bui_db.url"

#: The database used when nothing is configured.
sqlite_fname = "bui_db.sqlite3"

# Only the backends whose ``modified`` triggers are implemented, in
# core.timestamp_triggers.  Accepting a URL we cannot keep timestamps on
# would be worse than refusing it, since the failure would be silent.
_django_backends = {
    "sqlite": "django.db.backends.sqlite3",
    "mysql": "django.db.backends.mysql",
    "mariadb": "django.db.backends.mysql",
}


def get_database_url(data_dir):
    """
    Return the configured database URL, or ``None`` if there is none.

    Args:
        data_dir:    The user data directory to look for the URL file in.

    Returns:
        str or None:    The URL, stripped, or ``None`` to use the default
            SQLite database.
    """

    from_env = environ.get(url_env_var, "").strip()
    if from_env:
        return from_env

    url_path = path.join(str(data_dir), url_fname)
    if path.isfile(url_path):
        with open(url_path, "r", encoding="utf-8") as url_file:
            configured = url_file.read().strip()
        if configured:
            return configured

    return None


def _ensure_mysql_driver():
    """Make Django's MySQL backend work with whichever driver is installed.

    Django's backend imports ``MySQLdb``, supplied by ``mysqlclient``.
    AutoWISP does not depend on either driver -- the project database
    documents ``mysql+pymysql`` and leaves installing it to the user -- so
    accept ``pymysql`` as well rather than insisting on Django's default.

    The imports are deliberately not at module level: neither driver is a
    dependency, so importing one on the way in would make this module --
    and therefore ``settings`` -- unimportable on every SQLite install,
    which is all of them by default.
    """

    try:
        import MySQLdb  # noqa: F401  pylint: disable=unused-import,C0415

        return
    except ImportError:
        pass

    try:
        import pymysql  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise ImportError(
            "A MySQL/MariaDB browser-interface database needs a driver: "
            "install either 'mysqlclient' (Django's default) or 'pymysql'."
        ) from error

    pymysql.install_as_MySQLdb()


def get_databases(data_dir):
    """
    Return Django's ``DATABASES`` setting for the browser interface.

    Args:
        data_dir:    The user data directory, holding the default SQLite
            database and, optionally, the URL file naming another.

    Returns:
        dict:    Suitable for assigning to ``DATABASES``.

    Raises:
        ValueError:    If the configured URL names a backend whose
            ``modified`` trigger is not implemented.
    """

    url = get_database_url(data_dir)
    if url is None:
        return {
            "default": {
                "ENGINE": _django_backends["sqlite"],
                "NAME": path.join(str(data_dir), sqlite_fname),
            }
        }

    parsed = make_url(url)
    backend = parsed.get_backend_name()
    if backend not in _django_backends:
        raise ValueError(
            f"Unsupported browser-interface database backend {backend!r} "
            f"in {url_env_var} or {url_fname}. Supported: "
            + ", ".join(sorted(_django_backends))
            + "."
        )

    if backend == "sqlite":
        return {
            "default": {
                "ENGINE": _django_backends["sqlite"],
                # An SQLite URL carries the path as the database, and
                # query parameters (timeout, uri) that Django does not
                # take here; the path is all that transfers.
                "NAME": parsed.database,
            }
        }

    _ensure_mysql_driver()
    return {
        "default": {
            "ENGINE": _django_backends[backend],
            "NAME": parsed.database,
            "USER": parsed.username or "",
            "PASSWORD": parsed.password or "",
            "HOST": parsed.host or "",
            "PORT": str(parsed.port) if parsed.port else "",
        }
    }
