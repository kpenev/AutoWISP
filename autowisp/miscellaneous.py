"""Collection of small utils that is hard to classify."""

import os
import inspect

try:
    import git
except ImportError:
    git = None

RECOGNIZED_HAT_ID_PREFIXES = ["HAT", "UCAC4"]


def get_hat_source_id_str(source_id):
    """Return the string representation of 3-integer HAT-id."""

    return RECOGNIZED_HAT_ID_PREFIXES[
        source_id[0]
    ] + "-{src[1]:03d}-{src[2]:07d}".format(src=source_id)


def get_code_version_str():
    """Return a string identifying the version of the code being used.

    For an installed package - which is not a git repository - returns the
    version string recorded at build time in :mod:`autowisp._version` (see
    ``autowisp/_write_version.py``).

    When running from a git checkout with no build-generated version file,
    walks up from the *caller's* file to the enclosing git repository, so
    the returned hash (with a ``:dirty`` suffix when the working tree has
    uncommitted changes) identifies the entire working tree that produced
    the call. This requires GitPython, which is only a build dependency, so
    when it is not installed a placeholder string is returned instead.
    """

    try:
        from autowisp._version import code_version
    except ImportError:
        code_version = None
    if code_version:
        return code_version

    if git is None:
        return "Code version unavailable (GitPython not installed)."

    check_path = os.path.abspath(inspect.stack()[1].filename)
    repository = None
    while check_path != "/":
        check_path = os.path.dirname(check_path)
        try:
            repository = git.Repo(check_path)
            break
        except git.exc.InvalidGitRepositoryError:
            pass
    if repository is None:
        return "Caller not under git version control."
    head_sha = repository.commit().hexsha
    if repository.is_dirty():
        return head_sha + ":dirty"
    return head_sha


def collect_resource_snapshot():
    """Best-effort machine-memory snapshot (bytes), for diagnosing OOM.

    System memory pressure is the tell for an OOM / macOS-jetsam kill: a
    ``SIGKILL`` with no native traceback plus a nearly-full machine points
    at memory, not a crash. The dead worker's own peak RSS is gone by the
    time the parent looks, but the machine's RAM ceiling and the parent's
    RSS are strong signal. Cross-OS via ``psutil`` (a hard dependency).

    Never raises -- a failure yields a partial or empty dict rather than
    turning the recording of one error into a second error.

    Returns:
        dict:    Any of ``ram_total`` / ``ram_available`` (bytes),
            ``ram_percent_used`` (percent), ``process_rss`` (bytes).
    """

    snapshot = {}
    try:
        # Local import: keep this leaf module importable even if the
        # optional runtime were ever stripped down.
        import psutil  # pylint: disable=import-outside-toplevel
    except Exception:  # pylint: disable=broad-except
        return snapshot
    try:
        virtual_memory = psutil.virtual_memory()
        snapshot["ram_total"] = int(virtual_memory.total)
        snapshot["ram_available"] = int(virtual_memory.available)
        snapshot["ram_percent_used"] = float(virtual_memory.percent)
    except Exception:  # pylint: disable=broad-except
        pass
    try:
        snapshot["process_rss"] = int(psutil.Process().memory_info().rss)
    except Exception:  # pylint: disable=broad-except
        pass
    return snapshot
