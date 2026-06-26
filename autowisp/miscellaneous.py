"""Collection of small utils that is hard to classify."""

import os
import inspect

try:
    import git
except ImportError:
    pass

RECOGNIZED_HAT_ID_PREFIXES = ["HAT", "UCAC4"]


def get_hat_source_id_str(source_id):
    """Return the string representation of 3-integer HAT-id."""

    return RECOGNIZED_HAT_ID_PREFIXES[
        source_id[0]
    ] + "-{src[1]:03d}-{src[2]:07d}".format(src=source_id)


def get_code_version_str():
    """Return a string identifying the version of the code being used.

    Walks up from the *caller's* file to the enclosing git repository, so
    the returned hash (with a ``:dirty`` suffix when the working tree has
    uncommitted changes) identifies the entire working tree that produced
    the call.
    """

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
