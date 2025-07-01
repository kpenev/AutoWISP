"""Functions used by multiple bayesian sampling scripts."""

import os
import os.path
from datetime import datetime
import logging
import re
from glob import glob
import inspect
import sys

try:
    import git
except ImportError:
    pass


def get_code_version_str():
    """Return a string identifying the version of the code being used."""

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


default_config = {
    "task": "calculate",
    "fname_datetime_format": "%Y%m%d%H%M%S",
    "std_out_err_fname": "{task}_{now!s}_{pid:d}.outerr",
    "logging_fname": "{task}_{now!s}_{pid:d}.log",
    "logging_verbosity": "info",
    "logging_message_format": (
        "%(levelname)s %(asctime)s %(name)s: %(message)s | "
        "%(pathname)s.%(funcName)s:%(lineno)d"
    ),
}


def get_log_outerr_filenames(existing_pid=False, **config):
    """Return the filenames where `setup_process()` redirects log and output."""

    config.update(
        now=(
            "*"
            if existing_pid
            else datetime.now().strftime(config["fname_datetime_format"])
        ),
        pid=(existing_pid or os.getpid()),
    )

    if existing_pid == "*":
        pid_rex = re.compile(r"\{pid[^}]*\}")

        def prepare(format_str):
            return "*".join(pid_rex.split(format_str))

    else:

        def prepare(format_str):
            return format_str

    if config["std_out_err_fname"] is None:
        std_out_err_fname = None
    else:
        std_out_err_fname = prepare(config["std_out_err_fname"]).format_map(
            config
        )

    result = (
        prepare(config["logging_fname"]).format_map(config),
        std_out_err_fname,
    )

    if existing_pid:
        return tuple(sorted(glob(glob_str)) for glob_str in result)

    return result


def setup_process(**config):
    """
    Logging and I/O setup for the current processes.

    Args (keyword only):
        std_out_err_fname(str):    Format string for the standard output/error
            file name with substitutions including any keyword arguments passed
            to this function, ``now`` which gets replaced by current date/time,
            ``pid`` which gets replaced by the process ID, ``task`` which
            gets the value ``'calculate'`` by default but can be overwritten
            here.

        logging_fname(str):    Format string for the logging file name (see
            ``std_out_err_fname``).

        fname_datetime_format(str):    The format for the date and time string
            to be inserted in the file names.

        logging_message_format(str):    The format for the logging messages (see
            logging module documentation)

        logging_verbosity(str):    The verbosity of logging (see logging module
            documentation)

        All other keyword arguments are used to substitute into the format
            strings for the filenames.

    Returns:
        None
    """

    def ensure_directory(fname):
        """Make sure the directory containing the given name exists."""

        dirname = os.path.dirname(fname)
        if dirname and not os.path.exists(dirname):
            try:
                os.makedirs(dirname)
            except FileExistsError:
                if not os.path.isdir(dirname):
                    raise

    for param, value in default_config.items():
        if param not in config and (
            param != "logging_verbosity" or "verbose" not in config
        ):
            config[param] = value

    logging_fname, std_out_err_fname = get_log_outerr_filenames(**config)
    if std_out_err_fname is not None:
        ensure_directory(std_out_err_fname)
        sys.stdout = open(std_out_err_fname, "w", encoding="utf-8")
        sys.stderr = sys.stdout

    ensure_directory(logging_fname)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()
    logging_config = {
        "filename": logging_fname,
        "level": getattr(
            logging,
            config.get("logging_verbosity", config.get("verbose")).upper(),
        ),
        "format": config["logging_message_format"],
    }
    if config.get("logging_datetime_format") is not None:
        logging_config["datefmt"] = config["logging_datetime_format"]

    logging.basicConfig(**logging_config)


def setup_process_map(config):
    """Like `setup_process`, but more convenient for `multiprocessing.Pool`."""

    setup_process(**config)


if __name__ == "__main__":
    print(f"Code version: {get_code_version_str()}")
