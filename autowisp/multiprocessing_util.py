"""Functions used by multiple bayesian sampling scripts."""

import os
import os.path
from datetime import datetime
import logging
import re
from glob import glob
import inspect
import sys
import multiprocessing

import platformdirs

from autowisp.database.interface import set_sqlite_database
from autowisp.data_reduction.data_reduction_file import DataReductionFile

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

    if config.get("parent_pid"):
        result = tuple(
            os.path.join(
                os.path.dirname(fname),
                str(config["parent_pid"]),
                os.path.basename(fname),
            )
            for fname in result
        )

    if existing_pid:
        return tuple(sorted(glob(glob_str)) for glob_str in result)

    return result


def setup_process_map(db_fname, config):
    """
    Logging and I/O setup for the current processes.

    KWArgs:
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
        with open('C:\\WISP\\out\\file_multi.txt', mode='a', encoding='utf-8') as f:
            f.write(f'Ensuring directory for {fname}\n')
        dirname = os.path.dirname(fname)
        if dirname and not os.path.exists(dirname):
            try:
                os.makedirs(dirname)
                with open('C:\\WISP\\out\\file_multi.txt', mode='a', encoding='utf-8') as f:
                    f.write(f'Created directory {dirname}\n')
            except FileExistsError:
                if not os.path.isdir(dirname):
                    raise
    with open('C:\\WISP\\out\\file_multi.txt', mode='a', encoding='utf-8') as f:
        f.write(f'Setting up process map with DB {db_fname} and config {config}\n')
    for param, value in default_config.items():
        if param not in config and (
            param != "logging_verbosity" or "verbose" not in config
        ):
            config[param] = value

    with open(
        os.path.join(
            platformdirs.user_data_dir("autowisp"), "setup_process.outerr"
        ),
        "a",
        encoding="utf-8",
    ) as info_file:
        info_file.write(
            f"Setting up process with DB {db_fname} and configuration:\n\t"
            + "\n\t".join(
                f"{key!r}: {value!r}" for key, value in config.items()
            )
            +"\n"
        )
        # if os.name == "nt":
            # with open('C:\\WISP\\out\\file_multi.txt', mode='a', encoding='utf-8') as f:
            #     f.write('Setting multiprocessing executable to pythonw.exe\n')
            # multiprocessing.set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))
        logging_fname, std_out_err_fname = get_log_outerr_filenames(**config)
        info_file.write(
            f"Logging to {logging_fname!r}, "
            f"stdout/stderr to {std_out_err_fname!r}\n"
        )

    if std_out_err_fname is not None:
        with open('C:\\WISP\\out\\file_multi.txt', mode='a', encoding='utf-8') as f:
            f.write(f'Redirecting output to {std_out_err_fname}\n')
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout.close()
        sys.stderr.close()
        ensure_directory(std_out_err_fname)
        sys.stdout = open(  # pylint: disable=consider-using-with
            std_out_err_fname, "w", encoding="utf-8", buffering=1
        )
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
        "force": True,
    }
    if config.get("logging_datetime_format") is not None:
        logging_config["datefmt"] = config["logging_datetime_format"]

    logging.basicConfig(**logging_config)

    logging.info("Starting process with configuration: %s", repr(config))

    set_sqlite_database(db_fname)
    if "data_reduction_fname" in config:
        DataReductionFile.fname_template = config["data_reduction_fname"]


def setup_process(db_fname, **config):
    """Like `setup_process`, but more convenient for `multiprocessing.Pool`."""

    setup_process_map(db_fname, config)


if __name__ == "__main__":
    print(f"Code version: {get_code_version_str()}")
