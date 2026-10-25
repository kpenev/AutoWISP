"""Run the image processing pipeline in a detached mode."""

import logging
import os
import sys
from contextlib import contextmanager
from traceback import format_exc

from configargparse import ArgumentParser, DefaultsFormatter, SUPPRESS
from sqlalchemy import sql, update
import platformdirs

from autowisp.database.interface import (
    get_db_engine,
    start_db_session,
    set_project_home,
    get_project_home,
)
from autowisp.database.migrate import locked_connection
from autowisp.database.data_model import (  # pylint: disable=no-name-in-module
    PipelineRun,
)
from autowisp.database.image_processing import ImageProcessingManager
from autowisp.database.lightcurve_processing import LightCurveProcessingManager
from autowisp.database.pipeline_runs import find_earlier_run, record_run
from autowisp.database.user_interface import get_processing_sequence
from autowisp.error_context import get_error_context
from autowisp.error_cli import report_error
from autowisp.project_experiments import ProjectExperiments
from autowisp.exceptions import AutoWISPError, PipelineError, ResourceError
from autowisp.file_utilities import find_fits_fnames


def parse_command_line():
    """Return the command line configuration."""

    parser = ArgumentParser(
        description="Manually invoke the fully automated processing",
        default_config_files=[],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False,
    )
    parser.add_argument(
        "project_home", help="Path to the project home directory."
    )
    parser.add_argument(
        "--add-raw-images",
        "-i",
        nargs="+",
        default=[],
        help="Before processing add new raw images for processing. Can be "
        "specified as a combination of image files and directories which will"
        "be searched for FITS files.",
    )
    parser.add_argument(
        "--steps",
        nargs="*",
        default=None,
        help="Process only these steps. Given without any, only the raw "
        "images are added. Leave out to process every step.",
    )
    parser.add_argument(
        "--step-imtypes",
        nargs="+",
        default=[],
        help="Process the steps named here only for these image types, given "
        "as step:imagetype pairs (e.g. calibrate:flat calibrate:object). "
        "Steps not named here are processed for every image type.",
    )
    parser.add_argument(
        "--detached",
        action="store_true",
        help=SUPPRESS,  # Only used internally to detach in windows
    )
    logging.info("Parsed arguments: %s", parser.parse_args())
    return parser.parse_args()


def main(config):
    """Set up the project and run the pipeline, recording any error.

    The top-level handler records *any* exception that escapes the run --
    not just an :class:`AutoWISPError`. A plain exception from the
    orchestration layer (e.g. a bad configuration expression) is wrapped
    in a :class:`PipelineError` so it is still captured. The error is
    stamped with the pipeline-run snapshot (and the ``crashed`` time) if
    it lacks one, then reported via :func:`report_error` -- recording it
    as a queryable ``Error`` row plus sidecar and rendering a summary to
    stderr (which, for a detached run, is ``run_pipeline.out``) -- and
    re-raised. Reporting is best-effort and never masks the original
    error.
    """

    # Migrate here, in the one process that is definitely alone: workers
    # opening the project later only check the schema, since concurrent DDL
    # from a pool of them is not survivable.
    #
    # No assume_backed_up: a local SQLite database is copied aside and
    # migrated without ceremony, while a centralised one refuses and points
    # at wisp-migrate. Migrating a database several people share should be a
    # deliberate act, not a side effect of starting a photometry run.
    set_project_home(config.project_home, migrate=True)
    try:
        _run_pipeline(config)
    except AutoWISPError as exc:
        _record_escaping_error(exc)
        raise
    except Exception as exc:
        # A non-AutoWISP exception (e.g. a NameError from a bad config
        # expression) would otherwise crash the run unrecorded. Wrap it so
        # it is captured like any other failure, preserving the original
        # as the cause and its traceback for the sidecar.
        wrapped = PipelineError(str(exc) or type(exc).__name__)
        wrapped.__cause__ = exc
        wrapped.__traceback__ = exc.__traceback__
        _record_escaping_error(wrapped)
        raise


def _record_escaping_error(exc):
    """Stamp the pipeline-run snapshot (if missing) and report ``exc``."""

    if exc.pipeline_run is None:
        exc.with_pipeline_run(get_error_context().pipeline_run)
    report_error(exc)


@contextmanager
def _lock_run_start():
    """
    Let one run at a time record itself, on a server database.

    find_earlier_run() needs the runs' rows to become visible in the order of
    their ids. SQLite sees to that: an insert holds the write lock until the
    commit. A server hands the id out at the insert, so a run recorded later
    could commit first and check without seeing the earlier one. Its
    migration lock serializes them: recording takes too little time for
    sharing the lock with migrating to matter.

    Not taken on SQLite, where it would deadlock: the lock there makes every
    transaction of the engine wait for the write lock it holds.
    """

    engine = get_db_engine()
    if engine.dialect.name == "sqlite":
        yield
        return
    with locked_connection(engine):
        yield


def _run_pipeline(config):
    """Create the pipeline run and drive image + lightcurve processing."""

    # old code
    # db_fname = os.path.abspath(config.processing_database)
    # set_sqlite_database(db_fname)

    # with start_db_session() as db_session:
    #    dummy_processing = ProcessingManager(None)
    #    dummy_config = dummy_processing.get_config(
    #        dummy_processing.get_matched_expressions(Evaluator()),
    #        db_session,
    #        step_name="add_images_to_db",
    #    )[0]

    # dummy_config["task"] = "run_pipeline"
    # dummy_config["parent_pid"] = ""
    # dummy_config["processing_step"] = "none"
    # dummy_config["image_type"] = "none"

    # setup_process_map(db_fname, dummy_config)

    with _lock_run_start(), start_db_session() as db_session:
        # Also the error context's run, even before the first setup_process
        # call.
        pipeline_run = record_run(db_session).id

    try:
        _process(config, pipeline_run)
    finally:
        # However the run ends, since its process may go on: the BUI adds raw
        # images by calling main() in its own process, and the run would
        # otherwise look alive for as long as the BUI is.
        with start_db_session() as db_session:
            db_session.execute(
                update(PipelineRun).where(PipelineRun.id == pipeline_run)
                # pylint: disable-next=not-callable
                .values(finished=sql.func.now())
            )


def _get_enabled_processing(config):
    """
    Return the (step, image type) pairs to process, by name; None for all.

    A pair is enabled if ``--steps`` names its step, or is not given, and
    ``--step-imtypes`` names the pair, or names no pair of its step.
    ``--steps`` given empty enables nothing: only the raw images are added.

    The processing is given the enabled pairs as one set, and records what
    is pending for every pair, enabled or not, before skipping those not
    enabled, so that a later step needing one waits for it.

    Raises:
        PipelineError:    If an entry of ``--step-imtypes`` is not given as
            ``step:imagetype``.
    """

    if config.steps is None and not config.step_imtypes:
        return None
    image_types = {}
    for pair in config.step_imtypes:
        step_name, separator, image_type_name = pair.partition(":")
        step_name = step_name.strip()
        image_type_name = image_type_name.strip()
        if not separator or not step_name or not image_type_name:
            raise PipelineError(
                f"Step and image type {pair!r} is not given as "
                "step:imagetype, e.g. calibrate:flat."
            )
        image_types.setdefault(step_name, set()).add(image_type_name)
    with start_db_session() as db_session:
        return {
            (step.name, image_type.name)
            for step, image_type in get_processing_sequence(db_session)
            if (config.steps is None or step.name in config.steps)
            and image_type.name in image_types.get(step.name, {image_type.name})
        }


def _process(config, pipeline_run):
    """Add the images and process them, as the run with the given id."""

    # Only now that the run is recorded: an experiment being created at the
    # same time then either sees the run, or has locked the project already.
    ProjectExperiments(get_db_engine(), get_project_home()).refuse_run()

    enabled_processing = _get_enabled_processing(config)
    # Recorded first, then checked, so that of runs starting together
    # exactly one goes on (find_earlier_run). Only adding images is left to
    # go alongside a run.
    if enabled_processing is None or enabled_processing:
        with start_db_session() as db_session:
            earlier = find_earlier_run(db_session, pipeline_run)
            if earlier is not None:
                logging.warning(
                    "Not processing: run %d (process %d) is already "
                    "processing this project.",
                    earlier.id,
                    earlier.process_id,
                )
                return

    if enabled_processing is not None:
        logging.info("Processing only: %s", sorted(enabled_processing))

    processing = ImageProcessingManager(pipeline_run_id=pipeline_run)

    for img_to_add in config.add_raw_images:
        logging.info("Adding raw images from: %s", img_to_add)
        processing.add_raw_images(find_fits_fnames(os.path.abspath(img_to_add)))

    if enabled_processing is None or enabled_processing:
        logging.info(
            "Starting processing for project home %s...",
            get_project_home(),
        )
        sys.stdout.flush()
        sys.stderr.flush()

        processing(enabled_processing)
        logging.info("Processing completed.")
        sys.stdout.flush()
        sys.stderr.flush()

        LightCurveProcessingManager(pipeline_run_id=pipeline_run)(
            enabled_processing
        )


if __name__ == "__main__":
    with open(
        os.path.join(
            platformdirs.user_data_dir("autowisp"), "run_pipeline.out"
        ),
        "w",
        encoding="utf-8",
        buffering=1,
    ) as outf:
        sys.stdout = outf
        sys.stderr = outf

        if os.name == "posix":  # Linux/macOS
            from os import getpgid, setsid, fork
            import platform

            if platform.system() == "Darwin":
                # Double-fork is unsafe on macOS after Python runtime init.
                # Zombie reaping is handled by views.py (daemon thread).
                main(parse_command_line())
                sys.exit(0)

            try:
                setsid()
            except OSError:
                print(f"pid={os.getpid():d}  pgid={getpgid(0):d}")

            pid = fork()
            if pid < 0:
                raise ResourceError(
                    "Could not start the pipeline in the background: the "
                    "operating system refused to create a new process "
                    "(fork() failed)."
                )
            if pid != 0:
                sys.exit(0)

            setsid()
            main(parse_command_line())  # Run main function in child process

        elif os.name == "nt":  # Windows
            try:
                main(parse_command_line())
            except Exception as e:  # pylint: disable=broad-except
                with open(
                    "detached_process_error.log", "w", encoding="utf-8"
                ) as error_log:
                    error_log.write(f"Error in main: {format_exc()}\n")
