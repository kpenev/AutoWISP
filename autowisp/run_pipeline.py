"""Run the image processing pipeline in a detached mode."""

import logging
import os
import sys
import subprocess
from socket import getfqdn
from traceback import format_exc

from configargparse import ArgumentParser, DefaultsFormatter, SUPPRESS
from sqlalchemy import sql, update
import platformdirs

from autowisp.database.interface import (
    start_db_session,
    set_project_home,
    get_project_home,
)
from autowisp.database.data_model import (  # pylint: disable=no-name-in-module
    PipelineRun,
)
from autowisp.database.image_processing import ImageProcessingManager
from autowisp.database.lightcurve_processing import LightCurveProcessingManager
from autowisp.file_utilities import find_fits_fnames
from autowisp.evaluator import Evaluator


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
        nargs="+",
        default=None,
        help="Process using only the specified steps. Leave empty for full "
        "processing.",
    )
    parser.add_argument(
        "--detached",
        action="store_true",
        help=SUPPRESS,  # Only used internally to detach in windows
    )
    logging.info("Parsed arguments: %s", parser.parse_args())
    return parser.parse_args()


def main(config):
    """Avoid global variables."""

    set_project_home(config.project_home)

    #with start_db_session() as db_session:
    #    dummy_processing = ProcessingManager(None)
    #    dummy_config = dummy_processing.get_config(
    #        dummy_processing.get_matched_expressions(Evaluator()),
    #        db_session,
    #        step_name="add_images_to_db",
    #    )[0]

    #dummy_config["task"] = "run_pipeline"
    #dummy_config["parent_pid"] = ""
    #dummy_config["processing_step"] = "none"
    #dummy_config["image_type"] = "none"

    #setup_process_map(db_fname, dummy_config)

    with start_db_session() as db_session:
        pipeline_run = PipelineRun(
            host=getfqdn(),
            process_id=os.getpid(),
            started=sql.func.now(),  # pylint: disable=not-callable
        )
        db_session.add(pipeline_run)
        db_session.commit()
        pipeline_run = pipeline_run.id

    processing = ImageProcessingManager(pipeline_run_id=pipeline_run)

    for img_to_add in config.add_raw_images:
        logging.info("Adding raw images from: %s", img_to_add)
        processing.add_raw_images(find_fits_fnames(os.path.abspath(img_to_add)))

    if config.steps is None or config.steps:
        logging.info(
            "Starting processing for project home %s...",
            get_project_home(),
        )
        sys.stdout.flush()
        sys.stderr.flush()

        processing(limit_to_steps=config.steps)
        logging.info("Processing completed.")
        sys.stdout.flush()
        sys.stderr.flush()

        LightCurveProcessingManager(pipeline_run_id=pipeline_run)()

    with start_db_session() as db_session:
        db_session.execute(
            update(PipelineRun)
            .where(PipelineRun.id == pipeline_run)
            .values(finished=sql.func.now())  # pylint: disable=not-callable
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
                main(parse_command_line())
                sys.exit(0)

            try:
                setsid()
            except OSError:
                print(f"pid={os.getpid():d}  pgid={getpgid(0):d}")

            pid = fork()
            if pid < 0:
                raise RuntimeError("fork fail")
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
