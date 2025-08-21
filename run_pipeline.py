"""Run the image processing pipeline in a detached mode."""

import logging
import os
import sys
import subprocess
from traceback import format_exc

from configargparse import ArgumentParser, DefaultsFormatter, SUPPRESS

from autowisp.database.image_processing import ImageProcessingManager
from autowisp.database.lightcurve_processing import LightCurveProcessingManager
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
        'processing_database',
        help="Path to the processing database."
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

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)

    logging.debug("Config add_raw_images: %s", config.add_raw_images)
    logging.debug("Config steps: %s", config.steps)

    processing = ImageProcessingManager()
    for img_to_add in config.add_raw_images:
        logging.debug("Adding raw images from: %s", img_to_add)
        processing.add_raw_images(find_fits_fnames(os.path.abspath(img_to_add)))

    logging.debug("Starting processing...")
    processing(limit_to_steps=config.steps)
    logging.debug("Processing completed.")

    LightCurveProcessingManager()()


if __name__ == "__main__":
    if os.name == "posix":  # Linux/macOS
        from os import getpgid, setsid, fork

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
        from subprocess import DETACHED_PROCESS

        if "--detached" not in sys.argv:
            try:
                with open(
                    "detached_process.log", "w", encoding="utf-8"
                ) as log_file:
                    subprocess.Popen(  # pylint: disable=consider-using-with
                        [
                            sys.executable,
                            os.path.abspath(sys.argv[0]),
                            "--detached",
                        ]
                        + sys.argv[1:],  # Relaunch with --detached
                        creationflags=DETACHED_PROCESS,
                        stdout=log_file,
                        stderr=log_file,
                    )
                sys.exit(0)  # Exit parent process
            except Exception as e:  # pylint: disable=broad-except
                sys.stderr.write(f"Failed to detach: {format_exc()}\n")
                sys.exit(1)
        else:
            try:
                main(parse_command_line())
            except Exception as e:  # pylint: disable=broad-except
                with open(
                    "detached_process_error.log", "w", encoding="utf-8"
                ) as error_log:
                    error_log.write(f"Error in main: {format_exc()}\n")
