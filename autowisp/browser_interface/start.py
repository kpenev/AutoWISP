#!/usr/bin/env python3
"""Start django server wait to initialize and open in browser."""

import subprocess
import time
import sys
import socket
from http.client import HTTPConnection
import webbrowser
import os
import logging

from configargparse import ArgumentParser, DefaultsFormatter

from autowisp.browser_interface.django_project import settings


def parse_command_line():
    """Return command line configuration."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=[],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False,
    )
    parser.add_argument(
        "--hostname",
        default='',
        metavar="[<HOSTNAME>:]<PORT>",
        help="The port to run the surver on and optionally hostname. By "
        "default automatically finds an oppen port on localhost.",
    )
    parser.add_argument(
        "--verbose",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="The type of verbosity of logger.",
    )
    return parser.parse_args()


def wait_until_responsive(hostname, port):
    """Poll until the django server answers requests."""

    url = f"http://{hostname}:{port}"

    while True:
        conn = None
        try:
            conn = HTTPConnection(hostname, port, timeout=1)
            conn.request("HEAD", "/")
            response = conn.getresponse()
            if 200 <= response.status < 400:
                print(f"Server is ready at {url}")
                return
        except Exception:  # pylint: disable=broad-exception-caught
            time.sleep(0.5)
        finally:
            if conn:
                conn.close()


def find_free_port(hostname="localhost"):
    """Find an available port by binding to port 0."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((hostname, 0))
        return sock.getsockname()[1]


def migrate_and_serve(hostname, port):
    """Apply migrations, launch django and block until it exits."""

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "manage.py"),
    ]
    subprocess.run(
        cmd + ["migrate"], check=True, stdout=sys.stdout, stderr=sys.stderr
    )
    sys.stdout.flush()
    sys.stderr.flush()

    cmd.extend(["runserver", f"{hostname}:{port}"])
    print(f"Starting server with command: {' '.join(cmd)} in environment:")
    print("\n\t".join([f"{k}={v}" for k, v in os.environ.items()]))
    print("Python paths:\n\t" + "\n\t".join(sys.path))
    sys.stdout.flush()
    sys.stderr.flush()
    with subprocess.Popen(
        cmd, stdout=sys.stdout, stderr=sys.stderr
    ) as server_cmd:
        wait_until_responsive(hostname, port)
        webbrowser.open_new_tab(f"http://{hostname}:{port}")
        server_cmd.wait()


def main():
    """Set up config, logging and output redirection, then serve."""

    config = parse_command_line()

    os.makedirs(str(settings.BASE_DIR), exist_ok=True)
    filenames = {
        ext: str(settings.BASE_DIR / f"bui.{ext}")
        for ext in ["out", "err", "log"]
    }

    logging.basicConfig(
        level=getattr(logging, config.verbose.upper()),
        filename=filenames["log"],
        format="%(levelname)s %(asctime)s %(name)s: %(message)s | "
        "%(pathname)s.%(funcName)s:%(lineno)d",
        force=True,
    )

    hostname = "localhost"
    if len(config.hostname) > 1:
        if ":" in config.hostname:
            hostname, port = config.hostname.split(":")
        else:
            port = config.hostname
        port = int(port)
    else:
        port = find_free_port(hostname)

    # The redirected streams must be restored before the files are closed,
    # otherwise any later output (e.g. a warning or a traceback) is written to
    # a closed file and the interpreter dies with "lost sys.stderr".
    original_streams = (sys.stdout, sys.stderr)
    with open(filenames["out"], "w", encoding="utf-8") as outf, open(
        filenames["err"], "w", encoding="utf-8"
    ) as errf:
        try:
            sys.stdout = outf
            sys.stderr = errf
            migrate_and_serve(hostname, port)
        finally:
            sys.stdout, sys.stderr = original_streams


if __name__ == "__main__":
    main()
