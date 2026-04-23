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


def wait_for_server(hostname, port):
    """Waits for the Django server to respond to requests."""

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


def start_server():
    """Starts the Django development server."""

    filenames = {
        ext: str(settings.BASE_DIR / f"bui.{ext}")
        for ext in ["out", "err", "log"]
    }
    if not os.path.exists(str(settings.BASE_DIR)):
        os.makedirs(str(settings.BASE_DIR))

    with open(filenames["out"], "w", encoding="utf-8") as outf, open(
        filenames["err"], "w", encoding="utf-8"
    ) as errf:
        sys.stdout = outf
        sys.stderr = errf

        print("Test redirect")
        sys.stdout.flush()
        sys.stderr.flush()

    config = parse_command_line()
    logging_level = getattr(logging, config.verbose.upper())
    logging.basicConfig(
        level=logging_level,
        filename=filenames["log"],
        format="%(levelname)s %(asctime)s %(name)s: %(message)s | "
        "%(pathname)s.%(funcName)s:%(lineno)d",
        force=True,
    )

    with open(filenames["out"], "w", encoding="utf-8") as outf, open(
        filenames["err"], "w", encoding="utf-8"
    ) as errf:
        sys.stdout = outf
        sys.stderr = errf

        hostname = "localhost"
        if len(config.hostname) > 1:
            if ":" in config.hostname:
                hostname, port = config.hostname.split(":")
            else:
                port = config.hostname
            port = int(port)
        else:
            port = find_free_port(hostname)

        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(__file__), "manage.py"),
        ]
        subprocess.run(
            cmd + ["migrate"], check=True, stdout=sys.stdout, stderr=sys.stderr
        )
        sys.stdout.flush()
        sys.stderr.flush()

        cmd.extend(["runserver", f"{port}"])
        print(f"Starting server with command: {' '.join(cmd)} in environment:")
        print("\n\t".join([f"{k}={v}" for k, v in os.environ.items()]))
        print("Python paths:\n\t" + "\n\t".join(sys.path))
        sys.stdout.flush()
        sys.stderr.flush()
        with subprocess.Popen(
            cmd, stdout=sys.stdout, stderr=sys.stderr
        ) as server_cmd:
            wait_for_server(hostname, port)
            webbrowser.open_new_tab(f"http://{hostname}:{port}")
            server_cmd.wait()


if __name__ == "__main__":
    start_server()
