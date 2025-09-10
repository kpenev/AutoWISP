#!/usr/bin/env python3
"""Start django server wait to initialize and open in browser."""

import subprocess
import time
import sys
from http.client import HTTPConnection
import webbrowser
import os

from autowisp.browser_interface.django_project import settings


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


def start_server():
    """Starts the Django development server."""

    if not os.path.exists(str(settings.BASE_DIR)):
        os.makedirs(str(settings.BASE_DIR))

    with open(
        str(settings.BASE_DIR / "bui.out"), "w", encoding="utf-8"
    ) as outf, open(
        str(settings.BASE_DIR / "bui.err"), "w", encoding="utf-8"
    ) as errf:
        sys.stdout = outf
        sys.stderr = errf

        print("Test redirect")
        sys.stdout.flush()
        sys.stderr.flush()


    with open(
        str(settings.BASE_DIR / "bui.out"), "w", encoding="utf-8"
    ) as outf, open(
        str(settings.BASE_DIR / "bui.err"), "w", encoding="utf-8"
    ) as errf:
        sys.stdout = outf
        sys.stderr = errf

        if ":" in sys.argv[1]:
            hostname, port = sys.argv[1].split(":")
        else:
            port = sys.argv[1]
            hostname = "localhost"
        port = int(port)

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
        print('\n\t'.join([f"{k}={v}" for k, v in os.environ.items()]))
        print('Python paths:\n\t' + '\n\t'.join(sys.path))
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
