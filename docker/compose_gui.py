import os
import re
import sys
import subprocess
import threading
import time
import webbrowser
import urllib.request
import socket
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk
from tkinter import filedialog, messagebox


def _find_compose_path():
    """Locate compose.yaml.

    When running as a PyInstaller frozen executable, resolve the compose file
    from the executable directory.
    Returns a Path object (may not exist).
    """

    try:
        if getattr(sys, "frozen", False):
            exe_parent = Path(sys.executable).resolve().parent
            return exe_parent / "compose.yaml"
    except Exception:
        pass

    try:
        # Use the script directory (same folder as this file) as the
        # development fallback so compose.yaml next to compose_gui.py is
        # preferred (e.g. C:/.../AutoWISP/tools/compose.yaml).
        return Path(__file__).resolve().parent / "compose.yaml"
    except Exception:
        return Path("compose.yaml")


COMPOSE_PATH = _find_compose_path()
PROJECT_ROOT = COMPOSE_PATH.resolve().parent
MOUNT_TARGETS = {
    "storage": "/storage",
    "tmp": "/tmp",
    "bui": "/app_data/autowisp",
    "anet_narrow": "/anet_indices/narrow",
    "anet_wide": "/anet_indices/wide",
}
DEFAULT_LABELS = {
    "storage": "Storage folder:",
    "tmp": "Tmp folder:",
    "bui": "BUI folder:",
    "anet_narrow": "Anet narrow indices:",
    "anet_wide": "Anet wide indices:",
}


def read_compose_text():
    return COMPOSE_PATH.read_text(encoding="utf-8")


def target_matches(target_path, expected):
    return target_path == expected or target_path.startswith(
        expected.rstrip("/") + "/"
    )


def find_and_replace_sources(
    text,
    new_storage,
    new_tmp,
    new_bui=None,
    new_anet_narrow=None,
    new_anet_wide=None,
):
    # Work line-by-line to preserve formatting; find target:
    # /storage and /tmp and replace the nearest source: above them
    lines = text.splitlines()

    def replace_for_target(target, new_path):
        """Find a line whose target path starts with `target` and replace the
        nearest preceding `source:` line with new_path. This handles cases like
        `target: /storage/<container name>` by using prefix matching instead of
        exact equality.
        """
        target_re = re.compile(
            rf"^\s*target:\s*{re.escape(target.rstrip('/'))}(?:/.*)?\s*$"
        )
        source_re = re.compile(r"^(\s*)source:")
        for i, line in enumerate(lines):
            if target_re.match(line):
                for j in range(i - 1, -1, -1):
                    source_match = source_re.match(lines[j])
                    if source_match:
                        lines[j] = f"{source_match.group(1)}source: {new_path}"
                        return

    replacements = {
        "storage": new_storage,
        "tmp": new_tmp,
        "bui": new_bui,
        "anet_narrow": new_anet_narrow,
        "anet_wide": new_anet_wide,
    }
    for name, new_path in replacements.items():
        if new_path:
            replace_for_target(MOUNT_TARGETS[name], new_path)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def find_free_port(hostname="localhost"):
    """Find an available port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((hostname, 0))
        return str(sock.getsockname()[1])


def get_current_port():
    text = read_compose_text()
    lines = text.splitlines()
    # Find first port mapping like "8089:8089" and return the host port.
    m = None
    for line in lines:
        # strip leading/trailing spaces
        s = line.strip()
        # regex for optional quotes around mapping
        m = re.search(r'["\']?(\d+):(\d+)["\']?', s)
        if m:
            host = m.group(1)
            container = m.group(2)
            # determine if mapping used quotes
            quote = (
                '"' if '"' in s or '"' in line else ("'" if "'" in s else "")
            )
            return host, container, quote
    # default: find a free port on the host
    free_port = find_free_port()
    return free_port, "8089", '"'


def get_current_sources():
    """Return current source paths and user-facing labels for common targets.

    This inspects compose.yaml and looks for 'source:' lines above known
    'target:' entries. If the source uses the angle-bracket placeholder with
    a pipe (e.g. <IOdir| description>) the description (text after |) is
    extracted and returned as the label for that field in the GUI.

    Returns a tuple:
      (storage, tmp, bui, anet_narrow, anet_wide,
       storage_label, tmp_label, bui_label, anet_narrow_label, anet_wide_label)
    """
    text = read_compose_text()
    lines = text.splitlines()
    values = {name: "" for name in MOUNT_TARGETS}
    labels = DEFAULT_LABELS.copy()
    placeholder_re = re.compile(r"<([^>|]+)\|([^>]+)>")

    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("target:"):
            # extract the target path
            parts = s.split("target:", 1)
            if len(parts) < 2:
                continue
            target_path = parts[1].strip()
            # use prefix matching so targets like
            # '/storage/<container name>' are matched
            for name, expected_target in MOUNT_TARGETS.items():
                if target_matches(target_path, expected_target):
                    # find source above
                    for j in range(i - 1, -1, -1):
                        if lines[j].lstrip().startswith("source:"):
                            val = lines[j].split("source:", 1)[1].strip()
                            values[name] = val

                            # extract description for label if present
                            m = placeholder_re.search(val)
                            if m:
                                desc = m.group(2).strip()
                                if desc:
                                    labels[name] = (
                                        desc
                                        if desc.endswith(":")
                                        else desc + ":"
                                    )
                            break

    return SimpleNamespace(
        **values,
        **{f"{name}_label": label for name, label in labels.items()},
    )


class MountPoint:
    def __init__(
        self,
        root,
        row,
        label_text,
        initial_value,
        title,
        on_select=None,
        width=60,
    ):
        self.title = title
        self.on_select = on_select
        self.label = tk.Label(root, text=label_text)
        self.label.grid(row=row, column=0, sticky="w")
        self.var = tk.StringVar(value=initial_value)
        self.entry = tk.Entry(root, textvariable=self.var, width=width)
        self.entry.grid(row=row, column=1, padx=6, pady=6)
        self.browse_btn = tk.Button(root, text="Browse...", command=self.browse)
        self.browse_btn.grid(row=row, column=2, padx=6)

    def get(self):
        return self.var.get()

    def set(self, value):
        self.var.set(value)

    def set_state(self, state):
        self.entry.configure(state=state)
        self.browse_btn.configure(state=state)

    def browse(self):
        p = filedialog.askdirectory(
            initialdir=self.get() or os.getcwd(), title=self.title
        )
        if p:
            if self.on_select:
                self.on_select(p)
            else:
                self.set(p)


class ComposeEditorApp:
    def __init__(self, root):
        self.root = root
        root.title("AutoWISP Compose Editor")

        sources = get_current_sources()
        host_port, _, _ = get_current_port()

        # Use labels extracted from the compose YAML placeholders when available
        mount_defs = [
            (
                "storage",
                sources.storage_label,
                sources.storage,
                "Select storage folder",
                self.handle_storage_selected,
            ),
            ("tmp", sources.tmp_label, sources.tmp, "Select tmp folder", None),
            ("bui", sources.bui_label, sources.bui, "Select BUI folder", None),
            (
                "anet_narrow",
                sources.anet_narrow_label,
                sources.anet_narrow,
                "Select anet narrow indices folder",
                None,
            ),
            (
                "anet_wide",
                sources.anet_wide_label,
                sources.anet_wide,
                "Select anet wide indices folder",
                None,
            ),
        ]
        self.mounts = {}
        for row, (name, label_text, value, title, on_select) in enumerate(
            mount_defs
        ):
            if on_select:
                callback = lambda p, fn=on_select: fn(p, show_info=True)
            else:
                callback = (
                    lambda p, mount_name=name: self.handle_mount_selected(
                        mount_name, p
                    )
                )
            mount = MountPoint(
                root, row, label_text, value, title, on_select=callback
            )
            mount.entry.bind("<FocusOut>", self.handle_form_field_saved)
            mount.entry.bind("<Return>", self.handle_form_field_saved)
            self.mounts[name] = mount

        tk.Label(root, text="Port (host:container)").grid(
            row=5, column=0, sticky="w"
        )
        self.port_var = tk.StringVar(value=host_port)
        self.port_entry = tk.Entry(root, textvariable=self.port_var, width=20)
        self.port_entry.grid(row=5, column=1, sticky="w", padx=6, pady=6)
        self.port_entry.bind("<FocusOut>", self.handle_form_field_saved)
        self.port_entry.bind("<Return>", self.handle_form_field_saved)

        # Buttons: place inside a frame so layout is stable and
        # won't disappear if window is resized
        button_frame = tk.Frame(root)
        button_frame.grid(row=6, column=0, columnspan=3, pady=8)

        self.run_btn = tk.Button(
            button_frame, text="Start AutoWISP", command=self.run_docker
        )
        self.run_btn.grid(row=0, column=1, padx=6)

        self.update_image_btn = tk.Button(
            button_frame, text="Check for Update", command=self.update_image
        )
        self.update_image_btn.grid(row=0, column=2, padx=6)

        # Enforce storage selection at startup (Option A):
        # disable everything except storage and force the user to
        # pick a storage folder before proceeding.
        self.disable_all_except_storage()
        self.enforce_storage_at_startup()

    def handle_storage_selected(self, p, show_info=False):
        """Common handler when a storage folder is selected.

        - set storage path
        - auto-fill tmp, bui, astrometry paths under storage
        - create those folders if they do not exist
        - enable the previously-disabled widgets
        - optionally show an informational popup
        """
        # normalize path
        p = os.path.abspath(p)
        self.mounts["storage"].set(p)

        # derive defaults
        tmp = os.path.join(p, "tmp")
        bui = os.path.join(p, "BUI")
        # Provide separate astrometry folders for narrow and wide indices
        anet_narrow = os.path.join(p, "astrometry", "narrow")
        anet_wide = os.path.join(p, "astrometry", "wide")

        # set variables
        self.mounts["tmp"].set(tmp)
        self.mounts["bui"].set(bui)
        self.mounts["anet_narrow"].set(anet_narrow)
        self.mounts["anet_wide"].set(anet_wide)

        # create directories if missing (per Option A requirement)
        try:
            os.makedirs(tmp, exist_ok=True)
            os.makedirs(bui, exist_ok=True)
            os.makedirs(anet_narrow, exist_ok=True)
            os.makedirs(anet_wide, exist_ok=True)
        except Exception as e:
            messagebox.showwarning(
                "Warning", f"Failed to create some directories: {e}"
            )

        # enable UI now that storage is provided
        self.enable_all_widgets()

        # Inform the user and immediately persist the chosen paths to compose.yaml.
        # This writes changes as soon as the user selects folders.
        if show_info:
            messagebox.showinfo(
                "Storage selected",
                "Paths updated in the form and will be saved to compose.yaml.",
            )

        self.save_compose_settings()

    def handle_mount_selected(self, name, path):
        self.mounts[name].set(os.path.abspath(path))
        self.save_compose_settings()

    def handle_form_field_saved(self, _event=None):
        self.save_compose_settings()

    def mount_values(self):
        return {name: mount.get() for name, mount in self.mounts.items()}

    def save_compose_settings(self):
        try:
            values = self.mount_values()
            new_text = find_and_replace_sources(
                read_compose_text(),
                values["storage"],
                values["tmp"],
                new_bui=values["bui"],
                new_anet_narrow=values["anet_narrow"],
                new_anet_wide=values["anet_wide"],
            )
            new_text = find_and_replace_port(new_text, self.port_var.get())
            COMPOSE_PATH.write_text(new_text, encoding="utf-8")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save compose.yaml: {e}")
            return False

    def disable_all_except_storage(self):
        """
            Disable all entries/browse buttons and action buttons
            except the storage row.
        """
        for name, mount in self.mounts.items():
            if name == "storage":
                continue
            mount.set_state("disabled")

        # Action buttons
        self.run_btn.configure(state="disabled")

        # Keep storage entry and browse enabled
        self.mounts["storage"].set_state("normal")

    def enable_all_widgets(self):
        """Enable all previously-disabled widgets after storage selection."""
        for mount in self.mounts.values():
            mount.set_state("normal")

        self.run_btn.configure(state="normal")

    def update_image(self):
        try:
            if not COMPOSE_PATH.exists():
                messagebox.showerror(
                    "Error", f"compose file not found: {COMPOSE_PATH}"
                )
                return

            if not messagebox.askyesno(
                "Update image",
                (
                    "Stop and remove the wisp container,"
                    "then pull the latest kpenev/wisp image?"
                ),
            ):
                return

            cmd_str = (
                "docker compose stop wisp &&"
                "docker compose rm -f wisp &&"
                "docker pull kpenev/wisp & exit"
            )
            subprocess.Popen(
                ["cmd.exe", "/c", "start", "", "cmd", "/k", cmd_str],
                cwd=PROJECT_ROOT,
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update Docker image: {e}")

    def enforce_storage_at_startup(self):
        """Decide startup behaviour based on whether compose.yaml still contains
        the placeholder markers (i.e. it is unmodified). If the compose file is
        unmodified (contains placeholders like <IOdir|...), only the Storage
        selector remains enabled. If the compose file appears modified, enable
        all controls so the user can edit freely.
        """
        text = ""
        try:
            text = read_compose_text()
        except Exception:
            # If compose can't be read, default to strict mode (only storage enabled)
            text = ""

        # detect placeholder-style entries used in the original template
        placeholder_re = re.compile(r"<[^>|]+\|[^>]+>")

        if placeholder_re.search(text):
            # compose.yaml looks unmodified -> keep only storage enabled
            messagebox.showinfo(
                "Welcome!",
                (
                "This compose.yaml looks uninitialized."
                "Please select a storage folder first using the"
                "Storage Browse... button.",
                )
            )
            # leave other widgets disabled (they were disabled already)
            return
        else:
            # compose.yaml appears modified -> enable the UI immediately
            try:
                messagebox.showinfo(
                    "Welcome",
                    "compose.yaml already configured — you may change any paths now.",
                )
            except Exception:
                pass
            self.enable_all_widgets()
            return

    def run_docker(self):
        try:
            if not COMPOSE_PATH.exists():
                messagebox.showerror(
                    "Error", f"compose file not found: {COMPOSE_PATH}"
                )
                return

            # Save the current form values to compose.yaml before running docker.
            if not self.save_compose_settings():
                return

            cwd = os.getcwd()
            cmd_str = "docker compose up"
            subprocess.Popen(
                ["cmd.exe", "/c", "start", "", "/min", "cmd", "/k", cmd_str], cwd=cwd
            )
            # Poll the service URL and open the browser only when it responds
            try:
                port = int(self.port_var.get())
            except Exception:
                port = 8089

            url = f"http://localhost:{port}/"
            timeout = 600  # seconds
            poll_interval = 2  # seconds

            def _poll_and_open():
                end_time = time.time() + timeout
                while time.time() < end_time:
                    try:
                        with urllib.request.urlopen(url, timeout=3) as resp:
                            # getcode() works across Python versions
                            code = (
                                getattr(resp, "status", None) or resp.getcode()
                            )
                            if code and code < 400:
                                try:
                                    os.startfile(url)
                                except Exception:
                                    webbrowser.open(url)
                                return
                    except Exception:
                        # keep waiting
                        pass

                    time.sleep(poll_interval)

                # timed out
                self.root.after(
                    0,
                    lambda u=url: messagebox.showwarning(
                        "Still starting",
                        (
                            f"Timed out waiting for {u} to respond ({timeout}s).\n"
                            "Docker may still be starting;"
                            "open the URL manually when it is ready."
                        )
                    ),
                )

            threading.Thread(target=_poll_and_open, daemon=True).start()
        except Exception as e:
            messagebox.showerror(
                "Error", f"Failed to start docker compose: {e}"
            )


def find_and_replace_port(text, new_host_port):
    # Replace or add port mapping (e.g., "8089:8089") with new_host_port:8089
    # If no port mapping exists, add a ports section after the shm_size line
    lines = text.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        m = re.search(r'(["\']?)(\d+):(\d+)(["\']?)', s)
        if m:
            quote1 = m.group(1)
            container = m.group(3)
            quote2 = m.group(4)
            q = quote1 or quote2 or '"'
            lines[i] = line.replace(
                m.group(0), f"{q}{new_host_port}:{container}{q}"
            )
            return "\n".join(lines) + ("\n" if text.endswith("\n") else "")

    # No port mapping found; add ports section after shm_size line
    for i, line in enumerate(lines):
        if "shm_size:" in line:
            indent = len(line) - len(line.lstrip())
            port_lines = [
                " " * indent + "ports:",
                " " * (indent + 2) + f'- "{new_host_port}:8089"',
            ]
            lines = lines[: i + 1] + port_lines + lines[i + 1 :]
            return "\n".join(lines) + ("\n" if text.endswith("\n") else "")

    return text


def main():
    if not COMPOSE_PATH.exists():
        messagebox.showerror("Error", f"compose file not found: {COMPOSE_PATH}")
        return
    root = tk.Tk()
    app = ComposeEditorApp(root)
    # If the compose editor closed itself because the user cancelled storage
    # selection at startup, exit the program
    try:
        if not getattr(app, "root", None):
            return
    except Exception:
        pass
    root.mainloop()


if __name__ == "__main__":
    main()
