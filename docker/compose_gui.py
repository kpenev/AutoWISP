import os
import re
import sys
import difflib
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

def read_compose_text():
    return COMPOSE_PATH.read_text(encoding="utf-8")


def find_and_replace_sources(text, new_storage, new_tmp, new_bui=None, new_anet_narrow=None, new_anet_wide=None):
    # Work line-by-line to preserve formatting; find target: /storage and /tmp and replace the nearest source: above them
    lines = text.splitlines()

    def replace_for_target(target, new_path):
        """Find a line whose target path starts with `target` and replace the
        nearest preceding `source:` line with new_path. This handles cases like
        `target: /storage/<container name>` by using prefix matching instead of
        exact equality.
        """
        # TODO: simplify this by proper usage of regular expressions
        # find line index with target: {target} or target: {target}/...
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith("target:"):
                # extract the configured path after 'target:' and compare prefix
                tgt_val = s.split("target:", 1)[1].strip()
                if tgt_val == target or tgt_val.startswith(target.rstrip("/") + "/") or tgt_val.startswith(target + ""):

                    # search backwards for a source: line and replace it
                    for j in range(i - 1, -1, -1):
                        if lines[j].lstrip().startswith("source:"):
                            indent = lines[j][:len(lines[j]) - len(lines[j].lstrip())]
                            # keep YAML style, don't escape backslashes
                            lines[j] = f"{indent}source: {new_path}"
                            return

    replace_for_target("/storage", new_storage)
    replace_for_target("/tmp", new_tmp)
    # additional targets used in compose.yaml (only replace if value provided)
    if new_bui:
        replace_for_target("/app_data/autowisp", new_bui)
    if new_anet_narrow:
        replace_for_target("/anet_indices/narrow", new_anet_narrow)
    if new_anet_wide:
        replace_for_target("/anet_indices/wide", new_anet_wide)
    return "\n".join(lines) + ("\n" if text.endswith("\n") else "")


def find_free_port(hostname="localhost"):
    """Find an available port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((hostname, 0))
        return str(sock.getsockname()[1])


def get_current_port():
    text = read_compose_text()
    lines = text.splitlines()
    # Find first port mapping like "8089:8089" and return (host_port, container_port, quote_char)
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
            quote = '"' if '"' in s or '"' in line else ('\'' if "'" in s else '')
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
    storage = tmp = bui = anet_narrow = anet_wide = ""
    storage_label = "Storage folder:"
    tmp_label = "Tmp folder:"
    bui_label = "BUI folder:"
    anet_narrow_label = "Anet narrow indices:"
    anet_wide_label = "Anet wide indices:"

    lines = text.splitlines()
    placeholder_re = re.compile(r"<([^>|]+)\|([^>]+)>")

    targets = {
        "/storage": ("storage", "storage_label"),
        "/tmp": ("tmp", "tmp_label"),
        "/app_data/autowisp": ("bui", "bui_label"),
        "/anet_indices/narrow": ("anet_narrow", "anet_narrow_label"),
        "/anet_indices/wide": ("anet_wide", "anet_wide_label"),
    }

    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("target:"):
            # extract the target path
            parts = s.split("target:", 1)
            if len(parts) < 2:
                continue
            target_path = parts[1].strip()
            # use prefix matching so targets like '/storage/<container name>' are matched
            for key, (var_name, label_name) in targets.items():
                if target_path == key or target_path.startswith(key.rstrip("/") + "/"):
                    # find source above
                    for j in range(i - 1, -1, -1):
                        if lines[j].lstrip().startswith("source:"):
                            val = lines[j].split("source:", 1)[1].strip()
                            # set the variable
                            if var_name == "storage":
                                storage = val
                            elif var_name == "tmp":
                                tmp = val
                            elif var_name == "bui":
                                bui = val
                            elif var_name == "anet_narrow":
                                anet_narrow = val
                            elif var_name == "anet_wide":
                                anet_wide = val

                            # extract description for label if present
                            m = placeholder_re.search(val)
                            if m:
                                desc = m.group(2).strip()
                                if desc:
                                    text_label = desc if desc.endswith(":") else desc + ":"
                                    if label_name == "storage_label":
                                        storage_label = text_label
                                    elif label_name == "tmp_label":
                                        tmp_label = text_label
                                    elif label_name == "bui_label":
                                        bui_label = text_label
                                    elif label_name == "anet_narrow_label":
                                        anet_narrow_label = text_label
                                    elif label_name == "anet_wide_label":
                                        anet_wide_label = text_label
                            break

    return SimpleNamespace(
        storage=storage,
        tmp=tmp,
        bui=bui,
        anet_narrow=anet_narrow,
        anet_wide=anet_wide,
        storage_label=storage_label,
        tmp_label=tmp_label,
        bui_label=bui_label,
        anet_narrow_label=anet_narrow_label,
        anet_wide_label=anet_wide_label,
    )


class MountPoint:
    def __init__(self, root, row, label_text, initial_value, title, on_select=None, width=60):
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
        p = filedialog.askdirectory(initialdir=self.get() or os.getcwd(), title=self.title)
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
        storage = sources.storage
        tmp = sources.tmp
        bui = sources.bui
        anet_narrow = sources.anet_narrow
        anet_wide = sources.anet_wide
        storage_label = sources.storage_label
        tmp_label = sources.tmp_label
        bui_label = sources.bui_label
        anet_narrow_label = sources.anet_narrow_label
        anet_wide_label = sources.anet_wide_label
        host_port, container_port, quote = get_current_port()

        # Use labels extracted from the compose YAML placeholders when available
        mount_defs = [
            ("storage", storage_label, storage, "Select storage folder", self.handle_storage_selected),
            ("tmp", tmp_label, tmp, "Select tmp folder", None),
            ("bui", bui_label, bui, "Select BUI folder", None),
            ("anet_narrow", anet_narrow_label, anet_narrow, "Select anet narrow indices folder", None),
            ("anet_wide", anet_wide_label, anet_wide, "Select anet wide indices folder", None),
        ]
        self.mounts = {}
        for row, (name, label_text, value, title, on_select) in enumerate(mount_defs):
            callback = None
            if on_select:
                callback = lambda p, fn=on_select: fn(p, show_info=True)
            mount = MountPoint(root, row, label_text, value, title, on_select=callback)
            self.mounts[name] = mount
            setattr(self, f"{name}_var", mount.var)
            setattr(self, f"{name}_entry", mount.entry)
            setattr(self, f"{name}_browse_btn", mount.browse_btn)

        tk.Label(root, text="Port (host:container)").grid(row=5, column=0, sticky="w")
        self.port_var = tk.StringVar(value=host_port)
        self.port_entry = tk.Entry(root, textvariable=self.port_var, width=20)
        self.port_entry.grid(row=5, column=1, sticky="w", padx=6, pady=6)

        # Buttons: place inside a frame so layout is stable and won't disappear if window is resized
        button_frame = tk.Frame(root)
        button_frame.grid(row=6, column=0, columnspan=3, pady=8)

        self.run_btn = tk.Button(button_frame, text="Run 'docker compose up'", command=self.run_docker)
        self.run_btn.grid(row=0, column=1, padx=6)

        self.update_image_btn = tk.Button(button_frame, text="Update image", command=self.update_image)
        self.update_image_btn.grid(row=0, column=2, padx=6)

        # Enforce storage selection at startup (Option A): disable everything except storage
        # and force the user to pick a storage folder before proceeding.
        self.disable_all_except_storage()
        self.enforce_storage_at_startup()

    def handle_storage_selected(self, p, show_info=False):
        """Common handler when a storage folder is selected.

        - set storage_var
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
            messagebox.showwarning("Warning", f"Failed to create some directories: {e}")

        # enable UI now that storage is provided
        self.enable_all_widgets()

        # Inform the user and immediately persist the chosen paths to compose.yaml.
        # This writes changes as soon as the user selects folders.
        if show_info:
            messagebox.showinfo("Storage selected", "Paths updated in the form and will be saved to compose.yaml.")

        try:
            new_text = find_and_replace_sources(
                read_compose_text(),
                self.storage_var.get(),
                self.tmp_var.get(),
                new_bui=self.bui_var.get() if hasattr(self, 'bui_var') else None,
                new_anet_narrow=self.anet_narrow_var.get() if hasattr(self, 'anet_narrow_var') else None,
                new_anet_wide=self.anet_wide_var.get() if hasattr(self, 'anet_wide_var') else None,
            )
            # preserve port mappings as-is
            new_text = find_and_replace_port(new_text, self.port_var.get())
            COMPOSE_PATH.write_text(new_text, encoding="utf-8")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save compose.yaml: {e}")

    def disable_all_except_storage(self):
        """Disable all entries/browse buttons and action buttons except the storage row."""
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
                messagebox.showerror("Error", f"compose file not found: {COMPOSE_PATH}")
                return

            if not messagebox.askyesno(
                "Update image",
                "Stop and remove the wisp container, then pull the latest kpenev/wisp image?"
            ):
                return

            cmd_str = "docker compose stop wisp && docker compose rm -f wisp && docker pull kpenev/wisp & exit"
            subprocess.Popen(["cmd.exe", "/c", "start", "", "cmd", "/k", cmd_str], cwd=PROJECT_ROOT)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update Docker image: {e}")


    def apply(self):
        s = self.storage_var.get()
        t = self.tmp_var.get()
        p = self.port_var.get()
        if not Path(s).exists():
            messagebox.showerror("Error", f"Storage path does not exist: {s}")
            return
        if not Path(t).exists():
            messagebox.showerror("Error", f"Tmp path does not exist: {t}")
            return
        # Defensive check: ensure BUI path exists too (should have been created at storage selection)
        if hasattr(self, 'bui_var'):
            bpath = self.bui_var.get()
            if bpath and not Path(bpath).exists():
                messagebox.showerror("Error", f"BUI path does not exist: {bpath}")
                return
        # show a final preview
        new_text = find_and_replace_sources(
            read_compose_text(),
            s,
            t,
            new_bui=self.bui_var.get() if hasattr(self, 'bui_var') else None,
            new_anet_narrow=self.anet_narrow_var.get() if hasattr(self, 'anet_narrow_var') else None,
            new_anet_wide=self.anet_wide_var.get() if hasattr(self, 'anet_wide_var') else None,
        )
        new_text = find_and_replace_port(new_text, p)
        diff = list(difflib.unified_diff(read_compose_text().splitlines(keepends=True), new_text.splitlines(keepends=True), fromfile=str(COMPOSE_PATH), tofile=str(COMPOSE_PATH) + " (new)"))
        if not diff:
            messagebox.showinfo("No changes", "No changes to apply")
            return
        if not messagebox.askyesno("Confirm", "Apply changes to compose file?"):
            return
        try:
            # write directly; user can use Reset to restore from git if needed
            COMPOSE_PATH.write_text(new_text, encoding="utf-8")
            messagebox.showinfo("Success", "compose.yaml updated")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply changes: {e}")

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
                "This compose.yaml looks uninitialized. Please select a storage folder first using the Storage Browse... button."
            )
            # leave other widgets disabled (they were disabled already)
            return
        else:
            # compose.yaml appears modified -> enable the UI immediately
            try:
                messagebox.showinfo(
                    "Welcome",
                    "compose.yaml already configured — you may change any paths now."
                )
            except Exception:
                pass
            self.enable_all_widgets()
            return

    def run_docker(self):
        # Run docker compose up. Previously this always ran in the AutoWISP/docker
        # directory which prevented running the GUI + compose.yaml from arbitrary
        # locations. Use the compose file path directly so the command can be
        # executed from anywhere.
        try:
            if not COMPOSE_PATH.exists():
                messagebox.showerror("Error", f"compose file not found: {COMPOSE_PATH}")
                return
            
            # Save the port configuration to compose.yaml before running docker
            try:
                current_text = read_compose_text()
                new_text = find_and_replace_port(current_text, self.port_var.get())
                COMPOSE_PATH.write_text(new_text, encoding="utf-8")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save port configuration: {e}")
                return
            
            # Open a new cmd window in the current working directory and run
            # 'docker compose up'. This is intentionally simple: it behaves
            # like the user opened a terminal and ran 'docker compose up' in
            # the folder where the GUI was started. Avoids fiddling with
            # compose paths or -f arguments which previously caused quoting
            # issues when using Windows 'start'.
            cwd = os.getcwd()
            cmd_str = 'docker compose up'
            subprocess.Popen(["cmd.exe", "/c", "start", "", "cmd", "/k", cmd_str], cwd=cwd)
            # Poll the service URL and open the browser only when it responds
            try:
                port = int(self.port_var.get())
            except Exception:
                port = 8089

            url = f"http://localhost:{port}/"
            timeout = 180  # seconds
            poll_interval = 2  # seconds

            def _poll_and_open():
                end_time = time.time() + timeout
                first_wait = True
                while time.time() < end_time:
                    try:
                        with urllib.request.urlopen(url, timeout=3) as resp:
                            # getcode() works across Python versions
                            code = getattr(resp, 'status', None) or resp.getcode()
                            if code and code < 400:
                                try:
                                    webbrowser.open(url)
                                except Exception:
                                    pass
                                return
                    except Exception:
                        # keep waiting
                        pass

                    # Waiting status is intentionally not shown in the UI.
                    if first_wait:
                        first_wait = False
                    time.sleep(poll_interval)

                # timed out
                self.root.after(0, lambda u=url: messagebox.showwarning("Timeout", f"Timed out waiting for {u} to respond ({timeout}s)."))

            threading.Thread(target=_poll_and_open, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start docker compose: {e}")


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
            lines[i] = line.replace(m.group(0), f'{q}{new_host_port}:{container}{q}')
            return "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    
    # No port mapping found; add ports section after shm_size line
    for i, line in enumerate(lines):
        if 'shm_size:' in line:
            indent = len(line) - len(line.lstrip())
            port_lines = [
                ' ' * indent + 'ports:',
                ' ' * (indent + 2) + f'- "{new_host_port}:8089"'
            ]
            lines = lines[:i+1] + port_lines + lines[i+1:]
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
        if not getattr(app, 'root', None):
            return
    except Exception:
        pass
    root.mainloop()


if __name__ == "__main__":
    main()

