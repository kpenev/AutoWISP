import os
import re
import shutil
import datetime
import sys
import difflib
import subprocess
import threading
import time
import webbrowser
import urllib.request
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext


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
LOG_PATH = PROJECT_ROOT / "compose_gui.log"

try:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as lf:
        lf.write(f"{datetime.datetime.now().isoformat()} COMPOSE_PATH resolved: {COMPOSE_PATH} (exists={COMPOSE_PATH.exists()})\n")
except Exception:
    # non-fatal if logging fails
    pass


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
        # find line index with target: {target} or target: {target}/...
        for i, line in enumerate(lines):
            s = line.strip()
            if s.startswith("target:"):
                # extract the configured path after 'target:' and compare prefix
                tgt_val = s.split("target:", 1)[1].strip()
                if tgt_val == target or tgt_val.startswith(target.rstrip("/") + "/") or tgt_val.startswith(target + ""):
                    # If this is the storage target, update the container name
                    # portion of the target path (replace <container name> with
                    # the selected storage folder name).
                    if target == "/storage":
                        try:
                            container_name = os.path.basename(os.path.normpath(new_path))
                            indent_t = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
                            # replace any placeholder occurrence; if none, append
                            if "<container name>" in lines[i]:
                                lines[i] = lines[i].replace("<container name>", container_name)
                            else:
                                # if the target already has extra suffix, try to
                                # replace the trailing segment after /storage
                                parts = tgt_val.split("/storage", 1)
                                if len(parts) > 1 and parts[1]:
                                    lines[i] = f"{indent_t}target: /storage/{container_name}"
                                else:
                                    lines[i] = f"{indent_t}target: /storage/{container_name}"
                        except Exception:
                            # non-fatal: leave target line unchanged on errors
                            pass

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


def get_current_port():
    text = read_compose_text()
    lines = text.splitlines()
    # Find first port mapping like "8089:8089" and return (host_port, container_port, quote_char)
    m = None
    for line in lines:
        # strip leading/trailing spaces
        s = line.strip()
        # regex for optional quotes around mapping
        import re
        m = re.search(r'["\']?(\d+):(\d+)["\']?', s)
        if m:
            host = m.group(1)
            container = m.group(2)
            # determine if mapping used quotes
            quote = '"' if '"' in s or '"' in line else ('\'' if "'" in s else '')
            return host, container, quote
    # default
    return "8089", "8089", '"'


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

    return (
        storage,
        tmp,
        bui,
        anet_narrow,
        anet_wide,
        storage_label,
        tmp_label,
        bui_label,
        anet_narrow_label,
        anet_wide_label,
    )

def reset_compose_to_git():
    """Reset compose.yaml to HEAD version using git restore.

    This function runs 'git restore --staged' equivalent for the compose file
    by calling 'git restore <path>' in the project root. It returns (ok, msg).
    """
    try:
        # run git restore from PROJECT_ROOT
        p = str(COMPOSE_PATH)
        # Use subprocess; do not raise on non-zero so we can return message
        proc = subprocess.run(["git", "restore", p], cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            return False, proc.stderr.strip() or proc.stdout.strip()
        return True, "compose.yaml restored from git"
    except Exception as e:
        return False, str(e)


class ComposeEditorApp:
    def __init__(self, root):
        self.root = root
        root.title("AutoWISP Compose Editor")

        (
            storage,
            tmp,
            bui,
            anet_narrow,
            anet_wide,
            storage_label,
            tmp_label,
            bui_label,
            anet_narrow_label,
            anet_wide_label,
        ) = get_current_sources()
        host_port, container_port, quote = get_current_port()

        # Use labels extracted from the compose YAML placeholders when available
        tk.Label(root, text=storage_label).grid(row=0, column=0, sticky="w")
        self.storage_var = tk.StringVar(value=storage)
        self.storage_entry = tk.Entry(root, textvariable=self.storage_var, width=60)
        self.storage_entry.grid(row=0, column=1, padx=6, pady=6)
        # keep a reference to the storage browse button so we can enable/disable it
        self.storage_browse_btn = tk.Button(root, text="Browse...", command=self.browse_storage)
        self.storage_browse_btn.grid(row=0, column=2, padx=6)

        tk.Label(root, text=tmp_label).grid(row=1, column=0, sticky="w")
        self.tmp_var = tk.StringVar(value=tmp)
        self.tmp_entry = tk.Entry(root, textvariable=self.tmp_var, width=60)
        self.tmp_entry.grid(row=1, column=1, padx=6, pady=6)
        self.tmp_browse_btn = tk.Button(root, text="Browse...", command=self.browse_tmp)
        self.tmp_browse_btn.grid(row=1, column=2, padx=6)

        tk.Label(root, text=bui_label).grid(row=2, column=0, sticky="w")
        self.bui_var = tk.StringVar(value=bui)
        self.bui_entry = tk.Entry(root, textvariable=self.bui_var, width=60)
        self.bui_entry.grid(row=2, column=1, padx=6, pady=6)
        self.bui_browse_btn = tk.Button(root, text="Browse...", command=self.browse_bui)
        self.bui_browse_btn.grid(row=2, column=2, padx=6)

        tk.Label(root, text=anet_narrow_label).grid(row=3, column=0, sticky="w")
        self.anet_narrow_var = tk.StringVar(value=anet_narrow)
        self.anet_narrow_entry = tk.Entry(root, textvariable=self.anet_narrow_var, width=60)
        self.anet_narrow_entry.grid(row=3, column=1, padx=6, pady=6)
        self.anet_narrow_browse_btn = tk.Button(root, text="Browse...", command=self.browse_anet_narrow)
        self.anet_narrow_browse_btn.grid(row=3, column=2, padx=6)

        tk.Label(root, text=anet_wide_label).grid(row=4, column=0, sticky="w")
        self.anet_wide_var = tk.StringVar(value=anet_wide)
        self.anet_wide_entry = tk.Entry(root, textvariable=self.anet_wide_var, width=60)
        self.anet_wide_entry.grid(row=4, column=1, padx=6, pady=6)
        self.anet_wide_browse_btn = tk.Button(root, text="Browse...", command=self.browse_anet_wide)
        self.anet_wide_browse_btn.grid(row=4, column=2, padx=6)

        tk.Label(root, text="Port (host:container)").grid(row=5, column=0, sticky="w")
        self.port_var = tk.StringVar(value=host_port)
        self.port_entry = tk.Entry(root, textvariable=self.port_var, width=20)
        self.port_entry.grid(row=5, column=1, sticky="w", padx=6, pady=6)

        # Buttons: place inside a frame so layout is stable and won't disappear if window is resized
        button_frame = tk.Frame(root)
        button_frame.grid(row=6, column=0, columnspan=3, pady=8)

        # Action buttons (Preview/Apply removed — changes are saved immediately)
        self.reset_btn = tk.Button(button_frame, text="Reset compose.yaml (git restore)", command=self.reset_compose)
        self.reset_btn.grid(row=0, column=0, padx=6)
        self.run_btn = tk.Button(button_frame, text="Run 'docker compose up'", command=self.run_docker)
        self.run_btn.grid(row=0, column=1, padx=6)

        # Reduce height so the preview isn't excessively tall
        # Place the preview below the controls and buttons (row 7) so the
        # anet wide/narrow controls remain visible.
        self.preview_box = scrolledtext.ScrolledText(root, width=100, height=12, font=("Courier", 10))
        self.preview_box.grid(row=7, column=0, columnspan=3, padx=6, pady=6)

        # Enforce storage selection at startup (Option A): disable everything except storage
        # and force the user to pick a storage folder before proceeding.
        self.disable_all_except_storage()
        self.enforce_storage_at_startup()

    def browse_storage(self):
        p = filedialog.askdirectory(initialdir=self.storage_var.get() or os.getcwd(), title="Select storage folder")
        if p:
            # handle selection (may be called at startup or later)
            self.handle_storage_selected(p, show_info=True)

    def browse_tmp(self):
        p = filedialog.askdirectory(initialdir=self.tmp_var.get() or os.getcwd(), title="Select tmp folder")
        if p:
            self.tmp_var.set(p)

    def browse_bui(self):
        p = filedialog.askdirectory(initialdir=self.bui_var.get() or os.getcwd(), title="Select BUI folder")
        if p:
            self.bui_var.set(p)

    def browse_anet_narrow(self):
        p = filedialog.askdirectory(initialdir=self.anet_narrow_var.get() or os.getcwd(), title="Select anet narrow indices folder")
        if p:
            self.anet_narrow_var.set(p)

    def browse_anet_wide(self):
        p = filedialog.askdirectory(initialdir=self.anet_wide_var.get() or os.getcwd(), title="Select anet wide indices folder")
        if p:
            self.anet_wide_var.set(p)

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
        self.storage_var.set(p)

        # derive defaults
        tmp = os.path.join(p, "tmp")
        bui = os.path.join(p, "BUI")
        # Provide separate astrometry folders for narrow and wide indices
        anet_narrow = os.path.join(p, "astrometry", "narrow")
        anet_wide = os.path.join(p, "astrometry", "wide")

        # set variables
        self.tmp_var.set(tmp)
        self.bui_var.set(bui)
        self.anet_narrow_var.set(anet_narrow)
        self.anet_wide_var.set(anet_wide)

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
        # This writes changes as soon as the user selects folders (no Preview/Apply).
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
            with open(LOG_PATH, "a", encoding="utf-8") as lf:
                lf.write(f"{datetime.datetime.now().isoformat()} Applied changes from GUI\n")
            # update preview box to reflect saved changes
            try:
                self.preview_box.delete("1.0", tk.END)
                self.preview_box.insert(tk.END, "compose.yaml updated\n")
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save compose.yaml: {e}")

    def disable_all_except_storage(self):
        """Disable all entries/browse buttons and action buttons except the storage row."""
        # Entries
        self.tmp_entry.configure(state="disabled")
        self.bui_entry.configure(state="disabled")
        self.anet_narrow_entry.configure(state="disabled")
        self.anet_wide_entry.configure(state="disabled")

        # Browse buttons
        self.tmp_browse_btn.configure(state="disabled")
        self.bui_browse_btn.configure(state="disabled")
        self.anet_narrow_browse_btn.configure(state="disabled")
        self.anet_wide_browse_btn.configure(state="disabled")

        # Action buttons
        self.reset_btn.configure(state="disabled")
        self.run_btn.configure(state="disabled")

        # Keep storage entry and browse enabled
        self.storage_entry.configure(state="normal")
        self.storage_browse_btn.configure(state="normal")

    def enable_all_widgets(self):
        """Enable all previously-disabled widgets after storage selection."""
        self.tmp_entry.configure(state="normal")
        self.bui_entry.configure(state="normal")
        self.anet_narrow_entry.configure(state="normal")
        self.anet_wide_entry.configure(state="normal")

        self.tmp_browse_btn.configure(state="normal")
        self.bui_browse_btn.configure(state="normal")
        self.anet_narrow_browse_btn.configure(state="normal")
        self.anet_wide_browse_btn.configure(state="normal")

        self.reset_btn.configure(state="normal")
        self.run_btn.configure(state="normal")


    def preview(self):
        orig = read_compose_text().splitlines(keepends=True)
        new_text = find_and_replace_sources(
            read_compose_text(),
            self.storage_var.get(),
            self.tmp_var.get(),
            new_bui=self.bui_var.get() if hasattr(self, 'bui_var') else None,
            new_anet_narrow=self.anet_narrow_var.get() if hasattr(self, 'anet_narrow_var') else None,
            new_anet_wide=self.anet_wide_var.get() if hasattr(self, 'anet_wide_var') else None,
        )
        new_text = find_and_replace_port(new_text, self.port_var.get())
        new = new_text.splitlines(keepends=True)
        diff = difflib.unified_diff(orig, new, fromfile=str(COMPOSE_PATH), tofile=str(COMPOSE_PATH) + " (new)")
        self.preview_box.delete("1.0", tk.END)
        self.preview_box.insert(tk.END, "".join(diff) or "No changes\n")

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
            with open(LOG_PATH, "a", encoding="utf-8") as lf:
                lf.write(f"{datetime.datetime.now().isoformat()} Applied changes\n")
            messagebox.showinfo("Success", "compose.yaml updated")
            self.preview()
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

        try:
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
        except Exception:
            # On any error, fall back to keeping only storage enabled
            try:
                messagebox.showinfo("Welcome!", "Please select a storage folder when ready using the Storage Browse... button.")
            except Exception:
                pass
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
            timeout = 120  # seconds
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
                                # notify user and open browser on success
                                self.root.after(0, lambda u=url: messagebox.showinfo("Ready", f"Service is responding at {u} — opening browser"))
                                try:
                                    webbrowser.open(url)
                                except Exception:
                                    pass
                                return
                    except Exception:
                        # keep waiting
                        pass

                    # Update preview box with waiting status (throttle to avoid flooding)
                    if first_wait:
                        msg = f"Waiting for {url} to respond...\n"
                        self.root.after(0, lambda m=msg: self.preview_box.insert(tk.END, m))
                        first_wait = False
                    time.sleep(poll_interval)

                # timed out
                self.root.after(0, lambda u=url: messagebox.showwarning("Timeout", f"Timed out waiting for {u} to respond ({timeout}s)."))

            threading.Thread(target=_poll_and_open, daemon=True).start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start docker compose: {e}")

    def reset_compose(self):
        ok, msg = reset_compose_to_git()
        if ok:
            messagebox.showinfo("Reset", msg)
            # refresh GUI fields to reflect restored compose file
            (
                storage,
                tmp,
                bui,
                anet_narrow,
                anet_wide,
                storage_label,
                tmp_label,
                bui_label,
                anet_narrow_label,
                anet_wide_label,
            ) = get_current_sources()
            # update variables and labels
            self.storage_var.set(storage)
            self.tmp_var.set(tmp)
            if hasattr(self, 'bui_var'):
                self.bui_var.set(bui)
            if hasattr(self, 'anet_narrow_var'):
                self.anet_narrow_var.set(anet_narrow)
            if hasattr(self, 'anet_wide_var'):
                self.anet_wide_var.set(anet_wide)
        else:
            messagebox.showerror("Reset failed", msg)


def find_and_replace_port(text, new_host_port):
    # Replace the first mapping where container port exists (e.g., "8089:8089") with new_host_port:container_port preserving quotes
    import re
    lines = text.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        m = re.search(r'(["\']?)(\d+):(\d+)(["\']?)', s)
        if m:
            quote1 = m.group(1)
            host = m.group(2)
            container = m.group(3)
            quote2 = m.group(4)
            # build replacement preserving original quotes
            q = quote1 or quote2 or '"'
            lines[i] = line.replace(m.group(0), f'{q}{new_host_port}:{container}{q}')
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

