"""Define :class:`WalkFSView` allowing users to walk through the file system."""

import logging
import os
from os import path, scandir
import fnmatch
import re
import string

from django.views import View
from django.shortcuts import render


class WalkFSView(View):
    """Base class allowing user to walk through the file system."""

    _logger = logging.getLogger(__name__)

    _root_dir = [("Computer", "Computer")] if os.name == "nt" else [("/", "Computer")]

    template = 'core/walk_fs.html'
    url_name = None
    cancel_url_name = None

    def _get_context(self, config, search_dir):
        """Return the context required by the file system walk template."""

        result = {
            'url_name': self.url_name,
            'cancel_url_name': self.cancel_url_name,
        }
        filename_check = config.get("filename_filter", "[^.]")
        result["filename_filter"] = filename_check
        result["filename_filter_type"] = config.get(
            "filefilter_type", "Regular Expression"
        )
        if result["filename_filter_type"] != "Regular Expression":
            filename_check = fnmatch.translate(filename_check)
        try:
            filename_check = re.compile(filename_check)
        except re.error:
            filename_check = re.compile("")

        dirname_check = config.get("dirname_filter", "[^.]")
        result["dirname_filter"] = dirname_check
        result["dirname_filter_type"] = config.get(
            "dirfilter_type", "Regular Expression"
        )
        if result["dirname_filter_type"] != "Regular Expression":
            dirname_check = fnmatch.translate(dirname_check)
        try:
            dirname_check = re.compile(dirname_check)
        except re.error:
            print(f"Invalid REX: {dirname_check!r}")
            dirname_check = re.compile("")

        if search_dir is None:
            if os.name == "nt":
                current_dir = config.get("currentdir", "Computer")
                enter_dir = config.get("enter_dir")
                if enter_dir:
                    ent = str(enter_dir)
                    # If we are at the virtual root, clicking a drive enters that drive
                    if current_dir == "Computer":
                        # Normalize common drive inputs: "D", "D:", "D:\"
                        if re.fullmatch(r"[A-Za-z]:\\?", ent):
                            if not ent.endswith("\\"):
                                ent = ent + "\\"
                            search_dir = ent
                        elif path.isabs(ent):
                            search_dir = ent
                        else:
                            # Unexpected relative at root: keep at Computer
                            search_dir = "Computer"
                    else:
                        search_dir = ent if path.isabs(ent) else path.join(current_dir, ent)
                else:
                    search_dir = current_dir
            else:
                search_dir = config.get("currentdir", path.expanduser("~"))
                if "enter_dir" in config:
                    search_dir = path.join(search_dir, config["enter_dir"])
            result["currentdir"] = path.abspath(search_dir) if search_dir != "Computer" else "Computer"

        result["file_list"] = []
        result["dir_list"] = []

        if os.name == "nt" and search_dir == "Computer":
            # List available drives as "directories" under the virtual root
            for d in string.ascii_uppercase:
                drive_root = f"{d}:\\"
                if os.path.exists(drive_root):
                    result["dir_list"].append(drive_root)
        else:
            with scandir(search_dir) as dir_entries:
                for entry in dir_entries:
                    if entry.is_dir():
                        if dirname_check.match(entry.name):
                            result["dir_list"].append(entry.name)
                    elif filename_check.match(entry.name):
                        result["file_list"].append(entry.name)

        result["file_list"].sort()
        result["dir_list"].sort()

        if os.name == "nt":
            if search_dir == "Computer":
                parent_dir_list = self._root_dir[:]
            else:
                head = path.abspath(search_dir)
                parent_dir_list = self._root_dir[:]
                while True:
                    drive, tail = path.splitdrive(head)
                    # At drive root? Stop after adding it once.
                    if drive and (tail == "" or tail == "\\"):
                        parent_dir_list.insert(1, (f"{drive}\\", f"{drive[0]} Drive"))
                        break
                    parent_dir_list.insert(1, (head, path.basename(head)))
                    new_head = path.dirname(head)
                    if new_head == head:  # safety guard
                        break
                    head = new_head
        else:
            head = path.abspath(search_dir)
            #        parent_dir_list = [('/', 'Computer')]
            parent_dir_list = self._root_dir[:]
            #        while head and head != '/':
            while head and head not in [root[0] for root in self._root_dir]:
                parent_dir_list.insert(1, (head, path.basename(head)))
                head = path.dirname(head)

        result["parent_dir_list"] = parent_dir_list

        self._logger.debug("Context: %s", repr(result))
        return result

    def get(self, request, dirname=None):
        """Display the interface for selecting files."""

        return render(
            request,
            self.template,
            self._get_context(request.GET, dirname),
        )
