"""Define :class:`WalkFSView` allowing users to walk through the file system."""

import logging
import os
from os import path, scandir
import fnmatch
import re

from django.views import View
from django.shortcuts import render


class WalkFSView(View):
    """Base class allowing user to walk through the file system."""

    _logger = logging.getLogger(__name__)

    _root_dir = (
        [
            (drive + ":\\", f"{drive} Drive")
            for drive in "CDEFGHI"
            if os.path.exists(f"{drive}:\\")
        ]
        if os.name == "nt"
        else [("/", "Computer")]
    )

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
            search_dir = config.get("currentdir", path.expanduser("~"))
            if "enter_dir" in config:
                search_dir = path.join(search_dir, config["enter_dir"])
        result["file_list"] = []
        result["dir_list"] = []
        with scandir(search_dir) as dir_entries:
            for entry in dir_entries:
                if entry.is_dir():
                    if dirname_check.match(entry.name):
                        result["dir_list"].append(entry.name)
                elif filename_check.match(entry.name):
                    result["file_list"].append(entry.name)

        result["file_list"].sort()
        result["dir_list"].sort()

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
