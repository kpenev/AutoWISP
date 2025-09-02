"""Define the view allowing users to add new raw images for processing."""

import logging
from os import path
from argparse import Namespace

from django.http import HttpResponseRedirect
from django.urls import reverse

from autowisp.browser_interface.core.walk_fs_view import WalkFSView
from autowisp.file_utilities import find_fits_fnames
from autowisp.run_pipeline import main as run_pipeline


class SelectRawImages(WalkFSView):
    """A view for selecting raw images to add for processing."""

    _logger = logging.getLogger(__name__)

    template = "processing/select_raw_images.html"
    url_name = "processing:select_raw_images"
    cancel_url_name = "processing:progress"

    def _get_context(self, config, search_dir):
        """Return te context required by the file selection template."""

        self._logger.debug("Config: %s", repr(config))
        config = config.copy()
        if "filename_filter" not in config:
            config["filename_filter"] = r".*\.fits(.fz)?\Z"
        if "dirname_filter" not in config:
            config["dirname_filter"] = r"[^.]"

        return super()._get_context(config, search_dir)

    def post(self, request, *_args, **_kwargs):
        """Respond to user changing file selection configuration."""

        self._logger.debug("POST: %s", repr(request.POST))

        dir_name = request.POST["currentdir"]
        image_list = []
        # changed this line to use getList to handle multiple selections
        selected = request.POST.getlist("selected")

        for item_name in selected:
            full_path = path.join(dir_name, item_name)
            if path.isdir(full_path):
                self._logger.debug("Adding images under: %s", repr(full_path))
                image_list.extend(find_fits_fnames(full_path))
            else:
                self._logger.debug("Adding single image: %s", repr(full_path))
                assert path.isfile(full_path)
                image_list.append(full_path)

        try:
            run_pipeline(
                Namespace(
                    processing_database=request.session["project_db_path"],
                    add_raw_images=image_list,
                    steps=[],
                )
            )
        except OSError:
            self._logger.error(
                "OSError occurred while adding raw images", exc_info=True
            )
            return HttpResponseRedirect(reverse("processing:select_raw_images"))

        return HttpResponseRedirect(reverse("processing:progress"))
