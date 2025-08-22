"""Define the view for creating new AutoWISP projects."""

import os
from argparse import Namespace

from django.shortcuts import render, redirect

from autowisp.database.interface import set_sqlite_database
from autowisp.database.initialize_database import initialize_database
from autowisp.browser_interface.core.walk_fs_view import WalkFSView
from .models import Project


class CreateProjectView(WalkFSView):
    """View to create a new AutoWISP project."""

    template = "home/create_project.html"
    """The template used to display the project creation page."""

    url_name = "home:new_project"
    """The URL name for this view."""

    cancel_url_name = "home:new_project"
    """The URL name to redirect to when the cancel button is pressed."""

    mode = "create_project"
    """
    What mode this view is in.

    Possible values:
        ``"select_home"``: Display the project home director selection page.

        ``"create_dir"``: Allow specifying name of directory to create.

        ``"create_project"``: Create a new project in the specified directory.
    """

    def _get_context(self, config, search_dir):
        """Return the context required by the project creation template."""

        context = super()._get_context(config, search_dir)
        context["unselectable"] = context.pop("file_list")
        context["file_list"] = []
        return context

    @staticmethod
    def _get_path_overwrites(root_dir):
        """Return the config overwrites to place outputs under given root."""

        result = {
            "calibrated-fname": [
                (
                    None,
                    os.path.join(
                        root_dir, "CAL", "{RAWFNAME}_{CLRCHNL}.fits.fz"
                    ),
                )
            ],
            "data-reduction-fname": [
                (None, os.path.join(root_dir, "DR", "{RAWFNAME}_{CLRCHNL}.h5"))
            ],
            "master-photref-fname-format": [
                (
                    None,
                    os.path.join(
                        root_dir,
                        "MASTERS",
                        "mphotref_{TARGET}_{CLRCHNL}_{EXPTIME}sec"
                        "_iter{magfit_iteration:03d}.fits",
                    ),
                )
            ],
            "magfit-stat-fname-format": [
                (
                    None,
                    os.path.join(
                        root_dir,
                        "MASTERS",
                        "mfit_stat_{TARGET}_{CLRCHNL}_{EXPTIME}sec"
                        "_iter{magfit_iteration:03d}.txt",
                    ),
                )
            ],
            "lightcurve-catalog-fname": [
                (
                    None,
                    os.path.join(
                        root_dir,
                        "MASTERS",
                        "lc_catalog_{OBJECT}_{CLRCHNL}_{EXPTIME}.fits",
                    ),
                )
            ],
            "lc-fname": [(None, os.path.join(root_dir, "LC", "GDR3_{:d}.h5"))],
            "std-out-err-fname": [
                (
                    None,
                    os.path.join(
                        root_dir,
                        "LOGS",
                        "{processing_step:s}_{task:s}_{now:s}"
                        "_pid{pid:d}.outerr",
                    ),
                )
            ],
            "logging-fname": [
                (
                    None,
                    os.path.join(
                        root_dir,
                        "LOGS",
                        "{processing_step:s}_{task:s}_{now:s}"
                        "_pid{pid:d}.outerr",
                    ),
                )
            ],
            "stacked-master-fname": [
                (
                    None,
                    os.path.join(
                        root_dir,
                        "MASTERS",
                        "{IMAGETYP}_{OBS_SESN}_{CLRCHNL}.fits.fz",
                    ),
                )
            ],
            "high-flat-master-fname": [
                (
                    None,
                    os.path.join(
                        root_dir,
                        "MASTERS",
                        "{IMAGE_TYPE}_{OBS_SESN}_{CLRCHNL}.fits.fz",
                    ),
                )
            ],
            "low-flat-master-fname": [
                (
                    None,
                    os.path.join(
                        root_dir,
                        "MASTERS",
                        "low{IMAGE_TYPE}_{OBS_SESN}_{CLRCHNL}.fits.fz",
                    ),
                )
            ],
        }

        for cat_type in ["astrometry", "photometry", "magfit"]:
            result[f"{cat_type}-catalog"] = [
                (
                    None,
                    os.path.join(
                        root_dir, "MASTERS", "Gaia", "{checksum:s}.fits"
                    ),
                )
            ]
        return result

    def _create_project(self, config):
        """Create a new project following the given configuration."""

        proj = Project(
            name=config["project-name"],
            path=config["project-home"],
            description=config["project-description"],
        )
        proj.save()
        set_sqlite_database(os.path.join(proj.path, "autowisp.db"))
        initialize_database(
            Namespace(drop_hdf5_structure_tables=False, drop_all_tables=True),
            self._get_path_overwrites(proj.path),
        )

    def get(self, request, dirname=None):
        """
        Display the appropriate project cretion page per the current mode.

        The expected arguments depend on the mode:

        Args:
            dirname (str, optional): Directory name to display contents of
                when selecting project home or where new directory or new
                project will be created.
        """

        print(f"Mode: {self.mode!r}, dirname: {dirname!r}")
        if self.mode == "create_dir":
            print(f"Creating directory under {dirname!r}")
            context = self._get_context(request.GET, dirname)
            print(f"Context: {context!r}")
            return render(request, self.template, context)

        if self.mode == "create_project":
            print(f"Session: {request.session}")
            print(
                f"Create project in {request.session.get('project-home', '')!r}"
            )
            return render(
                request,
                "home/create_project.html",
                {
                    "path": request.session.get("project-home", ""),
                    "name": request.session.get("project-name", ""),
                    "description": request.session.get(
                        "project-description", ""
                    ),
                },
            )

        assert self.mode == "select_home", f"Invalid mode {self.mode!r}"
        return super().get(request, dirname=dirname)

    def post(self, request, *_args, **_kwargs):
        """
        Handle POST request to create a new directory.

        Args:
            request: The HTTP request object.
        """

        print(f"Received POST request {request!r}: {request.POST!r}")

        if "create-project" in request.POST:
            print(f"Creating project from {request.POST}")
            self._create_project(request.POST)
            return redirect("home:home")

        if "set-project-home" in request.POST:
            print(f"Setting project home from {request.POST}")
            request.session["project-home"] = request.POST["currentdir"]
            return redirect("home:new_project")

        request.session["project-name"] = request.POST.get("project-name", "")
        request.session["project-description"] = request.POST.get(
            "project-description", ""
        )
        if "create-dir" in request.POST:
            new_dir = os.path.join(
                request.POST["currentdir"], request.POST["create-dir"]
            )
            try:
                os.mkdir(new_dir)
            except OSError:
                print(f"Failed to create directory {new_dir!r}")
                return redirect(
                    "home:create_directory", dirname=request.POST["currentdir"]
                )
        else:
            new_dir = request.POST["currentdir"]

        return redirect("home:select_project_home", dirname=new_dir)
