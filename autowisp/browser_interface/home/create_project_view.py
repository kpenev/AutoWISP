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

    cancel_url_name = "home:home"
    """The URL name to redirect to when the cancel button is pressed."""

    mode = "select_home"
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

    def get(self, request, **kwargs):  # pylint: disable=arguments-differ
        """
        Display the appropriate project cretion page per the current mode.

        The expected arguments depend on the mode:

        For ``select_home`` mode:

            Args:
                dirname (str, optional): Directory name to display contents of
                    when selecting project home.

        For ``create_dir`` mode:

            Args:
                dirname (str, optional): Directory under which the new directory
                    will be created.

        For ``create_project`` mode:

            Args:
                dirname (str): The currently selected home directory where the
                    new project will be created.

                name(str, optional): The currently specified name of the new
                    project to create.

                description(str, optional): The currently specified description
                    of the new project to create.
        """

        if self.mode == "create_dir":
            context = self._get_context(request.GET, kwargs["dirname"])
            return render(request, "home/create_directory.html", context)

        if self.mode == "create_project":
            return render(
                request,
                "home/create_project.html",
                {
                    "properties": [
                        ("path", kwargs["dirname"]),
                        ("name", kwargs.get("name", "")),
                        ("description", kwargs.get("description", "")),
                    ]
                },
            )

        return super().get(request, dirname=kwargs.get("dirname", None))

    def post(self, request, *_args, **_kwargs):
        """
        Handle POST request to create a new directory.

        Args:
            request: The HTTP request object.
        """

        print(f"Receide POST request {request!r}: {request.POST!r}")

        if "create-project" in request.POST:
            print(f"Creating project from {request.POST}")
            proj = Project(
                name=request.POST["project-name"],
                path=request.POST["currentdir"],
                description=request.POST["project-description"],
            )
            proj.save()
            set_sqlite_database(os.path.join(proj.path, "autowisp.db"))
            initialize_database(
                Namespace(
                    drop_hdf5_structure_tables=False, drop_all_tables=True
                ),
                self._get_path_overwrites(proj.path)
            )

            return redirect("home:home")

        if "create-dir" in request.POST:
            new_dir = os.path.join(
                request.POST["currentdir"], request.POST["create-dir"]
            )
            try:
                os.mkdir(new_dir)
            except OSError:
                return redirect(
                    "home:create_directory", dirname=request.POST["currentdir"]
                )
        else:
            new_dir = request.POST["currentdir"]

        return redirect("home:new_project", dirname=new_dir)
