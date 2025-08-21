"""Define the view for creating new AutoWISP projects."""

import os
from argparse import Namespace

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

from django.shortcuts import render, redirect

from autowisp.database import interface as db_interface
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
            db_interface.db_engine = create_engine(
                (
                    "sqlite:///"
                    + os.path.join(proj.path, "autowisp.db")
                    + "?timeout=100&uri=true"
                ),
                echo=False,
                pool_pre_ping=True,
                pool_recycle=3600,
                poolclass=NullPool,
            )
            db_interface.Session = sessionmaker(
                db_interface.db_engine, expire_on_commit=False
            )
            initialize_database(
                Namespace(
                    drop_hdf5_structure_tables=False, drop_all_tables=True
                )
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
