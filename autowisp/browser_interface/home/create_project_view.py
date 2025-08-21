"""Define the view for creating new AutoWISP projects."""

import os

from django.shortcuts import render, redirect

from autowisp.browser_interface.core.walk_fs_view import WalkFSView
from .models import Project


class CreateProjectView(WalkFSView):
    """View to create a new AutoWISP project."""

    template = "home/select_project_home.html"
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

    def get(self, request, dirname=None):
        """
        Display appropriate project cretion page.

        Args:
            dirname (str, optional): Directory name to display contents of when
                selecting project home.

            project_home (str, optional): Directory to create a new project in.
        """

        if self.mode == "create_dir":
            context = self._get_context(request.GET, dirname)
            return render(request, "home/create_directory.html", context)

        if self.mode == "create_project":
            print(f"Creating project in: {dirname}")

        # if create_dir:
        #    dirname = os.path.join(dirname, create_dir)
        #    os.makedirs(dirname, exist_ok=False)

        return super().get(request, dirname=dirname)

    def post(self, request, *_args, **_kwargs):
        """
        Handle POST request to create a new directory.

        Args:
            request: The HTTP request object.
        """

        if self.mode == "select_home":
            proj = Project(name="New Project", path=request.POST['currentdir'])
            proj.save()
            return redirect("home:home")

        new_dir = os.path.join(
            request.POST["currentdir"], request.POST["create-dir"]
        )
        if new_dir != request.POST["currentdir"]:
            os.makedirs(new_dir, exist_ok=False)

        return redirect("home:new_project", dirname=new_dir)
