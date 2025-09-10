"""Define the view for creating new AutoWISP projects."""

import os
from argparse import Namespace
import re
import json
import copy

from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect

from autowisp.database.interface import set_sqlite_database
from autowisp.database.initialize_database import initialize_database
from autowisp import database
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

        ``"change_master_usage"``: Toggle the
    """

    db_fname = "autowisp.db"
    """The base filename of the SQLite database tracking all projects."""

    def _get_context(self, config, search_dir):
        """Return the context required by the home selection template."""

        context = super()._get_context(config, search_dir)
        context["unselectable"] = context.pop("file_list")
        context["file_list"] = []
        currentdir = context["parent_dir_list"][-1][0]
        if os.path.exists(os.path.join(currentdir, self.db_fname)):
            context["invalid_home_message"] = (
                f"Directory {currentdir} already appears to contain an AutoWISP"
                " project."
            )
            context["valid_home"] = False
        else:
            context["invalid_home_message"] = "valid home"
            context["valid_home"] = True
        return context

    @staticmethod
    def _get_path_overwrites(root_dir, config):
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
                        "mphotref_{TARGETID}_{CLRCHNL}_{EXPTIME}sec"
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
                        "mfit_stat_{TARGETID}_{CLRCHNL}_{EXPTIME}sec"
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
                        "{processing_step:s}_{task:s}_{now:s}_pid{pid:d}.log",
                    ),
                )
            ],
            "epd-statistics-fname": [
                (
                    None,
                    os.path.join(
                        root_dir,
                        "MASTERS",
                        "mphotref_{TARGETID}_{CLRCHNL}_{EXPTIME}sec.txt",
                    ),
                )
            ],
            "tfa-statistics-fname": [
                (
                    None,
                    os.path.join(
                        root_dir,
                        "MASTERS",
                        "mphotref_{TARGETID}_{CLRCHNL}_{EXPTIME}sec.txt",
                    ),
                )
            ],
        }

        if int(config["master-zero-enabled"]) or int(
            config["master-dark-enabled"]
        ):
            result["stacked-master-fname"] = [
                (
                    None,
                    os.path.join(
                        root_dir,
                        "MASTERS",
                        "{IMAGETYP}_{OBSSSNID}_{CLRCHNL}.fits.fz",
                    ),
                )
            ]

        if int(config["master-flat-enabled"]):
            result["high-flat-master-fname"] = [
                (
                    None,
                    os.path.join(
                        root_dir,
                        "MASTERS",
                        "{IMAGETYP}_{OBSSSNID}_{CLRCHNL}.fits.fz",
                    ),
                )
            ]
            result["low-flat-master-fname"] = [
                (
                    None,
                    os.path.join(
                        root_dir,
                        "MASTERS",
                        "low{IMAGETYP}_{OBSSSNID}_{CLRCHNL}.fits.fz",
                    ),
                )
            ]

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

    def _get_steps_and_masters(self, config):
        """Set the processing sequence and masters per configuration."""

        step_dependencies = copy.deepcopy(database.defaults.step_dependencies)
        master_info = copy.deepcopy(database.defaults.master_info)

        disabled_masters = [
            master_type
            for master_type in ["zero", "dark", "flat"]
            if int(config[f"master-{master_type}-enabled"]) == 0
        ]
        for i in range(len(step_dependencies) - 1, -1, -1):
            if step_dependencies[i][0] == "calibrate":
                if step_dependencies[i][1] in disabled_masters:
                    del step_dependencies[i]
                else:
                    assert step_dependencies[i][1] in [
                        "zero",
                        "dark",
                        "flat",
                        "object",
                    ]

                    for master_type in disabled_masters:
                        try:
                            step_dependencies[i][2].remove(
                                (
                                    "stack_to_master"
                                    + (
                                        "_flat" if master_type == "flat" else ""
                                    ),
                                    master_type,
                                )
                            )
                        except ValueError:
                            pass
            elif (
                step_dependencies[i][0].startswith("stack_to_master")
                and step_dependencies[i][1] in disabled_masters
            ):
                del step_dependencies[i]

        print("step_dependencies:", step_dependencies)

        for master_type in disabled_masters:
            if master_type == "flat":
                del master_info["highflat"]
                del master_info["lowflat"]
            else:
                del master_info[master_type]

        for master_type, master_config in master_info.items():
            if master_type in ["highflat", "lowflat"]:
                master_type = "flat"
            enabled = config[f"master-{master_type}-enabled"]
            assert enabled == "always" or int(enabled) == 1
            master_config["must_match"] = frozenset(
                filter(None, config.getlist(f"master-{master_type}-match"))
            )
            master_config["split_by"] = frozenset(
                filter(None, config.getlist(f"master-{master_type}-split"))
            )

        return step_dependencies, master_info

    def _create_project(self, config):
        """Create a new project following the given configuration."""

        db_fname = os.path.join(config["project-home"], self.db_fname)
        assert not os.path.exists(db_fname), (
            f"Directory {config['project-home']} appears to already contain a "
            "project."
        )

        proj = Project(
            name=config["project-name"],
            path=config["project-home"],
            description=config["project-description"],
        )
        proj.save()
        set_sqlite_database(db_fname)
        overwrites = {}

        config_rex = re.compile(
            r"^(?P<key>[^:=;#\s]+)\s*"
            r'(?:(?P<equal>[:=\s])\s*([\'"]?)(?P<value>.+?)?\3)?'
            r"\s*(?:\s[;#]\s*(?P<comment>.*?)\s*)?$"
        )

        for line in config["custom-config"].splitlines():
            parsed = config_rex.match(line)
            overwrites[parsed.group("key")] = [(None, parsed.group("value"))]
        overwrites.update(self._get_path_overwrites(proj.path, config))
        initialize_database(
            Namespace(drop_hdf5_structure_tables=False, drop_all_tables=True),
            *self._get_steps_and_masters(config),
            overwrites,
        )

    def _save_form(self, request):
        """Save the current state of the form to the session."""

        for key in [
            "project-name",
            "project-description",
            "project-home",
            "custom-config",
        ]:
            request.session[key] = request.POST.get(key, "")

        for master_type in database.defaults.master_info:
            if master_type == "lowflat":
                continue
            if master_type == "highflat":
                master_type = "flat"
            for param in ["enabled", "split", "match"]:
                key = f"master-{master_type}-{param}"
                request.session[key] = (
                    request.POST[key]
                    if param == "enabled"
                    else list(filter(None, request.POST.getlist(key)))
                )

    def _load_master_config(self, request):
        """Load master configuration from user-supplied JSON file."""

        config = json.load(request.FILES["import-master-config"])

        for master_type in database.defaults.master_info:
            if master_type in ["highflat", "lowflat"]:
                master_type = "flat"
            for param in ["enabled", "split", "match"]:
                request.session[f"master-{master_type}-{param}"] = config[
                    master_type
                ][param]

    def get(self, request, dirname=None):
        """
        Display the appropriate project cretion page per the current mode.

        The expected arguments depend on the mode:

        Args:
            dirname (str, optional): Directory name to display contents of
                when selecting project home or where new directory or new
                project will be created.
        """

        def get_master_usage(used_by):
            """Return usage string for master with given "used_by" entries."""

            if not used_by:
                return "output only"
            if used_by[0][2]:
                return "optional"
            return "required"

        def get_master_info(master_type, master_config):
            """Return the entry in context to add for the given master type."""

            if master_type == "highflat":
                master_type = "flat"
            return (
                master_type,
                str(
                    request.session.get(
                        f"master-{master_type}-enabled",
                        (
                            1
                            if master_type in ["zero", "dark", "flat"]
                            else "always"
                        ),
                    )
                ),
                get_master_usage(master_config["used_by"]),
                request.session.get(
                    f"master-{master_type}-split",
                    master_config["split_by"],
                ),
                request.session.get(
                    f"master-{master_type}-match",
                    master_config["must_match"],
                ),
            )

        print(f"Mode: {self.mode!r}, dirname: {dirname!r}")
        if self.mode == "create_dir":
            print(f"Creating directory under {dirname!r}")
            context = self._get_context(request.GET, dirname)
            print(f"Context: {context!r}")
            return render(request, self.template, context)

        if self.mode == "create_project":
            print("Session:")
            for key, value in request.session.items():
                print(f"\t{key} => {value}")
            print(
                f"Create project {request.session.get('project-name', '')} in "
                f"{request.session.get('project-home', '')!r}"
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
                    "config": request.session.get("custom-config", ""),
                    "master_info": [
                        get_master_info(master_type, master_config)
                        for (
                            master_type,
                            master_config,
                        ) in database.defaults.master_info.items()
                        if master_type != "lowflat"
                    ],
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

        if "create-project-submit" in request.POST:
            print(f"Creating project from {request.POST}")
            self._create_project(request.POST)
            return redirect("home:home")

        if "set-project-home" in request.POST:
            print(f"Setting project home from {request.POST}")
            request.session["project-home"] = request.POST["currentdir"]
            return redirect("home:new_project")

        if "import-master-config" in getattr(request, "FILES", []):
            self._save_form(request)
            self._load_master_config(request)
            return redirect("home:new_project")

        if "redirect" in request.POST:
            self._save_form(request)
            return HttpResponseRedirect(request.POST["redirect"])
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


class MasterConfigView(CreateProjectView):
    """Handle master configuration part of the project creation view."""

    def get(  # pylint: disable=arguments-renamed
        self, request, master_type, **kwargs
    ):
        """
        Handle a change to the master configuration by the user.

        If kwargs is empty, the master is toggled between enabled and
        disabled. Otherwise, it should contain a single key (``"split"`` or
        ``"match"``) with a value a single tuple specifying the change:
        ``(old expression, new expression)``. If old expression is None, the
        corresponding list is extended.
        """

        if not kwargs:
            assert master_type in ["zero", "dark", "flat"]
            key = f"master-{master_type}-enabled"
            request.session[key] = 1 - int(request.session[key])

        return super().get(request)
