"""Define the vies available on the home page."""

import json
import logging
import os
import re
from glob import iglob
from io import StringIO

import numpy
from django.shortcuts import render, redirect, HttpResponse
from sqlalchemy import select

from autowisp.catalog import read_catalog_file
from autowisp.database.defaults import master_info
from autowisp.database.interface import (
    set_project_home,
    start_db_session,
    get_db_engine,
    DB_URL_FNAME,
)
from autowisp.error_persistence import delete_all_error_sidecars
from autowisp.database.data_model.base import DataModelBase
from autowisp.database.data_model import (  # pylint: disable=no-name-in-module
    Configuration,
    Image,
    MasterFile,
    MasterType,
    Parameter,
    PipelineRun,
)
from autowisp.database.data_model.provenance.camera_channel import (
    CameraChannel,
)
from autowisp.fits_utilities import get_primary_header
from autowisp.light_curves.collect_light_curves import DecodingStringFormatter

from .create_project_view import (  # pylint: disable=unused-import
    CreateProjectView,
    MasterConfigView,
)
from .models import Project

logger = logging.getLogger(__name__)


def home(request):
    """Display the home page."""

    display_columns = [
        field.name
        for field in Project._meta.get_fields()  # pylint: disable=no-member, protected-access
        if field.name != "id"
    ]
    print(f"Projects: {Project.objects.all()}")  # pylint: disable=no-member
    missing_db_ids = {proj.id for proj in find_missing_databases()}
    context = {
        "columns": display_columns,
        "projects": {
            proj.id: [getattr(proj, col) for col in display_columns]
            for proj in Project.objects.all()  # pylint: disable=no-member
        },
        "missing_db_ids": missing_db_ids,
    }
    print(f"Context: {context!r}")  # Debugging output
    return render(request, "home/index.html", context)


def confirm_delete_projects(request):
    """Show a confirmation page listing projects and file-type checkboxes."""

    file_type_choices = [
        ("lightcurves", "Light curves"),
        ("data_reductions", "Data reduction files"),
        ("calibrated_images", "Calibrated images"),
        ("master_files", "Master files"),
        ("logs", "Log and output files"),
    ]

    project_ids = request.GET.getlist("project_ids")
    projects = Project.objects.filter(  # pylint: disable=no-member
        id__in=project_ids
    )
    context = {
        "project_ids": project_ids,
        "project_names": [p.name for p in projects],
        "file_type_choices": file_type_choices,
    }
    return render(request, "home/confirm_delete.html", context)


def _safe_remove(fpath, project_home):
    """Delete *fpath* after verifying it resides under *project_home*.

    Args:
        fpath(str):         Path to the file to delete.

        project_home(str):  Absolute path to the project home directory.

    Raises:
        ValueError:  If *fpath* is not under *project_home*.
    """

    real_home = os.path.realpath(project_home)
    if not os.path.isabs(fpath):
        fpath = os.path.join(real_home, fpath)
    real_fpath = os.path.realpath(fpath)
    if not real_fpath.startswith(real_home + os.sep):
        raise ValueError(
            f"Refusing to delete {fpath!r}: not under project home "
            f"{project_home!r}"
        )
    os.remove(fpath)


def delete_projects(request):
    """Delete selected projects and optionally their generated files."""

    project_ids = request.GET.getlist("project_ids")
    file_types = set(request.GET.getlist("file_types"))

    projects = Project.objects.filter(  # pylint: disable=no-member
        id__in=project_ids
    )
    missing_db_ids = {proj.id for proj in find_missing_databases()}
    for project in projects:
        project_home = project.path

        if project.id in missing_db_ids:
            logger.info(
                "Skipping file deletion for %s: database missing",
                project.name,
            )
            continue

        if "lightcurves" in file_types:
            delete_lightcurves(project_home)
        if "data_reductions" in file_types or "calibrated_images" in file_types:
            delete_image_products(
                project_home,
                data_reductions="data_reductions" in file_types,
                calibrated_images="calibrated_images" in file_types,
            )
        if "master_files" in file_types:
            delete_master_files(project_home)
        if "logs" in file_types:
            delete_logs(project_home)

        set_project_home(project_home)
        # The error sidecars are bound to the Error table dropped below, so
        # remove them before the table is gone. prune_empty_directories
        # (at the end) clears the emptied errors directories.
        delete_all_error_sidecars()
        DataModelBase.metadata.drop_all(get_db_engine())

        for db_file in ("autowisp.db", DB_URL_FNAME):
            db_file_path = os.path.join(project.path, db_file)
            if os.path.exists(db_file_path):
                _safe_remove(db_file_path, project_home)

        # The project is being removed entirely (DB + error sidecars
        # always go), so clear any directories left empty -- including the
        # emptied errors directory -- regardless of which file types were
        # selected. Only empty directories are removed, so kept data files
        # keep their directories.
        prune_empty_directories(project_home)

    projects.delete()
    request.session.flush()
    return redirect("home:home")


def delete_lightcurves(project_home):
    """Delete all lightcurve files generated by the project at *project_home*.

    Uses the lightcurve catalog(s) stored as ``MasterFile`` entries of type
    ``lightcurve_catalog`` to identify which individual lightcurve files were
    created, then removes them together with the catalog files themselves.

    The ``lc-fname`` configuration parameter is read from the project database
    to determine the filename pattern used when the lightcurves were created.

    Args:
        project_home(str):  Absolute path to the project home directory.
    """

    set_project_home(project_home)
    srcid_formatter = DecodingStringFormatter()

    with start_db_session() as db_session:
        lc_fname_pattern = db_session.scalar(
            select(Configuration.value)  # pylint: disable=no-member
            .join(Parameter)
            .where(Parameter.name == "lc-fname")
        )
        if lc_fname_pattern is None:
            logger.info("No lc-fname configuration found for %s", project_home)
            return

        catalog_fnames = db_session.scalars(
            select(MasterFile.filename)
            .join(MasterType)
            .where(MasterType.name == "lightcurve_catalog")
        ).all()

    for catalog_fname in catalog_fnames:
        if not os.path.exists(catalog_fname):
            logger.warning("Catalog file missing: %s", catalog_fname)
            continue

        catalog = read_catalog_file(catalog_fname)
        for source_id in catalog.index:
            lc_fname = srcid_formatter.format(
                lc_fname_pattern,
                *numpy.atleast_1d(source_id),
                PROJHOME=project_home,
            )
            if os.path.exists(lc_fname):
                _safe_remove(lc_fname, project_home)
                logger.info("Deleted lightcurve file: %s", lc_fname)

        _safe_remove(catalog_fname, project_home)
        logger.info("Deleted lightcurve catalog: %s", catalog_fname)


def delete_image_products(
    project_home, data_reductions=True, calibrated_images=True
):
    """Delete data reduction files and/or calibrated images for a project.

    Reads the ``data-reduction-fname`` and/or ``calibrated-fname``
    configuration parameters from the project database, then iterates over
    every ``Image``, reads its FITS header to resolve the templates, and
    removes the corresponding files.

    Args:
        project_home(str):  Absolute path to the project home directory.

        data_reductions(bool):  If True, delete data reduction files.

        calibrated_images(bool):  If True, delete calibrated image files.
    """

    if not data_reductions and not calibrated_images:
        return

    set_project_home(project_home)

    patterns = {}
    with start_db_session() as db_session:
        if data_reductions:
            patterns["data reduction"] = db_session.scalar(
                select(Configuration.value)  # pylint: disable=no-member
                .join(Parameter)
                .where(Parameter.name == "data-reduction-fname")
            )
        if calibrated_images:
            patterns["calibrated image"] = db_session.scalar(
                select(Configuration.value)  # pylint: disable=no-member
                .join(Parameter)
                .where(Parameter.name == "calibrated-fname")
            )

        patterns = {
            kind: pat for kind, pat in patterns.items() if pat is not None
        }
        if not patterns:
            logger.info("No filename configuration found for %s", project_home)
            return

        raw_fnames = db_session.scalars(
            select(Image.raw_fname)  # pylint: disable=no-member
        ).all()

        channel_names = db_session.scalars(
            select(CameraChannel.name).distinct()
        ).all()

    for raw_fname in raw_fnames:
        if not os.path.exists(raw_fname):
            logger.warning("Raw FITS file missing, skipping: %s", raw_fname)
            continue

        header = get_primary_header(raw_fname)
        header["PROJHOME"] = project_home
        base_fname = os.path.basename(raw_fname)
        for ext in [".fz", ".fits"]:
            if base_fname.endswith(ext):
                base_fname = base_fname[: -len(ext)]
        header["RAWFNAME"] = base_fname

        for kind, pattern in patterns.items():
            for channel_name in channel_names:
                header["CLRCHNL"] = channel_name
                try:
                    product_fname = pattern.format_map(header)
                except KeyError:
                    logger.warning(
                        "Raw FITS header missing keyword required to "
                        "find %s, skipping %s channel %s",
                        kind,
                        raw_fname,
                        channel_name,
                    )
                    continue
                if os.path.exists(product_fname):
                    _safe_remove(product_fname, project_home)
                    logger.info("Deleted %s file: %s", kind, product_fname)


def _pattern_to_glob(pattern, known_values, project_home):
    """
    Convert a filename pattern to glob by substituting unknown keys with ``*``.

    Known keys are replaced with their values from *known_values*; all
    remaining ``{key}`` or ``{key:spec}`` placeholders are replaced with
    ``*``.

    Args:
        pattern(str):       A Python format string with named placeholders.

        known_values(dict): Mapping of placeholder names to their values.

    Returns:
        str:  A glob pattern suitable for :func:`glob.iglob`.
    """

    def _replace(match):
        key = match.group("key")
        if key in known_values:
            return str(known_values[key])
        return "*"

    pattern = re.sub(r"\{(?P<key>\w+)(?::[^}]*)?\}", _replace, pattern)
    if os.path.isabs(pattern):
        return pattern
    return os.path.join(project_home, pattern)


def delete_master_files(project_home):
    """Delete all master files generated by the project at *project_home*.

    Queries the ``master_file`` table for every entry and removes the
    corresponding file from disk.

    Args:
        project_home(str):  Absolute path to the project home directory.
    """

    set_project_home(project_home)

    with start_db_session() as db_session:
        masters = db_session.execute(
            select(MasterFile.filename, MasterType.name).join(MasterType)
        ).all()
        extra_patterns = []
        for param_name in (
            "master-photref-fname-format",
            "magfit-stat-fname-format",
            "astrometry-catalog",
            "photometry-catalog",
            "magfit-catalog",
            "lc-catalog",
        ):
            pattern = db_session.scalar(
                select(Configuration.value)
                .join(Parameter)
                .where(Parameter.name == param_name)
            )
            if pattern is not None:
                extra_patterns.append(pattern)

    known = {"PROJHOME": project_home, "project_home": project_home}
    for pattern in extra_patterns:
        for fpath in iglob(_pattern_to_glob(pattern, known, project_home)):
            if os.path.isfile(fpath):
                _safe_remove(fpath, project_home)
                logger.info("Deleted master file: %s", fpath)

    for master_fname, master_type in masters:
        if os.path.exists(master_fname):
            _safe_remove(master_fname, project_home)
            logger.info("Deleted %s master file: %s", master_type, master_fname)


def delete_logs(project_home):
    """Delete all log and stdout/stderr output files for a project.

    Reads the ``logging-fname`` and ``std-out-err-fname`` configuration
    parameters from the project database, converts the patterns to globs
    by substituting unknown placeholders with ``*``, and removes all
    matching files.

    Args:
        project_home(str):  Absolute path to the project home directory.
    """

    set_project_home(project_home)

    with start_db_session() as db_session:
        log_pattern = db_session.scalar(
            select(Configuration.value)  # pylint: disable=no-member
            .join(Parameter)
            .where(Parameter.name == "logging-fname")
        )
        outerr_pattern = db_session.scalar(
            select(Configuration.value)  # pylint: disable=no-member
            .join(Parameter)
            .where(Parameter.name == "std-out-err-fname")
        )
        parent_pids = db_session.scalars(
            select(PipelineRun.process_id).distinct()
        ).all()

    known = {"project_home": project_home}
    glob_patterns = []
    for pattern in (log_pattern, outerr_pattern):
        if pattern is None:
            continue
        base_glob = _pattern_to_glob(pattern, known, project_home)
        glob_patterns.append(base_glob)
        for pid in parent_pids:
            glob_patterns.append(
                os.path.join(
                    os.path.dirname(base_glob),
                    str(pid),
                    os.path.basename(base_glob),
                )
            )

    if not glob_patterns:
        logger.info("No log filename configuration found for %s", project_home)
        return

    for glob_pattern in glob_patterns:
        for fpath in iglob(glob_pattern):
            if os.path.isfile(fpath):
                _safe_remove(fpath, project_home)
                logger.info("Deleted log file: %s", fpath)


def prune_empty_directories(project_home):
    """Remove all empty directories under *project_home*.

    Walks the directory tree bottom-up so that a parent directory is checked
    only after its children have already been removed if they were empty.
    The *project_home* directory itself is never removed.

    Args:
        project_home(str):  Absolute path to the project home directory.
    """

    for dirpath, _dirnames, _filenames in os.walk(project_home, topdown=False):
        if not os.listdir(dirpath):
            os.rmdir(dirpath)
            logger.info("Removed empty directory: %s", dirpath)


def find_missing_databases():
    """Return projects whose database is not accessible.

    A project's database is considered present when either its SQLite file
    (``autowisp.db``) or its centralised-DB connection file
    (``autowisp_db.url``) exists under the project home directory.

    Returns:
        list[Project]:  Projects for which neither database indicator exists.
    """

    missing = []
    for project in Project.objects.all():  # pylint: disable=no-member
        sqlite_path = os.path.join(project.path, "autowisp.db")
        url_file_path = os.path.join(project.path, DB_URL_FNAME)
        if not os.path.isfile(sqlite_path) and not os.path.isfile(
            url_file_path
        ):
            missing.append(project)
    return missing


def select_project(request, project_id):
    """Redirect to the processing progress page for the selected project."""

    request.session.flush()
    project = Project.objects.get(id=project_id)  # pylint: disable=no-member
    request.session["project_home"] = project.path
    request.session["project_name"] = project.name

    return redirect("processing:progress")


def reset_project_config(request):
    """Reset the configuration of project being created to defaults."""

    request.session.flush()
    return redirect("home:new_project")


def export_master_config(request):
    """Generate a JSON file with the current master config for new project."""

    master_config = {}
    for master_type in master_info:
        if master_type in ["highflat", "lowflat"]:
            master_type = "flat"
        master_config[master_type] = {
            param: (
                request.session[f"master-{master_type}-{param}"]
                if param == "enabled"
                else list(
                    filter(
                        None,
                        request.session[f"master-{master_type}-{param}"],
                    )
                )
            )
            for param in ["enabled", "split", "match"]
        }
    with StringIO() as export_stream:
        json.dump(master_config, export_stream, indent=4)
        return HttpResponse(
            export_stream.getvalue().encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Content-Disposition": (
                    'attachment; filename="master_config.json"'
                ),
            },
        )
