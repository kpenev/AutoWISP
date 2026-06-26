"""Build a shareable, scrubbed crash report for a recorded error.

The report gathers everything needed to diagnose a failure -- the error
record and its sidecar, the relevant per-process logs, the configuration
in effect, and environment provenance -- into a single zip the user can
hand to the maintainers.

Because logs and configuration can contain credentials (e.g. the Gaia
archive user/password threaded through the process configuration), every
text artifact is passed through the scrubbing helpers here before it
enters the report. Scrubbing is mandatory: nothing is written unscrubbed.
"""

import argparse
import importlib.metadata
import json
import logging
import os
import platform
import re
import shutil
import socket
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import Session

from autowisp.database.interface import (
    start_db_session,
    get_project_home,
    set_project_home,
)
from autowisp.database.image_processing import ImageProcessingManager
from autowisp.error_persistence import load_sidecar
from autowisp.exceptions import sanitize_for_json
from autowisp.miscellaneous import get_code_version_str

# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    Configuration,
    Error,
    Image,
    ImageProcessingProgress,
    Parameter,
    PipelineRun,
    Step,
)

# pylint: enable=no-name-in-module

git_id = "$Id$"

_logger = logging.getLogger(__name__)

#: Replacement written in place of a redacted secret value.
REDACTED = "***REDACTED***"  # pylint: disable=invalid-name

# Substring matched (case-insensitively) against a key to decide it names
# a secret. Covers the Gaia credentials this pipeline threads through the
# process config plus the usual credential-like names.
_SECRET = (  # pylint: disable=invalid-name
    r"gaia[_-]?(?:user|password)"
    r"|password|passwd|secret|token"
    r"|api[_-]?key|access[_-]?key|credentials?"
)

# A `key <:|=> value` assignment whose key names a secret, as it appears
# in dict reprs ('gaia_password': 'x'), JSON ("gaia_password": "x"), and
# ini/yaml-style config (gaia-password = x). The value -- quoted, or bare
# up to a comma or end of line -- is what gets redacted.
_SECRET_ASSIGNMENT = re.compile(
    rf"(?i)(['\"]?\b(?:{_SECRET})\b['\"]?\s*[:=]\s*)"
    r"('[^']*'|\"[^\"]*\"|[^\n,]*)"
)

# Matches a secret name anywhere in a mapping key.
_SECRET_KEY = re.compile(rf"(?i)(?:{_SECRET})")


def scrub_text(text):
    """Redact secret values from a blob of text (a log or config file).

    Replaces the value of any ``key: value`` / ``key = value`` assignment
    whose key names a secret with :data:`REDACTED`, leaving the key (and
    everything else) intact. Best-effort and never raises.

    Args:
        text(str):    The text to scrub.

    Returns:
        str:    The text with secret values redacted.
    """

    if not text:
        return text
    return _SECRET_ASSIGNMENT.sub(r"\1" + REDACTED, text)


def scrub_mapping(mapping):
    """Return a copy of ``mapping`` with secret-keyed values redacted.

    Recurses into nested dictionaries. A value is redacted when its key
    name matches a secret (e.g. ``gaia_password``, ``api_key``); other
    values are copied through unchanged.

    Args:
        mapping(dict):    The mapping to scrub.

    Returns:
        dict:    A scrubbed copy.
    """

    scrubbed = {}
    for key, value in mapping.items():
        if isinstance(key, str) and _SECRET_KEY.search(key):
            scrubbed[key] = REDACTED
        elif isinstance(value, dict):
            scrubbed[key] = scrub_mapping(value)
        else:
            scrubbed[key] = value
    return scrubbed


def scrub_config_values(db_session):
    """Redact secret configuration values in a database, in place.

    The project configuration stores credentials as ordinary rows (e.g. a
    ``gaia-password`` parameter), which a binary database file or SQL dump
    cannot be text-scrubbed for. This redacts the ``Configuration`` value
    of every parameter whose name names a secret.

    Intended for a *copy* of the project database destined for a crash
    report -- never the live database -- since it mutates the rows.

    Args:
        db_session:    A session connected to the database copy to scrub.

    Returns:
        int:    The number of configuration values redacted.
    """

    secret_param_ids = [
        param_id
        for param_id, name in db_session.execute(
            select(Parameter.id, Parameter.name)  # pylint: disable=no-member
        ).all()
        if name and _SECRET_KEY.search(name)
    ]
    if not secret_param_ids:
        return 0
    result = db_session.execute(
        update(Configuration)
        .where(
            Configuration.parameter_id.in_(  # pylint: disable=no-member
                secret_param_ids
            )
        )
        .values(value=REDACTED)
    )
    return result.rowcount or 0


# --- Locating the logs and processing record for an error. ------------


def find_error_progress(error_row, db_session=None):
    """Return the ``ImageProcessingProgress`` an error belongs to, or None.

    Resolves a step error to its processing record via run + step (and the
    image's type, when the error names an image), so the BUI can link the
    error to the matching log-review page. A pipeline/BUI error -- or a
    step with no recorded progress -- yields ``None`` rather than a wrong
    match.

    Args:
        error_row:    The ``Error`` row.

        db_session:    Optional active session; one is opened if omitted.

    Returns:
        ImageProcessingProgress or None
    """

    if db_session is None:
        with start_db_session() as own_session:
            return find_error_progress(error_row, own_session)

    if error_row.pipeline_run_id is None or not error_row.step_name:
        return None

    # pylint: disable=no-member
    step_id = db_session.scalar(
        select(Step.id).where(Step.name == error_row.step_name)
    )
    if step_id is None:
        return None

    query = select(ImageProcessingProgress).where(
        ImageProcessingProgress.run_id == error_row.pipeline_run_id,
        ImageProcessingProgress.step_id == step_id,
    )
    if error_row.image_id is not None:
        image_type_id = db_session.scalar(
            select(Image.image_type_id).where(Image.id == error_row.image_id)
        )
        if image_type_id is not None:
            query = query.where(
                ImageProcessingProgress.image_type_id == image_type_id
            )
    return db_session.scalars(
        query.order_by(ImageProcessingProgress.id.desc())
    ).first()
    # pylint: enable=no-member


def select_error_logs(error_row, db_session=None):
    """Return the per-process log files relevant to an error.

    Reuses the pipeline's own log-locating machinery
    (``ImageProcessingManager.find_processing_outputs``), so the
    configured ``logging_fname`` / ``std_out_err_fname`` naming is honored
    rather than assumed. Only the logs for the error's run and step are
    returned (the main-process log/outerr and the run's worker logs), not
    the whole log directory. Best-effort: returns the existing files it
    finds, or an empty list.

    Args:
        error_row:    The ``Error`` row.

        db_session:    Optional active session; one is opened if omitted.

    Returns:
        list[str]:    Absolute paths of the matching log files.
    """

    if db_session is None:
        with start_db_session() as own_session:
            return select_error_logs(error_row, own_session)

    progress = find_error_progress(error_row, db_session)
    if progress is None:
        return []

    try:
        main_logs, worker_logs = ImageProcessingManager(
            pipeline_run_id=None
        ).find_processing_outputs(progress, db_session)
    except Exception:  # pylint: disable=broad-except
        _logger.debug(
            "Could not locate logs for error %s",
            getattr(error_row, "id", None),
            exc_info=True,
        )
        return []

    candidates = list(main_logs)
    for entry in worker_logs:
        candidates.extend(entry)
    return sorted(
        {path for path in candidates if path and os.path.exists(path)}
    )


# --- Environment provenance. ------------------------------------------


def _package_versions(names):
    """Return ``{name: version}`` for the installed packages among ``names``.

    Packages that are not installed are simply omitted; never raises.
    """

    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def collect_provenance():
    """Return environment provenance for a crash report.

    Captures the machine and software building the report -- hostname, OS,
    Python and key package versions, and the current code version. The
    *failed run's* host and code version live on its ``PipelineRun`` row
    and are added separately by the report builder.

    Returns:
        dict:    Provenance fields, all JSON-serializable.
    """

    return {
        "report_generated": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "code_version": get_code_version_str(),
        "packages": _package_versions(
            (
                "autowisp",
                "astrowisp",
                "numpy",
                "scipy",
                "pandas",
                "sqlalchemy",
                "astropy",
            )
        ),
    }


# --- Assembling the report. -------------------------------------------


def _error_record(error_row, db_session):
    """The inline error fields plus its run's host/PID/code version."""

    record = {
        "id": error_row.id,
        "component": error_row.component,
        "step_name": error_row.step_name,
        "exception_class": error_row.exception_class,
        "user_message": error_row.user_message,
        "created": error_row.created,
        "subprocess_id": error_row.subprocess_id,
        "pipeline_run_id": error_row.pipeline_run_id,
        "image_id": error_row.image_id,
        "master_file_id": error_row.master_file_id,
    }
    if error_row.pipeline_run_id is not None:
        run = db_session.get(PipelineRun, error_row.pipeline_run_id)
        if run is not None:
            record["run_host"] = run.host
            record["run_process_id"] = run.process_id
            record["run_code_version"] = run.code_version
    return record


def _read_log_scrubbed(path, max_log_bytes):
    """Return a log's scrubbed text, head+tail truncated past the cap."""

    size = os.path.getsize(path)
    with open(path, "r", encoding="utf-8", errors="replace") as log_file:
        if size <= max_log_bytes:
            return scrub_text(log_file.read())
        half = max_log_bytes // 2
        head = log_file.read(half)
        log_file.seek(size - half)
        tail = log_file.read()
    omitted = size - 2 * half
    return scrub_text(
        f"{head}\n\n... [{omitted} bytes omitted; log truncated] ...\n\n{tail}"
    )


def _add_scrubbed_database(zip_file, manifest):
    """Add a copy of the SQLite project database with secrets redacted."""

    src = os.path.join(get_project_home(), "autowisp.db")
    if not os.path.exists(src):
        manifest["gaps"].append(
            {"artifact": "database", "reason": "no SQLite database found"}
        )
        return
    with tempfile.TemporaryDirectory() as work_dir:
        copy = os.path.join(work_dir, "autowisp.db")
        shutil.copy(src, copy)
        engine = create_engine(f"sqlite:///{copy}")
        try:
            with Session(engine) as scrub_session:
                scrub_config_values(scrub_session)
                scrub_session.commit()
        finally:
            engine.dispose()
        zip_file.write(copy, "database/autowisp.db")
    manifest["collected"].append("database/autowisp.db")


def build_crash_report(
    error_id, out_path=None, *, max_log_bytes=512 * 1024, db_session=None
):
    """Assemble a scrubbed crash-report zip for one error.

    Collects the error record, its detail sidecar, the per-process logs
    for its run/step, a credential-scrubbed copy of the SQLite project
    database, and environment provenance, plus a ``manifest.json``
    describing what was gathered. Every text artifact is scrubbed; the
    database is scrubbed in the copy. Read-only with respect to live
    pipeline state. Collection is best-effort: a source that cannot be
    read is noted as a gap in the manifest rather than failing the report.

    Args:
        error_id(int):    The error to report on.

        out_path(str or Path or None):    Destination zip; defaults to
            ``crash_report_error_<id>.zip`` in the current directory.

        max_log_bytes(int):    Per-log size cap; larger logs are head+tail
            truncated. Raise it for a more thorough (larger) report.

        db_session:    Optional active session; one is opened if omitted.

    Returns:
        Path:    The path to the written zip.

    Raises:
        ValueError:    If no error has the given id.
    """

    if db_session is None:
        with start_db_session() as own_session:
            return build_crash_report(
                error_id,
                out_path,
                max_log_bytes=max_log_bytes,
                db_session=own_session,
            )

    error_row = db_session.get(Error, error_id)
    if error_row is None:
        raise ValueError(f"No recorded error with id {error_id}.")

    out_path = Path(
        out_path or f"crash_report_error_{error_id}.zip"
    ).expanduser()
    manifest = {
        "schema_version": 1,
        "error_id": error_id,
        "generated": datetime.now(timezone.utc).isoformat(),
        "collected": [],
        "gaps": [],
    }

    def add_json(name, payload):
        zip_file.writestr(
            name, json.dumps(payload, indent=2, default=sanitize_for_json)
        )
        manifest["collected"].append(name)

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Error record.
        try:
            add_json("error.json", _error_record(error_row, db_session))
        except Exception as exc:  # pylint: disable=broad-except
            manifest["gaps"].append(
                {"artifact": "error.json", "reason": repr(exc)}
            )

        # Detail sidecar (scrubbed).
        try:
            sidecar = load_sidecar(error_row)
            if sidecar is None:
                manifest["gaps"].append(
                    {"artifact": "sidecar.json", "reason": "no sidecar"}
                )
            else:
                zip_file.writestr(
                    "sidecar.json",
                    scrub_text(
                        json.dumps(sidecar, indent=2, default=sanitize_for_json)
                    ),
                )
                manifest["collected"].append("sidecar.json")
        except Exception as exc:  # pylint: disable=broad-except
            manifest["gaps"].append(
                {"artifact": "sidecar.json", "reason": repr(exc)}
            )

        # Per-process logs (scrubbed, truncated).
        try:
            log_paths = select_error_logs(error_row, db_session)
            if not log_paths:
                manifest["gaps"].append(
                    {"artifact": "logs", "reason": "no matching logs found"}
                )
            for path in log_paths:
                try:
                    arcname = f"logs/{os.path.basename(path)}"
                    zip_file.writestr(
                        arcname, _read_log_scrubbed(path, max_log_bytes)
                    )
                    manifest["collected"].append(arcname)
                except Exception as exc:  # pylint: disable=broad-except
                    manifest["gaps"].append(
                        {"artifact": f"logs/{path}", "reason": repr(exc)}
                    )
        except Exception as exc:  # pylint: disable=broad-except
            manifest["gaps"].append({"artifact": "logs", "reason": repr(exc)})

        # Scrubbed database copy.
        try:
            _add_scrubbed_database(zip_file, manifest)
        except Exception as exc:  # pylint: disable=broad-except
            manifest["gaps"].append(
                {"artifact": "database", "reason": repr(exc)}
            )

        # Environment provenance.
        try:
            add_json("provenance.json", collect_provenance())
        except Exception as exc:  # pylint: disable=broad-except
            manifest["gaps"].append(
                {"artifact": "provenance.json", "reason": repr(exc)}
            )

        # Manifest last, so it records every gap seen above.
        zip_file.writestr(
            "manifest.json",
            json.dumps(manifest, indent=2, default=sanitize_for_json),
        )

    return out_path


def latest_error_id(db_session=None):
    """Return the id of the most recently recorded error, or None.

    Args:
        db_session:    Optional active session; one is opened if omitted.

    Returns:
        int or None
    """

    if db_session is None:
        with start_db_session() as own_session:
            return latest_error_id(own_session)
    return db_session.scalar(
        select(Error.id).order_by(  # pylint: disable=no-member
            Error.id.desc()  # pylint: disable=no-member
        )
    )


def crash_report_main():
    """CLI entry point for ``wisp-crash-report``."""

    parser = argparse.ArgumentParser(
        prog="wisp-crash-report",
        description=(
            "Bundle a recorded error into a single, credential-scrubbed "
            "zip (error record, sidecar, logs, a database copy, and "
            "provenance) to share with the maintainers."
        ),
    )
    parser.add_argument(
        "project_home", help="Path to the project home directory."
    )
    parser.add_argument(
        "error_id",
        nargs="?",
        type=int,
        default=None,
        help="The id of the error to report on (see the BUI error log). "
        "Omit and pass --last for the most recent error.",
    )
    parser.add_argument(
        "--last",
        action="store_true",
        help="Report on the most recently recorded error.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output zip path (default: crash_report_error_<id>.zip).",
    )
    parser.add_argument(
        "--max-log-bytes",
        type=int,
        default=512 * 1024,
        help="Per-log size cap before head+tail truncation; raise for a "
        "more thorough report (default: 524288).",
    )
    args = parser.parse_args()

    set_project_home(args.project_home)

    error_id = args.error_id
    if args.last:
        error_id = latest_error_id()
        if error_id is None:
            parser.error("no errors are recorded for this project")
    if error_id is None:
        parser.error("provide an error_id or use --last")

    try:
        out_path = build_crash_report(
            error_id, args.out, max_log_bytes=args.max_log_bytes
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Wrote crash report for error {error_id} to {out_path}")
