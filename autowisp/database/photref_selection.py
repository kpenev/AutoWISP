"""Photometric-reference selection helpers shared by the BUI and tests.

This module hosts the non-Django half of what the BUI's
``select_photref_views`` does:

- :func:`compute_photref_candidates` walks ``processing.pending`` for
  ``fit_magnitudes`` and groups the per-condition batches that still
  need a single photometric reference, leaving out entries
  :func:`get_unbound_entries` finds already bound.
- :func:`bind_images_to_photref` writes the ``ImageMasterSelection``
  rows for every batch image within ``max_photref_separation`` of the
  chosen photref.
- :func:`check_photref_fnames` refuses a photref whose magnitude fitting
  would write the same files as that of one already registered.
- :func:`record_single_photref` checks the chosen photref, registers it,
  and binds the batch to it.
- :func:`get_group_exclusions` reports what the magfit exclusion rule
  leaves out of a group still needing a photref, and
  :func:`get_offered_candidates` which of its images to offer as one.
- :func:`rank_photref_candidates` orders the offered images by a merit
  expression from the library, one of :func:`get_merit_expressions`.
- :func:`get_photref_exclusions` reports what each step's exclusion rule
  leaves out of the images bound to each photref.

The view module calls these to populate the Django session / handle
form submissions; the integration test calls them directly to mimic
"user picks a photref" without going through HTTP.
"""

from astropy.coordinates import SkyCoord
from astropy import units as astropy_units
import numpy
from sqlalchemy import select

from autowisp.data_reduction.data_reduction_file import DataReductionFile
from autowisp.database.image_processing import (
    ImageProcessingManager,
    get_master_expression_ids,
    record_photref_bindings,
    remove_failed_prerequisite,
)
from autowisp.database.interface import start_db_session
from autowisp.database.user_interface import get_processing_sequence
from autowisp.diagnostics.diagnostic_types import photometry_literal
from autowisp.diagnostics.exclusion_rules import (
    get_excluded,
    summarize_excluded,
)
from autowisp.diagnostics.expression_library import get_expressions
from autowisp.diagnostics.expression_series import get_custom_group_values
from autowisp.diagnostics.expressions import (
    get_channel_arity,
    get_photometry_arity,
)
from autowisp.evaluator import Evaluator
from autowisp.exceptions import ConfigurationError
from autowisp.magnitude_fitting.util import get_path_substitutions

# false positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    ConditionExpression,
    DiagnosticType,
    Image,
    ImageDiagnostics,
    ImageMasterSelection,
    MasterFile,
    MasterType,
    Step,
)

# pylint: enable=no-name-in-module


def get_unbound_entries(entries, master_type_id, db_session):
    """
    Return the entries not yet bound to a single photometric reference.

    Bindings are per (image, channel), and so is the check: a photref group
    need not be one channel. Under a ``must_match`` such as
    ``CLRCHNL[0].upper()``, ``G0`` and ``G1`` of one image are in one group,
    and an image bound in one of them still needs its other entry bound.

    Args:
        entries:    ``(image, channel, status)`` tuples of one photref group,
            as ``group_pending_by_conditions`` gives them.

        master_type_id(int):    The id of the ``single_photref`` master type.

        db_session:    Open SQLAlchemy session.

    Returns:
        list:    The entries with no binding, in the order given.
    """

    bound = set(
        db_session.execute(
            select(
                ImageMasterSelection.image_id, ImageMasterSelection.channel
            ).where(
                ImageMasterSelection.master_type_id == master_type_id,
                ImageMasterSelection.image_id.in_(
                    {image.id for image, _, _ in entries}
                ),
            )
        ).all()
    )
    return [entry for entry in entries if (entry[0].id, entry[1]) not in bound]


def compute_photref_candidates(processing, db_session):
    # pylint: disable=too-many-locals
    """Return the per-condition batches of images missing a photref.

    Holds the data-gathering half of what
    ``select_photref_views._get_missing_photref`` does. The BUI calls
    this (and then writes the result into the Django session); the
    integration test calls it directly.

    Builds
    ``processing.pending`` for the ``fit_magnitudes`` step (optionally
    falling back to "demo" mode where *every* candidate is treated as
    pending), strips images whose ``solve_astrometry`` prerequisite
    failed, groups the survivors by master-condition values, and for
    each non-empty group produces a ``(master_values, batch)`` tuple
    where ``batch`` is the list of ``(calibrated_fname, dr_fname,
    image_id, channel)`` entries :func:`bind_images_to_photref` expects.

    Args:
        processing:    A fresh ``ImageProcessingManager``. Its
            ``pending`` attribute is populated as a side effect.
        db_session:    Open SQLAlchemy session.

    Returns:
        dict with keys:

            ``"demo"`` (bool)
                True iff no images were actually pending
                ``fit_magnitudes`` -- the caller may then surface every
                candidate for inspection rather than only the unbound
                ones.

            ``"candidates"`` (list[dict])
                One entry per ``(step_id, image_type_id)`` in
                ``processing.pending``. Each entry has:

                * ``"step_id"`` (int)
                * ``"image_type_id"`` (int)
                * ``"master_expressions"`` (list[str]) -- the condition
                  expressions defining a photref's identity.
                * ``"groups"`` (list[tuple]) -- a tuple of
                  ``(list(master_values), batch)`` per group of
                  images sharing the same master-condition values.
    """

    master_type_id = db_session.scalar(
        select(MasterType.id).filter_by(name="single_photref")
    )
    magfit_steps = [
        entry
        for entry in get_processing_sequence(db_session)
        if entry[0].name == "fit_magnitudes"
    ]
    processing.set_pending(db_session, magfit_steps)
    for step in magfit_steps:
        for pending in processing.pending[(step[0].id, step[1].id)]:
            processing.evaluate_expressions_image(pending[0], db_session)

    # No images are actually pending fit_magnitudes (every per-step list is
    # empty) -- enter "demo" mode: tell processing to enumerate every
    # candidate (invert=True) so the BUI has something to surface.
    demo = not any(processing.pending.values())
    if demo:
        processing.set_pending(db_session, magfit_steps, True)

    astrom_step_id = db_session.scalar(
        select(Step.id).filter_by(name="solve_astrometry")
    )

    candidates = []
    for (
        step_id,
        image_type_id,
    ), pending_images in processing.pending.items():
        remove_failed_prerequisite(
            pending_images, image_type_id, astrom_step_id, db_session
        )
        master_expressions = [
            db_session.scalar(
                select(ConditionExpression.expression).filter_by(id=expr_id)
            )
            for expr_id in get_master_expression_ids(
                step_id, image_type_id, db_session
            )
        ]
        groups = []
        by_photref = processing.group_pending_by_conditions(
            pending_images,
            db_session,
            match_observing_session=False,
            step_id=step_id,
            masters_only=True,
        )
        for by_master_values, master_values in by_photref:
            unbound_images = (
                by_master_values
                if demo
                else get_unbound_entries(
                    by_master_values, master_type_id, db_session
                )
            )
            if not unbound_images:
                continue
            groups.append(
                (
                    list(master_values),
                    [
                        (
                            processing.get_step_input(
                                image, channel, "calibrated"
                            ),
                            processing.get_step_input(image, channel, "dr"),
                            image.id,
                            channel,
                        )
                        for image, channel, _ in unbound_images
                    ],
                )
            )
        candidates.append(
            {
                "step_id": step_id,
                "image_type_id": image_type_id,
                "master_expressions": master_expressions,
                "groups": groups,
            }
        )

    return {"demo": demo, "candidates": candidates}


def bind_images_to_photref(dr_fname, batch):
    # pylint: disable=too-many-locals
    """Write ImageMasterSelection rows for batch images near the photref.

    Reads the fit_magnitudes config to get ``max_photref_separation``
    (which may be conditional), then for each image in ``batch``
    computes the angular separation between the image center and the
    photref center. Images whose separation is within
    ``max_photref_separation * photref_diagonal_fov`` are bound to the
    photref via an upsert into ``ImageMasterSelection``.

    Args:
        dr_fname:    Path to the photref DR file that was just
            registered as a ``single_photref`` master via
            :meth:`ImageProcessingManager.add_masters`.
        batch:    List of ``(calibrated_fname, dr_fname, image_id,
            channel)`` tuples -- the candidate images from the same
            condition group. Only ``image_id`` and ``channel`` are
            consumed here; the first two slots exist for parity with
            ``compute_photref_candidates``'s return shape.

    Returns:
        list:    The ``(image_id, channel)`` entries bound, empty if the
            photref is not registered or its center is unknown.
    """

    with DataReductionFile(dr_fname, "r") as pf_dr:
        pf_header = pf_dr.get_frame_header()
    pf_rawfname = pf_header["RAWFNAME"]
    pf_channel = pf_header["CLRCHNL"]

    processing = ImageProcessingManager(pipeline_run_id=None)

    with start_db_session() as db_session:
        master_file = db_session.scalar(
            select(MasterFile).where(MasterFile.filename == dr_fname)
        )
        if master_file is None:
            return []

        pf_image_id = db_session.scalar(
            select(Image.id).where(  # pylint: disable=no-member
                Image.raw_fname.like(  # pylint: disable=no-member
                    f"%/{pf_rawfname}.%"
                )
            )
        )
        if pf_image_id is None:
            return []
        pf_diags = dict(
            db_session.execute(
                select(DiagnosticType.name, ImageDiagnostics.value)
                .join(
                    DiagnosticType,
                    ImageDiagnostics.diagnostic_id == DiagnosticType.id,
                )
                .where(
                    ImageDiagnostics.image_id == pf_image_id,
                    ImageDiagnostics.channel == pf_channel,
                    DiagnosticType.name.in_(
                        ["ra_center", "dec_center", "diagonal_fov"]
                    ),
                )
            ).all()
        )
        if not all(
            k in pf_diags for k in ("ra_center", "dec_center", "diagonal_fov")
        ):
            return []

        pf_center = SkyCoord(
            ra=pf_diags["ra_center"] * astropy_units.deg,
            dec=pf_diags["dec_center"] * astropy_units.deg,
            frame="icrs",
        )

        first_image_id, first_channel = batch[0][2], batch[0][3]
        first_image = db_session.get(Image, first_image_id)
        processing.evaluate_expressions_image(first_image, db_session)
        fit_config = processing.get_config(
            matched_expressions=None,
            db_session=db_session,
            image_id=first_image_id,
            channel=first_channel,
            step_name="fit_magnitudes",
        )[0]
        threshold_deg = (
            fit_config.get("max_photref_separation", 0.2)
            * pf_diags["diagonal_fov"]
        )

        new_bindings = []
        for _, _, image_id, channel in batch:
            img_diags = dict(
                db_session.execute(
                    select(DiagnosticType.name, ImageDiagnostics.value)
                    .join(
                        DiagnosticType,
                        ImageDiagnostics.diagnostic_id == DiagnosticType.id,
                    )
                    .where(
                        ImageDiagnostics.image_id == image_id,
                        ImageDiagnostics.channel == channel,
                        DiagnosticType.name.in_(["ra_center", "dec_center"]),
                    )
                ).all()
            )
            if "ra_center" not in img_diags or "dec_center" not in img_diags:
                continue
            img_center = SkyCoord(
                ra=img_diags["ra_center"] * astropy_units.deg,
                dec=img_diags["dec_center"] * astropy_units.deg,
                frame="icrs",
            )
            if (
                pf_center.separation(img_center).to_value(astropy_units.deg)
                <= threshold_deg
            ):
                new_bindings.append((image_id, channel, master_file.id))
        record_photref_bindings(new_bindings, master_file.type_id, db_session)
    return [(image_id, channel) for image_id, channel, _ in new_bindings]


def _get_magfit_fnames(processing, photref_fname, db_session):
    """
    Return the files magnitude fitting against a single photref may write.

    Args:
        processing(ImageProcessingManager):    Gives the ``fit_magnitudes``
            configuration that applies to the reference.

        photref_fname(str):    The DR file of the single photometric
            reference.

        db_session:    The database session to read the configuration in.

    Returns:
        dict:
            The master photometric reference and statistics file names of
            every iteration, each mapped to the option that formats it.
    """

    with DataReductionFile(photref_fname, "r") as photref_dr:
        header = photref_dr.get_frame_header()
    config = processing.get_config(
        processing.get_matched_expressions(Evaluator(header)),
        db_session,
        step_name="fit_magnitudes",
    )[0]
    # As MagnitudeFitting expands them. dict() first: a header may repeat a
    # keyword.
    substitutions = {**dict(header), **get_path_substitutions(config, header)}
    return {
        config[option].format_map(
            dict(substitutions, magfit_iteration=iteration)
        ): option
        for option in (
            "master_photref_fname_format",
            "magfit_stat_fname_format",
        )
        for iteration in range(config["max_magfit_iterations"] + 1)
    }


def check_photref_fnames(processing, photref_fname):
    """
    Raise if a single photref would share magfit files with a registered one.

    Compares file names rather than looking for files, so a clash is found
    before either master is built. ``fit_magnitudes`` refuses to overwrite a
    file as well, but only once the second master is being built.

    Args:
        processing(ImageProcessingManager):    Gives the configuration that
            applies to each reference.

        photref_fname(str):    The DR file of the single photometric
            reference about to be registered.

    Returns:
        None
    """

    with start_db_session() as db_session:
        new_fnames = _get_magfit_fnames(processing, photref_fname, db_session)
        for other_fname in db_session.scalars(
            select(MasterFile.filename)
            .join(MasterType)
            .where(
                MasterType.name == "single_photref",
                MasterFile.filename != photref_fname,
            )
        ).all():
            other_fnames = _get_magfit_fnames(
                processing, other_fname, db_session
            )
            clashes = sorted(new_fnames.keys() & other_fnames.keys())
            if clashes:
                raise ConfigurationError(
                    f"Single photometric references {photref_fname!r} and "
                    f"{other_fname!r} would both write {clashes[0]!r}. Make --"
                    + new_fnames[clashes[0]].replace("_", "-")
                    + " tell them apart, e.g. by including {FNUM}."
                )


def _get_rules(processing, dr_fname, db_session):
    """
    Return the exclusion rule each step has for the images like a DR file's.

    Read from the configuration through the conditions the file's header
    matches, as when the magnitude fitting file names of a single
    photometric reference are checked.

    Args:
        processing(ImageProcessingManager):    Gives the configuration.

        dr_fname(str):    A DR file of the images the rules are for.

        db_session:    An active SQLAlchemy database session.

    Returns:
        dict:    ``{step name: rule}``, in processing order, for each of
            ``fit_magnitudes``, ``epd`` and ``tfa`` with a rule set.
    """

    with DataReductionFile(dr_fname, "r") as dr_file:
        matched = processing.get_matched_expressions(
            Evaluator(dr_file.get_frame_header())
        )
    rules = {}
    for step_name, option in [
        ("fit_magnitudes", "magfit_exclusion_rule"),
        ("epd", "epd_exclusion_rule"),
        ("tfa", "tfa_exclusion_rule"),
    ]:
        rule = processing.get_config(matched, db_session, step_name=step_name)[
            0
        ].get(option)
        if rule:
            rules[step_name] = rule
    return rules


def _evaluate_rule(rule, members, db_session, *, before_magfit):
    """
    Return what *rule* excludes of the given observations fit together.

    Decided by the engine's own machinery, so what is reported is what the
    step will leave out.

    Args:
        rule(str):    The exclusion rule.

        members(list):    ``(image_id, channel)`` pairs fit together.

        db_session:    An active SQLAlchemy database session.

        before_magfit(bool):    Whether the rule is magnitude fitting's.

    Returns:
        dict:    ``rule``; ``num_images``, the observations fit together;
            and either ``verdicts``, as :func:`summarize_excluded` returns
            them with each photometry's ``label`` added, and ``excluded``,
            the members any verdict excludes, or ``error``, what the engine
            would refuse the rule with.
    """

    result = {"rule": rule, "num_images": len(members)}
    try:
        excluded = get_excluded(
            rule,
            members,
            db_session,
            before_magfit=before_magfit,
            report=False,
        )
    except ConfigurationError as error:
        result["error"] = str(error)
        return result
    except (ValueError, TypeError) as error:
        # Raised evaluating the rule rather than refusing it.
        result["error"] = f"{rule} failed: {error}"
        return result

    result["verdicts"] = [
        dict(
            verdict,
            label=(
                ""
                if verdict["photometry"] is None
                else photometry_literal(verdict["photometry"])
            ),
        )
        for verdict in summarize_excluded(excluded, len(members))
    ]
    result["excluded"] = set().union(*excluded.values())
    return result


def get_group_exclusions(batch, db_session):
    """
    Report what the magfit exclusion rule excludes of a group needing a ref.

    Reported before a reference is chosen, to inform the choice: the images
    fit together once it is are those of the group near enough to it, so
    the group is the estimate available when choosing. Only the magnitude
    fitting rule applies, the single photometric reference belonging to
    magnitude fitting.

    Args:
        batch:    The group's images, as :func:`compute_photref_candidates`
            gives them: ``(calibrated_fname, dr_fname, image_id, channel)``.

        db_session:    An active SQLAlchemy database session.

    Returns:
        dict or None:    As :func:`_evaluate_rule` returns, or None if no
            magfit rule is set for the group.
    """

    rule = _get_rules(
        ImageProcessingManager(pipeline_run_id=None), batch[0][1], db_session
    ).get("fit_magnitudes")
    if rule is None:
        return None
    return _evaluate_rule(
        rule,
        [(image_id, channel) for _, _, image_id, channel in batch],
        db_session,
        before_magfit=True,
    )


def get_offered_candidates(batch, exclusions):
    """
    Return which of a group's images to offer as its single photometric ref.

    An image the magnitude fitting rule excludes is not offered: as the
    reference it would spoil the first pass of the very fit the rule
    protects. Every image is offered where the rule decides nothing -- it is
    unset or refused -- or excludes them all, so that a broken or
    overzealous rule never leaves nothing to choose from.

    Args:
        batch:    The group's images, as for :func:`get_group_exclusions`.

        exclusions(dict or None):    What :func:`get_group_exclusions`
            returned for the group.

    Returns:
        list:    Per entry of *batch*, whether to offer it.
    """

    excluded = (exclusions or {}).get("excluded", set())
    offered = [
        (image_id, channel) not in excluded for _, _, image_id, channel in batch
    ]
    if not any(offered):
        return [True] * len(batch)
    return offered


def get_merit_expressions(library):
    """
    Return the library expressions candidates can be ranked by, by name.

    Those giving one value per entry of a photref group: taking at most one
    channel, which is bound to each entry's own, and no photometry, since
    the group has not been magnitude-fit.

    Args:
        library(dict):    The project's library, ``{name: expression}``.

    Returns:
        list:    The names, alphabetically.
    """

    return sorted(
        name
        for name in library
        if get_channel_arity(name, library) <= 1
        and not get_photometry_arity(name, library)
    )


def rank_photref_candidates(batch, merit, offered, db_session):
    """
    Return the offered entries of a photref group, best first.

    The merit and every diagnostic recorded for them are evaluated over the
    offered entries alone, so an aggregate such as ``nanrank`` ranks each
    candidate among the other candidates, and the images the magnitude
    fitting exclusion rule leaves out do not shift it.

    Args:
        batch:    The group's entries, as for :func:`get_group_exclusions`.

        merit(str or None):    The name of the library expression to rank
            by, one of :func:`get_merit_expressions`; None to keep the order
            of Julian date.

        offered:    Per entry of *batch*, whether to offer it, as
            :func:`get_offered_candidates` gives it.

        db_session:    An active SQLAlchemy database session.

    Returns:
        tuple:
            list:    The positions in *batch* of the offered entries: highest
                merit first and no merit (NaN) last, ties in order of Julian
                date. An entry whose image has no Julian date is left out.

            dict:    ``{quantity: array}``, the merit under its name and every
                diagnostic recorded for the offered entries, one value per
                position returned, in the same order.
    """

    position = {
        (image_id, channel): index
        for index, (_, _, image_id, channel) in enumerate(batch)
        if offered[index]
    }
    quantities = list(
        db_session.scalars(
            select(DiagnosticType.name)
            .join(
                ImageDiagnostics,
                ImageDiagnostics.diagnostic_id == DiagnosticType.id,
            )
            .where(
                ImageDiagnostics.image_id.in_(
                    {image_id for image_id, _ in position}
                ),
                ImageDiagnostics.channel.in_(
                    {channel for _, channel in position}
                ),
            )
            .distinct()
            .order_by(DiagnosticType.name)
        ).all()
    )
    if merit is not None:
        quantities.append(merit)
    values, members = get_custom_group_values(
        list(position), quantities, get_expressions(db_session), db_session
    )

    order = list(range(len(members)))
    if merit is not None:
        # Stable, so that ties stay in order of Julian date.
        order.sort(
            key=lambda index: (
                (1, 0.0)
                if numpy.isnan(values[merit][index])
                else (0, -values[merit][index])
            )
        )
    return (
        [position[members[index]] for index in order],
        {
            quantity: quantity_values[order]
            for quantity, quantity_values in values.items()
        },
    )


def get_photref_exclusions(db_session):
    """
    Report what each step's exclusion rule excludes of each reference's images.

    The images bound to a single photometric reference are fit together:
    ``fit_magnitudes`` splits its batches by reference, and EPD and TFA
    detrend the lightcurve points of one reference at once. The rule applied
    to them is the one configured for the reference.

    Args:
        db_session:    An active SQLAlchemy database session.

    Returns:
        list:    A dict per single photometric reference with images bound
            to it and per step with a rule set for it, by reference and then
            in processing order: ``photref``, its DR file, ``step``, and
            what :func:`_evaluate_rule` returns.
    """

    processing = ImageProcessingManager(pipeline_run_id=None)
    result = []
    for photref_id, photref_fname in db_session.execute(
        select(MasterFile.id, MasterFile.filename)
        .join(MasterType)
        .where(MasterType.name == "single_photref")
        .order_by(MasterFile.filename)
    ).all():
        members = db_session.execute(
            select(
                ImageMasterSelection.image_id, ImageMasterSelection.channel
            ).where(ImageMasterSelection.master_file_id == photref_id)
        ).all()
        if not members:
            continue
        for step_name, rule in _get_rules(
            processing, photref_fname, db_session
        ).items():
            result.append(
                {
                    "photref": photref_fname,
                    "step": step_name,
                    **_evaluate_rule(
                        rule,
                        members,
                        db_session,
                        before_magfit=step_name == "fit_magnitudes",
                    ),
                }
            )
    return result


def record_single_photref(dr_fname, batch):
    """
    Register a single photref and bind to it the batch images near it.

    Args:
        dr_fname(str):    The DR file to register as the reference.

        batch:    The candidate images, as for :func:`bind_images_to_photref`.

    Returns:
        list:    The ``(image_id, channel)`` entries of *batch* bound to it.
    """

    processing = ImageProcessingManager(pipeline_run_id=None)
    check_photref_fnames(processing, dr_fname)
    processing.add_masters(
        {
            "type": "single_photref",
            "filename": dr_fname,
            "preference_order": None,
            "disable": False,
        }
    )
    return bind_images_to_photref(dr_fname, batch)
