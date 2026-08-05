"""Photometric-reference selection helpers shared by the BUI and tests.

This module hosts the non-Django half of what the BUI's
``select_photref_views`` does:

- :func:`compute_photref_candidates` walks ``processing.pending`` for
  ``fit_magnitudes`` and groups the per-condition batches that still
  need a single photometric reference.
- :func:`bind_images_to_photref` writes the ``ImageMasterSelection``
  rows for every batch image within ``max_photref_separation`` of the
  chosen photref.

The view module calls these to populate the Django session / handle
form submissions; the integration test calls them directly to mimic
"user picks a photref" without going through HTTP.
"""

from astropy.coordinates import SkyCoord
from astropy import units as astropy_units
from sqlalchemy import select

from autowisp.data_reduction.data_reduction_file import DataReductionFile
from autowisp.database.image_processing import (
    ImageProcessingManager,
    get_master_expression_ids,
    remove_failed_prerequisite,
)
from autowisp.database.interface import start_db_session
from autowisp.database.user_interface import get_processing_sequence

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
    each non-empty group produces a ``(master_values,
    calculate_photref_merit_config, batch)`` tuple where ``batch`` is
    the list of ``(calibrated_fname, dr_fname, image_id, channel)``
    entries :func:`bind_images_to_photref` expects.

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
                  ``(list(master_values), config, batch)`` per group of
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
            if demo:
                unbound_images = by_master_values
            else:
                group_channel = by_master_values[0][1]
                bound_image_ids = set(
                    db_session.scalars(
                        select(ImageMasterSelection.image_id).where(
                            ImageMasterSelection.master_type_id
                            == master_type_id,
                            ImageMasterSelection.channel == group_channel,
                            ImageMasterSelection.image_id.in_(
                                [img.id for img, _, _ in by_master_values]
                            ),
                        )
                    ).all()
                )
                unbound_images = [
                    (img, ch, st)
                    for img, ch, st in by_master_values
                    if img.id not in bound_image_ids
                ]
            if not unbound_images:
                continue
            config = processing.get_config(
                matched_expressions=None,
                db_session=db_session,
                image_id=unbound_images[0][0].id,
                channel=unbound_images[0][1],
                step_name="calculate_photref_merit",
            )[0]
            groups.append(
                (
                    list(master_values),
                    config,
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
            return

        pf_image_id = db_session.scalar(
            select(Image.id).where(  # pylint: disable=no-member
                Image.raw_fname.contains(  # pylint: disable=no-member
                    f"{pf_rawfname}."
                )
            ) 
        )
        if pf_image_id is None:
            return
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
            return

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
                db_session.merge(
                    ImageMasterSelection(
                        image_id=image_id,
                        channel=channel,
                        master_type_id=master_file.type_id,
                        master_file_id=master_file.id,
                    )
                )
