#!/usr/bin/env python3

"""Define a class that automates the processing of light curves."""
import os

if os.name == "posix":
    from os import getpgid, setsid, fork
#pylint: disable=wrong-import-position
from os import path, getpid

import logging
import sys
import subprocess

from sqlalchemy import select, and_, literal, update, sql, delete
import numpy

from autowisp.multiprocessing_util import setup_process
from autowisp import DataReductionFile, LightCurveFile
from autowisp.catalog import read_catalog_file
from autowisp.database.interface import Session
from autowisp.database.processing import ProcessingManager
from autowisp.database.user_interface import get_processing_sequence
from autowisp.light_curves.collect_light_curves import DecodingStringFormatter
from autowisp import processing_steps

# False positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    Image,
    ImageType,
    InputMasterTypes,
    LightCurveStatus,
    LightCurveProcessingProgress,
    MasterFile,
    MasterType,
    ProcessingSequence,
    Step,
    StepDependencies,
)

# pylint: enable=no-name-in-module
#pylint: enable=wrong-import-position


class LightCurveProcessingManager(ProcessingManager):
    """
    Utilities for automated processing of lightcurves.

    Attrs:
        See `ProcessingManager`.

        _pending(dict):    Keys are 'EPD' and 'TFA' and values are lists
            of single photometric reference filenames for which the given
            detrending step is pending.
    """

    def _mark_progress(self, which, status=1, final=True):
        """
        Record that current step has finished processing given lightcurve(s).

        Returns:
            None
        """

        if isinstance(which, int):
            which = [which]

        assert status > 0
        # False positivie
        # pylint: disable=no-member
        with Session.begin() as db_session:
            # pylint: enable=no-member
            for star in which:
                if isinstance(star, int):
                    src_id = star
                else:
                    with LightCurveFile(star, "r+") as light_curve:
                        src_id = int(light_curve["Identifiers"][0][1])

                if final:
                    db_session.execute(
                        delete(LightCurveStatus).filter_by(id=src_id)
                    )
                else:
                    db_session.execute(
                        update(LightCurveStatus)
                        .where(id=src_id)
                        .values(status=status)
                    )

    def _cleanup_interrupted(self, db_session):
        """Don't do anything for lightcurves."""

    def _get_lc_fnames(
        self, *, step, db_sphotref, catalog_fname, lc_fname, db_session
    ):
        """Return the lightcurves to be processed by the current step."""

        def check_add(src_id, lc_fname):
            """Check if the given LC exists and mark src_id started if so."""

            if path.exists(lc_fname):
                db_session.add(
                    LightCurveStatus(
                        id=src_id,
                        progress_id=self._current_processing.id,
                        status=0,
                    )
                )
                return True
            return False

        previous = db_session.scalar(
            select(
                # False positive
                # pylint: disable=not-callable
                sql.func.count(LightCurveProcessingProgress.id)
                # pylint: enable=not-callable
            ).where(
                LightCurveProcessingProgress.step_id == step.id,
                LightCurveProcessingProgress.single_photref_id
                == (db_sphotref.id),
                LightCurveProcessingProgress.configuration_version
                == (self.step_version[step.name]),
                LightCurveProcessingProgress.id != self._current_processing.id,
            )
        )

        if previous:
            source_list = db_session.scalars(
                select(LightCurveStatus.id)
                .join(LightCurveProcessingProgress)
                .where(
                    LightCurveProcessingProgress.step_id == step.id,
                    LightCurveProcessingProgress.single_photref_id
                    == (db_sphotref.id),
                    LightCurveProcessingProgress.configuration_version
                    == (self.step_version[step.name]),
                )
            ).all()
        else:
            self._logger.debug(
                "Reading LC catalog file: %s", repr(catalog_fname)
            )
            catalog = read_catalog_file(catalog_fname)
            source_list = catalog.index
        srcid_formatter = DecodingStringFormatter()

        lc_fnames = map(
            lambda src_id: srcid_formatter.format(
                lc_fname, *numpy.atleast_1d(src_id)
            ),
            source_list,
        )
        if previous:
            lc_fnames = list(lc_fnames)
            for check in lc_fnames:
                assert path.exists(check)
            return lc_fnames
        return [
            lc
            for src_id, lc in zip(source_list, lc_fnames)
            if check_add(src_id, lc)
        ]

    def _specialize_config(
        self, *, step, step_config, db_sphotref, catalog, db_session
    ):
        """Add parts of configuration for step that depend on database."""

        step_config["image_type"] = self._current_image_type
        step_config["processing_step"] = step.name
        match_sphotref = f'sphotref == {db_sphotref.filename.encode("ascii")!r}'
        if not step_config["lc_points_filter_expression"]:
            step_config["lc_points_filter_expression"] = match_sphotref
        else:
            step_config["lc_points_filter_expression"] = (
                f'({step_config["lc_points_filter_expression"]}) and '
                + match_sphotref
            )
        for (
            option_name,
            input_master_type,
            input_master_type_id,
        ) in db_session.execute(
            select(InputMasterTypes.config_name, MasterType.name, MasterType.id)
            .join(MasterType)
            .join(ImageType, InputMasterTypes.image_type_id == ImageType.id)
            .where(
                InputMasterTypes.step_id == step.id,
                ImageType.name == self._current_image_type,
            )
        ).all():
            option_name = option_name.replace("-", "_")
            if input_master_type == "single_photref":
                step_config[option_name] = db_sphotref.filename
            elif input_master_type == "lightcurve_catalog":
                step_config[option_name] = catalog
            else:
                step_config[option_name] = db_session.scalar(
                    select(MasterFile.filename)
                    .join(
                        LightCurveProcessingProgress,
                        MasterFile.progress_id
                        == LightCurveProcessingProgress.id,
                    )
                    .where(
                        LightCurveProcessingProgress.single_photref_id
                        == db_sphotref.id
                    )
                    .where(MasterFile.type_id == input_master_type_id)
                )

    def _start_step(
        self,
        *,
        step,
        db_sphotref_image,
        sphotref_header,
        db_sphotref,
        db_session,
    ):
        """
        Record start of processing and return the LCs and configuration to use.

        Args:
            step(Step):    The database step to start.

            db_sphotref_image(Image):    The Image database object
                corresponding to the single photometric reference for which
                processing is starting.

            sphotref_header(dict-like):    The header of the single
                photometric reference.

            db_sphotref(MasterFile):    The database MasterFile instance
                corresponding to the single photometric reference.

            db_session:    Session for querying the database.

        Returns:
            []:
                List of the lightcurves to process

            dict:
                The complete configuration to use for the specified processing.
        """

        self._create_current_processing(
            step, ("single_photref", db_sphotref.id), db_session
        )

        catalog, step_config, lc_fname = self.get_step_config(
            step=step,
            db_sphotref=db_sphotref,
            db_sphotref_image=db_sphotref_image,
            sphotref_header=sphotref_header,
            db_session=db_session,
        )
        return (
            self._get_lc_fnames(
                step=step,
                db_sphotref=db_sphotref,
                catalog_fname=catalog,
                lc_fname=lc_fname,
                db_session=db_session,
            ),
            step_config,
        )

    def _check_ready(self, step, image_type, single_photref_fname, db_session):
        """
        Check if the given type of images is ready to process with given step.

        Args:
            step(Step):    The step to check for readiness.

            image_type_id(int):    The ID of the type of image to check for
                readiness (the image type of the single photometric reference).

            single_photref_fname(str):    The filename of the single photometric
                reference identifying light curve points to process.

            db_session(Session):    The database session to use.

        Returns:
            bool:    Whether all requirements for the specified processing are
                satisfied.
        """

        for required_step_name, required_imtype_id in db_session.execute(
            select(
                Step.name,
                StepDependencies.blocking_image_type_id,
            )
            .select_from(StepDependencies)
            .join(
                Step,
                StepDependencies.blocking_step_id == Step.id,
            )
            .where(StepDependencies.blocked_step_id == step.id)
            .where(StepDependencies.blocked_image_type_id == image_type.id)
        ).all():
            assert required_imtype_id == image_type.id
            if (
                required_step_name in self.pending
                and image_type.name in self.pending[required_step_name]
                and (
                    single_photref_fname
                    in self.pending[required_step_name][image_type.name]
                )
            ):
                self._logger.debug(
                    "Not ready for %s of lightcurve points corresponding to "
                    "single photometric reference %s because %s is pending.",
                    step.name,
                    repr(single_photref_fname),
                    required_step_name,
                )
                return False
        return True

    def _prepare_processing(
        self, step_name, single_photref_fname, limit_to_steps
    ):
        """Prepare for processing images of given type by a calibration step."""

        if limit_to_steps is not None and step_name not in limit_to_steps:
            self._logger.debug(
                "Skipping disabled %s for single photometric reference: %s",
                step_name,
                repr(single_photref_fname),
            )
            return None, None

        # pylint: disable=no-member
        with (
            Session.begin() as db_session,
            DataReductionFile(single_photref_fname, "r") as sphotref_dr,
        ):
            # pylint: enable=no-member
            db_sphotref = db_session.scalar(
                select(MasterFile).where(
                    MasterFile.filename == single_photref_fname
                )
            )
            step = db_session.scalar(select(Step).where(Step.name == step_name))
            header = sphotref_dr.get_frame_header()
            image = db_session.scalar(
                select(Image).where(
                    Image.raw_fname.contains(header["RAWFNAME"] + ".fits")
                )
            )
            self.evaluate_expressions_image(image, db_session)

            self._current_image_type = image.image_type.name
            setup_process(
                task="main",
                parent_pid="",
                processing_step=step_name,
                image_type=self._current_image_type,
                **self._processing_config,
            )

            if not self._check_ready(
                step, image.image_type, single_photref_fname, db_session
            ):
                return None, None

            return self._start_step(
                step=step,
                db_sphotref_image=image,
                sphotref_header=header,
                db_sphotref=db_sphotref,
                db_session=db_session,
            )

    def __init__(self, *args, **kwargs):
        """Initialize self._current_image_type in addition to normali init."""

        self._current_image_type = None
        super().__init__(*args, **kwargs)
        # pylint: disable=no-member
        with Session.begin() as db_session:
            # pylint: enable=no-member
            self.set_pending(db_session)

    @staticmethod
    def select_step_sphotref(db_session, pending=True, full_objects=False):
        """Return pening or non-pending step/single photref combinations."""

        master_cat_id = db_session.scalar(
            select(MasterType.id).where(MasterType.name == "lightcurve_catalog")
        )
        create_lc_step_id = db_session.scalar(
            select(Step.id).where(Step.name == "create_lightcurves")
        )
        return db_session.execute(
            select(
                Step if full_objects else Step.name,
                MasterFile if full_objects else MasterFile.filename,
            )
            .select_from(ProcessingSequence)
            .join(Step)
            .join(MasterFile, literal(True))
            .join(MasterType)
            .join(InputMasterTypes, InputMasterTypes.step_id == Step.id)
            .join(StepDependencies, StepDependencies.blocked_step_id == Step.id)
            .outerjoin(
                LightCurveProcessingProgress,
                and_(
                    (LightCurveProcessingProgress.step_id == Step.id),
                    (
                        LightCurveProcessingProgress.single_photref_id
                        == MasterFile.id
                    ),
                    LightCurveProcessingProgress.final == True,
                ),
            )
            .where(StepDependencies.blocking_step_id == create_lc_step_id)
            .where(MasterType.name == "single_photref")
            .where(InputMasterTypes.master_type_id == master_cat_id)
            .where(
                # pylint: disable=singleton-comparison
                LightCurveProcessingProgress.final == None
                if pending
                else LightCurveProcessingProgress.final == True
                # pylint: enable=singleton-comparison
            )
        ).all()

    def set_pending(self, db_session):
        """
        Set the unprocessed images and channels split by step and image type.

        Args:
            db_session(Session):    The database session to use.


        Returns:
            {step name: [str, ...]}:
                The filenames of the single photometric reference DR files for
                which lightcurves exist but the given steps has not been
                performend yet.
        """

        pending = self.select_step_sphotref(db_session)
        print("\n\t" + "\n\t".join([repr(e) for e in pending]))
        for step, sphotref_fname in pending:
            # pylint: disable=no-member
            with DataReductionFile(sphotref_fname, "r") as sphotref_dr:
                # pylint: enable=no-member
                image_type_name = db_session.scalar(
                    select(ImageType.name)
                    .select_from(Image)
                    .join(ImageType)
                    .where(
                        Image.raw_fname.contains(
                            sphotref_dr.get_frame_header()["RAWFNAME"] + ".fits"
                        )
                    )
                )

            if step not in self.pending:
                self.pending[step] = {}
            if image_type_name not in self.pending[step]:
                self.pending[step][image_type_name] = []
            self.pending[step][image_type_name].append(sphotref_fname)

    def get_step_config(
        self,
        *,
        step,
        db_sphotref,
        db_sphotref_image,
        sphotref_header,
        db_session,
    ):
        """Return the configuration to use for the given step/sphotref combo."""

        matched_expressions = self._evaluated_expressions[db_sphotref_image.id][
            sphotref_header["CLRCHNL"]
        ]["matched"]
        create_lc_cofig = self.get_config(
            matched_expressions, db_session, step_name="create_lightcurves"
        )[0]

        catalog = create_lc_cofig["lightcurve_catalog_fname"].format_map(
            sphotref_header
        )
        assert path.exists(catalog)

        step_config = self.get_config(
            matched_expressions, db_session, db_step=step
        )[0]
        self._specialize_config(
            step=step,
            step_config=step_config,
            db_sphotref=db_sphotref,
            catalog=catalog,
            db_session=db_session,
        )
        return catalog, step_config, create_lc_cofig["lc_fname"]

    def __call__(self, limit_to_steps=None):
        """Perform all the processing for the given steps (all if None)."""

        # False positivie
        # pylint: disable=no-member
        with Session.begin() as db_session:
            # pylint: enable=no-member
            processing_sequence = [
                (step.name, imtype.name)
                for step, imtype in get_processing_sequence(db_session)
            ]

        for step_name, imtype_name in processing_sequence:
            if step_name not in self.pending:
                continue
            for single_photref_fname in self.pending[step_name][imtype_name][:]:
                lc_fnames, configuration = self._prepare_processing(
                    step_name, single_photref_fname, limit_to_steps
                )
                if lc_fnames is None:
                    continue

                self._logger.debug(
                    "Starting %s on %d lightcurves for single photref %s with "
                    "configuration:\n\t%s",
                    step_name,
                    len(lc_fnames),
                    repr(single_photref_fname),
                    "\n\t".join(
                        f"{key}: {value!r}"
                        for key, value in configuration.items()
                    ),
                )

                step_module = getattr(processing_steps, step_name)
                new_masters = getattr(step_module, step_name)(
                    lc_fnames, 0, configuration, self._mark_progress
                )
                # False positivie
                # pylint: disable=no-member
                with Session.begin() as db_session:
                    # pylint: enable=no-member
                    # False positive
                    # pylint: disable=not-callable
                    self._current_processing = db_session.merge(
                        self._current_processing
                    )
                    self._current_processing.finished = sql.func.now()
                    self._current_processing.final = True
                    # pylint: enable=not-callable

                if new_masters:
                    self.add_masters(
                        new_masters, step_name, self._current_image_type
                    )
                self.pending[step_name][imtype_name].remove(
                    single_photref_fname
                )


def main():
    """Avoid global variables."""

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
    manager = LightCurveProcessingManager()
    print("Pending: " + repr(manager.pending))
    manager()


if __name__ == "__main__":
    if os.name == "posix":  # Linux/macOS
        try:
            setsid()
        except OSError:
            print(f"pid={getpid():d}  pgid={getpgid(0):d}")

        pid = fork()
        if pid < 0:
            raise RuntimeError("fork fail")
        if pid != 0:
            sys.exit(0)

        setsid()
        main()  # Run main function in child process

    elif os.name == "nt":  # Windows
        try:
            subprocess.Popen(
                [sys.executable, sys.argv[0]]
                + sys.argv[1:],  # Relaunch with same arguments
                creationflags=DETACHED_PROCESS,
            )
            sys.exit(0)  # Exit parent process
        except Exception as e:
            sys.stderr.write(f"Failed to detach: {e}\n")
            sys.exit(1)
