#!/usr/bin/env python3
# pylint: disable=too-many-lines

"""Handle data processing DB interactions."""

import logging
import os

from os import path, getpid

import sys
import subprocess

from sqlalchemy import sql, select, update, and_, or_
from configargparse import ArgumentParser, DefaultsFormatter

from autowisp.multiprocessing_util import (
    setup_process,
    get_log_outerr_filenames,
)
from autowisp.database.processing import ProcessingManager
from autowisp.database.interface import Session
from autowisp.file_utilities import find_fits_fnames
from autowisp import processing_steps
from autowisp.database.user_interface import get_processing_sequence
from autowisp.data_reduction.data_reduction_file import DataReductionFile

# False positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    StepDependencies,
    ImageProcessingProgress,
    ProcessedImages,
    Step,
    Image,
    ObservingSession,
    MasterType,
    InputMasterTypes,
    Condition,
    ConditionExpression,
)
from autowisp.database.data_model.provenance import (
    Camera,
    CameraChannel,
    CameraType,
)

# pylint: enable=no-name-in-module


class NoMasterError(ValueError):
    """Raised when no suitable master can be found for a batch of frames."""


# Intended to be used as simple callable
# pylint: disable=too-few-public-methods
class ExpressionMatcher:
    """
    Compare condition expressions for an image/channel to a target.

    Usually check if matched expressions and master expression values are
    identical, but also handles special case of calibrate step.
    """

    def _get_master_values(self, image_id, channel):
        """Return ready to compare masster expression values."""

        if channel is None:
            return tuple(
                self._get_master_values(image_id, channel)
                for channel in sorted(
                    filter(None, self._evaluated_expressions[image_id].keys())
                )
            )
        self._logger.debug(
            "Getting master expression values for expression ids %s, "
            "image %d, channel %s",
            repr(self._master_expression_ids),
            image_id,
            channel,
        )
        return tuple(
            self._evaluated_expressions[image_id][channel]["values"][
                expression_id
            ]
            for expression_id in self._master_expression_ids
        )

    def __init__(
        self,
        evaluated_expressions,
        ref_image_id,
        ref_channel,
        master_expression_ids,
        *,
        masters_only=False,
    ):
        """
        Set up comparison to the given evaluated expressions.

        """

        self._logger = logging.getLogger(__name__)
        self._evaluated_expressions = evaluated_expressions
        self._master_expression_ids = master_expression_ids
        reference_evaluated = evaluated_expressions[ref_image_id][ref_channel]
        self._ref_matched = reference_evaluated["matched"]
        self.ref_master_values = self._get_master_values(
            ref_image_id, ref_channel
        )
        self._masters_only = masters_only
        self._logger.debug(
            "Finding images matching expressions %s and values %s",
            repr(self._ref_matched),
            repr(self.ref_master_values),
        )

    def __call__(self, image_id, channel):
        """True iff the expressions for the given image/channel match."""

        image_evaluated = self._evaluated_expressions[image_id][channel]
        image_master_values = self._get_master_values(image_id, channel)

        self._logger.debug(
            "Comparing %s to %s and %s to %s",
            repr(image_evaluated["matched"]),
            repr(self._ref_matched),
            repr(image_master_values),
            repr(self.ref_master_values),
        )
        return (
            self._masters_only
            or image_evaluated["matched"] == self._ref_matched
        ) and image_master_values == self.ref_master_values


# pylint: enable=too-few-public-methods


def get_master_expression_ids(step_id, image_type_id, db_session):
    """
    List all condition expression IDs determining input or output masters.

    Args:
        step_id(int):    The ID of the step for which to return the master
            expression IDs.

        image_type_id(int):     The type of images being processed by the step
            for which to return the master expression IDs.

    Returns:
        [int]:
            The combined expression IDs reqired to determine which required
            masters can be used for the given step or which masters will be
            created by it.
    """

    return sorted(
        set(
            db_session.scalars(
                select(ConditionExpression.id)
                .select_from(InputMasterTypes)
                .join(MasterType)
                .join(
                    Condition,
                    # False positive
                    # pylint: disable=no-member
                    MasterType.condition_id == Condition.id,
                    # pylint: enable=no-member
                )
                .join(ConditionExpression)
                .where(InputMasterTypes.step_id == step_id)
                .where(InputMasterTypes.image_type_id == image_type_id)
                .group_by(ConditionExpression.id)
            ).all()
            + db_session.scalars(
                select(ConditionExpression.id)
                .select_from(MasterType)
                .join(
                    Condition,
                    or_(
                        # False positive
                        # pylint: disable=no-member
                        MasterType.condition_id == Condition.id,
                        (
                            MasterType.maker_image_split_condition_id
                            == Condition.id
                        ),
                        # pylint: enable-no-member
                    ),
                )
                .join(ConditionExpression)
                .where(MasterType.maker_step_id == step_id)
                .where(MasterType.maker_image_type_id == image_type_id)
            ).all()
        )
    )


def remove_failed_prerequisite(
    pending, pending_image_type_id, prereq_step_id, db_session
):
    """Remove from pending any entries that failed the prerequisite step."""

    prereq_statuses = [
        db_session.execute(
            select(ProcessedImages.status)
            .outerjoin(ImageProcessingProgress)
            .where(
                (ProcessedImages.image_id == image.id),
                ProcessedImages.channel == channel,
                ImageProcessingProgress.step_id == prereq_step_id,
                (
                    ImageProcessingProgress.image_type_id
                    == pending_image_type_id
                ),
            )
        ).scalar_one_or_none()
        for image, channel, _ in pending
    ]
    dropped = []
    for i in range(len(pending) - 1, -1, -1):
        if prereq_statuses[i] and prereq_statuses[i] < 0:
            dropped.append(pending.pop(i))

    return dropped


# pylint: disable=too-many-instance-attributes
class ImageProcessingManager(ProcessingManager):
    """
    Read configuration and record processing progress in the database.

    Attrs:
        See `ProcessingManager`.

        pending(dict):    Indexed by step ID, and image type ID list of
            (Image, channel name, status) tuples listing all the images of the
            given type that have not been processed by the currently selected
            version of the step in the key and their status if previous
            processing by that step was interrupted or None if not.

        _failed_dependencies(dict):    Dictionary with keys (step, image_type)
            that contains the list of images and channels that failed the given
            step.
    """

    def _set_calibration_config(self, config, first_image):
        """Retrun the specially formatted argument for the calibration step."""

        config["split_channels"] = self._get_split_channels(first_image)
        config["extra_header"] = {
            "OBS-SESN": first_image.observing_session.label
        }
        result = {
            (
                "split_channels",
                "".join(
                    repr(c)
                    for c in first_image.observing_session.camera.channels
                ),
            ),
            ("observing_session", config["extra_header"]["OBS-SESN"]),
        }
        self._logger.debug(
            "Calibration step configuration:\n%s",
            "\n\t".join((f"{k}: {v!r}" for k, v in config.items())),
        )
        return result

    def _split_by_master(self, batch, input_master_type):
        """Split the given list of images by the best master of given type."""

        result = {}

        for image, channel, status in batch:
            if channel is None:
                best_master = tuple(
                    (
                        channel,
                        self._evaluated_expressions[image.id][channel][
                            "masters"
                        ][input_master_type.master_type.name],
                    )
                    for channel in sorted(
                        filter(
                            None, self._evaluated_expressions[image.id].keys()
                        )
                    )
                )
            else:
                best_master = self._evaluated_expressions[image.id][channel][
                    "masters"
                ][input_master_type.master_type.name]

            if best_master in result:
                result[best_master].append((image, channel, status))
            else:
                result[best_master] = [(image, channel, status)]
        return result

    # Could not find good way to simplify
    # pylint: disable=too-many-locals
    def _get_batch_config(
        self, batch, master_expression_values, step, db_session
    ):
        """
        Split given batch of images by configuration for given step.

        The batch must already be split by all relevant condition expressions.
        Only splits batches by the best master for each image.

        Args:
            batch([Image, channel, status]):    List of database image instances
                and for channels which to find the configuration(s). The channel
                should be ``None`` for the ``calibrate`` step

            master_expression_values(tuple):    The values the expressions
                required to select input masters or to guarantee a unique output
                master. Should be provided in consistent order for all batches
                processed by the same step.

            step(Step):    The database step instance to configure.

            db_session:    Database session to use for queries.

        Returns:
            dict:
                keys:    guaranteed to match iff configuration, output master
                    conditions, and all best input master(s) match. In other
                    words, if this function is called separately on multiple
                    batches, it is safe to combine and process together those
                    that end up with the same key.

                values:
                    dict:    The configuration to use for the given (sub-)batch.

                    [Image]:     The (sub-)batch of images to process with given
                        configuration.
        """

        self._logger.debug("Finding configuration for batch: %s", repr(batch))
        first_image_expressions = self._evaluated_expressions[batch[0][0].id]
        config, config_key = self.get_config(
            first_image_expressions[batch[0][1]]["matched"],
            db_session,
            db_step=step,
        )
        config_key |= {master_expression_values}
        if step.name == "calibrate":
            config_key |= self._set_calibration_config(config, batch[0][0])
        config["processing_step"] = step.name
        config["image_type"] = batch[0][0].image_type.name

        result = {config_key: (config, batch)}
        for input_master_type in db_session.scalars(
            select(InputMasterTypes).filter_by(
                step_id=step.id, image_type_id=batch[0][0].image_type_id
            )
        ).all():
            for config_key, (config, sub_batch) in list(result.items()):
                del result[config_key]
                splits = self._split_by_master(sub_batch, input_master_type)

                for best_master, sub_batch in splits.items():
                    if best_master is None:
                        if input_master_type.optional:
                            assert config_key not in result
                            result[config_key] = (config, sub_batch)
                        else:
                            result[None] = (
                                "No master "
                                + input_master_type.master_type.name
                                + " found!",
                                sub_batch,
                            )

                    else:
                        new_config = dict(config)
                        new_config[
                            input_master_type.config_name.replace("-", "_")
                        ] = (
                            best_master
                            if isinstance(best_master, str)
                            else dict(best_master)
                        )
                        key_extra = {
                            (input_master_type.config_name, best_master)
                        }
                        result[config_key | key_extra] = (new_config, sub_batch)

        return result

    # pylint: enable=too-many-locals

    def _clean_pending_per_dependencies(
        self, db_session, from_step_id=None, from_image_type_id=None
    ):
        """Remove pending images from steps if they failed a required step."""

        dropped = {}
        for (step_id, image_type_id), pending in self.pending.items():
            if (
                from_image_type_id is not None
                and image_type_id != from_image_type_id
            ):
                continue
            for prereq_step_id in db_session.scalars(
                select(StepDependencies.blocking_step_id).where(
                    StepDependencies.blocked_step_id == step_id,
                    StepDependencies.blocked_image_type_id == image_type_id,
                    StepDependencies.blocking_image_type_id == image_type_id,
                )
            ):
                if from_step_id is not None and prereq_step_id != from_step_id:
                    continue
                if (step_id, image_type_id) not in dropped:
                    dropped[(step_id, image_type_id)] = []

                failed_prereq = remove_failed_prerequisite(
                    pending, image_type_id, prereq_step_id, db_session
                )
                self.pending[(step_id, image_type_id)] = pending

                dropped[(step_id, image_type_id)].extend(failed_prereq)

                self._logger.info(
                    "The following image/channel combinations failed %s. "
                    "Excluding from %s:\n\t%s",
                    db_session.scalar(
                        select(Step.name).filter_by(id=prereq_step_id)
                    ),
                    db_session.scalar(select(Step.name).filter_by(id=step_id)),
                    "\n\t".join(
                        image.raw_fname + ":" + channel
                        for image, channel in failed_prereq
                    ),
                )

        return dropped

    def _check_ready(self, step, image_type, db_session):
        """
        Check if the given type of images is ready to process with given step.

        Args:
            step(Step):    The step to check for readiness.

            image_type(ImageType):    The type of images to check for readiness.

            db_session(Session):    The database session to use.

        Returns:
            bool:    Whether all requirements for the specified processing are
                satisfied.
        """

        for requirement in db_session.execute(
            select(
                StepDependencies.blocking_step_id,
                StepDependencies.blocking_image_type_id,
            )
            .where(StepDependencies.blocked_step_id == step.id)
            .where(StepDependencies.blocked_image_type_id == image_type.id)
        ).all():
            if self.pending[requirement]:
                self._logger.debug(
                    "Not ready for %s of %d %s frames because of %d pending %s "
                    "type ID images for step ID %s:\n\t%s",
                    step.name,
                    len(self.pending[(step.id, image_type.id)]),
                    image_type.name,
                    len(self.pending[requirement]),
                    requirement[1],
                    requirement[0],
                    "\n\t".join(
                        f"{e[0]!r}: {e[1]!r}" for e in self.pending[requirement]
                    ),
                )
                return False
        return True

    def _get_interrupted(self, need_cleanup, db_session):
        """Return list of interrupted files and configuration for cleanup."""

        self.current_step = need_cleanup[0][2]
        self._current_processing = db_session.scalar(
            select(ImageProcessingProgress).where(
                ImageProcessingProgress.id == need_cleanup[0][1].progress_id
            )
        )
        input_type = getattr(
            processing_steps, self.current_step.name
        ).input_type

        for entry in need_cleanup:
            assert entry[2] == self.current_step

        pending = [
            (
                image,
                None if input_type == "raw" else processed.channel,
                processed.status,
            )
            for image, processed, _ in need_cleanup
        ]

        for image, _, __ in need_cleanup:
            if image.id not in self._evaluated_expressions:
                self.evaluate_expressions_image(image, db_session)

        cleanup_batches = self._get_config_batches(
            pending, input_type, db_session
        )
        result = {}
        for (config_key, status), (config, batch) in cleanup_batches.items():
            if config_key not in result:
                result[config_key] = (config, [])
            result[config_key][1].extend([(fname, status) for fname in batch])

        return list(result.values())

    def _cleanup_interrupted(self, db_session):
        """Cleanup previously interrupted processing for the current step."""

        need_cleanup = db_session.execute(
            select(Image, ProcessedImages, Step)
            .join(ProcessedImages)
            .join(ImageProcessingProgress)
            .join(Step)
            .where(~ProcessedImages.final)
            .order_by(Step.name)
        ).all()

        if not need_cleanup:
            return

        step_module = getattr(processing_steps, need_cleanup[0][2].name)

        for config, interrupted in self._get_interrupted(
            need_cleanup, db_session
        ):
            self._logger.warning(
                "Cleaning up interrupted %s processing of %d images:\n"
                "%s\n"
                "config: %s",
                need_cleanup[0][2],
                len(interrupted),
                repr(interrupted),
                repr(config),
            )
            new_status = step_module.cleanup_interrupted(interrupted, config)
            for _, processed, _ in need_cleanup:
                assert new_status >= -1
                assert new_status <= processed.status
                if new_status == -1:
                    db_session.delete(processed)
                else:
                    processed.status = new_status

    def _init_processed_ids(self, image, channels, step_input_type):
        """Prepare to record processing of the given image by current step."""

        if channels == [None]:
            channels = self._evaluated_expressions[image.id].keys()

        for channel_name in channels:
            if channel_name is None:
                continue

            step_input_fname = self.get_step_input(
                image, channel_name, step_input_type
            )

            if step_input_fname not in self._processed_ids:
                self._processed_ids[step_input_fname] = []
            self._processed_ids[step_input_fname].append(
                {"image_id": image.id, "channel": channel_name}
            )

    def _start_step(self, step, image_type, db_session):
        """
        Record the start of a processing step and return the images to process.

        Args:
            step(Step):    The database step to start.

            image_type(ImageType):    The database type of image to start
                processing.

            db_session:    Active session for database queries.

        Returns:
            [(Image, str)]:
                The list of images and channels to process.

            str:
                The type of input expected by the current step.
        """

        self._create_current_processing(
            step, ("image_type", image_type.id), db_session
        )

        pending_images = self.pending[(step.id, image_type.id)].copy()
        for image, channel, status in self._failed_dependencies.get(
            (step.id, image_type.id), []
        ):
            self._logger.info(
                "Prerequisite failed for %s of %s", step.name, image
            )
            db_session.add(
                ProcessedImages(
                    image_id=image.id,
                    channel=channel,
                    progress_id=self._current_processing.id,
                    status=-1,
                    final=True,
                )
            )
            self._some_failed = True

        self._processed_ids = {}
        step_input_type = getattr(processing_steps, step.name).input_type

        if step_input_type == "raw":
            added = set()
            new_pending = []
            for image, _, status in pending_images:
                if image.id not in added:
                    added.add(image.id)
                    new_pending.append((image, None, status))
            pending_images = new_pending

        for image, channel_name, _ in pending_images:
            self.evaluate_expressions_image(image, db_session)
            self._init_processed_ids(image, [channel_name], step_input_type)

        self._logger.info(
            "Starting %s step for %d %s images",
            self.current_step.name,
            len(pending_images),
            image_type.name,
        )

        return pending_images, step_input_type

    def _process_batch(
        self, batch, *, start_status, config, step_name, image_type_name
    ):
        """Run the current step for a batch of images given configuration."""

        step_module = getattr(processing_steps, step_name)

        new_masters = getattr(step_module, step_name)(
            batch,
            start_status,
            config,
            self._start_processing,
            self._end_processing,
        )
        if new_masters:
            self.add_masters(new_masters, step_name, image_type_name)

    def _start_processing(self, input_fname, status=0):
        """
        Mark in the database that processing the given file has begun.

        Args:
            input_fname:    The filename of the input (DR or FITS) that is about
                to begin processing.

        Returns:
            None
        """

        assert self.current_step is not None
        assert self._current_processing is not None
        self._logger.debug(
            "Starting processing IDs: %s",
            repr(self._processed_ids[input_fname]),
        )
        # False positivie
        # pylint: disable=no-member
        with Session.begin() as db_session:
            # pylint: enable=no-member
            for starting_id in self._processed_ids[input_fname]:
                db_session.add(
                    ProcessedImages(
                        **starting_id,
                        progress_id=self._current_processing.id,
                        status=status,
                        final=False,
                    )
                )

    def _end_processing(self, input_fname, status=1, final=True):
        """
        Record that the current step has finished processing the given file.

        Args:
            input_fname:    The filename of the input (DR or FITS) that was
                processed.

        Returns:
            None
        """

        assert self.current_step is not None
        assert self._current_processing is not None
        assert status != -1

        if status < 0:
            self._some_failed = True
        self._logger.debug(
            "Finished processing %s", repr(self._processed_ids[input_fname])
        )
        # False positivie
        # pylint: disable=no-member
        with Session.begin() as db_session:
            # pylint: enable-no-member
            for finished_id in self._processed_ids[input_fname]:
                db_session.execute(
                    update(ProcessedImages)
                    .where(ProcessedImages.image_id == finished_id["image_id"])
                    .where(ProcessedImages.channel == finished_id["channel"])
                    .where(
                        ProcessedImages.progress_id
                        == self._current_processing.id
                    )
                    .values(status=status, final=final)
                )

    # No good way to simplify
    # pylint: disable=too-many-locals
    def _get_config_batches(self, pending_images, step_input_type, db_session):
        """Return the batches of images to process with identical config."""

        result = {}
        check_image_type_id = pending_images[0][0].image_type_id
        for (
            by_condition,
            master_expression_values,
        ) in self.group_pending_by_conditions(
            pending_images,
            db_session,
            match_observing_session=self.current_step.name == "calibrate",
        ):
            for config_key, (config, batch) in self._get_batch_config(
                by_condition,
                master_expression_values,
                self.current_step,
                db_session,
            ).items():
                if config_key is None:
                    self._logger.warning(
                        "Excluding the following images from %s:\n\t%s",
                        config,
                        "\n\t".join(
                            [
                                self.get_step_input(
                                    image, channel, step_input_type
                                )
                                for image, channel, _ in batch
                            ]
                        ),
                    )
                    continue
                for image, channel, status in batch:
                    assert image.image_type_id == check_image_type_id

                    if (config_key, status) not in result:
                        result[config_key, status] = (config, [])
                    result[config_key, status][1].append(
                        self.get_step_input(image, channel, step_input_type)
                    )

        return result

    # pylint: enable=too-many-locals

    def _prepare_processing(self, step, image_type, limit_to_steps):
        """Prepare for processing images of given type by a calibration step."""

        # pylint: disable=no-member
        with Session.begin() as db_session:
            # pylint: enable=no-member
            setup_process(
                task="main",
                parent_pid="",
                processing_step=step.name,
                image_type=image_type.name,
                **self._processing_config,
            )
            step = db_session.merge(step)
            image_type = db_session.merge(image_type)

            self.set_pending(db_session, [(step, image_type)])

            if limit_to_steps is not None and step.name not in limit_to_steps:
                self._logger.debug(
                    "Skipping disabled %s for %s frames",
                    step.name,
                    image_type.name,
                )
                return step.name, image_type.name, None

            if not self._check_ready(step, image_type, db_session):
                return step.name, image_type.name, None

            pending_images, step_input_type = self._start_step(
                step, image_type, db_session
            )
            if not pending_images:
                return step.name, image_type.name, None

            return (
                step.name,
                image_type.name,
                self._get_config_batches(
                    pending_images, step_input_type, db_session
                ),
            )

    def _finalize_processing(self):
        """Update database and instance after processing."""

        # pylint: disable=no-member
        with Session.begin() as db_session:
            # pylint: enable=no-member
            self._current_processing = db_session.merge(
                self._current_processing
            )
            self._current_processing.finished = (
                # False positive
                # pylint: disable=not-callable
                sql.func.now()
                # pylint: enable=not-callable
            )
            pending = self.pending[
                (
                    self._current_processing.step_id,
                    self._current_processing.image_type_id,
                )
            ]

            self._logger.info(
                "Removing from pending all successful images for "
                "progress: %s",
                self._current_processing,
            )
            for finished_image_id, finished_channel in db_session.execute(
                select(ProcessedImages.image_id, ProcessedImages.channel)
                .where(
                    ProcessedImages.progress_id == self._current_processing.id
                )
                .where(
                    # pylint: disable=singleton-comparison
                    ProcessedImages.final
                    == True
                    # pylint: enable=singleton-comparison
                )
                .where(
                    or_(ProcessedImages.status > 0, ProcessedImages.status < -1)
                )
            ).all():
                found = False
                for i, (image, channel, _) in enumerate(pending):
                    if (
                        image.id == finished_image_id
                        and channel == finished_channel
                    ):
                        assert not found
                        del pending[i]
                        found = True
                        break
                if not found:
                    self._logger.error(
                        "Completed image ID %d, channel %s not found in "
                        "pending for step ID %d, image type ID %d:\n\t%s",
                        finished_image_id,
                        finished_channel,
                        self._current_processing.step_id,
                        self._current_processing.image_type_id,
                        "\n\t".join(f"{e[0]!r}: {e[1]!r}" for e in pending),
                    )
                    raise RuntimeError("Finished non-pending image!")

                self.pending[
                    (
                        self._current_processing.step_id,
                        self._current_processing.image_type_id,
                    )
                ] = pending

    #            if self._some_failed:
    #                dropped = self._clean_pending_per_dependencies(
    #                    db_session,
    #                    self._current_processing.step_id,
    #                    self._current_processing.image_type_id
    #                )
    #                for step_imtype, dropped_images in dropped.items():
    #                    if step_imtype in self._failed_dependencies:
    #                        self._failed_dependencies[
    #                            step_imtype
    #                        ].extend(
    #                            dropped_images
    #                        )
    #                    else:
    #                        self._failed_dependencies[step_imtype] = (
    #                            dropped_images
    #                        )

    def __init__(self, *args, **kwargs):
        """Initialize self._failed_dependencies in addition to normali init."""

        self._failed_dependencies = {}
        super().__init__(*args, **kwargs)

    def get_step_input(self, image, channel_name, step_input_type):
        """Return the name of the file required by the current step."""

        if step_input_type == "raw":
            return image.raw_fname

        if step_input_type.startswith("calibrated"):
            return self._evaluated_expressions[image.id][channel_name][
                "calibrated"
            ]

        if step_input_type == "dr":
            return self._evaluated_expressions[image.id][channel_name]["dr"]

        raise ValueError(f"Invalid step input type {step_input_type}")

    def set_pending(self, db_session, steps_imtypes=None, invert=False):
        """
        Set the unprocessed images and channels split by step and image type.

        Set the self.pending attribute to a dictionary with format ``{(step.id,
        image_type.id): (Image, str)}``, containing the images and channels of
        the specified type for which the specified step has not applied with the
        current configuration.

        Args:
            db_session(Session):    The database session to use.

            steps_imtypes(Step, ImageType):    The step image type combinations
                to determine pending images for. If unspecified, the full
                processing sequence defined in the database is used.

            invert(bool):    If True, returns successfully completed (not
                failed) instead of pending.


        Returns:
            None
        """

        status_select = (
            select(
                ProcessedImages.image_id,
                ProcessedImages.channel,
                sql.func.max(ProcessedImages.status).label("status"),
            )
            .join(ImageProcessingProgress)
            .where(ProcessedImages.status > 0)
            .where(ProcessedImages.final == 0)
            .group_by(ProcessedImages.image_id, ProcessedImages.channel)
        )

        for step, image_type in steps_imtypes or get_processing_sequence(
            db_session
        ):
            failed_prereq_subquery = (
                select(ProcessedImages.image_id, ProcessedImages.channel)
                .select_from(StepDependencies)
                .join(
                    ImageProcessingProgress,
                    and_(
                        StepDependencies.blocking_step_id
                        == ImageProcessingProgress.step_id,
                        StepDependencies.blocking_image_type_id
                        == ImageProcessingProgress.image_type_id,
                    ),
                )
                .join(ProcessedImages)
                .where(StepDependencies.blocked_step_id == step.id)
                .where(StepDependencies.blocked_image_type_id == image_type.id)
                .where(ProcessedImages.status < 0)
                .group_by(ProcessedImages.image_id, ProcessedImages.channel)
                .subquery()
            )

            processed_subquery = (
                select(ProcessedImages.image_id, ProcessedImages.channel)
                .join(ImageProcessingProgress)
                .where(ImageProcessingProgress.step_id == step.id)
                .where(ImageProcessingProgress.image_type_id == image_type.id)
                .where(
                    ImageProcessingProgress.configuration_version
                    == self.step_version[step.name]
                )
                .where(ProcessedImages.final)
            )

            status_subquery = (
                status_select.where(ImageProcessingProgress.step_id == step.id)
                .where(ImageProcessingProgress.image_type_id == image_type.id)
                .where(
                    ImageProcessingProgress.configuration_version
                    == self.step_version[step.name]
                )
                .subquery()
            )

            if invert:
                processed_subquery = processed_subquery.where(
                    ProcessedImages.status > 0
                )
            processed_subquery = processed_subquery.subquery()

            query = (
                select(Image, CameraChannel.name, status_subquery.c.status)
                .join(
                    ObservingSession,
                )
                .join(Camera)
                .join(CameraType)
                .join(CameraChannel)
                .outerjoin(
                    processed_subquery,
                    # False positive
                    # pylint: disable=no-member
                    and_(
                        Image.id == processed_subquery.c.image_id,
                        CameraChannel.name == processed_subquery.c.channel,
                    ),
                    # pylint: enable=no-member
                )
                .outerjoin(
                    failed_prereq_subquery,
                    and_(
                        Image.id == failed_prereq_subquery.c.image_id,
                        CameraChannel.name == failed_prereq_subquery.c.channel,
                    ),
                )
                .outerjoin(
                    status_subquery,
                    and_(
                        Image.id == status_subquery.c.image_id,
                        CameraChannel.name == status_subquery.c.channel,
                    ),
                )
                .where(
                    # False positive
                    # pylint: disable=no-member
                    Image.image_type_id
                    == image_type.id
                    # pylint: enable=no-member
                )
            )
            # This is how NULL comparison is done in SQLAlchemy
            # pylint: disable=singleton-comparison
            if invert:
                query = query.where(processed_subquery.c.image_id != None)
            else:
                query = query.where(processed_subquery.c.image_id == None)

            self.pending[(step.id, image_type.id)] = db_session.execute(
                query.where(failed_prereq_subquery.c.image_id == None)
            ).all()

            self._failed_dependencies[(step.id, image_type.id)] = (
                db_session.execute(
                    query.where(failed_prereq_subquery.c.image_id != None)
                ).all()
            )
            # pylint: enable=singleton-comparison

            self._logger.debug(
                "%s is pending for %d and failed dependencies for %d %s images",
                step.name,
                len(self.pending[(step.id, image_type.id)]),
                len(self._failed_dependencies[(step.id, image_type.id)]),
                image_type.name,
            )

        self._logger.debug("Pending: %s", repr(self.pending))

    def group_pending_by_conditions(
        self,
        pending_images,
        db_session,
        *,
        match_observing_session=False,
        step_id=None,
        masters_only=False,
    ):
        """
        Group pendig_images by condition expression values.

        Args:
            pending_images([Image, str]):    A list of the images (instance of
                Image DB class) and channels to group.

            db_session:    Database session to use for querries.

            match_observing_session:    Whether each group of images needs to
                be from the same observing session.

            step_id(int):    The ID of the step for which to group the pending
                images. If not specified, defaults to the current step.

            masters_only:    If True, grouping is done only by the values
                expressions required to determine the input or output masters
                for the current step.

        Returns:
            [([Image, str], tuple)]:
                Each entry is contains a list of the image/channel combinations
                matching a unique set of conditions and the second entry is the
                master expression values for all images in the list.
        """

        image_type_id = pending_images[0][0].image_type_id
        result = []
        master_expression_ids = get_master_expression_ids(
            step_id or self.current_step.id, image_type_id, db_session
        )
        while pending_images:
            self._logger.debug(
                "Finding images matching the same expressions as image id %d, "
                "channel %s",
                pending_images[-1][0].id,
                pending_images[-1][1],
            )
            batch = []
            match_expressions = ExpressionMatcher(
                self._evaluated_expressions,
                pending_images[-1][0].id,
                pending_images[-1][1],
                master_expression_ids,
                masters_only=masters_only,
            )
            observing_session_id = pending_images[-1][0].observing_session_id

            for i in range(len(pending_images) - 1, -1, -1):
                if (
                    not match_observing_session
                    or pending_images[i][0].observing_session_id
                    == observing_session_id
                ) and match_expressions(
                    pending_images[i][0].id, pending_images[i][1]
                ):
                    batch.append(pending_images.pop(i))
                else:
                    self._logger.debug("Not a match")

            self._logger.debug(
                "Image batch:\n\t%s",
                "\n\t".join(
                    f"{image.raw_fname}: {channel} status {status}"
                    for image, channel, status in batch
                ),
            )

            result.append((batch, match_expressions.ref_master_values))
        return result

    def find_processing_outputs(self, processing_progress, db_session=None):
        """Return all logging and output filenames for given processing ID."""

        if db_session is None:
            # False positivie
            # pylint: disable=no-member
            # pylint: disable=redefined-argument-from-local
            with Session.begin() as db_session:
                # pylint: enable=no-member
                # pylint: enable=redefined-argument-from-local
                return self.find_processing_outputs(
                    processing_progress, db_session
                )

        if not isinstance(processing_progress, ImageProcessingProgress):
            return self.find_processing_outputs(
                db_session.scalar(
                    select(ImageProcessingProgress).filter_by(
                        id=processing_progress
                    )
                ),
                db_session,
            )

        main_fnames = get_log_outerr_filenames(
            existing_pid=processing_progress.process_id,
            task="*",
            parent_pid="",
            processing_step=processing_progress.step.name,
            image_type=processing_progress.image_type.name,
            **self._processing_config,
        )
        logging.info("Main fnames: %s", repr(main_fnames))
        assert len(main_fnames[0]) == len(main_fnames[1]) == 1

        return (
            tuple(fname[0] for fname in main_fnames),
            get_log_outerr_filenames(
                existing_pid="*",
                task="*",
                parent_pid=processing_progress.process_id,
                processing_step=processing_progress.step.name,
                image_type=processing_progress.image_type.name,
                **self._processing_config,
            ),
        )

    def __call__(self, limit_to_steps=None):
        """Perform all the processing for the given steps (all if None)."""

        # False positivie
        # pylint: disable=no-member
        with Session.begin() as db_session:
            # pylint: enable=no-member
            processing_sequence = get_processing_sequence(db_session)

        DataReductionFile.get_file_structure()

        for step, image_type in processing_sequence:
            (step_name, image_type_name, processing_batches) = (
                self._prepare_processing(step, image_type, limit_to_steps)
            )
            self._logger.debug(
                "At start of %s step for %s images, pending:\n\t%s",
                step_name,
                image_type_name,
                "\n\t".join(
                    f"{key!r}: {len(val)}" for key, val in self.pending.items()
                ),
            )
            if processing_batches is None:
                continue

            self._finalize_processing()
            for (_, start_status), (
                config,
                batch,
            ) in processing_batches.items():
                # False positivie
                # pylint: disable=no-member
                with Session.begin() as db_session:
                    # pylint: enable=no-member
                    self._create_current_processing(
                        step, ("image_type", image_type.id), db_session
                    )

                self._logger.debug(
                    "Starting %s for a batch of %d %s images from status %s "
                    "with config:\n%s",
                    step_name,
                    len(batch),
                    image_type_name,
                    start_status,
                    repr(config),
                )

                self._process_batch(
                    batch,
                    start_status=start_status,
                    config=config,
                    step_name=step_name,
                    image_type_name=image_type_name,
                )
                self._logger.debug(
                    "Processed %s batch of %d images.", step_name, len(batch)
                )
                self._finalize_processing()
                self._logger.debug(
                    "After processing batch, pending:\n\t%s",
                    "\n\t".join(
                        f"{key!r}: {len(val)}"
                        for key, val in self.pending.items()
                    ),
                )

                self._some_failed = False

    def add_raw_images(self, image_collection):
        """Add the given RAW images to the database for processing."""

        # pylint: disable=no-member
        with Session.begin() as db_session:
            # pylint: enable=no-member
            default_expression_id = db_session.scalar(
                select(ConditionExpression.id).where(
                    ConditionExpression.notes == "Default expression"
                )
            )
            configuration = self.get_config(
                {default_expression_id},
                db_session,
                step_name="add_images_to_db",
            )[0]
        processing_steps.add_images_to_db.add_images_to_db(
            image_collection, configuration
        )


# pylint: enable=too-many-instance-attributes


def parse_command_line():
    """Return the command line configuration."""

    parser = ArgumentParser(
        description="Manually invoke the fully automated processing",
        default_config_files=[],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False,
    )
    parser.add_argument(
        "--add-raw-images",
        "-i",
        nargs="+",
        default=[],
        help="Before processing add new raw images for processing. Can be "
        "specified as a combination of image files and directories which will"
        "be searched for FITS files.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=None,
        help="Process using only the specified steps. Leave empty for full "
        "processing.",
    )
    parser.add_argument(
        "--detached",
        action="store_true",
        help="Indicates that the script is running as a detached process.",
    )
    logging.info("Parsed arguments: %s", parser.parse_args())
    return parser.parse_args()


def main(config):
    """Avoid global variables."""
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)

    logging.debug("Config add_raw_images: %s", config.add_raw_images)
    logging.debug("Config steps: %s", config.steps)

    processing = ImageProcessingManager()
    for img_to_add in config.add_raw_images:
        logging.debug("Adding raw images from: %s", img_to_add)
        processing.add_raw_images(find_fits_fnames(path.abspath(img_to_add)))

    logging.debug("Starting processing...")
    processing(limit_to_steps=config.steps)
    logging.debug("Processing completed.")


if __name__ == "__main__":
    if os.name == "posix":  # Linux/macOS
        from os import getpgid, setsid, fork

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
        main(parse_command_line())  # Run main function in child process

    elif os.name == "nt":  # Windows
        from subprocess import DETACHED_PROCESS

        if "--detached" not in sys.argv:
            try:
                with open("detached_process.log", "w") as log_file:
                    subprocess.Popen(
                        [
                            sys.executable,
                            os.path.abspath(sys.argv[0]),
                            "--detached",
                        ]
                        + sys.argv[1:],  # Relaunch with --detached
                        creationflags=DETACHED_PROCESS,
                        stdout=log_file,
                        stderr=log_file,
                    )
                sys.exit(0)  # Exit parent process
            except Exception as e:
                sys.stderr.write(f"Failed to detach: {e}\n")
                sys.exit(1)
        else:
            try:
                main(parse_command_line())
            except Exception as e:
                with open("detached_process_error.log", "w") as error_log:
                    error_log.write(f"Error in main: {e}\n")
