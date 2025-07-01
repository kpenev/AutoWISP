"""Define base class for processing images or lightcurves."""

from abc import ABC, abstractmethod
import logging
import os
from os import path, getpid
from tempfile import NamedTemporaryFile, TemporaryDirectory
from socket import getfqdn

from psutil import pid_exists, Process
from sqlalchemy import sql, select
from numpy import inf as infinity

from autowisp.multiprocessing_util import setup_process
from autowisp.database.interface import Session
from autowisp import Evaluator
from autowisp.fits_utilities import get_primary_header
from autowisp.image_calibration.fits_util import (
    add_required_keywords,
    add_channel_keywords,
)
from autowisp.database.user_interface import get_db_configuration
from autowisp.data_reduction.data_reduction_file import DataReductionFile
from autowisp.light_curves.light_curve_file import LightCurveFile
from autowisp import processing_steps

# False positive due to unusual importing
# pylint: disable=no-name-in-module
from autowisp.database.data_model import (
    Condition,
    ConditionExpression,
    Configuration,
    ImageType,
    ImageProcessingProgress,
    InputMasterTypes,
    LightCurveProcessingProgress,
    MasterFile,
    MasterType,
    Step,
)

# pylint: enable=no-name-in-module


class ProcessingInProgress(Exception):
    """Raised when a particular step is running in a different process/host."""

    def __init__(self, processing):
        self.step = processing.step.name
        if hasattr(processing, "image_type"):
            self.target = processing.image_type.name + " images"
        else:
            self.target = processing.sphotref.filename + " lightcurves"
        self.host = processing.host
        self.process_id = processing.process_id

    def __str__(self):
        return (
            f"Processing of {self.target} by {self.step} step on "
            f"{self.host!r} is still running with process id "
            f"{self.process_id!r}!"
        )


# pylint: disable=too-many-instance-attributes
class ProcessingManager(ABC):
    """
    Utilities for automated processing of images or lightcurves.

    Attrs:
        configuration(dict):    Indexed by parameter name with values further
            dictionaries with keys:

                ``version``: the actual version used including fallback

                ``value``: dict indexed by frozenset of expression IDs that an
                image must satisfy for the parameter to have a given value.

        condition_expressions({int: str}):    Dictionary of condition
            expressions that must be evaluated against the header of each input
            images to determine the exact values of the configuration parameters
            applicable to a given image. Keys are the condition expression IDs
            from the database and values are the actual expressions.

        step_version(dict):    Indexed by step name of the largest value of the
            actual version used for any parameter required by that step.

        current_step(Step):    The currently active step.

        _current_processing(ImageProcessingProgress):    The currently active
            step (the processing progress initiated the last time `start_step()`
            was called).

        _processed_ids(dict):    The keys are the filenames of the required
            inputs (DR or FITS) for the current step and the values are
            dictionaries with keys ``'image_id'`` and ``'channel'`` identifying
            what was processed.

        _evaluated_expressions(dict):    Indexed by image ID and then channel,
            dictionary containing dictionary with keys:

                * values: the values of the condition expressions for
                  the given image and channel indexed by their expression IDs.

                * matched: A set of the expression IDs for which the
                  corresponding expression converts to boolean True.

                * calibrated: the filename of the calibrated image

                * dr: the filename of the data reduction file

                * masters: A dictionary indexed by master type name of the best
                  master of the given type to apply to the image

            An additional entry with channel=None is included which contains
            just the common (intersection) set of expressions satisfied for all
            channels.

        _master_expressions(dict):    Indexed by master type, then tuple of
            expression values ordered by expression ID of the masters of the
            given type that match the given expression values.

        pending(dict):     Information about what images or lightcurves still
            need processing by the various steps. The format is different for
            image vs lightcurve processing managers.
    """

    def get_param_values(
        self, matched_expressions, parameters=None, db_session=None
    ):
        """
        Return the values to use for the given parameters.

        Args:
            matched_expressions(set):    Set of expression IDs that the image we
                are getting configuration for matches.

            parameters([] or str):    List of parameter names, or a step, or
                its name to get configuration for. Defaults to current step if
                not specified.

            as_args(bool):    If True, return a list of arguments ready to pass
                directly to one of the command line parser of the processing
                steps.

            db_session:    Session to use for DB queries. Only needed if
                specifying parameters by step name or using default.

        Returns:
            dict or list:    The values for the given parameters indexed by
                parameter name.
        """

        def get_value(param):
            """Return value for given parameter."""

            for required_expressions, value in self.configuration[param][
                "value"
            ].items():
                if required_expressions <= matched_expressions:
                    return value
            raise ValueError(f"No viable configuration found for {param}")

        if parameters is None:
            parameters = self.current_step

        if isinstance(parameters, str):
            parameters = [
                param.name
                for param in db_session.scalar(
                    select(Step).filter_by(name=parameters)
                ).parameters
            ]
        elif isinstance(parameters, Step):
            parameters = [param.name for param in parameters.parameters]

        return {param: get_value(param) for param in parameters}

    def _write_config_file(
        self,
        matched_expressions,
        outf,
        db_session,
        *,
        db_steps=None,
        step_names=None,
    ):
        """
        Write to given file configuration for given matched expressions.

        Returns:
            Set of tuples of parameters and values as set in the file. Used for
            comparing configurations.
        """

        # TODO: exclude master options
        if db_steps is None:
            if step_names is None:
                steps = db_session.scalars(select(Step).order_by(Step.id)).all()
            else:
                steps = [
                    db_session.execute(
                        select(Step).filter_by(name=name)
                    ).scalar_one()
                    for name in step_names
                ]
            return self._write_config_file(
                matched_expressions, outf, db_steps=steps, db_session=db_session
            )

        result = set()

        for step in db_steps:
            self._logger.debug("Getting configuration for %s step", step.name)
            outf.write(f"[{step.name}]\n")
            step_config = self.get_param_values(
                matched_expressions, [param.name for param in step.parameters]
            )
            self._logger.debug(
                "Adding configuration for %s step to config file", step.name
            )

            for param, value in step_config.items():
                if value is not None:
                    outf.write(f"    {param} = {value!r}\n")
                    self._logger.debug("    %s = %s", param, repr(value))
                    result.add((param, value))
                    self._logger.debug("    %s in result", param)

            outf.write("\n")

        return frozenset(result)

    def _get_best_master(self, candidate_masters, image_eval):
        """Find the best master from given list for given image/channel."""

        self._logger.debug(
            "Selecting best master for %s, channel %s from %s",
            repr(image_eval("RAWFNAME")),
            repr(image_eval("CLRCHNL")),
            repr(candidate_masters),
        )
        if not candidate_masters:
            return None

        best_master_value = infinity
        best_master_fname = None
        for master_fname, use_smallest in candidate_masters:
            assert use_smallest is not None
            master_value = image_eval(use_smallest)
            if master_value < best_master_value:
                best_master_value = master_value
                best_master_fname = master_fname
        assert best_master_fname
        return best_master_fname

    def _get_master(self, master_type, image_values, image_eval, db_session):
        """Return the master that should be used for the given image."""

        expressions = db_session.execute(
            select(ConditionExpression.id, ConditionExpression.expression)
            .join_from(
                Condition,
                ConditionExpression,
                Condition.expression_id == ConditionExpression.id,
            )
            .where(Condition.id == master_type.condition_id)
            .order_by(ConditionExpression.id)
        ).all()

        if master_type.name not in self._master_expressions:
            self._logger.debug(
                "Evaluating expressions for all masters of type: %s",
                repr(master_type.name),
            )
            self._master_expressions[master_type.name] = {}
            all_masters = db_session.execute(
                select(MasterFile.filename, MasterFile.use_smallest).filter_by(
                    type_id=master_type.id,
                    enabled=True,
                )
            ).all()
            for master in all_masters:
                master_eval = Evaluator(master.filename)
                expr_values = tuple(
                    master_eval(expr) for _, expr in expressions
                )
                if (
                    expr_values
                    not in self._master_expressions[master_type.name]
                ):
                    self._master_expressions[master_type.name][expr_values] = []
                self._master_expressions[master_type.name][expr_values].append(
                    master
                )

        expr_values = tuple(image_values[expr_id] for expr_id, _ in expressions)
        self._logger.debug(
            "Master %s available for: %s",
            master_type.name,
            repr(self._master_expressions[master_type.name].keys()),
        )

        candidates = self._master_expressions[master_type.name].get(
            expr_values, []
        )

        self._logger.debug("Candidate Masters: %s", repr(candidates))
        if len(candidates) == 1:
            return candidates[0].filename
        return self._get_best_master(candidates, image_eval)

    def _get_evaluated_entry(
        self, evaluate, image_type_id, calib_config, db_session
    ):
        """Return entry to add to self._evaluated_expressions."""

        evaluated_expressions = {
            "values": {
                expr_id: evaluate(expression)
                for expr_id, expression in self.condition_expressions.items()
            },
            "calibrated": calib_config["calibrated_fname"].format_map(
                evaluate.symtable
            ),
            "masters": {},
        }
        evaluated_expressions["matched"] = set(
            expr_id
            for expr_id, value in evaluated_expressions["values"].items()
            if value
        )

        for required_expressions, value in self.configuration[
            "data-reduction-fname"
        ]["value"].items():
            if required_expressions <= evaluated_expressions["matched"]:
                evaluated_expressions["dr"] = value.format_map(
                    evaluate.symtable
                )
                break
        assert "dr" in evaluated_expressions

        for master_type in db_session.scalars(
            select(MasterType)
            .join(InputMasterTypes)
            .where(InputMasterTypes.image_type_id == image_type_id)
            .distinct()
        ):
            if master_type.name not in ["epd_stat", "tfa_stat"]:
                evaluated_expressions["masters"][master_type.name] = (
                    self._get_master(
                        master_type,
                        evaluated_expressions["values"],
                        evaluate,
                        db_session,
                    )
                )

        return evaluated_expressions

    def get_matched_expressions(self, evaluate):
        """Return set of matching expressions given an evaluator for image."""

        def check(expr):
            """Return True if expression evaluates True."""

            try:
                return evaluate(expr)
            except NameError:
                return False

        return set(
            expr_id
            for expr_id, expression in self.condition_expressions.items()
            if check(expression)
        )

    def evaluate_expressions_image(self, image, db_session):
        """
        Return evaluator for header expressions for given image.

        Args:
            image(Image):     Instance of database Image for which to evaluate
                the condition expressions. The image header is augmented by
                ``IMAGE_TYPE`` keyword set to the name of the image type of the
                given image.

            db_session:    Used to select the best master.

            return_evaluator(bool):    Should an evaluator setup per the image
                header be returned for further use?

        Returns:
            Evaluator or None:
                Evaluator ready to evaluate additional expressions involving
                FITS headers. Only returned if ``return_evaluator`` is True.
        """

        if image.id in self._evaluated_expressions:
            return

        self._logger.debug("Evaluating expressions for: %s", repr(image))
        evaluate = Evaluator(get_primary_header(image.raw_fname, True))
        evaluate.symtable.update(
            IMAGE_TYPE=image.image_type.name,
            OBS_SESN=image.observing_session.label,
        )
        self._logger.debug(
            "Matched expressions: %s",
            repr(self.get_matched_expressions(evaluate)),
        )
        self._evaluated_expressions[image.id] = {}

        all_channel = {"matched": None, "values": None}
        for channel_name, channel_slice in self._get_split_channels(
            image
        ).items():
            self._logger.debug(
                "Adding channel keywords for channel %s of %s",
                channel_name,
                image.raw_fname,
            )
            add_channel_keywords(evaluate.symtable, channel_name, channel_slice)
            self._logger.debug(
                "Configuring for channel %s (%s) of %s",
                channel_name,
                evaluate.symtable["CLRCHNL"],
                image.raw_fname,
            )
            calib_config = self.get_config(
                self.get_matched_expressions(evaluate),
                db_session,
                step_name="calibrate",
            )[0]
            self._logger.debug(
                "Raw HDU for channel %s (%s) of %s: %s",
                channel_name,
                evaluate.symtable["CLRCHNL"],
                image.raw_fname,
                repr(calib_config["raw_hdu"]),
            )
            add_required_keywords(evaluate.symtable, calib_config, True)

            evaluated_expressions = self._get_evaluated_entry(
                evaluate, image.image_type_id, calib_config, db_session
            )

            if all_channel["matched"] is None:
                all_channel["matched"] = evaluated_expressions["matched"]
                all_channel["values"] = dict(evaluated_expressions["values"])
            else:
                all_channel["matched"] = (
                    all_channel["matched"] & evaluated_expressions["matched"]
                )
                # False positive
                # pylint: disable=unsubscriptable-object
                # pylint: disable=unsupported-delete-operation
                for expr_id in list(all_channel["values"].keys()):
                    if (
                        all_channel["values"][expr_id]
                        != evaluated_expressions["values"][expr_id]
                    ):
                        del all_channel["values"][expr_id]
                # pylint: enable=unsupported-delete-operation
                # pylint: enable=unsubscriptable-object

            self._evaluated_expressions[image.id][
                channel_name
            ] = evaluated_expressions

        self._evaluated_expressions[image.id][None] = {
            "matched": all_channel["matched"],
            "values": all_channel["values"],
        }
        self._logger.debug(
            "Evaluated expressions for image %s: %s",
            image,
            repr(self._evaluated_expressions[image.id]),
        )

    def get_product_fname(self, image_id, channel, product):
        """
        Return the ``dr`` or ``calibrated`` filename of specified image/channel.

        `self.evaluate_image_expressions()` must already have been called for
        this image.
        """

        return self._evaluated_expressions[image_id][channel][product]

    def get_master_fname(self, image_id, channel, master_type_name):
        """Return the filename of best master for a given image/channel."""

        return self._evaluated_expressions[image_id][channel]["masters"][
            master_type_name
        ]

    def _check_running_processing(
        self, running_processing, this_host, db_session
    ):
        """Check if any unfinished processing progresses are still running."""

        for processing in running_processing:
            if processing is not None and not processing.finished:
                if processing.host != this_host or (
                    pid_exists(processing.process_id)
                    and path.basename(
                        Process(processing.process_id).cmdline()[1]
                    )
                    == "processing.py"
                ):
                    raise ProcessingInProgress(processing)
                self._logger.warning(
                    "Processing progress %s appears to have crashed.",
                    processing,
                )
                # False positive
                # pylint: disable=not-callable
                processing.finished = sql.func.now()
                # pylint: enable=not-callable
                db_session.flush()

    def _create_current_processing(self, step, target, db_session):
        """Add a new ProcessingProgress at start of given step."""

        this_host = getfqdn()
        process_id = getpid()

        self.current_step = step

        progress_class = (
            ImageProcessingProgress
            if target[0] == "image_type"
            else LightCurveProcessingProgress
        )
        self._check_running_processing(
            db_session.scalars(
                select(progress_class).where(
                    (progress_class.step_id == self.current_step.id),
                    (getattr(progress_class, target[0] + "_id") == target[1]),
                    (
                        progress_class.configuration_version
                        == self.step_version[step.name]
                    ),
                )
            ).all(),
            this_host,
            db_session,
        )

        self._current_processing = progress_class(
            step_id=step.id,
            **{target[0] + "_id": target[1]},
            configuration_version=self.step_version[step.name],
            host=this_host,
            process_id=process_id,
            # False positive
            # pylint: disable=not-callable
            started=sql.func.now(),
            # pylint: enable=not-callable
            finished=None,
        )
        db_session.add(self._current_processing)
        db_session.flush()

    @abstractmethod
    def _cleanup_interrupted(self, db_session):
        """Cleanup previously interrupted processing for the current step."""

    def _get_split_channels(self, image):
        """Return the ``split_channels`` option for the given image."""

        return {
            channel.name: (
                slice(channel.y_offset, None, channel.y_step),
                slice(channel.x_offset, None, channel.x_step),
            )
            for channel in image.observing_session.camera.channels
        }

    def __init__(self, version=None, dummy=False):
        """
        Set the public class attributes per the given configuartion version.

        Args:
            version(int):    The version of the parameters to get. If a
                parameter value is not specified for this exact version use the
                value with the largest version not exceeding ``version``. By
                default us the latest configuration version in the database.

            dummy(bool):    If set to true, all logging is suppressed and no
                processing can be performed. Useful for reviewing the results
                of past processing.

        Returns:
            None
        """

        if dummy:
            logging.disable()
        DataReductionFile.get_file_structure()
        LightCurveFile.get_file_structure()

        self._logger = logging.getLogger(__name__)
        self.current_step = None
        self._current_processing = None
        self.configuration = {}
        self.condition_expressions = {}
        self._evaluated_expressions = {}
        self._master_expressions = {}
        self._processed_ids = {}
        self.pending = {}
        self._some_failed = False
        # False positivie
        # pylint: disable=no-member
        with Session.begin() as db_session:
            # pylint: enable=no-member
            if version is None:
                version = db_session.execute(
                    # False positivie
                    # pylint: disable=not-callable
                    # pylint: disable=no-member
                    select(sql.func.max(Configuration.version))
                    # pylint: enable=not-callable
                    # pylint: enable=no-member
                ).scalar_one()

            db_configuration = get_db_configuration(version, db_session)
            for config_entry in db_configuration:
                if config_entry.parameter.name not in self.configuration:
                    self.configuration[config_entry.parameter.name] = {
                        "version": config_entry.version,
                        "value": {},
                    }
                self.configuration[config_entry.parameter.name]["value"][
                    frozenset(
                        cond.expression_id for cond in config_entry.conditions
                    )
                ] = config_entry.value

                for cond in config_entry.conditions:
                    if cond.expression_id not in self.condition_expressions:
                        self.condition_expressions[cond.expression_id] = (
                            cond.expression.expression
                        )

            self._processing_config = self.get_config(
                self.get_matched_expressions(Evaluator()),
                db_session,
                step_name="add_images_to_db",
            )[0]
            del self._processing_config["processing_step"]
            del self._processing_config["image_type"]

            if not dummy:
                setup_process(
                    task="main",
                    parent_pid="",
                    processing_step="init_processing",
                    image_type="none",
                    **self._processing_config,
                )

            for master_type in db_session.scalars(select(MasterType)).all():
                for expression in (
                    master_type.match_expressions
                    + master_type.split_expressions
                ):
                    self.condition_expressions[expression.id] = (
                        expression.expression
                    )
            self._logger.debug(
                "Condition expressions to evaluate: %s",
                repr(self.condition_expressions),
            )

            self.step_version = {
                step.name: max(
                    self.configuration[param.name]["version"]
                    for param in step.parameters
                )
                for step in db_session.scalars(select(Step)).all()
            }

            if not dummy:
                self._cleanup_interrupted(db_session)

    def get_config(
        self,
        matched_expressions,
        db_session,
        *,
        db_step=None,
        step_name=None,
        image_id=None,
        channel=None,
    ):
        """Return the configuration for the given step for given expressions."""

        assert db_step or step_name
        if matched_expressions is None:
            assert image_id is not None and channel is not None
            matched_expressions = self._evaluated_expressions[image_id][
                channel
            ]["matched"]
        with TemporaryDirectory() as temp_dir:
            temp_file_path = path.join(temp_dir, "config_file.tmp")
            with open(temp_file_path, mode="w") as config_file:
                config_key = self._write_config_file(
                    matched_expressions,
                    config_file,
                    db_steps=[db_step] if db_step else None,
                    step_names=[step_name] if not db_step else None,
                    db_session=db_session,
                )
                self._logger.debug(
                    "Flushing config file %s", repr(config_file.name)
                )
                config_file.flush()
                os.fsync(
                    config_file.fileno()
                )  # Ensure data is written to disk (cross-platform)
                self._logger.debug(
                    "Wrote config file %s", repr(config_file.name)
                )
                return (
                    getattr(
                        processing_steps, db_step.name if db_step else step_name
                    ).parse_command_line(["-c", config_file.name]),
                    config_key,
                )

    @abstractmethod
    def set_pending(self, db_session):
        """
        Set the unprocessed images and channels split by step and image type.

        Args:
            db_session(Session):    The database session to use.

        Returns:
            {(step.id, image_type.id): (Image, str)}:
                The images and channels of the specified type for which the
                specified step has not applied with the current configuration.
        """

    def add_masters(self, new_masters, step_name=None, image_type_name=None):
        """
        Add new master files to the database.

        Args:
            new_masters(dict or iterable of dicts):    Information about the new
                mbaster(s) to add. Each dictionary should include:

                * type: The type of master being added.

                * filename: The full path to the new master file.

                * preference_order: Expression to select among multiple possible
                  masters. For each frame the expression for each candidate
                  master is evaluateed using the frame header and the master
                  with the smallest resulting value is used.

                * disable(bool): Optional. If set to True the masters are
                  recorded in the database, but not flagged enabled.

            step_name(str):    The name of the step that generated the
                masters.

            image_type_name(str):    The name of the type of images whose
                processing created the masters.
        """

        self._logger.debug(
            "Adding new masters from %s step for %s images:\n%s",
            step_name,
            image_type_name,
            repr(new_masters),
        )
        # False positivie
        # pylint: disable=no-member
        with Session.begin() as db_session:
            # pylint: enable=no-member

            master_id = (
                db_session.scalar(
                    # False positive
                    # pylint: disable=not-callable
                    select(sql.func.max(MasterFile.id))
                    # pylint: enable=not-callable
                )
                or 0
            ) + 1

            type_id_select = select(MasterType.id)
            if step_name is not None:
                assert image_type_name is not None
                type_id_select = (
                    type_id_select.join(ImageType)
                    .join(Step)
                    .where(
                        Step.name == step_name,
                        ImageType.name == image_type_name,
                    )
                )
            if isinstance(new_masters, dict):
                new_masters = (new_masters,)

            if self._current_processing is not None:
                self._current_processing = db_session.merge(
                    self._current_processing, load=False
                )

            for master in new_masters:
                if len(new_masters) > 1 or step_name is None:
                    master_type_id = db_session.scalar(
                        type_id_select.where(MasterType.name == master["type"])
                    )
                else:
                    master_type_id = db_session.scalar(type_id_select)
                db_session.add(
                    MasterFile(
                        id=master_id,
                        type_id=master_type_id,
                        progress_id=(
                            None
                            if self._current_processing is None
                            else self._current_processing.id
                        ),
                        filename=master["filename"],
                        use_smallest=master["preference_order"],
                        enabled=not master.get("disable", False),
                    )
                )
                master_id += 1

    def create_config_file(self, example_header, outf, steps=None):
        """
        Save configuration for processing given header to given output file.

        Args:
            example_header(str or dict-like):    The header to use
                to determine the values of the configuration parameters. Can be
                passed directly as a header instance or FITS or DR filename.

            outf(file or str):    The file to write the configuration to. Can be
                passed as something providing a write method or filename.
                Overwritten if exists.

            steps(list):    If specified, only configuration parameters required
                by these steps will be included.

            steps=None

        Returns:
            None
        """

        matched_expressions = self.get_matched_expressions(
            Evaluator(example_header)
        )
        # False positivie
        # pylint: disable=no-member
        with Session.begin() as db_session:
            # pylint: enable=no-member
            if isinstance(outf, str):
                with open(outf, "w", encoding="utf-8") as opened_outf:
                    self._write_config_file(
                        matched_expressions,
                        opened_outf,
                        step_names=steps,
                        db_session=db_session,
                    )
            else:
                self._write_config_file(
                    matched_expressions,
                    outf,
                    step_names=steps,
                    db_session=db_session,
                )

    @abstractmethod
    def __call__(self, limit_to_steps=None):
        """Perform all the processing for the given steps (all if None)."""

    # pylint: enable=too-many-locals
    # pylint: enable=too-many-branches


# pylint: enable=too-many-instance-attributes
