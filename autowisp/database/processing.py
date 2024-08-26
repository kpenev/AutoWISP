"""Define base class for processing images or lightcurves."""

from abc import ABC, abstractmethod
import logging
from os import path, getpid
from tempfile import NamedTemporaryFile
from socket import getfqdn

from psutil import pid_exists, Process
from sqlalchemy import sql, select
from asteval import asteval

from general_purpose_python_modules.multiprocessing_util import\
    setup_process

from autowisp.database.interface import Session
from autowisp import Evaluator
from autowisp.fits_utilities import get_primary_header
from autowisp.image_calibration.fits_util import\
    add_required_keywords,\
    add_channel_keywords
from autowisp.database.user_interface import\
    get_db_configuration
from autowisp import processing_steps
#False positive due to unusual importing
#pylint: disable=no-name-in-module
from autowisp.database.data_model import\
    Configuration,\
    ImageType,\
    ImageProcessingProgress,\
    LightCurveProcessingProgress,\
    MasterFile,\
    MasterType,\
    Step
#pylint: enable=no-name-in-module


class ProcessingInProgress(Exception):
    """Raised when a particular step is running in a different process/host."""

    def __init__(self, processing):
        self.step = processing.step.name
        if hasattr(processing, 'image_type'):
            self.target = processing.image_type.name + ' images'
        else:
            self.target = processing.sphotref.filename + ' lightcurves'
        self.host = processing.host
        self.process_id = processing.process_id

    def __str__(self):
        return (
            f'Processing of {self.target} by {self.step} step on '
            f'{self.host!r} is still running with process id '
            f'{self.process_id!r}!'
        )


#pylint: disable=too-many-instance-attributes
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

            An additional entry with channel=None is included which contains
            just the common (intersection) set of expressions satisfied for all
            channels.

        _pending(dict):     Information about what images or lightcurves still
            need processing by the various steps. The format is different for
            image vs lightcurve processing managers.
    """

    def get_param_values(self,
                         matched_expressions,
                         parameters=None,
                         db_session=None):
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

            for required_expressions, value in self.configuration[
                    param
            ][
                    "value"
            ].items():
                if required_expressions <= matched_expressions:
                    return value
            raise ValueError(f'No viable configuration found for {param}')

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


    def _write_config_file(self,
                           matched_expressions,
                           outf,
                           db_steps=None,
                           step_names=None):
        """
        Write to given file configuration for given matched expressions.

        Returns:
            Set of tuples of parameters and values as set in the file. Used for
            comparing configurations.
        """

        #TODO: exclude master options
        if db_steps is None:

            #False positivie
            #pylint: disable=no-member
            with Session.begin() as db_session:
            #pylint: enable=no-member

                if step_names is None:
                    steps = db_session.scalars(
                        select(Step).order_by(Step.id)
                    ).all()
                else:
                    steps = [
                        db_session.execute(
                            select(Step).filter_by(name=name)
                        ).scalar_one()
                        for name in step_names
                    ]
                return self._write_config_file(matched_expressions,
                                               outf,
                                               db_steps=steps)

        result = set()

        for step in db_steps:
            outf.write(f'[{step.name}]\n')
            step_config = self.get_param_values(
                matched_expressions,
                [
                    param.name
                    for param in step.parameters
                ]
            )
            for param, value in step_config.items():
                if value is not None:
                    outf.write(f'    {param} = {value!r}\n')
                    print(f'    {param} = {value!r}\n')
                    result.add((param, value))

            outf.write('\n')

        return frozenset(result)


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


    #TODO: see if can be simplified
    #pylint: disable=too-many-locals
    #pylint: disable=too-many-branches
    def evaluate_expressions_image(self,
                                   image,
                                   eval_channel=None,
                                   return_evaluator=False):
        """
        Return evaluator for header expressions for given image.

        Args:
            image(Image):     Instance of database Image for which to evaluate
                the condition expressions. The image header is augmented by
                ``IMAGE_TYPE`` keyword set to the name of the image type of the
                given image.

            eval_channel(str or None):    If given, the evaluator will involve
                all keywords that can be expected of the calibrated header for
                the given channel. Otherwise, an arbitatry channel header
                keywords will be available.

            return_evaluator(bool):    Should an evaluator setup per the image
                header be returned for further use?

        Returns:
            Evaluator or None:
                Evaluator ready to evaluate additional expressions involving
                FITS headers. Only returned if ``return_evaluator`` is True.
        """


        if image.id in self._evaluated_expressions:
            if not return_evaluator:
                return None
            image_expressions = self._evaluated_expressions[
                image.id
            ][
                eval_channel
            ]
            for product in ['dr', 'calibrated']:
                if (
                        product in image_expressions
                        and
                        path.exists(image_expressions[product])
                ):
                    return Evaluator(image_expressions[product])

        self._logger.debug('Evaluating expressions for: %s',
                           repr(image))
        result = None
        evaluate = Evaluator(get_primary_header(image.raw_fname, True))
        evaluate.symtable['IMAGE_TYPE'] = image.image_type.name
        evaluate.symtable['OBS_SESN'] = image.observing_session.label
        self._logger.debug('Matched expressions: %s',
                           repr(self.get_matched_expressions(evaluate)))
        skip_evaluate = image.id in self._evaluated_expressions
        if not skip_evaluate:
            self._evaluated_expressions[image.id] = {}

        all_channel={'matched': None, 'values': None}
        for channel_name, channel_slice in self._get_split_channels(
                image
        ).items():
            add_channel_keywords(evaluate.symtable,
                                 channel_name,
                                 channel_slice)

            calib_config = self.get_config(
                self.get_matched_expressions(evaluate),
                step_name='calibrate',
            )[0]
            self._logger.debug('Raw HDU for channel %s (%s) of %s: %s',
                               channel_name,
                               evaluate.symtable['CLRCHNL'],
                               image.raw_fname,
                               repr(calib_config['raw_hdu']))
            add_required_keywords(evaluate.symtable, calib_config, True)
            if result is None and return_evaluator:
                result = asteval.Interpreter()
                if return_evaluator and eval_channel is None:
                    result.symtable.update(evaluate.symtable)

            if return_evaluator and eval_channel == channel_name:
                result.symtable.update(evaluate.symtable)
            if skip_evaluate:
                continue
            evaluated_expressions = {
                'values': {expr_id: evaluate(expression)
                           for expr_id, expression in
                           self.condition_expressions.items()},
                'calibrated': calib_config['calibrated_fname'].format_map(
                    evaluate.symtable
                )
            }
            evaluated_expressions['matched'] = set(
                expr_id
                for expr_id, value in evaluated_expressions['values'].items()
                if value
            )

            if all_channel['matched'] is None:
                all_channel['matched'] = evaluated_expressions['matched']
                all_channel['values'] = dict(evaluated_expressions['values'])
            else:
                all_channel['matched'] = (all_channel['matched']
                                          &
                                          evaluated_expressions['matched'])
                #False positive
                #pylint: disable=unsubscriptable-object
                #pylint: disable=unsupported-delete-operation
                for expr_id in list(all_channel['values'].keys()):
                    if (
                        all_channel['values'][expr_id]
                        !=
                        evaluated_expressions['values'][expr_id]
                    ):
                        del all_channel['values'][expr_id]
                #pylint: enable=unsupported-delete-operation
                #pylint: enable=unsubscriptable-object

            for required_expressions, value in self.configuration[
                    'data-reduction-fname'
            ][
                    'value'
            ].items():
                if required_expressions <= evaluated_expressions['matched']:
                    evaluated_expressions['dr'] = value.format_map(
                        evaluate.symtable
                    )
                    break
            assert 'dr' in evaluated_expressions

            self._evaluated_expressions[image.id][channel_name] = (
                evaluated_expressions
            )

        if skip_evaluate:
            assert return_evaluator
            return result
        self._evaluated_expressions[image.id][None] = {
            'matched': all_channel['matched'],
            'values': all_channel['values'],
        }
        self._logger.debug('Evaluated expressions for image %s: %s',
                           image,
                           repr(self._evaluated_expressions[image.id]))

        return result if return_evaluator else None


    def _process_batch(self,
                       batch,
                       *,
                       start_status,
                       config,
                       step_name,
                       image_type_name):
        """Run the current step for a batch of images given configuration."""

        step_module = getattr(processing_steps, step_name)

        new_masters = getattr(step_module, step_name)(
            batch,
            start_status,
            config,
            self._start_processing,
            self._end_processing
        )
        if new_masters:
            self.add_masters(new_masters, step_name, image_type_name)


    def _check_running_processing(self,
                                  running_processing,
                                  this_host,
                                  db_session):
        """Check if any unfinished processing progresses are still running."""

        for processing in running_processing:
            if (
                processing is not None
                and
                not processing.finished
            ):
                if (
                        processing.host != this_host
                        or
                        (
                            pid_exists(processing.process_id)
                            and
                            path.basename(
                                Process(
                                    processing.process_id
                                ).cmdline()[1] == 'processing.py'
                            )
                        )
                ):
                    raise ProcessingInProgress(processing)
                self._logger.warning(
                    'Processing progress %s appears to have crashed.',
                    processing
                )
                #False positive
                #pylint: disable=not-callable
                processing.finished = sql.func.now()
                #pylint: enable=not-callable
                db_session.flush()


    def _create_current_processing(self, step, target, db_session):
        """Add a new ProcessingProgress at start of given step."""

        this_host  = getfqdn()
        process_id = getpid()

        self.current_step = step

        progress_class = (
            ImageProcessingProgress if target[0] == 'image_type'
            else LightCurveProcessingProgress
        )
        self._check_running_processing(
            db_session.scalars(
                select(
                    progress_class
                ).where(
                    (
                        progress_class.step_id
                        ==
                        self.current_step.id
                    ),
                    (
                        getattr(progress_class, target[0] + '_id')
                        ==
                        target[1]
                    ),
                    (
                        progress_class.configuration_version
                        ==
                        self.step_version[step.name]
                    )
                )
            ).all(),
            this_host,
            db_session
        )

        self._current_processing = progress_class(
            step_id=step.id,
            **{
                target[0] + '_id': target[1]
            },
            configuration_version=self.step_version[step.name],
            host=this_host,
            process_id=process_id,
            #False positive
            #pylint: disable=not-callable
            started=sql.func.now(),
            #pylint: enable=not-callable
            finished=None,
        )
        db_session.add(self._current_processing)
        db_session.flush()


    @abstractmethod
    def _start_processing(self, input_fname):
        """
        Mark in the database that processing the given file has begun.

        Args:
            input_fname:    The filename of the input (DR or FITS) that is about
                to begin processing.

        Returns:
            None
        """


    @abstractmethod
    def _end_processing(self, input_fname, status=1, final=True):
        """
        Record that the current step has finished processing the given file.

        Args:
            input_fname:    The filename of the input (DR or FITS) that was
                processed.

        Returns:
            None
        """


    @abstractmethod
    def _cleanup_interrupted(self, db_session):
        """Cleanup previously interrupted processing for the current step."""


    def _get_split_channels(self, image):
        """Return the ``split_channels`` option for the given image."""

        return {
            channel.name: (slice(channel.y_offset, None, channel.y_step),
                           slice(channel.x_offset, None, channel.x_step))
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
        self._logger = logging.getLogger(__name__)
        self.current_step = None
        self._current_processing = None
        self.configuration = {}
        self.condition_expressions = {}
        self._evaluated_expressions = {}
        self._processed_ids = {}
        self.pending = {}
        self._some_failed = False
        #False positivie
        #pylint: disable=no-member
        with Session.begin() as db_session:
        #pylint: enable=no-member

            if version is None:
                version = db_session.execute(
                    #False positivie
                    #pylint: disable=not-callable
                    #pylint: disable=no-member
                    select(sql.func.max(Configuration.version))
                    #pylint: enable=not-callable
                    #pylint: enable=no-member
                ).scalar_one()

            db_configuration = get_db_configuration(version, db_session)
            for config_entry in db_configuration:
                if config_entry.parameter.name not in self.configuration:
                    self.configuration[config_entry.parameter.name] = {
                        'version': config_entry.version,
                        'value': {}
                    }
                self.configuration[config_entry.parameter.name]['value'][
                    frozenset(
                        cond.expression_id
                        for cond in config_entry.conditions
                    )
                ] = config_entry.value

                for cond in config_entry.conditions:
                    if cond.expression_id not in self.condition_expressions:
                        self.condition_expressions[cond.expression_id] = (
                            cond.expression.expression
                        )

            self._processing_config = self.get_config(
                self.get_matched_expressions(Evaluator()),
                step_name='add_images_to_db',
            )[0]
            del self._processing_config['processing_step']
            del self._processing_config['image_type']

            if not dummy:
                setup_process(task='main',
                              parent_pid='',
                              processing_step='init_processing',
                              image_type='none',
                              **self._processing_config)

            for master_type in db_session.scalars(select(MasterType)).all():
                for expression in (
                    master_type.match_expressions
                    +
                    master_type.split_expressions
                ):
                    self.condition_expressions[expression.id] = (
                        expression.expression
                    )
            self._logger.debug('Condition expressions to evaluate: %s',
                               repr(self.condition_expressions))

            self.step_version = {
                step.name: max(
                    self.configuration[param.name]['version']
                    for param in step.parameters
                )
                for step in db_session.scalars(select(Step)).all()
            }

            if not dummy:
                self._cleanup_interrupted(db_session)
                self.set_pending(db_session)


    def get_config(self,
                   matched_expressions,
                   *,
                   db_step=None,
                   step_name=None,
                   image_id=None,
                   channel=None):
        """Return the configuration for the given step for given expressions."""

        assert db_step or step_name
        if matched_expressions is None:
            assert image_id is not None and channel is not None
            matched_expressions = self._evaluated_expressions[
                image_id
            ][
                channel
            ][
                'matched'
            ]
        with NamedTemporaryFile(mode='w') as config_file:
            config_key = self._write_config_file(
                matched_expressions,
                config_file,
                db_steps=[db_step] if db_step else None,
                step_names=[step_name] if not db_step else None
            )
            config_file.flush()
            self._logger.debug('Wrote config file %s', repr(config_file.name))
            return getattr(
                processing_steps,
                db_step.name if db_step else step_name
            ).parse_command_line(
                ['-c', config_file.name]
            ), config_key


    def get_candidate_masters(self, image, channel, master_type, db_session):
        """Return list of masters of given type that are applicable to image."""

        image_eval = self.evaluate_expressions_image(image,
                                                     channel,
                                                     True)
        print(f'Image keys: {image_eval.symtable.keys()!r}')
        candidate_masters = db_session.scalars(
            select(
                MasterFile
            ).filter_by(
                type_id=master_type.master_type_id,
                enabled=True,
            )
        ).all()
        result = []
        for master in candidate_masters:
            master_eval = Evaluator(master.filename)
            print(f'Master keys: {master_eval.symtable.keys()!r}')
            all_match = True
            for expr in master.master_type.match_expressions:
                if master_eval(expr.expression) != image_eval(expr.expression):
                    all_match = False
                    break
            if all_match:
                result.append(master)
        return result


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

        #False positivie
        #pylint: disable=no-member
        with Session.begin() as db_session:
        #pylint: enable=no-member

            master_id = (
                db_session.scalar(
                    #False positive
                    #pylint: disable=not-callable
                    select(sql.func.max(MasterFile.id))
                    #pylint: enable=not-callable
                ) or 0
            ) + 1

            type_id_select = select(
                MasterType.id
            )
            if step_name is not None:
                assert image_type_name is not None
                type_id_select = type_id_select.join(
                    ImageType
                ).join(
                    Step
                ).where(
                    Step.name == step_name,
                    ImageType.name == image_type_name
                )
            if isinstance(new_masters, dict):
                new_masters=(new_masters,)

            if self._current_processing is not None:
                self._current_processing = db_session.merge(
                    self._current_processing,
                    load=False
                )

            for master in new_masters:
                if len(new_masters) > 1 or step_name is None:
                    master_type_id = db_session.scalar(
                        type_id_select.where(
                            MasterType.name == master['type']
                        )
                    )
                else:
                    master_type_id = db_session.scalar(type_id_select)
                db_session.add(
                    MasterFile(
                        id=master_id,
                        type_id=master_type_id,
                        progress_id=(None if self._current_processing is None
                                     else self._current_processing.id),
                        filename=master['filename'],
                        use_smallest=master['preference_order'],
                        enabled=not master.get('disable', False)
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
        if isinstance(outf, str):
            with open(outf, 'w', encoding='utf-8') as opened_outf:
                self._write_config_file(matched_expressions,
                                        opened_outf,
                                        step_names=steps)
        else:
            self._write_config_file(matched_expressions,
                                    outf,
                                    step_names=steps)

    @abstractmethod
    def __call__(self, limit_to_steps=None):
        """Perform all the processing for the given steps (all if None)."""
    #pylint: enable=too-many-locals
    #pylint: enable=too-many-branches
#pylint: enable=too-many-instance-attributes
