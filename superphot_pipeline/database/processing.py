#!/usr/bin/env python3
#pylint: disable=too-many-lines

"""Handle data processing DB interactions."""

from tempfile import NamedTemporaryFile
import logging
from os import path, getpid
from socket import getfqdn

from sqlalchemy import sql, select, update, and_, or_
from asteval import asteval
import numpy
from configargparse import ArgumentParser, DefaultsFormatter
from psutil import pid_exists

from general_purpose_python_modules.multiprocessing_util import\
    setup_process,\
    get_log_outerr_filenames

from superphot_pipeline import Evaluator
from superphot_pipeline.database.interface import Session
from superphot_pipeline.fits_utilities import get_primary_header
from superphot_pipeline.image_calibration.fits_util import\
    add_required_keywords,\
    add_channel_keywords
from superphot_pipeline.file_utilities import find_fits_fnames
from superphot_pipeline import processing_steps
from superphot_pipeline.database.user_interface import\
    get_db_configuration,\
    get_processing_sequence
#False positive due to unusual importing
#pylint: disable=no-name-in-module
from superphot_pipeline.database.data_model import\
    StepDependencies,\
    ImageProcessingProgress,\
    ProcessedImages,\
    Configuration,\
    Step,\
    ImageType,\
    Image,\
    ObservingSession,\
    MasterFile,\
    MasterType,\
    RequiredMasterTypes,\
    Condition,\
    ConditionExpression
from superphot_pipeline.database.data_model.provenance import\
    Camera,\
    CameraChannel,\
    CameraType
#pylint: enable=no-name-in-module

class NoMasterError(ValueError):
    """Raised when no suitable master can be found for a batch of frames."""

class ProcessingInProgress(Exception):
    """Raised when a particular step is running in a different process/host."""

    def __init__(self, step, image_type, host, process_id):
        self.step = step
        self.image_type = image_type
        self.host = host
        self.process_id = process_id

#Intended to be used as simple callable
#pylint: disable=too-few-public-methods
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
            'Getting master expression values for expression ids %s, '
            'image %d, channel %s',
            repr(self._master_expression_ids),
            image_id,
            channel
        )
        return tuple(
            self._evaluated_expressions[
                image_id
            ][
                channel
            ][
                'values'
            ][
                expression_id
            ]
            for expression_id in self._master_expression_ids
        )


    def __init__(self,
                 evaluated_expressions,
                 ref_image_id,
                 ref_channel,
                 master_expression_ids):
        """
        Set up comparison to the given evaluated expressions.

        """

        self._logger = logging.getLogger(__name__)
        self._evaluated_expressions = evaluated_expressions
        self._master_expression_ids = master_expression_ids
        reference_evaluated = evaluated_expressions[ref_image_id][ref_channel]
        self._ref_matched = reference_evaluated['matched']
        self.ref_master_values = self._get_master_values(ref_image_id,
                                                          ref_channel)
        self._logger.debug(
            'Finding images matching expressions %s and values %s',
            repr(self._ref_matched),
            repr(self.ref_master_values)
        )

    def __call__(self, image_id, channel):
        """True iff the expressions for the given image/channel match."""

        image_evaluated = self._evaluated_expressions[image_id][channel]
        image_master_values = self._get_master_values(image_id, channel)

        self._logger.debug(
            'Comparing %s to %s and %s to %s',
            repr(image_evaluated['matched']),
            repr(self._ref_matched),
            repr(image_master_values),
            repr(self.ref_master_values)
        )
        return (
            image_evaluated['matched'] == self._ref_matched
            and
            image_master_values == self.ref_master_values
        )
#pylint: enable=too-few-public-methods


#pylint: disable=too-many-instance-attributes
class ProcessingManager:
    """
    Read configuration and record processing progress in the database.

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

        _progress(dict):    Indexed by step name, ImageProcessingProgress
            instances of all steps for which start_step was invoked.

        _current_processing(ImageProcessingProgress):    The currently active
            step (the processing progress initiated the last time `start_step()`
            was called).

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

        _processed_ids(dict):    The keys are the filenames of the required
            inputs (DR or FITS) for the current step and the values are
            dictionaries with keys ``'image_id'`` and ``'channel'`` identifying
            what was processed.

        _pending(dict):    Indexed by step ID, and image type ID list of
            (Image, channel name) tuples listing all the images of the given
            type that have not been processed by the currently selected version
            of the step in the key.
    """


    def _get_param_values(self,
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

        def get_param_value(param):
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

        return {param: get_param_value(param) for param in parameters}


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

        added_params = {'raw-hdu'}
        for step in db_steps:
            outf.write(f'[{step.name}]\n')
            step_config = self._get_param_values(
                matched_expressions,
                [
                    param.name
                    for param in step.parameters
                    if param.name not in added_params
                ]
            )
            for param, value in step_config.items():
                if value is not None:
                    outf.write(f'    {param} = {value!r}\n')
                    print(f'    {param} = {value!r}\n')
                    result.add((param, value))

            outf.write('\n')

        return frozenset(result)


    def _get_config(self, matched_expressions, db_step=None, step_name=None):
        """Return the configuration for the given step for given expressions."""

        assert db_step or step_name
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


    def _get_candidate_masters(self, image_eval, master_type, db_session):
        """Return list of masters of given type that are applicable to image."""

        print(f'Image keys: {image_eval.symtable.keys()!r}')
        candidate_masters = db_session.scalars(
            select(
                MasterFile
            ).filter_by(
                type_id=master_type.master_type_id,
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


    def _get_best_master(self, candidate_masters, image, channel):
        """Find the best master from given list for given image/channel."""

        if channel is None:
            return tuple(
                (
                    channel,
                    self._get_best_master(candidate_masters[channel],
                                          image,
                                          channel)
                )
                for channel in sorted(
                    filter(None, self._evaluated_expressions[image.id].keys())
                )
            )

        self._logger.debug('Selecting best master for %s, channel %s from %s',
                           repr(image),
                           repr(channel),
                           repr(candidate_masters))
        image_eval = self._evaluate_expressions_image(image,
                                                      channel,
                                                      True)
        best_master_value = numpy.inf
        best_master_fname = None
        for master in candidate_masters:
            assert master.use_smallest is not None
            master_value = image_eval(master.use_smallest)
            if master_value < best_master_value:
                best_master_value = master_value
                best_master_fname = master.filename
        assert best_master_fname
        return best_master_fname


    def _split_by_master(self, batch, required_master_type, db_session):
        """Split the given list of images by the best master of given type."""

        result = {}

        candidate_masters = {}
        if self.current_step.name == 'calibrate':
            channel_list = self._evaluated_expressions[batch[0][0].id].keys()
        else:
            assert batch[0][1] is not None
            channel_list = [batch[0][1]]

        for channel in channel_list:
            candidate_masters[channel] = self._get_candidate_masters(
                self._evaluate_expressions_image(batch[0][0],
                                                 channel,
                                                 True),
                required_master_type,
                db_session
            )
            if not candidate_masters[channel]:
                raise NoMasterError(
                    f'No master {required_master_type.master_type.name} '
                    f'found for image {batch[0][0].raw_fname} channel '
                    f'{channel}.'
                )
        if len(channel_list) == 1:
            candidate_masters = candidate_masters[channel_list[0]]

        if len(channel_list) == 1 and len(candidate_masters) == 1:
            result[candidate_masters[channel_list[0]][0].filename] = batch
        else:
            for image, channel in batch:
                best_master = self._get_best_master(candidate_masters,
                                                    image,
                                                    channel)
                if best_master in result:
                    result[best_master].append((image, channel))
                else:
                    result[best_master] = [(image, channel)]
        return result


    def _set_calibration_config(self, config, batch, db_session):
        """Retrun the specially formatted argument for the calibration step."""

        config['split_channels'] = self._get_split_channels(batch[0][0])
        config['extra_header'] = {
            'OBS-SESN': batch[0][0].observing_session.label
        }
        config['raw_hdu'] = {
            channel: self._get_param_values(
                first_image_expressions[channel]['matched'],
                ['raw-hdu'],
                db_session
            )['raw-hdu']
            for channel in filter(None, first_image_expressions.keys())
        }
        hdu_set = set(config['raw_hdu'].values())
        if len(hdu_set) == 1:
            config['raw_hdu'] = hdu_set.pop()
        else:
            assert len(hdu_set) == len(config['raw_hdu'])
            for channel, hdu in config['raw_hdu'].items():
                if hdu is not None:
                    config['raw_hdu'] = int(hdu)
        self._logger.debug('Calibration step configuration:\n%s',
                           '\n\t'.join(
                               (f'{k}: {v!r}' for k, v in config.items())
                           ))
        return {
            (
                'split_channels',
                ''.join(
                    repr(c)
                    for c in batch[0][0].observing_session.camera.channels
                )
            ),
            (
                'observing_session',
                config['extra_header']['OBS-SESN']
            ),
            tuple(
                sorted(config['raw_hdu'].items())
            )
        }


    #Could not find good way to simplify
    #pylint: disable=too-many-locals
    def _get_batch_config(self,
                          batch,
                          master_expression_values,
                          step,
                          db_session):
        """
        Split given batch of images by configuration for given step.

        The batch must already be split by all relevant condition expressions.
        Only splits batches by the best master for each image.

        Args:
            batch([Image, channel]):    List of database image instances and for
                channels which to find the configuration(s). The channel should
                be ``None`` for the ``calibrate`` step

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

        self._logger.debug('Finding configuration for batch: %s',
                           repr(batch))
        first_image_expressions = self._evaluated_expressions[batch[0][0].id]
        config, config_key = self._get_config(
            first_image_expressions[batch[0][1]]['matched'],
            db_step=step
        )
        config_key |= {master_expression_values}
        if step.name == 'calibrate':
            config_key |= self._set_calibration_config(config,
                                                       batch,
                                                       db_session)
        config['processing_step'] = step.name
        config['image_type'] = batch[0][0].image_type.name

        result = {
            config_key: (config, batch)
        }
        for required_master_type in db_session.scalars(
            select(RequiredMasterTypes).filter_by(
                step_id=step.id,
                image_type_id=batch[0][0].image_type_id
            )
        ).all():
            for config_key, (config, sub_batch) in list(result.items()):
                del result[config_key]
                try:
                    splits = self._split_by_master(sub_batch,
                                                   required_master_type,
                                                   db_session)
                except NoMasterError as no_master:
                    self._logger.error(str(no_master))
                    continue

                for best_master, sub_batch in splits.items():
                    new_config = dict(config)
                    new_config[
                        required_master_type.config_name.replace('-', '_')
                    ] = dict(best_master)
                    key_extra = {
                        (required_master_type.config_name, best_master)
                    }
                    result[config_key | key_extra] = (new_config, sub_batch)

        return result
    #pylint: enable=too-many-locals


    def _get_split_channels(self, image):
        """Return the ``split_channels`` option for the given image."""

        return {
            channel.name: (slice(channel.y_offset, None, channel.y_step),
                           slice(channel.x_offset, None, channel.x_step))
            for channel in image.observing_session.camera.channels
        }


    def _get_matched_expressions(self, evaluate):
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


    def _get_step_input(self, image, channel_name, step_input_type):
        """Return the name of the file required by the current step."""

        if step_input_type == 'raw':
            return image.raw_fname

        if step_input_type.startswith('calibrated'):
            return self._evaluated_expressions[
                image.id
            ][
                channel_name
            ][
                'calibrated'
            ]

        if step_input_type == 'dr':
            return self._evaluated_expressions[
                image.id
            ][
                channel_name
            ][
                'dr'
            ]

        raise ValueError(f'Invalid step input type {step_input_type}')


    #TODO: see if can be simplified
    #pylint: disable=too-many-locals
    #pylint: disable=too-many-branches
    def _evaluate_expressions_image(self,
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
                           repr(self._get_matched_expressions(evaluate)))
        self._evaluated_expressions[image.id] = {}
        all_channel={'matched': None, 'values': None}
        for channel_name, channel_slice in self._get_split_channels(
                image
        ).items():
            add_channel_keywords(evaluate.symtable,
                                 channel_name,
                                 channel_slice)

            calib_config = self._get_config(
                self._get_matched_expressions(evaluate),
                step_name='calibrate',
            )[0]
            self._logger.debug('Calibration config: %s', repr(calib_config))
            add_required_keywords(evaluate.symtable, calib_config, True)
            if result is None and return_evaluator:
                result = asteval.Interpreter()
                if return_evaluator and eval_channel is None:
                    result.symtable.update(evaluate.symtable)

            if return_evaluator and eval_channel == channel_name:
                result.symtable.update(evaluate.symtable)
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

        self._evaluated_expressions[image.id][None] = {
            'matched': all_channel['matched'],
            'values': all_channel['values'],
        }
        self._logger.debug('Evaluated expressions for image %s: %s',
                           image,
                           repr(self._evaluated_expressions[image.id]))

        return result if return_evaluator else None
    #pylint: enable=too-many-locals
    #pylint: enable=too-many-branches


    def _fill_pending(self, db_session):
        """
        Return the images and channels of a type not yet processed by a step.

        Args:
            step_id(Step):    The step to determine pending images for.

            image_type(ImageType):    The type of images to check for pending
                processing by the specified step.

            db_session(Session):    The database session to use.

        Returns:
            (Image, [str]):
                The images and channels of the specified type for which the
                specified step has not applied with the current configuration.
        """

        select_image_channel = select(
            Image,
            CameraChannel.name
        ).join(
            ObservingSession,
        ).join(
            Camera
        ).join(
            CameraType
        ).join(
            CameraChannel
        )

        for step, image_type in get_processing_sequence(db_session):
            processed_subquery = select(
                ProcessedImages.image_id,
                ProcessedImages.channel
            ).join(
                ImageProcessingProgress
            ).where(
                ImageProcessingProgress.step_id == step.id
            ).where(
                ImageProcessingProgress.configuration_version
                ==
                self.step_version[step.name]
            ).where(
                ProcessedImages.final
            ).subquery()

            self._pending[(step.id, image_type.id)] = db_session.execute(
                select_image_channel.outerjoin(
                    processed_subquery,
                    #False positive
                    #pylint: disable=no-member
                    and_(Image.id == processed_subquery.c.image_id,
                         CameraChannel.name == processed_subquery.c.channel),
                    #pylint: enable=no-member
                ).where(
                    #This is how NULL comparison is done in SQLAlchemy
                    #pylint: disable=singleton-comparison
                    processed_subquery.c.image_id == None
                    #pylint: enable=singleton-comparison
                ).where(
                    #False positive
                    #pylint: disable=no-member
                    Image.image_type_id == image_type.id
                    #pylint: enable=no-member
                )
            ).all()
            self._logger.debug(
                'Identified %d %s images for which %s is pending',
                len(self._pending[(step.id, image_type.id)]),
                image_type.name,
                step.name
            )
        self._logger.debug('Pending: %s', repr(self._pending))


    def _clean_pending_per_dependencies(self,
                                        db_session,
                                        from_step_id=None,
                                        from_image_type_id=None):
        """Remove pending images from steps if they failed a required step."""

        dropped = {}
        for (step_id, image_type_id), pending in self._pending.items():
            if (
                from_image_type_id is not None
                and
                image_type_id != from_image_type_id
            ):
                continue
            for prereq_step_id in db_session.scalars(
                select(
                    StepDependencies.blocking_step_id
                ).where(
                    StepDependencies.blocked_step_id == step_id,
                    StepDependencies.blocked_image_type_id == image_type_id,
                    StepDependencies.blocking_image_type_id == image_type_id
                )
            ):
                if from_step_id is not None and prereq_step_id != from_step_id:
                    continue
                prereq_statuses = [
                    db_session.execute(
                        select(
                            ProcessedImages.status
                        ).outerjoin(
                            ImageProcessingProgress
                        ).where(
                            (
                                ProcessedImages.image_id
                                ==
                                db_session.merge(image).id
                            ),
                            ProcessedImages.channel == channel,
                            ImageProcessingProgress.step_id == prereq_step_id,
                            (
                                ImageProcessingProgress.image_type_id
                                ==
                                image_type_id
                            )
                        )
                    ).scalar_one_or_none()
                    for image, channel in pending
                ]
                for i in range(len(pending) - 1, -1, -1):
                    if prereq_statuses[i] and prereq_statuses[i] < 0:
                        self._logger.info(
                            'Image %s, channel %s failed %s. Excluding from %s',
                            *pending[i],
                            db_session.scalar(
                                select(
                                    Step.name
                                ).filter_by(
                                    id=prereq_step_id
                                )
                            ),
                            db_session.scalar(
                                select(
                                    Step.name
                                ).filter_by(
                                    id=step_id
                                )
                            )
                        )

                        if (step_id, image_type_id) not in dropped:
                            dropped[(step_id, image_type_id)] = []
                        dropped[(step_id, image_type_id)].append(pending.pop(i))
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
                ).where(
                    StepDependencies.blocked_step_id == step.id
                ).where(
                    StepDependencies.blocked_image_type_id == image_type.id
                )
        ).all():
            if self._pending[requirement]:
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
        input_type = getattr(processing_steps,
                             self.current_step.name).input_type

        for entry in need_cleanup:
            assert entry[2] == self.current_step

        pending = [(image, None if input_type == 'raw' else processed.channel)
                   for image, processed, _ in need_cleanup]

        for image, _, __ in need_cleanup:
            if image.id not in self._evaluated_expressions:
                self._evaluate_expressions_image(image)

        cleanup_batches = self._get_batches(pending, input_type, db_session)
        return [
            (config, [(fname, status) for fname in batch])
            for (_, status), (config, batch) in cleanup_batches.items()
        ]


    def _cleanup_interrupted(self, db_session):
        """Cleanup previously interrupted processing for the current step."""

        need_cleanup = db_session.execute(
            select(
                Image,
                ProcessedImages,
                Step
            ).join(
                ProcessedImages
            ).join(
                ImageProcessingProgress
            ).join(
                Step
            ).where(
                ~ProcessedImages.final
            ).order_by(
                Step.name
            )
        ).all()

        if not need_cleanup:
            return

        step_module = getattr(processing_steps, need_cleanup[0][2].name)

        for config, interrupted in self._get_interrupted(need_cleanup,
                                                         db_session):
            self._logger.warning(
                'Cleaning up interrupted %s processing of %d images:\n'
                '%s\n'
                'config: %s',
                need_cleanup[0][2],
                len(interrupted),
                repr(interrupted),
                repr(config)
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

            step_input_fname = self._get_step_input(image,
                                                    channel_name,
                                                    step_input_type)

            if step_input_fname not in self._processed_ids:
                self._processed_ids[step_input_fname] = []
            self._processed_ids[step_input_fname].append(
                {'image_id': image.id, 'channel': channel_name}
            )


    def _create_current_processing(self, step, image_type, db_session):
        """Add a new ImageProcessingProgress at start of given step."""

        this_host  = getfqdn()
        process_id = getpid()

        self.current_step = step
        self._current_processing = db_session.execute(
            select(
                ImageProcessingProgress
            ).where(
                (
                    ImageProcessingProgress.step_id
                    ==
                    self.current_step.id
                ),
                (
                    ImageProcessingProgress.image_type_id
                    ==
                    image_type.id
                ),
                (
                    ImageProcessingProgress.configuration_version
                    ==
                    self.step_version[step.name]
                ),
                (
                    #This is how to check for NULL in sqlalchemy
                    #pylint: disable=singleton-comparison
                    ImageProcessingProgress.finished
                    ==
                    None
                    #pylint: enable=singleton-comparison
                )
            )
        ).scalar_one_or_none()

        if self._current_processing is not None:
            if (
                    self._current_processing.host != this_host
                    or
                    pid_exists(self._current_processing.process_id)
            ):
                raise ProcessingInProgress(step.name,
                                           image_type,
                                           self._current_processing.host,
                                           self._current_processing.process_id)
            self._logger.warning(
                'Processing progress %s appears to have crashed.',
                self._current_processing
            )
            #False positive
            #pylint: disable=not-callable
            self._current_processing.finished = sql.func.now()
            #pylint: enable=not-callable
            db_session.flush()

        self._current_processing = ImageProcessingProgress(
            step_id=step.id,
            image_type_id=image_type.id,
            configuration_version=self.step_version[step.name],
            host=this_host,
            process_id=process_id,
            #False positive
            #pylint: disable=not-callable
            started=sql.func.now(),
            #pylint: enable=not-callable
            finished=None
        )
        db_session.add(self._current_processing)
        db_session.flush()


    def _start_step(self, step, image_type, db_session):
        """
        Record the start of a processing step and return the images to process.

        Args:
            step_name(str):    The name of the step to start.

        Returns:
            [(Image, str)]:
                The list of images and channels to process.

            str:
                The type of input expected by the current step.
        """

        step = db_session.merge(step, load=False)
        image_type = db_session.merge(image_type, load=False)
        self._create_current_processing(step, image_type, db_session)

        pending_images = [
            (db_session.merge(image, load=False), channel)
            for image, channel in self._pending[(step.id, image_type.id)]
        ]
        for image, channel in self._failed_dependencies.get(
            (step.id, image_type.id),
            []
        ):
            image = db_session.merge(image, load=False)
            self._logger.info('Prerequisite failed for %s of %s',
                              step.name,
                              image)
            db_session.add(
                ProcessedImages(
                    image_id=image.id,
                    channel=channel,
                    progress_id=self._current_processing.id,
                    status=-1,
                    final=True
                )
            )
            self._some_failed = True

        self._processed_ids = {}
        step_input_type = getattr(processing_steps, step.name).input_type

        if step_input_type == 'raw':
            added = set()
            new_pending = []
            for image, _ in pending_images:
                if image.id not in added:
                    added.add(image.id)
                    new_pending.append((image, None))
            pending_images = new_pending

        for image, channel_name in pending_images:
            if image.id not in self._evaluated_expressions:
                self._evaluate_expressions_image(image)
            self._init_processed_ids(image, [channel_name], step_input_type)

        self._logger.info('Starting %s step for %d %s images',
                          self.current_step.name,
                          len(pending_images),
                          image_type.name)

        return pending_images, step_input_type


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
        if not new_masters:
            return
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
            ).join(
                ImageType
            ).join(
                Step
            ).where(
                Step.name == step_name,
                ImageType.name == image_type_name
            )
            if isinstance(new_masters, dict):
                new_masters=(new_masters,)

            for master in new_masters:
                if len(new_masters) > 1:
                    type_id_select = type_id_select.where(
                        MasterType.name == master['type']
                    )
                master_type_id = db_session.scalar(type_id_select)
                db_session.add(
                    MasterFile(
                        id=master_id,
                        type_id=master_type_id,
                        progress_id=db_session.merge(
                            self._current_processing,
                            load=False
                        ).id,
                        filename=master['filename'],
                        use_smallest=master['preference_order']
                    )
                )


    def _start_processing(self, input_fname):
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
        self._logger.debug('Starting processing IDs: %s',
                           repr(self._processed_ids[input_fname]))
        #False positivie
        #pylint: disable=no-member
        with Session.begin() as db_session:
        #pylint: enable=no-member
            for starting_id in self._processed_ids[input_fname]:
                db_session.add(
                    ProcessedImages(
                        **starting_id,
                        progress_id=db_session.merge(
                            self._current_processing,
                            load=False
                        ).id,
                        status=0,
                        final=False
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
        self._logger.debug('Finished processing %s',
                           repr(self._processed_ids[input_fname]))
        #False positivie
        #pylint: disable=no-member
        with Session.begin() as db_session:
        #pylint: enable=no-member
            db_session.execute(
                update(ProcessedImages),
                [
                    {
                        'image_id': finished_id['image_id'],
                        'channel': finished_id['channel'],
                        'progress_id': db_session.merge(
                            self._current_processing,
                            load=False
                        ).id,
                        'status': status,
                        'final': final
                    }
                    for finished_id in self._processed_ids[input_fname]
                ]
            )


    def _group_pending_by_conditions(self,
                                     pending_images,
                                     db_session,
                                     match_observing_session=False):
        """Group pendig_images grouped by condition expression values."""

        image_type_id = pending_images[0][0].image_type_id
        result = []
        master_expression_ids = (
            db_session.scalars(
                select(
                    ConditionExpression.id
                ).select_from(
                    RequiredMasterTypes
                ).join(
                    MasterType
                ).join(
                    Condition,
                    MasterType.condition_id == Condition.id
                ).join(
                    ConditionExpression
                ).where(
                    RequiredMasterTypes.step_id
                    ==
                    self.current_step.id
                ).where(
                    RequiredMasterTypes.image_type_id
                    ==
                    image_type_id
                ).group_by(
                    ConditionExpression.id
                )
            ).all()
            +
            db_session.scalars(
                select(
                    ConditionExpression.id
                ).select_from(
                    MasterType
                ).join(
                    Condition,
                    or_(
                        MasterType.condition_id == Condition.id,
                        (
                            MasterType.maker_image_split_condition_id
                            ==
                            Condition.id
                        )
                    )
                ).join(
                    ConditionExpression
                ).where(
                    MasterType.maker_step_id == self.current_step.id
                ).where(
                    MasterType.maker_image_type_id == image_type_id
                )
            ).all()
        )

        while pending_images:
            #pending_images = [
            #    (db_session.merge(image, load=False), channel)
            #    for image, channel in pending_images
            #]
            #self.current_step = db_session.merge(self.current_step,
            #                                     load=False)
            #self._current_processing = db_session.merge(
            #    self._current_processing,
            #    load=False
            #)

            self._logger.debug(
                'Finding images matching the same expressions as image id %d, '
                'channel %s',
                pending_images[-1][0].id,
                pending_images[-1][1]
            )
            batch = []
            match_expressions = ExpressionMatcher(
                self._evaluated_expressions,
                pending_images[-1][0].id,
                pending_images[-1][1],
                master_expression_ids
            )
            observing_session_id = pending_images[-1][0].observing_session_id

            for i in range(len(pending_images) - 1, -1, -1):
                if (
                    (
                        not match_observing_session
                        or
                        pending_images[i][0].observing_session_id
                        ==
                        observing_session_id
                    )
                    and
                    match_expressions(
                        pending_images[i][0].id,
                        pending_images[i][1]
                    )
                ):
                    batch.append(pending_images.pop(i))
                    self._logger.debug(
                        'Added image to batch, now:\n\t%s',
                        '\n\t'.join(
                            f'{image.raw_fname}: {channel}'
                            for image, channel in batch
                        )
                    )
                else:
                    self._logger.debug('Not a match')
            result.append((batch, match_expressions.ref_master_values))
        return result


    #No good way to simplify
    #pylint: disable=too-many-locals
    def _get_batches(self, pending_images, step_input_type, db_session):
        """Return the batches of images to process with identical config."""

        result = {}
        check_image_type_id = pending_images[0][0].image_type_id
        for (
            by_condition,
            master_expression_values
        ) in self._group_pending_by_conditions(
            pending_images,
            db_session,
            self.current_step.name == 'calibrate'
        ):
            for config_key, (config, batch) in self._get_batch_config(
                by_condition,
                master_expression_values,
                self.current_step,
                db_session
            ).items():
                batch_status = None
                for image, channel in batch:
                    assert image.image_type_id == check_image_type_id
                    status_select = select(
                        ProcessedImages.status
                    ).where(
                        ProcessedImages.image_id == image.id,
                        ProcessedImages.progress_id
                        ==
                        self._current_processing.id
                    )
                    if channel is not None:
                        status_select = status_select.where(
                            ProcessedImages.channel == channel
                        )
                    status = db_session.execute(
                        status_select.group_by(ProcessedImages.status)
                    ).scalar_one_or_none()

                    if status is not None:
                        if batch_status is None:
                            batch_status = status
                        else:
                            assert batch_status == status

                input_batch = [
                    self._get_step_input(*image_channel, step_input_type)
                    for image_channel in batch
                ]

                if (config_key, batch_status) in result:
                    result[config_key, batch_status][1].extend(input_batch)
                else:
                    result[config_key, batch_status] = (config, input_batch)

        return result
    #pylint: enable=too-many-locals


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
        self._pending = {}
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

            self._processing_config = self._get_config(
                self._get_matched_expressions(Evaluator()),
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
                self._fill_pending(db_session)
                self._failed_dependencies = (
                    self._clean_pending_per_dependencies(db_session)
                )
                self._logger.debug(
                    'Unsuccessful dependencies prevent:\n\t%s',
                    '\n\t'.join(
                        f'Step {step} for {len(failed)} image type {image_type}'
                        ' images'
                        for (step, image_type), failed in
                        self._failed_dependencies.items()
                    )
                )


    def find_processing_outputs(self, processing_progress, db_session=None):
        """Return all logging and output filenames for given processing ID."""

        if db_session is None:
            #False positivie
            #pylint: disable=no-member
            #pylint: disable=redefined-argument-from-local
            with Session.begin() as db_session:
            #pylint: enable=no-member
            #pylint: enable=redefined-argument-from-local
                return self.find_processing_outputs(processing_progress,
                                                  db_session)

        if not isinstance(processing_progress, ImageProcessingProgress):
            return self.find_processing_outputs(
                db_session.scalar(
                    select(
                        ImageProcessingProgress
                    ).filter_by(
                        id=processing_progress
                    )
                ),
                db_session
            )

        main_fnames = get_log_outerr_filenames(
            existing_pid=processing_progress.process_id,
            task='*',
            parent_pid='',
            processing_step=processing_progress.step.name,
            image_type=processing_progress.image_type.name,
            **self._processing_config
        )
        print('Main fnames: ' + repr(main_fnames))
        assert len(main_fnames[0]) == len(main_fnames[1]) == 1

        return (
            tuple(fname[0] for fname in main_fnames),
            get_log_outerr_filenames(
                existing_pid='*',
                task='*',
                parent_pid=processing_progress.process_id,
                processing_step=processing_progress.step.name,
                image_type=processing_progress.image_type.name,
                **self._processing_config
            )
        )


    def __call__(self, limit_to_steps=None):
        """Perform all the processing for the given step."""


        #False positivie
        #pylint: disable=no-member
        with Session.begin() as db_session:
        #pylint: enable=no-member
            processing_sequence = get_processing_sequence(db_session)

        for step, image_type in processing_sequence:
            #pylint: disable=no-member
            with Session.begin() as db_session:
            #pylint: enable=no-member
                step = db_session.merge(step)
                step_name = step.name
                image_type = db_session.merge(image_type)
                image_type_name = image_type.name
                setup_process(task='main',
                              parent_pid='',
                              processing_step=step_name,
                              image_type=image_type.name,
                              **self._processing_config)

                if limit_to_steps is not None and step not in limit_to_steps:
                    self._logger.debug('Skipping disabled %s for %s frames',
                                       step.name,
                                       image_type.name)
                    continue

                if not self._check_ready(step, image_type, db_session):
                    self._logger.debug(
                        'Not ready for %s of %d %s frames',
                        step.name,
                        len(self._pending[(step.id, image_type.id)]),
                        image_type.name
                    )

                self._some_failed = False
                pending_images, step_input_type = self._start_step(step,
                                                                   image_type,
                                                                   db_session)
                if not pending_images:
                    continue

                processing_batches = self._get_batches(pending_images,
                                                       step_input_type,
                                                       db_session)
            for (
                    (_, start_status),
                    (config, batch)
            ) in processing_batches.items():
                self._logger.debug('Starting %s for a batch of %d images.',
                                   step_name,
                                   len(batch))

                self._process_batch(batch,
                                    start_status=start_status,
                                    config=config,
                                    step_name=step_name,
                                    image_type_name=image_type_name)
                self._logger.debug('Processed %s batch of %d images.',
                                   step_name,
                                   len(batch))
            #pylint: disable=no-member
            with Session.begin() as db_session:
            #pylint: enable=no-member
                self._current_processing = db_session.merge(
                    self._current_processing
                )
                self._current_processing.finished = (
                    #False positive
                    #pylint: disable=not-callable
                    sql.func.now()
                    #pylint: enable=not-callable
                )
                if self._some_failed:
                    dropped = self._clean_pending_per_dependencies(
                        db_session,
                        self._current_processing.step_id,
                        self._current_processing.image_type_id
                    )
                    for step_imtype, dropped_images in dropped.items():
                        if step_imtype in self._failed_dependencies:
                            self._failed_dependencies[
                                step_imtype
                            ].extend(
                                dropped_images
                            )
                        else:
                            self._failed_dependencies[step_imtype] = (
                                dropped_images
                            )


    def add_raw_images(self, image_collection):
        """Add the given RAW images to the database for processing."""

        #pylint: disable=no-member
        with Session.begin() as db_session:
        #pylint: enable=no-member
            default_expression_id = db_session.scalar(
                select(
                    ConditionExpression.id
                ).where(
                    ConditionExpression.notes == 'Default expression'
                )
            )
        configuration = self._get_config({default_expression_id},
                                         step_name='add_images_to_db')[0]
        processing_steps.add_images_to_db.add_images_to_db(image_collection,
                                                           configuration)


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

        matched_expressions = self._get_matched_expressions(
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
#pylint: enable=too-many-instance-attributes


def parse_command_line():
    """Return the command line configuration."""

    parser = ArgumentParser(
        description='Manually invoke the fully automated processing',
        default_config_files=[],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False
    )
    parser.add_argument(
        '--add-raw-images', '-i',
        nargs='+',
        default=[],
        help='Before processing add new raw images for processing. Can be '
        'specified as a combination of image files and directories which will'
        'be searched for FITS files.'
    )
    parser.add_argument(
        '--steps',
        nargs='+',
        default=None,
        help='Process using only the specified steps. Leave empty for full '
        'processing.'
    )
    return parser.parse_args()


def main(config):
    """Avoid global variables."""

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
    for img_to_add in config.add_raw_images:
        ProcessingManager().add_raw_images(
            find_fits_fnames(path.abspath(img_to_add))
        )
    ProcessingManager()(limit_to_steps=config.steps)


if __name__ == '__main__':
    main(parse_command_line())
