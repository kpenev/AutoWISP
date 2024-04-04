#!/usr/bin/env python3
#pylint: disable=too-many-lines

"""Handle data processing DB interactions."""

from tempfile import NamedTemporaryFile
import logging
from os import path
from sys import argv

from sqlalchemy import sql, select, update, and_, or_
from asteval import asteval
import numpy

from general_purpose_python_modules.multiprocessing_util import setup_process

from superphot_pipeline import Evaluator
from superphot_pipeline.database.interface import Session
from superphot_pipeline.fits_utilities import get_primary_header
from superphot_pipeline.image_calibration.fits_util import\
    add_required_keywords,\
    add_channel_keywords
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


class ExpressionMatcher:
    """
    Compare condition expressions for an image/channel to a target.

    Usually check if matched expressions and master expression values are
    identical, but also handles special case of calibrate step.
    """

    def __init__(self,
                 evaluated_expressions,
                 ref_image_id,
                 ref_channel,
                 master_expression_ids):
        """
        Set up comparison to the given evaluated expressions.

        """

        reference_evaluated = evaluated_expressions[ref_image_id][ref_channel]
        self._ref_matched = reference_evaluated['matched']
        self._ref_master_values = tuple(
            reference_evaluated['values'][expression_id]
            for expression_id in master_expression_ids
        )
        self._logger.debug(
            'Finding images matching expressions %s and values %s',
            repr(self._ref_matched),
            repr(self._ref_master_values)
        )
        <++> HANDLE CALIBRATE <++>

    def __call__(self, image_id, channel):
        """True iff the expressions for the given image/channel match."""

        image_evaluated = self._evaluated_expressions[image_id][channel]
        image_master_values = tuple(
            image_evaluated['values'][expression_id]
            for expression_id in master_expression_ids
        )

        self._logger.debug(
            'Comparing %s to %s and %s to %s',
            repr(image_evaluated['matched']),
            repr(self._ref_matched),
            repr(image_master_values),
            repr(self._ref_master_values)
        )
        return (
            image_evaluated['matched'] == target_matched_expressions
            and
            image_master_expression_values
            ==
            target_master_expression_values
        )


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

        added_params = set()
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


    def _split_by_master(self,
                         batch,
                         required_master_type,
                         db_session):
        """Split the given list of images by the best master of given type."""

        result = {}
        candidate_masters = self._get_candidate_masters(
            self._evaluate_expressions_image(*batch[0],
                                             True),
            required_master_type,
            db_session
        )
        if not candidate_masters:
            raise ValueError(
                f'No master {required_master_type.master_type.name} '
                f'found for image {batch[0][0].raw_fname} channel '
                f'{batch[0][1]}.'
            )
        if len(candidate_masters) == 1:
            result[candidate_masters[0].filename] = batch
        else:
            for image, channel in batch:
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
                if best_master_fname in result:
                    result[best_master_fname].extend((image, channel))
                else:
                    result[best_master_fname] = [(image, channel)]
        return result


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

        config, config_key = self._get_config(
            self._evaluated_expressions[batch[0][0].id][batch[0][1]]['matched'],
            db_step=step
        )
        config_key |= {master_expression_values}
        if step.name == 'calibrate':
            config['split_channels'] = self._get_split_channels(batch[0][0])
            key_extra = {
                (
                    'split_channels',
                    ''.join(
                        repr(c)
                        for c in batch[0][0].observing_session.camera.channels
                    )
                )
            }
            config_key |= key_extra
        config['processing_step'] = step.name

        result = {
            config_key: (config, batch)
        }
        for required_master_type in db_session.scalars(
            select(RequiredMasterTypes).filter_by(
                step_id=step.id,
                image_type_id=batch[0][0].image_type_id
            )
        ).all():
            for config_key, sub_batch in list(result.items()):
                splits = self._split_by_master(sub_batch,
                                               required_master_type,
                                               db_session)


                del result[config_key]
                for best_master_fname, sub_batch in splits:
                    new_config = dict(config)
                    new_config[
                        required_master_type.config_name.replace('-', '_')
                    ] = best_master_fname
                    key_extra = {
                        (required_master_type.config_name, best_master_fname)
                    }
                    result[config_key | key_extra] = (new_config, sub_batch)

        return result


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
                if path.exists(image_expressions[product]):
                    return Evaluator(image_expressions[product])

        self._logger.debug('Evaluating expressions for: %s',
                           repr(image))
        result = None
        evaluate = Evaluator(get_primary_header(image.raw_fname, True))
        evaluate.symtable['IMAGE_TYPE'] = image.image_type.name
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
            add_required_keywords(evaluate.symtable, calib_config)
            if result is None:
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
                all_channel['values'] = evaluated_expressions['values']
            else:
                all_channel['matched'] = (all_channel['matched']
                                          &
                                          evaluated_expressions['matched'])
                for expr_id in list(all_channel['values'].keys()):
                    if (
                        all_channel['values'][expr_id]
                        !=
                        evaluated_expressions['values'][expr_id]
                    ):
                        del all_channel['values'][expr_id]


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

        interrupted = {}
        cleanup_step = need_cleanup[0][2]
        input_type = getattr(processing_steps, cleanup_step.name).input_type
        for image, processed, step in need_cleanup:

            assert step == cleanup_step

            if image.id not in self._evaluated_expressions:
                self._evaluate_expressions_image(image)

            (config_key, (config, _)), = self._get_batch_config(
                [image],
                processed.channel,
                step,
                db_session
            ).items()
            if config_key not in interrupted:
                interrupted[config_key] = [[], config]

            interrupted[config_key][0].append((
                self._get_step_input(image, processed.channel, input_type),
                processed.status
            ))
        return interrupted.values()


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

        for interrupted, config in self._get_interrupted(need_cleanup,
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

        self.current_step = step
        self._current_processing = db_session.execute(
            select(
                ImageProcessingProgress
            ).filter_by(
                step_id=self.current_step.id,
                configuration_version=self.step_version[step.name]
            )
        ).scalar_one_or_none()

        if self._current_processing is None:
            self._current_processing = ImageProcessingProgress(
                step_id=step.id,
                configuration_version=self.step_version[step.name]
            )
            db_session.add(self._current_processing)

        pending_images = [
            (db_session.merge(image, load=False), channel)
            for image, channel in self._pending[(step.id, image_type.id)]
        ]

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


    def _process_batch(self, batch, start_status, config, step_name):
        """Run the current step for a batch of images given configuration."""

        step_module = getattr(processing_steps, step_name)

        new_master = getattr(step_module, step_name)(
            batch,
            start_status,
            config,
            self._start_processing,
            self._end_processing
        )
        if not new_master:
            return
        #False positivie
        #pylint: disable=no-member
        with Session.begin() as db_session:
        #pylint: enable=no-member

            master_id = db_session.scalar(
                #False positive
                #pylint: disable=not-callable
                select(sql.func.max(MasterFile.id))
                #pylint: enable=not-callable
            ) + 1

            master_type_id = db_session.scalar(
                select(
                    MasterType.id
                ).where(
                    MasterType.name == new_master['type']
                )
            )
            db_session.add(
                MasterFile(
                    id=master_id,
                    type_id=master_type_id,
                    progress_id=db_session.merge(
                        self._current_processing,
                        load=False
                    ).id,
                    filename=new_master['filename'],
                    use_smallest=new_master['preference_order']
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


    def _group_pending_by_conditions(self, pending_images, db_session):
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

        if self.current_step.name == 'calibrate':
            return group_calibrate_pending(pending_images,
                                           master_expression_ids)

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

            for i in range(len(pending_images) - 1, -1, -1):
                if match_expressions(
                    pending_images[i][0].id,
                    pending_images[i][1]
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
            result.append((batch, target_master_expression_values))
        return result


    def _get_batches(self, pending_images, step_input_type, db_session):
        """Return the batches of images to process with identical config."""

        result = {}
        for (
            by_condition,
            master_expression_values
        ) in self._group_pending_by_conditions(
            pending_images,
            db_session
        ):
            for config_key, (config, batch) in self._get_batch_config(
                by_condition,
                master_expression_values,
                self.current_step,
                db_session
            ).items():
                setup_process(task='main', **config)

                batch_status = None
                for image, channel in batch:
                    status = db_session.execute(
                        select(
                            ProcessedImages.status
                        ).where(
                            ProcessedImages.image_id == image.id
                        ).where(
                            ProcessedImages.channel == channel
                        ).where(
                            ProcessedImages.progress_id
                            ==
                            self._current_processing.id
                        )
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


    def __init__(self, version=None):
        """
        Set the public class attributes per the given configuartion version.

        Args:
            version(int):    The version of the parameters to get. If a
                parameter value is not specified for this exact version use the
                value with the largest version not exceeding ``version``. By
                default us the latest configuration version in the database.

        Returns:
            None
        """

        self._logger = logging.getLogger(__name__)
        self.current_step = None
        self._current_processing = None
        self.configuration = {}
        self.condition_expressions = {}
        self._evaluated_expressions = {}
        self._processed_ids = {}
        self._pending = {}
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
                print(repr(config_entry.conditions))
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

            for master_type in db_session.scalars(select(MasterType)).all():
                for expression in master_type.match_expressions:
                    self.condition_expressions[expression.id] = (
                        expression.expression
                    )

            self.step_version = {
                step.name: max(
                    self.configuration[param.name]['version']
                    for param in step.parameters
                )
                for step in db_session.scalars(select(Step)).all()
            }

            self._cleanup_interrupted(db_session)
            self._fill_pending(db_session)


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

                self._process_batch(batch, start_status, config, step_name)
                self._logger.debug('Processed %s batch of %d images.',
                                   step_name,
                                   len(batch))


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


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
    ProcessingManager()()
    exit(1)

    ProcessingManager().create_config_file(argv[1], 'test.cfg', ['calibrate'])
