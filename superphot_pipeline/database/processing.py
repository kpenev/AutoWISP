#!/usr/bin/env python3

"""Handle data processing DB interactions."""

from tempfile import NamedTemporaryFile
import logging

from sqlalchemy import sql, select, update, and_
from general_purpose_python_modules.multiprocessing_util import setup_process

from superphot_pipeline import Evaluator
from superphot_pipeline.database.interface import Session
from superphot_pipeline.fits_utilities import get_primary_header
from superphot_pipeline.image_calibration.fits_util import\
    add_required_keywords,\
    add_channel_keywords
from superphot_pipeline import processing_steps
#False positive due to unusual importing
#pylint: disable=no-name-in-module
from superphot_pipeline.database.data_model import\
    ProcessingSequence,\
    StepDependencies,\
    ImageProcessingProgress,\
    ProcessedImages,\
    Configuration,\
    Step,\
    Image,\
    ImageType,\
    ObservingSession
from superphot_pipeline.database.data_model.provenance import\
    Camera,\
    CameraChannel,\
    CameraType
#pylint: enable=no-name-in-module


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

                * expressions: the condition expressions that are matched for
                  the given image and channel

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

    def _get_db_configuration(self, version, db_session):
        """Return list of Configuration instances given version."""

        #False positives:
        #pylint: disable=no-member
        param_version_subq = select(
            Configuration.parameter_id,
            #False positivie
            #pylint: disable=not-callable
            sql.func.max(Configuration.version).label('version'),
            #pylint: enable=not-callable
        ).filter(
            Configuration.version <= version
        ).group_by(
            Configuration.parameter_id
        ).subquery()

        return db_session.scalars(
            select(
                Configuration
            ).join(
                param_version_subq,
                sql.expression.and_(
                    (
                        Configuration.parameter_id
                        ==
                        param_version_subq.c.parameter_id
                    ),
                    (
                        Configuration.version
                        ==
                        param_version_subq.c.version
                    )
                )
            )
        ).all()
        #pylint: enable=no-member


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


    def _write_config_file(self, matched_expressions, outf, steps=None):
        """
        Write to given file configuration for given matched expressions.

        Returns:
            Set of tuples of parameters and values as set in the file. Used for
            comparing configurations.
        """

        result = set()
        #False positivie
        #pylint: disable=no-member
        with Session.begin() as db_session:
        #pylint: enable=no-member

            if steps is None:
                steps = db_session.scalars(select(Step).order_by(Step.id)).all()
            else:
                steps = [
                    db_session.execute(
                        select(Step).filter_by(name=step_name)
                    ).scalar_one()
                    for step_name in steps
                ]

            added_params = set()
            for this_step in steps:
                outf.write(f'[{this_step.name}]\n')
                step_config = self._get_param_values(
                    matched_expressions,
                    [
                        param.name
                        for param in this_step.parameters
                        if param.name not in added_params
                    ]
                )
                for param, value in step_config.items():
                    if value is not None:
                        outf.write(f'    {param} = {value}\n')
                        result.add((param, value))

                outf.write('\n')

        return frozenset(result)


    def _get_config(self, matched_expressions, step=None):
        """Return the configuration for the given step for given expressions."""

        with NamedTemporaryFile(mode='w') as config_file:
            config_key = self._write_config_file(matched_expressions,
                                                 config_file,
                                                 [step])
            config_file.flush()
            self._logger.debug('Wrote config file %s', repr(config_file.name))
            return getattr(processing_steps, step).parse_command_line(
                ['-c', config_file.name]
            ), config_key


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


    def _evaluate_expressions_image(self, image):
        """Add calibrated and DR filenames as attributes to given image."""

        evaluate = Evaluator(get_primary_header(image.raw_fname, True))
        calib_config = self._get_config(
            self._get_matched_expressions(evaluate),
            'calibrate',
        )[0]
        self._logger.debug('Calibration config: %s', repr(calib_config))
        add_required_keywords(evaluate.symtable, calib_config)

        self._evaluated_expressions[image.id] = {}
        all_channel_matched = None
        for channel_name, channel_slice in self._get_split_channels(
                image
        ).items():
            add_channel_keywords(evaluate.symtable,
                                 channel_name,
                                 channel_slice)
            matched_expressions = self._get_matched_expressions(evaluate)
            if all_channel_matched is None:
                all_channel_matched = matched_expressions
            else:
                all_channel_matched = all_channel_matched & matched_expressions
            self._evaluated_expressions[image.id][channel_name] = {
                'expressions': matched_expressions,
                'calibrated': calib_config['calibrated_fname'].format_map(
                    evaluate.symtable
                )
            }

            for required_expressions, value in self.configuration[
                    'data-reduction-fname'
            ][
                    'value'
            ].items():
                if required_expressions <= matched_expressions:
                    self._evaluated_expressions[
                        image.id
                    ][
                            channel_name
                    ][
                        'dr'
                    ] = value.format_map(evaluate.symtable)
                    break

        self._evaluated_expressions[image.id][None] = {
            'expressions': all_channel_matched
        }


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

        select_imgage_channel = select(
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

        for step, image_type in self._get_processing_sequence(db_session):
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
            ).subquery()

            self._pending[(step.id, image_type.id)] = db_session.execute(
                select_imgage_channel.outerjoin(
                    processed_subquery,
                    and_(Image.id == processed_subquery.c.image_id,
                         CameraChannel.name == processed_subquery.c.channel),
                ).where(
                    #This is how NULL comparison is done in SQLAlchemy
                    #pylint: disable=singleton-comparison
                    processed_subquery.c.image_id == None
                    #pylint: enable=singleton-comparison
                ).where(
                    Image.image_type_id == image_type.id
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


    def _cleanup_interrupted(self, db_session):
        """Cleanup previously interrupted processing for the current step."""

        need_cleanup = db_session.execute(
            select(
                Image,
                ProcessedImages,
                Step.name
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

        interrupted = []
        cleanup_step = need_cleanup[0][2]
        step_module = getattr(processing_steps, cleanup_step)
        matched_expressions = None
        for image, processed, step in need_cleanup:

            assert step == cleanup_step

            if image.id not in self._evaluated_expressions:
                self._evaluate_expressions_image(image)

            image_matches = self._evaluated_expressions[
                image.id
            ][
                processed.channel
            ][
                'expressions'
            ]
            if matched_expressions is None:
                matched_expressions = image_matches
                config, config_key = self._get_config(
                    matched_expressions,
                    step=step
                )
                config['processing_step'] = step
                if step == 'calibrate':
                    config['split_channels'] = self._get_split_channels(image)
            else:
                if matched_expressions != image_matches:
                    compare_config, compare_key = self._get_config(
                        matched_expressions,
                        step=step
                    )
                    if compare_key != config_key:
                        raise RuntimeError(
                            'Not all images with interrupted processing have '
                            'the same configuration: '
                            f'{config} vs. {compare_config}'
                        )

            interrupted.append((
                self._get_step_input(image,
                                     processed.channel,
                                     step_module.input_type),
                processed.status
            ))
        self._logger.warning(
            'Cleaning up interrupted %s processing of %d images:\n'
            '%s\n'
            'config: %s',
            cleanup_step,
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

        getattr(step_module, step_name)(
            batch,
            start_status,
            config,
            self._start_processing,
            self._end_processing
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
                           repr(self._processed_ids))
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
        self._logger.debug('Finished processing IDs: %s',
                           repr(self._processed_ids))
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


    def _group_pending_by_conditions(self, pending_images):
        """Group pendig_images grouped by condition expressions they satisfy."""

        result = []
        while pending_images:
#            pending_images = [
#                (db_session.merge(image, load=False), channel)
#                for image, channel in pending_images
#            ]
#            self.current_step = db_session.merge(self.current_step,
#                                                 load=False)
#            self._current_processing = db_session.merge(
#                self._current_processing,
#                load=False
#            )

            batch = []
            matched_expressions = self._evaluated_expressions[
                pending_images[-1][0].id
            ][
                pending_images[-1][1]
            ][
                'expressions'
            ]
            self._logger.debug(
                'Finding images matching expressions: %s',
                repr(matched_expressions)
            )

            for i in range(len(pending_images) - 1, -1, -1):
                self._logger.debug(
                    'Comparing %s to %s',
                    repr(
                        self._evaluated_expressions[
                            pending_images[i][0].id
                        ][
                            pending_images[i][1]
                        ][
                            'expressions'
                        ]
                    ),
                    repr(matched_expressions)
                )
                if self._evaluated_expressions[
                        pending_images[i][0].id
                ][
                    pending_images[i][1]
                ][
                    'expressions'
                ] == matched_expressions:
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
            result.append((matched_expressions, batch))
        return result


    def _get_batches(self, pending_images, step_input_type, db_session):
        """Return the batches of images to process with identical config."""

        result = {}
        for matched_expressions, batch in self._group_pending_by_conditions(
            pending_images
        ):
            config, config_key = self._get_config(
                matched_expressions,
                step=self.current_step.name
            )
            if self.current_step.name == 'calibrate':
                config['split_channels'] = self._get_split_channels(batch[0][0])
            config['processing_step']=self.current_step.name
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


    @staticmethod
    def _get_processing_sequence(db_session):
        """Return the sequence of step/image type pairs to process."""

        return db_session.execute(
            select(
                Step,
                ImageType
            ).select_from(
                ProcessingSequence
            ).join(
                Step,
                ProcessingSequence.step_id == Step.id
            ).join(
                ImageType,
                ProcessingSequence.image_type_id == ImageType.id
            )
        ).all()


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

            db_configuration = self._get_db_configuration(version, db_session)
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

            self.step_version = {
                step.name: max(
                    self.configuration[param.name]['version']
                    for param in step.parameters
                )
                for step in db_session.scalars(select(Step)).all()
            }

            self._fill_pending(db_session)


    def __call__(self, limit_to_steps=None):
        """Perform all the processing for the given step."""


        #False positivie
        #pylint: disable=no-member
        with Session.begin() as db_session:
        #pylint: enable=no-member
            self._cleanup_interrupted(db_session)

        for step, image_type in self._get_processing_sequence(db_session):
            if limit_to_steps is not None and step not in limit_to_steps:
                self._logger.debug('Skipping disabled %s for %s frames',
                                   step.name,
                                   image_type.name)
                continue

            #False positivie
            #pylint: disable=no-member
            with Session.begin() as db_session:
            #pylint: enable=no-member
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
                processing_batches = self._get_batches(pending_images,
                                                       step_input_type,
                                                       db_session)
            for (
                    (_, start_status),
                    (config, batch)
            ) in processing_batches.items():
                self._logger.debug('Starting %s for a batch of %d images.',
                                   step.name,
                                   len(batch))

                self._process_batch(batch, start_status, config, step.name)
                self._logger.debug('Processed %s batch of %d images.',
                                   step.name,
                                   len(batch))


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
                                        steps)
        else:
            self._write_config_file(matched_expressions, outf, steps)
#pylint: enable=too-many-instance-attributes


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)

    ProcessingManager()()
