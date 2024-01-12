#!/usr/bin/env python3

"""Handle data processing DB interactions."""

from tempfile import NamedTemporaryFile
import logging

from sqlalchemy import sql, select, update, tuple_, and_

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
    ImageProcessingProgress,\
    ProcessedImages,\
    Configuration,\
    Step,\
    Image
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
            instances of all steps for which star_step was invoked.

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


    def _write_config_file(self, matched_expressions, outf, steps=None):
        """Write to given file configuration for given matched expressions."""

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

                outf.write('\n')


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


    def _get_config(self, matched_expressions, step=None):

        with NamedTemporaryFile(mode='w') as config_file:
            self._write_config_file(matched_expressions,
                                    config_file,
                                    [step])
            config_file.flush()
            self._logger.debug('Wrote config file %s', repr(config_file.name))
            return getattr(processing_steps, step).parse_command_line(
                ['-c', config_file.name]
            )


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
        )
        self._logger.debug('Calibration config: %s', repr(calib_config))
        add_required_keywords(evaluate.symtable, calib_config)

        self._evaluated_expressions[image.id] = {}
        all_channel_matched = None
        for channel_name, channel_slice in (
                calib_config['split_channels'].items()
        ):
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


    def _get_pending_images(self, db_session):
        """
        Return the images and channels to process by the current step.

        Args:
            db_session(Session):    The database session to use.

        Returns:
            (Image, [str]):
                The images and channels for which all required inputs exist
                with correct versions but to which ``step`` has not been applied
                with the current configuration.
        """

        if not self.current_step.requires:
            return db_session.scalars(
                select(
                    Image
                ).outerjoin(
                    ProcessedImages
                ).where(
                    #That's how NULL comparison works in sqlalchemy
                    #pylint: disable=singleton-comparison
                    ProcessedImages.image_id == None
                    #pylint: enable=singleton-comparison
                )
            ).all()

        required_progress_ids = db_session.scalars(
            select(
                ImageProcessingProgress.id
            ).where(
                tuple_(
                    ImageProcessingProgress.step_id,
                    ImageProcessingProgress.configuration_version
                ).in_([
                    (step.id, self.step_version[step.name])
                    for step in self.current_step.requires
                ])
            )
        ).all()
        match_inputs_subq = select(
            ProcessedImages.image_id,
            ProcessedImages.channel,
            #False positive
            #pylint: disable=not-callable
            sql.func.count().label('num_satisfied')
            #pylint: enable=not-callable
        ).where(
            ProcessedImages.progress_id.in_(required_progress_ids)
        ).where(
            ProcessedImages.status > 0
        ).group_by(
            ProcessedImages.image_id,
            ProcessedImages.channel
        ).subquery()

        done_subq = select(
            ProcessedImages.image_id,
            ProcessedImages.channel,
        ).where(
            ProcessedImages.progress_id == self._current_processing.id
        ).subquery()

        pending_image_id_subq = select(
            match_inputs_subq
        ).outerjoin(
            done_subq,
            and_(
                match_inputs_subq.c.image_id
                ==
                done_subq.c.image_id,
                match_inputs_subq.c.channel
                ==
                done_subq.c.channel
            )
        ).where(
            match_inputs_subq.c.num_satisfied == len(required_progress_ids)
        ).where(
            #That's how NULL comparison works in sqlalchemy
            #pylint: disable=singleton-comparison
            done_subq.c.image_id == None
            #pylint: enable=singleton-comparison
        ).subquery()

        return db_session.execute(
            select(
                Image,
                pending_image_id_subq.c.channel
            ).join(
                pending_image_id_subq,
                #False positive
                #pylint: disable=no-member
                Image.id == pending_image_id_subq.c.image_id
                #pylint: enable=no-member
            )
        ).all()


    def _cleanup_interrupted(self, db_session):
        """Cleanup previously interrupted processing for the current step."""

        step_module = getattr(processing_steps, self.current_step.name)

        matched_expressions = None
        for image, processed in db_session.execute(
            select(
                Image, ProcessedImages
            ).join(
                ProcessedImages
            ).where(
                ProcessedImages.progress_id == self._current_processing.id
            ).where(
                ProcessedImages.status == 0
            )
        ):
            if image.id not in self._evaluated_expressions:
                self._evaluate_expressions_image(image)

            image_matches = self._evaluated_expressions[
                image.id
            ][
                processed.channel
            ][
                'expressions'
            ]
            if image_matches != matched_expressions:
                matched_expressions = image_matches
                config = self._get_config(
                    matched_expressions,
                    step=self.current_step.name
                )
                config['processing_step']=self.current_step.name

            interrupted_fname = self._get_step_input(image,
                                                     processed.channel,
                                                     step_module.input_type)
            self._logger.warning(
                'Cleaning up interrupted %s processing of %s',
                self.current_step.name,
                interrupted_fname
            )
            step_module.cleanup_interrupted(interrupted_fname, config)
            db_session.delete(processed)


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


    def _start_step(self, step_name):
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

        #False positivie
        #pylint: disable=no-member
        with Session.begin() as db_session:
        #pylint: enable=no-member
            self.current_step = db_session.execute(
                select(
                    Step
                ).filter_by(
                    name=step_name
                )
            ).scalar_one()
            self._current_processing = db_session.execute(
                select(
                    ImageProcessingProgress
                ).filter_by(
                    step_id=self.current_step.id,
                    configuration_version=self.step_version[step_name]
                )
            ).scalar_one_or_none()

            if self._current_processing is None:
                self._current_processing = ImageProcessingProgress(
                    step_id=self.current_step.id,
                    configuration_version=self.step_version[step_name]
                )
                db_session.add(self._current_processing)

            self._cleanup_interrupted(db_session)

            pending_images = self._get_pending_images(db_session)
            self._logger.debug('Pending images: %s', repr(pending_images))
            self._processed_ids = {}
            step_input_type = getattr(
                processing_steps,
                self.current_step.name
            ).input_type

            if step_input_type == 'raw':
                pending_images = [(image, None) for image in pending_images]

            for image, channel_name in pending_images:
                if image.id not in self._evaluated_expressions:
                    self._evaluate_expressions_image(image)
                self._init_processed_ids(image, [channel_name], step_input_type)

            return pending_images, step_input_type


    def _process_batch(self, batch, config, step_name):
        """Run the current step for a batch of images given configuration."""

        step_module = getattr(processing_steps, step_name)

        getattr(step_module, step_name)(
            batch,
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
                        status=0
                    )
                )


    def _end_processing(self, input_fname, status=1):
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
                        'status': status
                    }
                    for finished_id in self._processed_ids[input_fname]
                ]
            )


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


    def __call__(self, steps=None):
        """Perform all the processing for the given step."""


        if steps is None:
            #False positivie
            #pylint: disable=no-member
            with Session.begin() as db_session:
            #pylint: enable=no-member
                steps = db_session.scalars(select(Step.name)).all()

        for step_name in steps:
            pending_images, step_input_type = self._start_step(step_name)

            while pending_images:
                #False positivie
                #pylint: disable=no-member
                with Session.begin() as db_session:
                #pylint: enable=no-member
                    pending_images = [
                        (db_session.merge(image, load=False), channel)
                        for image, channel in pending_images
                    ]
                    self.current_step = db_session.merge(self.current_step,
                                                         load=False)
                    self._current_processing = db_session.merge(
                        self._current_processing,
                        load=False
                    )

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
                    config = self._get_config(
                        matched_expressions,
                        step=step_name
                    )
                    config['processing_step']=step_name
                    setup_process(task='main', **config)

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
                            batch.append(
                                self._get_step_input(
                                    *pending_images.pop(i),
                                    step_input_type
                                )
                            )
                            self._logger.debug(
                                'Added image to batch, now: %s',
                                repr(batch)
                            )
                        else:
                            self._logger.debug('Not a match')

                self._process_batch(batch, config, step_name)
                self._logger.debug(
                    'Processed batch leaving %d pending images',
                    len(pending_images)
                )


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

    ProcessingManager()(['calibrate', 'find_stars', 'solve_astrometry'])
