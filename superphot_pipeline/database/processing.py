"""Handle data processing DB interactions."""

from os import path
import logging

from sqlalchemy import sql, select, tuple_, and_

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

        _processed_ids(dict):    The keys are the filenames of the required
            inputs (DR or FITS) for the current step and the values are
            dictionaries with keys ``'image_id'`` and ``'channel'`` identifying
            what was processed.
    """

    def _get_db_configuration(self, version, db_session):
        """Return list of Configuration instances given version."""

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
                step_config = self._get_config(
                    matched_expressions,
                    [
                        param.name
                        for param in this_step.parameters
                        if param.name not in added_params
                    ]
                )
                for param, value in step_config.items():
                    outf.write(f'    {param.name} = {value}\n')

                outf.write('\n')


    def _get_config(self,
                    matched_expressions,
                    parameters=None,
                    as_args=False,
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

        if as_args:
            return [
                entry
                for param in parameters
                for entry in ['--' + param, get_param_value(param)]
            ]

        return {param: get_param_value(param) for param in parameters}


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


    def _evaluate_expressions_image(self, image, step_input_type):
        """Add calibrated and DR filenames as attributes to given image."""

        evaluate = Evaluator(get_primary_header(image.raw_fname))
        calib_config = processing_steps.calibrate.parse_command_line(
            self._get_config(
                self._get_matched_expressions(evaluate),
                'calibrate',
                as_args=True
            )
        )
        add_required_keywords(evaluate.symtable, calib_config)

        self._evaluated_expressions[image.id] = {}
        for channel_name, channel_slice in calib_config['split_channels']:
            add_channel_keywords(evaluate.symtable,
                                 channel_name,
                                 channel_slice)
            matched_expressions = self._get_matched_expressions(evaluate)
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
                    ] = evaluate(value).format_map(evaluate.symtable)
                    break
            if step_input_type == 'raw':
                step_input_fname = image.raw_fname
            elif step_input_type.startswith('calibrated'):
                step_input_fname = self._evaluated_expressions[
                    image.id
                ][
                    channel_name
                ][
                    'calibrated'
                ]
            elif step_input_type == 'dr':
                step_input_fname = self._evaluated_expressions[
                    image.id
                ][
                    channel_name
                ][
                    'dr'
                ]
            else:
                raise ValueError(f'Invalid step input type {step_input_type}')

            self._processed_ids[step_input_fname] = {
                'image_id': image.id,
                'channel': channel_name
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
                db_session.scalars(
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
            )

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
        ).group_by(
            ProcessedImages.image_id,
            ProcessedImages.channel
        )

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
                Image.id == pending_image_id_subq.c.image_id
            )
        ).all()


    def _start_step(self, step_name, db_session):
        """
        Record the start of a processing step and return the images to process.

        Args:
            step_name(str):    The name of the step to start.

        Returns:
            None
        """

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

        if self._current_processing is not None:
            self._current_processing = ImageProcessingProgress(
                step_id=self.current_step.id,
                configuration_version=self.step_version[step_name]
            )
            db_session.add(self._current_processing)

        pending_images = self._get_pending_images(db_session)
        self._processed_ids = {}
        step_input_type = getattr(
            processing_steps,
            self.current_step.name
        ).input_type
        for image, _ in pending_images:
            if image.id not in self._evaluated_expressions:
                self._evaluate_expressions_image(image, step_input_type)

        return pending_images


    def _process_batch(self, batch, config):
        """Run the current step for a batch of images given configuration."""

        step_module = getattr(processing_steps, self.current_step.name)

        getattr(
            step_module,
            self.current_step.name
        )(
            batch,
            config,
            self._record_processed_image
        )

    def _record_processed(self, input_fname):
        """
        Record that the given input filename was processed by the current step.

        Args:
            input_fname:    The filename of the input (DR or FITS) that was
                processed.

        Returns:
            None
        """

        assert self.current_step is not None
        assert self._current_processing is not None
        #False positivie
        #pylint: disable=no-member
        with Session.begin() as db_session:
        #pylint: enable=no-member
            db_session.add(
                ProcessedImages(
                    **processed_ids[input_fname],
                    processing_id=self._current_processing.id
                )
            )


    def __init__(self, version):
        """
        Set the public class attributes per the given configuartion version.

        Args:
            version(int):    The version of the parameters to get. If a
                parameter value is not specified for this exact version use the
                value with the largest version not exceeding ``version``.

        Returns:
            None
        """


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

        #False positivie
        #pylint: disable=no-member
        with Session.begin() as db_session:
        #pylint: enable=no-member
            if steps is None:
                steps = db_session.scalars(select(Step.name)).all()

        for step_name in steps:
            #False positivie
            #pylint: disable=no-member
            with Session.begin() as db_session:
            #pylint: enable=no-member
                pending_images = self._start_step(step_name, db_session)

            while pending_images:
                batch = [pending_images.pop()]
                matched_expressions = self._evaluated_expressions[
                    batch[0][0].id
                ][
                    batch[0][1]
                ][
                    'expressions'
                ]
                #False positivie
                #pylint: disable=no-member
                with Session.begin() as db_session:
                #pylint: enable=no-member
                    config = self._get_config(
                        matched_expressions,
                        db_session=db_session
                    )

                for i in range(len(pending_images) - 1, -1):
                    if matched_expressions[
                            pending_images[i][0].id
                    ][
                        pending_images[i][1]
                    ][
                        'expressions'
                    ] == matched_expressions:
                        batch.append(pending_images.pop(i))

                self._process_batch(batch, config)


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


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)

    test_fits = path.join(
        path.dirname(
            path.dirname(
                path.dirname(
                    path.abspath(__file__)
                )
            )
        ),
        'usage_examples',
        'test_data',
        '10-20170306',
        '10-464933_2_R1.fits.fz'
    )
    ProcessingManager(1).create_config_file(
        test_fits,
        'test.cfg'
    )
