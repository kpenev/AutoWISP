"""Handle data processing DB interactions."""

from sqlalchemy import sql

from superphot_pipeline.database.interface import Session
from superphot_pipeline.database.data_model import\
    Image,\
    ImageProcessingProgress

class ProcessingManager:
    """
    Read configuration and record processing progress in the database.

    Attrs:
        configuration(dict):    Indexed by parameter name with values further
            dictionaries with keys:

                ``version``: the actual version used including fallback

                ``value``: dict tuple of expression IDs that an image must
                satisfy for the parameter to have a given value.

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
    """

    def _get_db_configuration(self, version, db_session):
        """Return list of Configuration instances given version."""

        param_version_stmt = db_session.query(
            Configuration.parameter_id,
            sql.func.max(Configuration.version).label('version'),
        ).filter(
            Configuration.version <= version
        ).group_by(
            Configuration.parameter_id
        ).subquery()

        return db_session.query(
            Configuration
        ).join(
            param_version_stmt,
            sql.expression._and(
                (
                    Configuration.parameter_id
                    ==
                    param_version_stmt.c.parameter_id
                ),
                (
                    Configuration.version
                    ==
                    param_version_stmt.c.version
                )
            )
        ).all()


    def _get_condition_expressions(self, db_configurations):
        """Return the condition expressions required by the given configs."""

        with Session.begin() as db_session:


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


        self.configuration = {}
        self.condition_expressions = {}
        with Session.begin() as db_session:
            db_configuration = self._get_db_configuration(version, db_session)
            for config_entry in db_configuration:
                if config_entry.parameter.name not in self.configuration:
                    self.configuration[config_entry.parameter.name] = {
                        'version': config_entry.version,
                        'value': {}
                    }
            self.configuration[config_entry.parameter.name]['value'][
                tuple(
                    sorted([
                        expr.id
                        for expr in config_entry.condition.expressions
                    ])
                )
            ] = config_entry.value

            for expr in config_entry.condition.expressions:
                if expr.id not in self.condition_expressions:
                    self.condition_expressions[expr.id] = expr.expression

            for step in db_session.query(Step).all():
                self.step_version[step.name] = max([
                    self.configuration[param.name]['version']
                    for param in step.parameters
                ])


    def start_step(self, step_name):
        """
        Record the start of a processing step and return the config. to use.

        Args:
            step_name(str):    The name of the step to start.

        Returns:
            None
        """

        with Session.begin() as db_session:
            self.current_step = db_session.query(
                Step
            ).filter_by(
                name=step_name
            ).one()
            self._current_processing = ImageProcessingProgress(
                step_id=self.current_step.id,
                configuration_version=self.step_version[step_name]
            )
            db_session.add(self._current_processing)


    def record_processed_image(self, image):
        """
        Record that the given image was processed by the currently active step.

        Args:
            image_id:    The image (ID or `Image` instance) that was processed.

        Returns:
            None
        """


        with Session.begin() as db_session:
            if isinstance(image, int):
                image = db_session.query(
                    Image
                ).filter_by(
                    id=image
                ).one()

            self._current_processing.images.append(image)
