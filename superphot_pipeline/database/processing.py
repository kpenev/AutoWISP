"""Handle data processing DB interactions."""

from os import path
import logging

from sqlalchemy import sql

from superphot_pipeline import Evaluator
from superphot_pipeline.database.interface import Session
#False positive due to unusual importing
#pylint: disable=no-name-in-module
from superphot_pipeline.database.data_model import\
    Image,\
    ImageProcessingProgress,\
    Configuration,\
    Step
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
            sql.expression.and_(
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


#    def _get_condition_expressions(self, db_configurations):
#        """Return the condition expressions required by the given configs."""
#
#        with Session.begin() as db_session:


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
        self.configuration = {}
        self.condition_expressions = {}
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
                step.name: max([
                    self.configuration[param.name]['version']
                    for param in step.parameters
                ])
                for step in db_session.query(Step).all()
            }


    def start_step(self, step_name):
        """
        Record the start of a processing step and return the config. to use.

        Args:
            step_name(str):    The name of the step to start.

        Returns:
            None
        """

        #False positivie
        #pylint: disable=no-member
        with Session.begin() as db_session:
        #pylint: enable=no-member
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

        #False positivie
        #pylint: disable=no-member
        with Session.begin() as db_session:
        #pylint: enable=no-member
            if isinstance(image, int):
                image = db_session.query(
                    Image
                ).filter_by(
                    id=image
                ).one()

            self._current_processing.images.append(image)


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

        evaluate = Evaluator(example_header)

        satisfied_expressions = set(
            expr_id
            for expr_id, expression in self.condition_expressions.items()
            if evaluate(expression)
        )

        #False positivie
        #pylint: disable=no-member
        with Session.begin() as db_session:
        #pylint: enable=no-member

            if steps is None:
                steps = db_session.query(Step).order_by(Step.id).all()
            else:
                steps = [
                    db_session.query(Step).filter_by(name=step_name).one()
                    for step_name in steps
                ]

            if isinstance(outf, str):
                outf = open(outf, 'w')
                close = True
            else:
                close = False

            try:
                added_params = set()
                for this_step in steps:
                    outf.write(f'[{this_step.name}]\n')
                    for param in this_step.parameters:
                        for required_expressios, value in self.configuration[
                                param.name
                        ][
                            "value"
                        ].items():
                            if required_expressios <= satisfied_expressions:
                                if param.name not in added_params:
                                    if value is not None:
                                        outf.write(
                                            f'    {param.name} = {value}\n'
                                        )
                                    added_params.add(param.name)
                                break
                    outf.write('\n')
                if close:
                    outf.close()
            except:
                if close:
                    outf.close()
                raise

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
