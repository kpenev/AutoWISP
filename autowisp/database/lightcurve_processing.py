#!/usr/bin/env python3

"""Define a class that automates the processing of light curves."""
from os import path, getpid, getpgid, setsid
from socket import getfqdn

from sqlalchemy import sql, select, or_, and_, literal
import numpy

from general_purpose_python_modules.multiprocessing_util import\
    setup_process

from autowisp import DataReductionFile
from autowisp.catalog import read_catalog_file
from autowisp.database.interface import Session
from autowisp.database.processing import ProcessingManager
from autowisp.database.user_interface import get_processing_sequence
from autowisp.light_curves.collect_light_curves import DecodingStringFormatter
#False positive due to unusual importing
#pylint: disable=no-name-in-module
from autowisp.database.data_model import\
    Image,\
    InputMasterTypes,\
    LightCurveStatus,\
    LightCurveProcessingProgress,\
    MasterFile,\
    MasterType,\
    ProcessingSequence,\
    Step,\
    StepDependencies
#pylint: enable=no-name-in-module


class LightCurveProcessingManager(ProcessingManager):
    """
    Utilities for automated processing of lightcurves.

    Attrs:
        See `ProcessingManager`.

        _pending(dict):    Keys are 'EPD' and 'TFA' and values are lists
            of single photometric reference filenames for which the given
            detrending step is pending.
    """

    def _start_processing(self, input_fname):
        """
        Mark in the database that processing the given file has begun.

        Args:
            input_fname:    The filename of the input (DR or FITS) that is about
                to begin processing.

        Returns:
            None
        """

        #<++>


    def _end_processing(self, input_fname, status=1, final=True):
        """
        Record that the current step has finished processing the given file.

        Args:
            input_fname:    The filename of the input (DR or FITS) that was
                processed.

        Returns:
            None
        """

        #<++>


    def _cleanup_interrupted(self, db_session):
        """Cleanup previously interrupted processing for the current step."""

        #<++>


    def _clean_pending_per_dependencies(self, db_session):
        """
        Remove pending images/LCs from steps if they failed a required step.
        """

        #<++>


    def _start_step(self,
                    step,
                    db_sphotref_image,
                    sphotref_header):
        """
        Record start of processing and return the LCs and configuration to use.

        Args:
            step(Step):    The database step to start.

            sphotref_image_id(int):    The ID within the image table
                corresponding to the single photometric reference for which
                processing is starting.

            sphotref_master_id(int):    Same as above but the ID in the
                MasterFile table.

            db_session:    Database session to use for creating new processing
                progress.

        Returns:
            dict:
                The complete configuration to use for the specified processing.
        """

        matched_expressions = self._evaluated_expressions[
            db_sphotref_image.id
        ][
            sphotref_header['CLRCHNL']
        ][
            'matched'
        ]
        create_lc_cofig = self.get_config(
            matched_expressions,
            step_name='create_lightcurves'
        )[0]
        catalog = create_lc_cofig['lightcurve-catalog-fname'].format_map(
            sphotref_header
        )
        lc_fname = create_lc_cofig['lc-fname']
        assert path.exists(catalog)
        catalog = read_catalog_file(catalog)
        step_config = self.get_config(
            matched_expressions,
            db_step=step
        )
        srcid_formatter = DecodingStringFormatter()
        lc_fnames = map(
            lambda src_id: srcid_formatter.format(lc_fname,
                                                  *numpy.atleast_1d(src_id)),
            catalog.index
        )
        return (
            [lc for lc in lc_fnames if path.exists(lc)],
            step_config
        )


    def _check_ready(self, step, single_p, db_session):
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
            if self.pending[requirement]:
                self._logger.debug(
                    'Not ready for %s of %d %s frames because of %d pending %s '
                    'type ID images for step ID %s:\n\t%s',
                    step.name,
                    len(self.pending[(step.id, image_type.id)]),
                    image_type.name,
                    len(self.pending[requirement]),
                    requirement[1],
                    requirement[0],
                    '\n\t'.join(f'{db_session.merge(e[0])!r}: {e[1]!r}'
                                for e in self.pending[requirement])
                )
                return False
        return True



    def _prepare_processing(self, step, single_photref_fname, limit_to_steps):
        """Prepare for processing images of given type by a calibration step."""

        #pylint: disable=no-member
        with (
            Session.begin() as db_session,
            DataReductionFile(single_photref_fname) as sphotref_dr
        ):
        #pylint: enable=no-member
            step = db_session.merge(step)
            header = sphotref_dr.get_frame_header()
            image = db_session.scalar(
                select(Image).where(Image.raw_fname == header['RAWFNAME'])
            )
            self.evaluate_expressions_image(image, header['CLRCHNL'])

            setup_process(task='main',
                          parent_pid='',
                          processing_step=step.name,
                          image_type=image.image_type.name,
                          **self._processing_config)

            if limit_to_steps is not None and step.name not in limit_to_steps:
                self._logger.debug('Skipping disabled %s for %s frames',
                                   step.name,
                                   image.image_type.name)
                return None, None

            if not self._check_ready(step, image.image_type, db_session):
                return None, None

            return self._start_step(step, image, header)


    def set_pending(self, db_session):
        """
        Set the unprocessed images and channels split by step and image type.

        Args:
            db_session(Session):    The database session to use.


        Returns:
            {step name: [str, ...]}:
                The filenames of the single photometric reference DR files for
                which lightcurves exist but the given steps has not been
                performend yet.
        """

        master_cat_id = db_session.scalar(
            select(
                MasterType.id
            ).where(
                MasterType.name == 'lightcurve_catalog'
            )
        )
        create_lc_step_id = db_session.scalar(
            select(
                Step.id
            ).where(
                Step.name == 'create_lightcurves'
            )
        )
        pending = db_session.execute(
            select(
                Step.name,
                MasterFile.filename,
            ).select_from(
                ProcessingSequence
            ).join(
                Step
            ).join(
                MasterFile,
                literal(True)
            ).join(
                MasterType
            ).join(
                InputMasterTypes,
                InputMasterTypes.step_id == Step.id
            ).join(
                StepDependencies,
                StepDependencies.blocked_step_id == Step.id
            ).outerjoin(
                LightCurveProcessingProgress,
                and_(
                    (
                        LightCurveProcessingProgress.step_id
                        ==
                        Step.id
                    ),
                    (
                        LightCurveProcessingProgress.single_photref_id
                        ==
                        MasterFile.id
                    )
                )
            ).where(
                StepDependencies.blocking_step_id == create_lc_step_id
            ).where(
                MasterType.name == 'single_photref'
            ).where(
                InputMasterTypes.master_type_id == master_cat_id
            ).where(
                #pylint: disable=singleton-comparison
                or_(
                    LightCurveProcessingProgress.finished == False,
                    LightCurveProcessingProgress.finished == None
                )
                #pylint: enable=singleton-comparison
            )
        ).all()
        print('\n\t' + '\n\t'.join([repr(e) for e in pending]))
        for step, sphotref_fname in pending:
            if step not in self.pending:
                self.pending[step] = []
            self.pending[step].append(sphotref_fname)


    def __call__(self, limit_to_steps=None):
        """Perform all the processing for the given steps (all if None)."""

        #False positivie
        #pylint: disable=no-member
        with Session.begin() as db_session:
        #pylint: enable=no-member
            processing_sequence = get_processing_sequence(db_session,
                                                          'lightcurves')

        for step in processing_sequence:
            pass
            #<++>

if __name__ == '__main__':
    manager = LightCurveProcessingManager()
    print(repr(manager.pending))
