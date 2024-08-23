"""Define a class that automates the processing of light curves."""
from os import path, getpid, getpgid, setsid
from socket import getfqdn

from sqlalchemy import sql, select, or_


from autowisp.database.interface import Session
from autowisp.database.processing import ProcessingManager
from autowisp.database.user_interface import get_processing_sequence
#False positive due to unusual importing
#pylint: disable=no-name-in-module
from autowisp.database.data_model import\
    LightCurveStatus,\
    LightCurveProcessingProgress,\
    MasterFile,\
    MasterType
#pylint: enable=no-name-in-module


class LightcurveProcessingManager(ProcessingManager):
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

    def get_pending(self, db_session):
        """
        Return the unprocessed images and channels split by step and image type.

        Args:
            db_session(Session):    The database session to use.

            steps_imtypes(Step, ImageType):    The step image type combinations
                to determine pending images for. If unspecified, the full
                processing sequence defined in the database is used.

            invert(bool):    If True, returns successfully completed (not
                failed) instead of pending.


        Returns:
            {step name: [str, ...]}:
                The filenames of the single photometric reference DR files for
                which lightcurves exist but the given steps has not been
                performend yet.
        """

        return db_session.scalars(
            select(
                MasterFile.filename
            ).join(
                MasterType
            ).outerjoin(
                LightCurveProcessingProgress
            ).where(
                MasterType.name == 'single_photref'
            ).where(
                #pylint: disable=singleton-comparison
                or_(
                    LightCurveProcessingProgress.final == False,
                    LightCurveProcessingProgress.final == None
                )
                #pylint: enable=singleton-comparison
            )
        ).all()


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
