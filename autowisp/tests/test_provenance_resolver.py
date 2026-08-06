"""Unit tests for :mod:`autowisp.database.provenance_resolver`."""

from sqlalchemy.exc import NoResultFound

from autowisp.evaluator import Evaluator
from autowisp.database.interface import start_db_session
from autowisp.database.provenance_resolver import (
    get_or_create_observing_session,
)
from autowisp.tests import AutoWISPTestCase


def _make_header(**overrides):
    """Build a fake FITS-header dict matching the test survey configuration.

    Keys mirror the header expressions in ``test.cfg``; values match the
    rows imported from ``survey_instruments.json`` by
    :meth:`AutoWISPTestCase.setUp`.
    """

    header = {
        "VERSION": "0.0.X-1958",
        "CMSERIAL": "None",
        "TELNAME": "Canon 135mm f2 SN 230236",
        "MTID": "11",
        "OBSERVAT": "FLWO",
        "SITELAT": 31.68138,
        "SITELONG": -110.87857,
        "SITEALT": 2345.3,
        "NRACA": 10.2,
        "NDECCA": 45.0,
        "OBJECT": "G10124500_139",
        "DATE-OBS": "2020-01-01",
        "TIME-OBS": "00:00:00",
        "EXPTIME": 30.0,
    }
    header.update(overrides)
    return header


def _make_configuration():
    """Return a step configuration mirroring test.cfg's provenance fields."""

    return {
        "observer": "VERSION",
        "camera_serial_number": "CMSERIAL.strip()",
        "telescope_serial_number": "TELNAME",
        "mount_serial_number": "MTID",
        "observatory": "OBSERVAT",
        "observatory_location": ["SITELAT", "SITELONG", "SITEALT"],
        "target_ra": "NRACA",
        "target_dec": "NDECCA",
        "target_name": "OBJECT",
        "target_match_tolerance": 0.05,
        "observing_session_label": "OBJECT",
        "exposure_start_utc": 'DATE_OBS + "T" + TIME_OBS',
        "exposure_start_jd": None,
        "exposure_seconds": "EXPTIME",
    }


class TestProvenanceResolver(AutoWISPTestCase):
    """Unit tests for ``get_or_create_observing_session``."""

    def test_resolves_existing_survey_rows(self):
        """Resolver builds an ObservingSession from imported survey rows."""

        configuration = _make_configuration()
        evaluator = Evaluator(_make_header())
        with start_db_session() as db_session:
            session = get_or_create_observing_session(
                "object", evaluator, configuration, db_session
            )
            # Flush so the new ObservingSession is persistent and its
            # relationships (observer, camera, ...) can be loaded -- pending
            # objects don't lazy-load relationships from just the FK column.
            db_session.flush()
            self.assertEqual(session.observer.name, "0.0.X-1958")
            self.assertEqual(session.camera.camera_type.make, "Sony")
            self.assertEqual(
                session.telescope.serial_number,
                "Canon 135mm f2 SN 230236",
            )
            self.assertEqual(session.mount.serial_number, "11")
            self.assertEqual(session.observatory.name, "FLWO")
            self.assertEqual(session.target.name, "G10124500_139")
            self.assertEqual(session.label, "G10124500_139")

    def test_missing_survey_row_raises(self):
        """Resolver raises NoResultFound when a survey row is missing.

        Equivalent to the ``wisp-survey import`` step having been skipped:
        the matching ORM row simply isn't there. Use an unknown camera
        serial to trigger the same failure mode without tearing down the
        rest of the survey DB.
        """

        configuration = _make_configuration()
        evaluator = Evaluator(_make_header(CMSERIAL="unknown-serial"))
        with start_db_session() as db_session:
            with self.assertRaises(NoResultFound):
                get_or_create_observing_session(
                    "object", evaluator, configuration, db_session
                )
