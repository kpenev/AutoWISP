"""Views for displaying diagnostics for the calibration steps."""

from sqlalchemy import select

import pandas
from django.http import HttpResponseRedirect
from django.urls import reverse

from autowisp.database.interface import Session
from autowisp.catalog import read_catalog_file
#False positive
#pylint: disable=no-name-in-module
from autowisp.database.data_model import\
    MasterType,\
    MasterFile,\
    ImageType,\
    ImageProcessingProgress,\
    Step
#pylint: enable=no-name-in-module


def display_magfit_diagnostics(request, imtype, progress_id=None):
    """View displaying the scatter after magnitude fitting."""

    #False positive
    #pylint: disable=no-member
    with Session.begin() as db_session:
    #pylint: enable=no-member
        stat_subquery = select(
            MasterFile.progress_id,
            MasterFile.filename
        ).join(
            MasterType
        ).where(
            MasterType.name == 'magfit_stat'
        ).subquery()

        stat_cat_fnames = db_session.execute(
            select(
                stat_subquery.c.filename,
                MasterFile.filename
            ).join_from(
                MasterFile,
                stat_subquery,
                MasterFile.progress_id == stat_subquery.c.progress_id
            ).join(
                MasterType
            ).where(
                MasterType.name == 'magfit_catalog'
            )
        ).all()

    return HttpResponseRedirect(reverse('processing:progress'))


def display_diagnostics(request, step, imtype):
    """Common interface to all diagnostic views."""

    if step == 'fit_magnitudes':
        return display_magfit_diagnostics(request, imtype)
    return HttpResponseRedirect(reverse('processing:progress'))
