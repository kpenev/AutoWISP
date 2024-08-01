"""Views for displaying diagnostics for the calibration steps."""

from io import StringIO

import numpy
import matplotlib
from matplotlib import pyplot
from sqlalchemy import select

from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse

from autowisp import Evaluator
from autowisp.database.interface import Session
from autowisp.diagnostics.detrending import get_magfit_performance_data
#False positive
#pylint: disable=no-name-in-module
from autowisp.database.data_model import\
    MasterType,\
    MasterFile,\
    ImageType,\
    ImageProcessingProgress,\
    Condition,\
    ConditionExpression
#pylint: enable=no-name-in-module


def init_magfit_session(request):
    """Add to browser session which magfit runs can be diagnosed."""

    #False positive
    #pylint: disable=no-member
    with Session.begin() as db_session:
    #pylint: enable=no-member
        master_photref_fnames = db_session.execute(
            select(
                MasterFile.id,
                MasterFile.filename
            ).join(
                MasterType
            ).where(
                MasterType.name == 'master_photref'
            ).order_by(
                MasterFile.progress_id
            )
        ).all()
        match_expressions = db_session.scalars(
            select(
                ConditionExpression.expression
            ).join_from(
                MasterType,
                Condition,
                MasterType.condition_id == Condition.id
            ).join(
                ConditionExpression
            ).where(
                MasterType.name == 'master_photref'
            )
        ).all()
    color_map = matplotlib.colormaps.get_cmap('tab10')
    request.session['diagnostics'] = {
        'magfit': {
            'match_expressions': match_expressions,
            'mphotref': [
                (
                    mphotref_id,
                    tuple(
                        Evaluator(mphotref_fname)(expr)
                        for expr in match_expressions
                    ),
                    (
                        '#'
                        +
                        ''.join([
                            f'{int(numpy.round(c * 255)):02x}'
                            for c in color_map(mphotref_index % color_map.N)[:3]
                        ])
                    ),
                    None
                )
                for mphotref_index, (mphotref_id, mphotref_fname) in enumerate(
                    master_photref_fnames
                )
            ]
        }
    }


def refresh_diagnostics(request):
    """Reset all diagnostics related entries in the BUI session """

    del request.session['diagnostics']
    return display_magfit_diagnostics(request, None)


def display_magfit_diagnostics(request, imtype):
    """View displaying the scatter after magnitude fitting."""

    if (
        'diagnostics' not in request.session
        or
        'magfit' not in request.session['diagnostics']
    ):
        init_magfit_session(request)

    matplotlib.use('svg')
    pyplot.style.use('dark_background')

    for (
        mphotref_id,
        _,
        color,
        marker
    ) in request.session['diagnostics']['magfit']['mphotref']:
        if marker is None:
            continue
        data = get_magfit_performance_data(mphotref_id,
                                    0.8,
                                    'phot_g_mean_mag',
                                    True)
        pyplot.plot(
            data['magnitudes'],
            data['best_scatter'],
            marker=marker,
            markeredgecolor=(color if marker in 'x+.,1234|_' else 'none'),
            markerfacecolor=color
        )

    context = dict(request.session['diagnostics']['magfit'])
    with StringIO() as image_stream:
        pyplot.savefig(image_stream, format='svg')
        context['plot'] = image_stream.getvalue()
    return render(
        request,
        'processing/detrending_diagnostics.html',
        context
    )


def display_diagnostics(request, step, imtype):
    """Common interface to all diagnostic views."""

    if step == 'fit_magnitudes':
        return display_magfit_diagnostics(request, imtype)
    return HttpResponseRedirect(reverse('processing:progress'))
