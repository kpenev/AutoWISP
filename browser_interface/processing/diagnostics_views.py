"""Views for displaying diagnostics for the calibration steps."""

from io import StringIO
import json

import numpy
import matplotlib
from matplotlib import pyplot
from sqlalchemy import select

from django.shortcuts import render
from django.http import HttpResponseRedirect, JsonResponse
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

    return render(
        request,
        'processing/detrending_diagnostics.html',
        request.session['diagnostics']['magfit'],
    )


def display_diagnostics(request, step, imtype):
    """Common interface to all diagnostic views."""

    if step == 'fit_magnitudes':
        return display_magfit_diagnostics(request, imtype)
    return HttpResponseRedirect(reverse('processing:progress'))


def update_diagnostics_plot(request):
    """Update the session with new configuration for the diagnostics plot."""

    plot_config = json.loads(request.body.decode())

    matplotlib.use('svg')
    pyplot.style.use('dark_background')
    pyplot.clf()

    for mphotref_id, style in plot_config.items():
        data = get_magfit_performance_data(mphotref_id,
                                           0.8,
                                           'phot_g_mean_mag',
                                           True)
        pyplot.semilogy(
            data['magnitudes'],
            data['best_scatter'],
            linestyle='none',
            marker=style['marker'],
            markeredgecolor=(style['color'] if style['marker'] in 'x+.,1234|_'
                             else 'none'),
            markerfacecolor=style['color']
        )

    pyplot.xlabel('Gaia G')
    pyplot.ylabel('MAD')
    pyplot.grid(True, which='both', linewidth=0.2)

    with StringIO() as image_stream:
        pyplot.savefig(image_stream, format='svg')
        return JsonResponse({'plot_data': image_stream.getvalue()})
