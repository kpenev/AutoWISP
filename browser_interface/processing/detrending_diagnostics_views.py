"""Views for displaying diagnostics for the calibration steps."""

import logging
from io import StringIO, BytesIO
import json

import matplotlib
from matplotlib import pyplot
from sqlalchemy import select

from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse

from autowisp import Evaluator
from autowisp.bui_util import hex_color
from autowisp.data_reduction import DataReductionFile
from autowisp.database.interface import Session
from autowisp.database.lightcurve_processing import LightCurveProcessingManager
from autowisp.diagnostics.detrending import \
    find_magfit_stat_catalog,\
    get_detrending_performance_data
#False positive
#pylint: disable=no-name-in-module
from autowisp.database.data_model import\
    Condition,\
    ConditionExpression,\
    Image,\
    MasterType,\
    MasterFile\
#pylint: enable=no-name-in-module


def _guess_labels(photref_entries):
    """Guess what would make good labels for plotting."""

    num_expr = len(photref_entries[0]['expressions'])
    print(
        'Expression sets: '
        +
        repr([
            set(entry['expressions'][i] for entry in photref_entries)
            for i in range(num_expr)
        ])
    )
    use_expr = [
        len(set(entry['expressions'][i] for entry in photref_entries)) > 1
        for i in range(num_expr)
    ]
    print(f'Use expr flags: {use_expr!r}')
    for entry in photref_entries:
        entry['label'] = ':'.join(
            expr for expr, use in zip(entry['expressions'], use_expr) if use
        )


def _init_detrending_session(request):
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
    request.session['diagnostics'] = {
        'detrending': {
            'match_expressions': ['step'] + match_expressions,
            'photref': [
                {
                    'id': f'mfit_{mphotref_id!s}',
                    'filenames': find_magfit_stat_catalog(mphotref_id),
                    'expressions': ('MFIT',) + tuple(
                        str(Evaluator(mphotref_fname)(expr))
                        for expr in match_expressions
                    )
                }
                for mphotref_id, mphotref_fname in master_photref_fnames
            ],
            'plot_config': {
                'x_range': ['', ''],
                'y_range': ['', ''],
                'mag_expression': ['phot_g_mean_mag', 'Gaia G mag'],
                'marker_size': '5'
            }
        }
    }


def _init_lc_detrending_session(request):
    """Add to browser session which EPD and TFA runs can be diagnosed."""

    lc_processing = LightCurveProcessingManager(dummy=True)
    photref_entries = request.session['diagnostics']['detrending']['photref']
    match_expressions = request.session[
        'diagnostics'
    ][
        'detrending'
    ][
        'match_expressions'
    ][1:]
    print('Adding LC detrending info to request')
    #False positive
    #pylint: disable=no-member
    with Session.begin() as db_session:
        #pylint: enable=no-member
        for (
            db_step,
            db_sphotref
        ) in LightCurveProcessingManager.select_step_sphotref(db_session,
                                                              False,
                                                              True):
            if (
                    not db_step.name.startswith('generate_')
                or
                    not db_step.name.endswith('_statistics')
            ):
                continue

            print(f'Adding info for {db_step!r} step, '
                  f'{db_sphotref.filename!r} sphotref')
            with DataReductionFile(db_sphotref.filename, 'r') as sphotref_dr:
                sphotref_header = sphotref_dr.get_frame_header()

            db_sphotref_image = db_session.scalar(
                select(
                    Image
                ).where(
                    Image.raw_fname.contains(sphotref_header['RAWFNAME']
                                             +
                                             '.fits')
                )
            )
            lc_processing.evaluate_expressions_image(db_sphotref_image,
                                                     db_session)

            catalog_fname, step_config = lc_processing.get_step_config(
                step=db_step,
                db_sphotref=db_sphotref,
                db_sphotref_image=db_sphotref_image,
                sphotref_header=sphotref_header,
                db_session=db_session
            )[:2]
            detrending_name = db_step.name.split('_')[1]
            stat_fname = step_config[
                detrending_name + '_statistics_fname'
            ].format_map(
                sphotref_header
            )
            photref_entries.append(
                {
                    'id': f'{detrending_name}_{db_sphotref.id!s}',
                    'filenames': (catalog_fname, stat_fname),
                    'expressions': (
                        (detrending_name.upper(),)
                        +
                        tuple(
                            str(Evaluator(sphotref_header)(expr))
                            for expr in match_expressions
                        )
                    )
                }
            )


def refresh_detrending_diagnostics(request):
    """Reset all diagnostics related entries in the BUI session """

    if 'diagnostics' in request.session:
        del request.session['diagnostics']
    return redirect('/processing/display_detrending_diagnostics')


def display_detrending_diagnostics(request):
    """View displaying the scatter after magnitude fitting."""


    print('Using session with keys: ' + repr(request.session.keys()))

    if (
        'diagnostics' not in request.session
        or
        'detrending' not in request.session['diagnostics']
    ):
        print('Refreshing session')
        _init_detrending_session(request)
        _init_lc_detrending_session(request)
        photref_entries = request.session[
            'diagnostics'
        ][
            'detrending'
        ][
            'photref'
        ]
        color_map = matplotlib.colormaps.get_cmap('tab10')
        for photref_index, entry in enumerate(photref_entries):
            entry['color'] = hex_color(color_map(photref_index % color_map.N))
            entry['marker'] = ''
            entry['scale'] = '1.0'
            entry['min_fraction'] = '0.8'

        _guess_labels(photref_entries)

    print('Using session: '
          +
          repr(request.session['diagnostics']['detrending']))

    return render(
        request,
        'processing/detrending_diagnostics.html',
        request.session['diagnostics']['detrending'],
    )


def create_plot(session_detrending):
    """Create the diagnostic plot per configuration in session."""

    pyplot.clf()
    pyplot.cla()

    plot_config = session_detrending['plot_config']
    for photref_info in session_detrending['photref']:
        if not photref_info['marker']:
            continue

        data = get_detrending_performance_data(
            *photref_info['filenames'],
            photref_info['expressions'][0],
            min_unrejected_fraction=float(photref_info['min_fraction']),
            magnitude_expression=plot_config['mag_expression'][0],
            skip_first_stat=False
        )
        pyplot.semilogy(
            data['magnitudes'],
            data['best_scatter'],
            linestyle='none',
            marker=photref_info['marker'],
            markersize=(float(plot_config['marker_size'])
                        *
                        float(photref_info['scale'])),
            markeredgecolor=(
                photref_info['color']
                if photref_info['marker'] in 'x+.,1234|_' else
                'none'
            ),
            markerfacecolor=photref_info['color'],
            label=photref_info['label']
        )

    try:
        pyplot.xlim(*(float(v) for v in plot_config['x_range']))
    except ValueError:
        pass

    try:
        pyplot.ylim(*(float(v) for v in plot_config['y_range']))
    except ValueError:
        pass

    pyplot.xlabel(plot_config['mag_expression'][1])
    pyplot.ylabel('MAD')
    pyplot.grid(True, which='both', linewidth=0.2)
    pyplot.legend()


def update_detrending_diagnostics_plot(request):
    """Generate and respond with update plot, also update session."""

    plot_config = json.loads(request.body.decode())
    print('Plot config: ' + repr(plot_config))

    matplotlib.use('svg')
    pyplot.style.use('dark_background')

    session_detrending = request.session['diagnostics']['detrending']
    session_detrending['plot_config'].update(plot_config)

    to_update = session_detrending['photref']
    for photref_info in to_update:
        this_config = plot_config['datasets'].get(str(photref_info['id']))
        if this_config is None:
            photref_info['marker'] = ''
        else:
            photref_info.update(this_config)


    print('Updated session: '
          +
          repr(request.session['diagnostics']['detrending']))

    request.session.modified = True

    create_plot(session_detrending)

    with StringIO() as image_stream:
        pyplot.savefig(image_stream, bbox_inches='tight', format='svg')
        return JsonResponse({
            'plot_data': image_stream.getvalue(),
            'plot_config': (
                request.session['diagnostics']['detrending']['plot_config']
            )
        })


def download_detrending_diagnostics_plot(request):
    """Send the user the diagnostics plot as a PDF file."""

    matplotlib.use('pdf')
    pyplot.style.use('default')

    create_plot(request.session['diagnostics']['detrending'])

    with BytesIO() as image_stream:
        pyplot.savefig(image_stream, bbox_inches='tight', format='pdf')
        return HttpResponse(
            image_stream.getvalue(),
            headers={
                'Content-Type': 'application/pdf',
                'Content-Disposition': (
                    'attachment; filename="detrending_performance.pdf"'
                )
            }
        )
