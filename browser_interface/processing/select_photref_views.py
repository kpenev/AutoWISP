"""Implement the view for selecting single photometric reference."""

from io import StringIO
from functools import reduce

#from PIL.ImageTransform import AffineTransform
from django.shortcuts import render, redirect
import matplotlib
from matplotlib import pyplot
from sqlalchemy import select
import pandas

from autowisp.database.image_processing import\
    ImageProcessingManager,\
    get_master_expression_ids,\
    remove_failed_prerequisite
from autowisp.database.interface import Session
from autowisp.database.user_interface import get_processing_sequence
from autowisp.processing_steps.calculate_photref_merit import\
    calculate_photref_merit
#False positive due to unusual importing
#pylint: disable=no-name-in-module
from autowisp.database.data_model import\
    MasterType,\
    InputMasterTypes,\
    ConditionExpression,\
    Step
#pylint: enable=no-name-in-module
from autowisp.bui_util import encode_fits
from .display_fits_util import update_fits_display


def _get_missing_photref(request):
    """Add all frame sets missing photometric reference to the session."""

    assert 'need_photref' not in request.session
    processing = ImageProcessingManager(dummy=True)
    #False positivie
    #pylint: disable=no-member
    with Session.begin() as db_session:
    #pylint: enable=no-member
        master_type_id = db_session.scalar(
            select(MasterType.id).filter_by(name='single_photref')
        )
        magfit_steps = [entry for entry in get_processing_sequence(db_session)
                        if entry[0].name == 'fit_magnitudes']
        processing.set_pending(db_session, magfit_steps)
        for step in magfit_steps:
            for pending in processing.pending[
                    (step[0].id, step[1].id)
            ]:
                processing.evaluate_expressions_image(pending[0], db_session)

        request.session['demo'] = False
        if not reduce(lambda x, y: bool(x) or bool(y),
                      processing.pending.values(),
                      False):
            request.session['demo'] = True
            processing.set_pending(db_session,
                                   magfit_steps,
                                   True)

        astrom_step_id = db_session.scalar(
            select(Step.id).filter_by(name='solve_astrometry')
        )
        for (
                (step_id, image_type_id),
                pending_images
        ) in processing.pending.items():
            remove_failed_prerequisite(pending_images,
                                       image_type_id,
                                       astrom_step_id,
                                       db_session)
            input_master_type = db_session.scalar(
                select(InputMasterTypes).filter_by(
                    step_id=step_id,
                    image_type_id=image_type_id,
                    master_type_id=master_type_id
                )
            )
            request.session['need_photref'] = {
                'master_expressions': [
                    db_session.scalar(
                        select(
                            ConditionExpression.expression
                        ).filter_by(
                            id=expr_id
                        )
                    )
                    for expr_id in get_master_expression_ids(step_id,
                                                             image_type_id,
                                                             db_session)
                ],
                'master_values': []
            }

            by_photref = processing.group_pending_by_conditions(
                pending_images,
                db_session,
                match_observing_session=False,
                step_id=step_id,
                masters_only=True
            )
            for by_master_values, master_values in by_photref:
                if (
                    request.session['demo']
                    or
                    not processing.get_master_fname(
                        by_master_values[0][0].id,
                        by_master_values[0][1],
                        'single_photref'
                    )
                ):
                    config = processing.get_config(
                        matched_expressions=None,
                        db_session=db_session,
                        image_id=by_master_values[0][0].id,
                        channel=by_master_values[0][1],
                        step_name='calculate_photref_merit'
                    )[0]
                    request.session[
                        'need_photref'
                    ][
                        'master_values'
                    ].append(
                        (
                            list(master_values),
                            config,
                            [
                                (
                                    processing.get_step_input(image,
                                                              channel,
                                                              'calibrated'),
                                    processing.get_step_input(image,
                                                              channel,
                                                              'dr'),
                                    image.id,
                                    channel
                                )
                                for image, channel, _ in by_master_values
                            ]
                        )
                    )
    request.session.modified = True


def _get_merit_data(request, target_index):
    """Add to the session the merit information for selecting single ref."""

    if 'merit_info' not in request.session:
        request.session['merit_info'] = {}
    if str(target_index) not in request.session['merit_info']:
        print('Calculating merit for target ' + str(target_index))
        config, batch = (
            request.session['need_photref']['master_values'][target_index][1:]
        )
        request.session['merit_info'][str(target_index)] = (
            calculate_photref_merit(
                [entry[1] for entry in batch],
                config
            ).sort_values(
                by='merit',
                ascending=False
            ).to_json()
        )
    request.session.modified = True


def _create_merit_histograms(merit_data, image_index):
    """Create SVG histograms of various merit metrics showing image in each."""

    matplotlib.use('svg')
    pyplot.style.use('dark_background')
    result = []
    for column in merit_data.columns:
        if column == 'dr' or column.startswith('qnt_'):
            continue
        with StringIO() as image_stream:
            pyplot.hist(merit_data[column],
                        bins='auto',
                        linewidth=0,
                        color='white')
            pyplot.axvline(x=merit_data[column].iloc[image_index],
                           linewidth=5,
                           color='red')
            if column == 'merit':
                pyplot.suptitle('merit', fontsize=32)
            else:
                quantile = merit_data['qnt_' + column].iloc[image_index]
                pyplot.suptitle(column + f' ({quantile:.3f} quantile)',
                                fontsize=32)
            pyplot.savefig(image_stream, format='svg')
            result.append(image_stream.getvalue())
            pyplot.clf()
    return result


def select_photref_image(request,
                         *,
                         target_index,
                         recalculate=False):
    """Display the interface for reviewing canditate reference frames."""

    assert request.method == 'GET'
    print('Image view with request: ' + repr(request))
    update_fits_display(request)
    image_index = request.session['fits_display']['image_index']
    if recalculate:
        print('Deleting merit info')
        del request.session['merit_info']
    _get_merit_data(request, target_index)
    print('Merit info keys: ' + repr(request.session['merit_info'].keys()))

    merit_data = pandas.read_json(
        request.session['merit_info'][str(target_index)]
    )
    fits_fname = request.session[
        'need_photref'
    ][
        'master_values'
    ][
        target_index
    ][
        2
    ][
        #False positive
        #pylint:disable=no-member
        merit_data.index[image_index]
        #pylint:enable=no-member
    ][
        0
    ]
    context = {
        'target_index': target_index,
        #False positive
        #pylint: disable=no-member
        'num_images': merit_data.shape[0],
        #pylint: enable=no-member
        'histograms': _create_merit_histograms(merit_data, image_index),
        'view_config': request.session.get('view_config', 'undefined')
    }
    context.update(request.session['fits_display'])
    context.update(
        encode_fits(
            fits_fname,
            request.session['fits_display']['range'],
            request.session['fits_display']['transform']
        )
    )
    return render(
        request,
        'processing/select_photref_image.html',
        context
    )


def select_photref_target(request, recalc=False):
    """Display view to select which of the missing photrefs to define."""

    if recalc:
        request.session.flush()
        return redirect('/processing/select_photref_target')
    if 'need_photref' not in request.session:
        _get_missing_photref(request)

    print('Request master values: '
          +
          repr(request.session['need_photref']['master_values']))
    return render(
        request,
        'processing/select_photref_target.html',
        {
            'master_expressions': request.session[
                'need_photref'
            ][
                'master_expressions'
            ] + ['Num. Images'],
            'master_values': [
                target[0] + [len(target[2])]
                for target in request.session['need_photref']['master_values']
            ],
            'view_config': request.body
        }
    )


def record_photref_selection(request, target_index, image_index):
    """Record a single photometric reference frame selected by the user."""

    if request.session['demo']:
        print('Demo only! Not saving selected reference!')
        return None
    print('Merit info keys: ' + repr(request.session['merit_info'].keys()))
    merit_data = pandas.read_json(
        request.session['merit_info'][str(target_index)]
    )
    dr_fname = request.session[
        'need_photref'
    ][
        'master_values'
    ][
        target_index
    ][
        2
    ][
        #False positive
        #pylint:disable=no-member
        merit_data.index[image_index]
        #pylint:enable=no-member
    ][
        1
    ]
    del request.session[
        'need_photref'
    ][
        'master_values'
    ][
        target_index
    ]
    del request.session['merit_info'][str(target_index)]
    for shift_index in range(target_index + 1,
                             len(request.session[
                                 'need_photref'
                             ][
                                 'master_values'
                             ])):
        if str(shift_index) in request.session['merit_info']:
            request.session['merit_info'][str(shift_index - 1)] =\
                request.session['merit_info'].pop(str(shift_index))

    request.session.modified = True

    ImageProcessingManager(dummy=True).add_masters(
        {
            'type': 'single_photref',
            'filename': dr_fname,
            'preference_order': None,
            'disable': False
        }
    )
    return redirect('/processing/select_photref_target')
