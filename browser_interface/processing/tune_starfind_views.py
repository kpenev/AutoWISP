"""Implement views for tuning source extraction."""

import json

from django.shortcuts import render, redirect
from django.http import JsonResponse
from sqlalchemy import select

from autowisp import SourceFinder, Evaluator
from autowisp.database.interface import Session
from autowisp.database.processing import ProcessingManager
#False positive
#pylint: disable=no-name-in-module
from autowisp.database.data_model import\
    Step,\
    ImageType,\
    ProcessingSequence
from autowisp.database.data_model import provenance
#pylint: enable=no-name-in-module
from autowisp.bui_util import encode_fits

from .display_fits_util import update_fits_display


def _init_session(request, processing, db_session):
    """Set default django session entries first time the interface is opened"""


    if 'starfind' in request.session:
        return
    assert (
        len(processing.configuration['telescope-serial-number']['value'])
        ==
        1
    )
    assert (
        len(processing.configuration['camera-serial-number']['value'])
        ==
        1
    )

    grouping_expressions = []
    for component in ['Telescope', 'Camera']:
        sn_expression = list(
            processing.configuration.get(
                component.lower() + '-serial-number'
            )[
                'value'
            ].values()
        )[0]
        for instrument_type in db_session.scalars(
            select(getattr(provenance, component + 'Type'))
        ).all():
            serial_numbers = set(
                instrument.serial_number
                for instrument in getattr(instrument_type,
                                          component.lower() + 's')
            )
            grouping_expressions.append(
                (
                    f'{sn_expression} in {serial_numbers!r}',
                    f'{instrument_type.make} {instrument_type.model} '
                    f'{component.lower()}s'
                )
            )
    grouping_expressions.extend([
        ('CLRCHNL', '{value} channel'),
        (
            list(
                processing.configuration.get(
                    'exposure-seconds'
                )[
                    'value'
                ].values()
            )[0],
            '{value}s exposure'
        )
    ])

    request.session['starfind'] = {
        'grouping_expressions': grouping_expressions
    }


def _get_pending(request):
    """Add to ``request.session`` all image/channel pending star finding ."""

    processing = ProcessingManager(dummy=True)

    #False positivie
    #pylint: disable=no-member
    with Session.begin() as db_session:
    #pylint: enable=no-member
        _init_session(request, processing, db_session)
        if 'pending' in request.session['starfind']:
            return

        request.session['starfind']['pending'] = {}
        find_star_steps = db_session.execute(
            select(
                Step,
                ImageType
            ).select_from(
                ProcessingSequence
            ).join(
                Step,
                ProcessingSequence.step_id == Step.id
            ).join(
                ImageType,
                ProcessingSequence.image_type_id == ImageType.id
            ).where(
                Step.name == 'find_stars'
            )
        ).all()

        pending = processing.get_pending(db_session, find_star_steps)
        for step, imtype in find_star_steps:
            grouping = {}
            for image, channel in pending[step.id, imtype.id]:
                evaluator = processing.evaluate_expressions_image(image,
                                                                  channel,
                                                                  True)
                grouping_key = json.dumps([
                    evaluator(expr) for expr, _ in
                    request.session['starfind']['grouping_expressions']
                ])
                if grouping_key not in grouping:
                    grouping[grouping_key] = []
                grouping[grouping_key].append(
                    (
                        image.id,
                        channel,
                        processing.get_step_input(image, channel, 'calibrated')
                    )
                )
            request.session['starfind']['pending'][imtype.name] = sorted(
                grouping.items(),
                key=lambda item: len(item[1]),
                reverse=True,
            )

def select_starfind_batch(request, refresh=False):
    """Allow the user to select batch of images to tune star finding for."""

    if refresh:
        request.session.flush()
        return redirect('/processing/select_starfind_batch')

    _get_pending(request)

    if 'fits_display' in request.session:
        del request.session['fits_display']

    context = {
        'batches': []
    }
    for (
            imtype_name,
            imtype_batches
    ) in request.session['starfind']['pending'].items():
        batch_info = []
        for grouping_values, batch in imtype_batches:
            batch_info.append(
                (
                    ', '.join(
                        expr[1].format(value=value)
                        for value, expr in zip(
                            json.loads(grouping_values),
                            request.session['starfind']['grouping_expressions']
                        )
                        if not isinstance(value, bool) or value
                    ),
                    len(batch),
                )
            )

        context['batches'].append((imtype_name, batch_info))
    return render(
        request,
        'processing/select_starfind_batch.html',
        context
    )


def tune_starfind(request, imtype, batch_index):
    """Provide view allowing user to tune starfinding for given image batch."""

    batch = request.session['starfind']['pending'][imtype][batch_index]
    update_fits_display(request)
    image_index = request.session['fits_display']['image_index']
    context = encode_fits(
        batch[1][image_index][2],
        request.session['fits_display']['range'],
        request.session['fits_display']['transform']
    )
    context['num_images'] = len(batch[1])
    context.update(request.session['fits_display'])
    context['image_index1'] = context['image_index'] + 1
    context['fits_fname'] = batch[1][image_index][2]

    return render(
        request,
        'processing/tune_starfind.html',
        context
    )


def find_stars(request, fits_fname):
    """Run source extraction and respond with the results."""

    request_data = json.loads(request.body.decode())
    print(f'Request data: {request_data!r}')

    stars = SourceFinder(
        tool=request_data['srcfind-tool'],
        brightness_threshold=float(request_data['brightness-threshold']),
        filter_sources=request_data['filter-sources'],
        max_sources=int(request_data['max-sources'] or '0'),
        allow_overwrite=True,
        allow_dir_creation=True
    )(
        fits_fname
    )
    print('Found stars:\n' + repr(stars))

    return JsonResponse({'stars': [{'x': s['x'], 'y': s['y']} for s in stars]})
