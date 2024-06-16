"""Implement the view for selecting single photometric reference."""

from io import BytesIO, StringIO
from base64 import b64encode
from time import sleep

import numpy
from PIL import Image
from PIL.ImageTransform import AffineTransform
from django.shortcuts import render, redirect
import matplotlib
from matplotlib import colors, pyplot
from sqlalchemy import select
import pandas

from astropy.io import fits
from astropy.visualization import ZScaleInterval

from autowisp.database.processing import\
    ProcessingManager,\
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


def _log_transform(pixel_values, parameter=1000.0):
    """Perform the same log-transform as DS9."""

    return numpy.log(parameter * pixel_values + 1) / numpy.log(parameter)


def _pow_transform(pixel_values, parameter=1000.0):
    """Perform the same pow transfom as DS9."""

    return (numpy.power(parameter, pixel_values) - 1.0) / parameter


def _sqrt_transform(pixel_values):
    """Use square root of the pixel values as intensity."""

    return numpy.sqrt(pixel_values)


def _square_transform(pixel_values):
    """Use the square of the pixel values as intensity."""

    return numpy.square(pixel_values)


def _asinh_transform(pixel_values):
    """The asinh transform of DS9."""

    return numpy.arcsinh(10.0 * pixel_values) / 3.0


def _sinh_transform(pixel_values):
    """The sinh transform of DS9."""

    return numpy.sinh(3.0 * pixel_values) / 10.0


def _get_missing_photref(request):
    """Add all frame sets missing photometric reference to the session."""

    assert 'need_photref' not in request.session
    processing = ProcessingManager(dummy=True)
    #False positivie
    #pylint: disable=no-member
    with Session.begin() as db_session:
    #pylint: enable=no-member
        master_type_id = db_session.scalar(
            select(MasterType.id).filter_by(name='single_photref')
        )
        pending_photref = processing.get_pending(
            db_session,
            [entry for entry in get_processing_sequence(db_session)
             if entry[0].name == 'fit_magnitudes'],
        )
        astrom_step_id = db_session.scalar(
            select(Step.id).filter_by(name='solve_astrometry')
        )
        for (
                (step_id, image_type_id),
                pending_images
        ) in pending_photref.items():
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
                'master_values': {}
            }

            by_photref = processing.group_pending_by_conditions(
                pending_images,
                db_session,
                match_observing_session=False,
                step_id=step_id,
                masters_only=True
            )
            for target_ind, (by_master_values, master_values) in enumerate(
                by_photref
            ):
                if not processing.get_candidate_masters(
                    *by_master_values[0],
                    input_master_type,
                    db_session
                ):
                    config = processing.get_config(
                        matched_expressions=None,
                        image_id=by_master_values[0][0].id,
                        channel=by_master_values[0][1],
                        step_name='calculate_photref_merit'
                    )[0]
                    request.session[
                        'need_photref'
                    ][
                        'master_values'
                    ][
                        str(target_ind)
                    ] = (
                        master_values,
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
                            for image, channel in by_master_values
                        ]
                    )


def _get_merit_data(request, target_index):
    """Add to the session the merit information for selecting single ref."""

    target_index = str(target_index)
    if 'merit_info' not in request.session:
        request.session['merit_info'] = {}
    if target_index not in request.session['merit_info']:
        config, batch = (
            request.session['need_photref']['master_values'][target_index][1:]
        )
        request.session['merit_info'][target_index] = (
            calculate_photref_merit(
                [entry[1] for entry in batch],
                config
            ).sort_values(
                by='merit',
                ascending=False
            ).to_json()
        )


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
                pyplot.yscale('log')
                pyplot.suptitle('merit')
            else:
                quantile = merit_data['qnt_' + column].iloc[image_index]
                pyplot.suptitle(column + f' ({quantile} quantile)')
            pyplot.savefig(image_stream, format='svg')
            result.append(image_stream.getvalue())
            pyplot.clf()
    return result


def select_photref_image(request,
                         *,
                         target_index,
                         image_index=0,
                         values_range='zscale',
                         values_transform=None,
                         zoom=1.0):
    """Display the interface for reviewing canditate reference frames."""

    _get_merit_data(request, target_index)
    merit_data = pandas.read_json(
        request.session['merit_info'][str(target_index)]
    )
    print('Merit data:\n' + repr(merit_data))
    png_stream = BytesIO()
    with fits.open(
        request.session[
            'need_photref'
        ][
            'master_values'
        ][
            str(target_index)
        ][
            2
        ][
            #False positive
            #pylint:disable=no-member
            merit_data.index[image_index]
            #pylint:enable=no-member
        ][
            0
        ],
        'readonly'
    ) as frame:
        if values_range == 'zscale':
            limits = ZScaleInterval().get_limits(frame[1].data)
        elif values_range == 'minmax':
            limits = frame[1].data.min(), frame[1].data.max()
        else:
            limits = tuple(int(lim.strip())
                           for lim in values_range.split(','))
        pixel_values = colors.Normalize(
            *limits,
            True
        )(frame[1].data)
        if values_transform is not None and values_transform != 'None':
            transform_args = values_transform.split('-')
            transform = globals()['_' + transform_args.pop(0) + '_transform']
            transform_args = [float(arg) for arg in transform_args]
            pixel_values = transform(pixel_values, *transform_args)
        scaled_pixels = (
            pixel_values
            *
            255
        ).astype('uint8')
        image = Image.fromarray(scaled_pixels)
        apply_zoom = AffineTransform((1.0/zoom, 0, 0, 0, 1.0/zoom, 0.0))
        image.transform(
            size=(int(image.size[0] * zoom), int(image.size[1] * zoom)),
            method=apply_zoom
        ).save(
            png_stream,
            'png'
        )

    return render(
        request,
        'processing/select_photref_image.html',
        {
            'target_index': target_index,
            'image_index': image_index,
            'last_image': image_index < merit_data.shape[0] - 1,
            'image': b64encode(png_stream.getvalue()).decode('utf-8'),
            'histograms': _create_merit_histograms(merit_data, image_index),
            'range': values_range,
            'transform': values_transform,
            'transform_list': [
                entry[1:].split('_', 1)[0]
                for entry in globals()
                if(
                    entry[0] == '_'
                    and
                    entry.endswith('_transform')
                )
            ]
        }
    )


def select_photref_target(request, refresh=False):
    """Display view to select which of the missing photrefs to define."""

    if refresh:
        request.session.flush()
        return redirect('/processing/select_photref_target')
    request.session.set_expiry(0)
    if 'need_photref' not in request.session:
        _get_missing_photref(request)

    return render(
        request,
        'processing/select_photref_target.html',
        {
            'master_expressions': request.session[
                'need_photref'
            ][
                'master_expressions'
            ],
            'master_values': [
                request.session[
                    'need_photref'
                ][
                    'master_values'
                ][
                    str(target_ind)
                ][
                    0
                ]
                for target_ind in
                range(len(request.session['need_photref']['master_values']))
            ]
        }
    )
