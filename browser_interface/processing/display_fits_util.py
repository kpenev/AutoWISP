"""Utilities used by views that allow users to interact with FITS images."""

import json

from django.http import JsonResponse

def update_fits_display(request):
    """Updatet the displayed FITS image per user interaction."""

    if 'fits_display' not in request.session:
        request.session['fits_display'] = {
            'image_index': 0,
            'range': 'zscale',
            'transform': None,
        }

    if request.method != 'POST':
        return None

    request_data = json.loads(request.body.decode())
    print(f'Request data: {request_data!r}')
    change = request_data.pop('change')
    if change == 'next image':
        request.session['fits_display']['image_index'] += 1
    elif change == 'previous image':
        request.session['fits_display']['image_index'] -= 1
    else:
        assert len(change) == 1
        for param, value in change.items():
            request.session['fits_display'][param] = (
                int(value) - 1 if param == 'image_index'
                else value
            )
    request.session['view_config'] = json.dumps(request_data)
    request.session.modified = True

    return JsonResponse({'ok': True})
