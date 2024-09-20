"""Define the URL paths used by the processing BUI app."""
from django.urls import path

from . import views

app_name = 'results'

urlpatterns = [
    path('', views.display_lightcurve, name='results'),
    path('update_lightcurve_figure',
         views.update_lightcurve_figure,
         name='update_lightcurve_figure'),
    path('download_lightcurve_figure',
         views.download_lightcurve_figure,
         name='download_lightcurve_figure'),
    path('clear_lightcurve_buffer',
         views.clear_lightcurve_buffer,
         name='clear_lightcurve_buffer')
]
