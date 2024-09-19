"""Define the URL paths used by the processing BUI app."""
from django.urls import path

from . import views

app_name = 'results'

urlpatterns = [
    path('', views.display_lightcurve, name='results')
]
