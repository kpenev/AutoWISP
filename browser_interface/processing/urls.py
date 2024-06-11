"""Define the URL paths used by the processing BUI app."""
from django.urls import path

from . import views

app_name = 'processing'

urlpatterns = [
    path('', views.progress, name='progress'),
    path('start_processing', views.start_processing, name='start_processing'),
    path('select_raw_images/<path:dirname>/',
         views.SelectRawImages.as_view(),
         name='select_raw_images'),
    path('select_raw_images/',
         views.SelectRawImages.as_view(),
         name='select_raw_images'),
    path('review/<int:selected_processing_id>/<slug:what>/<slug:min_log_level>',
         views.review_single,
         name='review'),
    path('review/<int:selected_processing_id>/<slug:what>/<int:sub_process>/'
         '<slug:min_log_level>',
         views.review_single,
         name='review'),
    path('review/<int:selected_processing_id>/<slug:what>/<int:sub_process>',
         views.review_single,
         name='review'),
    path('review/<int:selected_processing_id>/<slug:min_log_level>',
         views.review,
         name='review'),
    path('select_photref',
         views.SelectPhotRef.as_view(),
         name='select_photref'),
    path('select_photref/<str:values_range>',
         views.SelectPhotRef.as_view(),
         name='select_photref'),
    path('select_photref/<str:values_range>/<slug:values_transform>',
         views.SelectPhotRef.as_view(),
         name='select_photref')
]
