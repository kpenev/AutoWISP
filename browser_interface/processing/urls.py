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
    path('select_photref_target',
         views.select_photref_target,
         name='select_photref_target'),
    path('select_photref_target/recalc',
         views.select_photref_target,
         {'recalc': True},
         name='select_photref_recalc'),
    path('select_photref_image/<int:target_index>',
         views.select_photref_image,
         name='select_photref_image'),
    path(
        '/'.join(['select_photref_image',
                  'recalc',
                  '<int:target_index>']),
        views.select_photref_image,
        {'recalculate': True},
        name='select_photref_image_recalc'
    ),
    path(
        'record_photref_selection/<int:target_index>/<int:image_index>',
        views.record_photref_selection,
        name='record_photref_selection'
    ),
    path(
        'update_fits_display',
        views.update_fits_display,
        name='update_fits_display'
    ),
    path(
        'select_starfind_batch',
        views.select_starfind_batch,
        name='select_starfind_batch'
    ),
    path(
        'select_starfind_batch/refresh',
        views.select_starfind_batch,
        {'refresh': True},
        name='select_starfind_batch_refresh'
    ),
    path(
        'tune_starfind/<slug:imtype>/<int:batch_index>',
        views.tune_starfind,
        name='tune_starfind'
    ),
    path(
        'find_stars/<path:fits_fname>',
        views.find_stars,
        name='find_stars'
    ),
    path(
        'project_catalog/<path:fits_fname>',
        views.project_catalog,
        name='project_catalog'
    ),
    path(
        'save_starfind_config/<slug:imtype>/<int:batch_index>',
        views.save_starfind_config,
        name='save_starfind_config'
    ),
    path(
        'diagnostics/<slug:step>/<slug:imtype>',
        views.display_detrending_diagnostics,
        name='diagnostics'
    ),
    path(
        'diagnostics/<slug:step>/<slug:imtype>/<slug:master_ids>',
        views.display_detrending_diagnostics,
        name='diagnostics'
    ),
    path(
        'display_detrending_diagnostics',
        views.display_detrending_diagnostics,
        name='display_detrending_diagnostics'
    ),
    path(
        'refresh_detrending_diagnostics',
        views.refresh_detrending_diagnostics,
        name='refresh_diagnostics'
    ),
    path(
        'update_detrending_diagnostics_plot',
        views.update_detrending_diagnostics_plot,
        name='update_diagnostics_plot'
    ),
    path(
        'download_detrending_diagnostics_plot',
        views.download_detrending_diagnostics_plot,
        name='download_diagnostics_plot'
    )
]
