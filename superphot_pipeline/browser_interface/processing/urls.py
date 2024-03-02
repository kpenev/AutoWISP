from django.urls import path

from . import views

app_name = 'processing'

urlpatterns = [
    path('', views.progress, name='progress'),
    path('add_raw_images/<slug:files_or_dir>/',
         views.add_raw_images,
         name='add_raw_images'),
    path('start_processing', views.start_processing, name='start_processing')
]
