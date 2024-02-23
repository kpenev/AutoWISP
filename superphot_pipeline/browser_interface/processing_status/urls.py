from django.urls import path

from . import views

app_name = 'processing_status'

urlpatterns = [
    path("", views.progress, name="progress"),
]
