"""
The browser interface's root URL configuration.

One prefix per application, each application routing its own URLs from
there, so that what a page's address begins with says which application
answers it.  ``home`` takes the root, being where a session starts and
what every other page's navigation leads back to.
"""

from django.contrib import admin
from django.urls import include, path

_apps = "autowisp.browser_interface"

urlpatterns = [
    path("", include(f"{_apps}.home.urls")),
    path("configuration/", include(f"{_apps}.configuration.urls")),
    path("processing/", include(f"{_apps}.processing.urls")),
    path("diagnostics/", include(f"{_apps}.diagnostics.urls")),
    path("results/", include(f"{_apps}.results.urls")),
    path("admin/", admin.site.urls),
]
