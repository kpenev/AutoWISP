"""Models describing the projects known to the browser interface."""

from django.db import models

from core.models import BuiModelBase


class Project(BuiModelBase):
    """Model to represent a project."""

    name = models.CharField(
        max_length=100,
        help_text="Enter the project name",
    )
    description = models.TextField(
        blank=True,
        help_text="Enter a brief description of the project",
    )
    path = models.TextField(
        help_text="The project root directory",
    )

    def __str__(self):
        return str(self.name)
