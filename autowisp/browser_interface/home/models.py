from django.db import models


class Project(models.Model):
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
    created = models.DateTimeField(
        auto_now_add=True,
        help_text="The date and time the project was created",
    )

    def __str__(self):
        return self.name


# Create your models here.
