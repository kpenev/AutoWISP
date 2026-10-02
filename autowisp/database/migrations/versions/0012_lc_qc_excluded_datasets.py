"""Add the lightcurve datasets recording what EPD and TFA left out of a fit.

EPD and TFA can be given a list of observations to leave out of the fit
while still correcting them, and record with each detrended photometry the
rule that produced the list, how many points it removed, and which. The
layout of lightcurves is kept in the project database, so projects created
before that need the three datasets added next to every detrended
magnitude they define.

They go into the structure versions that are already there. Nothing else
about the layout changes and a lightcurve need not contain every dataset
its structure lists, so existing lightcurves stay valid under the version
they were written with.

This revision changes rows, not the schema. The definitions below are a
copy of what ``_get_detrended_datasets`` creates for a new project, made so
that this revision keeps doing what it did if that function changes.
"""

import alembic
import sqlalchemy

# revision identifiers, used by Alembic.
revision = "0012_lc_qc_excluded_datasets"
down_revision = "0011_diagnostic_expression"
branch_labels = None
depends_on = None

MAGNITUDE_PATH_TAIL = "Magnitude"


def _reflect(connection, table):
    """Return the named table as it is in the database."""

    return sqlalchemy.Table(
        table, sqlalchemy.MetaData(), autoload_with=connection
    )


def _get_detrended_magnitudes(connection, datasets):
    """Return the lightcurve datasets holding EPD or TFA corrected magnitudes.

    Returns:
        [(int, str, str)]:
            The structure version id, pipeline key and path of each.
    """

    versions = _reflect(connection, "hdf5_structure_versions")
    products = _reflect(connection, "hdf5_products")
    return connection.execute(
        sqlalchemy.select(
            datasets.c.hdf5_structure_version_id,
            datasets.c.pipeline_key,
            datasets.c.abspath,
        )
        .join(versions, datasets.c.hdf5_structure_version_id == versions.c.id)
        .join(products, versions.c.hdf5_product_id == products.c.id)
        .where(
            products.c.pipeline_key == "light_curve",
            sqlalchemy.or_(
                datasets.c.pipeline_key.endswith(
                    ".epd.magnitude", autoescape=True
                ),
                datasets.c.pipeline_key.endswith(
                    ".tfa.magnitude", autoescape=True
                ),
            ),
        )
    ).all()


def _get_exclusion_datasets(magnitude_key, magnitude_path):
    """Return the datasets to add next to one detrended magnitude.

    Args:
        magnitude_key(str):    The pipeline key of the detrended magnitude,
            e.g. ``apphot.epd.magnitude``.

        magnitude_path(str):    Its path within the lightcurve.

    Returns:
        [dict]:
            The values of the columns to set for each dataset, except the
            structure version.
    """

    key_prefix = magnitude_key.rsplit(".", 1)[0]
    mode = key_prefix.rsplit(".", 1)[1]
    assert magnitude_path.endswith(MAGNITUDE_PATH_TAIL)
    root_path = magnitude_path[: -len(MAGNITUDE_PATH_TAIL)]
    return [
        {
            "pipeline_key": key_prefix + ".cfg.exclusion_rule",
            "abspath": root_path + "FitProperties/ExclusionRule",
            "dtype": "numpy.bytes_",
            "compression": "gzip",
            "compression_options": "9",
            "description": (
                "The rule that decided which observations to leave out of "
                f"the {mode} fit, empty if they were not decided by a rule."
            ),
        },
        {
            "pipeline_key": key_prefix + ".num_qc_excluded_points",
            "abspath": root_path + "FitProperties/NumberQCExcludedPoints",
            "dtype": "numpy.uint",
            "compression": "gzip",
            "compression_options": "9",
            "description": (
                "The number of corrected points that were left out of the "
                f"{mode} fit."
            ),
        },
        {
            "pipeline_key": key_prefix + ".qc_excluded",
            "abspath": root_path + "QCExcluded",
            "dtype": "numpy.bool_",
            "compression": "gzip",
            "compression_options": "9",
            "description": (
                f"Was each point left out of the {mode} fit, while still "
                "being corrected?"
            ),
        },
    ]


def upgrade():
    """Add the datasets that are not there yet.

    One can be without this revision having run: a re-run after an
    interrupted upgrade, or a project whose structure was filled by newer
    code than its recorded revision.
    """

    connection = alembic.op.get_bind()
    datasets = _reflect(connection, "hdf5_datasets")
    for version_id, magnitude_key, magnitude_path in _get_detrended_magnitudes(
        connection, datasets
    ):
        present = set(
            connection.scalars(
                sqlalchemy.select(datasets.c.pipeline_key).where(
                    datasets.c.hdf5_structure_version_id == version_id
                )
            )
        )
        for new_dataset in _get_exclusion_datasets(
            magnitude_key, magnitude_path
        ):
            if new_dataset["pipeline_key"] not in present:
                connection.execute(
                    datasets.insert().values(
                        hdf5_structure_version_id=version_id, **new_dataset
                    )
                )


def downgrade():
    """Remove the datasets from the structure, if they are there.

    Lightcurves that already contain them keep the data, which nothing
    refers to any more.
    """

    connection = alembic.op.get_bind()
    datasets = _reflect(connection, "hdf5_datasets")
    for version_id, magnitude_key, magnitude_path in _get_detrended_magnitudes(
        connection, datasets
    ):
        connection.execute(
            datasets.delete().where(
                datasets.c.hdf5_structure_version_id == version_id,
                datasets.c.pipeline_key.in_(
                    [
                        new_dataset["pipeline_key"]
                        for new_dataset in _get_exclusion_datasets(
                            magnitude_key, magnitude_path
                        )
                    ]
                ),
            )
        )
