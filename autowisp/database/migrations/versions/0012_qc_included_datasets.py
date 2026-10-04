"""Add the datasets and attributes recording the quality cuts of the fits.

EPD and TFA can be given a list of observations to leave out of the fit
while still correcting them, and record with each detrended photometry the
rule that produced the list, how many points it removed, and which points
the fit was derived from. Magnitude fitting records in each DR file, per
photometry, whether the image passes its quality cut, and lightcurves carry
that flag per point. The layouts of DR files and lightcurves are kept in
the project database, so projects created before that need the rows added:
the three EPD/TFA datasets next to every detrended magnitude, and the
magnitude fitting flag next to every photometry's magnitude fitting
results.

They go into the structure versions that are already there. Nothing else
about the layout changes and a file need not contain every element its
structure lists, so existing files stay valid under the version they were
written with.

This revision changes rows, not the schema. The definitions below are a
copy of what ``_get_detrended_datasets``, ``_get_magfit_attributes`` and
``_get_data_reduction_attribute_datasets`` create for a new project, made
so that this revision keeps doing what it did if those functions change.
"""

import alembic
import sqlalchemy

# revision identifiers, used by Alembic.
revision = "0012_qc_included_datasets"
down_revision = "0011_diagnostic_expression"
branch_labels = None
depends_on = None

MAGNITUDE_PATH_TAIL = "Magnitude"


def _reflect(connection, table):
    """Return the named table as it is in the database."""

    return sqlalchemy.Table(
        table, sqlalchemy.MetaData(), autoload_with=connection
    )


def _get_anchors(connection, table, product, condition, path_column):
    """Return the rows of a product's structure that new rows go next to.

    Args:
        table(sqlalchemy.Table):    ``hdf5_datasets`` or ``hdf5_attributes``.

        product(str):    The pipeline key of the HDF5 product.

        condition:    Selects the rows from ``table``.

        path_column(str):    The column of ``table`` saying where in the
            file the row's element is.

    Returns:
        [(int, str, str)]:
            The structure version id, pipeline key and path of each.
    """

    versions = _reflect(connection, "hdf5_structure_versions")
    products = _reflect(connection, "hdf5_products")
    return connection.execute(
        sqlalchemy.select(
            table.c.hdf5_structure_version_id,
            table.c.pipeline_key,
            table.c[path_column],
        )
        .join(versions, table.c.hdf5_structure_version_id == versions.c.id)
        .join(products, versions.c.hdf5_product_id == products.c.id)
        .where(products.c.pipeline_key == product, condition)
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
            "pipeline_key": key_prefix + ".qc_included",
            "abspath": root_path + "QCIncluded",
            "dtype": "numpy.bool_",
            "compression": "gzip",
            "compression_options": "9",
            "description": (
                f"Was each point among those the {mode} fit was derived "
                "from? Points the exclusion list left out are still "
                "corrected; points failing the points filter are not."
            ),
        },
    ]


def _get_magfit_flag(anchor_key, anchor_path, in_lightcurve):
    """Return the magnitude fitting flag to add for one photometry.

    Args:
        anchor_key(str):    The pipeline key of the element the flag goes
            next to: the photometry's ``magfit.fit_residual`` dataset in
            lightcurves, its ``magfit.cfg.single_photref`` attribute in DR
            files.

        anchor_path(str):    The path of that dataset, or the parent of that
            attribute.

        in_lightcurve(bool):    Is the flag a lightcurve dataset, rather than
            a DR attribute?

    Returns:
        dict:
            The values of the columns to set, except the structure version.
    """

    result = {
        "pipeline_key": anchor_key.split(".", 1)[0] + ".magfit.qc_included",
        "dtype": "numpy.bool_",
        "description": "Does the image pass the magnitude fitting quality "
        "cut?",
    }
    if in_lightcurve:
        result.update(
            abspath=anchor_path.rsplit("/", 1)[0] + "/QCIncluded",
            compression="gzip",
            compression_options="9",
        )
    else:
        result.update(parent=anchor_path, name="QCIncluded")
    return result


def _get_additions(connection):
    """Return what to add to each structure version.

    Returns:
        [(sqlalchemy.Table, int, [dict])]:
            The table, the structure version id, and the rows to add to it
            as returned by _get_exclusion_datasets() or _get_magfit_flag().
    """

    datasets = _reflect(connection, "hdf5_datasets")
    attributes = _reflect(connection, "hdf5_attributes")
    result = [
        (datasets, version_id, _get_exclusion_datasets(key, path))
        for version_id, key, path in _get_anchors(
            connection,
            datasets,
            "light_curve",
            sqlalchemy.or_(
                datasets.c.pipeline_key.endswith(
                    ".epd.magnitude", autoescape=True
                ),
                datasets.c.pipeline_key.endswith(
                    ".tfa.magnitude", autoescape=True
                ),
            ),
            "abspath",
        )
    ]
    for table, product, anchor_tail, path_column in [
        (datasets, "light_curve", ".magfit.fit_residual", "abspath"),
        (attributes, "data_reduction", ".magfit.cfg.single_photref", "parent"),
    ]:
        result.extend(
            (
                table,
                version_id,
                [_get_magfit_flag(key, path, product == "light_curve")],
            )
            for version_id, key, path in _get_anchors(
                connection,
                table,
                product,
                table.c.pipeline_key.in_(
                    [
                        photometry + anchor_tail
                        for photometry in ("shapefit", "apphot")
                    ]
                ),
                path_column,
            )
        )
    return result


def upgrade():
    """Add the rows that are not there yet.

    One can be without this revision having run: a re-run after an
    interrupted upgrade, or a project whose structure was filled by newer
    code than its recorded revision.
    """

    connection = alembic.op.get_bind()
    for table, version_id, new_rows in _get_additions(connection):
        present = set(
            connection.scalars(
                sqlalchemy.select(table.c.pipeline_key).where(
                    table.c.hdf5_structure_version_id == version_id
                )
            )
        )
        for new_row in new_rows:
            if new_row["pipeline_key"] not in present:
                connection.execute(
                    table.insert().values(
                        hdf5_structure_version_id=version_id, **new_row
                    )
                )


def downgrade():
    """Remove the rows from the structures, if they are there.

    Files that already contain the elements keep them, though nothing
    refers to them any more.
    """

    connection = alembic.op.get_bind()
    for table, version_id, new_rows in _get_additions(connection):
        connection.execute(
            table.delete().where(
                table.c.hdf5_structure_version_id == version_id,
                table.c.pipeline_key.in_(
                    [new_row["pipeline_key"] for new_row in new_rows]
                ),
            )
        )
