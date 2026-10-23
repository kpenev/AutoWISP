"""Store paths inside a project relative to its home, as ``{PROJHOME}/...``.

A project now holds no absolute path to anything inside it, so that moving or
copying its directory is all it takes to relocate it. This revision converts
what the database holds:

* the masters and raw images it records, where they lie inside the project;

* configuration values of path parameters; the log names named the home
  ``{project_home}``;

* the description of ``subpixmap``, whose default changed.

The files the project has produced hold such paths too, and this revision
converts them after the database: the calibrated images, the masters (and
the DR files a master photometric reference lists as fit, for every magfit
iteration), the DR files and the lightcurves (with the reference in their
points filters). They are found through the database as converted,
so this must run at exactly this revision, which is why it is not left to a
separate step. Every lightcurve is opened, so it can take a long time. A
file already converted is not changed again, so after an interruption, which
rolls back the database part, running it again only re-checks those.

A relative path, wherever it is found, is taken to be inside the project,
as the defaults the configured values replace are (``MASTERS/...``).

The data of a FITS file is copied byte for byte, never decoded: astropy
re-compresses a compressed image, changing its values, even when only the
header of the HDU changes.

The project home is given by ``migrate_project`` in
``config.attributes["project_home"]``. The parameter names and texts below,
how the products are found and where their paths are, are a copy, made so
that this revision keeps doing what it did if those change. The database is
read through reflected tables, and the files directly, never through the
classes that interpret them.

There is no going back: the products are not converted back, so a database
downgraded past this revision would not match them.
"""

import ast
import logging
import os
from pathlib import PurePath
import re
import tempfile

import alembic
from astropy.io import fits
from astropy.table import Table
import h5py
import numpy
import sqlalchemy

from autowisp.exceptions import DatabaseError
from autowisp.light_curves.collect_light_curves import DecodingStringFormatter

# revision identifiers, used by Alembic.
revision = "0018_projhome_paths"
down_revision = "0017_master_mask_description"
branch_labels = None
depends_on = None

_logger = logging.getLogger(__name__)

PROJHOME = "{PROJHOME}"

#: The keywords of FITS headers that name files.
_FILE_KEYWORDS = re.compile(r"M(BIAS|DARK|FLAT)FNM|ORIGF\d+")

#: Where products record the single photometric reference they used: an
#: attribute of DR files and a dataset of lightcurves.
_SPHOTREF_NAME = "SinglePhotometricReference"

#: The datasets of lightcurves holding the expression that selected the points
#: EPD and TFA fit, to which the engine adds the comparison below.
_FILTER_NAMES = ("Filter", "PointsFilterExpression")

#: The comparison selecting the points of one single photometric reference,
#: which is named by the ``repr`` of its bytes.
_SPHOTREF_MATCH = re.compile(
    r"(sphotref == )(b'(?:[^'\\]|\\.)*'|b\"(?:[^\"\\]|\\.)*\")"
)

#: The copy of the frame header in a DR file.
_DR_HEADER_PATH = "/FITSHeader"

#: The table of a master photometric reference listing the DR files fit.
_POPULATION_TABLE = "INMASTER"

#: Parameters whose values are paths, by how their values are converted.
PATH_PARAMETERS = {
    **{
        name: "project"
        for name in (
            "calibrated-fname",
            "stacked-master-fname",
            "high-flat-master-fname",
            "low-flat-master-fname",
            "data-reduction-fname",
            "astrometry-catalog",
            "astrometry-catalog-source-list",
            "photometry-catalog",
            "photometry-catalog-source-list",
            "magfit-catalog",
            "magfit-catalog-source-list",
            "master-photref-fname-format",
            "magfit-stat-fname-format",
            "lc-catalog",
            "lc-catalog-source-list",
            "lightcurve-catalog-fname",
            "lc-fname",
            "detrending-catalog",
            "epd-statistics-fname",
            "tfa-statistics-fname",
        )
    },
    "logging-fname": "log",
    "std-out-err-fname": "log",
    "master-mask": "channel",
    "anet-indices": "external",
    "subpixmap": "external",
    "tempstore-dir": "external",
}

SUBPIXMAP_DESCRIPTION = {
    "old": (
        "The sub-pixel sensitivity map to assume. If not specified uniform "
        "sensitivy is assumed. This is especially important if processing "
        "images from color cameras. For a standard Bayer array, the built-in "
        "map called ``dslr_subpixmap.fits`` can be used."
    ),
    "new": (
        "The sub-pixel sensitivity map to assume. This is especially "
        "important if processing images from color cameras. If not "
        "specified, images that are one channel of a 2x2 pattern (a "
        "standard Bayer array) use the built-in map "
        "``dslr_subpixmap.fits``, and other images assume uniform "
        "sensitivity."
    ),
}


def _get_stored_path(fname, project_home):
    """
    Return the path in ``{PROJHOME}`` form, or None if it is not inside.

    A relative path is taken to be inside, whether or not the file is still
    there: left relative, it would be read relative to whichever directory
    the code reading it runs in, and as a string it would never equal the
    stored form of the same file.

    The part after ``{PROJHOME}`` uses ``/`` on every system, as the stored
    form does everywhere else: stored paths are compared as strings.

    Args:
        fname(str):    The path as recorded, possibly a template.

        project_home(str):    The absolute path of the project home.

    Returns:
        str or None:    ``{PROJHOME}/<relative>``, or None for a path outside
        the project.
    """

    if fname.startswith(PROJHOME):
        return fname
    path = PurePath(fname)
    if not path.is_absolute():
        # On Windows, a path is absolute only with a drive. Joined to the
        # home, one rooted without a drive keeps only the drive of the home,
        # and need not be inside it.
        path = PurePath(project_home) / path
    if not path.is_relative_to(project_home):
        return None
    return PROJHOME + "/" + path.relative_to(project_home).as_posix()


def _convert_files(connection, table_name, column, project_home):
    """Convert the paths to files inside the project held by a column."""

    table = sqlalchemy.Table(
        table_name, sqlalchemy.MetaData(), autoload_with=connection
    )
    for row_id, fname in connection.execute(
        sqlalchemy.select(table.c.id, table.c[column])
    ).all():
        converted = _get_stored_path(fname, project_home)
        if converted is not None and converted != fname:
            connection.execute(
                table.update()
                .where(table.c.id == row_id)
                .values({column: converted})
            )


def _convert_path_entry(entry, kind, project_home):
    """Return one path of a configured value in ``{PROJHOME}`` form."""

    if kind == "channel":
        if ":" not in entry:
            _logger.warning(
                "%r is not of the form <channel>:<path>; left as it is.",
                entry,
            )
            return entry
        channel, entry = entry.split(":", 1)
        return (
            channel + ":" + _convert_path_entry(entry, "project", project_home)
        )
    if kind == "external":
        if not PurePath(entry).is_absolute():
            _logger.warning(
                "%r is a relative path, which must now be absolute; left as "
                "it is.",
                entry,
            )
        return entry
    if kind == "log":
        entry = entry.replace("{project_home}", PROJHOME)
    return _get_stored_path(entry, project_home) or entry


def _convert_value(value, kind, project_home):
    """Return a configured value of a path parameter in ``{PROJHOME}`` form."""

    if value.startswith("[") and value.endswith("]"):
        return (
            "["
            + ", ".join(
                _convert_path_entry(entry.strip(), kind, project_home)
                for entry in value[1:-1].split(",")
            )
            + "]"
        )
    return _convert_path_entry(value, kind, project_home)


def _convert_configuration(connection, project_home):
    """Convert the configured values of path parameters."""

    metadata = sqlalchemy.MetaData()
    parameter = sqlalchemy.Table(
        "parameter", metadata, autoload_with=connection
    )
    configuration = sqlalchemy.Table(
        "configuration", metadata, autoload_with=connection
    )
    for row_id, name, value in connection.execute(
        sqlalchemy.select(
            configuration.c.id, parameter.c.name, configuration.c.value
        )
        .join(parameter, configuration.c.parameter_id == parameter.c.id)
        .where(parameter.c.name.in_(PATH_PARAMETERS))
    ).all():
        if not value:
            continue
        converted = _convert_value(value, PATH_PARAMETERS[name], project_home)
        if converted != value:
            connection.execute(
                configuration.update()
                .where(configuration.c.id == row_id)
                .values(value=converted)
            )


def _resolve(stored_fname, project_home):
    """Return the file to open for a path in ``{PROJHOME}`` form."""

    return stored_fname.format(PROJHOME=project_home)


def _convert_header(header, project_home):
    """
    Convert a FITS header in place.

    Args:
        header(fits.Header):    The header to convert.

        project_home(str):    The absolute path of the project home.

    Returns:
        bool:    Was the header changed?
    """

    changed = False
    if "PROJHOME" in header:
        del header["PROJHOME"]
        changed = True
    for keyword in list(header.keys()):
        if not _FILE_KEYWORDS.fullmatch(keyword):
            continue
        converted = _get_stored_path(header[keyword], project_home)
        if converted is not None and converted != header[keyword]:
            header[keyword] = converted
            changed = True
    if changed and "CHECKSUM" in header:
        # It covers the header, so the change invalidates it; recomputing
        # it is cfitsio's business. DATASUM covers only the data, which is
        # not touched.
        del header["CHECKSUM"]
    return changed


def _migrate_fits(fname, project_home):
    """
    Convert the headers of a FITS file, copying its data byte for byte.

    The data is never decoded: astropy re-compresses a compressed image,
    changing its values, even when only the header of the HDU changes.

    Returns:
        bool:    Was the file changed?
    """

    with fits.open(fname, mode="readonly") as hdu_list:
        layout = [
            (info["hdrLoc"], info["datLoc"], info["datLoc"] + info["datSpan"])
            for info in (
                hdu_list.fileinfo(index) for index in range(len(hdu_list))
            )
        ]
    with open(fname, "rb") as fits_file:
        content = fits_file.read()

    pieces = []
    changed = False
    for header_start, data_start, data_end in layout:
        old_header = content[header_start:data_start]
        header = fits.Header.fromstring(old_header.decode("ascii"))
        if _convert_header(header, project_home):
            changed = True
            new_header = header.tostring(padding=True).encode("ascii")
            # Fill the blocks the header had, so that a header that shrank
            # can be written in place.
            while len(new_header) < len(old_header):
                header.append(fits.Card(), useblanks=False, end=True)
                new_header = header.tostring(padding=True).encode("ascii")
            pieces.append(new_header)
        else:
            pieces.append(old_header)
        pieces.append(content[data_start:data_end])
    if not changed:
        return False

    if sum(map(len, pieces)) == len(content):
        with open(fname, "r+b") as fits_file:
            position = 0
            for piece in pieces:
                if piece != content[position : position + len(piece)]:
                    fits_file.seek(position)
                    fits_file.write(piece)
                position += len(piece)
    else:
        # A header outgrew its blocks, moving everything after it.
        with tempfile.NamedTemporaryFile(
            dir=os.path.dirname(fname), delete=False
        ) as rewritten:
            for piece in pieces:
                rewritten.write(piece)
        os.replace(rewritten.name, fname)
    return True


def _migrate_population(fname, project_home):
    """
    Convert the DR files a master photometric reference lists as fit.

    The table is replaced, its column of names being as wide as the longest.
    Rewriting the file with astropy changes no value: unlike the images,
    a master photometric reference holds only binary tables.

    Returns:
        bool:    Was the file changed?
    """

    # Read into memory: replacing the table resizes the file, which astropy
    # does by removing it, and Windows refuses that while it is mapped.
    with fits.open(fname, mode="update", memmap=False) as hdu_list:
        if _POPULATION_TABLE not in hdu_list:
            return False
        population = hdu_list[_POPULATION_TABLE].data
        # A list: see _migrate_lc().
        old = list(population["dr_fname"])
        new = [_convert_text(value, project_home) for value in old]
        if all(converted is value for converted, value in zip(new, old)):
            return False
        hdu_list[hdu_list.index_of(_POPULATION_TABLE)] = (
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(
                        name="dr_fname",
                        format=f"{max(map(len, new))}A",
                        array=numpy.array(new, dtype=str),
                    ),
                    fits.Column(
                        name="included",
                        format="L",
                        array=population["included"],
                    ),
                ],
                name=_POPULATION_TABLE,
            )
        )
    return True


def _migrate_master(fname, project_home):
    """Convert a FITS master: its headers, and any population it lists."""

    headers_changed = _migrate_fits(fname, project_home)
    return _migrate_population(fname, project_home) or headers_changed


def _replace_dataset(group, name, data):
    """Replace a dataset by one holding ``data``, keeping its storage."""

    old = group[name]
    options = {
        option: getattr(old, option)
        for option in ("compression", "compression_opts", "shuffle")
        if getattr(old, option) is not None
    }
    attributes = dict(old.attrs)
    del group[name]
    new = group.create_dataset(name, data=data, **options)
    for attribute, value in attributes.items():
        new.attrs[attribute] = value


def _convert_text(value, project_home):
    """Return a recorded path as stored, keeping bytes as bytes."""

    as_bytes = isinstance(value, (bytes, numpy.bytes_))
    text = value.decode() if as_bytes else str(value)
    converted = _get_stored_path(text, project_home)
    if converted is None or converted == text:
        return value
    return converted.encode() if as_bytes else converted


def _convert_filter(value, project_home):
    """Return a points filter with its reference as stored, keeping the type."""

    as_bytes = isinstance(value, (bytes, numpy.bytes_))
    text = value.decode() if as_bytes else str(value)

    def convert_match(match):
        """Return the comparison with the reference converted."""

        # The literal is a repr, which doubles a Windows path's backslashes.
        old = ast.literal_eval(match[2])
        new = _convert_text(old, project_home)
        return match[0] if new is old else match[1] + repr(new)

    converted = _SPHOTREF_MATCH.sub(convert_match, text)
    if converted == text:
        return value
    return converted.encode() if as_bytes else converted


def _migrate_dr(fname, project_home):
    """Convert a DR file. Returns whether it was changed."""

    with h5py.File(fname, "r+") as dr_file:
        changed = False

        def convert_sphotref_attribute(_, obj):
            """Convert the attribute of each object of the file in turn."""

            nonlocal changed
            if _SPHOTREF_NAME not in obj.attrs:
                return
            old = obj.attrs[_SPHOTREF_NAME]
            new = _convert_text(old, project_home)
            if new is old:
                return
            obj.attrs[_SPHOTREF_NAME] = new

            changed = True

        # visititems() does not visit the root itself.
        convert_sphotref_attribute("/", dr_file)

        # Changing an attribute's value does not change what is visited.
        dr_file.visititems(convert_sphotref_attribute)

        if _DR_HEADER_PATH in dr_file:
            header = fits.Header.fromstring(
                dr_file[_DR_HEADER_PATH][()].tobytes().decode("ascii")
            )
            if _convert_header(header, project_home):
                _replace_dataset(
                    dr_file,
                    _DR_HEADER_PATH,
                    numpy.frombuffer(
                        b"".join(
                            card.image.encode("ascii") for card in header.cards
                        ),
                        dtype="i1",
                    ),
                )
                changed = True
    return changed


def _migrate_lc(fname, project_home):
    """Convert a lightcurve. Returns whether it was changed."""

    convert_by_name = {
        _SPHOTREF_NAME: _convert_text,
        **{name: _convert_filter for name in _FILTER_NAMES},
    }
    changed = False
    with h5py.File(fname, "r+") as light_curve:
        datasets = []
        light_curve.visititems(
            lambda name, obj: (
                datasets.append(obj)
                if isinstance(obj, h5py.Dataset)
                and name.rsplit("/", 1)[-1] in convert_by_name
                else None
            )
        )
        for dataset in datasets:
            convert = convert_by_name[dataset.name.rsplit("/", 1)[-1]]
            # A list, since iterating an array creates new objects each time,
            # and converting an unchanged value returns the same one.
            old = list(dataset[()])
            new = [convert(value, project_home) for value in old]
            if all(converted is value for converted, value in zip(new, old)):
                continue
            changed = True
            fixed_length = dataset.dtype.kind == "S"
            if fixed_length and max(map(len, new)) > dataset.dtype.itemsize:
                _replace_dataset(
                    dataset.parent,
                    dataset.name.rsplit("/", 1)[-1],
                    numpy.array(new, dtype=bytes),
                )
            else:
                for index, (converted, value) in enumerate(zip(new, old)):
                    if converted is not value:
                        dataset[index] = converted
    return changed


def _get_table(connection, name):
    """Return the table of the given name, as the database has it."""

    return sqlalchemy.Table(
        name, sqlalchemy.MetaData(), autoload_with=connection
    )


def _get_configured_values(connection, parameter_name):
    """Return every value a parameter can take under any condition."""

    parameter = _get_table(connection, "parameter")
    configuration = _get_table(connection, "configuration")
    return set(
        connection.execute(
            sqlalchemy.select(configuration.c.value)
            .join(parameter, configuration.c.parameter_id == parameter.c.id)
            .where(
                parameter.c.name == parameter_name,
                configuration.c.value.is_not(None),
            )
        ).scalars()
    )


def _find_masters(connection, project_home, master_type=None):
    """Return the files of the project's masters, optionally of one type."""

    master_file = _get_table(connection, "master_file")
    master_type_table = _get_table(connection, "master_type")
    query = sqlalchemy.select(master_file.c.filename).join(
        master_type_table, master_file.c.type_id == master_type_table.c.id
    )
    if master_type is not None:
        query = query.where(master_type_table.c.name == master_type)
    return [
        _resolve(fname, project_home)
        for fname in connection.execute(query).scalars()
    ]


def _read_raw_header(raw_fname):
    """Return the header of the first HDU of a raw frame holding an image."""

    with fits.open(raw_fname, mode="readonly") as raw_image:
        for hdu in raw_image:
            if hdu.header["NAXIS"] != 0:
                return hdu.header.copy()
    raise OSError(f"No image in {raw_fname!r}.")


def _find_image_products(connection, project_home, parameter_name):
    """
    Return the products a template names for each image and channel processed.

    Every value the template is configured with is tried, since which one
    applies depends on conditions, and the names that exist are returned.
    """

    image = _get_table(connection, "image")
    processed = _get_table(connection, "processed_images")
    templates = _get_configured_values(connection, parameter_name)
    result = set()
    for raw_fname, channels in _group_channels(
        connection.execute(
            sqlalchemy.select(image.c.raw_fname, processed.c.channel)
            .join(processed, processed.c.image_id == image.c.id)
            .distinct()
        ).all()
    ):
        raw_fname = _resolve(raw_fname, project_home)
        try:
            header = _read_raw_header(raw_fname)
        except OSError as error:
            _logger.warning(
                "Cannot read %r (%s), so the products made from it are not "
                "converted.",
                raw_fname,
                error,
            )
            continue
        base_fname = os.path.basename(raw_fname)
        for extension in (".fz", ".fits"):
            if base_fname.endswith(extension):
                base_fname = base_fname[: -len(extension)]
        for channel in channels:
            substitutions = dict(
                header,
                PROJHOME=project_home,
                RAWFNAME=base_fname,
                CLRCHNL=channel,
            )
            for template in templates:
                try:
                    fname = template.format_map(substitutions)
                except (KeyError, IndexError, ValueError):
                    continue
                if os.path.exists(fname):
                    result.add(fname)
    return sorted(result)


def _group_channels(rows):
    """Group ``(raw_fname, channel)`` rows by the raw image."""

    grouped = {}
    for raw_fname, channel in rows:
        grouped.setdefault(raw_fname, set()).add(channel)
    return sorted(grouped.items())


def _find_lightcurves(connection, project_home):
    """Return the lightcurves the project's lightcurve catalogues list."""

    templates = _get_configured_values(connection, "lc-fname")
    formatter = DecodingStringFormatter()
    result = set()
    for catalog_fname in _find_masters(
        connection, project_home, "lightcurve_catalog"
    ):
        if not os.path.exists(catalog_fname):
            _logger.warning(
                "The lightcurve catalogue %r is missing, so the lightcurves it "
                "lists are not converted.",
                catalog_fname,
            )
            continue
        for source_id in Table.read(catalog_fname)["source_id"]:
            for template in templates:
                fname = formatter.format(
                    template,
                    *numpy.atleast_1d(source_id),
                    PROJHOME=project_home,
                )
                if os.path.exists(fname):
                    result.add(fname)
    return sorted(result)


def _find_fits_masters(connection, project_home):
    """Return the masters that are FITS files and exist."""

    result = []
    for fname in _find_masters(connection, project_home):
        if not os.path.exists(fname):
            continue
        with open(fname, "rb") as master:
            if master.read(6) == b"SIMPLE":
                result.append(fname)
    return result


def _read_dr_header(dr_fname):
    """Return the copy of the frame header a DR file holds."""

    with h5py.File(dr_fname, "r") as dr_file:
        return fits.Header.fromstring(
            dr_file[_DR_HEADER_PATH][()].tobytes().decode("ascii")
        )


def _find_photref_iterations(connection, project_home):
    """
    Return the master photometric reference of every magfit iteration.

    The database records only the last iteration's. Each is named as magnitude
    fitting names it: the template filled with the single photometric
    reference's frame header and the iteration, counted from 0. Every value
    the template is configured with is tried, since which one applies depends
    on conditions, and the iterations up to the first missing one are kept.
    """

    templates = _get_configured_values(
        connection, "master-photref-fname-format"
    )
    result = set()
    for sphotref_fname in _find_masters(
        connection, project_home, "single_photref"
    ):
        try:
            header = _read_dr_header(sphotref_fname)
        except (OSError, KeyError) as error:
            _logger.warning(
                "Cannot read the frame header of %r (%s), so the master "
                "photometric references made from it are not converted.",
                sphotref_fname,
                error,
            )
            continue
        for template in templates:
            iteration_fnames = []
            while True:
                try:
                    fname = template.format_map(
                        dict(
                            header,
                            PROJHOME=project_home,
                            magfit_iteration=len(iteration_fnames),
                        )
                    )
                except (KeyError, IndexError, ValueError):
                    break
                # Without the iteration, a template names the same file again.
                if fname in iteration_fnames or not os.path.exists(fname):
                    break
                iteration_fnames.append(fname)
            result.update(iteration_fnames)
    return sorted(result)


#: For each kind of product, in the order converted: how to find its files,
#: and how to convert one.
_KINDS = {
    "calibrated_images": (
        lambda connection, home: _find_image_products(
            connection, home, "calibrated-fname"
        ),
        _migrate_fits,
    ),
    "masters": (_find_fits_masters, _migrate_master),
    "magfit_iteration_masters": (_find_photref_iterations, _migrate_master),
    "data_reduction": (
        lambda connection, home: _find_image_products(
            connection, home, "data-reduction-fname"
        ),
        _migrate_dr,
    ),
    "lightcurves": (_find_lightcurves, _migrate_lc),
}


def upgrade():
    """Convert the paths the database holds, then those the products hold."""

    connection = alembic.op.get_bind()
    project_home = alembic.context.config.attributes["project_home"]

    _convert_files(connection, "master_file", "filename", project_home)
    _convert_files(connection, "image", "raw_fname", project_home)
    _convert_configuration(connection, project_home)

    parameter = sqlalchemy.Table(
        "parameter", sqlalchemy.MetaData(), autoload_with=connection
    )
    connection.execute(
        parameter.update()
        .where(
            parameter.c.name == "subpixmap",
            parameter.c.description == SUBPIXMAP_DESCRIPTION["old"],
        )
        .values(description=SUBPIXMAP_DESCRIPTION["new"])
    )

    # Last: the products are found through the database as converted.
    for kind, (find, migrate) in _KINDS.items():
        migrate_fnames = find(connection, project_home)
        _logger.info(
            "Converting the paths in %d %s.", len(migrate_fnames), kind
        )
        changed = sum(migrate(fname, project_home) for fname in migrate_fnames)
        _logger.info("Changed %d of the %s.", changed, kind)


def downgrade():
    """Refuse: the products are not converted back."""

    raise DatabaseError(
        "A project cannot be taken back past AutoWISP 3.0: its products store "
        "paths as {PROJHOME}/..., and are not converted back, so the database "
        "would no longer match them. Restore the copy of the database taken "
        "before migrating, and the products from a backup, instead."
    )
