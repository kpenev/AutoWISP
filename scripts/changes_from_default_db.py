#!/usr/bin/env python3

"""Identify any changes to the database introduced since init with defalut."""

from sqlalchemy.orm import contains_eager

from autowisp.database.interface import Session
#Pylint false positive due to quirky imports.
#pylint: disable=no-name-in-module
from autowisp.database.data_model import\
    HDF5Product,\
    HDF5StructureVersion
#pylint: enable=no-name-in-module
from autowisp.database.initialize_light_curve_structure import\
    get_default_light_curve_structure
from autowisp.database.initialize_data_reduction_structure import\
    get_default_data_reduction_structure

def report_key_mismatch(default_key_set, config_key_set):
    """Return string reporting mismatch between the set of keys defined"""

    report = ''
    for message, missing_keys in [
            (
                'In default but not configured',
                (default_key_set - config_key_set)
            ),
            (
                'Configured but not in default',
                (config_key_set - default_key_set)
            )
    ]:
        if missing_keys:
            report += ('\t\t' + message + ':\n'
                       +
                       '\t\t\t'+ '\n\t\t\t'.join(missing_keys))
    return report

def report_changes_in_hdf5_component(default, configured, component):
    """Report any differences between default and configured HDF5 attribute."""

    assert default.pipeline_key == configured.pipeline_key

    to_check = dict(
        attribute=['parent',
                   'name',
                   'dtype'],
        dataset=['abspath',
                 'dtype',
                 'compression',
                 'compression_options',
                 'scaleoffset',
                 'shuffle'],
        link=['abspath',
              'target']
    )
    report = ''
    for check in to_check[component]:
        default_value = getattr(default, check)
        config_value = getattr(configured, check)
        if (
                default_value != config_value
                and
                (
                    default_value is not None
                    or
                    config_value is not False
                )
        ):
            report += ('\t\t\t%s mismatch: %s (default) vs %s (configured)\n'
                       %
                       (check.title(), repr(default_value), repr(config_value)))
    if report:
        return (
            '\t\t%s %s:\n' % (component.title(), default.pipeline_key)
            +
            report
        )
    return ''

def report_changes_in_hdf5_structure(default_structure, db_session):
    """Compare default structure for one HDF5 file type to current config."""

    configured_structure = db_session.query(
        HDF5Product
    ).join(
        HDF5Product.structure_versions
    ).options(
        contains_eager(
            HDF5Product.structure_versions
        ).subqueryload(
            HDF5StructureVersion.datasets
        )
    ).options(
        contains_eager(
            HDF5Product.structure_versions
        ).subqueryload(
            HDF5StructureVersion.attributes
        )
    ).options(
        contains_eager(
            HDF5Product.structure_versions
        ).subqueryload(
            HDF5StructureVersion.links
        )
    ).filter(
        HDF5Product.pipeline_key == default_structure.pipeline_key
    ).order_by(
        HDF5StructureVersion.version.desc()
    ).first()

    report = ''
    for structure_component in ['attributes', 'datasets', 'links']:
        default_components = getattr(default_structure.structure_versions[0],
                                     structure_component)
        config_components = getattr(configured_structure.structure_versions[0],
                                    structure_component)

        default_pipeline_keys = [comp.pipeline_key
                                 for comp in default_components]
        config_pipeline_keys = [comp.pipeline_key
                                for comp in config_components]

        component_report = report_key_mismatch(set(default_pipeline_keys),
                                               set(config_pipeline_keys))
        for default_index, pipeline_key in enumerate(default_pipeline_keys):
            try:
                config_index = config_pipeline_keys.index(pipeline_key)
            except ValueError:
                continue
            component_report += report_changes_in_hdf5_component(
                default_components[default_index],
                config_components[config_index],
                structure_component[:-1]
            )
        if component_report:
            report += ('\t' + structure_component + ':\n'
                       +
                       component_report)

    if report:
        print(default_structure.pipeline_key  + ':')
        print(report)

def report_changes():
    """Create the report without polluting global scope."""

    with Session.begin() as db_session:
        report_changes_in_hdf5_structure(
            get_default_light_curve_structure(db_session),
            db_session
        )
        report_changes_in_hdf5_structure(
            get_default_data_reduction_structure(),
            db_session
        )

if __name__ == '__main__':
    report_changes()
