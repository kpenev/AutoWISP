"""Functions for creating light curves from DR files."""

from .lc_data_reader import LCDataReader

def organize_configurations(configurations_list):
    """
    Fill the cfg attributes and the cfg ID columns of the data slice.

    Args:
        configurations_list:    The list of configurations returned by
            LCDataReader for the data currently in the slice. It is assumed that
            the configurations follow the same order as the data slice entries.

    Returns:
        dict:
            The keys are configuration components and the values are
            dictionaries with keys the coordinates along each dimension of
            the configuration index dataset and values lists of
            the configurations for the component with indices in the list
            corresponding to the entries added to the config ID columns in
            ReadLCData.lc_data_slice.
    """

    result = {component: dict()
              for component in LCDataReader.config_components}
    for frame_index, configurations in enumerate(configurations_list):
        for component, component_config in configurations.items():
            for dim_values, config in component_config:
                if dim_values not in result[component]:
                    result[component][dim_values] = dict()
                if config in result[component][dim_values]:
                    config_id = result[component][dim_values][config]
                else:
                    config_id = len(result[component][dim_values])
                    result[component][dim_values][config] = config_id
                LCDataReader.set_field_entry(
                    component + '.' + LCDataReader.cfg_index_id,
                    config_id,
                    frame_index,
                    dim_values
                )
    return result

def print_organized_configurations(organized_config):
    """Print the result of organize_configurations() nicely formatted."""

    print('Organized configurations:')
    for component, component_config in organized_config.items():
        print('\t' + component + ':')
        for dim_values, config_list in component_config.items():
            dim_id = dict(
                zip(
                    filter(
                        lambda dim: dim not in ['frame', 'source'],
                        LCDataReader.config_components[component][0]
                    ),
                    dim_values
                )
            )
            print('\t\t' + repr(dim_id) + ':')
            for config in config_list:
                print('\t\t\t' + repr(config))
