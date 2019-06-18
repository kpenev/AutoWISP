"""Functions for parsing files generated with HAT tools."""

import scipy

def parse_transformation(filename):
    """
    Parse a transformation file suitable for grtrans.

    Args:
        filename(str):    The name of the file to parse.

    Returns:
        dict:
            All quantity defining the transformation, e.g. tape, order, scale,
            offset and coefficients for each transformed variable.

        dict:
            Information about the properties of the transformation contained in
            the comments.
    """

    transformation = dict()
    info = dict()
    with open(filename, 'r') as trans_file:
        for line in trans_file:
            if line.strip()[0] == '#':
                split_line = line.strip().lstrip('#').strip().split(':', 1)
                if len(split_line) > 1:
                    quantity, value = split_line
                    if value:
                        info[quantity.strip().lower()] = value.strip()
            else:
                quantity, value = line.split('=')
                quantity = quantity.strip().lower()
                value = value.strip()
                if quantity == 'order':
                    value = int(value)
                elif quantity == 'scale':
                    value = float(value)
                elif quantity in ['offset', 'basisshift']:
                    value = tuple(float(v.strip()) for v in value.split(','))
                elif quantity != 'type':
                    value = scipy.array(
                        [float(v.strip()) for v in value.split(',')]
                    )
                transformation[quantity] = value

    return transformation, info

def parse_anmatch_transformation(filename):
    """Parse transformation files generate by anmatch."""

    def parse_multivalue(info, value_parser, head='', tail=''):

        values, description = info.rstrip(')').split('(')
        if head:
            assert description.startswith(head)
            description = description[len(head):]
        if tail:
            assert description.endswith(tail)
            description = description[:-len(tail)]

        keys = (k.strip() for k in description.strip().split(','))
        values = (value_parser(v) for v in values.strip().split())
        return dict(zip(keys, values))

    transformation, info = parse_transformation(filename)
    for info_key in ['residual', 'unitarity']:
        info[info_key] = float(info[info_key])

    info['points'] = parse_multivalue(info['points'], int, head='number of:')

    info['ratio'] = float(info['ratio'].split(None, 1)[0]) / 100.0

    info['timing'] = parse_multivalue(info['timing'],
                                      float,
                                      tail = ': in seconds')

    del info['all']

    info['2mass'] = parse_multivalue(info['2mass'], float)
    for size_char in 'wh':
        info['2mass']['image' + size_char] = int(
            info['2mass']['imag' + size_char]
        )
    return transformation, info
