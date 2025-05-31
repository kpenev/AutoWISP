"""Collection of small utils that is hard to classify."""

RECOGNIZED_HAT_ID_PREFIXES = ["HAT", "UCAC4"]


def get_hat_source_id_str(source_id):
    """Return the string representation of 3-integer HAT-id."""

    return RECOGNIZED_HAT_ID_PREFIXES[
        source_id[0]
    ] + "-{src[1]:03d}-{src[2]:07d}".format(src=source_id)
