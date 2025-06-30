"""Define base class for all pipeline processing steps enforcing conventions."""


# TODO: add recovery and logging.
# pylint: disable=too-few-public-methods
class Processor:
    """
    Providing and enforcing uniform interface for config, recovery, and logging.
    """

    default_configuration = {}

    def __init__(self, **configuration):
        """Prepare to process with the given configuration."""

        self.configuration = dict(self.default_configuration)
        self.configuration.update(configuration)

    def __call__(self, **configuration):
        """Add/overwrite any configuration parameters at time of processing."""

        for key, value in self.configuration.items():
            if key not in configuration:
                configuration[key] = value
        return configuration


# pylint: enable=too-few-public-methods
