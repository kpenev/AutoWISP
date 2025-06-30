"""Define a class for splitting sources in groups for PRF fitting."""

from math import inf
import numpy


# Pylint does not count __init__ and __call__, but should.
# pylint: disable=too-few-public-methods
class SplitSources:
    """
    Split sources in groups based on user-defined conditions.

    Currently splitting by distance from the center of the frame and by
    magnitude are supported.
    """

    def _find_mag_splits(self, sorted_magnitudes):
        """
        Find the magnitude boundaries that satisfy the source count limits.

        Args:
            sorted_magnitudes:    The sorted magnitudes of the sources which
                pass the current radius selection.

        Returns:
            boundaries:    The values ofthe magnitudes where to split the input
                list of sources in order to ensure that groups have exactly the
                minimum number of sources specified at construction, except the
                group containing the faintest sources which could be up to two
                time larger in order to ensure no sources are left without a
                group.
        """

        return (
            [-inf]
            + [
                (
                    sorted_magnitudes[split_index - 1]
                    + sorted_magnitudes[split_index]
                )
                / 2.0
                for split_index in range(
                    self.mag_split_min_sources,
                    len(sorted_magnitudes) - self.mag_split_min_sources + 1,
                    self.mag_split_min_sources,
                )
            ]
            + [inf]
        )

    def __init__(
        self,
        magnitude_column,
        *,
        radius_splits=(),
        manual_mag_splits=(),
        mag_split_by_source_count=None,
    ):
        """
        Set-up the splitting.

        Args:
            magnitude_column:    The filter in which to do the splitting by
                magnitude. Should be one of the columns available in
                the the input sources.

            radius_splits:    The boundaries in radius (measured in pixels) at
                which to split fit groups.

            manual_mag_splits:    The boundaries in magnitude at which to split
                fit groups. It is an error to specify both manual_mag_splits
                and mag_split_by_source_count.

            mag_split_by_source_count:    If specied, magnitude splitting is
                done to ensure at least this many sources in each group. It is
                an error to specify both manual_mag_splits
                and mag_split_by_source_count.

        Returns:
            None
        """

        self.magnitude_column = magnitude_column
        self.radius_splits = sorted(set([0] + list(radius_splits) + [inf]))

        if manual_mag_splits:
            assert mag_split_by_source_count is None
            self.mag_splits = sorted(
                set([-inf] + list(manual_mag_splits) + [inf])
            )
        elif mag_split_by_source_count is None:
            self.mag_splits = [-inf, inf]

        self.mag_split_min_sources = mag_split_by_source_count
        assert self.radius_splits[0] == 0

    # TODO: See if it can be simplified
    # pylint: disable=too-many-locals
    def __call__(self, sources, image_resolution):
        """
        Return an array of integers grouping PRF fitting sources as specified.

        Args:
            sources(structured array or pandas.DataFrame):    The sources to
                split. Must define at least `'ID'`, ``'x'`, `'y'`, `'phqual'`,
                `'objtype'`, and `'doublestar'` fields as well as any magnitudes
                on which splitting is to be done.

            image_resolution:    The resolution of the image in pixels (y, x).

        Returns:
            grouping:    A numpy integer array indicating for each
                source the PRF fitting group it is in. The grouping of sources
                is controlled by the splitting arguments specified at
                construction time with each group of sources getting a unique
                positive integer. Sources which should not be used for PRF
                fitting are assigned a group ID of `-1`.

            in_frame:    A numpy boolean array indicating for each source
                whether it's center lies inside the frame or not.
        """

        print(f"Splitting {len(sources):d} sources")

        in_frame = numpy.logical_and(
            numpy.logical_and(
                sources["x"] > 0,
                sources["x"] < image_resolution[1],
            ),
            numpy.logical_and(
                sources["y"] > 0,
                sources["y"] < image_resolution[0],
            ),
        )

        good_sources = in_frame
        for column_name, value in [
            ("phqual", "AAA"),
            ("objtype", 0),
            ("doublestar", 0),
        ]:
            try:
                good_sources = numpy.logical_and(
                    good_sources, sources[column_name] == value
                )
            except KeyError:
                pass

        square_radius = numpy.square(
            sources["x"] - image_resolution[1] / 2.0
        ) + numpy.square(sources["y"] - image_resolution[0] / 2.0)

        grouping = numpy.empty(len(sources), dtype=numpy.int32)
        grouping[numpy.logical_not(good_sources)] = -1

        group_id = 0

        for min_radius, max_radius in zip(
            self.radius_splits, self.radius_splits[1:]
        ):

            radius_group_sources = numpy.logical_and(
                numpy.logical_and(
                    square_radius >= min_radius**2,
                    square_radius < max_radius**2,
                ),
                good_sources,
            )

            if self.mag_split_min_sources is None:
                magnitude_splits = self.mag_splits
            else:
                magnitude_splits = self._find_mag_splits(
                    numpy.sort(
                        sources[self.magnitude_column][radius_group_sources]
                    )
                )

            for min_magnitude, max_magnitude in zip(
                magnitude_splits, magnitude_splits[1:]
            ):
                group_sources = numpy.logical_and(
                    numpy.logical_and(
                        sources[self.magnitude_column] >= min_magnitude,
                        sources[self.magnitude_column] < max_magnitude,
                    ),
                    radius_group_sources,
                )
                grouping[group_sources] = group_id
                group_id += 1

        return grouping, in_frame

    # pylint: enable=too-many-locals


# pylint: enable=too-few-public-methods
