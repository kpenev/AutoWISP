"""Sort DR files by their single photometric reference merit function."""

import pandas

from astropy import units
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

from autowisp.astrometry import Transformation
from autowisp import DataReductionFile


def get_matched_sources(dr_fname, path_substitutions):
    """Convenience wrapper around `DataReductionFile.get_matched_sources()`."""

    with DataReductionFile(dr_fname, 'r') as dr:
        return dr.get_matched_sources(**path_substitutions)


def get_average_matched_sources(dr_fnames,
                                source_average='median',
                                frame_average='median',
                                **path_substitutions):
    """Return the average of the matched sources in all DR files."""

    source_averaged = pandas.DataFrame(
        getattr(get_matched_sources(fname, path_substitutions),
                source_average)()
        for fname in dr_fnames
    )
    return getattr(source_averaged, frame_average)()


def get_merit_info(dr_fname, **dr_path_substitutions):
    """Return the properties relevant for calculating the merit of given DR."""

    with DataReductionFile(dr_fname, 'r') as dr_file:
        header = dr_file.get_frame_header()
        astrometry = Transformation()
        astrometry.read_transformation(dr_file, **dr_path_substitutions)
        location = EarthLocation(lat=header['SITELAT'] * units.deg,
                                 lon=header['SITELONG'] * units.deg,
                                 height=header['SITEALT'] * units.m)
        obs_time = Time(header['JD-OBS'], format='jd', location=location)
        source_coords = SkyCoord(
            ra=astrometry.pre_projection_center[0] * units.deg,
            dec=astrometry.pre_projection_center[1] * units.deg,
            frame='icrs'
        )
        altitude = source_coords.transform_to(
            AltAz(obstime=obs_time, location=location)
        ).alt.to_value(units.deg)

        return {
            'zenith_distance': 90.0 - altitude
        }
