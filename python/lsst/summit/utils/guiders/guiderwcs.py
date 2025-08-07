# This file is part of summit_utils.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

__all__ = [
    "make_init_guider_wcs",
]

# Additional imports
from dataclasses import dataclass
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time

from lsst.afw import cameraGeom
from lsst.geom import Angle, degrees
from lsst.obs.base import createInitialSkyWcsFromBoresight
from lsst.obs.lsst.translators.lsst import SIMONYI_LOCATION

if TYPE_CHECKING:
    from lsst.geom import SkyWcs


def make_init_guider_wcs(camera, visitInfo) -> dict[str, SkyWcs]:
    """
    Parameters
    ----------
    camera : lsst.afw.cameraGeom.Camera
        Camera object
    visitInfo : lsst.afw.image.VisitInfo
        visit info from an Image

    Returns
    -------
    cam_wcs : dictionary
        WCS for each detector, keyed by detector Id

    """
    orientation = visitInfo.getBoresightRotAngle()
    boresight = visitInfo.getBoresightRaDec()

    # Get WCS for each CCD in camera
    cam_wcs = {}

    for det in camera:
        if det.getType() == cameraGeom.DetectorType.GUIDER:
            # get WCS
            args = boresight, orientation, det
            cam_wcs[det.getName()] = createInitialSkyWcsFromBoresight(*args, flipX=False)

    return cam_wcs


def get_camera_rot_angle(visitinfo) -> float:
    """Get the camera rotation angle from visitInfo

    Parameters
    ----------
    visitInfo : lsst.afw.image.VisitInfo
        visit info from an Image

    Returns
    -------
    orientation : float
        Camera rotation angle in radians

    """
    camRot = (
        visitinfo.getBoresightParAngle().asDegrees() - visitinfo.getBoresightRotAngle().asDegrees() - 90.0
    )
    camRotAngle = Angle(camRot, degrees)
    camRotAngleW = camRotAngle.wrapNear(Angle(0.0))
    return camRotAngleW.asDegrees()


# Data class for drift results
@dataclass
class DriftResult:
    ra_point: float
    dec_point: float
    ra_real: float
    dec_real: float
    delta_ra_arcsec: float
    delta_dec_arcsec: float
    az_drift_arcsec: float
    el_drift_arcsec: float
    total_drift_arcsec: float
    el_start: float
    pixel_offset: tuple[float, float]

    def __str__(self, expid=""):
        s = (
            f"Exposure summary: {expid}\n"
            f"Telescope pointing (RA, Dec): ({self.ra_point:.6f}, {self.dec_point:.6f})\n"
            f"Actual pointing   (RA, Dec) : ({self.ra_real:.6f}, {self.dec_real:.6f})\n"
            f"Pointing error (RA, Dec).   : ({self.delta_ra_arcsec:.1f},"
            f"{self.delta_dec_arcsec:.1f}) arcseconds\n"
            f"Starting elevation.         : {self.el_start:.2f} degrees\n"
            f"Pixel offset from origin   : ({self.pixel_offset[0]:.4f}, {self.pixel_offset[1]:.4f}) pixels\n"
            f"\n"
            f"-------------------- Drift Summary --------------------\n"
            f"Azimuth drift   : {self.az_drift_arcsec:.2f} arcseconds\n"
            f"Elevation drift : {self.el_drift_arcsec:.2f} arcseconds\n"
            f"Total drift.    : {self.total_drift_arcsec:.2f} arcseconds\n"
        )
        return s

    def summary(self, expid=""):
        print(self.__str__(expid))


def getObsAltAz(ra, dec, pressure, hum, temperature, wl, time):
    # given the RA/Dec and other variables
    skyLocation = SkyCoord(ra * u.deg, dec * u.deg)
    altAz1 = AltAz(
        obstime=time,
        location=SIMONYI_LOCATION,
        pressure=pressure,
        temperature=temperature,
        relative_humidity=hum,
        obswl=wl,
    )
    obsAltAz1 = skyLocation.transform_to(altAz1)
    return obsAltAz1


def DeltaAltAz(ra, dec, pressure, hum, temperature, wl, time1, time2):
    # This calculates the change in AltAz during an exposure
    obsAltAz1 = getObsAltAz(ra, dec, pressure, hum, temperature, wl, time1)
    obsAltAz2 = getObsAltAz(ra, dec, pressure, hum, temperature, wl, time2)
    # 1 is at the beginning of the exposure, 2 is at the end
    # el, az are the actual values, prime values reflect the pointing model
    # These are all in degrees
    el1 = obsAltAz1.alt.deg
    az1 = obsAltAz1.az.deg
    el2 = obsAltAz2.alt.deg
    az2 = obsAltAz2.az.deg
    # Change values are the change f
    # rom the beginning to the end of the exposure, in arcseconds
    azChange = (az2 - az1) * 3600.0
    elChange = (el2 - el1) * 3600.0
    return [azChange, elChange]


def calculate_drift(metadata, cWcs, rWcs):
    """
    Calculate telescope drift from exposure metadata and WCS.

    Parameters
    ----------
    metadata : dict-like
        Must contain at least: 'FILTBAND', 'PRESSURE', 'AIRTEMP', 'HUMIDITY',
        'MJD-BEG', 'MJD-END', 'RASTART', 'DECSTART', 'ELSTART'
    cWcs, rWcs : lsst.afw.geom.SkyWcs
        Calibrated and raw WCS objects

    Returns
    -------
    DriftResult

    Example
    -------
    expId = 2025072300537
    rawExp = butler.get('raw', detector=94,
                        exposure=expId, instrument=instrument)
    calExp = butler.get('preliminary_visit_image',
                        detector=94, visit=expId, instrument=instrument)

    md = rawExp.getMetadata()
    cWcs = calExp.getWcs()
    rWcs = rawExp.getWcs()

    drift547 = calculate_drift(md, cWcs, rWcs)
    drift547.summary(2025072300537)
    """
    wavelengths = {"u": 3671, "g": 4827, "r": 6223, "i": 7546, "z": 8691, "y": 9712}
    filter = metadata["FILTBAND"]
    wl = wavelengths[filter] * u.angstrom
    pressure = metadata["PRESSURE"] * u.pascal
    temperature = metadata["AIRTEMP"] * u.Celsius
    hum = metadata["HUMIDITY"]
    time1 = Time(metadata["MJD-BEG"], format="mjd", scale="tai")
    time2 = Time(metadata["MJD-END"], format="mjd", scale="tai")
    ra_point = metadata["RASTART"]
    dec_point = metadata["DECSTART"]
    el_start = metadata["ELSTART"]

    azChangePoint, elChangePoint = DeltaAltAz(
        ra_point, dec_point, pressure, hum, temperature, wl, time1, time2
    )

    # Use WCS to get actual (ra, dec) at image center
    calExpSkyCenter = cWcs.pixelToSky(rWcs.getPixelOrigin())
    ra_real = calExpSkyCenter.getRa().asDegrees()
    dec_real = calExpSkyCenter.getDec().asDegrees()
    delta_ra = (ra_real - ra_point) * 3600.0
    delta_dec = (dec_real - dec_point) * 3600.0

    # Calculate pixel offset from raw WCS origin to calibrated WCS origin
    pixel_offset = tuple(cWcs.getPixelOrigin() - rWcs.getPixelOrigin())

    azChangeReal, elChangeReal = DeltaAltAz(ra_real, dec_real, pressure, hum, temperature, wl, time1, time2)

    az_drift = azChangeReal - azChangePoint
    el_drift = elChangeReal - elChangePoint
    total_drift = np.sqrt(el_drift**2 + (az_drift * np.cos(np.radians(el_start))) ** 2)

    return DriftResult(
        ra_point=ra_point,
        dec_point=dec_point,
        ra_real=ra_real,
        dec_real=dec_real,
        delta_ra_arcsec=delta_ra,
        delta_dec_arcsec=delta_dec,
        az_drift_arcsec=az_drift,
        el_drift_arcsec=el_drift,
        total_drift_arcsec=total_drift,
        el_start=el_start,
        pixel_offset=pixel_offset,
    )
