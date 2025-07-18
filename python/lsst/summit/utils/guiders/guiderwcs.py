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

from lsst.afw import cameraGeom
from lsst.geom import Angle, degrees
from lsst.obs.base import createInitialSkyWcsFromBoresight


def make_init_guider_wcs(camera, visitInfo) -> dict[str, any]:
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
