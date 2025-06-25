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

__all__ = [
    "make_init_guider_wcs",
]

from lsst.afw import cameraGeom
from lsst.obs.base import createInitialSkyWcsFromBoresight


def make_init_guider_wcs(camera, visitInfo):
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
            cam_wcs[det.getId()] = createInitialSkyWcsFromBoresight(boresight, orientation, det, flipX=False)

    return cam_wcs
