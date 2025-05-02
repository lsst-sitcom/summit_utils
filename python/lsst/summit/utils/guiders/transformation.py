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
    "focal_to_pixel",
    "pixel_to_focal",
    "stamp_to_ccd",
    "amp_to_ccdview"
]

import numpy as np
from lsst.afw import cameraGeom
from lsst.obs.lsst import LsstCam
from lsst.obs.lsst.cameraTransforms import LsstCameraTransforms

def focal_to_pixel(fpx, fpy, det):
    """
    Parameters
    ----------
    fpx, fpy : array
        Focal plane position in millimeters in DVCS
        See https://lse-349.lsst.io/
    det : lsst.afw.cameraGeom.Detector
        Detector of interest.

    Returns
    -------
    x, y : array
        Pixel coordinates.
    """
    tx = det.getTransform(cameraGeom.FOCAL_PLANE, cameraGeom.PIXELS)
    x, y = tx.getMapping().applyForward(np.vstack((fpx, fpy)))
    return x.ravel(), y.ravel()

def pixel_to_focal(x, y, det):
    """
    Parameters
    ----------
    x, y : array
        Pixel coordinates.
    det : lsst.afw.cameraGeom.Detector
        Detector of interest.

    Returns
    -------
    fpx, fpy : array
        Focal plane position in millimeters in DVCS
        See https://lse-349.lsst.io/
    """
    tx = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
    fpx, fpy = tx.getMapping().applyForward(np.vstack((x, y)))
    return fpx.ravel(), fpy.ravel()

def stamp_to_ccd(stamp,ccdimg,detector,camera,ampName,ampCol,ampRow):
    """ 
    Parameters
    ----------
    stamp : array
        raw ROI array in amp coordinates
    ccdimg : array
        ccd image array in ccd coordinates
    detector : lsst.afw.cameraGeom.Detector
        Detector of interest.
    camera : lsst.afw.cameraGeom.Camera 
        Camera object
    ampName : str
        amplifier name, eg. C00
    ampCol : int
        starting column number for ROI
    ampRow : int
        starting row number for ROI

    Returns
    -------
    ccdimg : array
        ccd image array filled with ROI
    """
    
    stamp_ccd = amp_to_ccdview(stamp,detector,ampName)
    ampRows,ampCols = stamp_ccd.shape

    # get corner0 of the location for the ROI
    lct = LsstCameraTransforms(camera,detector.getName())  
    corner0_CCDX,corner0_CCDY = lct.ampPixelToCcdPixel(ampCol,ampRow,ampName)
    corner0_CCDX = int(corner0_CCDX)
    corner0_CCDY = int(corner0_CCDY)

    # get opposite corner, corner2, of the ROI
    amp = detector[ampName]
    if amp.getRawFlipX():
        corner2_CCDX = corner0_CCDX - ampCols
    else:
        corner2_CCDX = corner0_CCDX + ampCols

    if amp.getRawFlipY():
        corner2_CCDY = corner0_CCDY - ampRows
    else:
        corner2_CCDY = corner0_CCDY + ampRows

    # now place the ROI
    try:
        ccdimg[min(corner0_CCDY,corner2_CCDY):max(corner0_CCDY,corner2_CCDY),
               min(corner0_CCDX,corner2_CCDX):max(corner0_CCDX,corner2_CCDX)] = stamp_ccd
    except:
        print(detector.getName(),ampName)
        print('Y: ',min(corner0_CCDY,corner2_CCDY),max(corner0_CCDY,corner2_CCDY))
        print('X: ',min(corner0_CCDX,corner2_CCDX),max(corner0_CCDX,corner2_CCDX))

    return ccdimg

def amp_to_ccdview(stamp,detector,ampName):
    """ 
    Parameters
    ----------
    stamp : array
        raw ROI array in amp coordinates
    detector : lsst.afw.cameraGeom.Detector
        Detector of interest.
    ampName : str
        amplifier name, eg. C00

    Returns
    -------
    img : array
        ROI image flipped to be in CCD coordinate orientation
    """

    """ Converts an image array from Amp view to CCD (DMCV) view
    """
    amp = detector[ampName]
    img = stamp.copy()
    if amp.getRawFlipX():
        img = np.fliplr(img)
    if amp.getRawFlipY():
        img = np.flipud(img)
    return img
