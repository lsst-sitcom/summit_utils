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
    "mk_rot",
    "mk_ccd_to_dvcs",
    "mk_roi_bboxes",
    "stamp_to_ccd",
    "get_detector_amp",
    "convert_roi",
    "focal_to_pixel",
    "pixel_to_focal",
    "amp_to_ccdview",
    "convert_focal_to_altaz",
    "convert_pixels_to_altaz",
    "convert_to_focal_plane",
    "convert_roi_to_ccd",
    "convert_ccd_to_roi",
    "convert_ccd_to_dvcs",
]

from typing import TYPE_CHECKING, Any

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time

from lsst.afw import cameraGeom
from lsst.afw.cameraGeom import Detector
from lsst.afw.image import ImageF
from lsst.geom import AffineTransform, Box2D, Box2I, Extent2D, Point2D
from lsst.obs.lsst import LsstCam  # noqa F401
from lsst.obs.lsst.cameraTransforms import LsstCameraTransforms  # noqa F401
from lsst.obs.lsst.translators.lsst import SIMONYI_LOCATION  # noqa F401

if TYPE_CHECKING:
    from astropy.time import Time  # noqa F811

    from lsst.summit.utils.guiders.reading import GuiderData


def convert_pixel_to_radec(wcs: Any, x_flat: np.ndarray, y_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Map detector-pixel coordinates to ICRS RA and DEC (in radians).

    Parameters
    ----------
    wcs : object
        World Coordinate System object with pixelToSkyArray method.
    xFlat : np.ndarray
        Flattened array of x pixel coordinates.
    yFlat : np.ndarray
        Flattened array of y pixel coordinates.

    Returns
    -------
    raFlat : np.ndarray
        Array of right ascension values in radians.
    decFlat : np.ndarray
        Array of declination values in radians.
    """
    return wcs.pixelToSkyArray(x_flat, y_flat)


def convert_pixels_to_altaz(
    wcs: Any, time: Time, xPix: np.ndarray, yPix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert detector-pixel coordinates to Altitude and Azimuth (degrees)
    in a vectorized manner.

    Parameters
    ----------
    wcs : object
        World Coordinate System object with pixelToSkyArray method.
    time : astropy.time.Time
        Observation time.
    xPix : np.ndarray
        Array of x pixel coordinates.
    yPix : np.ndarray
        Array of y pixel coordinates.

    Returns
    -------
    az : np.ndarray
        Array of azimuth values in degrees (same shape as xPix/yPix).
    alt : np.ndarray
        Array of altitude values in degrees (same shape as xPix/yPix).
    """

    # 1) make sure we have numpy arrays, remember their shape
    x_arr = np.asarray(xPix)
    y_arr = np.asarray(yPix)
    shp = x_arr.shape

    # 2) flatten for the WCS call
    x_flat = x_arr.ravel()
    y_flat = y_arr.ravel()

    # 3) scalarSKyWcs → ICRS RA/Dec (radians) in bulk:
    #    (pixelToSkyArray expects floats and returns two 1D arrays)
    ra_flat, dec_flat = convert_pixel_to_radec(wcs, x_flat, y_flat)

    # 4) assemble a single SkyCoord and transform once
    sc_icrs = SkyCoord(
        ra=ra_flat * u.rad,
        dec=dec_flat * u.rad,
        frame="icrs",
        obstime=time,
        location=SIMONYI_LOCATION,
    )
    # Transform to AltAz
    sc_altaz = sc_icrs.transform_to(AltAz(obstime=time, location=SIMONYI_LOCATION))

    # 5) reshape back to original grid
    az = sc_altaz.az.deg.reshape(shp)
    alt = sc_altaz.alt.deg.reshape(shp)
    return az, alt


def convert_focal_to_altaz(
    wcs: Any, time: Time, detector: Detector, xFocal: np.ndarray, yFocal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map focal-plane coordinates (mm) to Altitude/Azimuth (degrees) by
    chaining focal → pixel → AltAz transformations.

    Parameters
    ----------
    wcs : object
        World Coordinate System object with pixelToSkyArray method.
    time : astropy.time.Time
        Observation time.
    detector : lsst.afw.cameraGeom.Detector
        Detector object.
    xFocal : np.ndarray
        Array of x focal-plane coordinates (mm).
    yFocal : np.ndarray
        Array of y focal-plane coordinates (mm).

    Returns
    -------
    az : np.ndarray
        Array of azimuth values in degrees.
    alt : np.ndarray
        Array of altitude values in degrees.
    """
    x_pix, y_pix = focal_to_pixel(xFocal, yFocal, detector)
    return convert_pixels_to_altaz(wcs, time, x_pix, y_pix)


# Aaron's code to make transformations from ROI coordinates to Focal Plane and
# sky coordinates.
def mk_rot(detNquarter: int, direction: int = 1) -> AffineTransform:
    """
    Create an AffineTransform representing a rotation
    by multiples of 90 degrees.

    Parameters
    ----------
    detNquarter : int
        Number of 90 degree counter-clockwise rotations (modulo 4).
    direction : int, optional
        1 for forward rotation (default), -1 for reverse rotation (transpose)

    Returns
    -------
    rotation : lsst.geom.AffineTransform
        The rotation transformation.

    Raises
    ------
    ValueError
        If direction is not 1 or -1.
    """
    rot = {}
    rot[0] = np.array([[1.0, 0.0], [0.0, 1.0]])
    rot[1] = np.array([[0.0, -1], [1.0, 0.0]])
    rot[2] = np.array([[-1.0, 0.0], [0.0, -1.0]])
    rot[3] = np.array([[0.0, 1.0], [-1.0, 0.0]])

    irot = {}
    irot[0] = rot[0].transpose()
    irot[1] = rot[1].transpose()
    irot[2] = rot[2].transpose()
    irot[3] = rot[3].transpose()

    nq = np.mod(detNquarter, 4)
    if direction == 1:
        rotation = AffineTransform(rot[nq])
    elif direction == -1:
        rotation = AffineTransform(irot[nq])
    else:
        raise ValueError(f"direction must be either +/- 1, got {direction}")
    return rotation


def mk_ccd_to_dvcs(
    bboxCcd: Box2I,
    detNquarter: int,
) -> tuple[AffineTransform, AffineTransform]:
    """
    Create forward and backward AffineTransforms for converting between
    CCD pixel coordinates and DVCS (focal plane) stamp coordinates.

    Parameters
    ----------
    bboxCcd : lsst.geom.Box2I
        Bounding box for the ROI in CCD pixel coordinates.
    detNquarter : int
        Number of 90-degree CCW rotations to align CCD view to DVCS view.

    Returns
    -------
    forwards : lsst.geom.AffineTransform
        Transform from CCD pixel coordinates (CCD view) to stamp
        pixel coordinates (DVCS view).
    backwards : lsst.geom.AffineTransform
        Transform from stamp pixel coordinates (DVCS view) to
        CCD pixel coordinates (CCD view).

    Notes
    -----
    Useful for mapping between the raw detector coordinates and the orientation
    of the focal plane as used in DVCS.
    """
    # Use shared integer rotation matrices
    nq = np.mod(det_nquarter, 4)
    rot = ROTATION_MATRICES
    irot = INVERSE_ROTATION_MATRICES
    # number of 90deg CCW rotations
    nq = np.mod(detNquarter, 4)

    # get LL,Size of the CCD view BBox
    llpt_ccd = Extent2D(bboxCcd.getCorners()[0])
    nx, ny = bboxCcd.getDimensions()

    # get translations to use for each NQ value
    boxtranslation = {}
    boxtranslation[0] = Extent2D(0, 0)
    boxtranslation[1] = Extent2D(ny - 1, 0)
    boxtranslation[2] = Extent2D(nx - 1, ny - 1)
    boxtranslation[3] = Extent2D(0, nx - 1)

    # build the forware Transform
    ftranslation = AffineTransform.makeTranslation(-llpt_ccd)
    frotation = AffineTransform(rot[nq])
    fboxtranslation = AffineTransform.makeTranslation(boxtranslation[nq])
    # ordering is third*second*first
    forwards = fboxtranslation * frotation * ftranslation

    btranslation = AffineTransform.makeTranslation(llpt_ccd)
    brotation = AffineTransform(irot[nq])
    bboxtranslation = AffineTransform.makeTranslation(-boxtranslation[nq])
    backwards = btranslation * brotation * bboxtranslation

    return forwards, backwards


def mk_roi_bboxes(md, camera):
    """
    Construct the bounding box for a Guider ROI stamp
    in full CCD view coordinates.

    Parameters
    ----------
    md : lsst.daf.base.PropertyList
        Metadata from one Guider CCD.
    camera : lsst.obs.lsst.LsstCam
        LsstCam object.

    Returns
    -------
    ccdViewBbox : lsst.geom.Box2I
        Bounding box for the ROI in CCD pixel coordinates.
    """

    # get need info from the stamp metadata
    roiCol = md["ROICOL"]
    roiRow = md["ROIROW"]
    roiCols = md["ROICOLS"]
    roiRows = md["ROIROWS"]

    # get detector and amp
    detector, ampName = get_detector_amp(md, camera)
    amp = detector[ampName]

    # get corner0 of the location for the ROI
    lct = LsstCameraTransforms(camera, detector.getName())
    corner0_CCDX, corner0_CCDY = lct.ampPixelToCcdPixel(roiCol, roiRow, ampName)
    corner0_CCDX = int(corner0_CCDX)
    corner0_CCDY = int(corner0_CCDY)

    # get opposite corner, corner2, of the ROI
    # depending on the RawFlipX,Y it could be either below or above corner0
    if amp.getRawFlipX():
        corner2_CCDX = corner0_CCDX - (roiCols - 1)
    else:
        corner2_CCDX = corner0_CCDX + (roiCols - 1)

    if amp.getRawFlipY():
        corner2_CCDY = corner0_CCDY - (roiRows - 1)
    else:
        corner2_CCDY = corner0_CCDY + (roiRows - 1)

    # now make the CCD view BBox, here we want the LowerLeft Point to be the
    # smallest x,y
    ll_x = min(corner0_CCDX, corner2_CCDX)
    ur_x = max(corner0_CCDX, corner2_CCDX)
    ll_y = min(corner0_CCDY, corner2_CCDY)
    ur_y = max(corner0_CCDY, corner2_CCDY)
    ll_CCD = Point2D(ll_x, ll_y)
    ur_CCD = Point2D(ur_x, ur_y)
    ccd_view_bbox = Box2I(Box2D(ll_CCD, ur_CCD))

    return ccd_view_bbox


def get_detector_amp(md, camera):
    """
    Unpack detector and amplifier information from metadata.

    Parameters
    ----------
    md : lsst.daf.base.PropertyList
        Metadata from one Guider CCD.
    camera : lsst.obs.lsst.LsstCam
        LsstCam object.

    Returns
    -------
    detector : lsst.afw.cameraGeom.Detector
        CCD Detector object.
    ampName : str
        Amplifier name, e.g., 'C00'.
    """
    raftBay = md["RAFTBAY"]
    ccdSlot = md["CCDSLOT"]

    segment = md["ROISEG"]
    ampName = "C" + segment[7:]
    detName = raftBay + "_" + ccdSlot
    detector = camera[detName]

    return detector, ampName


def convert_roi(roi, md, detector, ampName, camera, view="dvcs"):
    """
    Convert ROI image from raw amplifier view to CCD or DVCS views,
    or embed in full CCD image.

    Parameters
    ----------
    roi : np.ndarray
        ROI array (raw, amplifier coordinates).
    md : lsst.daf.base.PropertyList
        Metadata from one Guider CCD.
    detector : lsst.afw.cameraGeom.Detector
        CCD detector object.
    ampName : str
        Amplifier name, e.g., 'C00'.
    camera : lsst.obs.lsst.LsstCam
        LsstCam object.
    view : str, optional
        Desired output view: 'dvcs' (default), 'ccd', or 'ccdfull'.

    Returns
    -------
    imf : lsst.afw.image.ImageF
        Image in the requested view.

    Notes
    -----
    - 'dvcs': Oriented as in the focal plane.
    - 'ccd': Oriented as in CCD coordinates.
    - 'ccdfull': ROI embedded in a full CCD-sized image.
    """

    # convert image to ccd view
    roi_ccdview = amp_to_ccdview(roi, detector, ampName)

    # convert image to dvcs view (nb. np.rot90 works opposite to
    # afwMath.rotateImageBy90)
    roi_dvcsview = np.rot90(roi_ccdview, -detector.getOrientation().getNQuarter())

    # output ImageF
    if view == "dvcs":
        ny, nx = roi.shape
        imf = ImageF(nx, ny, 0.0)
        imf.array[:] = roi_dvcsview
    elif view == "ccd":
        ny, nx = roi.shape
        imf = ImageF(nx, ny, 0.0)
        imf.array[:] = roi_ccdview
    elif view == "ccdfull":
        # make the full CCD and place the ROI inside it
        nx, ny = detector.getBBox().getDimensions()
        imf = ImageF(nx, ny, 0.0)
        segment = md["ROISEG"]
        ampName = "C" + segment[7:]
        roiCol = md["ROICOL"]
        roiRow = md["ROIROW"]
        imf.array[:] = stamp_to_ccd(roi, imf.array[:], detector, camera, ampName, roiCol, roiRow)

    return imf


def focal_to_pixel(fpx: np.ndarray, fpy: np.ndarray, det: Detector) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert focal plane coordinates (mm, DVCS) to detector pixel coordinates.

    Parameters
    ----------
    fpx : np.ndarray
        Focal plane x coordinates (mm).
    fpy : np.ndarray
        Focal plane y coordinates (mm).
    det : lsst.afw.cameraGeom.Detector
        Detector of interest.

    Returns
    -------
    x : np.ndarray
        Pixel x coordinates.
    y : np.ndarray
        Pixel y coordinates.
    """
    tx = det.getTransform(cameraGeom.FOCAL_PLANE, cameraGeom.PIXELS)
    x, y = tx.getMapping().applyForward(np.vstack((fpx, fpy)))
    return x.ravel(), y.ravel()


def pixel_to_focal(x: np.ndarray, y: np.ndarray, det: Detector) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert pixel coordinates to focal plane coordinates (mm, DVCS).

    Parameters
    ----------
    x : np.ndarray
        Pixel x coordinates.
    y : np.ndarray
        Pixel y coordinates.
    det : lsst.afw.cameraGeom.Detector
        Detector of interest.

    Returns
    -------
    fpx : np.ndarray
        Focal plane x coordinates (mm).
    fpy : np.ndarray
        Focal plane y coordinates (mm).
    """
    tx = det.getTransform(cameraGeom.PIXELS, cameraGeom.FOCAL_PLANE)
    fpx, fpy = tx.getMapping().applyForward(np.vstack((x, y)))
    return fpx.ravel(), fpy.ravel()


def stamp_to_ccd(stamp, ccdimg, detector, camera, ampName, ampCol, ampRow):
    """
    Place ROI (stamp) into an existing full CCD array, all in CCD view.

    Parameters
    ----------
    stamp : np.ndarray
        Raw ROI array in amplifier coordinates.
    ccdimg : np.ndarray
        CCD image array in CCD coordinates (to be filled).
    detector : lsst.afw.cameraGeom.Detector
        Detector of interest.
    camera : lsst.afw.cameraGeom.Camera
        Camera object.
    ampName : str
        Amplifier name, e.g., 'C00'.
    ampCol : int
        Starting column number for ROI.
    ampRow : int
        Starting row number for ROI.

    Returns
    -------
    ccdimg : np.ndarray
        CCD image array with the ROI inserted in the appropriate location.

    Notes
    -----
    This function may print debugging information if placement fails.
    """

    stamp_ccd = amp_to_ccdview(stamp, detector, ampName)
    ampRows, ampCols = stamp_ccd.shape

    # get corner0 of the location for the ROI
    lct = LsstCameraTransforms(camera, detector.getName())
    corner0_CCDX, corner0_CCDY = lct.ampPixelToCcdPixel(ampCol, ampRow, ampName)
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
        ccdimg[
            min(corner0_CCDY, corner2_CCDY) : max(corner0_CCDY, corner2_CCDY),
            min(corner0_CCDX, corner2_CCDX) : max(corner0_CCDX, corner2_CCDX),
        ] = stamp_ccd
    except Exception:
        print(detector.getName(), ampName)
        print("Y: ", min(corner0_CCDY, corner2_CCDY), max(corner0_CCDY, corner2_CCDY))
        print("X: ", min(corner0_CCDX, corner2_CCDX), max(corner0_CCDX, corner2_CCDX))

    return ccdimg


def amp_to_ccdview(stamp, detector, ampName):
    """
    Convert a Guider ROI stamp image from amplifier view to CCD view.

    Parameters
    ----------
    stamp : np.ndarray
        Raw ROI array in amplifier coordinates.
    detector : lsst.afw.cameraGeom.Detector
        Detector of interest.
    ampName : str
        Amplifier name, e.g., 'C00'.

    Returns
    -------
    img : np.ndarray
        ROI image array flipped to be in CCD coordinate orientation.
    """
    amp = detector[ampName]
    img = stamp.copy()
    if amp.getRawFlipX():
        img = np.fliplr(img)
    if amp.getRawFlipY():
        img = np.flipud(img)
    return img


def convert_to_focal_plane(xccd: np.ndarray, yccd: np.ndarray, detNum: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert from CCD pixel coordinates to focal plane coordinates (mm).

    Parameters
    ----------
    xccd : np.ndarray
        Array of x CCD pixel coordinates.
    yccd : np.ndarray
        Array of y CCD pixel coordinates.
    detNum : int
        Detector number (e.g., 189 for R22_S11).

    Returns
    -------
    xfp : np.ndarray
        Focal plane x coordinates (mm).
    yfp : np.ndarray
        Focal plane y coordinates (mm).
    """
    if len(xccd) > 0:
        detector = LsstCam.getCamera()[detNum]
        # Convert the star positions to focal plane coordinates
        xfp, yfp = pixel_to_focal(xccd, yccd, detector)
    else:
        xfp, yfp = np.array([]), np.array([])
    return xfp, yfp


def convert_to_altaz(
    xccd: np.ndarray, yccd: np.ndarray, wcs: Any, obsTime: Time
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert CCD pixel coordinates to altitude and azimuth (degrees).

    Parameters
    ----------
    xccd : np.ndarray
        Array of x CCD pixel coordinates.
    yccd : np.ndarray
        Array of y CCD pixel coordinates.
    wcs : lsst.afw.image.Wcs
        WCS object for the guider detector.
    obsTime : astropy.time.Time
        Observation time.

    Returns
    -------
    alt : np.ndarray
        Array of altitude coordinates (degrees).
    az : np.ndarray
        Array of azimuth coordinates (degrees).
    """
    if len(xccd) > 0:
        az, alt = convert_pixels_to_altaz(wcs, obsTime, xccd, yccd)
    else:
        alt, az = np.array([]), np.array([])

    return alt, az


def convert_roi_to_ccd(
    xroi: np.ndarray, yroi: np.ndarray, guiderData: GuiderData, guiderName: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert ROI coordinates to CCD pixel coordinates.

    Parameters
    ----------
    xroi : np.ndarray
        Array of x ROI coordinates (pixels within the ROI).
    yroi : np.ndarray
        Array of y ROI coordinates (pixels within the ROI).
    guiderData : GuiderData
        GuiderData object containing the guider datasets and view information.
    guiderName : str
        Name of the guider (e.g., 'R44_SG0').

    Returns
    -------
    xccd : np.ndarray
        Array of x CCD pixel coordinates.
    yccd : np.ndarray
        Array of y CCD pixel coordinates.

    Raises
    ------
    ValueError
        If the view in GuiderData is not supported.
    """
    view = guiderData.view
    stamps = guiderData.datasets[guiderName]

    if np.isscalar(xroi):
        xroi = np.array([xroi])
    if np.isscalar(yroi):
        yroi = np.array([yroi])

    if len(xroi) == 0:
        return np.array([]), np.array([])

    box, _, roi2ccd = stamps.getArchiveElements()[0]
    if view == "ccd":
        # convert roi coords to ccd coords
        # by adding the lower left corner of the box
        lower_left_corner = box.getMin()
        xmin, ymin = lower_left_corner.getX(), lower_left_corner.getY()
        xccd, yccd = xroi + xmin, yroi + ymin

    elif view == "dvcs":
        # convert roi coords to ccd coords using the roi2ccd transform
        # roi2ccd is an AffineTransform that converts
        # from roi coords to ccd coords
        xccd, yccd = roi2ccd(xroi, yroi)
    else:
        raise ValueError(f"Unsupported view '{view}' in convert_roi_to_ccd" "must be 'ccd' or 'dvcs'")

    return xccd, yccd


def convert_ccd_to_roi(
    xccd: np.ndarray, yccd: np.ndarray, guiderData: GuiderData, guiderName: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert CCD pixel coordinates to ROI pixel coordinates.

    Parameters
    ----------
    xccd : np.ndarray
        Array of x CCD pixel coordinates.
    yccd : np.ndarray
        Array of y CCD pixel coordinates.
    guiderData : GuiderData
        GuiderData object containing the guider datasets and view information.
    guiderName : str
        Name of the guider (e.g., 'R44_SG0').

    Returns
    -------
    xroi : np.ndarray
        Array of x ROI pixel coordinates.
    yroi : np.ndarray
        Array of y ROI pixel coordinates.

    Raises
    ------
    ValueError
        If the view in GuiderData is not supported.
    """
    view = guiderData.view
    stamps = guiderData.datasets[guiderName]

    if len(xccd) == 0:
        return np.array([]), np.array([])

    box, ccd2dvcs, _ = stamps.getArchiveElements()[0]

    if view == "ccd":
        lower_left_corner = box.getMin()
        xmin, ymin = lower_left_corner.getX(), lower_left_corner.getY()
        xroi, yroi = xccd - xmin, yccd - ymin

    elif view == "dvcs":
        xroi, yroi = ccd2dvcs(xccd, yccd)

    else:
        raise ValueError(f"Unsupported view '{view}' in convert_ccd_to_roi", "must be 'ccd' or 'dvcs'")

    return xroi, yroi


def convert_ccd_to_dvcs(
    xccd: np.ndarray, yccd: np.ndarray, guiderData: GuiderData, guiderName: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert CCD pixel coordinates to DVCS (focal plane)
    coordinates (pixels or mm).

    Parameters
    ----------
    xccd : np.ndarray
        Array of x CCD pixel coordinates.
    yccd : np.ndarray
        Array of y CCD pixel coordinates.
    guiderData : GuiderData
        GuiderData object containing the guider datasets and view information.
    guiderName : str
        Name of the guider (e.g., 'R44_SG0').

    Returns
    -------
    xdvcs : np.ndarray
        Array of x DVCS focal plane coordinates.
    ydvcs : np.ndarray
        Array of y DVCS focal plane coordinates.
    """
    stamps = guiderData.datasets[guiderName]

    if len(xccd) == 0:
        return np.array([]), np.array([])

    box, ccd2dvcs, _ = stamps.getArchiveElements()[0]
    xdvcs, ydvcs = ccd2dvcs(xccd, yccd)
    return xdvcs, ydvcs
