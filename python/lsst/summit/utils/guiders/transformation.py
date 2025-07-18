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
    Map detector-pixel → ICRS RA/Dec (radians).
    """
    return wcs.pixelToSkyArray(x_flat, y_flat)


def convert_pixels_to_altaz(
    wcs: Any, time: Time, x_pix: np.ndarray, y_pix: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized conversion of detector-pixel coords → AltAz (deg).

    Parameters
    ----------
    x_pix : `np.ndarray`
        Same-shaped arrays of pixel coordinates.
    y_pix : `np.ndarray`
        Same-shaped arrays of pixel coordinates.

    Returns
    -------
    az, alt : np.ndarray
        Arrays of the same shape as x_pix/y_pix giving Az and Alt in
        degrees.
    """

    # 1) make sure we have numpy arrays, remember their shape
    x_arr = np.asarray(x_pix)
    y_arr = np.asarray(y_pix)
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
    wcs: Any, time: Time, detector: Detector, x_focal: np.ndarray, y_focal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map focal-plane mm (or any XY) → Alt/Az by going
    focal → pixel → AltAz.
    """
    # You must provide focal_to_pixel or similar
    x_pix, y_pix = focal_to_pixel(x_focal, y_focal, detector)
    return convert_pixels_to_altaz(wcs, time, x_pix, y_pix)


# Aaron's code to make transformations from ROI coordinates to Focal Plane and
# sky coordinates.
def mk_rot(det_nquarter: int, direction: int = 1) -> AffineTransform:
    """
    Make rotation Transform

    Parameters
    ----------
    det_nquarter: int
        Number of 90 degree CCW rotations

    direction: int
        =1 for forward, =-1 for reverse

    Returns
    -------
    rotation: AffineTransformation
        rotation transformation

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

    nq = np.mod(det_nquarter, 4)
    if direction == 1:
        rotation = AffineTransform(rot[nq])
    elif direction == -1:
        rotation = AffineTransform(irot[nq])
    else:
        raise ValueError(f"direction must be either +/- 1, got {direction}")
    return rotation


def mk_ccd_to_dvcs(
    bbox_ccd: Box2I,
    det_nquarter: int,
) -> tuple[AffineTransform, AffineTransform]:
    """
    Make transformations suitable for Guider stamps, to go from a view of the
    stamp in CCD pixel coordinates (ie. with 0,0 in the Lower Left of C00) to
    DVCS view (ie. orientated in the Focal Plane correctly in the DVCS, with
    0,0 at the Lower Left of the ROI)

    Parameters
    ----------
    bbox_ccd : Box2I
        Bounding Box for the ROI in CCD pixel coordinates
    det_nquarter : int
        The number of 90degree quarters needed to rotate from CCD view to DVCS
        view

    Returns
    -------
    forwards : AffineTransform
        A transform from CCD pixel coordinates in the CCD view to Stamp pixel
        coordinates in DVCS view

    backwards : AffineTransform
        A transform from Stamp pixel coordinates in DVCS view to CCD pixel
        coordinates in the CCD view

    Example
    -------

    detector = camera[189] bbox_ccd = Box2I(Point2I(300,600),Point2I(350,650))
    ft,bt = mk_ccd_to_dvcs(box_ccd,detector.getOrientation().getNQuarter())

    # given a sky coordinate, find point on the stamp in DVCS view pt_ccd =
    wcs.skyToPixel(sky_coord) pt_stamp_dvcs = ft(pt_ccd)

    # given a point on the stamp in DVCS view, find the skyCoord pt_ccd =
    bt(pt_stamp_dvcs) sky_coord = wcs.pixelToSky(pt_ccd)

    """
    # build integer rotation matrices
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

    # number of 90deg CCW rotations
    nq = np.mod(det_nquarter, 4)

    # get LL,Size of the CCD view BBox
    llpt_ccd = Extent2D(bbox_ccd.getCorners()[0])
    nx, ny = bbox_ccd.getDimensions()

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
    Make bounding box for a Guider stamp in the full CCD view

    Parameters
    ----------
    md : lsst.daf.base.PropertyList
        Metadata from one Guider CCD

    camera: lsst.obs.lsst.LsstCam
        LsstCam object

    Returns
    -------
    ccd_view_bbox : Box2I
        Bounding box for the stamps inside the full CCD in CCD view
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
    Unpack Detector and Amplifier info.

    Parameters
    ----------
    md : lsst.daf.base.PropertyList
        Metadata from one Guider CCD

    camera: lsst.obs.lsst.LsstCam
        LsstCam object

    Returns
    -------
    detector : lsst.afw.cameraGeom.Detector
        CCD Detector

    ampName : String
        Amplifier name, eg. C00
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
    convert ROI image from raw to CCD or DVCS views

    Parameters
    ----------
    roi : numpy.ndarray
        ROI array

    md : lsst.daf.base.PropertyList
        Metadata from one Guider CCD

    detector : lsst.afw.cameraGeom.Detector
        CCD detector object

    ampName : str
        Amplifier Name, eg. C00

    camera : lsst.obs.lsst.LsstCam
        LsstCam object

    view : string
        Desired view for ROI. Default is 'dvcs', other option is 'ccd'

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


def stamp_to_ccd(stamp, ccdimg, detector, camera, ampName, ampCol, ampRow):
    """
    Place ROI (stamp) into existing full CCD array, all in CCD view
    (Should rewrite to use mk_roi_bboxes(md,camera))

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
    Comvert a Guider ROI stamp image from Amp view to CCD view

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
    amp = detector[ampName]
    img = stamp.copy()
    if amp.getRawFlipX():
        img = np.fliplr(img)
    if amp.getRawFlipY():
        img = np.flipud(img)
    return img


def convert_to_focal_plane(xccd: np.ndarray, yccd: np.ndarray, detNum: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert from ccd pixels coordinates to focal plane coordinates.

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
    xfp, yfp : np.ndarray
        Arrays of x and y focal plane coordinates (mm).

    """
    if len(xccd) > 0:
        detector = LsstCam.getCamera()[detNum]
        # Convert the star positions to focal plane coordinates
        xfp, yfp = pixel_to_focal(xccd, yccd, detector)
    else:
        xfp, yfp = np.array([]), np.array([])
    return xfp, yfp


def convert_to_altaz(
    xccd: np.ndarray, yccd: np.ndarray, wcs: Any, obs_time: Time
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert the ccd pixel coordinates to altaz coordinates.

    Parameters
    ----------
    xccd : np.ndarray
        Array of x CCD pixel coordinates.
    yccd : np.ndarray
        Array of y CCD pixel coordinates.
    wcs : lsst.afw.image.Wcs
        WCS object for the guider detector.
    obs_time : astropy.time.Time
        Observation time.

    Returns
    -------
    alt, az : np.ndarray
        Arrays of alt and az coordinates (degrees).
    """
    if len(xccd) > 0:
        az, alt = convert_pixels_to_altaz(wcs, obs_time, xccd, yccd)
    else:
        alt, az = np.array([]), np.array([])

    return alt, az


def convert_roi_to_ccd(
    xroi: np.ndarray, yroi: np.ndarray, guiderData: GuiderData, guiderName: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert roi coordinates to CCD pixel coordinates.

    Parameters
    ----------
    xroi : np.ndarray
        Array of x roi coordinates (pixels within the ROI).
    yroi : np.ndarray
        Array of y roi coordinates (pixels within the ROI).
    guiderData : GuiderData
        GuiderData object containing the guider datasets and view information.
    guiderName : str
        Name of the guider (e.g., 'R44_SG0', etc.)

    Returns
    -------
    xccd, yccd : np.ndarray
        Arrays of CCD pixel coordinates.
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
        raise ValueError(f"Unsupported view '{view}' in convert_roi_to_ccd", "must be 'ccd' or 'dvcs'")

    return xccd, yccd


def convert_ccd_to_roi(
    xccd: np.ndarray, yccd: np.ndarray, guiderData: GuiderData, guiderName: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert CCD pixel coordinates to ROI coordinates.

    Parameters
    ----------
    xccd : np.ndarray
        Array of x CCD pixel coordinates.
    yccd : np.ndarray
        Array of y CCD pixel coordinates.
    guiderData : GuiderData
        GuiderData object containing the guider datasets and view information.
    guiderName : str
        Name of the guider (e.g., 'R44_SG0', etc.)

    Returns
    -------
    xroi, yroi : np.ndarray
        Arrays of ROI pixel coordinates.
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
    Convert CCD pixel coordinates to DVCS focal plane coordinates (mm).

    Parameters
    ----------
    xccd : np.ndarray
        Array of x CCD pixel coordinates.
    yccd : np.ndarray
        Array of y CCD pixel coordinates.
    guiderData : GuiderData
        GuiderData object containing the guider datasets and view information.
    guiderName : str
        Name of the guider (e.g., 'R44_SG0', etc.)

    Returns
    -------
    xdvcs, ydvcs : np.ndarray
        Arrays of x and y DVCS focal plane coordinates (pixel).
    """
    stamps = guiderData.datasets[guiderName]

    if len(xccd) == 0:
        return np.array([]), np.array([])

    box, ccd2dvcs, _ = stamps.getArchiveElements()[0]
    xdvcs, ydvcs = ccd2dvcs(xccd, yccd)
    return xdvcs, ydvcs
