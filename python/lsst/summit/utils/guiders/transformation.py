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
from lsst.geom import Point2D, Box2D, Box2I
from lsst.geom import AffineTransform
from lsst.afw.image import ImageF

# Aaron's code to make transformations from ROI coordinates to Focal Plane and sky coordinates.

def mk_rot(det_nquarter):
    """ 
    Make rotation Transform
    """
    rot = {}
    rot[0] = np.array([[1.,0.],[0.,1.]])
    rot[1] = np.array([[0.,-1],[1.,0.]])
    rot[2] = np.array([[-1.,0.],[0.,-1.]])
    rot[3] = np.array([[0.,1.],[-1.,0.]])

    nq = np.mod(det_nquarter,4)
    frotation = AffineTransform(rot[nq])
    return frotation

def mk_ccd_to_dvcs(llpt_ccd,det_nquarter):
    """
    Make transformations suitable for Guider stamps, to go from
    a view of the stamp in CCD pixel coordinates (ie. with 0,0 in the Lower Left) 
    to DVCS view (ie. orientated in the Focal Plane correctly in the DVCS) 

    Parameters
    ----------
    llpt_ccd : Extent2I
        The Lower Left point in the CCD pixel coordinates
    det_nquarter : int
        The number of 90degree quarters needed to rotate from CCD view to DVCS view

    Returns
    -------
    forwards : AffineTransform
        A transform from CCD pixel coordinates in the CCD view to Stamp pixel coordinates in DVCS view

    backwards : AffineTransform
        A transform from Stamp pixel coordinates in DVCS view to CCD pixel coordinates in the CCD view

    Example
    -------

    detector = camera[189]
    llpt_ccd = Extent2D(100.,200.)
    ft,bt = mk_ccd_to_dvcs(llpt_ccd,detector.getOrientation().getNQuarter())

    # given a sky coordinate, find point on the stamp in DVCS view 
    pt_ccd = wcs.skyToPixel(sky_coord)
    pt_stamp_dvcs = ft(pt_ccd)

    # given a point on the stamp in DVCS view, find the skyCoord
    pt_ccd = bt(pt_stamp_dvcs)
    sky_coord = wcs.pixelToSky(pt_ccd)
    
    """
    rot = {}
    rot[0] = np.array([[1.,0.],[0.,1.]])
    rot[1] = np.array([[0.,-1],[1.,0.]])
    rot[2] = np.array([[-1.,0.],[0.,-1.]])
    rot[3] = np.array([[0.,1.],[-1.,0.]])

    irot = {}
    irot[0] = rot[0].transpose()
    irot[1] = rot[1].transpose()
    irot[2] = rot[2].transpose()
    irot[3] = rot[3].transpose()

    nq = np.mod(det_nquarter,4)
    
    ftranslation = AffineTransform(-llpt_ccd)
    frotation = AffineTransform(rot[nq])
    forwards = frotation*ftranslation # ordering is second*first

    btranslation = AffineTransform(llpt_ccd)
    brotation = AffineTransform(irot[nq])
    backwards = btranslation*brotation

    return forwards,backwards


def mk_roi_bboxes(md,camera):
    """
    Make bounding boxes for a Guider stamp in CCD and DVCS view

    Parameters
    ----------
    md : 
        Metadata from one Guider CCD

    camera:
        LsstCam 

    Returns
    -------
    ccd_view_bbox : Box2I
        Bounding box for the stamps in the full CCD in CCD view

    dvcs_view_bbox : Box2I
        Bounding box for the stamps in the full CCD but rotated to DVCS view
    """

    # get need info from the stamp metadata
    roiCol    = md['ROICOL']
    roiRow    = md['ROIROW']
    roiCols   = md['ROICOLS']
    roiRows   = md['ROIROWS']

    # get detector and amp
    detector,ampName = get_detector_amp(md,camera)
    amp = detector[ampName]

    # get corner0 of the location for the ROI
    lct = LsstCameraTransforms(camera,detector.getName())
    corner0_CCDX,corner0_CCDY = lct.ampPixelToCcdPixel(roiCol,roiRow,ampName)
    corner0_CCDX = int(corner0_CCDX)
    corner0_CCDY = int(corner0_CCDY)

    # get opposite corner, corner2, of the ROI
    # depending on the RawFlipX,Y it could be either below or above corner0
    if amp.getRawFlipX():
        corner2_CCDX = corner0_CCDX - roiCols
    else:
        corner2_CCDX = corner0_CCDX + roiCols

    if amp.getRawFlipY():
        corner2_CCDY = corner0_CCDY - roiRows
    else:
        corner2_CCDY = corner0_CCDY + roiRows

    # now make the CCD view BBox, here we want the LowerLeft Point to be the 
    # smallest x,y
    ll_x = min(corner0_CCDX,corner2_CCDX)
    ur_x = max(corner0_CCDX,corner2_CCDX)
    ll_y = min(corner0_CCDY,corner2_CCDY)
    ur_y = max(corner0_CCDY,corner2_CCDY)
    ll_CCD = Point2D(ll_x,ll_y)
    ur_CCD = Point2D(ur_x,ur_y)
    ccd_view_bbox = Box2I(Box2D(ll_CCD,ur_CCD))
    
    # next make the DVCS view BBox, by rotating by NQuarters
    frot = mk_rot(detector.getOrientation().getNQuarter())
    ll_rot = frot(ll_CCD)
    ur_rot = frot(ur_CCD)
    ll_x_dvcs = min(ll_rot.getX(),ur_rot.getX())
    ur_x_dvcs = max(ll_rot.getX(),ur_rot.getX())
    ll_y_dvcs = min(ll_rot.getY(),ur_rot.getY())
    ur_y_dvcs = max(ll_rot.getY(),ur_rot.getY())
    ll_DVCS = Point2D(ll_x_dvcs,ur_x_dvcs)
    ur_DVCS = Point2D(ll_y_dvcs,ur_y_dvcs)
    dvcs_view_bbox = Box2I(Box2D(ll_DVCS,ur_DVCS))
    
    return ccd_view_bbox,dvcs_view_bbox

def get_detector_amp(md,camera):
    """
    Unpack Detector and Amplifier info. 
    
    """
    raftBay = md['RAFTBAY']
    ccdSlot = md['CCDSLOT']
    obsId = md['OBSID']
    dayObs = int(obsId[5:13])
    seqNum = int(obsId[14:])
            
    segment = md['ROISEG']
    ampName = 'C'+ segment[7:]
    detName = raftBay + '_' + ccdSlot
    detector = camera[detName]

    return detector,ampName

def convert_roi(roi,detector,ampName,camera,view='dvcs'):
    """
    convert roi to CCD or DVCS views

    """

    # convert image to ccd view
    roi_ccdview = amp_to_ccdview(roi,detector,ampName)

    # convert image to dvcs view
    roi_dvcsview = np.rot90(roi_ccdview,-detector.getOrientation().getNQuarter())

    # output ImageF
    ny,nx = roi.shape
    imf = ImageF(nx,ny,0.)

    if view=='dvcs':
        imf.array[:] = roi_dvcsview
    elif view=='ccd':
        imf.array[:] = roi_ccdview

    return imf
    
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
