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

import logging

import numpy as np

from lsst.afw.geom.ellipses import Quadrupole
from lsst.geom import LinearTransform, radians

__all__ = [
    "convertE1E2",
    "convertIxxIyyIxy"
]

logger = logging.getLogger(__name__)

def convertE1E2(
    e1: list,
    e2: list,
    physical_rotator_angle: float,
    input_coordinate_system: str,
    output_coordinate_system: str,
    sky_angle: float | None=None,
) -> list:
    """Transform ellipticities between coordinate systems for LSSTCam images.
    
    Given an input e1, e2 with a specified coordinate system and rotator 
    angle (and sky angle if needed), convert the ellipticity between CCS, 
    DVCS, OCS, Alt/Az, equatorial, and North-West (NW) coordinate systems.
    
    See https://smtn-019.lsst.io/#on-sky-and-hardware-rotation-angles for more details.
    
    Parameters
    ----------
    e1 : `list`
        A list of input e1 ellipticities.
    e2 : `list`
        A list of input e2 ellipticities.
    physical_rotator_angle : `float`
        Camera rotator angle in degrees. Equivalent to the rotator telescope position 
        (RotTelPos or rtp)
    input_coordinate_system : `string`
        The input coordinate system, which is the system that e1 and e2 are in.
        Options are: 'altaz', 'ccs', 'dvcs', 'equatorial', 'nw' 'ocs'.
    output_coordinate_system : `string`
        The input coordinate system, which is the system that e1 and e2 are in.
        Options are: 'altaz', 'ccs', 'dvcs', 'equatorial', 'nw', 'ocs'.
    sky_angle : `float` | None
        Optional: Sky angle in degrees. This is the orientation of the +𝑌DVCS axis (projected on the sky) 
        measured east of north in the International Celestial Reference Frame (ICRF).
        Also called RotSkyPos or given by the boresightRotAngle if the rotType=Sky 
        (which is true for all FBS observations and many engineering observations).
        Only needed if the coordinate system includes equatorial or nw.
        
    Returns
    -------
    e1_out : `list`
        A list of output e1 ellipticities.
    e2_out : `list`
        A list of output e2 ellipticities.
    """
    
    allowed_coordinates = ['altaz', 'ccs', 'dvcs', 'ocs', 'equatorial', 'nw']
    
    if input_coordinate_system not in allowed_coordinates:
        raise ValueError(f"Input coordinate system must be in {allowed_coordinates}")
    if output_coordinate_system not in allowed_coordinates:
        raise ValueError(f"Output coordinate system must be in {allowed_coordinates}")
    if (("nw" or "equatorial") in [input_coordinate_system, output_coordinate_system]) and (sky_angle is None):
        raise ValueError(f"Sky angle must be provided when using nw or equatorial coordinate systems")
    
    if input_coordinate_system == output_coordinate_system:
        return e1, e2
    
    # convert e1 and e2 into Ixx, Iyy, Ixy components 
    # letting the trace T (Ixx + Iyy) be 1.
    Ixx = [(1 + e1_input)/2 for e1_input in e1]
    Iyy = [(1 - e1_input)/2 for e1_input in e1]
    Ixy = [(e2_input)/2 for e2_input in e2]
    
    if input_coordinate_system == "nw":
        # if input coordinate system is "nw", immediately convert to dvcs
        _, _, _, e1_out, e2_out = convertIxxIyyIxy(Ixx, Iyy, Ixy, -1*sky_angle)
        if output_coordinate_system == 'dvcs':
            return e1_out, e2_out
        else:
            input_coordinate_system = "dvcs"
    
    if input_coordinate_system == 'dvcs':
        if output_coordinate_system == 'ccs':
            # for ccs you swap x and y relative to dvcs
            _, _, _, e1_out, e2_out = convertIxxIyyIxy(Iyy, Ixx, Ixy, 0)
            return e1_out, e2_out
        elif output_coordinate_system in ['ocs', 'altaz']:
            # for ocs and altaz you swap x and y relative to dvcs
            _, _, _, e1_out, e2_out = convertIxxIyyIxy(Iyy, Ixx, Ixy, -1*physical_rotator_angle)
            if output_coordinate_system == 'ocs':
                return e1_out, e2_out
            else:
                return e1_out, [-1*ellipticity for ellipticity in e2_out]
        elif output_coordinate_system == 'nw':
            # for NW coordinate system, you don't need to transpose x and y
            _, _, _, e1_out, e2_out = convertIxxIyyIxy(Ixx, Iyy, Ixy, sky_angle)
            return e1_out, e2_out

    if input_coordinate_system == 'altaz':
        if output_coordinate_system == 'ocs':
            return e1, [-1*ellipticity for ellipticity in e2]
        elif output_coordinate_system in ['ccs', 'dvcs', 'nw']:
            _, _, _, e1_out, e2_out = convertIxxIyyIxy(Ixx, Iyy, [-1*xy for xy in Ixy], physical_rotator_angle)
            if output_coordinate_system == 'ccs':
                return e1_out, e2_out
            else:
                e1_dvcs = [-1* ellipticity for ellipticity in e1_out]
                if output_coordinate_system == 'dvcs':
                    return e1_dvcs, e2_out
                else:
                    # rotate into NW using sky angle
                    Ixx_dvcs = [(1 + dvcs)/2 for dvcs in e1_dvcs]
                    Iyy_dvcs = [(1 - dvcs)/2 for dvcs in e1_dvcs]
                    Ixy_dvcs = [(e2_input)/2 for e2_input in e2_out]
                    _, _, _, e1_out, e2_out = convertIxxIyyIxy(Ixx_dvcs, Iyy_dvcs, Ixy_dvcs, sky_angle)
                    return e1_out, e2_out
                    
            
    elif input_coordinate_system == 'ccs':
        if output_coordinate_system in ['altaz', 'ocs']:
            _, _, _, e1_out, e2_out = convertIxxIyyIxy(Ixx, Iyy, Ixy, -1*physical_rotator_angle)
            if output_coordinate_system == 'ocs':
                return e1_out, e2_out
            else:
                return e1_out, [-1*ellipticity for ellipticity in e2_out]
        elif output_coordinate_system in ['dvcs', 'nw']:
            e1_dvcs = [-1* ellipticity for ellipticity in e1_out]
            if output_coordinate_system == 'dvcs':
                return e1_dvcs, e2_out
            else:
                # rotate into NW using sky angle
                Ixx_dvcs = [(1 + dvcs)/2 for dvcs in e1_dvcs]
                Iyy_dvcs = [(1 - dvcs)/2 for dvcs in e1_dvcs]
                Ixy_dvcs = [(e2_input)/2 for e2_input in e2_out]
                _, _, _, e1_out, e2_out = convertIxxIyyIxy(Ixx_dvcs, Iyy_dvcs, Ixy_dvcs, sky_angle)
                return e1_out, e2_out
        
    elif input_coordinate_system == 'ocs':
        if output_coordinate_system == 'altaz':
            return e1, [-1*ellipticity for ellipticity in e2]
        elif output_coordinate_system in ['ccs', 'dvcs', 'nw']:
            _, _, _, e1_out, e2_out = convertIxxIyyIxy(Ixx, Iyy, Ixy, physical_rotator_angle)
            if output_coordinate_system == 'ccs':
                return e1_out, e2_out
            else:
                e1_dvcs = [-1* ellipticity for ellipticity in e1_out]
                if output_coordinate_system == 'dvcs':
                    return e1_dvcs, e2_out
                else:
                    # rotate into NW using sky angle
                    Ixx_dvcs = [(1 + dvcs)/2 for dvcs in e1_dvcs]
                    Iyy_dvcs = [(1 - dvcs)/2 for dvcs in e1_dvcs]
                    Ixy_dvcs = [(e2_input)/2 for e2_input in e2_out]
                    _, _, _, e1_out, e2_out = convertIxxIyyIxy(Ixx_dvcs, Iyy_dvcs, Ixy_dvcs, sky_angle)
                    return e1_out, e2_out
            
    return [], []

def convertIxxIyyIxy(
    Ixx: list,
    Iyy: list,
    Ixy: list,
    rotation_angle: float,
) -> list:    
    """Rotate Ixx, Iyy, Ixy by a given rotation angle.
    
    Parameters
    ----------
    Ixx : `list`
        A list of input Ixx quantities.
    Iyy : `list`
        A list of input Iyy quantities.
    Ixy : `list`
        A list of input Ixy quantities.
    rotation_angle : `float`
        Angle of rotation in degrees.
        
    Returns
    -------
    Ixx_out : `list`
        A list of output Ixx quantities.
    Iyy_out : `list`
        A list of output Iyy quantities.
    Ixy_out : `list`
        A list of output Ixy quantities.
    e1_out : `list`
        A list of output e1 quantities
    e2_out : `list
        A list of output e2 quantitites
    """
    
    # Build rotation matrix
    rotation_angle_rad = rotation_angle * np.pi / 180
    srot, crot = np.sin(rotation_angle_rad), np.cos(rotation_angle_rad)
    rot = np.array([[crot, -srot], [srot, crot]]) @ np.array([[0, 1], [1, 0]]) @ np.array([[-1, 0], [0, 1]])
    
    # perform the rotations
    transform = LinearTransform(rot)
    rotShapes = []
    for xx, yy, xy in [*zip(Ixx, Iyy, Ixy)]:
        shape = Quadrupole(xx, yy, xy)
        rotShape = shape.transform(transform)
        rotShapes.append(rotShape)
    Ixx_out = [sh.getIxx() for sh in rotShapes]
    Iyy_out = [sh.getIyy() for sh in rotShapes]
    Ixy_out = [sh.getIxy() for sh in rotShapes]
    
    T_out = [xx + yy for xx, yy in [*zip(Ixx_out, Iyy_out)]]
    
    e1_out = [(xx + yy)/T for xx, yy, T in [*zip(Ixx_out, Iyy_out, T_out)]]
    e2_out = [2*xy/T for xy, T in [*zip(Ixy_out, T_out)]]
    
    return Ixx_out, Iyy_out, Ixy_out, e1_out, e2_out