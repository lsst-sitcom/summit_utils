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

import numpy as np

import matplotlib.pyplot as plt
from astropy.coordinates import Angle
import astropy.units as u

from lsst.obs.lsst.translators.latiss import AUXTEL_LOCATION
from .. import quickSmooth


# TODO: Add some of Craig's nice overlay stuff here

def plot(exp, icSrc=None, filteredSources=None, saveAs=None,
         clipMin=1, clipMax=1000000):
    """Plot an exposure, overlaying the selected sources and compass arrows.

    Plots the exposure on a logNorm scale, with the brightest sources, as
    selected by the configuration, overlaid with an x. Plots compass arrows
    for both north/east and az/el. Optionally saves the output to a file if
    ``saveAs`` is supplied.

    Parameters
    ----------
    exp : `lsst.afw.image.Exposure`
        The exposure to get the astrometry for.
    icSrc : `lsst.afw.table.SourceCatalog`
        The source catalog for the exposure.
    filteredSources : `lsst.afw.table.SourceCatalog`, optional
        The filtered source catalog. If supplied, shows which sources were
        selected.
    saveAs : `str`, optional
        Saves the plot to this filename if specified.
    clipMin : `float`
        Clip values in the image below this value to ``clipMin``.
    clipMax : `float`
        Clip values in the image above this value to ``clipMax``.
    """
    plt.figure(figsize=(16, 16))
    arr = exp.image.array
    arr = np.clip(arr, clipMin, clipMax)
    arr = quickSmooth(arr)  # smooth slightly to help visualize
    plt.imshow(np.arcsinh(arr)/10,
               interpolation='None', cmap='gray', origin='lower')

    height, width = exp.image.array.shape
    leftFraction = .15  # fraction into the image to start the N/E compass
    rightFraction = .225  # fraction into the image to start the az/el compass
    fontsize = 20  # for the compass labels
    compassSize = 500
    textDistance = 650
    compassCenter = (leftFraction*width, leftFraction*height)
    compassAzElCent = ((1 - rightFraction)*width, rightFraction*height)

    vi = exp.getInfo().getVisitInfo()
    az, _ = vi.boresightAzAlt
    _, dec = vi.boresightRaDec
    rotpa = vi.boresightRotAngle

    az = Angle(az.asDegrees(), u.deg)
    dec = Angle(dec.asDegrees(), u.deg)
    rotpa = Angle(rotpa.asDegrees(), u.deg)

    if icSrc:
        plt.scatter(icSrc['base_SdssCentroid_x'], icSrc['base_SdssCentroid_y'], color='red', marker='x')
    if filteredSources:
        markerStyle = dict(marker='o', linestyle='', markersize=20, linewidth=10, color='green',
                           markeredgecolor='green', fillstyle='none')
        plt.plot(filteredSources['base_SdssCentroid_x'],
                 filteredSources['base_SdssCentroid_y'],
                 **markerStyle)
    plt.arrow(compassCenter[0],
              compassCenter[1],
              -compassSize*np.sin(rotpa),
              compassSize*np.cos(rotpa),
              color='green', width=20)
    plt.text(compassCenter[0] - textDistance*np.sin(rotpa),
             compassCenter[1] + textDistance*np.cos(rotpa),
             'N',
             color='green', fontsize=fontsize, weight='bold')
    plt.arrow(compassCenter[0],
              compassCenter[1],
              compassSize*np.cos(rotpa),
              compassSize*np.sin(rotpa),
              color='green', width=20)
    plt.text(compassCenter[0] + textDistance*np.cos(rotpa),
             compassCenter[1] + textDistance*np.sin(rotpa),
             'E',
             color='green', fontsize=fontsize, weight='bold')

    sinTheta = np.cos(AUXTEL_LOCATION.lat)/np.cos(dec)*np.sin(az)
    theta = Angle(np.arcsin(sinTheta))
    rotAzEl = rotpa - theta - Angle(90.0 * u.deg)
    plt.arrow(compassAzElCent[0],
              compassAzElCent[1],
              -compassSize*np.sin(rotAzEl),
              compassSize*np.cos(rotAzEl),
              color='cyan', width=20)
    plt.text(compassAzElCent[0] - textDistance*np.sin(rotAzEl),
             compassAzElCent[1] + textDistance*np.cos(rotAzEl),
             'EL',
             color='cyan', fontsize=fontsize, weight='bold')
    plt.arrow(compassAzElCent[0],
              compassAzElCent[1],
              compassSize*np.cos(rotAzEl),
              compassSize*np.sin(rotAzEl),
              color='cyan', width=20)
    plt.text(compassAzElCent[0] + textDistance*np.cos(rotAzEl),
             compassAzElCent[1] + textDistance*np.sin(rotAzEl),
             'AZ',
             color='cyan', fontsize=fontsize, weight='bold')

    plt.ylim(0, height)
    plt.tight_layout()

    if saveAs:
        plt.savefig(saveAs)
    plt.show()