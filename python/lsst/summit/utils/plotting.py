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

from lsst.afw.detection import FootprintSet
from lsst.afw.table import SourceCatalog
import lsst.geom as geom
import lsst.summit.utils as utils
import lsst.afw.image as Image

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import astropy.visualization as vis


def makePlot(inData, imageType='image', ax=None, centroids=None, compass=False, coordGrid=False,
             stretch=None, percentile=99., cmap='gray_r', anchorPix=250,
             legend=True, savePlotAs=None):

    """Make a plot.

    Parameters
    ----------
    inData : `numpy.array`, `lsst.afw.image.exposure`, or
             `lsst.afw.image.Image`
        The input data.
    imageType : `str`, optional
        If input data is an exposure, plot either 'image', or 'masked' image.
        Defaults to 'image'.
    ax : `matplotlib.axes.Axes`, optional
         The Matplotlib axis containing the image data plot.
    centroids : `list`
        The centroids parameter represents a collection of centroid data.
        It can be a combination of different types of data:

        - List of tuples: Each tuple is a centroid with its (X,Y) coordinates.
        - FootprintSet: lsst.afw.detection.FootprintSet object.
        - SourceCatalog: A lsst.afw.table.SourceCatalog object.

        You can provide any combination of these data types within the list.
        The function will plot the centroid data accordingly.
    compass : `bool`, optional
        Add compass to the plot? Defaults to False.
    coordGrid : `bool`, optional
        Add coordinate grid to the plot? Defaults to False.
    stretch : `str', optional
        Changes mapping of colors for the image. Avaliable options:
        css, log, power, asinh, linear, sqrt. Defaults to None.
    percentile : `float', optional
        Parameter for astropy.visualization.PercentileInterval:
        The fraction of pixels to keep. The same fraction of pixels
        is eliminated from both ends. Here: defaults to 99.
    cmap : `str`, optional
        matplotlib colormap. Defaults to 'gray_r'.
    anchorPix : `int`, optional
        The coordinate of pixel to anchor the compass.
    legend : `bool',optional
       Add legend to plot.
    savePlotAs : `str`, optional
        The name of the file to save the plot as, including the file extension.
        The extention must be supported by `matplotlib.pyplot`.
        If None (default) plot will not be saved.

    Returns
    -------
    ax : `matplotlib.axes.Axes`
        A plot showing image data.
    """

    if not ax:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

    match inData:
        case np.ndarray():
            imData = inData
        case Image._maskedImage.MaskedImageF():
            imData = inData.getImage().array
        case Image._image.ImageF():
            imData = inData.array
        case Image._exposure.ExposureF():
            match imageType:
                case 'image':
                    imData = inData.image.array
                case 'masked':
                    imData = inData.getMaskedImage().getImage().array
        case _:
            raise TypeError("This function accepts numpy array, lsst.afw.image.Exposure components. "
                  f"Got {type(inData)}: {inData}")

    # Add stretching
    match stretch:
        case 'css':
            quantiles = utils.getQuantiles(imData, 256)
            norm = colors.BoundaryNorm(quantiles, 256)
        case 'asinh':
            norm = vis.ImageNormalize(imData,
                                      interval=vis.PercentileInterval(percentile),
                                      stretch=vis.AsinhStretch(a=0.1))
        case 'power':
            norm = vis.ImageNormalize(imData,
                                      interval=vis.PercentileInterval(percentile),
                                      stretch=vis.PowerStretch(a=2))
        case 'log':
            norm = vis.ImageNormalize(imData,
                                      interval=vis.PercentileInterval(percentile),
                                      stretch=vis.LogStretch(a=1))
        case 'linear':
            norm = vis.ImageNormalize(imData,
                                      interval=vis.PercentileInterval(percentile),
                                      stretch=vis.LinearStretch())
        case 'sqrt':
            norm = vis.ImageNormalize(imData,
                                      interval=vis.PercentileInterval(percentile),
                                      stretch=vis.SqrtStretch())
        case _:
            norm = None

    ax.imshow(imData, cmap=cmap, origin='lower', norm=norm)

    # Add grid
    if coordGrid:
        ax.grid(color='white', ls='solid')

    # Add compass
    if compass:
        color = 'r'
        try:
            wcs = inData.getWcs()
        except AttributeError:
            wcs = None

        if wcs:
            anchorRa, anchorDec = wcs.pixelToSky(anchorPix, anchorPix)
            east = wcs.skyToPixel(geom.SpherePoint(anchorRa + 30.0 * geom.arcseconds, anchorDec))
            north = wcs.skyToPixel(geom.SpherePoint(anchorRa, anchorDec + 30. * geom.arcseconds))

            ax.arrow(anchorPix, anchorPix, north[0]-anchorPix, north[1]-anchorPix,
                     head_width=1., head_length=1., color=color)
            ax.arrow(anchorPix, anchorPix, east[0]-anchorPix, east[1]-anchorPix,
                     head_width=1., head_length=1., color=color)
            ax.text(north[0], north[1], 'N', color=color)
            ax.text(east[0], east[1], 'E', color=color)

    # Add centroids
    if centroids:
        cCycle = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
        # Create a dict with points from different sources
        cenDict = {}
        c_fs, c_sc, c_lst = 0, 0, 0  # index for color
        for cenSet in centroids:
            match cenSet:
                case FootprintSet():
                    fs = FootprintSet.getFootprints(cenSet)
                    xy = [_.getCentroid() for _ in fs]
                    key = 'footprintSet'+str(c_fs)
                    cenDict[key] = {'data': xy}
                    cenDict[key]['m'] = '+'
                    cenDict[key]['c'] = cCycle[c_fs]
                    c_fs += 1
                case SourceCatalog():
                    key = 'SourceCatalog'+str(c_sc)
                    xy = list(zip(cenSet.getX(), cenSet.getY()))
                    cenDict[key] = {'data': xy}
                    cenDict[key]['m'] = 'x'
                    cenDict[key]['c'] = cCycle[c_sc]
                    c_sc += 1
                case list():
                    key = 'tupleList'+str(c_lst)
                    cenDict[key] = {'data': cenSet}
                    cenDict[key]['m'] = 's'
                    cenDict[key]['c'] = cCycle[c_lst]
                    c_lst += 1
                case _:  # inData type cannot be used here
                    raise TypeError("This function accepts a list of SourceCatalog, \
                                     list of tuples, and/or FootprintSet. "
                                    f"Got {type(cenSet)}: {cenSet}")

        for cSet in cenDict:
            ax.plot(*zip(*cenDict[cSet]['data']),
                    marker=cenDict[cSet]['m'],
                    markeredgecolor=cenDict[cSet]['c'],
                    markerfacecolor='None',
                    linestyle='None', label=cSet)

        if legend:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    if savePlotAs:
        plt.savefig(savePlotAs)

    return ax
