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
from lsst.summit.utils import getQuantiles
import lsst.afw.image as Image

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import astropy.visualization as vis


def plot(inputData,
         figure=None,
         centroids=None,
         title=None,
         showCompass=False,
         stretch='linear',
         percentile=99.,
         cmap='gray_r',
         compassLocation=250,
         addLegend=True,
         savePlotAs=None):

    """Make a plot.

    Parameters
    ----------
    inputData : `numpy.array`, `lsst.afw.image.Exposure`,
        `lsst.afw.image.Image`, or `lsst.afw.image.MaskedImage`
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
    title : `str`, optional
        Title for the plot.
    showCompass : `bool`, optional
        Add compass to the plot? Defaults to False.
    stretch : `str', optional
        Changes mapping of colors for the image. Avaliable options:
        ccs, log, power, asinh, linear, sqrt. Defaults to linear.
    percentile : `float', optional
        Parameter for astropy.visualization.PercentileInterval:
        The fraction of pixels to keep. The same fraction of pixels
        is eliminated from both ends. Here: defaults to 99.
    cmap : `str`, optional
        matplotlib colormap. Defaults to 'gray_r'.
    compassLocation : `int`, optional
        How far in from the bottom left of the image to display the compass.
        By default, compass will be placed at pixel (X,Y) = (250,250).
    addLegend : `bool', optional
       Add legend to the plot.
    savePlotAs : `str`, optional
        The name of the file to save the plot as, including the file extension.
        The extention must be supported by `matplotlib.pyplot`.
        If None (default) plot will not be saved.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The rendered image.
    """

    if not figure:
        figure = plt.figure(figsize=(10, 10))
        ax = figure.add_subplot(111)

    match inputData:
        case np.ndarray():
            imageData = inputData
        case Image.MaskedImage():
            imageData = inputData.getImage().array
        case Image.Image():
            imageData = inputData.array
        case Image.Exposure():
            imageData = inputData.image.array
        case _:
            raise TypeError("This function accepts numpy array, lsst.afw.image.Exposure components. "
                  f"Got {type(inputData)}: {inputData}")

    match stretch:
        case 'ccs':
            quantiles = getQuantiles(imageData, 256)
            norm = colors.BoundaryNorm(quantiles, 256)
        case 'asinh':
            norm = vis.ImageNormalize(imageData,
                                      interval=vis.PercentileInterval(percentile),
                                      stretch=vis.AsinhStretch(a=0.1))
        case 'power':
            norm = vis.ImageNormalize(imageData,
                                      interval=vis.PercentileInterval(percentile),
                                      stretch=vis.PowerStretch(a=2))
        case 'log':
            norm = vis.ImageNormalize(imageData,
                                      interval=vis.PercentileInterval(percentile),
                                      stretch=vis.LogStretch(a=1))
        case 'linear':
            norm = vis.ImageNormalize(imageData,
                                      interval=vis.PercentileInterval(percentile),
                                      stretch=vis.LinearStretch())
        case 'sqrt':
            norm = vis.ImageNormalize(imageData,
                                      interval=vis.PercentileInterval(percentile),
                                      stretch=vis.SqrtStretch())
        case _:
            norm = None

    ax.imshow(imageData, cmap=cmap, origin='lower', norm=norm)

    if showCompass:
        color = 'r'
        try:
            wcs = inputData.getWcs()
        except AttributeError:
            wcs = None

        if wcs:
            anchorRa, anchorDec = wcs.pixelToSky(compassLocation, compassLocation)
            east = wcs.skyToPixel(geom.SpherePoint(anchorRa + 30.0 * geom.arcseconds, anchorDec))
            north = wcs.skyToPixel(geom.SpherePoint(anchorRa, anchorDec + 30. * geom.arcseconds))

            ax.arrow(compassLocation, compassLocation,
                     north[0]-compassLocation, north[1]-compassLocation,
                     head_width=1., head_length=1., color=color)
            ax.arrow(compassLocation, compassLocation,
                     east[0]-compassLocation, east[1]-compassLocation,
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
                case _:
                    raise TypeError("This function accepts a list of SourceCatalog, \
                                     list of tuples, or FootprintSet. "
                                    f"Got {type(cenSet)}: {cenSet}")

        for cSet in cenDict:
            ax.plot(*zip(*cenDict[cSet]['data']),
                    marker=cenDict[cSet]['m'],
                    markeredgecolor=cenDict[cSet]['c'],
                    markerfacecolor='None',
                    linestyle='None', label=cSet)

        if addLegend:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    if title:
        ax.set_title(title)
    if savePlotAs:
        plt.savefig(savePlotAs)

    return figure
