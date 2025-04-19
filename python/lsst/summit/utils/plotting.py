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

import logging

import astropy.visualization as vis
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import lsst.afw.detection as afwDetection
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.geom as geom
from lsst.summit.utils.utils import getImageArray, getQuantiles


def drawCompass(
    ax: matplotlib.axes.Axes,
    wcs: afwGeom.SkyWcs,
    compassLocation: int = 300,
    arrowLength: float = 300.0,
) -> matplotlib.axes.Axes:
    """
    Draw the compass.
    The arrowLength is the length of compass arrows (arrows should have
    the same length).
    The steps here are:
    - transform the (compassLocation, compassLocation) to RA, DEC coordinates
    - move this point in DEC to get N; in RA to get E directions
    - transform N and E points back to pixel coordinates
    - find linear solutions for lines connecting the center of
      the compass with N and E points
    - find points along those lines located at the distance of
      arrowLength form the (compassLocation, compassLocation).
    - there will be two points for each linear solution.
      Choose the correct one.
    - centers of the N/E labels will also be located on those lines.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        The axes on which the compass will be drawn.
    wcs : `lsst.afw.geom.SkyWcs`
        WCS from exposure.
    compassLocation : `int`, optional
        How far in from the bottom left of the image to display the compass.
    arrowLength : `float`, optional
        The length of the compass arrow.
    Returns
    -------
    ax : `matplotlib.axes.Axes`
        The axes with the compass.
    """

    anchorRa, anchorDec = wcs.pixelToSky(compassLocation, compassLocation)
    east = wcs.skyToPixel(geom.SpherePoint(anchorRa + 30.0 * geom.arcseconds, anchorDec))
    north = wcs.skyToPixel(geom.SpherePoint(anchorRa, anchorDec + 30.0 * geom.arcseconds))
    labelPosition = arrowLength + 50.0

    for xy, label in [(north, "N"), (east, "E")]:
        if compassLocation == xy[0]:
            xTip = compassLocation
            xTipLabel = compassLocation
            if xy[1] > compassLocation:
                yTip = compassLocation + arrowLength
                yTipLabel = compassLocation + labelPosition
            else:
                yTip = compassLocation - arrowLength
                yTipLabel = compassLocation - labelPosition
        else:
            slope = (xy[1] - compassLocation) / (xy[0] - compassLocation)
            xTipProjection = arrowLength / np.sqrt(1.0 + slope**2)
            xTipLabelProjection = labelPosition / np.sqrt(1.0 + slope**2)

            if xy[0] > compassLocation:
                xTip = compassLocation + xTipProjection
                xTipLabel = compassLocation + xTipLabelProjection
            elif xy[0] < compassLocation:
                xTip = compassLocation - xTipProjection
                xTipLabel = compassLocation - xTipLabelProjection
            yTip = slope * (xTip - compassLocation) + compassLocation
            yTipLabel = slope * (xTipLabel - compassLocation) + compassLocation

        color = "r"
        ax.arrow(
            compassLocation,
            compassLocation,
            xTip - compassLocation,
            yTip - compassLocation,
            head_width=30.0,
            length_includes_head=True,
            color=color,
        )
        ax.text(xTipLabel, yTipLabel, label, ha="center", va="center", color=color)
    return ax


def plot(
    inputData: np.ndarray | afwImage.Exposure | afwImage.Image | afwImage.MaskedImage,
    figure: matplotlib.figure.Figure | None = None,
    centroids: list[tuple[int, int]] | None = None,
    footprints: (
        afwDetection.FootprintSet | afwDetection.Footprint | list[afwDetection.Footprint] | None
    ) = None,
    sourceCat: afwTable.SourceCatalog = None,
    title: str | None = None,
    showCompass: bool = True,
    stretch: str = "linear",
    percentile: float = 99.0,
    cmap: str = "gray",
    compassLocation: int = 300,
    addLegend: bool = False,
    savePlotAs: str | None = None,
    logger: logging.Logger | None = None,
) -> matplotlib.figure.Figure:
    """Plot an input image accommodating different data types and additional
    features, like: overplotting centroids, compass (if the input image
    has a WCS), stretching, plot title, and legend.

    Parameters
    ----------
    inputData : `numpy.array` or
                `lsst.afw.image.Exposure` or
                `lsst.afw.image.Image`, or
                `lsst.afw.image.MaskedImage`
        The input data.
    figure : `matplotlib.figure.Figure`, optional
         The matplotlib figure that will be used for plotting.
    centroids : `list`
        The centroids parameter as a list of tuples.
        Each tuple is a centroid with its (X,Y) coordinates.
    footprints:  `lsst.afw.detection.FootprintSet` or
                 `lsst.afw.detection.Footprint` or
                 `list` of `lsst.afw.detection.Footprint`
        The footprints containing centroids to plot.
    sourceCat: `lsst.afw.table.SourceCatalog`:
        An `lsst.afw.table.SourceCatalog` object containing centroids
        to plot.
    title : `str`, optional
        Title for the plot.
    showCompass : `bool`, optional
        Add compass to the plot? Defaults to True.
    stretch : `str', optional
        Changes mapping of colors for the image. Avaliable options:
        ccs, log, power, asinh, linear, sqrt, midtone. Defaults to linear.
    percentile : `float', optional
        Parameter for astropy.visualization.PercentileInterval.
        Sets lower and upper limits for a stretch. This parameter
        will be ignored if stretch='ccs'.
    cmap : `str`, optional
        The colormap to use for mapping the image values to colors. This can be
        a string representing a predefined colormap. Default is 'gray'.
    compassLocation : `int`, optional
        How far in from the bottom left of the image to display the compass.
        By default, compass will be placed at pixel (x,y) = (300,300).
    addLegend : `bool', optional
       Option to add legend to the plot. Recommended if centroids come from
       different sources. Default value is False.
    savePlotAs : `str`, optional
        The name of the file to save the plot as, including the file extension.
        The extention must be supported by `matplotlib.pyplot`.
        If None (default) plot will not be saved.
    logger : `logging.Logger`, optional
        The logger to use for errors, created if not supplied.
    Returns
    -------
    figure : `matplotlib.figure.Figure`
        The rendered image.
    """

    if not figure:
        figure = plt.figure(figsize=(10, 10))

    ax = figure.add_subplot(111)

    if not logger:
        logger = logging.getLogger(__name__)

    imageData = getImageArray(inputData)

    if np.isnan(imageData).all():
        im = ax.imshow(imageData, origin="lower", aspect="equal")
        logger.warning("The imageData contains only NaN values.")
    else:
        interval = vis.PercentileInterval(percentile)
        match stretch:
            case "ccs":
                quantiles = getQuantiles(imageData, 256)
                norm = colors.BoundaryNorm(quantiles, 256)
            case "asinh":
                norm = vis.ImageNormalize(imageData, interval=interval, stretch=vis.AsinhStretch(a=0.1))
            case "power":
                norm = vis.ImageNormalize(imageData, interval=interval, stretch=vis.PowerStretch(a=2))
            case "log":
                norm = vis.ImageNormalize(imageData, interval=interval, stretch=vis.LogStretch(a=1))
            case "linear":
                norm = vis.ImageNormalize(imageData, interval=interval, stretch=vis.LinearStretch())
            case "sqrt":
                norm = vis.ImageNormalize(imageData, interval=interval, stretch=vis.SqrtStretch())
            case "midtone":
                imageData = stretchDataMidTone(imageData)
                # no interval in this norm as imageData is now [0, 1] aready
                norm = vis.ImageNormalize(imageData, stretch=vis.LinearStretch())
            case _:
                raise ValueError(
                    f"Invalid value for stretch : {stretch}. "
                    "Accepted options are: ccs, asinh, power, log, linear, sqrt."
                )

        im = ax.imshow(imageData, cmap=cmap, origin="lower", norm=norm, aspect="equal", interpolation="auto")

        if stretch != "midtone":
            div = make_axes_locatable(ax)
            cax = div.append_axes("right", size="5%", pad=0.05)
            figure.colorbar(im, cax=cax)

    if showCompass:
        try:
            assert hasattr(inputData, "getWcs"), "inputData does not have a getWcs method"
            wcs = inputData.getWcs()
        except AssertionError:
            logger.warning("Failed to get WCS from input data. Compass will not be plotted.")
            wcs = None

        if wcs:
            arrowLength = min(imageData.shape) * 0.05
            ax = drawCompass(ax, wcs, compassLocation=compassLocation, arrowLength=arrowLength)

    if centroids:
        ax.plot(
            *zip(*centroids),
            marker="x",
            markeredgecolor="r",
            markerfacecolor="None",
            linestyle="None",
            label="List of centroids",
        )

    if sourceCat:
        ax.scatter(
            sourceCat.getX(),
            sourceCat.getY(),
            marker="o",
            edgecolors="c",  # cyan rings
            c="None",  # empty cicrles (no fill)
            label="Source catalog",
        )

    if footprints:
        match footprints:
            case afwDetection.FootprintSet():
                fs = afwDetection.FootprintSet.getFootprints(footprints)
                xy = [_.getCentroid() for _ in fs]
            case afwDetection.Footprint():
                xy = [footprints.getCentroid()]
            case list():
                xy = []
                for i, ft in enumerate(footprints):
                    try:
                        ft.getCentroid()
                    except AttributeError:
                        raise TypeError(
                            "Cannot get centroids for one of the "
                            "elements from the footprints list. "
                            "Expected lsst.afw.detection.Footprint, "
                            f"got {type(ft)} for footprints[{i}]"
                        )
                    xy.append(ft.getCentroid())
            case _:
                raise TypeError(
                    "This function works with FootprintSets, "
                    "single Footprints, and iterables of Footprints. "
                    f"Got {type(footprints)}"
                )

        ax.plot(
            *zip(*xy),
            marker="x",
            markeredgecolor="b",
            markerfacecolor="None",
            linestyle="None",
            label="Footprints centroids",
        )

    if addLegend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=5)

    if title:
        ax.set_title(title)

    if savePlotAs:
        plt.savefig(savePlotAs)

    return figure


def _computeMtf(image: np.ndarray, midtonesBalance: float) -> np.ndarray:
    """
    Compute the midtones transfer function (MTF) for an image.

    Parameters
    ----------
    image : `np.ndarray`
        Input image normalized to [0, 1].
    midtonesBalance : `float`
        Balance parameter controlling midtones emphasis.

    Returns
    -------
    image : `np.ndarray`
        Image after applying the MTF, with NaNs replaced by fallback values.
    """
    M = np.full(image.shape, midtonesBalance)
    maskHalf = image == 0.5
    maskZero = image == 0
    maskOne = image == 1
    fallback = (maskHalf * 0.5) * (1 - maskZero) + maskOne
    result = (M - 1) * image / ((2 * M - 1) * image - M)
    nanMask = ~np.isfinite(result)
    result[nanMask] = fallback[nanMask]
    return result


def _applyClip(image: np.ndarray, clipLow: float, clipHigh: float) -> np.ndarray:
    """
    Linearly clip and scale image values between clipLow and clipHigh.

    Parameters
    ----------
    image : `np.ndarray`
        Input image normalized to [0, 1].
    clipLow : `float`
        Lower clipping threshold.
    clipHigh : `float`
        Upper clipping threshold.

    Returns
    -------
    clipped : `np.ndarray`
        Clipped and scaled image, with values in [0, 1].
    """
    belowLow = image < clipLow
    aboveHigh = image > clipHigh
    scaled = (image - clipLow) / (clipHigh - clipLow)
    return np.clip(scaled * (~belowLow) + aboveHigh, 0.0, 1.0)


def _applyExpansion(image: np.ndarray, outMin: float, outMax: float) -> np.ndarray:
    """
    Expand image dynamic range from [outMin, outMax] to [0, 1].

    Parameters
    ----------
    image : `np.ndarray`
        Input image after MTF.
    outMin : `float`
        Minimum output value (usually 0.0).
    outMax : `float`
        Maximum output value (usually 1.0).

    Returns
    -------
    expanded : `np.ndarray`
        Expanded image in [0, 1].
    """
    return (image - outMin) / (outMax - outMin)


def _applyDisplayFunction(
    image: np.ndarray, midtonesBalance: float, clipLow: float, clipHigh: float, outMin: float, outMax: float
) -> np.ndarray:
    """
    Apply the full display function: clip, MTF, then expansion.

    Parameters
    ----------
    image : `np.ndarray`
        Input image normalized to [0, 1].
    midtonesBalance : `float`
        Midtones balance parameter.
    clipLow : `float`
        Lower clipping threshold.
    clipHigh : `float`
        Upper clipping threshold.
    outMin : `float`
        Minimum output of expansion.
    outMax : `float`
        Maximum output of expansion.

    Returns
    -------
    np.ndarray
        Stretched image ready for display.
    """
    clipped = _applyClip(image, clipLow, clipHigh)
    mtf = _computeMtf(clipped, midtonesBalance)
    return _applyExpansion(mtf, outMin, outMax)


def _computeDisplayParameters(data: np.ndarray) -> tuple[float, float, float, float, float]:
    """
    Compute parameters for display function based on data statistics.

    Parameters
    ----------
    data : `np.ndarray`
        Normalized image data array.

    Returns
    -------
    tuple[float, float, float, float, float]
        midtonesBalance, clipLow, clipHigh, outMin, outMax
    """
    median = np.median(data)
    deviations = np.abs(data.ravel() - median)
    madn = 1.4826 * np.median(np.sort(deviations))
    targetBackground = 0.25
    clippingFactor = -2.8

    aboveHalf = median > 0.5

    if not aboveHalf and madn != 0:
        clipLow = min(1.0, max(0.0, median + clippingFactor * madn))
    else:
        clipLow = 0.0

    if aboveHalf and madn != 0:
        clipHigh = min(1.0, max(0.0, median - clippingFactor * madn))
    else:
        clipHigh = 1.0

    if median <= 0.5:
        midtonesBalance = (
            (targetBackground - 1)
            * (median - clipLow)
            / ((2 * targetBackground - 1) * (median - clipLow) - targetBackground)
        )
    else:
        midtonesBalance = (
            (clipHigh - median - 1)
            * targetBackground
            / (2 * (clipHigh - median - 1) * targetBackground - (clipHigh - median))
        )

    return midtonesBalance, clipLow, clipHigh, 0.0, 1.0


def stretchDataMidTone(
    imageLike: np.ndarray | afwImage.Exposure | afwImage.Image | afwImage.MaskedImage,
) -> np.ndarray:
    """
    Normalize and stretch image data from an Exposure object using the Midtone
    Transfer Function (MTF).

    This is following:
        https://pixinsight.com/doc/docs/XISF-1.0-spec/XISF-1.0-spec.html
        #__XISF_Data_Objects_:_XISF_Image_:_Display_Function__

    Parameters
    ----------
    imageLike : `numpy.ndarray`, `lsst.afw.image.Exposure`,
        `lsst.afw.image.Image`, or `lsst.afw.image.MaskedImage`
        The image-like object containg the data to be stretched.

    Returns
    -------
    stretched : `np.ndarray`
        The stretched image array.
    """
    data = getImageArray(imageLike)

    pedestal = np.min(data)
    if pedestal >= 0.0:
        norm = np.max(data)
        normalized = data / norm
    else:
        norm = np.max(data - pedestal)
        normalized = (data - pedestal) / norm

    midtonesBalance, clipLow, clipHigh, outMin, outMax = _computeDisplayParameters(normalized)
    stretched = _applyDisplayFunction(normalized, midtonesBalance, clipLow, clipHigh, outMin, outMax)
    return stretched
