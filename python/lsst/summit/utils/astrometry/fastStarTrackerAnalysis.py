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

import typing
from dataclasses import dataclass

import galsim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.geom as geom
from lsst.summit.utils.utils import detectObjectsInExp
from lsst.utils.iteration import ensure_iterable

__all__ = (
    "tifToExp",
    "getBboxAround",
    "getFlux",
    "getBackgroundLevel",
    "countOverThresholdPixels",
    "sortSourcesByFlux",
    "findFastStarTrackerImageSources",
    "checkResultConsistency",
    "plotSourceMovement",
    "plotSource",
    "plotSourcesOnImage",
)


def tifToExp(filename):
    """Open a tif image as an exposure.

    Opens the file, sets a blank mask plane, and converts the data to
    `np.float32` and returns an exposure, currently with no visitInfo.

    TODO: DM-38422 Once we have a way of getting the expTime, set that,
    and the frequency at which the images were taken.

    Parameters
    ----------
    filename : `str`
        The full path to the file to load.

    Returns
    -------
    exp : `lsst.afw.image.Exposure`
        The exposure.
    """
    im = Image.open(filename)
    data = im.getdata()
    data = np.asarray(data, dtype=np.float32)
    data = data.reshape(im.height, im.width)
    img = afwImage.ImageF(data)
    maskedIm = afwImage.MaskedImageF(img)
    exp = afwImage.ExposureF(maskedIm)
    return exp


def fitsToExp(filename):
    """Open a fits file as an exposure.

    Parameters
    ----------
    filename : `str`
        The full path to the file to load.

    Returns
    -------
    exp : `lsst.afw.image.Exposure`
        The exposure.
    """
    exp = afwImage.ExposureF(filename)
    return exp


def openFile(filename):
    """Open a file as an exposure.

    Parameters
    ----------
    filename : `str`
        The full path to the file to load.

    Returns
    -------
    exp : `lsst.afw.image.Exposure`
        The exposure.
    """
    if filename.endswith(".tif"):
        return tifToExp(filename)
    elif filename.endswith(".fits"):
        return fitsToExp(filename)
    else:
        raise ValueError("File type not recognized")


def getBboxAround(centroid, boxSize, exp):
    """Get a bbox centered on a point, clipped to the exposure.

    If the bbox would extend beyond the bounds of the exposure it is clipped to
    the exposure, resulting in a non-square bbox.

    Parameters
    ----------
    centroid : `lsst.geom.Point`
        The source centroid.
    boxsize : `int`
        The size of the box to centre around the centroid.
    exp : `lsst.afw.image.Exposure`
        The exposure, so the bbox can be clipped to not overrun the bounds.

    Returns
    -------
    bbox : `lsst.geom.Box2I`
        The bounding box, centered on the centroid unless clipping to the
        exposure causes it to be non-square.
    """
    bbox = geom.BoxI().makeCenteredBox(centroid, geom.Extent2I(boxSize, boxSize))
    bbox = bbox.clippedTo(exp.getBBox())
    return bbox


def getFlux(cutout, backgroundLevel=0):
    """Get the flux inside a cutout, subtracting the image-background.

    Here the flux is simply summed, and if the image background level is
    supplied, it is subtracted off, assuming it is constant over the cutout. A
    more accurate(?) flux is obtained by the hsm model fit.

    Parameters
    ----------
    cutout : `np.array`
        The cutout as a raw array.
    backgroundLevel : `float`, optional
        If supplied, this is subtracted as a constant background level.

    Returns
    -------
    flux : `float`
        The flux of the source in the cutout.
    """
    rawFlux = np.sum(cutout)
    if not backgroundLevel:
        return rawFlux

    return rawFlux - (cutout.size * backgroundLevel)


def getBackgroundLevel(exp, nSigma=3):
    """Calculate the clipped image mean and stddev of an exposure.

    Testing shows on images like this, 2 rounds of sigma clipping is more than
    enough so this is left fixed here.

    Parameters
    ----------
    exp : `lsst.afw.image.Exposure`
        The exposure.
    nSigma : `float`, optional
        The number of sigma to clip to for the background estimation.

    Returns
    -------
    mean : `float`
        The clipped mean, as an estimate of the background level
    stddev : `float`
        The clipped standard deviation, as an estimate of the background noise.
    """
    sctrl = afwMath.StatisticsControl()
    sctrl.setNumSigmaClip(nSigma)
    sctrl.setNumIter(2)  # this is always plenty here
    statTypes = afwMath.MEANCLIP | afwMath.STDEVCLIP
    stats = afwMath.makeStatistics(exp.maskedImage, statTypes, sctrl)
    std, _ = stats.getResult(afwMath.STDEVCLIP)
    mean, _ = stats.getResult(afwMath.MEANCLIP)
    return mean, std


def countOverThresholdPixels(cutout, bgMean, bgStd, nSigma=15):
    """Get the number of pixels in the cutout which are 'in the source'.

    From the one image I've looked at so far, the drop-off is quite slow
    probably due to some combination of focus, plate scale, star brightness,
    pointing quality etc, so the default nSigma is 15 here as that looked about
    right when I plotted it by eye.

    Parameters
    ----------
    cutout : `np.array`
        The cutout to measure.
    bgMean : `float`
        The background level.
    bgStd : `float`
        The clipped standard deviation in the image.
    nSigma : `float`, optional
        The number of sigma above background at which to count pixels as being
        over threshold.

    Returns
    -------
    nPix : `int`
        The number of pixels above threshold.
    """
    inds = np.where(cutout > (bgMean + 0 * bgStd))
    return len(inds[0])


def sortSourcesByFlux(sources, reverse=False):
    """Sort the sources by flux, returning the brightest first.

    Parameters
    ----------
    sources : `list` of
              `lsst.summit.utils.astrometry.fastStarTrackerAnalysis.Source`
        The list of sources to sort.
    reverse : `bool`, optional
        Return the brightest at the start of the list if ``reverse`` is
        ``False``, or the brightest last if ``reverse`` is ``True``.

    Returns
    -------
    sources : `list` of
              `lsst.summit.utils.astrometry.fastStarTrackerAnalysis.Source`
        The sources, sorted by flux.
    """
    # invert reverse because we want brightest first by default, but want the
    # reverse arg to still behave as one would expect
    return sorted(sources, key=lambda s: s.rawFlux, reverse=not reverse)


@dataclass(slots=True)
class Source:
    """A dataclass for FastStarTracker analysis results."""

    # raw numbers
    centroidX: float = np.nan  # in image coordinates
    centroidY: float = np.nan  # in image coordinates
    rawFlux: float = np.nan
    nPix: int | float = np.nan
    bbox: geom.Box2I | None = None
    cutout: np.ndarray | None = None
    localCentroidX: float = np.nan  # in cutout coordinates
    localCentroidY: float = np.nan  # in cutout coordinates

    # numbers from the hsm moments fit
    hsmFittedFlux: float = np.nan
    hsmCentroidX: float = np.nan
    hsmCentroidY: float = np.nan
    moments: galsim.hsm.ShapeData | None = None  # keep the full fit even though we pull some things out too

    imageBackground: float = np.nan
    imageStddev: float = np.nan
    nSourcesInImage: int | float = np.nan
    parentImageWidth: int | float = np.nan
    parentImageHeight: int | float = np.nan

    def __repr__(self):
        """Print everything except the full details of the moments."""
        retStr = ""
        for itemName in self.__slots__:
            v = getattr(self, itemName)
            if isinstance(v, int):  # print ints as ints
                retStr += f"{itemName} = {v}\n"
            elif isinstance(v, float):  # but round floats at 3dp
                retStr += f"{itemName} = {v:.3f}\n"
            elif itemName == "moments":  # and don't spam the full moments
                retStr += f"moments = {type(v)}\n"
            elif itemName == "bbox":  # and don't spam the full moments
                retStr += f"bbox = lsst.geom.{repr(v)}\n"
            elif itemName == "cutout":  # and don't spam the full moments
                if v is None:
                    retStr += "cutout = None\n"
                else:
                    retStr += f"cutout = {type(v)}\n"
        return retStr


class NanSource:
    def __getattribute__(self):
        return np.nan


def findFastStarTrackerImageSources(filename, boxSize, attachCutouts=True):
    """Analyze a single FastStarTracker image.

    Parameters
    ----------
    filename : `str`
        The full
    boxSize : `int`
        The size of the box to put around each source for measurement.
    attachCutouts : `bool`, optional
        Attach the cutouts to the ``Source`` objects? Useful for
        debug/plotting but adds memory usage.

    Returns
    -------
    sources : `list` of
              `lsst.summit.utils.astrometry.fastStarTrackerAnalysis.Source`
        The sources in the image, sorted by rawFlux.
    """
    exp = openFile(filename)
    footprintSet = detectObjectsInExp(exp)
    footprints = footprintSet.getFootprints()
    bgMean, bgStd = getBackgroundLevel(exp)

    sources = []
    for footprint in footprints:
        source = Source()
        source.nSourcesInImage = len(footprints)
        source.parentImageWidth, source.parentImageHeight = exp.getDimensions()

        centroid = footprint.getCentroid()
        bbox = getBboxAround(centroid, boxSize, exp)
        source.bbox = bbox
        cutout = exp.image[bbox].array
        if attachCutouts:
            source.cutout = cutout
        source.centroidX = centroid[0]
        source.centroidY = centroid[1]
        source.rawFlux = getFlux(cutout, bgMean)
        source.imageBackground = bgMean
        source.imageStddev = bgStd
        source.nPix = countOverThresholdPixels(cutout, bgMean, bgStd)

        moments = galsim.hsm.FindAdaptiveMom(galsim.Image(cutout))
        source.moments = moments
        source.hsmFittedFlux = moments.moments_amp
        source.hsmCentroidX = moments.moments_centroid.x + bbox.minX - 1
        source.hsmCentroidY = moments.moments_centroid.y + bbox.minY - 1
        source.localCentroidX = moments.moments_centroid.x - 1
        source.localCentroidY = moments.moments_centroid.y - 1
        sources.append(source)
    return sortSourcesByFlux(sources)


def checkResultConsistency(results, maxAllowableShift=5, silent=False):
    """Check if a set of results are self-consistent.

    Check the number of detected sources are the same in each image, that no
    sources have been removed from each image's source list, and that all the
    input images were the same size (because we read out sub frames, and
    analyzing these with full frame data invalidates the centroid coordinates).

    Also displays the maximum (x, y) movements between adjacent exposures, and
    the mean and stddev of the main source's flux.

    Parameters
    ----------
    results : `dict` of `list` of
              `lsst.summit.utils.astrometry.fastStarTrackerAnalysis.Source`
        A dict, keyed by sequence number, with each value being a list of the
        sources found in the image, e.g. as returned by
        ``findFastStarTrackerImageSources()``.
    maxAllowableShift : `float`
        The biggest centroid shift between adjacent images allowable before
        something is considered to have gone wrong.
    silent : `bool`, optional
        Print some useful checks and measurements if ``False``, otherwise just
        return whether the results appear nominally OK silently (for use when
        being called by other code rather than users).

    Returns
    -------
    consistent : `bool`
        Are the results nominally consistent?
    """
    if isinstance(results, typing.ValuesView):  # in case we're passed a .values()
        results = list(results)

    sourceCounts = set([len(sourceSet) for sourceSet in results])
    if sourceCounts == {0}:  # none of the images contain any detections
        if not silent:
            print("No images contain any sources. Results are technically consistent, but also useless.")
        # this is technically consistent, so return True, but any downstream
        # code which tries to make plots with these will fail, of course.
        return True

    if 0 in ([len(sourceSet) for sourceSet in results]):
        if not silent:
            print(
                "Some results contain no sources. Results are therefore fundamentally inconsistent"
                " and other checks cannot be run"
            )
        return False

    consistent = True
    toPrint = []
    nSources = set([sourceSet[0].nSourcesInImage for sourceSet in results])
    if len(nSources) != 1:
        toPrint.append(f"❌ Images contain a variable number of sources: {nSources}")
        consistent = False
    else:
        n = nSources.pop()
        toPrint.append(f"✅ All images contain the same nominal number of sources at detection stage: {n}")

    nSourcesCounted = set([len(sourceSet) for sourceSet in results])
    if len(nSourcesCounted) != 1:
        toPrint.append(
            f"❌ Number of actual sources in each sourceSet varies, got: {nSourcesCounted}."
            " If some were manually removed you can ignore this"
        )
        consistent = False
    else:
        n = nSourcesCounted.pop()
        toPrint.append(f"✅ All results contain the same number of actual sources per image: {n}")

    widths = set([sourceSet[0].parentImageWidth for sourceSet in results])
    heights = set([sourceSet[0].parentImageHeight for sourceSet in results])
    if len(widths) != 1 or len(heights) != 1:
        toPrint.append(f"❌ Input images were of variable dimenions! {widths=}, {heights=}")
        consistent = False
    else:
        toPrint.append("✅ All input images were of the same dimensions")

    if len(results) > 1:  # can't np.diff an array of length 1 so these are not useful/defined
        # now the basic checks have passed, do some sanity checks on the
        # maximum deltas for the primary sources
        sources = [sourceSet[0] for sourceSet in results]
        dx = np.diff([s.centroidX for s in sources])
        dy = np.diff([s.centroidY for s in sources])
        maxMovementX = np.max(dx)
        maxMovementY = np.max(dy)
        happyOrSad = "✅"
        if max(maxMovementX, maxMovementY) > maxAllowableShift:
            consistent = False
            happyOrSad = "❌"

        toPrint.append(
            f"{happyOrSad} Maximum centroid movement of brightest object between images in (x, y)"
            f" = ({maxMovementX:.2f}, {maxMovementY:.2f}) pix"
        )

        fluxStd = np.nanstd([s.rawFlux for s in sources])
        fluxMean = np.nanmean([s.rawFlux for s in sources])
        toPrint.append(f"Mean and stddev of flux from brightest object = {fluxMean:.1f} ± {fluxStd:.1f} ADU")

    if not silent:
        for line in toPrint:
            print(line)

    return consistent


def plotSourceMovement(results, sourceIndex=0, allowInconsistent=False):
    """Plot the centroid movements and fluxes etc for a set of results.

    By default the brightest source in each image is plotted, but this can be
    changed by setting ``sourceIndex`` to values greater than 0 to move through
    the list of sources in each image.

    Parameters
    ----------
    results : `dict` of `list` of
              `lsst.summit.utils.astrometry.fastStarTrackerAnalysis.Source`
        A dict, keyed by sequence number, with each value being a list of the
        sources found in the image, e.g. as returned by
        ``findFastStarTrackerImageSources()``.
    sourceIndex : `int`, optional
        If there is more than one source in every image, which source number
        should the plot be made for? Defaults to zero, which is the brightest
        source by default.
    allowInconsistent : `bool`, optional
        Make the plots even if the input results appear to be inconsistent?

    Returns
    -------
    figs : `list` of `matplotlib.figure.Figure`
        The figures. The first is the source's flux and x, y movement over the
        image sequence, and the second is a scatter plot of the x and y, with
        the color showing the position in the sequence.
    """
    opts = {
        "marker": "o",
        "markersize": 6,
        "linestyle": "-",
    }

    consistent = checkResultConsistency(results.values(), silent=True)
    if not consistent and not allowInconsistent:
        checkResultConsistency(results.values(), silent=False)  # print the problem if we're raising
        raise ValueError("The sources were found to be inconsistent and allowInconsistent=False")

    sourceDict = {k: v[sourceIndex] if len(v) > sourceIndex else NanSource() for k, v in results.items()}
    seqNums = list(sourceDict.keys())
    sources = list(sourceDict.values())

    axisLabelSize = 18

    figs = []
    fig = plt.figure(figsize=(10, 16))
    ax1, ax2, ax3 = fig.subplots(3, sharex=True)
    fig.subplots_adjust(hspace=0)

    ax1.plot(seqNums, [s.rawFlux for s in sources], label="Raw Flux", **opts)
    ax1.plot(seqNums, [s.hsmFittedFlux for s in sources], label="Fitted Flux", **opts)
    ax1.set_ylabel("Flux (ADU)", size=axisLabelSize)
    ax1.legend()

    ax2.plot(seqNums, [s.centroidX for s in sources], label="Raw centroid x", **opts)
    ax2.plot(
        seqNums,
        [s.hsmCentroidX for s in sources],
        label="Fitted centroid x",
        **opts,
    )
    ax2.set_ylabel("x-centroid (pixels)", size=axisLabelSize)
    ax2.legend()

    ax3.plot(seqNums, [s.centroidY for s in sources], label="Raw centroid y", **opts)
    ax3.plot(
        seqNums,
        [s.hsmCentroidY for s in sources],
        label="Fitted centroid y",
        **opts,
    )
    ax3.set_ylabel("y-centroid (pixels)", size=axisLabelSize)
    ax3.set_xlabel("SeqNum", size=axisLabelSize)
    ax3.legend()

    figs.append(fig)

    fig = plt.figure(figsize=(10, 10))
    ax4 = fig.subplots(1)

    colors = np.arange(len(sources))
    # gnuplot2 has a nice balance of nothing white, and having an intuitive
    # progression of colours so the eye can pick out trends on the point cloud.
    axRef = ax4.scatter(
        [s.centroidX for s in sources],
        [s.centroidY for s in sources],
        c=colors,
        cmap="gnuplot2",
    )
    ax4.set_xlabel("x-centroid (pixels)", size=axisLabelSize)
    ax4.set_ylabel("y-centroid (pixels)", size=axisLabelSize)
    ax4.set_aspect("equal", "box")
    # move the colorbar
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(axRef, cax=cax)
    cbar.set_label("Image number in series", size=axisLabelSize * 0.75)
    figs.append(fig)

    return figs


# -------------- plotting tools


def bboxToMatplotlibRectanle(bbox):
    """Convert a bbox to a matplotlib Rectangle for plotting.

    Parameters
    ----------
    results : `lsst.geom.Box2I` or `lsst.geom.Box2D`
        The bbox to convert.

    Returns
    -------
    rectangle : `bool`
        The rectangle.
    """
    ll = bbox.minX, bbox.minY
    width, height = bbox.getDimensions()
    return Rectangle(ll, width, height)


def plotSourcesOnImage(parentFilename, sources):
    """Plot one of more source on top of an image.

    Parameters
    ----------
    parentFilename : `str`
        The full path to the parent (.tif) file.
    sources : `list` of
              `lsst.summit.utils.astrometry.fastStarTrackerAnalysis.Source` or
              `lsst.summit.utils.astrometry.fastStarTrackerAnalysis.Source`
        The sources found in the image.
    """
    exp = openFile(parentFilename)
    data = exp.image.array

    fig = plt.figure(figsize=(16, 8))
    ax = fig.subplots(1)

    plt.imshow(data, interpolation="None", origin="lower")

    sources = ensure_iterable(sources)
    patches = []
    for source in sources:
        ax.scatter(source.centroidX, source.centroidY, color="red", marker="x")  # mark the centroid
        patch = bboxToMatplotlibRectanle(source.bbox)
        patches.append(patch)

    # move the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax)

    # plot the bboxes on top
    pc = PatchCollection(patches, edgecolor="r", facecolor="none")
    ax.add_collection(pc)

    plt.tight_layout()


def plotSource(source):
    """Plot a single source.

    Parameters
    ----------
    source : `lsst.summit.utils.astrometry.fastStarTrackerAnalysis.Source`
        The source to plot.
    """
    if source.cutout is None:
        raise RuntimeError(
            "Can only plot sources with attached cutouts. Either set attachCutouts=True "
            "in findFastStarTrackerImageSources() or try using plotSourcesOnImage() instead"
        )

    fig = plt.figure(figsize=(16, 8))
    ax = fig.subplots(1)

    plt.imshow(source.cutout, interpolation="None", origin="lower")  # plot the image
    ax.scatter(source.localCentroidX, source.localCentroidY, color="red", marker="x", s=200)  # mark centroid

    # move the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cax=cax)

    plt.tight_layout()
