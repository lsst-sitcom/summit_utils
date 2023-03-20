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

import galsim
import typing

import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from PIL import Image

import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.geom as geom

from lsst.summit.utils.utils import detectObjectsInExp

__all__ = ('tifToExp',
           'getBboxAround',
           'getFlux',
           'getBackgroundLevel',
           'countOverThresholdPixels',
           'sortSourcesByFlux',
           'analyzeFastStarTrackerImage',
           'checkResultConsistency',
           'plotResults'
           )


def tifToExp(filename):
    """Open a tif image as an exposure.

    Opens the file, sets a blank mask plane, and converts the data to
    `np.float32` and returns an exposure, currently with no visitInfo.

    Once we have a way of getting the expTime, set that.

    Parameters
    ----------
    filename : `str`
        The full path to the file to load.
    """
    im = Image.open(filename)
    data = im.getdata()
    data = np.asarray(data, dtype=np.float32)
    data = data.reshape(im.height, im.width)
    img = afwImage.ImageF(data)
    maskedIm = afwImage.MaskedImageF(img)
    exp = afwImage.ExposureF(maskedIm)
    return exp


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

    nPix = cutout.shape[0] * cutout.shape[1]
    return rawFlux - (nPix*backgroundLevel)


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
    inds = np.where(cutout > (bgMean + 0*bgStd))
    return len(inds[0])


def sortSourcesByFlux(sources, reverse=False):
    """Sort the sources by flux, returning the brightest first.

    Parameters
    ----------
    sources : `list` of
              `lsst.summit.utils.astrometry.starTrackerAnalysis.Source`
        The list of sources to sort.
    reverse : `bool`, optional
        Return the brightest at the start of the list if ``reverse`` is
        ``False``, or the brightest last if ``reverse`` is ``True``.

    Returns
    -------
    sources : `list` of
              `lsst.summit.utils.astrometry.starTrackerAnalysis.Source`
        The sources, sorted by flux.
    """
    # invert reverse because we want brightest first by default, but want the
    # reverse arg to still behave as one would expect
    return sorted(sources, key=lambda s: s.rawFlux, reverse=not reverse)


@dataclass(slots=True)
class Source:
    """A dataclass for FastStarTracker analysis results.
    """
    # raw numbers
    centroidX: float = np.nan
    centroidY: float = np.nan
    rawFlux: float = np.nan
    nPix: int = np.nan

    # numbers from the hsm moments fit
    hsmFittedFlux: float = np.nan
    hsmCentroidX: float = np.nan
    hsmCentroidY: float = np.nan
    moments: galsim.hsm.ShapeData = None  # keep the full fit even though we pull some things out too

    imageBackground: float = np.nan
    imageStddev: float = np.nan
    nSourcesInImage: int = np.nan
    parentImageWidth: int = np.nan
    parentImageHeight: int = np.nan

    def __repr__(self):
        """Print everything except the full details of the moments.
        """
        retStr = ''
        for itemName in self.__slots__:
            v = getattr(self, itemName)
            if isinstance(v, int):  # print ints as ints
                retStr += f'{itemName} = {v}\n'
            elif isinstance(v, float):  # but round floats at 3dp
                retStr += f'{itemName} = {v:.3f}\n'
            elif itemName == 'moments':  # and don't spam the full moments
                retStr += 'moments = <galsim.hsm.ShapeData>\n'

        # retStr = '\n'.join([f'{item} = {getattr(self, item):.2f}'
        #                     for item in self.__dict__ if item != 'moments'])
        # retStr += '\n' + 'moments = <galsim.hsm.ShapeData>'
        return retStr


def analyzeFastStarTrackerImage(filename, boxSize):
    """Analyze a single FastStarTracker image.

    Parameters
    ----------
    filename : `str`
        The full
    boxSize : `int`
        The size of the box to put around each source for measurement.

    Returns
    -------
    sources : `list` of
              `lsst.summit.utils.astrometry.starTrackerAnalysis.Source`
        The sources in the image, sorted by rawFlux.
    """
    exp = tifToExp(filename)
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
        cutout = exp.image[bbox].array
        rawFlux = getFlux(cutout, bgMean)
        source.centroidX = centroid[0]
        source.centroidY = centroid[1]
        source.rawFlux = rawFlux
        source.imageBackground = bgMean
        source.imageStddev = bgStd
        source.nPix = countOverThresholdPixels(cutout, bgMean, bgStd)

        moments = galsim.hsm.FindAdaptiveMom(galsim.Image(cutout))
        source.moments = moments
        source.hsmFittedFlux = moments.moments_amp
        source.hsmCentroidX = moments.moments_centroid.x + bbox.minX - 1
        source.hsmCentroidY = moments.moments_centroid.y + bbox.minY - 1
        sources.append(source)
    return sortSourcesByFlux(sources)


def checkResultConsistency(results, silent=False):
    """Check if a set of results are self-consistent.

    For each image, check the number of detected sources are the same in each
    image, that no sources have been removed from each image's source list, and
    that all the input images were the same size (because we read out sub
    frames, and analyzing these with full frame data invalidates the centroid
    coordinates).

    Also displays the maximum (x, y) movements between adjacent exposures, and
    the mean and stddev of the main source's flux.

    Parameters
    ----------
    results : `dict` of `list` of
              `lsst.summit.utils.astrometry.starTrackerAnalysis.Source`
        A dict, keyed by sequence number, with each value being a list of the
        sources found in the image, e.g. as returned by
        ``analyzeFastStarTrackerImage()``.
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

    if 0 in ([len(sourceSet) for sourceSet in results]):
        if not silent:
            print('Some results contain no sources. Results are therefore fundamentally inconsistent'
                  ' and other checks cannot be run')
        return False

    consistent = True
    nSources = set([sourceSet[0].nSourcesInImage for sourceSet in results])
    if len(nSources) != 1:
        if not silent:
            print(f'Images contain a variable number of sources: {nSources}')
        consistent = False
    else:
        if not silent:
            n = nSources.pop()
            print(f'✅ All images contain the same nominal number of sources at detection stage: {n}')

    nSourcesCounted = set([len(sourceSet) for sourceSet in results])
    if len(nSourcesCounted) != 1:
        if not silent:
            print(f'WARNING: Number of actual sources in each sourceSet varies, got: {nSourcesCounted}.'
                  ' If some were manually removed you can ignore this')
        consistent = False
    else:
        if not silent:
            n = nSourcesCounted.pop()
            print(f'✅ All results contain the same number of actual sources per image: {n}')

    widths = set([sourceSet[0].parentImageWidth for sourceSet in results])
    heights = set([sourceSet[0].parentImageWidth for sourceSet in results])
    if len(widths) != len(heights) != 1:
        if not silent:
            print(f'Input images were of variable dimenions! {widths=}, {heights=}')
        consistent = False
    else:
        if not silent:
            print('✅ All input images were of the same dimensions')

    if not consistent:
        return False

    if not silent:
        # now the basic checks have passed, do some sanity checks on the
        # maximum deltas for the primary sources
        sources = [sourceSet[0] for sourceSet in results]
        dx = np.diff([s.centroidX for s in sources])
        dy = np.diff([s.centroidY for s in sources])
        maxMovementX = np.max(dx)
        maxMovementY = np.max(dy)
        print('Maximum centroid movement between images in (x, y) = '
              f'({maxMovementX:.2f}, {maxMovementY:.2f}) pix')

        fluxStd = np.nanstd([s.rawFlux for s in sources])
        fluxMean = np.nanmean([s.rawFlux for s in sources])
        print(f'Mean and stddev of fluxes = {fluxMean:.1f} ± {fluxStd:.1f} ADU')

    return consistent


def plotResults(results, sourceIndex=0, allowInconsistent=False):
    """Plot the centroid movements and fluxes etc for a set of results.

    Parameters
    ----------
    results : `dict` of `list` of
              `lsst.summit.utils.astrometry.starTrackerAnalysis.Source`
        A dict, keyed by sequence number, with each value being a list of the
        sources found in the image, e.g. as returned by
        ``analyzeFastStarTrackerImage()``.
    sourceIndex : `int`, optional
        If there is more than one source in every image, which source number
        should the plot be made for? Defaults to zero, which is the brightest
        source by default.
    allowInconsistent : `bool`, optional
        Make the plots even if the input results appear to be inconsistent?
    """
    consistent = checkResultConsistency(results.values(), silent=True)
    if not consistent and not allowInconsistent:
        checkResultConsistency(results.values(), silent=False)  # print the problem if we're raising
        raise ValueError('Inconsistent sources and allowInconsistent=False')

    sourceDict = {k: v[sourceIndex] for k, v in results.items()}
    seqNums = list(sourceDict.keys())
    sources = list(sourceDict.values())

    axisLabelSize = 18

    fig = plt.figure(figsize=(10, 16))
    ax1, ax2, ax3 = fig.subplots(3, sharex=True)
    fig.subplots_adjust(hspace=0)

    ax1.plot(seqNums, [s.rawFlux for s in sources], label='Raw Flux')
    ax1.plot(seqNums, [s.hsmFittedFlux for s in sources], label='Fitted Flux')
    ax1.set_ylabel('Flux (ADU)', size=axisLabelSize)
    ax1.legend()

    ax2.plot(seqNums, [s.centroidX for s in sources], label='Raw centroid x')
    ax2.plot(seqNums, [s.hsmCentroidX for s in sources], label='Fitted centroid x')
    ax2.set_ylabel('x-centroid (pixels)', size=axisLabelSize)
    ax2.legend()

    ax3.plot(seqNums, [s.centroidY for s in sources], label='Raw centroid y')
    ax3.plot(seqNums, [s.hsmCentroidY for s in sources], label='Fitted centroid y')
    ax3.set_ylabel('y-centroid (pixels)', size=axisLabelSize)
    ax3.set_xlabel('SeqNum', size=axisLabelSize)
    ax3.legend()

    fig = plt.figure(figsize=(10, 10))
    ax4 = fig.subplots(1)

    # TODO: add a linearly spaced color map to these points to show progression
    # of seqNums
    # npts = len(seqNums)
    # spacing = np.linspace(0, 1, npts)
    # cmap = matplotlib.colors.Colormap('jet', npts)

    ax4.scatter([s.centroidX for s in sources], [s.centroidY for s in sources])
    ax4.set_xlabel('x-centroid (pixels)', size=axisLabelSize)
    ax4.set_ylabel('y-centroid (pixels)', size=axisLabelSize)
