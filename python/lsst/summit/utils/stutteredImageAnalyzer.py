from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# rsp imports
import lsst.summit.utils.butlerUtils as butlerUtils
from lsst.ip.isr.isrTask import IsrTask
from lsst.daf.butler import Butler


import galsim

from lsst.afw.image import Image
from lsst.geom import Box2I, Point2I, Extent2I
from lsst.meas.extensions import shapeHSM
import lsst.afw.math as afwMath

from scipy.fft import fft
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

from dataclasses import dataclass
from typing import List


def get_strip(exp, strip_num, strip_height):
    """Extract the data for a single strip from the exposure.

    Parameters
    ----------
    exp : `lsst.afw.image.Exposure`
        The exposure to get the strip from.
    strip_num : `int`
        The strip number to get.
    strip_height : `int`
        The height of each strip in pixels.

    Returns
    -------
    strip : `lsst.afw.image.Exposure`
        The strip from the exposure.
    """
    strip_slice = get_strip_slice(strip_num, strip_height)
    return exp[strip_slice]


def get_strip_slice(strip_num, strip_height):
    """Get a slice object for a single strip.

    Get a slice object for a given strip, such that it can be used to
    directly de-reference the strip from the exposure using
    ``exp[stripSlice]``.

    Parameters
    ----------
    strip_num : `int`
        The strip number to get.
    strip_height : `int`
        The height of each strip in pixels.

    Returns
    -------
    strip_slice : `slice`
        The slice object for the strip.
    """
    return slice(None, None), slice(strip_num*strip_height, (strip_num+1) * strip_height)


def get_stutter_properties(exp):
    """Get the exposure time, strip height, and strip count
    from the header.
    Parameters
    ----------
    exp : `lsst.afw.image.Exposure`
        The exposure to get the properties from.

    Returns
    -------
    strip_height : `int`
        The height of each strip in pixels.
    n_strips : `int`
        The number of strips in the image.
    strip_exp_time : `float`
        The exposure time of the strips in seconds.
    """
    md = exp.getMetadata().toDict()
    strip_height = md.get('STUTTER ROWS')
    strip_exp_time = md.get('STUTTER DELAY')
    n_strips = md.get('STUTTER NSHIFTS')
    properties = (strip_height, n_strips, strip_exp_time)
    if not all(properties):
        raise RuntimeError(f'Failed to calculated stutter properties in image: {properties}')
    return properties


def run_isr(data_id, butler, do_flat=True, do_bias=True):
    """Run ISR on the raw image with appropriate parameters (if not looking
    at quick look exposures).
    Parameters
    ----------
    data_id: `dict`
        The day_obs, sequence number, detector, etc. for locating the image
        of interest.
    do_flat: `binary`
        Whether to apply flatfield in ISR.
    do_bias: `binary`
        Whether to do bias correction.

    Returns
    -------
    exp : `lsst.afw.image.Exposure`
        The exposure with ISR performed.
    """
    isrConfig = IsrTask.ConfigClass()
    isrConfig.doLinearize = False
    isrConfig.doBias = do_bias
    isrConfig.doFlat = do_flat
    isrConfig.doDark = False
    isrConfig.doFringe = False
    isrConfig.doDefect = True
    isrConfig.doWrite = False
    isrConfig.doSaturation = False
    isrConfig.doNanInterpolation = False
    isrConfig.doNanMasking = False
    isrConfig.doSaturation = False
    isrConfig.doSaturationInterpolation = False
    isrConfig.doWidenSaturationTrails = False

    isrTask = IsrTask(config=isrConfig)

    raw = butler.get('raw', data_id)
    bias = butler.get('bias', data_id)
    defects = butler.get('defects', data_id)
    flat = butler.get('flat', data_id)
    exp = isrTask.run(raw, bias=bias, defects=defects, flat=flat).exposure
    return exp


class StutteredImageAnalyzer():
    """Analysis class for stuttered images.

    Class contains options for analyzing stuttered images.

    For getting the exposure, you can either grab a quick look exposure or
    run_isr with this class. For doing background subtraction, you can subtract
    background by median or sigma clipping
    """

    def __init__(self):
        """Initialize stuttered image analyzer.
        """

    def subtract_background(self, exp, method='median'):
        """Subtract the background from the image in-place.

        The background is calculated by taking the median of sigma-clipped mean
        of each strip, and subtracting it from the image. The calculated
        background levels which were subtracted are returned.

        Parameters
        ----------
        exp : `lsst.afw.image.Exposure`
            The exposure to subtract the background from.
        method : `str`, optional
            The method to use to calculate the background. Can be either
            'median' or 'sigmaclip'. Default is 'median'.

        Returns
        -------
        backgrounds : `list` of `float`
            The background levels which were subtracted from each strip.
        """
        if method not in ('median', 'sigmaclip'):
            raise ValueError(f'Unrecognised background subtraction method {method}')
        strip_height, n_strips, _ = get_stutter_properties(exp)

        if method == 'sigmaclip':
            sctrl = afwMath.StatisticsControl()
            sctrl.setNumSigmaClip(3)
            sctrl.setNumIter(3)
            statTypes = afwMath.MEANCLIP

        backgrounds = []
        for strip_num in range(2*n_strips):  # 2x for the two halves
            strip = get_strip(exp, strip_num, strip_height)

            if method == 'median':
                background = np.nanmedian(strip.image.array)
            elif method == 'sigmaclip':
                stats = afwMath.makeStatistics(strip.maskedImage, statTypes, sctrl)
                background, _ = stats.getResult(afwMath.MEANCLIP)

            backgrounds.append(background)
            strip.image.array -= background
        return backgrounds

    def detect_sources(self, exp, threshold=100, do_plot=True, do_background_subtract=True, sigma=3,
                       min_peak_width=5, min_prominence=2, min_distance=50, min_height=0.5):
        """ Detect all sources in the image.

        Find mean x positions of all sources that are above a given
        threshold. Uses scipy method of find_peaks.

        Parameters
        ----------
        exp : `lsst.afw.image.Exposure`
            The exposure, probably with background subtracted, although this is
            not essential.

        Returns
        -------
        sources : `pandas dataframe`
            Pandas dataframe with x and y location within single strip and
            column saying which half it's in.
        """
        strip_height, n_strips, strip_exp_time = get_stutter_properties(exp)
        sources = []

        for image_half in range(2):
            # take the mean and smooth the image half
            x_profile = np.nanmean(exp.image.array[image_half*(strip_height*n_strips):(image_half+1) *
                                                   (strip_height*n_strips), :], axis=0)
            filtered_x_profile = gaussian_filter(x_profile, sigma=sigma)

            # detect sources in the image half
            peak_locations, peak_information = find_peaks(filtered_x_profile, height=min_height,
                                                          distance=min_distance, width=min_peak_width,
                                                          prominence=min_prominence)
            half_sources = pd.DataFrame(data={'x_peak': peak_locations,
                                              'peak_value': peak_information['peak_heights'],
                                              'image_half': np.zeros(len(peak_locations)) + image_half})
            if image_half == 0:
                sources = half_sources
            else:
                sources = pd.concat([sources, half_sources], ignore_index=True)
            if do_plot:
                # plot the mean x values and the found peaks
                plt.figure(figsize=(10, 5))
                plt.plot(x_profile)
                plt.plot(filtered_x_profile)
                plt.plot(peak_locations, np.ones_like(peak_locations)*50, 'x')
                plt.yscale('log')

                # plot the image with the peaks marked
                plt.figure(figsize=(10, 10))
                plt.imshow(np.arcsinh(10*exp.image.array[image_half*(strip_height*n_strips):(image_half+1) *
                                                         (strip_height*n_strips), :])/10, cmap="magma",
                           origin="lower")
                plt.plot(peak_locations, np.ones_like(peak_locations)*strip_height*n_strips/2, 'bx')
        return sources

    def catalog_all_sources(self, exp, do_plot=True, method='sigmaclip', threshold=100, sigma=2,
                            do_background_subtract=True):
        """ Make a catalog of all sources in the image

        Find mean x and y positions of all sources that are above a given
        threshold.

        Parameters
        ----------
        exp : `lsst.afw.image.Exposure`
            The exposure, probably with background subtracted, although this is
            not essential.

        sources : `pandas dataframe`
            Pandas dataframe with x and y location within single strip and
            column saying which half it's in.

        Returns
        -------
        source_catalog : `dataclass`
            Pandas dataframe with x and y location within single strip and
            column saying which half it's in. XXX it isn't yet
        """
        strip_height, n_strips, strip_exp_time = get_stutter_properties(exp)

        # first subtract backgrounds and detect sources
        backgrounds = self.subtract_background(exp, method=method)
        sources = self.detect_sources(exp, threshold=threshold, do_plot=do_plot)
        source_catalog = pd.DataFrame(data={'centroidX': np.full_like((3, 3), np.nan),
                                            'centroidY': np.full_like((3, 3), np.nan),
                                            'rawFlux': np.full_like((3, 3), np.nan)
                                            })

        # now loop through each strip and look for the sources based on their x position
        for image_half in range(2):
            for strip_num in range(n_strips - 1):
                # XXX is this symmetric around half images?
                strip = get_strip(exp, strip_num + n_strips * image_half, strip_height)
                for x_peak in sources.x_peak:
                    x = 2

        return source_catalog, backgrounds

    def filter_brightest_sources(self, sources, distance_threshold):
        """
        Retain only the brightest sources if two or more sources are too close together.

        Args:
        - df (DataFrame): Input dataframe with columns 'x' (for x-coordinates) and 'brightness'.
        - distance_threshold (float): Distance threshold to determine closeness of sources.

        Returns:
        - DataFrame: Filtered dataframe retaining only the brightest sources.
        """
        # Sort by brightness in descending order
        sources = sources.sort_values(by='peak_value', ascending=False).reset_index()

        to_keep = []

        while not sources.empty:
            # Consider the brightest source first
            current_x = sources.iloc[0]['x_peak']

            # Determine sources that are too close to the current source
            mask = (abs(sources['x_peak'] - current_x) < distance_threshold) & (sources.image_half == sources.loc[0]['image_half'])
            print('mask', mask)
            # Add the current (brightest) source's index to the list of sources to keep
            to_keep.append(sources[mask].index[0])
            print('keep', to_keep)
            # Remove all sources that were too close
            sources = sources[~mask]

        # Filter the original dataframe to retain only the selected sources
        filtered_sources = sources.loc[sources.index.isin(to_keep)].sort_values(by='image_half').reset_index(drop=True)

        return filtered_sources


@dataclass
class StutteredImage:
    """Class contains all the information about a stuttered image"""
    n_strips: int = np.nan
    strip_height: int = np.nan
    strip_exp_time: float = np.nan
    backgrounds: list
    source_catalog: StutteredImageSourceCatalog


@dataclass
class StutteredImageSourceCatalog:
    """Class is a list of all stuttered objects in an image"""
    sources: List(StutteredObject)


@dataclass
class StutteredObject:
    """Class that is a list of all detected sources for a given object"""


@dataclass
class StutteredSource:
    """
    # raw numbers
    centroidX: float = np.nan  # in image coordinates
    centroidY: float = np.nan  # in image coordinates
    rawFlux: float = np.nan
    nPix: int = np.nan
    # bbox: geom.Box2I = None
    cutout: np.array = None
    localCentroidX: float = np.nan  # in cutout coordinates
    localCentroidY: float = np.nan  # in cutout coordinates

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
    """

    def __repr__(self):
        """Print everything except the full details of the moments."""
        retStr = ''
        for itemName in self.__slots__:
            v = getattr(self, itemName)
            if isinstance(v, int):  # print ints as ints
                retStr += f'{itemName} = {v}\n'
            elif isinstance(v, float):  # but round floats at 3dp
                retStr += f'{itemName} = {v:.3f}\n'
            elif itemName == 'moments':  # and don't spam the full moments
                retStr += f'moments = {type(v)}\n'
            elif itemName == 'bbox':  # and don't spam the full moments
                retStr += f'bbox = lsst.geom.{repr(v)}\n'
            elif itemName == 'cutout':  # and don't spam the full moments
                if v is None:
                    retStr += 'cutout = None\n'
                else:
                    retStr += f'cutout = {type(v)}\n'
        return retStr
