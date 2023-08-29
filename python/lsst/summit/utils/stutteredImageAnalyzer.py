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
from photutils.detection import find_peaks


class StutteredImageAnalyzer():
    """Analysis class for stuttered images.

    Class contains options for analyzing stuttered images.

    For getting the exposure, you can either grab a quick look exposure or
    run_isr with this class. For doing background subtraction, you can subtract
    background by median or sigma clipping
    """

    def __init__(self):
        """Initialize stuttered image analyzer. Make butler.
        """
        self.butler = butlerUtils.makeDefaultLatissButler(embargo=True)

    def get_stutter_properties(self, exp):
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

    def run_isr(self, data_id, do_flat=True, do_bias=True):
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

        raw = self.butler.get('raw', data_id)
        bias = self.butler.get('bias', data_id)
        defects = self.butler.get('defects', data_id)
        flat = self.butler.get('flat', data_id)
        exp = isrTask.run(raw, bias=bias, defects=defects, flat=flat).exposure
        return exp

    def get_strip(self, exp, strip_num, strip_height):
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
        strip_slice = self.get_strip_slice(strip_num, strip_height)
        return exp[strip_slice]

    def get_strip_slice(self, strip_num, strip_height):
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
        strip_height, n_strips, _ = self.get_stutter_properties(exp)

        if method == 'sigmaclip':
            sctrl = afwMath.StatisticsControl()
            sctrl.setNumSigmaClip(3)
            sctrl.setNumIter(3)
            statTypes = afwMath.MEANCLIP

        backgrounds = []
        for strip_num in range(2*n_strips):  # 2x for the two halves
            strip = self.get_strip(exp, strip_num, strip_height)

            if method == 'median':
                background = np.nanmedian(strip.image.array)
            elif method == 'sigmaclip':
                stats = afwMath.makeStatistics(strip.maskedImage, statTypes, sctrl)
                background, _ = stats.getResult(afwMath.MEANCLIP)

            backgrounds.append(background)
            strip.image.array -= background
        return backgrounds

    def detect_sources(self, exp, threshold=5000, do_plot=False):
        """ Detect all sources in the image.

        Find mean x and y positions of all sources that are above a given
        threshold.

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
        strip_height, n_strips, strip_exp_time = self.get_stutter_properties(exp)
        summed_strip = []
        sources = []
        for image_half in range(2):
            for strip_num in range(int(n_strips)):
                # sum all strips together
                strip = self.get_strip(exp, strip_num + n_strips * image_half, strip_height)
                if strip_num == 0:
                    summed_strip = strip.image.array
                else:
                    summed_strip += strip.image.array
            # detect sources in full strip
            peak_locations = find_peaks(summed_strip, threshold=threshold, box_size=strip_height)
            self.peak_locations = peak_locations
            half_sources = pd.DataFrame(data={'x_peak': peak_locations['x_peak'],
                                              'y_peak': peak_locations['y_peak'],
                                              'peak_value': peak_locations['peak_value'],
                                              'image_half': np.zeros(len(peak_locations)) + image_half})
            if image_half == 0:
                sources = half_sources
            else:
                sources = pd.concat([sources, half_sources], ignore_index=True)
            if do_plot:
                plt.figure(figsize=(20, 20))
                plt.imshow(np.arcsinh(10*summed_strip)/10, cmap="magma", origin="lower")
                plt.plot(sources.x_peak[sources.image_half == image_half],
                         sources.y_peak[sources.image_half == image_half], 'rx')
                plt.show()
            summed_strip = []
        return sources
