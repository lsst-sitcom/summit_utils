import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

import galsim

from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

import lsst.afw.math as afwMath

from dataclasses import dataclass, field
from typing import List

from statistics import mean, stdev


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
    return slice(None, None), slice(strip_num * strip_height, (strip_num + 1) * strip_height)


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
    strip_height = md.get("STUTTER ROWS")
    strip_exp_time = md.get("STUTTER DELAY")
    n_strips = md.get("STUTTER NSHIFTS")
    properties = (strip_height, n_strips, strip_exp_time)
    if not all(properties):
        raise RuntimeError(f"Failed to calculated stutter properties in image: {properties}")
    return properties


class StutteredImageAnalyzer:
    """Analysis class for stuttered images.

    Class contains options for analyzing stuttered images.

    For getting the exposure, you can either grab a quick look exposure or
    run_isr with this class. For doing background subtraction, you can subtract
    background by median or sigma clipping
    """

    def __init__(self):
        """Initialize stuttered image analyzer."""
        self.log = logging.getLogger(type(self).__name__)

    def subtract_background(self, exp, method="median"):
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
        if method not in ("median", "sigmaclip"):
            raise ValueError(f"Unrecognised background subtraction method {method}")
        strip_height, n_strips, _ = get_stutter_properties(exp)

        if method == "sigmaclip":
            sctrl = afwMath.StatisticsControl()
            sctrl.setNumSigmaClip(3)
            sctrl.setNumIter(3)
            statTypes = afwMath.MEANCLIP

        backgrounds = []
        for strip_num in range(2 * n_strips):  # 2x for the two halves
            strip = get_strip(exp, strip_num, strip_height)

            if method == "median":
                background = np.nanmedian(strip.image.array)
            elif method == "sigmaclip":
                stats = afwMath.makeStatistics(strip.maskedImage, statTypes, sctrl)
                background, _ = stats.getResult(afwMath.MEANCLIP)

            backgrounds.append(background)
            strip.image.array -= background
        return backgrounds

    def detect_sources(
        self,
        exp,
        threshold=100,
        do_plot=True,
        do_background_subtract=True,
        sigma=3,
        min_peak_width=12,
        min_prominence=2,
        min_distance=50,
        min_height=5,
    ):
        """Detect all sources in the image.

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
            x_profile = np.nanmean(
                exp.image.array[
                    image_half * (strip_height * n_strips) : (image_half + 1) * (strip_height * n_strips), :
                ],
                axis=0,
            )
            filtered_x_profile = gaussian_filter(x_profile, sigma=sigma)

            # detect sources in the image half
            peak_locations, peak_information = find_peaks(
                filtered_x_profile,
                height=min_height,
                distance=min_distance,
                width=min_peak_width,
                prominence=min_prominence,
            )

            half_sources = pd.DataFrame(
                data={
                    "x_peak": peak_locations,
                    "peak_value": peak_information["peak_heights"],
                    "image_half": np.zeros(len(peak_locations)) + image_half,
                }
            )
            if image_half == 0:
                sources = half_sources
            else:
                sources = pd.concat([sources, half_sources], ignore_index=True)
            if do_plot:
                # plot the mean x values and the found peaks
                plt.figure(figsize=(10, 5))
                plt.plot(x_profile)
                plt.plot(filtered_x_profile)
                plt.plot(peak_locations, np.ones_like(peak_locations) * 50, "x")
                plt.xlim(left=0)
                plt.yscale("log")

                # plot the image with the peaks marked
                plt.figure(figsize=(10, 10))
                plt.imshow(
                    np.arcsinh(
                        10
                        * exp.image.array[
                            image_half
                            * (strip_height * n_strips) : (image_half + 1)
                            * (strip_height * n_strips),
                            :,
                        ]
                    )
                    / 10,
                    cmap="magma",
                    origin="lower",
                )
                plt.plot(peak_locations, np.ones_like(peak_locations) * strip_height * n_strips / 2, "bx")
        return sources

    def find_mean_y_position(
        self, exp, sources, n_strips, strip_height, sigma=2, y_threshold=5, do_plot=True
    ):
        """
        Find the mean y position of each source.

        Parameters:
        - half_section: The input 2D array.
        - sources: A list of indices of the columns you're interested in.
        - strip_height: The length of each strip.
        - sigma: Standard deviation for the Gaussian filter.

        Returns:
        - The sources dataframe with a mean y_peak location
        """

        sources["y_peak"] = np.nan

        for image_half in range(2):
            # define the relevant part of the image
            half_section = exp.image.array[
                image_half * (strip_height * n_strips) : (image_half + 1) * (strip_height * n_strips), :
            ]
            if image_half == 1:
                # flip the direction of the second half of the image
                half_section = half_section[::-1]

            # Step 1: Filter the image by columns of interest
            filtered_array = half_section[:, sources.x_peak.loc[sources.image_half == image_half]]

            # Step 2: Sum the even strips
            reshaped_array = filtered_array[: n_strips * strip_height].reshape(n_strips, strip_height, -1)
            summed_array = np.sum(reshaped_array, axis=0)

            # Step 3: Smooth the columns using Gaussian filter
            smoothed_array = gaussian_filter(summed_array, sigma=sigma, axes=0)

            # Step 4: Find the max value
            max_indices = np.argmax(smoothed_array, axis=0)
            max_values = np.max(smoothed_array, axis=0)

            sources.loc[sources.image_half == image_half, "y_peak"] = max_indices

            # Step 5: Remove all sources where there is not sufficient contrast
            min_values = np.min(smoothed_array, axis=0)
            min_values[min_values < 0] = 1  # getting rid of negative values

            sources = sources.drop(
                sources.loc[(sources.image_half == image_half)].index[(max_values / min_values < y_threshold)]
            ).reset_index(drop=True)

            if do_plot:
                plt.figure()
                plt.plot(smoothed_array)
                plt.plot(max_indices, max_values, "x")
                plt.yscale("log")
                plt.show()

                # plot the image with the remaining peaks marked
                plt.figure(figsize=(10, 10))
                plt.imshow(
                    np.arcsinh(
                        10
                        * exp.image.array[
                            image_half
                            * (strip_height * n_strips) : (image_half + 1)
                            * (strip_height * n_strips),
                            :,
                        ]
                    )
                    / 10,
                    cmap="magma",
                    origin="lower",
                )
                plt.plot(
                    sources.x_peak.loc[sources.image_half == image_half],
                    np.ones_like(sources.x_peak[sources.image_half == image_half])
                    * strip_height
                    * n_strips
                    / 2,
                    "bx",
                )

        return sources

    def catalog_all_sources(
        self,
        exp,
        sources,
        n_strips,
        strip_height,
        do_plot=True,
        do_plot_all=False,
        sigma=2,
        do_background_subtract=True,
        max_mom2_iter=400,
        fwhm=20,
    ):
        """Make a catalog of all sources in the image

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

        # now create the appropriate dataclasses for each source
        stuttered_object_catalog = []

        # loop through each strip and find sources based on their x position
        for image_half in range(2):
            if do_plot:
                centroid_x = []
                centroid_y = []
            # define the relevant part of the image
            half_section = exp.image.array[
                image_half * (strip_height * n_strips) : (image_half + 1) * (strip_height * n_strips), :
            ]
            if image_half == 1:
                # flip the direction of the second half of the image
                half_section = half_section[::-1]
            for index, source in sources.loc[sources.image_half == image_half].iterrows():
                x_min = int(source.x_peak - strip_height / 2)
                x_max = int(source.x_peak + strip_height / 2)
                for strip in range(n_strips):
                    if strip == 0:
                        y_max = int(source.y_peak + strip_height / 2)
                        if source.y_peak < strip_height / 2:
                            y_min = 0
                        else:
                            y_min = int(source.y_peak - strip_height / 2)
                    elif strip == n_strips:
                        y_min = int((strip_height * strip) + source.y_peak - strip_height / 2)
                        if source.y_peak > strip_height / 2:
                            y_max = int(strip_height * strip - 1)
                        else:
                            y_max = int((strip_height * strip) + source.y_peak + strip_height / 2)
                    else:
                        y_min = int((strip_height * strip) + source.y_peak - strip_height / 2)
                        y_max = int((strip_height * strip) + source.y_peak + strip_height / 2)

                    footprint = half_section[y_min:y_max, x_min:x_max]

                    new_params = galsim.hsm.HSMParams(max_mom2_iter=max_mom2_iter)
                    galImage = galsim.Image(footprint)

                    try:
                        moments = galsim.hsm.FindAdaptiveMom(
                            galImage, guess_sig=fwhm / 2, hsmparams=new_params
                        )

                        if moments.moments_amp > 0:
                            stuttered_object = StutteredObject(
                                index,
                                image_half,
                                strip,
                                source.x_peak,
                                source.y_peak,
                                source.peak_value,
                                moments.moments_amp,
                                moments.moments_centroid.x + x_min - 1,
                                moments.moments_centroid.y + y_min - 1,
                                moments.moments_centroid.y - 1 + source.y_peak - strip_height / 2,
                                moments.moments_sigma,
                                moments.moments_rho4,
                                moments,
                            )
                            stuttered_object_catalog.append(stuttered_object)

                            if do_plot:
                                centroid_x.append(moments.moments_centroid.x + x_min - 1)
                                centroid_y.append(moments.moments_centroid.y + y_min - 1)
                        else:
                            self.log.debug(f"Total intensity of source {source}, strip {strip},  is negative. Skipping.")

                        if do_plot_all:
                            plt.figure()
                            plt.title(f'source {source}, strip {strip}')
                            plt.imshow(footprint)
                            plt.plot(
                                moments.moments_centroid.x - 1,
                                moments.moments_centroid.y - 1,
                                "rx",
                                markersize=15,
                            )
                            plt.show()

                    except RuntimeError:
                        self.log.debug(f'Failed to fit image of source {source}, strip {strip}.')
                        if do_plot_all:
                            plt.figure()
                            plt.imshow(footprint)
                            plt.show()
                        pass

            if do_plot:
                plt.figure(figsize=(10, 20))
                plt.imshow(np.arcsinh(10 * half_section) / 10, cmap="magma", origin="lower")
                plt.plot(centroid_x, centroid_y, "x")
                plt.show()

        return stuttered_object_catalog

    def run(
        self,
        exp,
        data_id=np.nan,
        do_plot=True,
        do_plot_all=False,
        method="sigmaclip",
        threshold=100,
        sigma=2,
        do_background_subtract=True,
        flux_threshold=3,
        min_object_number=5,
        y_threshold=5,
    ):
        """Fully analyze a stuttered image

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
        stuttered_image : `dataclass`
            Dataclass with information about the image and sources therein.
        """
        strip_height, n_strips, strip_exp_time = get_stutter_properties(exp)

        # first subtract backgrounds and detect sources
        backgrounds = self.subtract_background(exp, method=method)
        if do_plot:
            plt.plot(backgrounds)
            plt.show()

        sources = self.detect_sources(exp, threshold=threshold, do_plot=do_plot)
        sources = self.find_mean_y_position(
            exp, sources, n_strips, strip_height, sigma=sigma, do_plot=do_plot, y_threshold=y_threshold
        )

        stuttered_object_catalog = self.catalog_all_sources(
            exp, sources, n_strips, strip_height, do_plot_all=do_plot_all
        )

        filtered_stuttered_object_catalog = self.filter_objects(
            sources,
            stuttered_object_catalog,
            flux_threshold=flux_threshold,
            min_object_number=min_object_number,
        )

        if do_plot:
            for image_half in range(2):
                # define the relevant part of the image
                half_section = exp.image.array[
                    image_half * (strip_height * n_strips) : (image_half + 1) * (strip_height * n_strips), :
                ]
                if image_half == 1:
                    # flip the direction of the second half of the image
                    half_section = half_section[::-1]

                plt.figure(figsize=(10, 20))
                plt.imshow(np.arcsinh(10 * half_section) / 10, cmap="magma", origin="lower")
                plt.plot(
                    [
                        source.fitted_centroid_x
                        for source in filtered_stuttered_object_catalog
                        if source.image_half == image_half
                    ],
                    [
                        source.fitted_centroid_y
                        for source in filtered_stuttered_object_catalog
                        if source.image_half == image_half
                    ],
                    "x",
                )
                plt.show()

        return StutteredImage(
            filtered_stuttered_object_catalog, n_strips, strip_height, strip_exp_time, backgrounds, data_id
        )

    def filter_objects(self, sources, stuttered_object_catalog, flux_threshold=3, min_object_number=5):
        """Remove all objects that are too dim relative to the other objects,
        and all sources where there are too few

        Parameters
        ----------

        sources : `pandas dataframe`
            Pandas dataframe with x and y location within single strip and
            column saying which half it's in.

        stuttered_object_catalog : `dataclass`
            Dataclass with all detected sources in it

        flux_threshold : `float`
            flux threshold at which to consider the source

        min_object_number : `integer`
            The minimum number of objects for which to do calculations

        Returns
        -------
        filtered_stuttered_object_catalog : `dataclass`
            Stuttered object catalog with bad sources removed
        """

        filtered_stuttered_object_catalog = []

        for index in sources.index:
            # Filter out sources with flux outside threshold.
            fluxes = [
                object.fitted_flux for object in stuttered_object_catalog if object.source_number == index
            ]

            if len(fluxes) > 0:
                mean_flux = mean(fluxes)
                std_flux = stdev(fluxes)

                # add object to the catalog only if flux is within threshold
                # or it's the first object in the strip and too bright
                source_objects = [
                    object
                    for object in stuttered_object_catalog
                    if (
                        (
                            (object.source_number == index)
                            and (np.abs(object.fitted_flux - mean_flux) <= flux_threshold * std_flux)
                        )
                        or ((object.fitted_flux == fluxes[-1]) and (fluxes[-1] > mean_flux))
                    )
                ]

                # convert the strips in this list to stutter number,
                # where 0 is the final stutter, or the highest strip number
                first_strip = source_objects[-1].strip

                for object in source_objects:
                    object.stutter_number = first_strip - object.strip

                if len(source_objects) >= min_object_number:
                    # only add objects if there are enough of them
                    source_objects = self.calculate_source_variation(source_objects)

                    filtered_stuttered_object_catalog.append(source_objects)

        # flatten list
        filtered_stuttered_object_catalog = [
            object for source_list in filtered_stuttered_object_catalog for object in source_list
        ]

        return filtered_stuttered_object_catalog

    def calculate_source_variation(self, object_list):
        """Find the changes in each source position, flux relative to its mean

        Parameters
        ----------

        stuttered_object_catalog : `dataclass`
            Dataclass with all detected sources in it

        Returns
        -------
        filled_stuttered_object_catalog : `dataclass`
            Stuttered object catalog with information about the individual
            source relative to the mean source position and flux
        """

        # set mean_flux, std_flux, and differential flux
        fluxes = [object.fitted_flux for object in object_list]
        mean_flux = mean(fluxes)
        std_flux = stdev(fluxes)

        x_pos = [object.fitted_centroid_x for object in object_list]
        mean_x = mean(x_pos)
        std_x = stdev(x_pos)

        y_pos = [object.fitted_centroid_y_in_strip for object in object_list]
        mean_y = mean(y_pos)
        std_y = stdev(y_pos)

        for object in object_list:
            object.mean_flux = mean_flux
            object.std_flux = std_flux
            object.differential_flux = object.fitted_flux / mean_flux

            object.fitted_mean_x = mean_x
            object.fitted_std_x = std_x
            object.differential_centroid_x = object.fitted_centroid_x - mean_x

            object.fitted_mean_y = mean_y
            object.fitted_std_y = std_y
            object.differential_centroid_y = object.fitted_centroid_y_in_strip - mean_y

        return object_list


@dataclass
class StutteredObject:
    """Class contains information about a source"""

    # general source info
    source_number: float = np.nan
    image_half: float = np.nan
    strip: float = np.nan
    mean_x: float = np.nan
    mean_y: float = np.nan
    mean_peak: float = np.nan

    # numbers from the hsm moments fit
    fitted_flux: float = np.nan
    fitted_centroid_x: float = np.nan
    fitted_centroid_y: float = np.nan
    fitted_centroid_y_in_strip: float = np.nan
    fitted_sigma: float = np.nan
    fitted_rho4: float = np.nan
    moments: galsim.hsm.ShapeData = None  # keep the full fit even though we pull some things out too
    fitted_centroid_error: float = fitted_sigma / np.sqrt(fitted_flux)

    # variation relative to source mean
    stutter_number: float = np.nan
    fitted_mean_x: float = np.nan
    fitted_mean_y: float = np.nan
    mean_flux: float = np.nan
    fitted_std_x: float = np.nan
    fitted_std_y: float = np.nan
    std_flux: float = np.nan
    differential_flux: float = np.nan
    differential_centroid_x: float = np.nan
    differential_centroid_y: float = np.nan


@dataclass
class StutteredImage:
    """Class contains all the information about a stuttered image"""

    source_catalog: List[StutteredObject]
    n_strips: int = 40
    strip_height: int = np.nan
    strip_exp_time: float = np.nan
    backgrounds: list = field(default_factory=list)
    data_id: int = np.nan
