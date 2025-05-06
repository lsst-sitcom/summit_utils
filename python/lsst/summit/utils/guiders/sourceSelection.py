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

__all__ = [
    "starGuideFinder"
]

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from astropy.nddata import CCDData, Cutout2D
from astropy.stats import mad_std
from astropy.stats import sigma_clipped_stats, SigmaClip

from photutils.detection import DAOStarFinder
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_threshold, detect_sources
from photutils.utils import circular_footprint

from lsst.obs.lsst.cameraTransforms import LsstCameraTransforms
from lsst.summit.utils.guiders.transformation import pixel_to_focal
from lsst.summit.utils.guiders.reading import readGuiderData

# To avoid warnings
# import logging
# import astroquery
# # 1) Grab the astroquery logger
# logger = logging.getLogger('astroquery')
# # 2) Raise its level to ERROR (so WARNINGs are hidden)
# logger.setLevel(logging.ERROR)

class starGuideFinder:
    """
    Class to find stars in the Guider data.

    Example
    -------
    >>> from lsst.obs.lsst import LsstCam
    >>> from read import readGuiderData
    >>> from starGuideFinder import starGuideFinder
    
    >>> seqNum, dayObs, raft, ccd = 591, 20250425, "R40", "SG0"
    >>> camera = LsstCam.getCamera()
    >>> reader = readGuiderData(seqNum, dayObs, raft, ccd)
    
    >>> finder = starGuideFinder(reader, camera)
    >>> finder.run()
    >>> print("Check the reference catalog, sorted by SNR")
    >>> finder.ref_catalog
    
    >>> finder.plot_stacked_sources()
    >>> finder.plot_drifts_with_errors()
    >>> finder.plot_scatter_stamp()
    
    """
    def __init__(self, readerObject, camera, fwhm=10.0):
        """
        Initialize the starGuideFinder class.
        Parameters
        ----------
        readerObject : readGuiderData
            Instance of the readGuiderData class.
        camera : Camera
            Instance of the Camera class.
        """
        self.reader = readerObject
        self.fwhm = fwhm

        # Set the ROI parameters
        self.roiRow = self.reader.roiRow
        self.roiCol = self.reader.roiCol
        self.roiRows = self.reader.roiRows
        self.roiCols = self.reader.roiCols
        
        # Set the number of stamps
        self.nStamps = self.reader.nStamps
        self.timestamp = self.reader.timestamp

        # The amplifier name
        self.ampName = self.reader.ampName

        # Set the camera object
        self.camera = camera
        self.detector = self.camera[self.reader.key]
        self.lct = LsstCameraTransforms(self.camera, self.reader.key)

        # Get the lowest corner of the amplifier
        self.get_amplifier_lowest_corner()
        pass

    def get_amplifier_lowest_corner(self):
        """
        Get the lowest corner of the amplifier in CCD coordinates.
        Returns
        -------
        tuple
            The lowest corner of the amplifier in CCD coordinates.
        """
        a, b = self.lct.ampPixelToCcdPixel(self.roiCol,self.roiRow,self.ampName)
        c, d = self.lct.ampPixelToCcdPixel(self.roiCol+self.roiCols,self.roiRow+self.roiRows,self.ampName)
        begin_x, begin_y = min([a,c]), min([b,d])
        self.min_x = begin_x
        self.min_y = begin_y
        pass
    
    def run(self):
        """
        Run the star detection and tracking process.

        Returns
        -------
        motions : ndarray
            Array of star offsets across all stamps.
        magOffsets : ndarray
            Array of magnitude offsets for all stars across all stamps.
        """
        # Stack the images
        stacked_image = self.stack_guider_images()

        # Build the reference catalog
        self.build_ref_catalog()

        # Track motion for all stars
        motions, magOffsets = self.track_stars()
        return motions, magOffsets

    def track_stars(self, threshold_sigma=5.0, roundness_max=0.5):
        """
        Track all the stars across all stamps.

        Parameters
        ----------
        threshold_sigma : float
            Detection threshold (sigma above background).

        Returns
        -------
        stars : list
            List of detected stars in each stamp.
        """
        nstars = len(self.ref_catalog)
        motions = np.zeros((nstars, 2, self.nStamps))
        magOffsets = np.zeros((nstars, self.nStamps))
        
        for star_index in range(nstars):
            offsets = self.track_star_motion(star_index, 
                                             threshold_sigma=threshold_sigma, 
                                             roundness_max=roundness_max)
            dx = np.array(offsets)[:,0]
            dy = np.array(offsets)[:,1]
            mag_off = np.array(offsets)[:,2]

            # Save the motion data
            motions[star_index] = np.c_[dx, dy].T
            magOffsets[star_index] = mag_off

        self.motions = motions
        self.magOffsets = magOffsets
        return motions, magOffsets
    
    def track_star_motion(self, star_id, threshold_sigma=5.0, roundness_max=0.5):
        """
        Track the motion of a specific star across all stamps.

        Parameters
        ----------
        star_id : int
            ID of the star to track.

        Returns
        -------
        motion_data : list
            List of tuples containing (stamp_index, x, y) for the tracked star.
        """
        motion_data = []

        # Get the star position
        star = self.ref_catalog.iloc[star_id]
        x = star['xcentroid']
        y = star['ycentroid']
        
        for stamp_index in range(self.nStamps):
            # Take the stamp
            stamp = self.image_list[stamp_index]
            
            # Subtract the background
            isr = stamp - self.sky_bkg.background

            # make cutout of the star
            cutout = Cutout2D(isr, (x, y), size=50, mode='partial', fill_value=np.nan)

            # Stats
            mean, median, std = sigma_clipped_stats(cutout.data, sigma=3.0)

            # Do source detection
            sources = detect_stars_filtered(cutout.data, self.fwhm, threshold_sigma, 
                                            roundness_max=roundness_max, median=median, std=std)

            # get the highest flux star
            if sources is not None:
                sources = sources[sources['flux'] == np.max(sources['flux'])]
                x_star = sources['xcentroid'].data[0]
                y_star = sources['ycentroid'].data[0]
                mag_diff = -2.5 * np.log(sources['flux'].data[0] / star['flux'])
                # shift to original image coordinates
                x_star += cutout.xmin_original
                y_star += cutout.ymin_original

                motion_data.append((x_star-x, y_star-y, mag_diff))
            else:
                # print(f"No stars found in stamp {stamp_index} for star ID {star_id}.")
                motion_data.append((np.nan, np.nan, np.nan))

        return motion_data

    def stack_guider_images(self):
        """
        Stack guider images 

        Returns
        -------
        stacked_image : 2D numpy array
            Stacked image of the detected stars.
        """
        # Stack the images using ccdproc
        # Read the images from the reader
        image_list = [self.reader.read(i) for i in range(self.nStamps)]        
        stacked = np.nanmedian(np.array(image_list), axis=0)
        
        # TODO: Check if ccdproc is installed
        # ccd_list = [CCDData(img, unit=u.adu) for img in image_list]
        # combiner = ccdp.Combiner(ccd_list)
        # combiner.sigma_clipping(func=np.ma.median, sigma=5.0)
        # stacked_ccd = combiner.median_combine()
        # stacked = stacked_ccd.data

        self.image_list = image_list
        self.stacked = stacked
        return stacked

    def build_ref_catalog(self, threshold_sigma=5.0, snr_threshold=20, edge_size=20):
        """
        Build a reference catalog of stars from the stacked image.

        Parameters
        ----------
        threshold_sigma : float
            Detection threshold (sigma above background).

        Returns
        -------
        ref_catalog : astropy Table
            Reference catalog of detected stars.
        """
        # Stack the images and detect stars
        if self.stacked is None:
            stacked_image = self.stack_guider_images()
        else:
            stacked_image = self.stacked

        # Build the background model
        median, mean, std, mask = background_model(stacked_image, fwhm=self.fwhm)
        self.bias = median
        self.noise_ref = std
        self.mask_ref = mask

        # model the background
        self.sky_bkg = Background2D(stacked_image, box_size=50, filter_size=3, bkg_estimator=MedianBackground(), mask=mask)

        # Detect stars in the stacked image
        # and filter out streaks
        # Use the background model to subtract the background
        isr = stacked_image - self.sky_bkg.background
        ref_catalog = detect_stars_filtered(isr, fwhm=self.fwhm, threshold_sigma=threshold_sigma, 
                                            roundness_max=0.2, median=0, std=std)


        if ref_catalog is None or len(ref_catalog) == 0:
            print("No stars found in the reference catalog.")
            return None

        # Save the reference catalog
        self.ref_catalog = ref_catalog[["xcentroid", "ycentroid", "flux", "flux_err", "snr", "roundness1", "roundness2"]].to_pandas()

        # Filter out sources that are too close to the edges
        # and have low SNR
        self.filter_ref_catalog(snr_threshold=snr_threshold)
        self.mask_edge_ref_catalog(edge=edge_size)
        
        # Add additional information to the reference catalog
        self.add_ref_catalog_info(median, std)

        self.ref_catalog = self.ref_catalog.sort_values(by='snr', ascending=False)
        self.ref_catalog = self.ref_catalog.reset_index(drop=True)

    def filter_ref_catalog(self, snr_threshold=20):
        """
        Filter the reference catalog based on SNR.

        Parameters
        ----------
        snr_threshold : float
            Minimum SNR threshold for stars to be included in the catalog.
        """
        # Filter out sources with low SNR
        self.ref_catalog = self.ref_catalog[self.ref_catalog['snr'] > snr_threshold]
        
    def mask_edge_ref_catalog(self, edge=10):
        """
        Mask the edges of the reference catalog.

        Parameters
        ----------
        edge : int
            Number of pixels from the edge to mask.
        """
        # Filter out sources that are too close to the edges
        xmax, ymax = self.stacked.shape
        x_max = xmax - edge
        y_max = ymax - edge
        x_min = edge
        y_min = edge
        self.edges_frame = (x_min, y_min, x_max, y_max)
        self.ref_catalog = self.ref_catalog[(self.ref_catalog['xcentroid'] > x_min) & (self.ref_catalog['xcentroid'] < x_max) &
                                            (self.ref_catalog['ycentroid'] > y_min) & (self.ref_catalog['ycentroid'] < y_max)]
        

    def add_ref_catalog_info(self, median, std):
        """
        Add additional information to the reference catalog.
        Parameters
        ----------
        median : float
            Median value of the background.
        std : float
            Standard deviation of the background.
        """
        # Add additional information to the reference catalog
        self.ref_catalog['bias'] = median
        self.ref_catalog['noise'] = std
        self.ref_catalog['timestamp'] = self.timestamp[0].iso
        self.ref_catalog['expId'] = self.reader.expId
        self.ref_catalog['ampName'] = self.ampName
        self.ref_catalog['stamp'] = -99
        self.ref_catalog['id'] = np.arange(len(self.ref_catalog))

        # Convert the star positions to CCD coordinates
        self.ref_catalog['xpixel'] = self.ref_catalog['xcentroid']+self.min_x
        self.ref_catalog['ypixel'] = self.ref_catalog['ycentroid']+self.min_y
        xfp, yfp = pixel_to_focal(self.ref_catalog['xpixel'], self.ref_catalog['ypixel'], self.detector)
        self.ref_catalog['xfp'] = xfp
        self.ref_catalog['yfp'] = yfp

    def get_cutout_star(self, star_id, size=50):
        """
        Get a cutout of a specific star.

        Parameters
        ----------
        star_id : int
            ID of the star to get the cutout for.
        size : int
            Size of the cutout.

        Returns
        -------
        cutout : Cutout2D
            Cutout of the star.
        """
        # Get the star position
        star = self.ref_catalog.iloc[star_id]
        x = star['xcentroid']
        y = star['ycentroid']

        # Get the cutout
        cutout = Cutout2D(self.stacked, (x, y), size=size, mode='partial', fill_value=np.nan)
        return cutout.data
    
    def get_cutout_stamp(self, stamp_id, size=50):
        """
        Get a cutout of a specific stamp for the best star (highest SNR).

        Parameters
        ----------
        stamp_id : int
            ID of the stamp to get the cutout for.
        
        size : int
            Size of the cutout.

        Returns
        -------
        cutout : Cutout2D
            Cutout of the star.
        """
        # Get the star position; the best star is the first one in the catalog
        star = self.ref_catalog.iloc[0]
        x = star['xcentroid']
        y = star['ycentroid']

        # Get the cutout
        cutout = Cutout2D(self.image_list[stamp_id], (x, y), size=size, mode='partial', fill_value=np.nan)
        return cutout.data
        
    def plot_stacked_sources(self,
                            lo=10,
                            hi=98,
                            marker_color='firebrick',
                            marker_size=12,
                            annotate_ids=False,
                            ax=None):
        """
        Show the stacked image with your reference‐catalog positions overlaid.

        Parameters
        ----------
        lo, hi : float
            Percentiles for contrast stretch on the stack.
        marker_color : str
            Color for the source markers.
        marker_size : float
            Marker size in points^2.
        annotate_ids : bool
            If True, draw each source's catalog ID next to the marker.
        ax : matplotlib Axes, optional
            If None, one will be created.

        Returns
        -------
        ax : matplotlib Axes
        """
        # 1) Build the stack if not already done
        stacked = self.stack_guider_images()

        # 2) Compute display limits
        vmin, vmax = np.nanpercentile(stacked, [lo, hi])

        # 3) Create axes if needed
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        # 4) Show the image
        im = ax.imshow(stacked,
                    origin='lower',
                    cmap='Greys',
                    vmin=vmin, vmax=vmax)

        # 5) Overlay markers at each reference position
        xs = self.ref_catalog['xcentroid']
        ys = self.ref_catalog['ycentroid']
        
        ax.plot(xs, ys,
                markersize=marker_size,
                marker='x',
                color=marker_color,
                linestyle='',
                label='Reference sources')

        # 6) Optionally annotate IDs
        if annotate_ids and 'id' in self.ref_catalog.colnames:
            for row in self.ref_catalog:
                ax.text(row['xcentroid'] + 3,
                        row['ycentroid'] + 3,
                        str(row['id']),
                        color=marker_color, fontsize=8)

        # 7) Clean up axes
        ax.set_xlim(0, stacked.shape[1])
        ax.set_ylim(0, stacked.shape[0])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_title("Stacked guider image with reference sources"+"\n"+f"({self.reader.key}, {self.reader.expId})")

        # Plot the edges of the frame
        # Unpack your edges
        x_min, y_min, x_max, y_max = self.edges_frame

        # Compute width and height
        width  = x_max - x_min
        height = y_max - y_min

        # Create a Rectangle with no fill, grey edge, dashed line
        rect = Rectangle(
            (x_min, y_min),    # lower-left corner
            width, height,
            fill=False,
            edgecolor=marker_color,
            linestyle='--',
            linewidth=2.5
        )

        # Add it to your axes
        ax.add_patch(rect)

        fig.tight_layout()
        return fig, ax


    def plot_drifts_with_errors(self,
                                motions=None,
                                stamp_axis=None,
                                figsize=(6,4),
                                fig=None,
                                ax=None,
                                **plot_kw):
        """
        Plot the median drift ± robust σ (from MAD) for ΔX and ΔY.

        Parameters
        ----------
        motions : ndarray (n_stars, 2, n_stamps), optional
            If None, will call self.track_stars().
        stamp_axis : array-like, optional
            X-axis values (e.g. 0,1,2,… or timestamps). Defaults to range(n_stamps).
        figsize : tuple
            Figure size.
        plot_kw : dict
            Additional keywords passed to plt.errorbar (markersize, alpha, etc.).

        Returns
        -------
        fig, ax : matplotlib objects
        """
        # 1) get the motions array
        if motions is None:
            if hasattr(self, 'motions'):
                motions = self.motions
            else:
                motions, _ = self.track_stars()
        nstars, _, nstamps = motions.shape

        # 2) define the x-axis
        if stamp_axis is None:
            stamp_axis = np.arange(nstamps)

        # 3) compute median and MAD-based sigma across stars, per stamp
        #    motions[:,0,:] is ΔX for all stars, all stamps
        med_dx = np.nanmedian(motions[:,0,:], axis=0)
        med_dy = np.nanmedian(motions[:,1,:], axis=0)

        sig_dx = mad_std(motions[:,0,:], axis=0, ignore_nan=True)
        sig_dy = mad_std(motions[:,1,:], axis=0, ignore_nan=True)

        sig_dx[np.isnan(sig_dx)] = np.nanmedian(sig_dx)
        sig_dy[np.isnan(sig_dy)] = np.nanmedian(sig_dy)

        # 4) do the errorbar plot
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        defaults = dict(fmt='o', capsize=3, markersize=5, alpha=0.8)
        defaults.update(plot_kw)

        ax.errorbar(stamp_axis, med_dx, yerr=sig_dx,
                    color='k', label='CCD ΔX', **defaults)
        ax.errorbar(stamp_axis, med_dy, yerr=sig_dy,
                    color='firebrick', label='CCD ΔY', **defaults)
        ax.axhline(0, color='grey', lw=1, ls='--')

        # --- new jitter annotation ---
        jitter_pix   = np.nanstd(motions)
        jitter_arcsec = jitter_pix * 0.2
        txt = f"Jitter (rms): {jitter_pix:.2f} pixel, {jitter_arcsec:.2f} arcsec "
        ax.text(0.02, 0.98, txt,
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=12, color='grey',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # 5) polish
        ax.set_xlabel("Stamp #")
        ax.set_ylabel("Median offset (pixels)")
        ax.set_title(f"Star drift over {nstamps} stamps ({nstars} stars)"+"\n"+f"({self.reader.key}, {self.reader.expId})")
        ax.legend(frameon=False)
        ax.grid(True, ls=':', color='grey', alpha=0.5)
        # fig.tight_layout()
        return fig, ax

    def plot_scatter_stamp(self,
                            magOffsets=None,
                            stamp_axis=None,
                            figsize=(8,5),
                            **plot_kw):
        """

        Returns
        -------
        fig, ax : matplotlib objects
        """
        # 1) get the motions array
        if magOffsets is None:
            motions, magOffsets = self.track_stars()
        nstars, nstamps = magOffsets.shape

        # 2) define the x-axis
        if stamp_axis is None:
            stamp_axis = np.arange(nstamps)

        # 4) do the errorbar plot
        fig, ax = plt.subplots(figsize=figsize)
        defaults = dict(fmt='o', capsize=3, markersize=5, alpha=0.8)
        defaults.update(plot_kw)

        for i in range(nstars):
            p = ax.plot(stamp_axis, magOffsets[i]-np.nanmedian(magOffsets[i]), alpha=0.5, ls='--', lw=0.5)
            c = p[0].get_color()
            ax.scatter(stamp_axis, magOffsets[i]-np.nanmedian(magOffsets[i]), color=c, alpha=0.75, label=f'Star {i+1}')
        ax.axhline(0, color='grey', lw=1, ls='--')
    
        
        # --- new jitter annotation ---
        jitter_pix   = mad_std(magOffsets, ignore_nan=True)
        # jitter_arcsec = jitter_pix * 0.2
        txt = f"\\sigma (rms): {jitter_pix:.2f} mag "
        ax.text(0.02, 0.98, txt,
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=12, color='grey',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # 5) polish
        ax.set_xlabel("Stamp #")
        ax.set_ylabel("Mag Offset: stamp-ref [mag]")
        ax.set_title(f"Star flux variation over {nstamps} stamps ({nstars} stars)"+"\n"+f"({self.reader.key}, {self.reader.expId})")
        ax.legend(frameon=False, ncol=2)
        ax.grid(True, ls=':', color='grey', alpha=0.5)
        fig.tight_layout()
        return fig, ax

def stats_background(image, fwhm=10.0):
    """
    Compute the background statistics of an image.

    Parameters
    ----------
    image : 2D array
        Input image.
    fwhm : float
        FWHM for star detection.

    Returns
    -------
    mean, median, std : float
        Mean, median, and standard deviation of the background.
    """
    sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    return mean, median, std

def background_model(image, fwhm=10.0):
    sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    threshold = detect_threshold(image, nsigma=2.0, sigma_clip=sigma_clip)
    segment_img = detect_sources(image, threshold, npixels=10)
    footprint = circular_footprint(radius=1*fwhm)
    
    mask = segment_img.make_source_mask(footprint=footprint)
    mean, median, std = sigma_clipped_stats(image, sigma=3.0, mask=mask)
    streak_mask = np.abs(image-median)>2*std
    mean, median, std = sigma_clipped_stats(image, sigma=3.0, mask=mask|streak_mask)
    final_mask = mask|streak_mask
    return mean, median, std, final_mask
        
def detect_stars_filtered(image, fwhm=10.0, threshold_sigma=5.0, roundness_max=1.5, median=None, std=None):
    """
    Detect stars and filter out elongated sources (e.g., streaks).

    Parameters
    ----------
    image : 2D array
        Input image.
    fwhm : float
        FWHM for star detection.
    threshold_sigma : float
        Detection threshold (sigma above background).
    roundness_max : float
        Maximum allowed elongation.

    Returns
    -------
    filtered_sources : astropy Table
        Table of detected sources after filtering.
    """
    gain = 1.0  # Gain in e-/ADU
    if median is None or std is None:
        mean, median, std, _ = background_model(image, fwhm)
    
    bkg_sigma = std
    threshold = threshold_sigma * bkg_sigma

    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold, roundhi=roundness_max)
    sources = daofind(image - median)
    
    if sources is None or len(sources) == 0:
        return None

    # Compute flux error:
    npix = sources['npix']
    flux = sources['flux']
    # Poisson term + background term
    flux_err = np.sqrt( flux / gain + npix * std**2 )
    sources['flux_err'] = flux_err
    sources['snr'] = flux / flux_err
    return sources
