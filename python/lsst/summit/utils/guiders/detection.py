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

__all__ = ["starGuideFinder"]

import sep
import pandas as pd
import numpy as np

import astropy.units as u
import matplotlib.pyplot as plt
from astropy.nddata import CCDData, Cutout2D
from astropy.stats import mad_std
from astropy.stats import sigma_clipped_stats, SigmaClip

from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_threshold, detect_sources
from photutils.utils import circular_footprint

from lsst.obs.lsst.cameraTransforms import LsstCameraTransforms
from lsst.summit.utils.guiders.transformation import pixel_to_focal
from lsst.summit.utils.guiders.reading import readGuiderData
from matplotlib.patches import Rectangle

class starGuideFinder:
    """
    Class to find stars in the Guider data.

    Example
    -------
    >>> from lsst.obs.lsst import LsstCam
    >>> from read import readGuiderData
    >>> from starGuideFinder import starGuideFinder
    
    >>> seqNum, dayObs = 591, 20250425
    >>> camera = LsstCam.getCamera()
    >>> reader = readGuiderData(seqNum, dayObs)
    
    >>> finder = starGuideFinder(reader, camera)
    >>> finder.run()
    >>> print("Check the reference catalog, sorted by SNR")
    >>> finder.ref_catalog
    
    >>> finder.plot_stacked_sources()
    >>> finder.plot_drifts_with_errors()
    >>> finder.plot_scatter_stamp()
    
    """
    def __init__(self, readerObject, detname, camera=None, fwhm=6.0, 
                snr_threshold=3.0, nstamps_min=30, edge_size=30, max_ellip=0.1):
        """
        Initialize the starGuideFinder class.
        Parameters
        ----------
        readerObject : readGuiderData
            Instance of the readGuiderData class.
        detname : str
            Name of the detector. E.g. 'R22_S11'.
        camera : Camera
            Instance of the Camera class.
        fwhm : float
            Full width at half maximum for star detection.
        snr_threshold : float
            Signal-to-noise ratio threshold for star detection.
        nstamps_min : int
            Minimum number of stamps a star must be detected in to be considered valid.
        edge_size : int
            Size of the edge to mask in pixels.            
        max_ellip : float
            Maximum ellipticity for star detection.
        """
        self.reader = readerObject
        self.expId = self.reader.expId

        # Source detection parameters
        self.fwhm = fwhm

        # Quality control parameters
        self.snr_threshold = snr_threshold
        self.nstamps_min = nstamps_min
        self.max_ellip = max_ellip

        # Set the ROfI parameters
        self.roiRow = self.reader.roiRow
        self.roiCol = self.reader.roiCol
        self.roiRows = self.reader.roiRows
        self.roiCols = self.reader.roiCols
        
        # Set the number of stamps
        self.nStamps = self.reader.nStamps
        self.timestamp = self.reader.timestamp

        # Set detector attributes
        self.detname = detname
        self.reader.set_detector(detname)
        self.ampName = self.reader.ampName
        self.edge_size = edge_size

        # Set the camera object
        self.camera = camera or self.reader.camera
        self.detector = self.camera[self.detname]
        self.lct = LsstCameraTransforms(self.camera, self.detname)

        # Initialize attributes
        self.stars = pd.DataFrame()

        # Get the lowest corner of the amplifier
        self.get_amplifier_lowest_corner()
        pass

    @classmethod
    def run_all_guiders(cls, reader, camera=None, fwhm=12.0, snr_threshold=3.0, nstamps_min=30, max_ellip=0.1):
        """
        Run star detection on all guider detectors and return a master star table.

        Parameters
        ----------
        reader : readGuiderData
            Reader object already loaded.
        camera : Camera, optional
            Camera object. If None, will use reader.camera.
        fwhm, snr_threshold, nstamps_min, mag_offset_bias_max, mag_offset_std_max :
            Parameters forwarded to each starGuideFinder instance.

        Returns
        -------
        pandas.DataFrame
            Master catalog of selected stars from all guider detectors.
        """
        stars_list = []
        for detname in reader.getGuiderNames():
            # print(f"Processing {detname}")
            finder = cls(reader, detname, camera=camera,
                        fwhm=fwhm, snr_threshold=snr_threshold,
                        nstamps_min=nstamps_min
                        , max_ellip=max_ellip)
            finder.run()
            # print(f"Found {len(finder.ref_catalog)} stars in {detname} with SNR > {finder.snr_threshold}")
            stars_list.append(finder.stars)

        # Filter out empty DataFrames
        stars_list = [df for df in stars_list if not df.empty]

        if not stars_list:
            return pd.DataFrame()
        
        stars = pd.concat(stars_list, ignore_index=True)
        return stars

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
        self.stacked_image = self.stack_guider_images()

        # Build the reference catalog
        self.build_ref_catalog()

        # If no reference catalog was built, exit early
        if len(self.ref_catalog)==0:
            return

        # Track motion for all stars
        self.track_stars()

        # Select the best star per stamp
        self.select_best_star_per_stack()

        # convert to focal plane coordinates/ altaz
        self.convert_to_focal_plane()
        self.convert_to_altaz()

        # Set unique IDs
        self.stars = self.set_unique_id(self.stars)

        # Compute offsets
        self.stars = self.compute_offsets(self.stars)
        pass

    def set_unique_id(self, stars):
        # 1) Build a detector→index map (0,1,2,…)
        det_map = self.reader.guiders

        # 2) Create a numeric “global” star_id:
        #    global_id = det_index * 10000 + local star_id
        stars['det_id']     = stars['detector'].map(det_map)
        stars['star_local'] = stars['star_id'].astype(int)
        stars['star_id']    = stars['det_id']*10000 + stars['star_local']
        stars['expId']      = self.expId

        # 3) Drop the helpers if you like
        stars = stars.drop(columns=['star_local'])
        return stars

    def compute_offsets(self, stars):
        """
        Compute the offsets for each star in the catalog.
        """
        df = stars.copy()

        # Compute all your offsets
        df['dx']    = df['xpixel'] - df['xpixel_ref']
        df['dy']    = df['ypixel'] - df['ypixel_ref']
        df['dxfp']  = df['xfp']    - df['xfp_ref']
        df['dyfp']  = df['yfp']    - df['yfp_ref']
        df['dalt']  = (df['alt']    - df['alt_ref'])*3600
        df['daz']   = (df['az']     - df['az_ref'])*3600

        # Drop columns with ref
        # df.drop(columns=['xpixel_ref', 'ypixel_ref', 'xfp_ref', 'yfp_ref', 'alt_ref', 'az_ref'], inplace=True)
        return df

    def select_best_star_per_stack(self):
        """
        Select the best star per stamp based on:
        1) SNR
        2) number of stamps
        3) photometric stability

        Returns
        -------
        best_stars : list
            List of best stars for each stamp.
        """
        df = self.output_catalog.copy()
        df.drop(columns=['detector', 'ampname'], inplace=True, errors='ignore')
        # count the number of valid flux stamps for each star
        n_stamps = df.groupby('star_id').count()['flux'].values
        # compute the median SNR for each star
        med_snr = df.groupby('star_id').median()['snr'].values

        cut  = med_snr > self.snr_threshold
        cut &= n_stamps > self.nstamps_min
        starids = df.groupby('star_id').count().loc[cut].index
        
        # filter the stars
        mask = df['star_id'].isin(starids)
        df = df[mask].copy()
        df.sort_values(['stamp', 'snr'], ascending=[True, False], inplace=True)
        df.reset_index(drop=True, inplace=True)

        df['detector'] = self.detname
        df['ampname'] = self.ampName
        self.stars = df
        pass

    def track_star_stamp(self,
                        star_id: int,
                        ) -> pd.DataFrame:
        """
        Track one star across all stamps, returning a DataFrame with:
        ['star_id','stamp','xpixel','ypixel',
        'flux','flux_err','snr','roundness1','roundness2']
        The first row is stamp = -1 (the reference).
        """
        rows = []
        # Pull ref-catalog row
        ref = self.ref_catalog.iloc[star_id].copy()
        ref_x, ref_y = ref['xcentroid'], ref['ycentroid']
        ref_flux     = ref['flux']
        fwhm = self.fwhm
        ampname = ref['ampname']
        bkg = ref['noise']/self.nStamps
        bias = ref['bias']/self.nStamps

        ref['stamp'] = -1
        ref['star_id'] = star_id
        ref['mag_offset'] = 0.0  # reference is always 0 mag offset

        # # --- reference row ---
        sel_columns = [
            'star_id', 'stamp', 'ampname',
            'xcentroid', 'ycentroid', 'xpixel', 'ypixel', 'xerr', 'yerr',
            'flux', 'flux_err', 'snr', 'mag_offset',
            'ixx', 'iyy', 'ixy', 'ixx_err', 'iyy_err', 'ixy_err', 'fwhm',
        ]
        rows.append(ref[sel_columns].copy())

        mask_cutout = Cutout2D(self.mask_streak, (ref_x, ref_y), size=50, mode='partial', fill_value=False)

        # --- per‐stamp measurements ---
        # the first two stamps are taken the shutter is fully open
        for si in range(2, self.nStamps):
            stamp = self.image_list[si]
            isr   = stamp - self.sky_bkg.background/self.nStamps

            cutout = Cutout2D(isr, (ref_x, ref_y),
                            size=50, mode='partial', fill_value=np.nan)
            
            _, median, std = sigma_clipped_stats(cutout.data, sigma=3.0, mask=mask_cutout.data)
            # data = np.where(mask_cutout.data, cutout.data, 0)
            # sources = run_sextractor(cutout.data-median, aperture_radius=self.fwhm, th=5, max_ellip=0.2, mask=mask_cutout.data)
            sources = measure_star_in_aperture(cutout.data-median, aperture_radius=fwhm, std_bkg=std, gain=1.0, mask=mask_cutout.data)

            if len(sources) == 0:
                # No sources detected in this stamp, skip it
                continue
            sources['star_id'] = star_id
            sources['stamp'] = si
            sources['ampname'] = ampname

            # Centroid in amplifier roi coordinates
            sources['xcentroid'] += cutout.xmin_original
            sources['ycentroid'] += cutout.ymin_original

            # pixel in ccd coordinates
            sources['xpixel'] = sources['xcentroid'] + self.min_x
            sources['ypixel'] = sources['ycentroid'] + self.min_y
            rows.append(sources.iloc[0])

        df = pd.DataFrame(rows)
        
        # Define the reference as the median of the other stamps
        flux_med = np.nanmedian(df['flux'][1:]) + 1e-12  # avoid division by zero
        df['mag_offset'] = -2.5 * np.log10(df['flux']/flux_med)
        df['xcentroid_ref'] = np.nanmedian(df['xcentroid'][1:])
        df['ycentroid_ref'] = np.nanmedian(df['ycentroid'][1:])
        df['xpixel_ref'] = np.nanmedian(df['xpixel'][1:])
        df['ypixel_ref'] = np.nanmedian(df['ypixel'][1:])

        # if len(df)>1:
        #     df['mag_offset'] = -2.5 * np.log10(df['flux']/df['flux'][1:].median())
        #     df.iloc[0, df.columns.get_loc('xcentroid')] = np.median(df.iloc[1:, df.columns.get_loc('xcentroid')])
        #     df.iloc[0, df.columns.get_loc('ycentroid')] = np.median(df.iloc[1:, df.columns.get_loc('ycentroid')])
        #     df.iloc[0, df.columns.get_loc('xpixel')] = np.median(df.iloc[1:, df.columns.get_loc('xpixel')])
        #     df.iloc[0, df.columns.get_loc('ypixel')] = np.median(df.iloc[1:, df.columns.get_loc('ypixel')])
        return df

    def track_stars(self,
                    threshold_sigma: float = 3.0,
                    roundness_max: float = 0.5
                ) -> pd.DataFrame:
        """
        Track all reference stars; return one big DataFrame with every
        (star_id, stamp) row, including the reference (stamp=-1).
        """
        dfs = []
        for i in range(len(self.ref_catalog)):
            df_i = self.track_star_stamp(i)
            dfs.append(df_i)
        
        if not dfs:
            # return an empty DataFrame with the expected columns
            cols = ['star_id', 'stamp', 'detector', 'det_id', 'ampname',
                    'xcentroid', 'ycentroid', 'xpixel', 'ypixel', 
                    'xerr', 'yerr', 'xfp', 'yfp', 'alt', 'az',
                    'ixx', 'iyy', 'ixy', 'ixx_err', 'iyy_err', 
                    'ixy_err', 'fwhm',
                    'flux', 'flux_err', 'snr', 'mag_offset',
                    'dx', 'dy', 'dxfp', 'dyfp', 'dalt', 'daz']
            self.output_catalog = pd.DataFrame(columns=cols)
            return 
        
        output = pd.concat(dfs, ignore_index=True)
        # filter SNR
        output = output[output['snr'] > self.snr_threshold]
        output = output[output['flux'] > 1e-6]
        output.sort_values(['star_id', 'stamp'], inplace=True)
        output.reset_index(inplace=True, drop=True)

        self.output_catalog = output
        pass
    
    def stack_guider_images(self):
        """
        Stack guider images 

        Returns
        -------
        stacked_image : 2D numpy array
            Stacked image of the detected stars.
        """
        image_list = [self.reader.read(stamp, self.detname) 
                      for stamp in range(self.nStamps)]
        # parallel overscan region correction
        image_list = [img-np.nanmedian(img,axis=0) for img in image_list]
    
        # stack with the sum
        stacked = np.nansum(np.array(image_list), axis=0)

        self.image_list = image_list
        self.stacked = stacked
        return stacked

    def build_ref_catalog(self, threshold_sigma=3.0, edge_size=None):
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
        
        if edge_size is None:
            edge_size = self.edge_size

        # Find Bad columns 
        streak_mask = find_bad_columns(stacked_image, nsigma=2)

        # Build the background model
        median, mean, std, mask = background_model(stacked_image, fwhm=self.fwhm, streak_mask=streak_mask)

        self.bias = median
        self.noise_ref = std
        self.mask_source = mask
        self.mask_streak = streak_mask

        # model the background
        try:
            self.sky_bkg = Background2D(stacked_image, box_size=50, filter_size=3, bkg_estimator=MedianBackground(), mask=mask)
        except Exception as e:
            print(f"Error building background model: {e}")
            self.sky_bkg = Background2D(stacked_image, box_size=50, filter_size=3, bkg_estimator=MedianBackground(), exclude_percentile=30)

        # Detect stars in the stacked image
        # and filter out streaks
        # Use the background model to subtract the background
        # isr = stacked_image - self.sky_bkg.background
        # ref_catalog = detect_stars_filtered(isr, fwhm=self.fwhm, threshold_sigma=threshold_sigma, 
        #                                     roundness_max=0.2, median=0, std=std)

        ref_catalog = run_sextractor(stacked_image, aperture_radius=self.fwhm, th=threshold_sigma, max_ellip=self.max_ellip, mask=streak_mask)

        sel_columns = ['star_id', 'stamp', 'detector', 'det_id', 'ampname', 
                        'xcentroid', 'ycentroid', 'xpixel', 'ypixel', 'xerr', 'yerr', 'xfp', 'yfp', 'alt', 'az',
                        'ixx', 'iyy', 'ixy', 'ixx_err', 'iyy_err', 'ixy_err', 'fwhm',
                        'flux', 'flux_err', 'snr', 'mag_offset',
                        'dx', 'dy', 'dxfp', 'dyfp', 'dalt', 'daz',
                        ]
        
        if ref_catalog is None or len(ref_catalog) == 0:
            print(f"Error: No stars found in the reference catalog for {self.detname}.")
            self.ref_catalog = pd.DataFrame(columns=sel_columns)
            self.stars = pd.DataFrame(columns=sel_columns)
            return
        # Save the reference catalog
        self.ref_catalog = ref_catalog

        # Save the centroid in ccd coordinates
        self.ref_catalog['xpixel'] = self.ref_catalog['xcentroid']+self.min_x
        self.ref_catalog['ypixel'] = self.ref_catalog['ycentroid']+self.min_y

        # Filter out sources that are too close to the edges
        # and have low SNR
        self.filter_ref_catalog(snr_threshold=self.snr_threshold*np.sqrt(self.nStamps))
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
        
    def mask_edge_ref_catalog(self, edge=20):
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
        self.ref_catalog['ampname'] = self.ampName
        self.ref_catalog['stamp'] = -1
        self.ref_catalog['id'] = np.arange(len(self.ref_catalog))

        # Convert the star positions to CCD coordinates
        self.ref_catalog['xpixel'] = self.ref_catalog['xcentroid']+self.min_x
        self.ref_catalog['ypixel'] = self.ref_catalog['ycentroid']+self.min_y
        pass

    def convert_to_focal_plane(self):
        """
        Convert the star positions to focal plane coordinates.
        """
        if len(self.stars['xpixel'])>0:
            # Convert the star positions to focal plane coordinates
            xfp, yfp = pixel_to_focal(self.stars['xpixel'], self.stars['ypixel'], self.detector)
            xfp_ref, yfp_ref = pixel_to_focal(self.stars['xpixel_ref'], self.stars['ypixel_ref'], self.detector)
        else:
            xfp, yfp = None, None
            xfp_ref, yfp_ref = None, None

        self.stars['xfp'] = xfp
        self.stars['yfp'] = yfp
        self.stars['xfp_ref'] = xfp_ref
        self.stars['yfp_ref'] = yfp_ref
        pass
    
    def convert_to_altaz(self):
        """
        Convert the star positions to altaz coordinates.
        """
        if len(self.stars['xfp'])>0:
            from transformation import CoordinatesToAltAz
            coord = CoordinatesToAltAz(self.reader.seqNum, self.reader.dayObs, self.detname,
                                    butler=self.reader.butler)
            az, alt = coord.convert_pixels_to_altaz(self.stars['xpixel'], self.stars['ypixel'])
            az_ref, alt_ref = coord.convert_pixels_to_altaz(self.stars['xpixel_ref'], self.stars['ypixel_ref'])
        else:
            az, alt = None, None
            az_ref, alt_ref = None, None
        self.stars['alt'] = alt
        self.stars['az'] = az
        self.stars['alt_ref'] = alt_ref
        self.stars['az_ref'] = az_ref
        pass

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
        im2 = ax.imshow(self.mask_streak,
                    origin='lower',
                    cmap='Reds',
                    alpha=0.5,
                    vmin=0, vmax=1,
                    interpolation='nearest')

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
        # ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_title("Stacked guider image with reference sources"+"\n"+f"({self.detname}, {self.expId})")

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

    def plot_drifts_with_errors(self, stars=None,
                                figsize=(6,4),
                                fig=None,
                                ax=None,
                                **plot_kw):
        """
        Plot the median drift ± robust σ (from MAD) for ΔX and ΔY per stamp.
        """
        if stars is None:
            if not hasattr(self, 'stars'):
                raise ValueError("No output catalog found. Run run() first.")
            stars = self.stars

        # Remove the stacked sources
        stars = stars[stars['stamp'] != -1].copy()

        # Group by stamp
        grouped = stars.groupby('stamp')
        stamps = np.array(sorted(stars['stamp'].unique()))

        # Per-stamp median
        med_dx = grouped['dxfp'].median().to_numpy()*100
        med_dy = grouped['dyfp'].median().to_numpy()*100

        # Per-stamp robust sigma (MAD)
        sig_dx = grouped['dxfp'].apply(lambda x: mad_std(x, ignore_nan=True)).to_numpy()*100
        sig_dy = grouped['dyfp'].apply(lambda x: mad_std(x, ignore_nan=True)).to_numpy()*100
        err_x = grouped['xerr'].median().to_numpy()
        err_y = grouped['yerr'].median().to_numpy()
        sig_dx = np.hypot(sig_dx, err_x)
        sig_dy = np.hypot(sig_dy, err_y)

        nstars = stars['star_id'].nunique()
        nstamps = stars['stamp'].nunique()

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        defaults = dict(fmt='o', capsize=3, markersize=5, alpha=0.8)
        defaults.update(plot_kw)

        # # Optional: show all data as faded dots for context
        # ax.scatter(stars['stamp'], stars['dx'], color='k', marker='x', alpha=0.25, label='CCD ΔX (all)')
        # ax.scatter(stars['stamp'], stars['dy'], color='firebrick', marker='+', alpha=0.25, label='CCD ΔY (all)')

        # Median drift/error bars
        ax.errorbar(stamps, med_dx, yerr=sig_dx, color='k', label='Median ΔX', **defaults)
        ax.errorbar(stamps, med_dy, yerr=sig_dy, color='firebrick', label='Median ΔY', **defaults)

        ax.axhline(0, color='grey', lw=1, ls='--')

        # --- new jitter annotation ---
        jitter_pix = np.nanstd(stars[['dx', 'dy']].to_numpy())
        jitter_arcsec = jitter_pix * 0.2
        txt = f"Jitter (rms): {jitter_pix:.2f} pixel, {jitter_arcsec:.2f} arcsec"
        ax.text(0.02, 0.98, txt,
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=12, color='grey',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # Polish
        ax.set_xlabel("Stamp #")
        ax.set_ylabel("Offset (pixels)")
        ax.set_title(f"Star drift over {nstamps} stamps ({nstars} stars)"
                    + f"\n({getattr(self, 'detname', '?')}, {getattr(self, 'expId', '?')})")
        ax.legend(frameon=False, loc='upper right', ncol=2)
        ax.grid(True, ls=':', color='grey', alpha=0.5)
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

    @staticmethod
    def measure_jitter_stats(stars: pd.DataFrame) -> pd.DataFrame:
        """
        Compute global jitter statistics across all guiders.

        Parameters
        ----------
        stars : pd.DataFrame
            Concatenated star table across all guider detectors.

        Returns
        -------
        stars : pd.DataFrame
            DataFrame with new jitter statistic columns broadcasted to all rows.
        """
        time = (stars.stamp.to_numpy() + 0.5) * 0.3  # seconds
        time = time.astype(np.float64)
        az = stars.daz.to_numpy()
        alt = stars.dalt.to_numpy()

        # Linear fits
        coefs_az = np.polyfit(time, az, 1)
        coefs_alt = np.polyfit(time, alt, 1)

        # Stats
        jitter_stats = {
            "jitter_az": mad_std(az),
            "jitter_alt": mad_std(alt),
            "jitter_corr_az": mad_std(az - np.polyval(coefs_az, time)),
            "jitter_corr_alt": mad_std(alt - np.polyval(coefs_alt, time)),
            "drift_rate_az": coefs_az[0],
            "drift_rate_alt": coefs_alt[0],
            "offset_zero_az": coefs_az[1],
            "offset_zero_alt": coefs_alt[1],
        }
        
        return jitter_stats

    @staticmethod
    def format_jitter_summary(jitter_stats: dict) -> str:
        """
        Pretty string summary of jitter stats from run_all_guiders.
        """
        if not jitter_stats:
            return "No jitter statistics available."

        js = jitter_stats
        summary = (
            f"\nGlobal Jitter Summary Across All Guiders\n"
            f"{'-'*45}\n"
            f"  - Jitter (AZ):         {js['jitter_az']:.3f} arcsec (raw)\n"
            f"  - Jitter (ALT):        {js['jitter_alt']:.3f} arcsec (raw)\n"
            f"  - Jitter (AZ):         {js['jitter_corr_az']:.3f} arcsec (linear corr)\n"
            f"  - Jitter (ALT):        {js['jitter_corr_alt']:.3f} arcsec (linear corr)\n"
            f"  - Drift Rate (AZ):     {15*js['drift_rate_az']:.3f} arcsec per exposure\n"
            f"  - Drift Rate (ALT):    {15*js['drift_rate_alt']:.3f} arcsec per exposure\n"
            f"  - Zero Offset (AZ):    {js['offset_zero_az']:.3f} arcsec\n"
            f"  - Zero Offset (ALT):   {js['offset_zero_alt']:.3f} arcsec"
        )
        return summary



    @staticmethod
    def measure_photometric_variation(stars: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Fit mag_offset vs time across all rows, compute drift rate, zero-point, and RMS scatter,
        then add these as constant columns to `stars`.
        """
        mo = stars["mag_offset"].to_numpy()
        mask = np.isfinite(mo)
        if not mask.any():
            phot_stats = {
                "mag_offset_rate": np.nan,
                "mag_offset_zero": np.nan,
                "mag_offset_rms": np.nan,
            }
        else:
            time = (stars["stamp"].to_numpy()[mask] + 0.5) * 0.3  # seconds
            mo_valid = mo[mask]
            coef = np.polyfit(time, mo_valid, 1)
            rate, zero = coef
            resid = mo_valid - np.polyval(coef, time)
            rms = mad_std(resid)
            phot_stats = {
                "mag_offset_rate": rate,
                "mag_offset_zero": zero,
                "mag_offset_rms": rms,
            }
            
        return phot_stats

    @staticmethod
    def format_photometric_summary(phot_stats: dict) -> str:
        """
        Pretty-print summary of photometric variation statistics.
        """
        if not phot_stats:
            return "No photometric statistics available."

        return (
            "\nPhotometric Variation Summary\n"
            "-------------------------------\n"
            f"  - Mag Drift Rate:      {phot_stats['mag_offset_rate']:.5f} mag/sec\n"
            f"  - Mag Zero Offset:     {phot_stats['mag_offset_zero']:.5f} mag\n"
            f"  - Mag RMS (detrended): {phot_stats['mag_offset_rms']:.5f} mag"
        )

    @staticmethod
    def format_stats_per_guider(stars_per_guiders: dict) -> str:
        """
        Formats the per-guider star counts (e.g. {'N_R22_S11': 5, ...}) as a table.
        """
        if not stars_per_guiders:
            return "No guider star counts available."

        lines = ["",
            f"{'Guider':<20} {'N_stars':>8}",
            "-" * 30,
        ]
        for guider, n_stars in sorted(stars_per_guiders.items()):
            lines.append(f"{guider:<20} {n_stars:8d}")
        return "\n".join(lines)
    
    @staticmethod
    def format_stats_per_guider_full(stars_per_guiders: dict, n_guider: int, n_unique_stars: int, n_measurements: int, fraction_valid: float) -> str:
        """
        Formats per-guider star counts with extra global stats.
        """
        out = "-" * 50
        out += f"\nNumber of Stars: {n_unique_stars}\n"
        out +=  f"Number of Guiders: {n_guider}\n"
        out += f"Total valid measurements: {n_measurements:,d}"
        out += f"\nFraction valid stamps:    {fraction_valid:.3f}\n"
        out += starGuideFinder.format_stats_per_guider(stars_per_guiders)
        return out
    
    
    @classmethod
    def run_guide_stats(cls, reader, fwhm=10, snr_threshold=10, 
                        camera=None, nstamps_min=30, vebose=True, max_ellip=0.2):
        """
        Run all guiders, then produce a one‐row DataFrame containing:
        - Number of valid guiders
        - Number of unique stars
        - Number of stars per guider (e.g., N_R02_S11, N_R22_S11, etc.)
        - Number of star measurements across all guiders and stamps
        - Global jitter statistics (az/alt, detrended, rates, zero‐points)
        - Global photometric variation statistics (drift rate, zero‐point, RMS)
        - Fraction of valid stamp measurements
        """
        # 1) Run detections across all guiders
        stars = cls.run_all_guiders(reader, fwhm=fwhm, snr_threshold=snr_threshold,
                                    camera=camera, nstamps_min=nstamps_min, max_ellip=max_ellip)
        if stars.empty:
            cols = [
                'n_guiders',
                'n_unique_stars',
                'fraction_valid_stamps',
                'n_measurements',
            ]
            # Dynamically include per‐guider star counts, jitter, and photometry keys
            # Use placeholders for column names
            example_jitter = ['jitter_az','jitter_alt','jitter_corr_az','jitter_corr_alt',
                            'offset_rate_az','offset_rate_alt','offset_zero_az','offset_zero_alt']
            example_phot = ['mag_offset_rate','mag_offset_zero','mag_offset_rms']
            cols += [f'N_{det}' for det in []] + example_jitter + example_phot
            return pd.DataFrame(), pd.DataFrame(columns=cols)


        # 2) Number of valid guiders
        n_guiders = stars['detector'].nunique()

        # 3) Number of unique stars
        n_unique_stars = stars['star_id'].nunique()

        # 4) Number of stars per guider
        stars_per_guiders = {}
        counts = (stars.groupby('detector')['star_id'].nunique().to_dict())
        for det in reader.guiders.keys():
            stars_per_guiders[f'N_{det}'] = counts.get(det, 0)


        # 5) Number of star measurements (stamp >= 0 and xpixel not NaN)
        mask_valid = (stars['stamp'] >= 0) & (stars['xpixel'].notna())
        n_measurements = int(mask_valid.sum())

        # 6) Global jitter stats (adds jitter cols to stars)
        jitter_stats = cls.measure_jitter_stats(stars)

        # 7) Global photometric variation stats (adds photometry cols to stars)
        phot_stats = cls.measure_photometric_variation(stars)

        # 8) Fraction of valid stamp measurements:
        #    total possible = n_unique_stars * reader.nStamps
        total_possible = n_unique_stars * reader.nStamps
        fraction_valid = n_measurements / total_possible if total_possible > 0 else np.nan

        # 9) Assemble summary dict
        summary = {
            'n_guiders': n_guiders,
            'n_stars': n_unique_stars,
            'n_measurements': n_measurements,
            'fraction_valid_stamps': fraction_valid,
            **stars_per_guiders,
            **jitter_stats,
            **phot_stats
        }
        # 10) Return as single‐row DataFrame
        stats = pd.DataFrame([summary])

        if vebose:
            print(cls.format_stats_per_guider_full(stars_per_guiders, n_guiders, n_unique_stars, n_measurements, fraction_valid))        
            print(cls.format_jitter_summary(jitter_stats))
            print(cls.format_photometric_summary(phot_stats))
            print("-" * 50)
            
        return stars, stats

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

def background_model(image, fwhm=10.0, streak_mask=None):
    sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    threshold = detect_threshold(image, nsigma=5.0, sigma_clip=sigma_clip, mask=streak_mask)
    segment_img = detect_sources(image, threshold, npixels=fwhm/2., mask=streak_mask)
    footprint = circular_footprint(radius=2*fwhm)
    
    if footprint is None:
        mask = np.zeros_like(image, dtype=bool)
    else:
        mask = segment_img.make_source_mask(footprint=footprint)
    mean, median, std = sigma_clipped_stats(image, sigma=3.0, mask=mask | streak_mask)
    return mean, median, std, mask
        
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
    flux_err = np.sqrt( np.abs(flux) / gain + npix * std**2 )
    sources['flux_err'] = flux_err
    sources['snr'] = flux / flux_err
    return sources

def measure_star_in_aperture(
    cutout_data,
    aperture_radius=5,
    std_bkg=1.0,
    gain=1.0,
    mask=None,
):
    """
    Measure centroid, moments, and flux in a circular aperture, ignoring any
    pixels flagged in `mask`.

    Parameters
    ----------
    cutout_data : 2D ndarray
        Background-subtracted cutout image.
    aperture_radius : float
        Radius of aperture in pixels.
    std_bkg : float
        Background RMS per pixel.
    gain : float
        e-/ADU gain.
    mask : 2D bool ndarray or None
        True = pixel to ignore (e.g. bad column, cosmic ray, star mask). Must
        be same shape as cutout_data.  If None, no extra masking.

    Returns
    -------
    pandas.DataFrame with one row and columns:
      xcentroid, ycentroid, xerr, yerr,
      ixx, iyy, ixy, ixx_err, iyy_err, ixy_err,
      flux, flux_err, fwhm, snr
    """
    h, w = cutout_data.shape
    y, x = np.indices((h, w))
    x0, y0 = w/2, h/2

    # Combine aperture and external mask
    ap_mask = ((x - x0)**2 + (y - y0)**2) <= aperture_radius**2
    if mask is not None:
        valid1 = ap_mask & (~mask)
    else:
        valid1 = ap_mask

    # 1) initial flux & centroid
    data1 = np.where(valid1, cutout_data, 0.0)
    flux1 = np.nansum(data1)
    if flux1 <= 1e-12:
        # no signal → return all NaNs/zeros
        return pd.DataFrame([{
            "xcentroid": np.nan, "ycentroid": np.nan,
            "xerr": np.nan, "yerr": np.nan,
            "ixx": np.nan, "iyy": np.nan, "ixy": np.nan,
            "ixx_err": np.nan, "iyy_err": np.nan, "ixy_err": np.nan,
            "flux": 0.0, "flux_err": 0.0,
            "fwhm": np.nan, "snr": 0.0
        }])

    xcen1 = np.nansum(x * data1) / flux1
    ycen1 = np.nansum(y * data1) / flux1

    # 2) re-centered aperture
    valid2 = ((x - xcen1)**2 + (y - ycen1)**2) <= aperture_radius**2
    if mask is not None:
        valid2 = valid2 & (~mask)
    data2 = np.where(valid2, cutout_data, 0.0)
    flux2 = np.nansum(data2)
    npix2 = np.count_nonzero(valid2)

    # 3) second moments
    dx = x - xcen1
    dy = y - ycen1
    if flux2 > 0:
        ixx = np.nansum(dx**2 * data2) / flux2
        iyy = np.nansum(dy**2 * data2) / flux2
        ixy = np.nansum(dx * dy * data2) / flux2

        # centroid errors (from moments & flux)
        xerr = np.sqrt(np.abs(ixx)) / np.sqrt(flux2)
        yerr = np.sqrt(np.abs(iyy)) / np.sqrt(flux2)

        # per-pixel variance: poisson + background
        var = np.abs(data2)/gain + std_bkg**2
        ixx_err = np.sqrt(np.nansum(dx**4 * var) / flux2**2)
        iyy_err = np.sqrt(np.nansum(dy**4 * var) / flux2**2)
        ixy_err = np.sqrt(np.nansum((dx**2)*(dy**2) * var) / flux2**2)
    else:
        ixx = iyy = ixy = xerr = yerr = ixx_err = iyy_err = ixy_err = np.nan

    # 4) flux error & FWHM
    flux_err = np.sqrt(flux2/gain + npix2 * std_bkg**2)
    sigma = np.sqrt(np.abs(ixx + iyy)/2.0) if np.isfinite(ixx+iyy) else np.nan
    fwhm = 2.355 * sigma
    snr = flux2 / np.maximum(flux_err, 1e-12)

    return pd.DataFrame([{
        "xcentroid": xcen1,
        "ycentroid": ycen1,
        "xerr": xerr,
        "yerr": yerr,
        "ixx": ixx,
        "iyy": iyy,
        "ixy": ixy,
        "ixx_err": ixx_err,
        "iyy_err": iyy_err,
        "ixy_err": ixy_err,
        "flux": flux2,
        "flux_err": flux_err,
        "fwhm": fwhm,
        "snr": snr
    }])

def find_bad_columns(img, mask=None, nsigma=3.0):
    """
    Identify bad columns in an image using per-column sigma-clipped statistics.

    Parameters
    ----------
    img : 2D ndarray
        The input image (can contain NaNs).
    mask : 2D bool ndarray or None
        True where pixels should be ignored in the stats (e.g. masked or bad pixels).
    nsigma : float
        number of stdevs from the median to be considered a bada column. 

    Returns
    -------
    bad_mask : 2D bool ndarray
        True for all pixels in columns flagged as bad.
    """
    # 1) Compute per-column sigma-clipped stats
    mean_cols, median_cols, std_cols = sigma_clipped_stats(
        img,
        sigma=3.0,
        maxiters=5,
        mask=mask,
        axis=0  # collapse over rows → one stat per column
    )

    # 2) Determine threshold: median of medians + thresh_sigma * median of sigmas
    global_med_of_meds = np.nanmedian(median_cols)
    global_med_of_stds = np.nanmedian(std_cols)
    threshold = global_med_of_meds + nsigma * global_med_of_stds

    # 3) Find columns whose median exceeds that threshold
    bad_cols = np.where(median_cols > threshold)[0]

    # 4) Build a 2D mask marking entire columns as bad
    bad_mask = np.zeros_like(img, dtype=bool)
    if bad_cols.size:
        bad_mask[:, bad_cols] = True
        
    return bad_mask

def run_sextractor(img, th=10, median=0, std=None, bkg_size=50, aperture_radius=5, max_ellip=0.1, gain=1.0, mask=None):
    """
    Vectorized SEP photometry with centroid errors, outputs a pandas DataFrame.
    Only returns nearly round, bright sources.
    """
    # Mask bad pixels
    bad_mask = ~np.isfinite(img) | (img < 0)
    img_clean = np.where(bad_mask, 0.0, img)

    # Background subtraction
    if std is None:
        bkg = sep.Background(img_clean, mask=bad_mask, bw=bkg_size, bh=bkg_size)
        img_sub = img_clean - bkg
        std = bkg.globalrms
    else:
        img_sub = img_clean - median

    # Detection
    objects = sep.extract(img_sub, th, err=std, mask=bad_mask|mask)
    if len(objects) == 0:
        return pd.DataFrame()

    # Gather properties
    xcen, ycen = objects['x'], objects['y']
    ixx, iyy, ixy = objects['x2'], objects['y2'], objects['xy']
    ixx_err, iyy_err, ixy_err = objects['errx2'], objects['erry2'], objects['errxy']

    flux, fluxerr, _ = sep.sum_circle(
        img_clean, xcen, ycen, aperture_radius, err=std, mask=bad_mask|mask, gain=gain
    )
    fwhm = 2.355 * np.sqrt(0.5 * (ixx + iyy))

    denom = ixx + iyy + 1e-12
    e1 = (ixx - iyy) / denom
    e2 = (2 * ixy) / denom

    # Filter: round and bright sources only
    mask = np.abs(e1) < max_ellip  # nearly round
    mask &= np.abs(e2) < max_ellip  # nearly round

    # Compute centroid errors from moment errors (astrometry error estimate)
    # For a Gaussian, σ_xcentroid ≈ sqrt(ixx_err) / sqrt(flux)
    # (cf. https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/IRAC_Instrument_Handbook.pdf, Table 2.9)
    centroid_x_err = np.sqrt(np.abs(ixx_err)) / np.sqrt(np.maximum(flux, 1e-6))
    centroid_y_err = np.sqrt(np.abs(iyy_err)) / np.sqrt(np.maximum(flux, 1e-6))

    df = pd.DataFrame({
        "xcentroid": xcen[mask],
        "ycentroid": ycen[mask],
        "xerr": centroid_x_err[mask],
        "yerr": centroid_y_err[mask],
        "ixx": ixx[mask],
        "iyy": iyy[mask],
        "ixy": ixy[mask],
        "ixx_err": ixx_err[mask],
        "iyy_err": iyy_err[mask],
        "ixy_err": ixy_err[mask],
        "fwhm": fwhm[mask],
        "e1": e1[mask],
        "e2": e2[mask],
        "flux": flux[mask],
        "flux_err": fluxerr[mask],
        "snr": flux[mask] / np.maximum(fluxerr[mask], 1e-6)
    })

    return df

def make_stars_pairs(df0, seqNum=300):
    df = df0[df0['seqNum']==seqNum].copy()
    starids = df['star_id'].values
    seqnum = df['seqNum'].values
    regions = df['region'].values
    snr = df['snr'].values
    flux = df['flux'].values
    
    stars1, stars2 = [], []
    for r1, r2 in [(0,3), (1,2)]:
        m1 = (regions == r1)
        m2 = (regions == r2)
        ix = np.argsort(snr)

        s1 = starids[ix][m1]
        s2 = starids[ix][m2]
        stars1.append(s1)
        stars2.append(s2)


if __name__ == "__main__":
    # Example usage
    # from reading import readGuiderData
    from lsst.summit.utils.guiders.reading import readGuiderData
    from lsst.summit.utils.guiders.sourceSelection import starGuideFinder
    import pandas as pd

    seqNum, dayObs = 591, 20250425
    reader = readGuiderData(seqNum, dayObs, view='ccd')
    reader.load()

    # Run the source detection for all guider
    # return the stars DataFrame with all the measurements
    # return some stats information of the number of stars, jitter, photometric variance
    stars, stats = starGuideFinder.run_guide_stats(reader, fwhm=10, snr_threshold=10)

    ##### Some stats
    # The number of valid (not nan) stamp measurements per star
    stars.groupby('star_id')[['stamp','xpixel']].count()
    
    # The centroid error for each stamp
    stars.groupby('stamp')[['dalt','daz']].std()

    # The mean jitter in arcsec
    stars[['dalt','daz']].std()
