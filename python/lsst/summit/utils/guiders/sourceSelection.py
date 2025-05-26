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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.nddata import Cutout2D
from astropy.stats import SigmaClip, mad_std, sigma_clipped_stats
from matplotlib.patches import Rectangle
from photutils.background import Background2D, MedianBackground
from photutils.detection import DAOStarFinder
from photutils.segmentation import detect_sources, detect_threshold
from photutils.utils import circular_footprint

from lsst.obs.lsst.cameraTransforms import LsstCameraTransforms
from lsst.summit.utils.guiders.reading import ReadGuiderData
from lsst.summit.utils.guiders.transformation import pixel_to_focal

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

    def __init__(
        self,
        readerObject,
        detname,
        camera=None,
        fwhm=12.0,
        snr_threshold=3.0,
        nstamps_min=10,
        mag_offset_bias_max=0.1,
        mag_offset_std_max=0.1,
    ):
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
        """
        self.reader = readerObject
        self.expId = self.reader.expId

        # Source detection parameters
        self.fwhm = fwhm

        # Quality control parameters
        self.snr_threshold = snr_threshold
        self.nstamps_min = nstamps_min
        self.mag_offset_bias_max = mag_offset_bias_max
        self.mag_offset_std_max = mag_offset_std_max

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

        # Set the camera object
        self.camera = camera or self.reader.camera
        self.detector = self.camera[self.detname]
        self.lct = LsstCameraTransforms(self.camera, self.detname)

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
        a, b = self.lct.ampPixelToCcdPixel(self.roiCol, self.roiRow, self.ampName)
        c, d = self.lct.ampPixelToCcdPixel(
            self.roiCol + self.roiCols, self.roiRow + self.roiRows, self.ampName
        )
        begin_x, begin_y = min([a, c]), min([b, d])
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
        stars["det_id"] = stars["detector"].map(det_map)
        stars["star_local"] = stars["star_id"].astype(int)
        stars["star_id"] = stars["det_id"] * 10000 + stars["star_local"]
        stars["expId"] = self.expId

        # 3) Drop the helpers if you like
        stars = stars.drop(columns=["star_local"])
        return stars

    def compute_offsets(self, stars):
        """
        Compute the offsets for each star in the catalog.
        """
        # 1) Pull out the reference (stamp = -1) for all coords
        refs = stars.loc[
            stars["stamp"] == -1, ["star_id", "xpixel", "ypixel", "xfp", "yfp", "alt", "az"]
        ].rename(
            columns={
                "xpixel": "x_ref",
                "ypixel": "y_ref",
                "xfp": "xfp_ref",
                "yfp": "yfp_ref",
                "alt": "alt_ref",
                "az": "az_ref",
            }
        )

        # 2) Merge them back into the full table
        df = stars.merge(refs, on="star_id", how="left")

        # 3) Compute all your offsets
        df["dx"] = df["xpixel"] - df["x_ref"]
        df["dy"] = df["ypixel"] - df["y_ref"]
        df["dxfp"] = df["xfp"] - df["xfp_ref"]
        df["dyfp"] = df["yfp"] - df["yfp_ref"]
        df["dalt"] = (df["alt"] - df["alt_ref"]) * 3600
        df["daz"] = (df["az"] - df["az_ref"]) * 3600

        # 4) Finally drop the reference rows themselves
        df = df.loc[df["stamp"] != -1].reset_index(drop=True)
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
        df = self.output_catalog
        # count the number of valid flux stamps for each star
        n_stamps = df.groupby("star_id").count()["flux"].values
        # compute the median SNR for each star
        med_snr = df.groupby("star_id").median()["snr"].values
        # compute the median magOffset for each star
        med_magOffset = df.groupby("star_id").median()["mag_offset"].values
        # compute the std of the magOffset for each star
        std_magOffset = df.groupby("star_id").std()["mag_offset"].values

        cut = med_snr > self.snr_threshold
        cut &= n_stamps > self.nstamps_min
        cut &= np.abs(med_magOffset) < self.mag_offset_bias_max
        cut &= std_magOffset < self.mag_offset_std_max
        starids = df.groupby("star_id").count().loc[cut].index

        # filter the stars
        mask = df["star_id"].isin(starids)
        df = df[mask].copy()
        df.sort_values(["stamp", "snr"], ascending=[True, False], inplace=True)
        df.reset_index(drop=True, inplace=True)

        df["detector"] = self.detname
        self.stars = df
        pass

    def build_motion_dataframe(self) -> pd.DataFrame:
        """
        ref_catalog : pandas.DataFrame
            Must have columns:
            ['xcentroid','ycentroid','flux','flux_err','snr'].
        motions : ndarray, shape (n_stars, 2, n_stamps)
            motions[i,0,j] = Δx, motions[i,1,j] = Δy
        magOffsets : ndarray, shape (n_stars, n_stamps)
            per‐star magnitude offsets
        Returns
        -------
        df : pandas.DataFrame
            columns = [
            'star_id','stamp','xpixel','ypixel',
            'dx','dy','mag_offset','flux','flux_err','snr'
            ]
        """
        if not hasattr(self, "motions"):
            raise ValueError("No motions found. Please run track_stars() first.")

        n_stars, _, n_stamps = self.motions.shape

        # --- Build the reference‐catalog part (stamp = -1) ---
        cols = ["xcentroid", "ycentroid", "flux", "flux_err", "snr"]
        ref = self.ref_catalog[cols].copy().reset_index(drop=True)
        ref = ref.rename(columns={"xcentroid": "xpixel", "ycentroid": "ypixel"})
        ref["stamp"] = -1
        ref["mag_offset"] = 0.0
        ref["star_id"] = np.arange(1, n_stars + 1)  # 1-based ID

        # --- Build the per-stamp measurements ---
        # 1) star_id and stamp indices
        star_ids = np.repeat(np.arange(1, n_stars + 1), n_stamps)
        stamp_idxs = np.tile(np.arange(n_stamps), n_stars)

        # 2) flattened reference positions
        x0 = np.repeat(ref["xpixel"].values, n_stamps)
        y0 = np.repeat(ref["ypixel"].values, n_stamps)

        # 3) flattened motions & magOffsets
        dx_flat = self.motions[:, 0, :].ravel()
        dy_flat = self.motions[:, 1, :].ravel()
        mo_flat = self.magOffsets.ravel()
        snr_flat = self.snrStamps.ravel()

        # 4) compute absolute positions at each stamp
        x_pix = x0 + dx_flat
        y_pix = y0 + dy_flat

        # 5) repeat the flux, flux_err, snr from the ref for each stamp
        flux_flat = np.repeat(ref["flux"].values, n_stamps)
        flux_err_flat = np.repeat(ref["flux_err"].values, n_stamps)

        # 6) assemble the DataFrame
        meas = pd.DataFrame(
            {
                "star_id": star_ids,
                "stamp": stamp_idxs,
                "xpixel": x_pix,
                "ypixel": y_pix,
                "mag_offset": mo_flat,
                "flux": flux_flat,
                "flux_err": flux_err_flat,
                "snr": snr_flat,
            }
        )

        # --- Combine and return ---
        df = pd.concat([ref, meas], ignore_index=True, sort=False)
        df.sort_values("star_id", inplace=True)
        return df.reset_index(drop=True)

    def track_star_stamp(
        self, star_id: int, threshold_sigma: float = 5.0, roundness_max: float = 0.5
    ) -> pd.DataFrame:
        """
        Track one star across all stamps, returning a DataFrame with:
        ['star_id','stamp','xpixel','ypixel',
        'flux','flux_err','snr','roundness1','roundness2']
        The first row is stamp = -1 (the reference).
        """
        rows = []
        # Pull ref-catalog row
        ref = self.ref_catalog.iloc[star_id]
        ref_x, ref_y = ref["xcentroid"], ref["ycentroid"]
        ref_flux = ref["flux"]
        ref_ferr = ref["flux_err"]
        ref_snr = ref["snr"]
        ref_r1 = ref.get("roundness1", np.nan)
        ref_r2 = ref.get("roundness2", np.nan)

        # --- reference row ---
        rows.append(
            {
                "star_id": star_id,
                "stamp": -1,
                "xpixel": ref_x,
                "ypixel": ref_y,
                "flux": ref_flux,
                "flux_err": ref_ferr,
                "mag_offset": 0.0,
                "snr": ref_snr,
                "roundness1": ref_r1,
                "roundness2": ref_r2,
            }
        )

        # --- per‐stamp measurements ---
        for si in range(self.nStamps):
            stamp = self.image_list[si]
            isr = stamp - self.sky_bkg.background

            cutout = Cutout2D(isr, (ref_x, ref_y), size=50, mode="partial", fill_value=np.nan)
            _, median, std = sigma_clipped_stats(cutout.data, sigma=3.0)

            sources = detect_stars_filtered(
                cutout.data,
                fwhm=self.fwhm,
                threshold_sigma=threshold_sigma,
                roundness_max=roundness_max,
                median=median,
                std=std,
            )

            if sources is not None and len(sources) > 0:
                # pick the brightest detection
                src = sources[sources["flux"] == np.max(sources["flux"])][0]
                x_meas, y_meas = src["xcentroid"], src["ycentroid"]
                x_full = x_meas + cutout.xmin_original
                y_full = y_meas + cutout.ymin_original

                flux = src["flux"]
                flux_err = src["flux_err"]
                snr = src["snr"]  # *np.sqrt(self.nStamps)
                r1 = src["roundness1"]
                r2 = src["roundness2"]
                magOffset = -2.5 * np.log10(flux / ref_flux)
            else:
                x_full = y_full = flux = flux_err = snr = r1 = r2 = magOffset = np.nan

            rows.append(
                {
                    "star_id": star_id,
                    "stamp": si,
                    "xpixel": x_full,
                    "ypixel": y_full,
                    "flux": flux,
                    "flux_err": flux_err,
                    "mag_offset": magOffset,
                    "snr": snr,
                    "roundness1": r1,
                    "roundness2": r2,
                }
            )

        return pd.DataFrame(rows)

    def track_stars(self, threshold_sigma: float = 3.0, roundness_max: float = 0.5) -> pd.DataFrame:
        """
        Track all reference stars; return one big DataFrame with every
        (star_id, stamp) row, including the reference (stamp=-1).
        """
        dfs = []
        for i in range(len(self.ref_catalog)):
            df_i = self.track_star_stamp(i, threshold_sigma=threshold_sigma, roundness_max=roundness_max)
            dfs.append(df_i)

        if not dfs:
            # return an empty DataFrame with the expected columns
            cols = [
                "star_id",
                "stamp",
                "xpixel",
                "ypixel",
                "flux",
                "flux_err",
                "mag_offset",
                "snr",
                "roundness1",
                "roundness2",
            ]

            self.output_catalog = pd.DataFrame(columns=cols)
            return

        output = pd.concat(dfs, ignore_index=True)
        # filter SNR
        output = output[output["snr"] > self.snr_threshold]
        output.sort_values(["star_id", "stamp"], inplace=True)
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
        # Stack the images using ccdproc
        # Read the images from the reader
        image_list = [self.reader.read(stamp, self.detname) for stamp in range(self.nStamps)]
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

    def build_ref_catalog(self, threshold_sigma=3.0, edge_size=20):
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
        self.sky_bkg = Background2D(
            stacked_image, box_size=50, filter_size=3, bkg_estimator=MedianBackground(), mask=mask
        )

        # Detect stars in the stacked image
        # and filter out streaks
        # Use the background model to subtract the background
        isr = stacked_image - self.sky_bkg.background
        ref_catalog = detect_stars_filtered(
            isr, fwhm=self.fwhm, threshold_sigma=threshold_sigma, roundness_max=0.2, median=0, std=std
        )

        if ref_catalog is None or len(ref_catalog) == 0:
            print("No stars found in the reference catalog.")
            return None

        # Save the reference catalog
        self.ref_catalog = ref_catalog[
            ["xcentroid", "ycentroid", "flux", "flux_err", "snr", "roundness1", "roundness2"]
        ].to_pandas()

        # Filter out sources that are too close to the edges
        # and have low SNR
        self.filter_ref_catalog(snr_threshold=self.snr_threshold * np.sqrt(self.nStamps))
        self.mask_edge_ref_catalog(edge=edge_size)

        # Add additional information to the reference catalog
        self.add_ref_catalog_info(median, std)

        self.ref_catalog = self.ref_catalog.sort_values(by="snr", ascending=False)
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
        self.ref_catalog = self.ref_catalog[self.ref_catalog["snr"] > snr_threshold]

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
        self.ref_catalog = self.ref_catalog[
            (self.ref_catalog["xcentroid"] > x_min)
            & (self.ref_catalog["xcentroid"] < x_max)
            & (self.ref_catalog["ycentroid"] > y_min)
            & (self.ref_catalog["ycentroid"] < y_max)
        ]

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
        self.ref_catalog["bias"] = median
        self.ref_catalog["noise"] = std
        self.ref_catalog["timestamp"] = self.timestamp[0].iso
        self.ref_catalog["expId"] = self.reader.expId
        self.ref_catalog["ampname"] = self.ampName
        self.ref_catalog["stamp"] = -1
        self.ref_catalog["id"] = np.arange(len(self.ref_catalog))

        # Convert the star positions to CCD coordinates
        self.ref_catalog["xpixel"] = self.ref_catalog["xcentroid"] + self.min_x
        self.ref_catalog["ypixel"] = self.ref_catalog["ycentroid"] + self.min_y
        pass

    def convert_to_focal_plane(self):
        """
        Convert the star positions to focal plane coordinates.
        """
        if len(self.stars["xpixel"]) > 0:
            # Convert the star positions to focal plane coordinates
            xfp, yfp = pixel_to_focal(self.stars["xpixel"], self.stars["ypixel"], self.detector)
        else:
            xfp, yfp = None, None
        self.stars["xfp"] = xfp
        self.stars["yfp"] = yfp
        pass

    def convert_to_altaz(self):
        """
        Convert the star positions to altaz coordinates.
        """
        if len(self.stars["xfp"]) > 0:
            from transformation import CoordinatesToAltAz

            coord = CoordinatesToAltAz(
                self.reader.seqNum, self.reader.dayObs, self.detname, butler=self.reader.butler
            )
            az, alt = coord.convert_pixels_to_altaz(self.stars["xpixel"], self.stars["ypixel"])
        else:
            az, alt = None, None
        self.stars["alt"] = alt
        self.stars["az"] = az
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
        x = star["xcentroid"]
        y = star["ycentroid"]

        # Get the cutout
        cutout = Cutout2D(self.stacked, (x, y), size=size, mode="partial", fill_value=np.nan)
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
        x = star["xcentroid"]
        y = star["ycentroid"]

        # Get the cutout
        cutout = Cutout2D(self.image_list[stamp_id], (x, y), size=size, mode="partial", fill_value=np.nan)
        return cutout.data

    def plot_stacked_sources(
        self, lo=10, hi=98, marker_color="firebrick", marker_size=12, annotate_ids=False, ax=None
    ):
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
        _ = ax.imshow(stacked, origin="lower", cmap="Greys", vmin=vmin, vmax=vmax)

        # 5) Overlay markers at each reference position
        xs = self.ref_catalog["xcentroid"]
        ys = self.ref_catalog["ycentroid"]

        ax.plot(
            xs,
            ys,
            markersize=marker_size,
            marker="x",
            color=marker_color,
            linestyle="",
            label="Reference sources",
        )

        # 6) Optionally annotate IDs
        if annotate_ids and "id" in self.ref_catalog.colnames:
            for row in self.ref_catalog:
                ax.text(
                    row["xcentroid"] + 3, row["ycentroid"] + 3, str(row["id"]), color=marker_color, fontsize=8
                )

        # 7) Clean up axes
        ax.set_xlim(0, stacked.shape[1])
        ax.set_ylim(0, stacked.shape[0])
        # ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_title(
            "Stacked guider image with reference sources" + "\n" + f"({self.reader.key}, {self.reader.expId})"
        )

        # Plot the edges of the frame
        # Unpack your edges
        x_min, y_min, x_max, y_max = self.edges_frame

        # Compute width and height
        width = x_max - x_min
        height = y_max - y_min

        # Create a Rectangle with no fill, grey edge, dashed line
        rect = Rectangle(
            (x_min, y_min),  # lower-left corner
            width,
            height,
            fill=False,
            edgecolor=marker_color,
            linestyle="--",
            linewidth=2.5,
        )

        # Add it to your axes
        ax.add_patch(rect)

        fig.tight_layout()
        return fig, ax

    def plot_drifts_with_errors(
        self, motions=None, stamp_axis=None, figsize=(6, 4), fig=None, ax=None, **plot_kw
    ):
        """
        Plot the median drift ± robust σ (from MAD) for ΔX and ΔY.

        Parameters
        ----------
        motions : ndarray (n_stars, 2, n_stamps), optional
            If None, will call self.track_stars().
        stamp_axis : array-like, optional
            X-axis values (e.g. 0,1,2,… or timestamps). Defaults to
            range(n_stamps).
        figsize : tuple
            Figure size.
        plot_kw : dict
            Additional keywords passed to plt.errorbar (markersize, alpha,
            etc.).

        Returns
        -------
        fig, ax : matplotlib objects
        """
        # 1) get the motions array
        if motions is None:
            if hasattr(self, "motions"):
                motions = self.motions
            else:
                motions, _ = self.track_stars()
        nstars, _, nstamps = motions.shape

        # 2) define the x-axis
        if stamp_axis is None:
            stamp_axis = np.arange(nstamps)

        # 3) compute median and MAD-based sigma across stars, per stamp
        #    motions[:,0,:] is ΔX for all stars, all stamps
        med_dx = np.nanmedian(motions[:, 0, :], axis=0)
        med_dy = np.nanmedian(motions[:, 1, :], axis=0)

        sig_dx = mad_std(motions[:, 0, :], axis=0, ignore_nan=True)
        sig_dy = mad_std(motions[:, 1, :], axis=0, ignore_nan=True)

        sig_dx[np.isnan(sig_dx)] = np.nanmedian(sig_dx)
        sig_dy[np.isnan(sig_dy)] = np.nanmedian(sig_dy)

        # 4) do the errorbar plot
        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        defaults = dict(fmt="o", capsize=3, markersize=5, alpha=0.8)
        defaults.update(plot_kw)

        ax.errorbar(stamp_axis, med_dx, yerr=sig_dx, color="k", label="CCD ΔX", **defaults)
        ax.errorbar(stamp_axis, med_dy, yerr=sig_dy, color="firebrick", label="CCD ΔY", **defaults)
        ax.axhline(0, color="grey", lw=1, ls="--")

        # --- new jitter annotation ---
        jitter_pix = np.nanstd(motions)
        jitter_arcsec = jitter_pix * 0.2
        txt = f"Jitter (rms): {jitter_pix:.2f} pixel, {jitter_arcsec:.2f} arcsec "
        ax.text(
            0.02,
            0.98,
            txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            color="grey",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        # 5) polish
        ax.set_xlabel("Stamp #")
        ax.set_ylabel("Median offset (pixels)")
        ax.set_title(
            f"Star drift over {nstamps} stamps ({nstars} stars)"
            + "\n"
            + f"({self.reader.key}, {self.reader.expId})"
        )
        ax.legend(frameon=False)
        ax.grid(True, ls=":", color="grey", alpha=0.5)
        # fig.tight_layout()
        return fig, ax

    def plot_scatter_stamp(self, magOffsets=None, stamp_axis=None, figsize=(8, 5), **plot_kw):
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
        defaults = dict(fmt="o", capsize=3, markersize=5, alpha=0.8)
        defaults.update(plot_kw)

        for i in range(nstars):
            p = ax.plot(stamp_axis, magOffsets[i] - np.nanmedian(magOffsets[i]), alpha=0.5, ls="--", lw=0.5)
            c = p[0].get_color()
            ax.scatter(
                stamp_axis,
                magOffsets[i] - np.nanmedian(magOffsets[i]),
                color=c,
                alpha=0.75,
                label=f"Star {i + 1}",
            )
        ax.axhline(0, color="grey", lw=1, ls="--")

        # --- new jitter annotation ---
        jitter_pix = mad_std(magOffsets, ignore_nan=True)
        # jitter_arcsec = jitter_pix * 0.2
        txt = f"\\sigma (rms): {jitter_pix:.2f} mag "
        ax.text(
            0.02,
            0.98,
            txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            color="grey",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        # 5) polish
        ax.set_xlabel("Stamp #")
        ax.set_ylabel("Mag Offset: stamp-ref [mag]")
        ax.set_title(
            f"Star flux variation over {nstamps} stamps ({nstars} stars)"
            + "\n"
            + f"({self.reader.key}, {self.reader.expId})"
        )
        ax.legend(frameon=False, ncol=2)
        ax.grid(True, ls=":", color="grey", alpha=0.5)
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
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    return mean, median, std


def background_model(image, fwhm=10.0):
    sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    threshold = detect_threshold(image, nsigma=2.0, sigma_clip=sigma_clip)
    segment_img = detect_sources(image, threshold, npixels=10)
    footprint = circular_footprint(radius=1 * fwhm)

    mask = segment_img.make_source_mask(footprint=footprint)
    mean, median, std = sigma_clipped_stats(image, sigma=3.0, mask=mask)
    streak_mask = np.abs(image - median) > 2 * std
    mean, median, std = sigma_clipped_stats(image, sigma=3.0, mask=mask | streak_mask)
    final_mask = mask | streak_mask
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
    npix = sources["npix"]
    flux = sources["flux"]
    # Poisson term + background term
    flux_err = np.sqrt(flux / gain + npix * std**2)
    sources["flux_err"] = flux_err
    sources["snr"] = flux / flux_err
    return sources


if __name__ == "__main__":
    # Example usage
    # from reading import readGuiderData
    seqNum, dayObs = 591, 20250425
    reader = ReadGuiderData(seqNum, dayObs, view="ccd")
    reader.load()

    starsList = []
    for raftccd in reader.getGuiderNames():
        print(f"Processing {raftccd}")
        finder = starGuideFinder(reader, raftccd, fwhm=10)
        finder.run()
        starsList.append(finder.stars)

    # Concatenate all the catalogs
    stars = pd.concat(starsList, ignore_index=True)

    # Some stats:
    # The number of valid (not nan) stamp measurements per star
    stars.groupby("star_id")[["stamp", "xpixel"]].count()

    # The centroid error for each stamp
    stars.groupby("stamp")[["dalt", "daz"]].std()

    # The mean jitter in arcsec
    stars[["dalt", "daz"]].std()
