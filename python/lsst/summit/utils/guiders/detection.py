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

__all__ = ["StarGuideFinder"]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sep
from astropy.nddata import Cutout2D
from astropy.stats import SigmaClip, mad_std, sigma_clipped_stats
from matplotlib.patches import Rectangle

from lsst.obs.lsst.cameraTransforms import LsstCameraTransforms
from lsst.summit.utils.guiders.reading import GuiderDataReader
from lsst.summit.utils.guiders.transformation import pixel_to_focal

# from photutils.detection import DAOStarFinder # replaced by sep
# from photutils.background import Background2D, MedianBackground
# from photutils.segmentation import detect_threshold, detect_sources
# from photutils.utils import circular_footprint


DEFAULT_COLUMNS = [
    "xcentroid",
    "ycentroid",
    "xpixel",
    "ypixel",
    "xpixel_ref",
    "ypixel_ref",
    "xfp",
    "yfp",
    "xfp_ref",
    "yfp_ref",
    "alt",
    "az",
    "alt_ref",
    "az_ref",
    "detector",
    "det_id",
    "amp_name",
    "expId",
    "star_id",
    "stamp",
    "timestamp",
    "filter",
    "mag_offset",
    "ixx",
    "iyy",
    "ixy",
    "ixx_err",
    "iyy_err",
    "ixy_err",
    "fwhm",
    "e1",
    "e2",
    "flux",
    "flux_err",
    "snr",
    "dx",
    "dy",
    "dxfp",
    "dyfp",
    "dalt",
    "daz",
]


class StarGuideFinder:
    """
    Class to find stars in the Guider data and
    measure astrometric/photometric statistics.

    Parameters
    ----------
    reader : GuiderDataReader
        Reader object with loaded guider exposures.
    detector_name : str
        Name of the detector (e.g., 'R22_S11').
    camera : Camera, optional
        Camera model; if None, uses reader.camera.
    psf_fwhm : float, default=6.0
        Expected PSF full-width at half-maximum in pixels.
    min_snr : float, default=3.0
        Minimum signal-to-noise ratio for stars in the reference catalog.
    min_stamp_detections : int, default=30
        Minimum number of stamps a star must be detected
        in to be considered valid.
    edge_margin : int, default=30
        Pixels from CCD edge to exclude reference sources.
    max_ellipticity : float, default=0.1
        Maximum allowed ellipticity for sources.
    """

    def __init__(
        self,
        reader,
        detector_name,
        camera=None,
        psf_fwhm=6.0,
        min_snr=3.0,
        min_stamp_detections=30,
        edge_margin=30,
        max_ellipticity=0.1,
    ):
        self.reader = reader
        self.view = reader.view
        self.exp_id = reader.exp_id
        self.detector_name = detector_name
        self.reader.set_detector(detector_name)
        self.amp_name = reader.amp_name
        self.filter = reader.filter

        # detection and QC parameters
        self.psf_fwhm = psf_fwhm
        self.min_snr = min_snr
        self.min_stamp_detections = min_stamp_detections
        self.edge_margin = edge_margin
        self.max_ellipticity = max_ellipticity

        # ROI geometry
        self.roi_row = reader.roi_row
        self.roi_col = reader.roi_col
        self.roi_rows = reader.roi_rows
        self.roi_cols = reader.roi_cols
        self.n_stamps = reader.n_stamps
        self.timestamp = reader.timestamp

        # camera and transforms
        self.camera = camera or reader.camera
        self.detector = self.camera[self.detector_name]
        self.lct = LsstCameraTransforms(self.camera, self.detector_name)

        # initialize outputs
        self.stars = pd.DataFrame(columns=DEFAULT_COLUMNS)
        self.output_catalog = pd.DataFrame(columns=DEFAULT_COLUMNS)

        # compute amplifier offset
        self.get_amplifier_lowest_corner()

    @classmethod
    def run_all_guiders(
        cls, reader, camera=None, psf_fwhm=12.0, min_snr=3.0, min_stamp_detections=30, max_ellipticity=0.1
    ):
        """
        Run detection and tracking on all guider detectors.

        Returns a concatenated DataFrame of all selected stars.
        """
        stars_list = []
        for det in reader.get_guider_names():
            finder = cls(
                reader,
                det,
                camera=camera,
                psf_fwhm=psf_fwhm,
                min_snr=min_snr,
                min_stamp_detections=min_stamp_detections,
                max_ellipticity=max_ellipticity,
            )
            stars_list.append(finder.run_source_detection())

        # Filter out empty DataFrames
        stars_list = [df for df in stars_list if not df.empty]
        if not stars_list:
            return pd.DataFrame()
        stars = pd.concat(stars_list, ignore_index=True)
        return stars

    def run_source_detection(self):
        """
        Executes detection & tracking for *one* guider:
        1. Stack the sequence of guider exposures.
        2. Build a reference catalog from the stacked image.
        3. If no stars found, return early (empty DataFrame).
        4. Track each reference star across all stamps (makes cutouts)
        5. Filter out stars within `min_stamp_detections` stamps.
        6. Convert star positions to CCD pixels if the view is DVCS.
        7. Convert positions to focal-plane (xfp,yfp) and AltAz (alt,az).
        8. Assign a unique global `star_id` per guider.
        9. Compute pixel & arcsecond offsets (`dx,dy,dalt,daz`).
        10. Return the populated `self.stars` DataFrame.
        """
        # Stack the images
        self.stacked_image = self.stack_guider_images()

        # Build the reference catalog
        self.build_ref_catalog()

        # If no reference catalog was built, exit early
        if len(self.ref_catalog) == 0:
            return self.stars

        # Track motion for all stars
        self.track_stars()

        # Filter the minimum number of detections per stamp
        self.filter_min_stamp_detections()

        # convert to focal plane coordinates/ altaz
        # action to check weather the view is dvcs
        self.convert_if_dvcs_to_ccd()
        self.convert_to_focal_plane()
        self.convert_to_altaz()

        # Set unique IDs
        self.set_unique_id()

        # Compute offsets
        self.compute_offsets()
        return self.stars

    def convert_if_dvcs_to_ccd(self):
        """
        Convert xcentroid/ycentroid to CCD pixels if the view is DVCS.
        """
        if self.view == "dvcs":
            # Convert xcentroid/ycentroid to CCD pixels
            stamps = self.reader.dataset[self.detector_name]

            # get CCD<->DVCS translation from the stamps
            _, _, dvcs = stamps.getArchiveElements()[0]

            xy = self.stars[["xpixel", "ypixel"]].to_numpy()
            xy_ref = self.stars[["xpixel_ref", "ypixel_ref"]].to_numpy()

            x_ccd, y_ccd = dvcs(xy[:, 0], xy[:, 1])
            x_ccd_ref, y_ccd_ref = dvcs(xy_ref[:, 0], xy_ref[:, 1])
            self.stars["xpixel"] = x_ccd
            self.stars["ypixel"] = y_ccd
            self.stars["xpixel_ref"] = x_ccd_ref
            self.stars["ypixel_ref"] = y_ccd_ref
        elif self.view == "ccd":
            # No conversion needed for CCD view
            pass
        pass

    def set_unique_id(self):
        # 1) Build a detector→index map (0,1,2,…)
        det_map = self.reader.guiders

        # 2) Create a numeric “global” star_id:
        #    global_id = det_index * 10000 + local star_id
        self.stars["det_id"] = self.stars["detector"].map(det_map)
        star_local = self.stars["star_id"].astype(int)
        self.stars["star_id"] = self.stars["det_id"] * 10000 + star_local
        self.stars["expId"] = self.exp_id

    def get_amplifier_lowest_corner(self):
        """
        Compute the lowest corner of the amplifier in CCD coordinates.
        """
        a, b = self.lct.ampPixelToCcdPixel(self.roi_col, self.roi_row, self.amp_name)
        c, d = self.lct.ampPixelToCcdPixel(
            self.roi_col + self.roi_cols, self.roi_row + self.roi_rows, self.amp_name
        )
        self.min_x = int(min(a, c))
        self.min_y = int(min(b, d))
        pass

    def compute_offsets(self):
        """
        Compute the offsets for each star in the catalog.
        """
        # Compute all your offsets
        self.stars["dx"] = self.stars["xpixel"] - self.stars["xpixel_ref"]
        self.stars["dy"] = self.stars["ypixel"] - self.stars["ypixel_ref"]
        self.stars["dxfp"] = self.stars["xfp"] - self.stars["xfp_ref"]
        self.stars["dyfp"] = self.stars["yfp"] - self.stars["yfp_ref"]
        self.stars["dalt"] = (self.stars["alt"] - self.stars["alt_ref"]) * 3600
        self.stars["daz"] = (self.stars["az"] - self.stars["az_ref"]) * 3600

        # Correct for cos(alt) in daz
        self.stars["daz"] = np.cos(self.stars["alt_ref"] * np.pi / 180) * self.stars["daz"]
        pass

    def filter_min_stamp_detections(self):
        """
        Select the best star per stamp based on the number of detections.
        """
        df = self.output_catalog.copy()
        df.drop(columns=["detector", "amp_name"], inplace=True, errors="ignore")

        n_stamps = df.groupby("star_id").count()["flux"].values

        cut = n_stamps > self.min_stamp_detections
        starids = df.groupby("star_id").count().loc[cut].index

        df = df[df["star_id"].isin(starids)].copy()
        df.sort_values(["stamp", "snr"], ascending=[True, False], inplace=True)
        df.reset_index(drop=True, inplace=True)

        df["detector"] = self.detector_name
        df["amp_name"] = self.amp_name
        df["filter"] = self.filter
        self.stars = df.copy()

    def track_star_stamp(
        self,
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
        ref_x, ref_y = ref["xcentroid"], ref["ycentroid"]
        fwhm = self.psf_fwhm
        amp_name = ref["amp_name"]

        ref["stamp"] = -1
        ref["star_id"] = star_id
        ref["mag_offset"] = 0.0  # reference is always 0 mag offset

        # # --- reference row ---
        sel_columns = [
            "star_id",
            "stamp",
            "amp_name",
            "filter",
            "xcentroid",
            "ycentroid",
            "xpixel",
            "ypixel",
            "xerr",
            "yerr",
            "flux",
            "flux_err",
            "snr",
            "mag_offset",
            "ixx",
            "iyy",
            "ixy",
            "ixx_err",
            "iyy_err",
            "ixy_err",
            "fwhm",
        ]
        rows.append(ref[sel_columns].copy())

        mask_cutout = Cutout2D(self.mask_streak, (ref_x, ref_y), size=50, mode="partial", fill_value=False)

        # --- per‐stamp measurements ---
        # the first two stamps are taken the shutter is fully open
        for si in range(2, self.n_stamps):
            stamp = self.image_list[si]
            isr = stamp - np.nanmedian(stamp)

            cutout = Cutout2D(isr, (ref_x, ref_y), size=50, mode="partial", fill_value=np.nan)

            _, median, std = sigma_clipped_stats(cutout.data, sigma=3.0, mask=mask_cutout.data)
            sources = measure_star_in_aperture(
                cutout.data - median, aperture_radius=fwhm, std_bkg=std, gain=1.0, mask=mask_cutout.data
            )

            if len(sources) == 0:
                # No sources detected in this stamp, skip it
                continue

            sources["star_id"] = star_id
            sources["stamp"] = si
            sources["amp_name"] = amp_name
            sources["filter"] = self.filter

            # Centroid in amplifier roi coordinates
            sources["xcentroid"] += cutout.xmin_original
            sources["ycentroid"] += cutout.ymin_original

            # pixel in image view coordinates
            # if ccd view, ccd pixel
            sources["xpixel"] = sources["xcentroid"] + self.min_x
            sources["ypixel"] = sources["ycentroid"] + self.min_y
            rows.append(sources.iloc[0])

        df = pd.DataFrame(rows)

        if len(df) == 1:
            return

        # Define the reference as the median of the other stamps
        flux_med = np.nanmedian(df["flux"][1:]) + 1e-12  # avoid division by zero
        df["mag_offset"] = -2.5 * np.log10((df["flux"] + 1e-12) / flux_med)

        df["xcentroid_ref"] = np.nanmedian(df["xcentroid"][1:])
        df["ycentroid_ref"] = np.nanmedian(df["ycentroid"][1:])
        df["xpixel_ref"] = np.nanmedian(df["xpixel"][1:])
        df["ypixel_ref"] = np.nanmedian(df["ypixel"][1:])
        return df

    def track_stars(self) -> pd.DataFrame:
        """
        Track all reference stars; return one big DataFrame with every
        (star_id, stamp) row, including the reference (stamp=-1).
        """
        dfs = []
        for i in range(len(self.ref_catalog)):
            df_i = self.track_star_stamp(i)
            if df_i is not None and not df_i.empty:
                dfs.append(df_i)

        if not dfs:
            return self.output_catalog

        # Concatenate all DataFrames
        output = pd.concat(dfs, ignore_index=True)

        # filter SNR
        output = output[output["snr"] > self.min_snr]
        # filter min flux
        output = output[output["flux"] > 1e-6]
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
        image_list = [self.reader.read(stamp, self.detector_name) for stamp in range(self.n_stamps)]
        # parallel overscan region correction
        image_list = [img - np.nanmedian(img, axis=0) for img in image_list]

        # stack with the sum
        stacked = np.nansum(np.array(image_list), axis=0)

        self.image_list = image_list
        self.stacked = stacked
        return stacked

    def build_ref_catalog(self, threshold_sigma=3.0, edge_margin=None):
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

        if edge_margin is None:
            edge_margin = self.edge_margin

        # Find Bad columns
        streak_mask = find_bad_columns(stacked_image, nsigma=2)
        self.mask_streak = streak_mask

        # # model the background

        ref_catalog = run_sextractor(
            stacked_image,
            aperture_radius=self.psf_fwhm,
            th=threshold_sigma,
            max_ellipticity=self.max_ellipticity,
            mask=streak_mask,
        )

        sel_columns = [
            "star_id",
            "stamp",
            "detector",
            "det_id",
            "amp_name",
            "xcentroid",
            "ycentroid",
            "xpixel",
            "ypixel",
            "xerr",
            "yerr",
            "xfp",
            "yfp",
            "alt",
            "az",
            "ixx",
            "iyy",
            "ixy",
            "ixx_err",
            "iyy_err",
            "ixy_err",
            "fwhm",
            "flux",
            "flux_err",
            "snr",
            "mag_offset",
            "dx",
            "dy",
            "dxfp",
            "dyfp",
            "dalt",
            "daz",
        ]

        if ref_catalog is None or len(ref_catalog) == 0:
            print(f"Error: No stars found in the reference catalog for {self.detector_name}.")
            self.ref_catalog = pd.DataFrame(columns=sel_columns)
            self.stars = pd.DataFrame(columns=sel_columns)
            return
        # Save the reference catalog
        self.ref_catalog = ref_catalog

        # Save the centroid in ccd coordinates
        self.ref_catalog["xpixel"] = self.ref_catalog["xcentroid"] + self.min_x
        self.ref_catalog["ypixel"] = self.ref_catalog["ycentroid"] + self.min_y

        # Filter out sources that are too close to the edges
        # and have low SNR
        self.filter_ref_catalog(snr_threshold=self.min_snr * np.sqrt(self.n_stamps))
        self.mask_edge_ref_catalog(edge=edge_margin)

        # Add additional information to the reference catalog
        median, _, std = sigma_clipped_stats(stacked_image, sigma=3.0, mask=streak_mask)
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
        self.ref_catalog["expId"] = self.reader.exp_id
        self.ref_catalog["amp_name"] = self.amp_name
        self.ref_catalog["stamp"] = -1
        self.ref_catalog["filter"] = self.filter
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
            xfp_ref, yfp_ref = pixel_to_focal(
                self.stars["xpixel_ref"], self.stars["ypixel_ref"], self.detector
            )
        else:
            xfp, yfp = None, None
            xfp_ref, yfp_ref = None, None

        self.stars["xfp"] = xfp
        self.stars["yfp"] = yfp
        self.stars["xfp_ref"] = xfp_ref
        self.stars["yfp_ref"] = yfp_ref
        pass

    def convert_to_altaz(self):
        """
        Convert the star positions to altaz coordinates.
        """
        if len(self.stars["xfp"]) > 0:
            from transformation import CoordinatesToAltAz

            coord = CoordinatesToAltAz(
                self.reader.seqNum, self.reader.dayObs, self.detector_name, butler=self.reader.butler
            )
            az, alt = coord.convert_pixels_to_altaz(self.stars["xpixel"], self.stars["ypixel"])
            az_ref, alt_ref = coord.convert_pixels_to_altaz(
                self.stars["xpixel_ref"], self.stars["ypixel_ref"]
            )
        else:
            az, alt = None, None
            az_ref, alt_ref = None, None
        self.stars["alt"] = alt
        self.stars["az"] = az
        self.stars["alt_ref"] = alt_ref
        self.stars["az_ref"] = az_ref
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
        _ = ax.imshow(
            self.mask_streak, origin="lower", cmap="Reds", alpha=0.5, vmin=0, vmax=1, interpolation="nearest"
        )

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
            "Stacked guider image with reference sources" + "\n" + f"({self.detector_name}, {self.exp_id})"
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

    def plot_drifts_with_errors(self, stars=None, figsize=(6, 4), fig=None, ax=None, **plot_kw):
        """
        Plot the median drift ± robust σ (from MAD) for ΔX and ΔY per stamp.
        """
        if stars is None:
            if not hasattr(self, "stars"):
                raise ValueError("No output catalog found. Run run_source_detection() first.")
            stars = self.stars

        # Remove the stacked sources
        stars = stars[stars["stamp"] != -1].copy()

        # Group by stamp
        grouped = stars.groupby("stamp")
        stamps = np.array(sorted(stars["stamp"].unique()))

        # Per-stamp median
        med_dx = grouped["dxfp"].median().to_numpy() * 100
        med_dy = grouped["dyfp"].median().to_numpy() * 100

        # Per-stamp robust sigma (MAD)
        sig_dx = grouped["dxfp"].apply(lambda x: mad_std(x, ignore_nan=True)).to_numpy() * 100
        sig_dy = grouped["dyfp"].apply(lambda x: mad_std(x, ignore_nan=True)).to_numpy() * 100
        err_x = grouped["xerr"].median().to_numpy()
        err_y = grouped["yerr"].median().to_numpy()
        sig_dx = np.hypot(sig_dx, err_x)
        sig_dy = np.hypot(sig_dy, err_y)

        nstars = stars["star_id"].nunique()
        n_stamps = stars["stamp"].nunique()

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        defaults = dict(fmt="o", capsize=3, markersize=5, alpha=0.8)
        defaults.update(plot_kw)

        # Median drift/error bars
        ax.errorbar(stamps, med_dx, yerr=sig_dx, color="k", label="Median ΔX", **defaults)
        ax.errorbar(stamps, med_dy, yerr=sig_dy, color="firebrick", label="Median ΔY", **defaults)

        ax.axhline(0, color="grey", lw=1, ls="--")

        # --- new std_centroid annotation ---
        std_centroid_pix = np.nanstd(stars[["dx", "dy"]].to_numpy())
        std_centroid_arcsec = std_centroid_pix * 0.2
        txt = f"std_centroid (rms): {std_centroid_pix:.2f} pixel, {std_centroid_arcsec:.2f} arcsec"
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

        # Polish
        ax.set_xlabel("Stamp #")
        ax.set_ylabel("Offset (pixels)")
        ax.set_title(
            f"Star drift over {n_stamps} stamps ({nstars} stars)"
            + f"\n({getattr(self, 'detname', '?')}, {getattr(self, 'exp_id', '?')})"
        )
        ax.legend(frameon=False, loc="upper right", ncol=2)
        ax.grid(True, ls=":", color="grey", alpha=0.5)
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
        nstars, n_stamps = magOffsets.shape

        # 2) define the x-axis
        if stamp_axis is None:
            stamp_axis = np.arange(n_stamps)

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

        # --- new std_centroid annotation ---
        std_centroid_pix = mad_std(magOffsets, ignore_nan=True)
        # std_centroid_arcsec = std_centroid_pix * 0.2
        txt = f"\\sigma (rms): {std_centroid_pix:.2f} mag "
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
            f"Star flux variation over {n_stamps} stamps ({nstars} stars)"
            + "\n"
            + f"({self.reader.key}, {self.reader.exp_id})"
        )
        ax.legend(frameon=False, ncol=2)
        ax.grid(True, ls=":", color="grey", alpha=0.5)
        fig.tight_layout()
        return fig, ax

    @staticmethod
    def format_std_centroid_summary(stats_df: pd.DataFrame) -> str:
        """
        Pretty string summary of centroid stdev. stats from run_all_guiders.
        """
        # handle both dicts and DataFrames
        if (isinstance(stats_df, pd.DataFrame) and stats_df.empty) or (
            isinstance(stats_df, dict) and not stats_df
        ):
            return "No centroid stdev. statistics available."

        # if it's a DataFrame, extract the one-row dict
        if isinstance(stats_df, pd.DataFrame):
            stats = stats_df.iloc[0].to_dict()
        else:
            stats = stats_df

        js = stats
        summary = (
            f"\nGlobal centroid stdev. Summary Across All Guiders\n"
            f"{'-' * 45}\n"
            f"  - centroid stdev.  (AZ): {js['std_centroid_az']:.3f} arcsec (raw)\n"
            f"  - centroid stdev. (ALT): {js['std_centroid_alt']:.3f} arcsec (raw)\n"
            f"  - centroid stdev.  (AZ): {js['std_centroid_corr_az']:.3f} arcsec (linear corr)\n"
            f"  - centroid stdev. (ALT): {js['std_centroid_corr_alt']:.3f} arcsec (linear corr)\n"
            f"  - Drift Rate       (AZ): {15 * js['drift_rate_az']:.3f} arcsec per exposure\n"
            f"  - Drift Rate      (ALT): {15 * js['drift_rate_alt']:.3f} arcsec per exposure\n"
            f"  - Zero Offset      (AZ): {js['offset_zero_az']:.3f} arcsec\n"
            f"  - Zero Offset     (ALT): {js['offset_zero_alt']:.3f} arcsec"
        )
        return summary

    @staticmethod
    def format_photometric_summary(phot_stats: pd.DataFrame) -> str:
        """
        Pretty-print summary of photometric variation statistics.
        """
        if (isinstance(phot_stats, pd.DataFrame) and phot_stats.empty) or (
            isinstance(phot_stats, dict) and not phot_stats
        ):
            return "No photometric statistics available."

        # if it's a DataFrame, extract the one-row dict
        if isinstance(phot_stats, pd.DataFrame):
            stats = phot_stats.iloc[0].to_dict()
        else:
            stats = phot_stats

        return (
            "\nPhotometric Variation Summary\n"
            "-------------------------------\n"
            f"  - Mag Drift Rate:      {stats['mag_offset_rate']:.5f} mag/sec\n"
            f"  - Mag Zero Offset:     {stats['mag_offset_zero']:.5f} mag\n"
            f"  - Mag RMS (detrended): {stats['mag_offset_rms']:.5f} mag"
        )

    @staticmethod
    def format_stats_summary(summary: pd.DataFrame) -> str:
        """
        Pretty-print only the stats that are present in `summary`.
        Expects keys like:
          n_guiders, n_stars, n_measurements, fraction_valid_stamps,
          N_<detector>, std_centroid_*, mag_offset_*, etc.
        """
        if (isinstance(summary, pd.DataFrame) and summary.empty) or (
            isinstance(summary, dict) and not summary
        ):
            return "No summary statistics available."
        # if it's a DataFrame, extract the one-row dict
        if isinstance(summary, pd.DataFrame):
            summary = summary.iloc[0].to_dict()

        lines = ["-" * 50]

        # Basic overall stats
        lines.append(f"Number of Guiders: {int(summary['n_guiders'])}")
        lines.append(f"Number of Unique Stars: {int(summary['n_stars'])}")
        lines.append(f"Total Measurements: {int(summary['n_measurements'])}")
        frac = summary["fraction_valid_stamps"]
        lines.append(f"Fraction Valid Stamps: {frac:.3f}")

        # Per-guider counts (keys begin with 'N_')
        guider_keys = sorted(k for k in summary if k.startswith("N_"))
        if guider_keys:
            lines.append("\nStars per Guider:")
            for k in guider_keys:
                lines.append(f"  - {k[2:]}: {int(summary[k])}")
        return "\n".join(lines)

    @classmethod
    def run_guide_stats(
        cls,
        reader,
        camera=None,
        psf_fwhm=12.0,
        min_snr=3.0,
        min_stamp_detections=30,
        max_ellipticity=0.1,
        vebose=False,
    ):
        """
        Run all guiders, then produce a one‐row DataFrame containing:
        - Number of valid guiders
        - Number of unique stars
        - Number of stars per guider (e.g., N_R02_S11, N_R22_S11, etc.)
        - Number of star measurements across all guiders and stamps
        - Global centroid stdev. statistics (az/alt, rates, zero-points)
        - Global photometric variation statistics (drift rate, zero-point, RMS)
        - Fraction of valid stamp measurements
        """
        # 1) Run detections across all guiders
        stars = cls.run_all_guiders(
            reader,
            psf_fwhm=psf_fwhm,
            min_snr=min_snr,
            camera=camera,
            min_stamp_detections=min_stamp_detections,
            max_ellipticity=max_ellipticity,
        )

        # 2) Build the stats via our new helper
        stats = assemble_stats(stars, reader)

        if vebose:
            print(cls.format_stats_summary(stats))
            print(cls.format_std_centroid_summary(stats))
            print(cls.format_photometric_summary(stats))
            print("-" * 50)

        return stars, stats


def assemble_stats(stars: pd.DataFrame, reader) -> pd.DataFrame:
    """
    Given a (possibly empty) stars DataFrame and a reader,
    compute and return the one‐row summary stats DataFrame.
    """
    # 1) If empty, build an empty stats frame with the right columns
    if stars.empty:
        return make_empty_summary(reader)

    # 2) Number of valid guiders
    n_guiders = stars["detector"].nunique()

    # 3) Number of unique stars
    n_unique = stars["star_id"].nunique()

    # 4) Stars per guider
    counts = stars.groupby("detector")["star_id"].nunique().to_dict()
    stars_per_guiders = {f"N_{det}": counts.get(det, 0) for det in reader.guiders.keys()}

    # 5) Valid measurements
    mask_valid = (stars["stamp"] >= 0) & (stars["xpixel"].notna())
    n_meas = int(mask_valid.sum())

    # 6/7) Global std_centroid & photometric stats
    std_centroid = measure_std_centroid_stats(stars)
    phot = measure_photometric_variation(stars)

    # 8) Fraction valid
    total_possible = n_unique * reader.n_stamps
    frac_valid = n_meas / total_possible if total_possible > 0 else np.nan

    # 9) Assemble
    summary = {
        "n_guiders": n_guiders,
        "n_stars": n_unique,
        "n_measurements": n_meas,
        "fraction_valid_stamps": frac_valid,
        **stars_per_guiders,
        **std_centroid,
        **phot,
    }
    df = pd.DataFrame([summary])
    df["seqNum"] = reader.seqNum
    df["filter"] = reader.filter
    df["expId"] = reader.exp_id
    return df


def measure_std_centroid_stats(stars: pd.DataFrame) -> pd.DataFrame:
    """
    Compute global std_centroid statistics across all guiders.

    Parameters
    ----------
    stars : pd.DataFrame
        Concatenated star table across all guider detectors.

    Returns
    -------
    stars : pd.DataFrame
        DataFrame with new std_centroid statistic columns.
    """
    time = (stars.stamp.to_numpy() + 0.5) * 0.3  # seconds
    time = time.astype(np.float64)
    az = stars.daz.to_numpy()
    alt = stars.dalt.to_numpy()

    # Linear fits
    coefs_az = np.polyfit(time, az, 1)
    coefs_alt = np.polyfit(time, alt, 1)

    # Stats
    std_centroid_stats = {
        "std_centroid_az": mad_std(az),
        "std_centroid_alt": mad_std(alt),
        "std_centroid_corr_az": mad_std(az - np.polyval(coefs_az, time)),
        "std_centroid_corr_alt": mad_std(alt - np.polyval(coefs_alt, time)),
        "drift_rate_az": coefs_az[0],
        "drift_rate_alt": coefs_alt[0],
        "offset_zero_az": coefs_az[1],
        "offset_zero_alt": coefs_alt[1],
    }
    return std_centroid_stats


def measure_photometric_variation(stars: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Fit mag_offset vs time across all rows, compute drift rate,
    zero-point, and RMS scatter, then add these as constant columns to `stars`.
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


def make_empty_summary(reader):
    """
    Build a one‐row “zeroed” summary table with the full set of columns,
    filling in seqNum, dayObs, exp_id, and filter from the reader.

    Parameters
    ----------
    reader : your Reader class instance
        Must have attributes:
          - seqNum, dayObs, exp_id
          - filter (or you can replace with reader.filterName)
          - guiders: a dict (or iterable) of guider names

    Returns
    -------
    pd.DataFrame
        One‐row DataFrame with all summary columns set to zero,
        except seqNum/dayObs/exp_id/filter filled in.
    """
    # 1) Basic summary columns
    cols = [
        "n_guiders",
        "n_stars",
        "fraction_valid_stamps",
        "n_measurements",
    ]

    # 2) Per-guider counts
    guider_cols = [f"N_{g}" for g in reader.guiders.keys()]
    cols += guider_cols
    zero_row = {c: 0 for c in cols}

    # 3) Centroid statistics
    std_cols = [
        "std_centroid_az",
        "std_centroid_alt",
        "std_centroid_corr_az",
        "std_centroid_corr_alt",
        "offset_rate_az",
        "offset_rate_alt",
        "offset_zero_az",
        "offset_zero_alt",
        "drift_rate_az",
        "drift_rate_alt",
    ]
    for c in std_cols:
        zero_row[c] = np.nan

    # 4) Photometric stats
    phot_cols = ["mag_offset_rate", "mag_offset_zero", "mag_offset_rms"]
    cols += phot_cols

    for c in phot_cols:
        zero_row[c] = np.nan

    # 5) Add metadata columns
    meta_cols = ["seqNum", "dayObs", "expId", "filter"]
    cols += meta_cols

    # 7) Overwrite the metadata values
    zero_row["seqNum"] = reader.seqNum
    zero_row["dayObs"] = reader.dayObs
    zero_row["expId"] = reader.exp_id
    zero_row["filter"] = getattr(reader, "filter", None)

    return pd.DataFrame([zero_row])


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
    # sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    mean, median, std = sigma_clipped_stats(image, sigma=3.0)
    return mean, median, std


def background_model(image, fwhm=10.0, streak_mask=None):
    from photutils.segmentation import detect_sources, detect_threshold
    from photutils.utils import circular_footprint

    sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    threshold = detect_threshold(image, nsigma=5.0, sigma_clip=sigma_clip, mask=streak_mask)
    segment_img = detect_sources(image, threshold, npixels=fwhm / 2.0, mask=streak_mask)
    footprint = circular_footprint(radius=2 * fwhm)

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
    from photutils.detection import DAOStarFinder  # replaced by sep

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
    flux_err = np.sqrt(np.abs(flux) / gain + npix * std**2)
    sources["flux_err"] = flux_err
    sources["snr"] = flux / flux_err
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
        e1, e2,  # ellipticity components
      flux, flux_err, fwhm, snr
    """
    h, w = cutout_data.shape
    y, x = np.indices((h, w))
    x0, y0 = w / 2, h / 2

    # Combine aperture and external mask
    ap_mask = ((x - x0) ** 2 + (y - y0) ** 2) <= aperture_radius**2
    if mask is not None:
        valid1 = ap_mask & (~mask)
    else:
        valid1 = ap_mask

    # 1) initial flux & centroid
    data1 = np.where(valid1, cutout_data, 0.0)
    flux1 = np.nansum(data1)
    if flux1 <= 1e-12:
        # no signal → return all NaNs/zeros
        return pd.DataFrame(
            [
                {
                    "xcentroid": np.nan,
                    "ycentroid": np.nan,
                    "xerr": np.nan,
                    "yerr": np.nan,
                    "ixx": np.nan,
                    "iyy": np.nan,
                    "ixy": np.nan,
                    "ixx_err": np.nan,
                    "iyy_err": np.nan,
                    "ixy_err": np.nan,
                    "flux": 0.0,
                    "flux_err": 0.0,
                    "fwhm": np.nan,
                    "snr": 0.0,
                }
            ]
        )

    xcen1 = np.nansum(x * data1) / flux1
    ycen1 = np.nansum(y * data1) / flux1

    # 2) re-centered aperture
    valid2 = ((x - xcen1) ** 2 + (y - ycen1) ** 2) <= aperture_radius**2
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
        var = np.abs(data2) / gain + std_bkg**2
        ixx_err = np.sqrt(np.nansum(dx**4 * var) / flux2**2)
        iyy_err = np.sqrt(np.nansum(dy**4 * var) / flux2**2)
        ixy_err = np.sqrt(np.nansum((dx**2) * (dy**2) * var) / flux2**2)
    else:
        ixx = iyy = ixy = xerr = yerr = ixx_err = iyy_err = ixy_err = np.nan

    # 4) flux error & FWHM
    flux_err = np.sqrt(flux2 / gain + npix2 * std_bkg**2)
    sigma = np.sqrt(np.abs(ixx + iyy) / 2.0) if np.isfinite(ixx + iyy) else np.nan
    fwhm = 2.355 * sigma
    snr = flux2 / np.maximum(flux_err, 1e-12)

    return pd.DataFrame(
        [
            {
                "xcentroid": xcen1,
                "ycentroid": ycen1,
                "xerr": xerr,
                "yerr": yerr,
                "ixx": ixx,
                "iyy": iyy,
                "ixy": ixy,
                "e1": (ixx - iyy) / (ixx + iyy + 1e-12),
                "e2": (2 * ixy) / (ixx + iyy + 1e-12),
                "ixx_err": ixx_err,
                "iyy_err": iyy_err,
                "ixy_err": ixy_err,
                "flux": flux2,
                "flux_err": flux_err,
                "fwhm": fwhm,
                "snr": snr,
            }
        ]
    )


def find_bad_columns(img, mask=None, nsigma=3.0):
    """
    Identify bad columns in an image using per-column sigma-clipped statistics.

    Parameters
    ----------
    img : 2D ndarray
        The input image (can contain NaNs).
    mask : 2D bool ndarray or None
        True where pixels should be ignored in the stats (e.g. masked pixels).
    nsigma : float
        number of stdevs from the median to be considered a bada column.

    Returns
    -------
    bad_mask : 2D bool ndarray
        True for all pixels in columns flagged as bad.
    """
    # 1) Compute per-column sigma-clipped stats
    mean_cols, median_cols, std_cols = sigma_clipped_stats(
        img, sigma=3.0, maxiters=5, mask=mask, axis=0  # collapse over rows → one stat per column
    )

    # 2) Determine threshold
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


def run_sextractor(
    img, th=10, median=0, std=None, bkg_size=50, aperture_radius=5, max_ellipticity=0.1, gain=1.0, mask=None
):
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
    objects = sep.extract(img_sub, th, err=std, mask=bad_mask | mask)
    if len(objects) == 0:
        return pd.DataFrame()

    # Gather properties
    xcen, ycen = objects["x"], objects["y"]
    ixx, iyy, ixy = objects["x2"], objects["y2"], objects["xy"]
    ixx_err, iyy_err, ixy_err = objects["errx2"], objects["erry2"], objects["errxy"]

    flux, fluxerr, _ = sep.sum_circle(
        img_clean, xcen, ycen, aperture_radius, err=std, mask=bad_mask | mask, gain=gain
    )
    fwhm = 2.355 * np.sqrt(0.5 * (ixx + iyy))

    denom = ixx + iyy + 1e-12
    e1 = (ixx - iyy) / denom
    e2 = (2 * ixy) / denom

    # Filter: round and bright sources only
    mask = np.abs(e1) < max_ellipticity  # nearly round
    mask &= np.abs(e2) < max_ellipticity  # nearly round

    # Compute centroid errors from moment errors (astrometry error estimate)
    # For a Gaussian, σ_xcentroid ≈ sqrt(ixx_err) / sqrt(flux)
    # (cf. https://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/iracinstrumenthandbook/IRAC_Instrument_Handbook.pdf, Table 2.9)  # noqa: W505 E501
    centroid_x_err = np.sqrt(np.abs(ixx_err)) / np.sqrt(np.maximum(flux, 1e-6))
    centroid_y_err = np.sqrt(np.abs(iyy_err)) / np.sqrt(np.maximum(flux, 1e-6))

    df = pd.DataFrame(
        {
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
            "snr": flux[mask] / np.maximum(fluxerr[mask], 1e-6),
        }
    )

    return df


def build_star_pairs(df0, seqNum=300):
    df = df0[df0["seqNum"] == seqNum].copy()
    starids = df["star_id"].values
    regions = df["region"].values
    snr = df["snr"].values

    stars1, stars2 = [], []
    for r1, r2 in [(0, 3), (1, 2)]:
        m1 = regions == r1
        m2 = regions == r2
        ix = np.argsort(snr)

        s1 = starids[ix][m1]
        s2 = starids[ix][m2]
        stars1.append(s1)
        stars2.append(s2)


if __name__ == "__main__":
    # Example usage
    import pandas as pd

    from lsst.summit.utils.guiders.detection import starGuideFinder

    seqNum, dayObs = 591, 20250425
    reader = GuiderDataReader(seqNum, dayObs, view="ccd")
    reader.load()

    # Run the source detection for all guider
    # return the stars DataFrame with all the measurements
    # return some stats information of
    # the number of stars, std_centroid, photometric variance
    stars, stats = starGuideFinder.run_guide_stats(reader, psf_fwhm=10, min_snr=10)

    # Some stats
    # The number of valid (not nan) stamp measurements per star
    stars.groupby("star_id")[["stamp", "xpixel"]].count()

    # The centroid error for each stamp
    stars.groupby("stamp")[["dalt", "daz"]].std()

    # The mean std_centroid in arcsec
    stars[["dalt", "daz"]].std()
