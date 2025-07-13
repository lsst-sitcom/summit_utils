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

__all__ = ["GuiderStarTracker"]

from typing import Any
import numpy as np
import pandas as pd
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats

# from lsst.obs.lsst.cameraTransforms import LsstCameraTransforms
from lsst.obs.lsst import LsstCam
from lsst.summit.utils.guiders.transformation import convert_pixels_to_altaz, pixel_to_focal
from lsst.summit.utils.guiders.reading import GuiderReader
from lsst.summit.utils.guiders.reading import GuiderData

DEFAULT_COLUMNS = [
    "xroi",
    "yroi",
    "xccd",
    "yccd",
    "xccd_ref",
    "yccd_ref",
    "xfp",
    "yfp",
    "xfp_ref",
    "yfp_ref",
    "alt",
    "az",
    "alt_ref",
    "az_ref",
    "detector",
    "detid",
    "ampname",
    "expid",
    "starid",
    "stamp",
    "timestamp",
    "filter",
    "magoffset",
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


class GuiderStarTracker:
    """
    Class to track stars in the Guider data.

    Parameters
    ----------
    guider : GuiderData
        Guider dataclass instance containing guider data.
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
        guider: GuiderData,
        psf_fwhm: float = 6.0,
        min_snr: float = 3.0,
        min_stamp_detections: int = 30,
        edge_margin: int = 20,
        max_ellipticity: float = 0.2,
    ) -> None:
        self.guider = guider
        self.n_stamps = len(self.guider.timestamps)

        # detection and QC parameters
        self.psf_fwhm = psf_fwhm
        self.min_snr = min_snr
        self.min_stamp_detections = min_stamp_detections
        self.edge_margin = edge_margin
        self.max_ellipticity = max_ellipticity

        # initialize outputs
        self.stars = pd.DataFrame(columns=DEFAULT_COLUMNS)

    # STOPPED HERE: SAT, 12, JULY, 2025
    def track_guider_stars(self, ref_catalog: None | pd.DataFrame = None) -> pd.DataFrame:
        """
        Track stars across guider exposures using a reference catalog.

        Parameters
        ----------
        ref_catalog : pd.DataFrame
            Reference catalog with known star positions per detector.

        Returns
        -------
        stars : pd.DataFrame
            DataFrame with tracked stars and their properties,
            including positions, fluxes, and residual offsets.
        """
        if ref_catalog is None:
            ref_catalog = build_reference_catalog(
                self.guider,
                min_snr=self.min_snr,
                edge_margin=self.edge_margin,
                aperture_radius=self.psf_fwhm,
                max_ellipticity=self.max_ellipticity,
            )
        tracked_star_tables = []
        for guiderName in self.guider.getGuiderNames():
            ref = ref_catalog[ref_catalog["detector"] == guiderName].copy()
            if len(ref) > 1:
                raise ValueError(f"Multiple rows found for guider {guiderName} in the reference catalog.")
            stars = self.run_tracking_star(ref, guiderName)
            tracked_star_tables.append(stars)

        # Concatenate all stars into a single DataFrame
        if tracked_star_tables:
            tracked_star_catalog = pd.concat(tracked_star_tables, ignore_index=True)

        else:
            tracked_star_catalog = pd.DataFrame(columns=DEFAULT_COLUMNS)
            return tracked_star_catalog

        # Set unique IDs
        tracked_star_catalog = self.set_unique_id(tracked_star_catalog)

        # Compute offsets
        tracked_star_catalog = self.compute_offsets(tracked_star_catalog)
        return tracked_star_catalog

    def run_tracking_star(
        self,
        ref: pd.DataFrame,
        guiderName: str,
    ) -> pd.DataFrame:
        """
        Track one star across all stamps for one guider.

        The first two stamps are taken while the shutter is opening,
        so we skip them.

        Parameters
        ----------
        ref : pd.DataFrame
            One-row DataFrame with reference star info for this guider.
            Must contain columns 'xroi', 'yroi', 'starid'.
        guiderName : str
            Name of the guider (e.g., 'R22_S11').
        Returns
        -------
        pd.DataFrame or None
            DataFrame with one row per stamp where the star was detected,
            or None if the star was not detected in any stamp.
            Columns include:
              - starid, stamp, ampname, filter
              - xroi, yroi (centroid in roi coordinates)
              - xccd, yccd (centroid in CCD pixel coordinates)
              - xfp, yfp (centroid in focal plane coordinates)
              - alt, az (centroid in alt/az coordinates)
              - flux, flux_err, fwhm, snr
              - ixx, iyy, ixy, ixx_err, iyy_err, ixy_err
              - e1, e2 (ellipticity components)
            If the star was not detected in any stamp (or only one),
            returns None.

        """
        if ref.empty:
            return pd.DataFrame(columns=DEFAULT_COLUMNS)  # no reference star for this guider

        # Pull ref-catalog row
        ref_x, ref_y = ref["xroi"].iloc[0], ref["yroi"].iloc[0]
        starid = ref["starid"].iloc[0]
        amp_name = self.guider.getGuiderAmpName(guiderName)

        fwhm = self.psf_fwhm
        image_list = self.guider.datasets[guiderName]
        rows = []

        # --- per‐stamp measurements ---
        for i, stampObject in enumerate(image_list):
            si = stampObject.metadata.get("DAQSTAMP", i)
            stamp = stampObject.stamp_im.image.array

            # Skip first stamp (shutter opening)
            if si == 0:
                continue

            # make isr and cutout
            isr = stamp - np.nanmedian(stamp, axis=0)
            cutout = Cutout2D(isr, (ref_x, ref_y), size=50, mode="partial", fill_value=np.nan)

            _, median, std = sigma_clipped_stats(cutout.data, sigma=3.0)
            sources = measure_star_in_aperture(
                cutout.data - median, aperture_radius=fwhm, std_bkg=std, gain=1.0
            )

            if len(sources) == 0:
                # No sources detected in this stamp, skip it
                continue

            sources["starid"] = starid
            sources["stamp"] = si
            sources["ampname"] = amp_name
            sources["filter"] = self.guider.header["filter"]

            # Centroid in amplifier roi coordinates
            sources["xroi"] += cutout.xmin_original
            sources["yroi"] += cutout.ymin_original

            # Convert roi to ccd/focal-plane and alt/az coordinates
            xccd, yccd = self.convert_roi_to_ccd(sources, guiderName)
            xfp, yfp = self.convert_to_focal_plane(xccd, yccd, guiderName)
            alt, az = self.convert_to_altaz(xccd, yccd, self.guider.wcs)

            # Add reference positions
            sources["xccd"] = xccd
            sources["yccd"] = yccd
            sources["xfp"] = xfp
            sources["yfp"] = yfp
            sources["alt"] = alt
            sources["az"] = az
            sources["detector"] = guiderName
            rows.append(sources.iloc[0])

        df = pd.DataFrame(rows)
        if len(df) < 1:
            return pd.DataFrame(columns=DEFAULT_COLUMNS)  # only one detection, ignore it
        else:
            return df

    def convert_if_dvcs_to_ccd(
        self, x_dvcs: np.ndarray, y_dvcs: np.ndarray, guiderName: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Check if xccd/yccd CCD pixels are in DVCS coordinates system.
        """
        view = self.guider.view
        if view == "dvcs":
            # Convert xroi/yroi to CCD pixels
            stamps = self.guider.datasets[guiderName]

            # get CCD<->DVCS translation from the stamps
            _, _, dvcs = stamps.getArchiveElements()[0]

            x_ccd, y_ccd = dvcs(x_dvcs, y_dvcs)
            return x_ccd, y_ccd

        elif view == "ccd":
            # No conversion needed for CCD view
            return x_dvcs, y_dvcs

        else:
            raise ValueError(f"Unknown guider view '{view}'. Expected 'dvcs' or 'ccd'.")

    def set_unique_id(self, stars) -> pd.DataFrame:
        # 1) Build a detector→index map (0,1,2,…)
        det_map = self.guider.guiderNameMap

        # 2) Create a numeric “global” starid:
        #    global_id = det_index * 10000 + local starid
        stars["detid"] = stars["detector"].map(det_map)
        stars["trackid"] = stars["starid"] * 100 + stars["stamp"]
        stars["expid"] = self.guider.header["expid"]
        stars["filter"] = self.guider.header["filter"]
        return stars

    def compute_offsets(self, stars: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the offsets for each star in the catalog.
        """
        # make reference positions
        stars["xccd_ref"] = stars.groupby("starid")["xccd"].transform("median")
        stars["yccd_ref"] = stars.groupby("starid")["yccd"].transform("median")
        stars["xfp_ref"] = stars.groupby("starid")["xfp"].transform("median")
        stars["yfp_ref"] = stars.groupby("starid")["yfp"].transform("median")
        stars["alt_ref"] = stars.groupby("starid")["alt"].transform("median")
        stars["az_ref"] = stars.groupby("starid")["az"].transform("median")

        # Compute all your offsets
        stars["dx"] = stars["xccd"] - stars["xccd_ref"]
        stars["dy"] = stars["yccd"] - stars["yccd_ref"]
        stars["dxfp"] = stars["xfp"] - stars["xfp_ref"]
        stars["dyfp"] = stars["yfp"] - stars["yfp_ref"]
        stars["dalt"] = (stars["alt"] - stars["alt_ref"]) * 3600
        stars["daz"] = (stars["az"] - stars["az_ref"]) * 3600

        # Correct for cos(alt) in daz
        stars["daz"] = np.cos(stars["alt_ref"] * np.pi / 180) * stars["daz"]

        # compute mag offset
        stars["flux_ref"] = stars.groupby("starid")["flux"].transform("median")
        stars["magoffset"] = -2.5 * np.log10((stars["flux"] + 1e-12) / (stars["flux_ref"] + 1e-12))
        stars["magoffset"] = stars["magoffset"].replace([np.inf, -np.inf], np.nan)

        return stars

    def convert_to_focal_plane(
        self, xccd: np.ndarray, yccd: np.ndarray, detName: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert the star positions to focal plane coordinates.
        """
        if len(xccd) > 0:
            detNum = self.guider.getGuiderDetNum(detName)
            detector = LsstCam.getCamera()[detNum]
            # Convert the star positions to focal plane coordinates
            xfp, yfp = pixel_to_focal(xccd, yccd, detector)
        else:
            xfp, yfp = np.array([]), np.array([])
        return xfp, yfp

    def convert_to_altaz(self, xccd: np.ndarray, yccd: np.ndarray, wcs: Any) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert the star positions to altaz coordinates.

        Parameters
        ----------
        xccd : np.ndarray
            Array of x CCD pixel coordinates.
        yccd : np.ndarray
            Array of y CCD pixel coordinates.
        wcs : lsst.afw.image.Wcs
            WCS object for the guider detector.
        """
        nmid = self.n_stamps // 2
        obs_time = self.guider.timestamps[nmid]

        if len(xccd) > 0:
            alt, az = convert_pixels_to_altaz(wcs, obs_time, xccd, yccd)
        else:
            alt, az = np.array([]), np.array([])

        return alt, az

    # TODO: Double-check this conversion
    def convert_roi_to_ccd(self, df: pd.DataFrame, guiderName: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert roi coordinates to CCD pixel coordinates.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'xroi' and 'yroi' columns.
        guiderName : str
            Name of the guider (e.g., 'R22_S11').

        Returns
        -------
        xccd, yccd : np.ndarray
            Arrays of CCD pixel coordinates.
        """
        if df.empty:
            return np.array([]), np.array([])

        # min_x, min_y = self.guider.getGuiderAmpMinXY(guiderName)
        min_x, min_y = 0.0, 0.0
        xccd = df["xroi"].to_numpy() + min_x
        yccd = df["yroi"].to_numpy() + min_y

        # convert to ccd pixel if dvcs view is set
        xccd, yccd = self.convert_if_dvcs_to_ccd(xccd, yccd, guiderName)
        return xccd, yccd


def measure_star_in_aperture(
    cutout_data: np.ndarray,
    aperture_radius: float = 5,
    std_bkg: float = 1.0,
    gain: float = 1.0,
    mask: np.ndarray | None = None,
) -> pd.DataFrame:
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
                    "xroi": np.nan,
                    "yroi": np.nan,
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
                "xroi": xcen1,
                "yroi": ycen1,
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


def run_sextractor(
    img: np.ndarray,
    th: float = 10,
    median: float = 0,
    std: float | None = None,
    bkg_size: int = 50,
    aperture_radius: float = 5,
    max_ellipticity: float = 0.1,
    gain: float = 1.0,
) -> pd.DataFrame:
    """
    Vectorized SEP photometry with centroid errors, outputs a pandas DataFrame.
    Only returns nearly round, bright sources.
    """
    import sep

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
    objects = sep.extract(img_sub, th, err=std, mask=bad_mask)
    if len(objects) == 0:
        return pd.DataFrame()

    # Gather properties
    xcen, ycen = objects["x"], objects["y"]
    ixx, iyy, ixy = objects["x2"], objects["y2"], objects["xy"]
    ixx_err, iyy_err, ixy_err = objects["errx2"], objects["erry2"], objects["errxy"]

    flux, fluxerr, _ = sep.sum_circle(
        img_clean, xcen, ycen, aperture_radius, err=std, mask=bad_mask, gain=gain
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
            "xroi": xcen[mask],
            "yroi": ycen[mask],
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


def build_reference_catalog(
    guider: GuiderData,
    aperture_radius: float = 6.0,
    max_ellipticity: float = 0.2,
    min_snr: float = 3.0,
    edge_margin: int = 20,
) -> pd.DataFrame:
    """
    Build a reference catalog of stars from the first stamp of each guider.

    Parameters
    ----------
    guider : GuiderData
        Guider dataclass instance containing guider data.
    min_snr : float
        Minimum signal-to-noise ratio for stars to include.
    edge_margin : int
        Pixels from CCD edge to exclude sources.

    Returns
    -------
    pd.DataFrame
        Reference catalog with default columns
    """
    table_list = []
    for guiderName in guider.getGuiderNames():
        stamp = guider.getStackedStampArray(detName=guiderName, is_isr=True)
        _, median, std = sigma_clipped_stats(stamp, sigma=3.0)
        sources = run_sextractor(
            stamp - median,
            th=min_snr * std,
            median=median,
            std=std,
            aperture_radius=aperture_radius,
            max_ellipticity=max_ellipticity,
            gain=1.0,
        )
        if len(sources) == 0:
            continue

        # Filter out edge sources
        h, w = stamp.shape
        sources = sources[
            (sources["xroi"] > edge_margin)
            & (sources["xroi"] < w - edge_margin)
            & (sources["yroi"] > edge_margin)
            & (sources["yroi"] < h - edge_margin)
        ]
        if len(sources) == 0:
            continue

        # select only bright sources
        sources = sources[sources["snr"] >= min_snr]
        sources.sort_values(by="snr", ascending=False, inplace=True)
        sources.reset_index(drop=True, inplace=True)

        # pick the brightest source only
        bright = sources.iloc[[0]].copy()

        detNum = guider.getGuiderDetNum(guiderName)
        bright["detector"] = guiderName
        bright["detid"] = detNum
        bright["starid"] = detNum * 1000 + 1
        table_list.append(bright)

    if len(table_list) == 0:
        raise RuntimeError("No sources found in any guider for the reference catalog.")

    ref_catalog = pd.concat(table_list, ignore_index=True)
    return ref_catalog


if __name__ == "__main__":
    from lsst.daf.butler import Butler

    butler = Butler("embargo", collections=["LSSTCam/raw/guider", "LSSTCam/raw/all"])

    seqNum, dayObs = 461, 20250425
    reader = GuiderReader(butler, view="dvcs", verbose=True)
    guider = reader.get(dayObs=dayObs, seqNum=seqNum)

    star_tracker = GuiderStarTracker(guider, psf_fwhm=6.0)
    stars = star_tracker.track_guider_stars(ref_catalog=None)

    print(f"Tracked {len(stars)} stars in {len(stars['starid'].unique())} unique stars.")
    stars.to_csv(f"tracked_stars_{dayObs}_{seqNum:06d}.csv", index=False)
    print(f"Wrote tracked stars to tracked_stars_{dayObs}_{seqNum:06d}.csv")

    print(stars.head())
    print(stars.groupby("detector").size())
