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

import logging
from typing import TYPE_CHECKING

import galsim
import numpy as np
import pandas as pd
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats

from lsst.afw.image import ExposureF, ImageF, MaskedImageF
from lsst.summit.utils.guiders.reading import GuiderReader
from lsst.summit.utils.guiders.transformation import (
    convert_roi_to_ccd,
    convert_to_altaz,
    convert_to_focal_plane,
)
from lsst.summit.utils.utils import detectObjectsInExp

if TYPE_CHECKING:
    from lsst.summit.utils.guiders.reading import GuiderData

DEFAULT_COLUMNS = (
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
)


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
        guiderData: GuiderData,
        psf_fwhm: float = 6.0,
        min_snr: float = 3.0,
        min_stamp_detections: int = 30,
        edge_margin: int = 20,
        max_ellipticity: float = 0.2,
    ) -> None:
        self.log = logging.getLogger(__name__)
        self.guiderData = guiderData
        self.n_stamps = len(self.guiderData.timestamps)

        # detection and QC parameters
        self.psf_fwhm = psf_fwhm
        self.min_snr = min_snr
        self.min_stamp_detections = min_stamp_detections
        self.edge_margin = edge_margin
        self.max_ellipticity = max_ellipticity

        # initialize outputs
        self.stars = pd.DataFrame(columns=DEFAULT_COLUMNS)

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
            self.log.info("Using self-generated refcat")
            ref_catalog = build_reference_catalog(
                self.guiderData,
                min_snr=self.min_snr,
                edge_margin=self.edge_margin,
                aperture_radius=self.psf_fwhm,
                max_ellipticity=self.max_ellipticity,
            )
        tracked_star_tables = []
        for guiderName in self.guiderData.getGuiderNames():
            ref = ref_catalog[ref_catalog["detector"] == guiderName].copy()
            # take the first star only
            if len(ref) > 1:
                ref = ref.sort_values(by="snr", ascending=False).iloc[[0]].copy()

            stars = self.track_star_across_stamps(ref, guiderName)
            tracked_star_tables.append(stars)

        # Concatenate all stars into a single DataFrame
        if tracked_star_tables:
            tracked_star_catalog = pd.concat(tracked_star_tables, ignore_index=True)

        else:
            self.log.warning("No stars detected in any guider. Returning empty catalog.")
            tracked_star_catalog = pd.DataFrame(columns=DEFAULT_COLUMNS)
            return tracked_star_catalog

        # Set unique IDs
        tracked_star_catalog = self.set_unique_id(tracked_star_catalog)

        # Compute offsets
        tracked_star_catalog = self.compute_offsets(tracked_star_catalog)
        return tracked_star_catalog

    def track_star_across_stamps(
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
            # no reference star for this guider
            return pd.DataFrame(columns=DEFAULT_COLUMNS)

        # Pull ref-catalog row
        ref_x, ref_y = ref["xroi"].iloc[0], ref["yroi"].iloc[0]
        starid = ref["starid"].iloc[0]
        amp_name = self.guiderData.getGuiderAmpName(guiderName)

        fwhm = self.psf_fwhm
        image_list = self.guiderData.datasets[guiderName]
        rows = []

        # get some basic info
        detNum = self.guiderData.getGuiderDetNum(guiderName)
        nmid = len(self.guiderData.timestamps) // 2
        obstime = self.guiderData.timestamps[nmid]

        # --- per‐stamp measurements ---
        for i, stampObject in enumerate(image_list):
            if not (si := stampObject.metadata.get("DAQSTAMP")):
                self.log.warning(
                    (
                        f"Stamp {i} in {guiderName} has no DAQSTAMP metadata,",
                        " defaulting to using position in list instead.",
                    )
                )
                si = i

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
            sources["filter"] = self.guiderData.header["filter"]

            # Centroid in amplifier roi coordinates
            sources["xroi"] += cutout.xmin_original
            sources["yroi"] += cutout.ymin_original

            # Convert roi to ccd/focal-plane and alt/az coordinates
            xccd, yccd = convert_roi_to_ccd(
                sources["xroi"].values,
                sources["yroi"].values,
                self.guiderData,
                guiderName,
            )
            xfp, yfp = convert_to_focal_plane(xccd, yccd, detNum)
            alt, az = convert_to_altaz(xccd, yccd, self.guiderData.wcs, obstime)

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

    def set_unique_id(self, stars) -> pd.DataFrame:
        # 1) Build a detector→index map (0,1,2,…)
        det_map = self.guiderData.guiderNameMap

        # 2) Create a numeric “global” starid:
        #    global_id = det_index * 10000 + local starid
        stars["detid"] = stars["detector"].map(det_map)
        stars["trackid"] = stars["starid"] * 1000 + stars["stamp"]
        stars["expid"] = self.guiderData.header["expid"]
        stars["filter"] = self.guiderData.header["filter"]
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


def run_galsim_detection(
    image: np.ndarray,
    th: float = 10,
    bkg_std: float = 15,
    aperture_radius: float = 5,
    max_ellipticity: float = 0.1,
    gain: float = 1.0,
) -> pd.DataFrame:
    """
    Replaces SEP with LSST's detectObjectsInExp and GalSim moments.
    """
    # Step 1: Convert numpy image to MaskedImage and Exposure
    exposure = ExposureF(MaskedImageF(ImageF(image)))

    # Step 3: Detect sources
    footprints = detectObjectsInExp(exposure, th)

    if not footprints:
        return pd.DataFrame(
            columns=[
                "xroi",
                "yroi",
                "xerr",
                "yerr",
                "fwhm",
                "e1",
                "e2",
                "flux",
                "flux_err",
                "snr",
            ]
        )

    results = []

    for fp in footprints.getFootprints():
        # Create a cutout of the image around the footprint
        bbox = fp.getBBox()
        try:
            subimage = exposure.getMaskedImage().getImage()[bbox]
        except Exception:
            continue

        array = subimage.array
        if not np.isfinite(array).all():
            continue

        # GalSim adaptive moment
        try:
            gs_img = galsim.Image(array)
            hsm_res = galsim.hsm.FindAdaptiveMom(gs_img, strict=False)
            flag = hsm_res.error_message == ""
            xcentroid_gs = hsm_res.moments_centroid.x if flag else np.nan
            ycentroid_gs = hsm_res.moments_centroid.y if flag else np.nan
            flux_gs = hsm_res.moments_amp if flag else np.nan
            sigma_gs = hsm_res.moments_sigma if flag else np.nan
            fwhm_gs = 2.355 * sigma_gs if flag else np.nan
            n_pix = np.pi * fwhm_gs**2
            flux_err = np.sqrt(flux_gs / gain + n_pix * bkg_std**2)

        except (galsim.errors.GalSimError, RuntimeError):
            continue

        # Ellipticity filtering using axis ratio
        if np.hypot(hsm_res.observed_shape.e1, hsm_res.observed_shape.e2) > max_ellipticity:
            continue

        # Build row
        result = {
            "xroi": xcentroid_gs + bbox.getMinX(),
            "yroi": ycentroid_gs + bbox.getMinY(),
            "xerr": sigma_gs / np.sqrt(flux_gs) if flux_gs > 0 else np.nan,
            "yerr": sigma_gs / np.sqrt(flux_gs) if flux_gs > 0 else np.nan,
            "fwhm": fwhm_gs,
            "e1": hsm_res.observed_shape.e1,
            "e2": hsm_res.observed_shape.e2,
            "flux": flux_gs,
            "flux_err": flux_err,
            "snr": flux_gs / flux_err if flux_err > 0 else 0.0,
        }
        results.append(result)

    df = pd.DataFrame(results)
    if df.empty:
        return df
    else:
        df.sort_values("snr", ascending=False, inplace=True)
        dfout = df[(df["snr"] >= th) & (df["flux"] > 0) & (df["flux_err"] > 0)]
        return dfout.reset_index(drop=True)


# TODO: Replace SEP with galsim
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
        array = guider.getStackedStampArray(detName=guiderName, is_isr=True)
        _, median, std = sigma_clipped_stats(array, sigma=3.0)
        sources = run_galsim_detection(
            array, th=min_snr, max_ellipticity=max_ellipticity, bkg_std=std + median
        )

        if len(sources) == 0:
            continue

        # Filter out edge sources
        h, w = array.shape
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
        bright["starid"] = detNum * 100
        table_list.append(bright)

    if len(table_list) == 0:
        raise RuntimeError("No sources found in any guider for the reference catalog.")

    ref_catalog = pd.concat(table_list, ignore_index=True)
    return ref_catalog


if __name__ == "__main__":
    seqNum, dayObs = 461, 20250425
    reader = GuiderReader(view="dvcs", verbose=True)
    guider = reader.get(dayObs=dayObs, seqNum=seqNum)

    star_tracker = GuiderStarTracker(guider, psf_fwhm=6.0)

    # the ref catalog will be provided by the user, .e.g gaia
    # if not, the class will self-generate one is based on the stack
    # of the stamps of each guider
    stars = star_tracker.track_guider_stars(ref_catalog=None)

    print(stars.head())
    print(stars.groupby("detector").size())
