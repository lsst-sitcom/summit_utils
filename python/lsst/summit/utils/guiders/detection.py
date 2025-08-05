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
import warnings
from typing import TYPE_CHECKING

import galsim
import numpy as np
import pandas as pd
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from erfa import ErfaWarning

from lsst.afw.image import ExposureF, ImageF, MaskedImageF
from lsst.pex.exceptions import InvalidParameterError
from lsst.summit.utils.utils import detectObjectsInExp

from .reading import GuiderReader
from .transformation import convertRoiToCcd, convertToAltaz, convertToFocalPlane

if TYPE_CHECKING:
    from .reading import GuiderData

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
    "elapsed_time",
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
    guiderData : GuiderData
        Guider dataclass instance containing guider data.
    psfFwhm : float, default=6.0
        Expected PSF full-width at half-maximum in pixels.
    minSnr : float, default=3.0
        Minimum signal-to-noise ratio for stars in the reference catalog.
    minStampDetections : int, default=30
        Minimum number of stamps a star must be detected
        in to be considered valid.
    edgeMargin : int, default=20
        Pixels from CCD edge to exclude reference sources.
    maxEllipticity : float, default=0.2
        Maximum allowed ellipticity for sources.
    """

    def __init__(
        self,
        guiderData: GuiderData,
        psfFwhm: float = 6.0,
        minSnr: float = 3.0,
        minStampDetections: int = 30,
        edgeMargin: int = 20,
        maxEllipticity: float = 0.2,
    ) -> None:
        self.log = logging.getLogger(__name__)
        self.guiderData = guiderData
        self.nStamps = len(self.guiderData.timestamps)

        # detection and QC parameters
        self.psfFwhm = psfFwhm
        self.minSnr = minSnr
        self.minStampDetections = minStampDetections
        self.edgeMargin = edgeMargin
        self.maxEllipticity = maxEllipticity

        # initialize outputs
        self.stars = pd.DataFrame(columns=DEFAULT_COLUMNS)

    def trackGuiderStars(self, refCatalog: None | pd.DataFrame = None) -> pd.DataFrame:
        """
        Track stars across guider exposures using a reference catalog.

        Parameters
        ----------
        refCatalog : pd.DataFrame
            Reference catalog with known star positions per detector.

        Returns
        -------
        stars : pd.DataFrame
            DataFrame with tracked stars and their properties,
            including positions, fluxes, and residual offsets.
        """
        if refCatalog is None:
            self.log.info("Using self-generated refcat")
            refCatalog = build_reference_catalog(
                self.guiderData,
                min_snr=self.minSnr,
                edge_margin=self.edgeMargin,
                aperture_radius=self.psfFwhm,
                max_ellipticity=self.maxEllipticity,
            )

        if refCatalog.empty:
            self.log.warning("Reference catalog is empty. No stars to track.")
            return pd.DataFrame(columns=DEFAULT_COLUMNS)

        trackedStarTables = []
        for guiderName in self.guiderData.guiderNames:
            refStar = refCatalog[refCatalog["detector"] == guiderName].copy()
            # take the first star only
            if len(refStar) > 1:
                refStar = refStar.sort_values(by="snr", ascending=False).iloc[[0]].copy()

            stars = self.trackStarAcrossStamps(refStar, guiderName)
            trackedStarTables.append(stars)

        filteredTables = [df for df in trackedStarTables if not df.empty and not df.isna().all(axis=None)]
        # Concatenate all stars into a single DataFrame
        if filteredTables:
            trackedStarCatalog = pd.concat(filteredTables, ignore_index=True)
        else:
            self.log.warning("No stars detected in any guider. Returning empty catalog.")
            trackedStarCatalog = pd.DataFrame(columns=DEFAULT_COLUMNS)
            return trackedStarCatalog

        # Filter out stars with insufficient detections (xroi, yroi)
        trackedStarCatalog = trackedStarCatalog.groupby("starid").filter(
            lambda x: x["xroi"].notna().sum() >= self.minStampDetections
        )

        # Set unique IDs
        trackedStarCatalog = self.setUniqueId(trackedStarCatalog)

        # Compute offsets
        trackedStarCatalog = self.computeOffsets(trackedStarCatalog)
        return trackedStarCatalog

    def trackStarAcrossStamps(
        self,
        refStar: pd.DataFrame,
        guiderName: str,
        cutoutSize: int = 25,
    ) -> pd.DataFrame:
        """
        Track one star across all stamps for one guider.

        The first two stamps are taken while the shutter is opening,
        so we skip them.

        Parameters
        ----------
        refStar : pd.DataFrame
            One-row DataFrame with reference star info for this guider.
            Must contain columns 'xroi', 'yroi', 'starid'.
        guiderName : str
            Name of the guider (e.g., 'R22_S11').
        cutoutSize : int, optional
            Size of the cutout region in pixels. Default is 25.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per stamp where the star was detected,
            or empty DataFrame if the star was not detected in any stamp.
            Columns include:
              - starid, stamp, ampname, filter
              - xroi, yroi (centroid in roi coordinates)
              - xccd, yccd (centroid in CCD pixel coordinates)
              - xfp, yfp (centroid in focal plane coordinates)
              - alt, az (centroid in alt/az coordinates)
              - flux, flux_err, fwhm, snr
              - ixx, iyy, ixy, ixx_err, iyy_err, ixy_err
              - e1, e2 (ellipticity components)
        """
        if refStar.empty:
            # no reference star for this guider
            return pd.DataFrame(columns=DEFAULT_COLUMNS)

        # Pull ref-catalog row
        ref_x, ref_y = refStar["xroi"].iloc[0], refStar["yroi"].iloc[0]
        starid = refStar["starid"].iloc[0]
        ampName = self.guiderData.getGuiderAmpName(guiderName)

        imageList = self.guiderData.datasets[guiderName]
        rows = []

        # get some basic info
        detNum = self.guiderData.getGuiderDetNum(guiderName)
        obsTime = Time(self.guiderData.header["start_time"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ErfaWarning)
            elapsedTime = self.guiderData.timestamps - self.guiderData.timestamps[0]

        # --- per‐stamp measurements ---
        for i, stampObject in enumerate(imageList[1:]):
            if not (si := stampObject.metadata.get("DAQSTAMP")):
                self.log.warning(
                    (
                        f"Stamp {i} in {guiderName} has no DAQSTAMP metadata,",
                        " defaulting to using position in list instead.",
                    )
                )
                si = i

            stamp = stampObject.stamp_im.image.array

            if np.count_nonzero(stamp != 0) < 1:
                self.log.warning(f"Stamp {si} in {guiderName} is empty, skipping it.")
                continue

            # Skip first stamp (shutter opening)
            if si == 0:
                continue

            # make isr and cutout
            isr = stamp - np.nanmedian(stamp, axis=0)
            cutout = Cutout2D(isr, (ref_x, ref_y), size=cutoutSize, mode="partial", fill_value=np.nan)
            _, median, std = sigma_clipped_stats(cutout.data[: cutoutSize // 2, : cutoutSize // 2], sigma=3.0)
            sources_df = run_galsim(cutout.data - median, bkg_std=std, gain=1.0)
            if len(sources_df) == 0:
                # No sources detected in this stamp, skip it
                continue

            sources_df = pd.DataFrame(sources_df, index=np.arange(len(sources_df)))
            sources_df["starid"] = starid
            sources_df["stamp"] = si
            sources_df["ampname"] = ampName
            sources_df["filter"] = self.guiderData.header["filter"]
            sources_df["timestamp"] = self.guiderData.timestamps[si]
            sources_df["elapsed_time"] = elapsedTime[si].sec

            # Centroid in amplifier roi coordinates
            sources_df["xroi"] += cutout.xmin_original
            sources_df["yroi"] += cutout.ymin_original

            # Convert roi to ccd/focal-plane and alt/az coordinates
            xccd, yccd = convertRoiToCcd(
                sources_df["xroi"],
                sources_df["yroi"],
                self.guiderData,
                guiderName,
            )
            wcs = self.guiderData.wcs[guiderName]
            xfp, yfp = convertToFocalPlane(xccd, yccd, detNum)
            alt, az = convertToAltaz(xccd, yccd, wcs, obsTime)

            # Convert fwhm to arcseconds
            pixel_scale = wcs.getPixelScale().asArcseconds()
            sources_df["fwhm"] *= pixel_scale

            # Rotate e1, e2 to alt/az coordinates
            camera_angle = float(self.guiderData.header["CAM_ROT_ANGLE"])
            e1_altaz, e2_altaz = rotate_ellipticity(sources_df["e1"], sources_df["e2"], 90 + camera_angle)

            # Add reference positions
            sources_df["xccd"] = xccd
            sources_df["yccd"] = yccd
            sources_df["xfp"] = xfp
            sources_df["yfp"] = yfp
            sources_df["alt"] = alt
            sources_df["az"] = az
            sources_df["e1_altaz"] = e1_altaz
            sources_df["e2_altaz"] = e2_altaz
            sources_df["detector"] = guiderName
            rows.append(sources_df.iloc[0])

        df = pd.DataFrame(rows)
        if len(df) < 1:
            return pd.DataFrame(columns=DEFAULT_COLUMNS)  # only one detection, ignore it
        else:
            return df

    def setUniqueId(self, stars: pd.DataFrame) -> pd.DataFrame:
        """
        Assign unique IDs to tracked stars.

        Parameters
        ----------
        stars : pd.DataFrame
            DataFrame with detected stars.

        Returns
        -------
        pd.DataFrame
            DataFrame with additional unique ID columns.
        """
        # 1) Build a detector→index map (0,1,2,…)
        detMap = self.guiderData.guiderNameMap

        # 2) Create a numeric “global” starid:
        #    global_id = det_index * 10000 + local starid
        stars["detid"] = stars["detector"].map(detMap)
        stars["trackid"] = stars["starid"] * 1000 + stars["stamp"]
        stars["expid"] = self.guiderData.header["expid"]
        stars["filter"] = self.guiderData.header["filter"]
        return stars

    def computeOffsets(self, stars: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the offsets for each star in the catalog.

        Parameters
        ----------
        stars : pd.DataFrame
            DataFrame with detected stars.

        Returns
        -------
        pd.DataFrame
            DataFrame with additional offset columns.
        """
        # make reference positions
        stars["xroi_ref"] = stars.groupby("starid")["xroi"].transform("median")
        stars["yroi_ref"] = stars.groupby("starid")["yroi"].transform("median")
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
        stars["flux_ref"] = pd.to_numeric(stars["flux_ref"], errors="coerce")
        stars["flux"] = pd.to_numeric(stars["flux"], errors="coerce")
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
    cutout_data : np.ndarray
        Background-subtracted cutout image.
    aperture_radius : float
        Radius of aperture in pixels.
    std_bkg : float
        Background RMS per pixel.
    gain : float
        e-/ADU gain.
    mask : np.ndarray or None
        True = pixel to ignore (e.g. bad column, cosmic ray, star mask). Must
        be same shape as cutout_data. If None, no extra masking.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row containing centroid, moments, ellipticity,
        flux, flux error, FWHM, and SNR.
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


def run_source_detection(
    image: np.ndarray,
    th: float = 10,
    bkg_std: float = 15,
    max_ellipticity: float = 0.1,
    gain: float = 1.0,
) -> pd.DataFrame:
    """
    Detect sources in an image using LSST's detectObjectsInExp and measure
    their roperties with GalSim adaptive moments.

    Parameters
    ----------
    image : np.ndarray
        2D image array.
    th : float
        Detection threshold.
    bkg_std : float
        Background RMS per pixel.
    max_ellipticity : float
        Maximum allowed ellipticity for sources.
    gain : float
        Detector gain (e-/ADU).

    Returns
    -------
    pd.DataFrame
        DataFrame with detected source properties.
    """
    # Step 1: Convert numpy image to MaskedImage and Exposure
    exposure = ExposureF(MaskedImageF(ImageF(image)))

    try:
        # Step 3: Detect sources
        footprints = detectObjectsInExp(exposure, th)
    except InvalidParameterError:
        logging.warning(
            "InvalidParameterError: Standard deviation must be > 0. Possibly empty or uniform image."
        )
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    if not footprints:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    results = []

    for fp in footprints.getFootprints():
        # Create a cutout of the image around the footprint
        bbox = fp.getBBox()
        subimage = exposure.getMaskedImage().getImage()[bbox]

        array = subimage.array
        if not np.isfinite(array).all():
            continue

        result = run_galsim(array, gain=gain, bkg_std=bkg_std)
        result["xroi"] += bbox.getMinX()
        result["yroi"] += bbox.getMinY()

        results.append(result)
    df = pd.DataFrame(results)
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
    aperture_radius : float
        Radius of aperture in pixels.
    max_ellipticity : float
        Maximum allowed ellipticity for sources.
    min_snr : float
        Minimum signal-to-noise ratio for stars to include.
    edge_margin : int
        Pixels from CCD edge to exclude sources.

    Returns
    -------
    pd.DataFrame
        Reference catalog with default columns.
    """
    table_list = []
    for guiderName in guider.guiderNames:
        array = guider.getStackedStampArray(detName=guiderName, isIsr=True)
        _, median, std = sigma_clipped_stats(array, sigma=3.0)
        sources = run_source_detection(
            array, th=min_snr, max_ellipticity=max_ellipticity, bkg_std=std + median
        )
        if sources.empty:
            # logging.warning(f"Guider {guiderName} has no sources detected.")
            continue

        # Filter out edge sources
        h, w = array.shape
        mask = (
            (sources["xroi"] > edge_margin)
            & (sources["xroi"] < w - edge_margin)
            & (sources["yroi"] > edge_margin)
            & (sources["yroi"] < h - edge_margin)
            & (sources["snr"] >= min_snr)
            & (np.sqrt(sources["e1"] ** 2 + sources["e2"] ** 2) <= max_ellipticity)
            & (sources["flux"] > 0)  # Ensure positive flux
            & (sources["flux_err"] > 0)  # Ensure positive flux error
            & (sources["fwhm"] > 1)  # Ensure positive FWHM at least 1 pixel
        )
        sources = sources[mask]

        if len(sources) == 0:
            # logging.warning(f"Guider {guiderName}
            # has no sources after SNR/ellip cut.")
            continue

        sources["fwhm_inv"] = 1 / sources["fwhm"]
        sources.sort_values(by=["snr", "fwhm_inv"], ascending=False, inplace=True)
        sources.reset_index(drop=True, inplace=True)

        # pick the brightest source only
        bright = sources.iloc[[0]].copy()

        detNum = guider.getGuiderDetNum(guiderName)
        bright["detector"] = guiderName
        bright["detid"] = detNum
        bright["starid"] = detNum * 100
        table_list.append(bright)

    if len(table_list) == 0:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)

    ref_catalog = pd.concat(table_list, ignore_index=True)
    return ref_catalog


def run_galsim(
    array: np.ndarray,
    gain: float = 1.0,
    bkg_std: float = 0.0,
) -> pd.DataFrame:
    """
    Measure star properties using GalSim adaptive moments.

    Parameters
    ----------
    array : np.ndarray
        2D image array.
    gain : float
        Detector gain (e-/ADU).
    bkg_std : float
        Background RMS per pixel.

    Returns
    -------
    pd.DataFrame
        DataFrame with measured properties (one row).
    """
    gs_img = galsim.Image(array)
    hsm_res = galsim.hsm.FindAdaptiveMom(gs_img, strict=False)
    success = hsm_res.error_message == ""

    if success:
        xcentroid = hsm_res.moments_centroid.x
        ycentroid = hsm_res.moments_centroid.y
        flux = hsm_res.moments_amp
        sigma = hsm_res.moments_sigma
        e1 = hsm_res.observed_shape.e1
        e2 = hsm_res.observed_shape.e2
        g1 = hsm_res.observed_shape.g1
        g2 = hsm_res.observed_shape.g2
        fwhm = 2.355 * sigma

        # Calculate errors using GalSim's error estimation
        gs_dict = galsim_error(array, hsm_res, gain=gain, bkg_std=bkg_std, is_gain=True)
        xerr = gs_dict["M10"]
        yerr = gs_dict["M01"]

        # Calculate SNR and flux error
        e = np.sqrt(e1**2 + e2**2)
        n_eff = 2 * np.pi * sigma**2 * np.sqrt(1 - e**2)
        shot_noise = np.sqrt(n_eff * bkg_std**2)
        flux_err = np.sqrt(flux / gain + shot_noise**2)
        snr = flux / shot_noise if flux > 0 else 0.0

        # Calculate second moments
        ixx = sigma**2 * (1 + g1)
        iyy = sigma**2 * (1 - g1)
        ixy = sigma**2 * g2
    else:
        xcentroid = ycentroid = sigma = flux = fwhm = np.nan
        e1 = e2 = g1 = g2 = ixx = iyy = ixy = np.nan
        flux_err = snr = xerr = yerr = 0.0

    result = {
        "xroi": xcentroid,
        "yroi": ycentroid,
        "xerr": xerr,
        "yerr": yerr,
        "e1": e1,
        "e2": e2,
        "ixx": ixx,
        "iyy": iyy,
        "ixy": ixy,
        "fwhm": fwhm,
        "flux": flux,
        "flux_err": flux_err,
        "snr": snr,
    }
    return result


def galsim_error(
    array: np.ndarray, gs: galsim.hsm, gain: float = 1.0, bkg_std: float = 0.0, is_gain: bool = False
) -> dict:
    """
    Estimate variance of second moments from a GalSim HSMShapeData object.

    Code based on:
    https://github.com/rmjarvis/Piff/blob/60286d107438dbc0d21e2fe215f0b63d0c107e6a/piff/util.py#L296

    Parameters
    ----------
    array : np.ndarray
        2D image data used for moment measurement.
    gs : galsim.hsm.HSMShapeData
        Result object from galsim.hsm.FindAdaptiveMom.
    gain : float
        Detector gain (e-/ADU).
    bkg_std : float
        Background RMS per pixel.
    is_gain : bool
        If True, include gain in variance weighting.

    Returns
    -------
    dict
        Dictionary with error estimates for M00, M10, M01, M11, M20, M02.
    """
    if not gs or gs.error_message != "":
        return {}

    # define galsim params
    x0 = gs.moments_centroid.x
    y0 = gs.moments_centroid.y
    sigma = gs.moments_sigma
    e1 = gs.observed_shape.e1
    e2 = gs.observed_shape.e2
    flux = gs.moments_amp

    # make galsim gaussian kernel
    kernel = make_elliptical_gaussian_star(
        shape=(array.shape[0], array.shape[1]), e1=e1, e2=e2, flux=1, sigma=sigma, center=(x0, y0)
    )

    # make weight = 1/variance
    weight = np.ones_like(array) / (bkg_std**2 + 1e-9)
    if is_gain:
        weight = np.ones_like(array) / (bkg_std**2 + np.abs(flux * kernel / gain))

    mask = weight == 0.0
    data = array.copy()
    if np.any(mask):
        kernel_masked = kernel.copy()
        data[mask] = kernel_masked[mask] * np.sum(data[~mask]) / np.sum(kernel_masked[~mask])

    # start the variance estimate
    u, v = np.meshgrid(np.arange(array.shape[1]) - x0, np.arange(array.shape[0]) - y0)
    usq = u**2
    vsq = v**2
    uv = u * v
    rsq = usq + vsq
    usqmvsq = usq - vsq

    # Notation:
    #   W = kernel
    #   I = data
    #   V = var(data) -- used below.
    WI = kernel * data
    M00 = np.nansum(WI)

    # WV = W^2 1/w
    WV = (kernel**2).astype(float)
    WV[~mask] /= weight[~mask]  # Only use 1/w where w != 0
    WV[mask] /= np.median(weight[~mask])
    WV = WV / float(M00**2)

    rsq2 = rsq * rsq
    WIrsq = WI * rsq
    WIuv = WI * uv
    M11 = np.sum(WIrsq)
    M22 = np.sum(WI * rsq2)
    M20 = np.sum(WI * usqmvsq)
    M02 = 2 * np.sum(WIuv)

    A = 1 / (3 - M22 / M11**2)
    B = 2 / (4 - M22 / M11**2)
    dM00 = 1 - A * (rsq / M11 - 1)  # We'll need this combination a lot below, so save it.

    varM00 = np.sum(WV * dM00**2)
    varM10 = 4 * np.sum(WV * usq)
    varM01 = 4 * np.sum(WV * vsq)
    varM11 = 4 * A**2 * np.sum(WV * (rsq - M11) ** 2)
    varM20 = 4 * np.sum(WV * (B * usqmvsq + A * M20 * (rsq / M11 - 1)) ** 2)
    varM02 = 4 * np.sum(WV * (2 * B * uv + A * M02 * (rsq / M11 - 1)) ** 2)

    err = np.sqrt(np.array([varM00, varM10, varM01, varM11, varM20, varM02]))
    return dict(M00=err[0], M10=err[1], M01=err[2], M11=err[3], M20=err[4], M02=err[5])


def make_elliptical_gaussian_star(
    shape: tuple[int, int],
    flux: float,
    sigma: float,
    e1: float,
    e2: float,
    center: tuple[float, float],
) -> np.ndarray:
    """
    Create an elliptical 2D Gaussian star with specified parameters.

    Parameters
    ----------
    shape : tuple[int, int]
        (ny, nx) shape of the output image.
    flux : float
        Total flux of the star.
    sigma : float
        Base Gaussian size.
    e1 : float
        Ellipticity component e1.
    e2 : float
        Ellipticity component e2.
    center : tuple[float, float]
        (x0, y0) center of the Gaussian.

    Returns
    -------
    np.ndarray
        Image array with the elliptical Gaussian.
    """
    y, x = np.indices(shape)
    x0, y0 = center
    u = x - x0
    v = y - y0

    # Second-moment matrix elements
    Ixx = sigma**2 * (1 + e1)
    Iyy = sigma**2 * (1 - e1)
    Ixy = sigma**2 * e2

    # Inverse covariance matrix
    det = Ixx * Iyy - Ixy**2
    invIxx = Iyy / det
    invIyy = Ixx / det
    invIxy = -Ixy / det

    # Quadratic form: u^2 * invIxx + v^2 * invIyy + 2uv * invIxy
    r2 = invIxx * u**2 + invIyy * v**2 + 2 * invIxy * u * v

    e = np.sqrt(e1**2 + e2**2)
    norm = flux / (2 * np.pi * sigma**2 * np.sqrt(1 - e**2))
    image = norm * np.exp(-0.5 * r2)
    return image


def rotate_ellipticity(e1, e2, theta_deg):
    """
    Rotate ellipticity components (e1, e2) by theta_deg degrees.

    Parameters
    ----------
    e1 : float or array-like
        Ellipticity component e1.
    e2 : float or array-like
        Ellipticity component e2.
    theta_deg : float
        Rotation angle in degrees.

    Returns
    -------
    tuple
        (e1_rot, e2_rot) rotated ellipticity components.
    """
    theta = np.deg2rad(theta_deg)
    cos2t = np.cos(2 * theta)
    sin2t = np.sin(2 * theta)
    e1_rot = e1 * cos2t + e2 * sin2t
    e2_rot = -e1 * sin2t + e2 * cos2t
    return e1_rot, e2_rot


if __name__ == "__main__":
    seqNum, dayObs = 461, 20250425
    reader = GuiderReader(view="dvcs", verbose=True)
    guider = reader.get(dayObs=dayObs, seqNum=seqNum)

    starTracker = GuiderStarTracker(guider, psfFwhm=6.0)

    # the ref catalog will be provided by the user, .e.g gaia
    # if not, the class will self-generate one is based on the stack
    # of the stamps of each guider
    stars = starTracker.trackGuiderStars(refCatalog=None)

    print(stars.head())
    print(stars.groupby("detector").size())
