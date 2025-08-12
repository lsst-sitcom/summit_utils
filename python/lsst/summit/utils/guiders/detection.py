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

import logging
from dataclasses import asdict, dataclass, field

__all__ = [
    "runSourceDetection",
    "buildReferenceCatalog",
    "trackStarAcrossStamp",
    "makeBlankCatalog",
    "runGalSim",
]

import galsim
import numpy as np
import pandas as pd
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats

from lsst.afw.image import ExposureF, ImageF, MaskedImageF
from lsst.summit.utils.utils import detectObjectsInExp

from .reading import GuiderData


def trackStarAcrossStamp(
    refCenter: tuple[float, float],
    guiderData: GuiderData,
    guiderName: str,
    gain: int = 1,
    cutoutSize: int = 30,
) -> pd.DataFrame:
    """
    Track a star across all guider stamps, returning a DataFrame with
    per-stamp centroids, shapes, fluxes, and residuals.
    """
    gd = guiderData
    expid = gd.expid

    # loop over stamps
    results = []
    for i in range(len(gd)):
        cutout = getCutouts(gd[guiderName, i], refCenter, cutoutSize)
        array = cutout.data

        if np.count_nonzero(array == 0) < 1:
            # Blank Stamp, skip this stamp
            continue

        # Calculate read noise and background std
        _, median, bkgStd = sigma_clipped_stats(array.flatten(), sigma=3.0)

        # 2)  Track the star across all stamps for this guider
        star = runGalSim(array - median, gain=gain, bkgStd=bkgStd)
        star.xroi += cutout.xmin_original
        star.yroi += cutout.ymin_original
        sf = star.toDataFrame()
        sf["stamp"] = i  # Add stamp index
        results.append(sf)

    # 3)  Concatenate & quality-cut
    stars = pd.concat(results, ignore_index=True)
    if stars.empty:
        return makeBlankCatalog()

    # 4)  Add metadata
    stars["detector"] = guiderName
    stars["expid"] = expid
    stars["ampname"] = gd.getGuiderAmpName(guiderName)
    stars["detid"] = gd.getGuiderDetNum(guiderName)
    stars["filter"] = gd.header.get("filter", "UNKNOWN")
    return stars


# ----------------------------------------------------------------------

# === Make Blank Catalog ===
_DEFAULT_COLUMNS: str = (
    "trackid detector expid timestamp dalt daz dtheta dx dy "
    "xroi yroi xccd yccd xroi_ref yroi_ref xccd_ref yccd_ref "
    "dxfp dyfp xfp yfp alt az xfp_ref yfp_ref alt_ref az_ref "
    "theta theta_ref flux flux_err magoffset snr "
    "ixx iyy ixy fwhm e1 e2 e1_altaz e2_altaz "
    "ampname elapsed_time stamp detid filter "
)
DEFAULT_COLUMNS: tuple[str, ...] = tuple(_DEFAULT_COLUMNS.split())


def makeBlankCatalog() -> pd.DataFrame:
    """
    Create a blank DataFrame with the default columns for a star catalog.
    """
    return pd.DataFrame(columns=DEFAULT_COLUMNS)


# ----------------------------------------------------------------------
# === Core GalSim Detection ===


@dataclass
class StarMeasurement:
    xroi: float = field(default=np.nan)
    yroi: float = field(default=np.nan)
    xerr: float = field(default=0.0)
    yerr: float = field(default=0.0)
    e1: float = field(default=np.nan)
    e2: float = field(default=np.nan)
    ixx: float = field(default=np.nan)
    iyy: float = field(default=np.nan)
    ixy: float = field(default=np.nan)
    fwhm: float = field(default=np.nan)
    flux: float = field(default=np.nan)
    flux_err: float = field(default=0.0)
    snr: float = field(default=0.0)

    def toDataFrame(self) -> pd.DataFrame:
        """
        Return a single-row DataFrame for this measurement.
        NaN/Inf values are kept as-is (pandas handles them well).
        """
        data = asdict(self)
        return pd.DataFrame([data])


def runSourceDetection(
    image: np.ndarray,
    threshold: float = 10,
    bkgStd: float = 15,
    gain: float = 1.0,
) -> pd.DataFrame:
    """
    Detect sources in an image and measure their properties.

    Parameters
    ----------
    image : np.ndarray
        2D image array.
    threshold : float
        Detection threshold.
    bkgStd : float
        Background RMS per pixel.
    gain : float
        Detector gain (e-/ADU).

    Returns
    -------
    pd.DataFrame
        DataFrame with detected source properties.
    """
    # Step 1: Convert numpy image to MaskedImage and Exposure
    exposure = ExposureF(MaskedImageF(ImageF(image)))

    # Step 2: Detect sources
    footprints = detectObjectsInExp(exposure, nSigma=threshold)

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

        star = runGalSim(array, gain=gain, bkgStd=bkgStd)
        star.xroi += bbox.getMinX()
        star.yroi += bbox.getMinY()

        results.append(star)
    df = pd.concat([star.toDataFrame() for star in results], ignore_index=True)
    return df


def runGalSim(
    imageArray: np.ndarray,
    gain: float = 1.0,
    bkgStd: float = 0.0,
) -> StarMeasurement:
    """
    Measure star properties with GalSim adaptive moments.

    Returns a DataFrame with centroid, moments, shape, flux, FWHM, and SNR.
    """
    gsImg = galsim.Image(imageArray)
    hsmRes = galsim.hsm.FindAdaptiveMom(gsImg, strict=False)
    success = hsmRes.error_message == ""

    if not success:
        result = StarMeasurement()
    else:
        xCentroid = hsmRes.moments_centroid.x
        yCentroid = hsmRes.moments_centroid.y
        flux = hsmRes.moments_amp
        sigma = hsmRes.moments_sigma
        e1 = hsmRes.observed_shape.e1
        e2 = hsmRes.observed_shape.e2
        fwhm = 2.355 * sigma

        # Calculate errors using GalSim's error estimation
        xErr, yErr = galSimError(imageArray, hsmRes, gain=gain, bkgStd=bkgStd, isGain=True)

        # Calculate SNR and flux error
        ellipticity = np.sqrt(e1**2 + e2**2)
        nEff = 2 * np.pi * sigma**2 * np.sqrt(1 - ellipticity**2)
        shotNoise = np.sqrt(nEff * bkgStd**2)
        fluxErr = np.sqrt(flux / gain + shotNoise**2)
        snr = flux / shotNoise if flux > 0 else 0.0

        # Calculate second moments
        ixx = sigma**2 * (1 + e1)
        iyy = sigma**2 * (1 - e1)
        ixy = sigma**2 * e2

        result = StarMeasurement(
            xroi=xCentroid,
            yroi=yCentroid,
            xerr=xErr,
            yerr=yErr,
            e1=e1,
            e2=e2,
            ixx=ixx,
            iyy=iyy,
            ixy=ixy,
            fwhm=fwhm,
            flux=flux,
            flux_err=fluxErr,
            snr=snr,
        )
    return result


def galSimError(
    imageArray: np.ndarray,
    gs: galsim.hsm,
    gain: float = 1.0,
    bkgStd: float = 0.0,
    isGain: bool = False,
) -> tuple[float, float]:
    """
    Estimate centroid errors from GalSim HSMShapeData.
    """
    if not gs or gs.error_message != "":
        return 0.0, 0.0

    x0 = gs.moments_centroid.x
    y0 = gs.moments_centroid.y
    sigma = gs.moments_sigma
    e1 = gs.observed_shape.e1
    e2 = gs.observed_shape.e2
    flux = gs.moments_amp

    kernel = makeEllipticalGaussianStar(
        shape=(imageArray.shape[0], imageArray.shape[1]),
        e1=e1,
        e2=e2,
        flux=1,
        sigma=sigma,
        center=(x0, y0),
    )

    weight = np.ones_like(imageArray) / (bkgStd**2 + 1e-9)
    if isGain:
        weight = np.ones_like(imageArray) / (bkgStd**2 + np.abs(flux * kernel / gain))

    mask = weight == 0.0
    data = imageArray.copy()
    if np.any(mask):
        kernelMasked = kernel.copy()
        data[mask] = kernelMasked[mask] * np.sum(data[~mask]) / np.sum(kernelMasked[~mask])

    u, v = np.meshgrid(np.arange(imageArray.shape[1]) - x0, np.arange(imageArray.shape[0]) - y0)
    usq = u**2
    vsq = v**2
    WI = kernel * data
    M00 = np.nansum(WI)
    WV = (kernel**2).astype(float)
    WV[~mask] /= weight[~mask]
    WV[mask] /= np.median(weight[~mask])
    WV = WV / float(M00**2)

    varM10 = 4 * np.sum(WV * usq)
    varM01 = 4 * np.sum(WV * vsq)
    xerr = np.sqrt(varM10)
    yerr = np.sqrt(varM01)
    return xerr, yerr


def makeEllipticalGaussianStar(
    shape: tuple[int, int],
    flux: float,
    sigma: float,
    e1: float,
    e2: float,
    center: tuple[float, float],
) -> np.ndarray:
    """
    Create an elliptical 2D Gaussian star with specified parameters.
    """
    y, x = np.indices(shape)
    x0, y0 = center
    u = x - x0
    v = y - y0

    # Second-moment matrix elements
    ixx = sigma**2 * (1 + e1)
    iyy = sigma**2 * (1 - e1)
    ixy = sigma**2 * e2

    # Inverse covariance matrix
    det = ixx * iyy - ixy**2
    invIxx = iyy / det
    invIyy = ixx / det
    invIxy = -ixy / det

    # Quadratic form: u^2 * invIxx + v^2 * invIyy + 2uv * invIxy
    r2 = invIxx * u**2 + invIyy * v**2 + 2 * invIxy * u * v

    e = np.sqrt(e1**2 + e2**2)
    norm = flux / (2 * np.pi * sigma**2 * np.sqrt(1 - e**2))
    image = norm * np.exp(-0.5 * r2)
    return image


# === Reference Catalog Construction ===
def buildReferenceCatalog(
    guiderData: "GuiderData",
    log: logging.Logger,
    maxEllipticity: float = 0.2,
    minSnr: float = 3.0,
    edgeMargin: int = 20,
) -> pd.DataFrame:
    """
    Build a reference star catalog from each guider's coadded stamp.
    Returns a DataFrame with the brightest star per guider.
    """
    expId = guiderData.expid
    tableList = []
    for guiderName in guiderData.guiderNames:
        array = guiderData.getStampArrayCoadd(guiderName)
        array = np.where(array < 0, 0, array)  # Ensure no negative values
        _, median, std = sigma_clipped_stats(array, sigma=3.0)
        sources = runSourceDetection(
            array,
            threshold=minSnr,
            bkgStd=std + median,
            gain=1.0,
        )
        if sources.empty:
            log.warning(f"No sources detected in `buildReferenceCatalog`" f"for {guiderName} in {expId}. ")
            continue

        # Filter out edge sources
        h, w = array.shape
        mask = (
            (sources["xroi"] > edgeMargin)
            & (sources["xroi"] < w - edgeMargin)
            & (sources["yroi"] > edgeMargin)
            & (sources["yroi"] < h - edgeMargin)
            & (sources["snr"] >= minSnr)
            & (np.sqrt(sources["e1"] ** 2 + sources["e2"] ** 2) <= maxEllipticity)
            & (sources["flux"] > 0)  # Ensure positive flux
            & (sources["fwhm"] > 1)  # Ensure positive FWHM at least 1 pixel
        )
        # Ensure we have sources to process
        if np.count_nonzero(mask) == 0:
            log.warning(
                f"No sources after filering in `buildReferenceCatalog`" f"for {guiderName} in {expId}."
            )
            continue

        sources = sources[mask]
        sources["fwhm_inv"] = 1 / sources["fwhm"]
        sources.sort_values(by=["snr", "fwhm_inv"], ascending=False, inplace=True)
        sources.reset_index(drop=True, inplace=True)

        # pick the brightest source only
        bright = sources.iloc[[0]].copy()

        detNum = guiderData.getGuiderDetNum(guiderName)
        bright["detector"] = guiderName
        bright["detid"] = detNum
        bright["starid"] = detNum * 100
        tableList.append(bright)

    if len(tableList) == 0:
        log.warning(f"`buildReferenceCatalog` failed" f" - no stars detected in any guider for {expId}.")
        return makeBlankCatalog()

    refCatalog = pd.concat(tableList, ignore_index=True)
    return refCatalog


def getCutouts(imageArray: np.ndarray, refCenter: tuple[float, float], cutoutSize: int = 25) -> Cutout2D:
    """
    Get a cutout at the reference position from an image array.
    """
    refX, refY = refCenter
    return Cutout2D(imageArray, (refX, refY), size=cutoutSize, mode="partial", fill_value=np.nan)
