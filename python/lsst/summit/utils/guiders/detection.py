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

_DEFAULT_COLUMNS: str = (
    "trackid detector expid elapsed_time dalt daz dtheta dx dy "
    "fwhm xroi yroi xccd yccd xroi_ref yroi_ref xccd_ref yccd_ref "
    "dxfp dyfp xfp yfp alt az xfp_ref yfp_ref alt_ref az_ref "
    "xerr yerr theta theta_err theta_ref flux flux_err magoffset snr "
    "ixx iyy ixy e1 e2 e1_altaz e2_altaz "
    "ampname timestamp stamp detid filter "
)
DEFAULT_COLUMNS: tuple[str, ...] = tuple(_DEFAULT_COLUMNS.split())


def makeBlankCatalog() -> pd.DataFrame:
    """
    Create a blank DataFrame with the default columns for a star catalog.

    Returns
    -------
    catalog : `pd.DataFrame`
        Empty catalog with the default schema.
    """
    return pd.DataFrame(columns=DEFAULT_COLUMNS)


@dataclass(frozen=True, slots=True)
class GuiderStarTrackerConfig:
    """Configuration for the GuiderStarTracker.

    Parameters
    ----------
    minSnr : `float`
        Minimum signal-to-noise ratio for star detection.
    minStampDetections : `int`
        Minimum number of detections across all stamps for a star to be
        considered valid.
    edgeMargin : `int`
        Margin in pixels to avoid edge effects in the image.
    maxEllipticity : `float`
        Maximum allowed ellipticity for a star to be considered valid.
    cutOutSize : `int`
        Size of the cutout around the star for tracking.
    aperSizeArcsec : `float`
        Aperture size in arcseconds for star detection.
    gain : `float`
        Gain factor for the guider data, used in flux calculations.
    """

    minSnr: float = 10.0
    minStampDetections: int = 30
    edgeMargin: int = 25
    maxEllipticity: float = 0.3
    cutOutSize: int = 50
    aperSizeArcsec: float = 3.0
    gain: float = 1.0


def trackStarAcrossStamp(
    refCenter: tuple[float, float],
    guiderData: GuiderData,
    guiderName: str,
    config: GuiderStarTrackerConfig = GuiderStarTrackerConfig(),
) -> pd.DataFrame:
    """
    Track a star across all guider stamps and compute centroid, shape, and
    flux.

    GalSim is used for centroid and shape measurements. Flux is measured with
    aperture photometry.

    Parameters
    ----------
    refCenter : `tuple[float, float]`
        Reference position (x, y) in pixel coordinates for the star.
    guiderData : `GuiderData`
        Guider data containing image stamps and metadata.
    guiderName : `str`
        Name of the guider to process.
    config : `GuiderStarTrackerConfig`
        Configuration parameters for the star tracker.

    Returns
    -------
    stars : `pd.DataFrame`
        DataFrame containing the tracked star measurements across all stamps.
    """
    gd = guiderData
    expid = gd.expid
    wcs = gd.getWcs(guiderName)
    pixelScale = wcs.getPixelScale().asArcseconds()

    # Initialize parameters from config
    aperRadius = config.aperSizeArcsec / pixelScale
    cutOutSize = config.cutOutSize
    gain = config.gain

    # check if the ref center is within the image bounds
    stampShape = gd[guiderName, 0].shape
    if not (0 <= refCenter[0] < stampShape[1]) or not (0 <= refCenter[1] < stampShape[0]):
        return makeBlankCatalog()

    # loop over stamps
    results = []
    for i in range(len(gd)):
        data = gd[guiderName, i]
        star = measureStarOnStamp(data, refCenter, cutOutSize, aperRadius, gain=gain).toDataFrame()

        # Add stamp index
        if not star.empty:
            star["stamp"] = i
            results.append(star)

    # 3)  Concatenate
    if not results:
        return makeBlankCatalog()
    stars = pd.concat(results, ignore_index=True)

    # 4)  Add metadata
    stars["detector"] = guiderName
    stars["expid"] = expid
    stars["ampname"] = gd.getGuiderAmpName(guiderName)
    stars["detid"] = gd.getGuiderDetNum(guiderName)
    stars["filter"] = gd.header.get("filter", "UNKNOWN")
    return stars


def annulusBackgroundSubtraction(data: np.ndarray, annulus: tuple[float, float]) -> tuple[np.ndarray, float]:
    """
    Subtract background from the data using an annulus.

    Parameters
    ----------
    data : `np.ndarray`
        Image cutout data.
    annulus : `tuple[float, float]`
        Inner and outer radii (pixels) defining the background annulus.

    Returns
    -------
    dataBkgSub : `np.ndarray`
        Background-subtracted data.
    bkgStd : `float`
        Standard deviation of the background estimation.
    """
    rin, rout = annulus
    x0, y0 = data.shape[1] // 2, data.shape[0] // 2
    x, y = np.indices(data.shape)
    annMask = ((x - x0) ** 2 + (y - y0) ** 2 >= rin**2) & ((x - x0) ** 2 + (y - y0) ** 2 <= rout**2)
    annMask &= np.isfinite(data)
    _, bkgSub, bkgStd = sigma_clipped_stats(data[annMask], sigma=3.0)
    dataBkgSub = data - bkgSub
    return dataBkgSub, bkgStd


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
        Convert this measurement to a single-row DataFrame.

        Returns
        -------
        row : `pd.DataFrame`
            Single-row DataFrame with measurement fields, or empty if invalid.
        """
        d = asdict(self)
        # Only drop the column if xroi is NaN (i.e., measurement failed)
        if not np.isfinite(d.get("xroi", np.nan)):
            # Return an empty DataFrame with all the keys as columns,
            return pd.DataFrame(columns=list(d.keys()))
        # Otherwise, return all columns, even if some are NaN
        return pd.DataFrame([d])

    def aperturePhotometry(
        self, cutout: np.ndarray, radius: float, bkgStd: float = 1.0, gain: float = 1.0
    ) -> None:
        """
        Perform aperture photometry on a cutout image.

        Parameters
        ----------
        cutout : `np.ndarray`
            2D cutout image (background-subtracted).
        radius : `float`
            Aperture radius in pixels.
        bkgStd : `float`
            Background RMS per pixel.
        gain : `float`
            Detector gain (e-/ADU).
        """
        x0, y0 = self.xroi, self.yroi
        if np.isfinite(x0) and np.isfinite(y0):
            ny, nx = cutout.shape
            y, x = np.indices((ny, nx))
            x0, y0 = self.xroi, self.yroi

            # Background mask
            aperMask = (x - x0) ** 2 + (y - y0) ** 2 <= radius**2

            # Aperture sum
            fluxNet = np.nansum(cutout[aperMask])
            fluxNet = np.clip(fluxNet, 0, None)  # Ensure non-negative flux
            npix = aperMask.sum()

            # Flux error
            fluxErr = np.sqrt(fluxNet / gain + npix * bkgStd**2)
            snr = fluxNet / (fluxErr + 1e-9) if fluxErr > 0 else 0.0

            # Update the measurement
            self.flux = fluxNet
            self.flux_err = fluxErr
            self.snr = snr


def runSourceDetection(
    image: np.ndarray,
    threshold: float = 10,
    cutOutSize: int = 25,
    aperRadius: int = 5,
    gain: float = 1.0,
) -> pd.DataFrame:
    """
    Detect sources in an image and measure their properties.

    Parameters
    ----------
    image : `np.ndarray`
        2D image array.
    threshold : `float`
        Detection threshold in sigma units.
    cutOutSize : `int`
        Size of the cutout around each detected source (pixels).
    aperRadius : `int`
        Aperture radius in pixels for photometry.
    gain : `float`
        Detector gain (e-/ADU).

    Returns
    -------
    sources : `pd.DataFrame`
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
        # only single peaked stars
        if len(fp.getPeaks()) > 1:
            continue

        # Create a cutout of the image around the footprint
        refCenter = tuple(fp.getCentroid())
        star = measureStarOnStamp(image, refCenter, cutOutSize, aperRadius, gain).toDataFrame()
        if not star.empty:
            results.append(star)

    df = pd.concat([sf for sf in results], ignore_index=True)
    return df


def measureStarOnStamp(
    stamp: np.ndarray,
    refCenter: tuple[float, float],
    cutOutSize: int,
    aperRadius: int,
    gain: float = 1.0,
) -> StarMeasurement:
    """
    Measure a star on a single stamp: background subtraction, shape, centroid,
    photometry.

    Parameters
    ----------
    stamp : `np.ndarray`
        Full stamp array.
    refCenter : `tuple[float, float]`
        Reference (x, y) pixel position for the cutout center.
    cutOutSize : `int`
        Size of the cutout in pixels.
    aperRadius : `int`
        Aperture radius in pixels for photometry.
    gain : `float`
        Detector gain (e-/ADU).

    Returns
    -------
    measurement : `StarMeasurement`
        StarMeasurement object with populated fields (may be empty on failure).
    """
    cutout = getCutouts(stamp, refCenter, cutoutSize=cutOutSize)
    data = cutout.data

    if np.all(data == 0) | (not np.isfinite(data).all()):
        return StarMeasurement()

    # 1) Subtract the background
    annulus = (aperRadius * 1.0, aperRadius * 2)
    dataBkgSub, bkgStd = annulusBackgroundSubtraction(data, annulus)

    # 2)  Track the star across all stamps for this guider
    star = runGalSim(dataBkgSub, gain=gain, bkgStd=bkgStd)

    # 3) Make aperture photometry measurements
    # Galsim flux is the normalization of the Gaussian, not w/ fixed aper.
    star.aperturePhotometry(dataBkgSub, aperRadius, gain=gain, bkgStd=bkgStd)

    # 4)  Add centroid and shape in amplifier roi coordinates
    star.xroi += cutout.xmin_original
    star.yroi += cutout.ymin_original
    return star


def runGalSim(
    imageArray: np.ndarray,
    gain: float = 1.0,
    bkgStd: float = 0.0,
) -> StarMeasurement:
    """
    Measure star properties with GalSim adaptive moments.

    Parameters
    ----------
    imageArray : `np.ndarray`
        Background-subtracted image cutout.
    gain : `float`
        Detector gain (e-/ADU).
    bkgStd : `float`
        Background RMS per pixel.

    Returns
    -------
    result : `StarMeasurement`
        Resulting measurement (empty if measurement failed).
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
        snr = flux / (shotNoise + 1e-9) if shotNoise > 0 else 0.0

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

    Parameters
    ----------
    imageArray : `np.ndarray`
        Image cutout used for measurement.
    gs : `galsim.hsm`
        GalSim HSM shape data result object.
    gain : `float`
        Detector gain (e-/ADU).
    bkgStd : `float`
        Background RMS per pixel.
    isGain : `bool`
        Whether to include gain-dependent weighting.

    Returns
    -------
    xerr : `float`
        Estimated x centroid uncertainty (pixels).
    yerr : `float`
        Estimated y centroid uncertainty (pixels).
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

    Parameters
    ----------
    shape : `tuple[int, int]`
        (ny, nx) output array shape.
    flux : `float`
        Total flux (normalization).
    sigma : `float`
        Gaussian sigma (pixels).
    e1 : `float`
        Ellipticity component e1.
    e2 : `float`
        Ellipticity component e2.
    center : `tuple[float, float]`
        (x0, y0) centroid position in pixels.

    Returns
    -------
    image : `np.ndarray`
        Generated model image.
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
    guiderData: GuiderData,
    log: logging.Logger,
    config: GuiderStarTrackerConfig = GuiderStarTrackerConfig(),
) -> pd.DataFrame:
    """
    Build a reference star catalog from each guider's coadded stamp.

    Parameters
    ----------
    guiderData : `GuiderData`
        Guider dataset containing stamps and metadata.
    log : `logging.Logger`
        Logger for warnings and diagnostics.
    config : `GuiderStarTrackerConfig`
        Star tracker configuration.

    Returns
    -------
    refCatalog : `pd.DataFrame`
        Concatenated reference catalog of brightest stars per guider.
    """
    expId = guiderData.expid
    minSnr = config.minSnr
    gain = config.gain
    cutOutSize = config.cutOutSize

    tableList = []
    for guiderName in guiderData.guiderNames:
        pixelScale = guiderData.getWcs(guiderName).getPixelScale().asArcseconds()
        aperRadius = int(config.aperSizeArcsec / pixelScale)

        array = guiderData.getStampArrayCoadd(guiderName)
        array = np.where(array < 0, 0, array)  # Ensure no negative values
        sources = runSourceDetection(
            array,
            threshold=minSnr,
            aperRadius=aperRadius,
            cutOutSize=cutOutSize,
            gain=gain,
        )
        if sources.empty:
            log.warning(f"No sources detected in `buildReferenceCatalog`" f"for {guiderName} in {expId}. ")
            continue

        sources.sort_values(by=["snr"], ascending=False, inplace=True)
        sources.reset_index(drop=True, inplace=True)

        detNum = guiderData.getGuiderDetNum(guiderName)
        sources["detector"] = guiderName
        sources["detid"] = detNum
        sources["starid"] = detNum * 100
        tableList.append(sources)

    if len(tableList) == 0:
        log.warning(f"`buildReferenceCatalog` failed" f" - no stars detected in any guider for {expId}.")
        return makeBlankCatalog()

    refCatalog = pd.concat(tableList, ignore_index=True)
    return refCatalog


def getCutouts(imageArray: np.ndarray, refCenter: tuple[float, float], cutoutSize: int = 25) -> Cutout2D:
    """
    Get a cutout at the reference position from an image array.

    Parameters
    ----------
    imageArray : `np.ndarray`
        Full image array.
    refCenter : `tuple[float, float]`
        (x, y) center for the cutout in pixels.
    cutoutSize : `int`
        Size (pixels) of the square cutout.

    Returns
    -------
    cutout : `Cutout2D`
        Astropy Cutout2D object.
    """
    refX, refY = refCenter
    return Cutout2D(imageArray, (refX, refY), size=cutoutSize, mode="partial", fill_value=np.nan)
