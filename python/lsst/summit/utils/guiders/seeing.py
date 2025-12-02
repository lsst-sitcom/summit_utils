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

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from lsst.summit.utils.guiders.metrics import GuiderMetricsBuilder

__all__ = [
    "rotateCentroids",
    "groupStarsPerDetector",
    "fwhmVonkarman",
    "r0FromVariance",
    "calcCentroidStats",
    "getROICenters",
    "alignTimes",
    "getCorrectedXy",
    "GuiderSeeing",
    "CorrelationAnalysis",
]


MM_TO_PIXEL = 100.0


def rotateCentroids(
    centroids1: NDArray[np.float64],
    centroids2: NDArray[np.float64],
    center1: NDArray[np.float64],
    center2: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rotate two sets of centroid coordinates into a common local frame.

    The local coordinate system is defined by the vector connecting ``center1``
    to ``center2``. That direction becomes the local y-axis (parallel), and its
    perpendicular becomes the local x-axis (perpendicular). Both centroid sets
    are projected into this frame.

    Parameters
    ----------
    centroids1 : `ndarray`
        Shape (N, 2). Original (x, y) centroids from the first detector, in
        focal-plane pixels.
    centroids2 : `ndarray`
        Shape (M, 2). Original (x, y) centroids from the second detector, in
        focal-plane pixels.
    center1 : `ndarray`
        Shape (2,). Mean (x, y) position of ``centroids1``, used to define the
        rotation axis.
    center2 : `ndarray`
        Shape (2,). Mean (x, y) position of ``centroids2``, used to define the
        rotation axis.

    Returns
    -------
    centroids1Rot : `ndarray`
        Shape (N, 2). Rotated coordinates of ``centroids1`` in the common
        frame. Columns correspond to (x_perp, y_parallel).
    centroids2Rot : `ndarray`
        Shape (M, 2). Rotated coordinates of ``centroids2`` in the same frame.
    """
    yHat = (center2 - center1).astype(float)
    yHat /= np.linalg.norm(yHat)

    y1Rot = np.dot(yHat[np.newaxis, :], centroids1.T).reshape(-1)
    y2Rot = np.dot(yHat[np.newaxis, :], centroids2.T).reshape(-1)

    xHat = np.array([yHat[1], -yHat[0]], dtype=float)

    x1Rot = np.dot(xHat[np.newaxis, :], centroids1.T).reshape(-1)
    x2Rot = np.dot(xHat[np.newaxis, :], centroids2.T).reshape(-1)

    centroids1Rot = np.column_stack((x1Rot, y1Rot))
    centroids2Rot = np.column_stack((x2Rot, y2Rot))

    return centroids1Rot, centroids2Rot


def groupStarsPerDetector(starsDetrended: pd.DataFrame, fluxCut: float) -> dict[str, pd.DataFrame]:
    """Group detrended star rows by detector and apply a mean-flux threshold.

    Parameters
    ----------
    starsDetrended : `pandas.DataFrame`
        Detrended star table containing at least columns ``'detector'`` and
        ``'flux'``.
    fluxCut : `float`
        Minimum mean flux required to keep a detector.

    Returns
    -------
    starList : `dict[str, pandas.DataFrame]`
        Mapping from detector ID to filtered dataframe for that detector.
        Detectors below the flux threshold are omitted.
    """
    groups = dict(tuple(starsDetrended.groupby("detector", sort=False)))

    toDrop: list[str] = []
    for det, df in groups.items():
        meanFlux = float(np.nanmean(pd.to_numeric(df["flux"], errors="coerce")))
        if not (meanFlux > fluxCut):
            toDrop.append(det)
    for det in toDrop:
        groups.pop(det, None)
    return groups


def fwhmVonkarman(r0: float | NDArray[np.float64], outerScale: float = 25.0) -> NDArray[np.float64]:
    """Atmospheric FWHM (Vonkármán PSF) from Fried parameter ``r0`` at 500 nm.

    Parameters
    ----------
    r0 : `float` or `ndarray`
        Fried parameter (meters) referenced to 500 nm.
    outerScale : `float`, optional
        Outer scale of turbulence (meters).

    Returns
    -------
    fwhm : `ndarray`
        Atmospheric seeing FWHM (arcsec) at 500 nm under Vonkármán model.
    """
    r0Arr = np.asarray(r0, dtype=float)
    kolmTerm = (0.976 * 0.5) / (4.85 * r0Arr)
    vkCorrection = np.sqrt(1.0 - 2.183 * (r0Arr / outerScale) ** 0.356)
    return (kolmTerm * vkCorrection).astype(np.float64)


def r0FromVariance(variance: float | NDArray[np.float64]) -> NDArray[np.float64]:
    """Infer Fried parameter ``r0`` from centroid-motion variance.

    Parameters
    ----------
    var : `float` or `ndarray`
        Variance proxy for centroid motion (units consistent with model).

    Returns
    -------
    r0 : `ndarray`
        Fried parameter (same unit system as the 0.1 scaling in formula).

    Notes
    -----
    Uses a emperically found scaling: ``r0 ≈ (var/0.1)^(-3/5) * 0.1``.
    """
    variance = np.asarray(variance, dtype=float)
    return ((variance / 0.1) ** (-3.0 / 5.0) * 0.1).astype(np.float64)


@dataclass(frozen=True, slots=True)
class GuiderSeeing:
    """Container for tomographic seeing results.

    Attributes
    ----------
    total : `float`
        Total seeing full width at half maximum (FWHM) in arcseconds.
    low : `float`
        Estimated FWHM contribution from the low-altitude turbulence layer
        (typically < 0.5 km), inferred from longer baseline correlations.
    mid : `float`
        Estimated FWHM contribution from the mid-altitude turbulence layer
        (~0.5-1.5 km), representing the residual between low- and high-layer
        components.
    high : `float`
        Estimated FWHM contribution from the high-altitude turbulence layer (>
        1.5 km), inferred from the shorter baseline correlations of centroid
        motion.
    """

    total: float
    low: float
    mid: float
    high: float


class CorrelationAnalysis:
    """Correlation measurement on guider star centroid motions.

    Provides utilities to:
    - filter detectors by flux threshold,
    - estimate and correct common drift across detectors, and
    - measure per-detector variance and pairwise correlations of
      corrected centroid motions.

    Parameters
    ----------
    stars : `pandas.DataFrame`
        Table of star measurements. Must contain columns:
        ``'detector'``, ``'stamp'`` (time), ``'xfp'``, ``'yfp'``, ``'flux'``,
        ``'expid'``, and ``'filter'``.
    expid : `int`
        Exposure ID to select. Only rows with this exposure ID are retained.
    fluxCut : `float`, optional
        Minimum mean flux required to keep a detector.

    Raises
    ------
    ValueError
        If no rows with the given ``expid`` are found in ``stars``.
    ValueError
        If more than one unique filter is present in the selected
        exposure.
    """

    def __init__(self, stars: pd.DataFrame, expid: int, fluxCut: float = 4e3) -> None:
        self.expid = expid
        self.fluxCut = fluxCut

        # Early exit if no data
        mask = stars["expid"].eq(expid)
        if not mask.any():
            raise ValueError(f"Stars dataframe has no data with expid={expid}")

        starsFiltered = stars[mask].copy()

        # Wavelength information from LSST bandpass centers (nm).
        # TODO: DM-52059 Use official filter wavelengths once they're available
        lamMap = {
            "u": 372.4,
            "g": 480.7,
            "r": 622.1,
            "i": 755.9,
            "z": 868.0,
            "y": 975.3,
        }
        band = starsFiltered["filter"].unique()
        if len(band) != 1:
            raise ValueError("cannot have more than one filter per exposure")
        self.wavelength = float(lamMap[str(band[0])])

        # Initialize metrics builder and detrend stars.
        # These classes/objects are external to this module.
        dummyNmissingStampPlaceholder = 0
        metricsBuilder = GuiderMetricsBuilder(starsFiltered, dummyNmissingStampPlaceholder)
        self.metrics = metricsBuilder.buildMetrics(self.expid)
        starsDetrended = metricsBuilder.detrendStars()

        # Build star list per detector with a flux threshold.
        self.starList = groupStarsPerDetector(starsDetrended, fluxCut)
        self.detectors = list(self.starList.keys())

    def measureVariance(self) -> list[float]:
        """Compute per-detector variances of corrected centroid motions.

        Requires that detrending has already been applied. Variances are scaled
        to units of (pix x MM_TO_PIXEL)^2.

        Returns
        -------
        out : `list[float]`
            Flattened list of per-detector variances: ``[vx_det1, vy_det1,
            vx_det2, vy_det2, ...]``. If a detector was excluded or lacks
            corrected coordinates, entries are ``NaN``.
        """
        # Verify detrending outputs are present (handled upstream).
        # Leaving this note instead of commented-out long lines.

        out: list[float] = []
        for df in self.starList.values():
            meanFlux = float(np.nanmean(pd.to_numeric(df["flux"], errors="coerce")))
            hasCorr = "dxfp_corr" in df and "dyfp_corr" in df
            if meanFlux > self.fluxCut and hasCorr:
                vx = np.nanvar(pd.to_numeric(df["dxfp_corr"], errors="coerce")) * (MM_TO_PIXEL**2)
                vy = np.nanvar(pd.to_numeric(df["dyfp_corr"], errors="coerce")) * (MM_TO_PIXEL**2)
                out.extend([vx, vy])
            else:
                out.extend([np.nan, np.nan])
        return out

    @staticmethod
    def getBaselinePairs(kind: str) -> tuple[tuple[str, str], ...]:
        """Produce preset detector baseline pairs.

        Parameters
        ----------
        kind : `str`
            One of ``{'adjacent', 'cross', 'diagonal'}``.

        Returns
        -------
        pairs : `tuple` of `tuple`
            Immutable sequence of 2-element detector pairs ``(('R00_SG0',
            'R00_SG1'), ...)``.

        Raises
        ------
        KeyError
            If ``kind`` is not a recognized preset.
        """
        presets: dict[str, tuple[tuple[str, str], ...]] = {
            "adjacent": (
                ("R00_SG0", "R00_SG1"),
                ("R04_SG0", "R04_SG1"),
                ("R40_SG0", "R40_SG1"),
                ("R44_SG0", "R44_SG1"),
            ),
            "cross": (
                ("R00_SG0", "R40_SG1"),
                ("R04_SG1", "R44_SG0"),
                ("R00_SG1", "R04_SG0"),
                ("R40_SG0", "R44_SG1"),
            ),
            "diagonal": (
                ("R00_SG0", "R44_SG1"),
                ("R00_SG1", "R44_SG0"),
                ("R40_SG0", "R04_SG1"),
                ("R40_SG1", "R04_SG0"),
            ),
        }
        return presets[kind]

    def measurePairwiseStats(
        self, kind: str
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Compute pairwise correlations, covariances, and differential
        variances between centroid motions of detector pairs.

        The detector pairs are chosen by a preset specified by ``kind``. For
        each pair of detectors, star centroids are rotated into the local
        (perpendicular, parallel) frame defined by the vector that connects
        their mean positions. The method then computes:
        - Pearson correlation coefficients,
        - covariance (scaled to centipix²), and
        - variance of the centroid differences (scaled to centipix²),
        separately for the perpendicular (x) and parallel (y) components.

        Parameters
        ----------
        kind : `str`
            Preset name passed to ``self.getBaselinePairs()``.

        Returns
        -------
        corrs : `ndarray`
            Shape (P, 2). Correlation coefficients per pair: columns
            ``[corrPerp, corrPar]``.
        covs : `ndarray`
            Shape (P, 2). Covariances per pair (scaled by ``MM_TO_PIXEL^2``):
            columns ``[covPerp, covPar]``.
        diffVars : `ndarray`
            Shape (P, 2). Variances of centroid differences per pair (scaled by
            ``MM_TO_PIXEL^2``): columns ``[diffVarPerp, diffVarPar]``.
        """
        pairs = self.getBaselinePairs(kind)

        corrs: list[list[float]] = []
        covs: list[list[float]] = []
        diffVars: list[list[float]] = []

        for det1, det2 in pairs:
            df1 = self.starList.get(det1)
            df2 = self.starList.get(det2)
            if df1 is None or df2 is None:
                corrs.append([np.nan, np.nan])
                covs.append([np.nan, np.nan])
                diffVars.append([np.nan, np.nan])
                continue

            i1, i2 = alignTimes(df1, df2)
            if i1 is None or i2 is None:
                corrs.append([np.nan, np.nan])
                covs.append([np.nan, np.nan])
                diffVars.append([np.nan, np.nan])
                continue

            roiCenter1, roiCenter2 = getROICenters(df1, df2)
            starCentroid1 = getCorrectedXy(df1)
            starCentroid2 = getCorrectedXy(df2)
            alignedCentroid1, alignedCentroid2 = rotateCentroids(
                starCentroid1, starCentroid2, roiCenter1, roiCenter2
            )

            pairCorrs: list[float] = []
            pairCovs: list[float] = []
            pairDvars: list[float] = []
            for comp in (0, 1):  # 0 = ⟂ (x), 1 = ∥ (y)
                corr, cov, dvar = calcCentroidStats(alignedCentroid1[i1, comp], alignedCentroid2[i2, comp])
                pairCorrs.append(corr)
                pairCovs.append(cov)
                pairDvars.append(dvar)

            corrs.append(pairCorrs)
            covs.append(pairCovs)
            diffVars.append(pairDvars)

        return (
            np.asarray(corrs, dtype=np.float64),
            np.asarray(covs, dtype=np.float64),
            np.asarray(diffVars, dtype=np.float64),
        )

    def calcTomographicCoeffs(self) -> dict[str, tuple[tuple[float, float], ...]]:
        """Return nested coefficient tuples for tomographic seeing layer
        weights for the low and high layers.

        Structure mirrors the correlation data order: ((diag_x, diag_y),
        (cross_x, cross_y), (adj_x, adj_y)).

        Returns
        -------
        coeffs : `dict`
            Dictionary with keys ``'low'`` and ``'high'`` mapping to the
            corresponding coefficient tuples.
        """
        # TODO: add ROI position dependence
        return {
            "low": ((0, 0.51), (0.49, 0), (0, 0)),
            "high": ((0, 0), (0, 0), (0.872, 0.128)),
        }

    def measureTomographicSeeing(self) -> GuiderSeeing:
        """Estimate tomographic seeing by splitting the total FWHM into
        low/mid/high altitude contributions.

        This method:
        1) computes pairwise correlations for three detector-pair presets
           (adjacent, cross, diagonal),
        2) estimates a total Fried parameter from the median centroid variance
           across detectors, and
        3) converts to total FWHM using a Vonkármán model and scales to the
           exposure wavelength,
        4) splits the total into low/mid/high layers using empirical weights
           based on correlations.

        Returns
        -------
        result : `GuiderSeeing`
            Dataclass containing total, low-, mid-, and high-altitude FWHM
            components (arcsec) for the exposure.
        """
        corrsAdj, _, _ = self.measurePairwiseStats("adjacent")
        corrsCross, _, _ = self.measurePairwiseStats("cross")
        corrsDiag, _, _ = self.measurePairwiseStats("diagonal")
        var = self.measureVariance()

        r0 = r0FromVariance(float(np.nanmedian(var)))
        # Wavelength scaling: r0 ∝ λ^(6/5) ⇒ FWHM ∝ λ^(-1/5)
        fwhmTot = float(fwhmVonkarman(r0) * (self.wavelength / 500.0) ** (-1.0 / 5.0))

        coeffs = self.calcTomographicCoeffs()
        weightsLow = (
            coeffs["low"][0][0] * float(np.nanmedian(corrsDiag[:, 0]))
            + coeffs["low"][0][1] * float(np.nanmedian(corrsDiag[:, 1]))
            + coeffs["low"][1][0] * float(np.nanmedian(corrsCross[:, 0]))
            + coeffs["low"][1][1] * float(np.nanmedian(corrsCross[:, 1]))
            + coeffs["low"][2][0] * float(np.nanmedian(corrsAdj[:, 0]))
            + coeffs["low"][2][1] * float(np.nanmedian(corrsAdj[:, 1]))
        )
        weightsHigh = (
            1
            - coeffs["high"][0][0] * float(np.nanmedian(corrsDiag[:, 0]))
            - coeffs["high"][0][1] * float(np.nanmedian(corrsDiag[:, 1]))
            - coeffs["high"][1][0] * float(np.nanmedian(corrsCross[:, 0]))
            - coeffs["high"][1][1] * float(np.nanmedian(corrsCross[:, 1]))
            - coeffs["high"][2][0] * float(np.nanmedian(corrsAdj[:, 0]))
            - coeffs["high"][2][1] * float(np.nanmedian(corrsAdj[:, 1]))
        )
        weightsMid = 1.0 - weightsLow - weightsHigh

        def safeGuard(w: float) -> float:
            if np.isnan(w):
                return np.nan
            return np.sqrt(max(0.0, float(w)))

        return GuiderSeeing(
            total=fwhmTot,
            low=fwhmTot * safeGuard(weightsLow),
            mid=fwhmTot * safeGuard(weightsMid),
            high=fwhmTot * safeGuard(weightsHigh),
        )


def calcCentroidStats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Compute Pearson correlation, covariance, and (a-b) variance for arrays.

    Parameters
    ----------
    a : `ndarray`
        First 1D array (component of rotated centroids).
    b : `ndarray`
        Second 1D array (component of rotated centroids).

    Returns
    -------
    corr : `float`
        Pearson correlation coefficient.
    cov : `float`
        Covariance scaled to centipix² (i.e., multiplied by ``MM_TO_PIXEL^2``).
    dvar : `float`
        Variance of (a - b) scaled to centipix² (i.e., multiplied by
        ``MM_TO_PIXEL^2``).
    """
    if np.all(np.isnan(a)) or np.all(np.isnan(b)):
        return np.nan, np.nan, np.nan
    corr = float(np.corrcoef(a, b)[0, 1])
    cov = float(np.cov(a, b, ddof=1)[0, 1]) * (MM_TO_PIXEL**2)
    dvar = float(np.var(a - b, ddof=1)) * (MM_TO_PIXEL**2)
    return corr, cov, dvar


def getROICenters(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean (x, y) centers for two detector dataframes.

    Parameters
    ----------
    df1 : `pandas.DataFrame`
        First detector dataframe; must contain ``'xfp'``, ``'yfp'``.
    df2 : `pandas.DataFrame`
        Second detector dataframe; must contain ``'xfp'``, ``'yfp'``.

    Returns
    -------
    center1 : `ndarray`
        Shape (2,). Mean of (xfp, yfp) for ``df1``.
    center2 : `ndarray`
        Shape (2,). Mean of (xfp, yfp) for ``df2``.
    """
    c1 = np.array(
        [
            np.nanmean(pd.to_numeric(df1["xfp"], errors="coerce")),
            np.nanmean(pd.to_numeric(df1["yfp"], errors="coerce")),
        ],
        float,
    )
    c2 = np.array(
        [
            np.nanmean(pd.to_numeric(df2["xfp"], errors="coerce")),
            np.nanmean(pd.to_numeric(df2["yfp"], errors="coerce")),
        ],
        float,
    )
    return c1, c2


def alignTimes(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Align two detector dataframes on common ``elapsed_time`` stamps.

    Parameters
    ----------
    df1 : `pandas.DataFrame`
        First detector dataframe; must contain ``'elapsed_time'``.
    df2 : `pandas.DataFrame`
        Second detector dataframe; must contain ``'elapsed_time'``.

    Returns
    -------
    i1 : `ndarray` or `None`
        Indices into ``df1`` for rows matching the common timestamps, or
        ``None`` if no overlap.
    i2 : `ndarray` or `None`
        Indices into ``df2`` for rows matching the common timestamps, or
        ``None`` if no overlap.
    """
    t, i1, i2 = np.intersect1d(df1["elapsed_time"], df2["elapsed_time"], return_indices=True)
    return (i1, i2) if t.size else (None, None)


def getCorrectedXy(df: pd.DataFrame) -> np.ndarray:
    """Return corrected (x, y) centroid columns as a 2D array.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Detector dataframe containing columns ``'dxfp_corr'`` and
        ``'dyfp_corr'``.

    Returns
    -------
    xy : `ndarray`
        Shape (N, 2) array with columns (dxfp_corr, dyfp_corr).
    """
    return np.column_stack([df["dxfp_corr"], df["dyfp_corr"]]).astype(np.float64)
