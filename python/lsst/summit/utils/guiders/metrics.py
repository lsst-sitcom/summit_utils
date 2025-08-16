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

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from astropy.stats import mad_std

from lsst.summit.utils.utils import RobustFitResult, RobustFitter

__all__ = ["GuiderMetricsBuilder"]


class GuiderMetricsBuilder:
    """
    Measure and organize guider performance metrics for a given exposure.

    This class wraps the computation of both exposure-level counts (number of
    guiders, stars, measurements, fraction of valid stamps) and per-quantity
    trend metrics (ALT drift, AZ drift, rotator, photometry, PSF). Trend
    metrics include slope, intercept, trend RMSE, global scatter, outlier
    fraction, slope significance, and sample size.

    Parameters
    ----------
    starCatalog : `pandas.DataFrame`
        Catalog of guider star measurements, containing at least the columns
        required for the counts and trend metrics: ``expid``, ``elapsed_time``,
        and the measurement columns for each metric (e.g., ``dalt``, ``daz``,
        ``dtheta``, ``magoffset``, ``fwhm``).
    """

    def __init__(self, starCatalog: pd.DataFrame) -> None:
        self.starCatalog = starCatalog
        self.log = logging.getLogger(__name__)

        # Store the basic variable names for metrics
        self.baseVarsCols = {
            "altDrift": "dalt",
            "azDrift": "daz",
            "mag": "magoffset",
            "rotator": "dtheta",
            "psf": "fwhm",
        }
        self.baseVars = list(self.baseVarsCols.keys())

        # keep track if the metrics were build
        self.isBuilt = False

    def buildMetrics(self, expid: int) -> pd.DataFrame:
        """
        Compute all metrics for the specified exposure ID.

        Parameters
        ----------
        expid : `int`
            Exposure ID to compute metrics for.

        Returns
        -------
        metricsDf : `pandas.DataFrame`
            Single-row DataFrame with all computed metrics for the specified
            exposure ID. Columns include exposure counts (e.g., ``n_guiders``,
            ``n_stars``, per-guider flags) and each metric prefix
            (``alt_drift``, ``az_drift``, ``rotator``, ``mag``, ``psf``)
            expanded with its statistic names.
        """
        self.expid = expid
        stars = self.starCatalog

        # early exit if no data
        mask = stars["expid"].eq(expid)
        if not mask.any():
            self.isBuilt = False
            self.log.warning(f"No data found for expid={expid}. Returning empty metrics DataFrame.")
            return pd.DataFrame(columns=self.metricsColumns)  # FIX: property

        # build metrics
        self.countsDf = computeExposureCounts(stars, expid)
        self.altDriftData: guiderDriftResult = computeTrendMetrics(stars, "elapsed_time", "dalt", expid)
        self.azDriftData: guiderDriftResult = computeTrendMetrics(stars, "elapsed_time", "daz", expid)
        self.rotatorData: guiderDriftResult = computeTrendMetrics(stars, "elapsed_time", "dtheta", expid)
        self.magData: guiderDriftResult = computeTrendMetrics(stars, "elapsed_time", "magoffset", expid)
        self.psfData: guiderDriftResult = computeTrendMetrics(stars, "elapsed_time", "fwhm", expid)

        # Set the built state to true
        self.isBuilt = True

        # build MetricResult objects to a DataFrame
        return self.toDataFrame()

    def toDataFrame(self) -> pd.DataFrame:
        """
        Assemble all computed counts and trend metrics into a single row.

        Returns
        -------
        metricsDf : `pandas.DataFrame`
            DataFrame with one row for the currently set ``expid``. Columns
            include exposure counts (e.g., ``n_guiders``, ``n_stars``,
            per-guider flags) and each metric prefix (``alt_drift``,
            ``az_drift``, ``rotator``, ``mag``, ``psf``) expanded with its
            statistic names.
        """
        if not self.isBuilt:
            raise RuntimeError("Metrics have not been built. Call buildMetrics(expid) first.")

        trendsDf = mergeGuiderDriftResults(
            [
                ("alt_drift", self.altDriftData),
                ("az_drift", self.azDriftData),
                ("rotator", self.rotatorData),
                ("mag", self.magData),
                ("psf", self.psfData),
            ],
            index=self.expid,
        )
        return pd.concat([self.countsDf, trendsDf], axis=1)

    @property
    def metricsColumns(self) -> list[str]:
        """
        List of expected output column names for the metrics DataFrame.

        Combines the base count columns and each metric prefix with all
        statistic suffixes.

        Returns
        -------
        columns : `list` of `str`
            All column names in the order they will appear in the DataFrame
            returned by ``toDataFrame()`` or ``buildMetrics()``.
        """
        baseCols = ["n_guiders", "n_stars", "n_measurements", "fraction_valid_stamps"]
        statVars = [
            "slope",
            "intercept",
            "trend_rmse",
            "global_std",
            "outlier_frac",
            "slope_significance",
            "nsize",
        ]
        columns = baseCols[:]
        for var in self.baseVars:
            for stat in statVars:
                # convert camelCase to snake_case
                varSnakeCase = "".join(["_" + c.lower() if c.isupper() else c for c in var]).lstrip("_")
                columns.append(f"{varSnakeCase}_{stat}")
        return columns

    def printSummary(self) -> None:
        """
        Print a human-readable summary of all metrics.

        Each metric's slope is scaled by the exposure time.

        """
        # Guard if buildMetrics found no data
        if not self.isBuilt:
            raise RuntimeError("Metrics have not been built. Call buildMetrics(expid) first.")

        # set units (ensure consistency with y units!)
        self.altDriftData.units = "arcsec"
        self.azDriftData.units = "arcsec"
        self.rotatorData.units = "arcsec"
        self.magData.units = "mag"
        self.psfData.units = "arcsec"

        # exptime
        exptime = self.countsDf["exptime"].values[0]

        # Print summaries
        header1 = makeHeader("Guider Metrics Summary")
        print("\n".join(header1))
        print("Exposure ID:", self.expid)
        print(f"Exposure time: {exptime:.2f} sec")
        printExposureCounts(self.countsDf)

        self.azDriftData.pprint("Az")
        self.altDriftData.pprint("Alt")
        self.rotatorData.pprint("Rotator")
        self.magData.pprint("Mag")
        self.psfData.pprint("PSF FWHM")


def computeTrendMetrics(
    stars: pd.DataFrame,
    timeCol: str,
    yCol: str,
    expid: int,
) -> guiderDriftResult:
    """
    Compute robust linear trend metrics for a given measurement column versus
    time within a single exposure.

    The function fits a robust linear model to the specified `yCol` as a
    function of `timeCol` for rows matching the given `expid`. It returns a
    `MetricResult` containing the slope, intercept, trend RMSE, robust global
    scatter, outlier fraction, slope significance, and sample size.

    Parameters
    ----------
    stars : `pandas.DataFrame`
        Table of star measurements containing at least the columns `timeCol`,
        `yCol`, and `expid`.
    timeCol : `str`
        Name of the time column (e.g., ``"elapsed_time"``).
    yCol : `str`
        Name of the dependent variable column to fit (e.g., ``"dalt"``,
        ``"magoffset"``).
    expid : `int`
        Exposure identifier used to filter the rows.

    Returns
    -------
    metrics : `MetricResult`
        Dataclass containing the computed trend metrics. If there are no data
        in `yCol` after filtering, all fields are set to NaN/None and `nsize`
        is zero.

    Raises
    ------
    KeyError
        If `timeCol` or `yCol` are not present in `stars`.
    """
    s = stars.loc[stars["expid"].eq(expid), [timeCol, yCol]].dropna()
    if s.empty or s[yCol].nunique() < 2:
        empty_mask = np.zeros((0,), dtype=bool)
        return guiderDriftResult(
            fit=RobustFitResult(
                slope=np.nan,
                intercept=np.nan,
                scatter=np.nan,
                outlierMask=empty_mask,
                slopePValue=np.nan,
                slopeStdErr=np.nan,
                slopeTValue=np.nan,
                interceptPValue=np.nan,
                interceptStdErr=np.nan,
                interceptTValue=np.nan,
            ),
            globalStd=np.nan,
            nsize=0,
            units="",
            exptime=1.0,
        )

    x = s[timeCol].to_numpy()
    y = s[yCol].to_numpy()
    exptime = float(s[timeCol].max())
    global_std = float(mad_std(y))

    fitter = RobustFitter()
    fit_res = fitter.fit(x, y)

    return guiderDriftResult(
        fit=fit_res,
        globalStd=global_std,
        nsize=int(y.size),
        units="",
        exptime=exptime,
    )


@dataclass(slots=True)
class guiderDriftResult:
    """
    Metrics for guider data derived from a robust linear trend fit.

    This dataclass wraps a `RobustFitResult` with guider-specific
    fields. It stores the global scatter, number of valid points,
    and domain metadata such as units and exposure time. Properties
    provide easy access to slope, intercept, trend RMSE, outlier
    fraction, and slope significance.

    Parameters
    ----------
    fit : `RobustFitResult`
        The result of the robust fit containing slope and intercept.
    globalStd : `float`
        Robust global standard deviation of the dependent values.
    nsize : `int`
        Number of valid points used in the fit.
    units : `str`, optional
        Units of the dependent variable. Default is empty string.
    exptime : `float`, optional
        Exposure time in seconds. Default is 1.0.
    """

    fit: RobustFitResult  # composition, not duplication
    globalStd: float  # robust global std of y
    nsize: int
    units: str = ""
    exptime: float = 1.0

    def __post_init__(self) -> None:
        assert self.fit is not None, "fit must be provided"

    @property
    def slope(self) -> float:
        return self.fit.slope

    @property
    def intercept(self) -> float:
        return self.fit.intercept

    @property
    def trendRmse(self) -> float:
        return self.fit.scatter

    @property
    def outlierFrac(self) -> float:
        m = self.fit.outlierMask
        return float(np.count_nonzero(m)) / float(m.size) if m.size else np.nan

    @property
    def slopeSignificance(self) -> float | None:
        t = self.fit.slopeTValue
        return abs(float(t)) if t is not None else None

    def toDataFrame(self, prefix: str, index: int = 0) -> pd.DataFrame:
        """
        Convert the stored metrics into a single-row DataFrame.

        Parameters
        ----------
        prefix : `str`
            Prefix to add to each metric's column name in the output.
        index : `int`, optional
            Index value for the returned DataFrame row.

        Returns
        -------
        metrics : `pandas.DataFrame`
            Single-row DataFrame containing the numeric/statistical fields of
            this result, prefixed with `prefix`.
        """
        row = {
            f"{prefix}_slope": self.slope,
            f"{prefix}_intercept": self.intercept,
            f"{prefix}_trend_rmse": self.trendRmse,
            f"{prefix}_global_std": self.globalStd,
            f"{prefix}_outlier_frac": self.outlierFrac,
            f"{prefix}_slope_significance": self.slopeSignificance,
            f"{prefix}_nsize": self.nsize,
        }
        return pd.DataFrame([row], index=[index])

    def pprint(self, title: str) -> None:
        """
        Print the stored metrics in a formatted, human-readable block.

        Parameters
        ----------
        title : `str`
            Title to display for the metric block.
        """
        if not title:
            title = "Metric"
        units = self.units
        exptime = self.exptime

        header = makeHeader(f"Metrics Summary: {title}", nchar=40)
        print("\n".join(header))
        slope_per_exp = self.slope * exptime
        print(f"  Slope          : {slope_per_exp:.3f} {units} per exposure")
        sig = "—" if self.slopeSignificance is None else f"{self.slopeSignificance:.1f}"
        print(f"  Slope signif.  : {sig} sigma")
        print(f"  Intercept      : {self.intercept:.3f} {units}")
        print(f"  Trend RMSE     : {self.trendRmse:.3f} {units}")
        print(f"  Global std     : {self.globalStd:.3f} {units}")
        print(f"  Outlier frac   : {self.outlierFrac:.2%}")
        print(f"  N (points)     : {self.nsize:d}\n")


def mergeGuiderDriftResults(pairs: list[tuple[str, guiderDriftResult]], index: int) -> pd.DataFrame:
    """
    Merge multiple `MetricResult` objects into a single-row DataFrame with
    prefixed column names.

    Each `MetricResult` is converted to a dictionary, filtered to keep only a
    standard set of metric fields, and all keys are prefixed with the provided
    label (e.g., ``"alt_drift"``, ``"mag"``) to distinguish different
    measurement types. The merged results are returned as a single-row
    DataFrame.

    Parameters
    ----------
    pairs : `list` of `tuple` of (`str`, `MetricResult`)
        List of (prefix, metric result) pairs. The prefix is used as the column
        name prefix for that result's metrics.
    index : `int`
        Index value for the returned DataFrame row.

    Returns
    -------
    merged : `pandas.DataFrame`
        Single-row DataFrame containing all merged metrics with column names of
        the form ``<prefix>_<metric_field>``.
    """
    row: dict[str, float | int | None] = {}
    keep = ("slope", "intercept", "trendRmse", "globalStd", "outlierFrac", "slopeSignificance", "nsize")
    for prefix, mr in pairs:
        d = asdict(mr)
        filtered = {k: d.get(k) for k in keep}
        row.update(_prefixKeys(filtered, prefix))
    return pd.DataFrame([row], index=[index])


def computeExposureCounts(stars: pd.DataFrame, expid: int) -> pd.DataFrame:
    """
    Compute guider/star/measurement counts for the given expid.

    Parameters
    ----------
    stars : `pandas.DataFrame`
        Star measurement rows.
    expid : `int`
        Exposure ID to filter on.

    Returns
    -------
    countsDf : `pandas.DataFrame`
        Single-row DataFrame with counts for the exposure.
    """
    s = stars.loc[stars["expid"].eq(expid)]
    exptime = float(s["elapsed_time"].max())

    # Guiders and stars per guider
    nGuiders = s["detector"].nunique()
    nUnique = s["detid"].nunique()
    counts = s.groupby("detector")["detid"].nunique().to_dict()
    guiderNames = sorted(s["detector"].unique())
    guidersPresent = {f"{det}": (counts.get(det, 0) > 0) for det in guiderNames}

    # Valid measurements
    maskValid = (s["stamp"] >= 0) & (s["xccd"].notna())
    nMeas = int(maskValid.sum())

    # Fraction of valid stamps (protect against div-by-zero)
    nStamps = s["stamp"].nunique()
    totalPossible = nUnique * nStamps
    fracValid = (nMeas / totalPossible) if totalPossible > 0 else np.nan

    row = {
        "n_guiders": nGuiders,
        "n_stars": nUnique,
        "n_measurements": nMeas,
        "fraction_valid_stamps": fracValid,
        "exptime": exptime,
    }
    row.update(guidersPresent)
    return pd.DataFrame([row], index=[expid])


def printExposureCounts(countsDf: pd.DataFrame, precision: int = 3) -> None:
    """
    Print exposure-level counts from a single-row counts DataFrame.

    Parameters
    ----------
    countsDf : `pandas.DataFrame`
        DataFrame with a single row containing exposure counts (e.g.,
        n_guiders, n_stars, per-guider flags).
    precision : `int`, optional
        Number of decimal places for fractional values.
    """
    if countsDf.empty:
        print("Exposure counts ─\nNo data available.")
        return

    # Get first row as dict for easy .get()
    row = countsDf.iloc[0].to_dict()

    lines = makeHeader("Exposure Counts")
    lines += [
        f"Guiders     : {int(row.get('n_guiders', 0))}",
        f"Unique stars: {int(row.get('n_stars', 0))}",
        f"Measurements: {int(row.get('n_measurements', 0))}",
    ]
    frac = row.get("fraction_valid_stamps")
    if isinstance(frac, (int, float, np.floating)):
        lines.append(f"Valid stamps: {float(frac):.{precision}f}")

    # Per-guider boolean flags if present
    guider_flags = [k for k in row.keys() if k.startswith("R") and "_SG" in k]
    if guider_flags:
        present = [g for g in sorted(guider_flags) if bool(row.get(g))]
        lines.append(f"Guiders used: {', '.join(present) if present else '—'}")

    print("\n".join(lines))
    print()


def _prefixKeys(d: dict, prefix: str) -> dict:
    """
    Return a new dictionary with all keys prefixed.

    The keys of the input dictionary are prefixed with the provided string and
    an underscore. This is typically used to namespace metric names by a
    measurement type or category.

    Parameters
    ----------
    d : `dict`
        Input dictionary with string keys and any values.
    prefix : `str`
        Prefix to prepend to each key.

    Returns
    -------
    prefixed : `dict`
        New dictionary where each key is formatted as
        ``f"{prefix}_{original_key}"`` and values are unchanged.

    Examples
    --------
    >>> _prefixKeys({"slope": 1.2, "rmse": 0.1}, "alt_drift")
    {'alt_drift_slope': 1.2, 'alt_drift_rmse': 0.1}
    """
    return {f"{prefix}_{k}": v for k, v in d.items()}


def makeHeader(title: str, nchar: int = 40) -> list[str]:
    """
    Create a formatted header block with horizontal lines.

    The header consists of a top line, a centered title line with padding, and
    a bottom line. The line width is the maximum of `nchar` and the title
    length plus 10 characters.

    Parameters
    ----------
    title : `str`
        Text to display in the header.
    nchar : `int`, optional
        Minimum width of the horizontal lines. Increased automatically if the
        title requires more space.

    Returns
    -------
    header_lines : `list` of `str`
        List of three strings: the top line, the title line, and the bottom
        line.
    """
    width = max(nchar, len(title) + 10)
    line = "─" * width
    header = f"{' ' * 5}{title}{' ' * 5}"
    return [line, header, line]


def makeLine(nchar: int = 40) -> str:
    """
    Create a horizontal line of box-drawing characters.

    Parameters
    ----------
    nchar : `int`, optional
        Number of characters in the line. Default is 40.

    Returns
    -------
    line : `str`
        String consisting of `nchar` repetitions of the '─' character.
    """
    return "─" * nchar
