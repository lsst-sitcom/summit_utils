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

from lsst.summit.utils.utils import RobustFitter

__all__ = ["GuiderMetrics"]


class GuiderMetrics:
    """
    Measure and organize guider performance metrics for a given exposure.

    This class wraps the computation of both exposure-level counts
    (number of guiders, stars, measurements, fraction of valid stamps)
    and per-quantity trend metrics (ALT drift, AZ drift, rotator,
    photometry, PSF). Trend metrics include slope, intercept, trend RMSE,
    global scatter, outlier fraction, slope significance, and sample size.

    Parameters
    ----------
    starCatalog : `pandas.DataFrame`
        Catalog of guider star measurements, containing at least
        the columns required for the counts and trend metrics:
        ``expid``, ``elapsed_time``, and the measurement columns
        for each metric (e.g., ``dalt``, ``daz``, ``dtheta``,
        ``magoffset``, ``fwhm``).
    """

    def __init__(self, starCatalog: pd.DataFrame) -> None:
        """
        Initialize the metrics object with a star measurement catalog.

        Parameters
        ----------
        starCatalog : `pandas.DataFrame`
            Full guider star measurement table for one or more exposures.
            Must include ``expid`` to identify individual exposures and
            the columns required for all desired metrics.
        """
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

        # Metric results are populated by buildMetrics; initialize for mypy
        self.countsDf: pd.DataFrame | None = None
        self.altDriftData: MetricResult | None = None
        self.azDriftData: MetricResult | None = None
        self.rotatorData: MetricResult | None = None
        self.magData: MetricResult | None = None
        self.psfData: MetricResult | None = None

        # keep track if the metrics were build
        self.isBuild = False

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
            Single-row DataFrame with all computed metrics for the
            specified exposure ID. Columns include exposure counts
            (e.g., ``n_guiders``, ``n_stars``, per-guider flags) and
            each metric prefix (``alt_drift``, ``az_drift``, ``rotator``,
            ``mag``, ``psf``) expanded with its statistic names.
        """
        self.expid = expid

        # early exit if no data
        mask = self.starCatalog["expid"].eq(expid)
        if not mask.any():  # faster/clearer
            self.isBuild = False
            self.log.warning(f"No data found for expid={expid}. Returning empty metrics DataFrame.")
            return pd.DataFrame(columns=self.metricsColumns)  # FIX: property

        # build metrics
        self.countsDf = computeExposureCounts(self.starCatalog, expid)

        # compute trend metrics for each variable
        for var, col in self.baseVarsCols.items():
            setattr(self, f"{var}Data", computeTrendMetrics(self.starCatalog, "elapsed_time", col, expid))

        # Set the build state to true
        self.isBuild = True

        # build MetricResult objects to a DataFrame
        return self.toDataFrame()

    def toDataFrame(self) -> pd.DataFrame:
        """
        Assemble all computed counts and trend metrics into a single row.

        Returns
        -------
        metricsDf : `pandas.DataFrame`
            DataFrame with one row for the currently set ``expid``.
            Columns include exposure counts (e.g., ``n_guiders``,
            ``n_stars``, per-guider flags) and each metric prefix
            (``alt_drift``, ``az_drift``, ``rotator``, ``mag``, ``psf``)
            expanded with its statistic names.
        """
        if not self.isBuild:
            raise RuntimeError("Metrics have not been built. Call buildMetrics(expid) first.")

        assert self.countsDf is not None
        assert self.altDriftData is not None
        assert self.azDriftData is not None
        assert self.rotatorData is not None
        assert self.magData is not None
        assert self.psfData is not None

        trendsDf = mergeMetricResults(
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

        Combines the base count columns and each metric prefix with
        all statistic suffixes.

        Returns
        -------
        columns : `list` of `str`
            All column names in the order they will appear in the
            DataFrame returned by ``toDataFrame()`` or ``buildMetrics()``.
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
        ]  # FIX: include nsize
        columns = baseCols[:]
        for var in self.baseVars:
            for stat in statVars:
                # convert camelCase to snake_case
                varSnakeCase = "".join(["_" + c.lower() if c.isupper() else c for c in var]).lstrip("_")
                columns.append(f"{varSnakeCase}_{stat}")
        return columns

    def printSummary(self, exptime: float = 1.0) -> None:
        """
        Print a human-readable summary of all metrics.

        Each metric's slope is scaled by the provided exposure time.

        Parameters
        ----------
        exptime : `float`, optional
            Exposure time in seconds. Used to convert slope units
            from per-second to per-exposure in the printed output.
        """
        # Guard if buildMetrics found no data
        if not self.isBuild:
            raise RuntimeError("Metrics have not been built. Call buildMetrics(expid) first.")

        assert self.countsDf is not None
        assert self.altDriftData is not None
        assert self.azDriftData is not None
        assert self.rotatorData is not None
        assert self.magData is not None
        assert self.psfData is not None

        # set units (ensure consistency with y units!)
        self.altDriftData.units = "arcsec"
        self.azDriftData.units = "arcsec"
        self.rotatorData.units = "arcsec"
        self.magData.units = "mag"
        self.psfData.units = "arcsec"

        # Print summaries

        header1 = makeHeader("Guider Metrics Summary")
        print("\n".join(header1))
        print("Exposure ID:", self.expid)
        print(f"Exposure time: {exptime:.2f} sec")
        printExposureCounts(self.countsDf)

        self.azDriftData.pprint("Az", exptime)
        self.altDriftData.pprint("Alt", exptime)
        self.rotatorData.pprint("Rotator", exptime)
        self.magData.pprint("Mag", exptime)
        self.psfData.pprint("PSF FWHM", exptime)


def computeTrendMetrics(
    stars: pd.DataFrame,
    timeCol: str,
    yCol: str,
    expid: int,
) -> MetricResult:
    """
    Compute robust linear trend metrics for a given measurement column
    versus time within a single exposure.

    The function fits a robust linear model to the specified `yCol`
    as a function of `timeCol` for rows matching the given `expid`.
    It returns a `MetricResult` containing the slope, intercept, trend
    RMSE, robust global scatter, outlier fraction, slope significance,
    and sample size.

    Parameters
    ----------
    stars : `pandas.DataFrame`
        Table of star measurements containing at least the columns
        `timeCol`, `yCol`, and `expid`.
    timeCol : `str`
        Name of the time column (e.g., ``"elapsed_time"``).
    yCol : `str`
        Name of the dependent variable column to fit
        (e.g., ``"dalt"``, ``"magoffset"``).
    expid : `int`
        Exposure identifier used to filter the rows.

    Returns
    -------
    metrics : `MetricResult`
        Dataclass containing the computed trend metrics.
        If there are no data in `yCol` after filtering,
        all fields are set to NaN/None and `nsize` is zero.

    Raises
    ------
    KeyError
        If `timeCol` or `yCol` are not present in `stars`.
    """
    if timeCol not in stars.columns or yCol not in stars.columns:
        raise KeyError(f"Columns '{timeCol}' and/or '{yCol}' not found in stars DataFrame.")

    s = stars.loc[stars["expid"].eq(expid), [timeCol, yCol]].dropna()
    if s.empty or s[yCol].nunique() < 2:
        return MetricResult(
            slope=np.nan,
            intercept=np.nan,
            trend_rmse=np.nan,
            global_std=np.nan,
            outlier_frac=np.nan,
            slope_significance=None,
            nsize=0,
        )

    xArr = s[timeCol].to_numpy()
    yArr = s[yCol].to_numpy()
    stdGlobal = float(mad_std(yArr))
    fitter = RobustFitter(xArr, yArr)

    coefs = fitter.reportBestValues()
    inlierMask = ~fitter.outlier_mask
    slopeT = getattr(coefs, "slope_tvalue", None)
    slope_sig = abs(float(slopeT)) if slopeT is not None else None
    return MetricResult(
        slope=float(coefs.slope),
        intercept=float(coefs.intercept),
        trend_rmse=float(coefs.scatter),
        global_std=stdGlobal,
        outlier_frac=1.0 - (inlierMask.sum() / len(xArr)),
        slope_significance=slope_sig,
        nsize=int(yArr.size),  # or int(inlierMask.sum()) if you prefer inliers
    )


@dataclass(slots=True)
class MetricResult:
    """
    Container for trend metrics of a single quantity y(t) in an
    exposure.

    Holds the results of fitting a robust linear trend to a
    measurement column (`yCol`) as a function of time (`timeCol`),
    along with descriptive statistics for the distribution and
    fit quality.

    Parameters
    ----------
    slope : `float`
        Best-fit slope dy/dx from robust regression, in `yCol` units
        per unit time.
    intercept : `float`
        Best-fit intercept (y at x = 0) from robust regression, in
        `yCol` units.
    trend_rmse : `float`
        RMS of residuals from the fitted trend, in `yCol` units.
    global_std : `float`
        Robust global standard deviation of `yCol`, ignoring the
        trend.
    outlier_frac : `float`
        Fraction of data points flagged as outliers by the fitter.
    slope_significance : `float` or `None`
        t-statistic for the slope significance, or `None` if not
        available.
    nsize : `int`
        Number of valid points used in the fit.
    units : `str`
        Units of the dependent variable. Empty string if none.
    """

    slope: float
    intercept: float
    trend_rmse: float
    global_std: float
    outlier_frac: float
    slope_significance: float | None = None
    nsize: int = 0
    units: str = ""

    def pprint(self, title: str, exptime: float = 1.0) -> None:
        """
        Print the stored metrics in a formatted, human-readable block.

        Parameters
        ----------
        title : `str`
            Title to display for the metric block.
        exptime : `float`, optional
            Exposure time in seconds. The slope will be multiplied by this
            value to convert from per-second to per-exposure units.
        """
        if not title:
            title = "Metric"
        units = self.units
        header = makeHeader(f"Metrics Summary: {title}", nchar=40)
        print("\n".join(header))
        slope_per_exp = self.slope * exptime
        print(f"  Slope          : {slope_per_exp:.3f} {units} per exposure")
        sig = "—" if self.slope_significance is None else f"{self.slope_significance:.1f}"
        print(f"  Slope signif.  : {sig} sigma")
        print(f"  Intercept      : {self.intercept:.3f} {units}")
        print(f"  Trend RMSE     : {self.trend_rmse:.3f} {units}")
        print(f"  Global std     : {self.global_std:.3f} {units}")
        print(f"  Outlier frac   : {self.outlier_frac:.2%}")
        print(f"  N (points)     : {self.nsize:d}\n")

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
        prefix = prefix
        resDict = asdict(self)
        keep = (
            "slope",
            "intercept",
            "trend_rmse",
            "global_std",
            "outlier_frac",
            "slope_significance",
            "nsize",
        )
        row = _prefixKeys({k: resDict[k] for k in keep}, prefix)
        return pd.DataFrame([row], index=[index])


def mergeMetricResults(pairs: list[tuple[str, "MetricResult"]], index: int) -> pd.DataFrame:
    """
    Merge multiple `MetricResult` objects into a single-row DataFrame
    with prefixed column names.

    Each `MetricResult` is converted to a dictionary, filtered to keep
    only a standard set of metric fields, and all keys are prefixed
    with the provided label (e.g., ``"alt_drift"``, ``"mag"``) to
    distinguish different measurement types. The merged results are
    returned as a single-row DataFrame.

    Parameters
    ----------
    pairs : `list` of `tuple` of (`str`, `MetricResult`)
        List of (prefix, metric result) pairs. The prefix is used as
        the column name prefix for that result's metrics.
    index : `int`
        Index value for the returned DataFrame row.

    Returns
    -------
    merged : `pandas.DataFrame`
        Single-row DataFrame containing all merged metrics with column
        names of the form ``<prefix>_<metric_field>``.
    """
    row: dict[str, float | int | None] = {}
    keep = ("slope", "intercept", "trend_rmse", "global_std", "outlier_frac", "slope_significance", "nsize")
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
    }
    row.update(guidersPresent)
    return pd.DataFrame([row], index=[expid])


def printExposureCounts(countsDf: pd.DataFrame, precision: int = 3) -> None:
    """
    Print exposure-level counts from a single-row counts DataFrame.

    Parameters
    ----------
    countsDf : `pandas.DataFrame`
        DataFrame with a single row containing exposure counts
        (e.g., n_guiders, n_stars, per-guider flags).
    precision : `int`, optional
        Number of decimal places for fractional values.

    Returns
    -------
    None
    """
    nchar = 40
    if countsDf.empty:
        print("Exposure counts ─\nNo data available.")
        return

    # Get first row as dict for easy .get()
    row = countsDf.iloc[0].to_dict()

    lines = makeHeader("Exposure Counts", nchar=nchar)
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

    The keys of the input dictionary are prefixed with the provided
    string and an underscore. This is typically used to namespace
    metric names by a measurement type or category.

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

    The header consists of a top line, a centered title line with
    padding, and a bottom line. The line width is the maximum of
    `nchar` and the title length plus 10 characters.

    Parameters
    ----------
    title : `str`
        Text to display in the header.
    nchar : `int`, optional
        Minimum width of the horizontal lines. Increased automatically
        if the title requires more space.

    Returns
    -------
    header_lines : `list` of `str`
        List of three strings: the top line, the title line, and the
        bottom line.
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
