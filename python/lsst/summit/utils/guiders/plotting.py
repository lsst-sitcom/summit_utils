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
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.stats import mad_std
from matplotlib import animation
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from lsst.summit.utils.utils import RobustFitter
from lsst.utils.plotting.figures import make_figure

if TYPE_CHECKING:
    from .reading import GuiderData

__all__ = ["GuiderDataPlotter", "GuiderPlotter"]

LIGHT_BLUE = "#6495ED"


@dataclass(frozen=True)
class MosaicLayout:
    grid: List[Tuple[str, ...]] = field(
        default_factory=lambda: [
            (".", "R40_SG1", "R44_SG0", "."),
            ("R40_SG0", "center", ".", "R44_SG1"),
            ("R00_SG1", ".", ".", "R04_SG0"),
            ("arrow", "R00_SG0", "R04_SG1", "."),
        ]
    )

    def build(
        self,
        *,
        figsize: Tuple[float, float] = (12, 12),
        hspace: float = 0.0,
        wspace: float = 0.0,
        constrained_layout: bool = False,
    ) -> tuple[plt.Figure, dict[str, plt.Axes]]:
        # TODO: use DM code to make figure mosaic
        fig, axs = plt.subplot_mosaic(
            cast(Any, self.grid),
            figsize=figsize,
            gridspec_kw=dict(hspace=hspace, wspace=wspace),
            constrained_layout=constrained_layout,
            sharex=False,  # for the mosaic, do not share axes
            sharey=False,  # for the mosaic, do not share axes
        )
        return fig, axs


stripPlotKwargs: dict[str, dict] = {
    "centroidAltAz": {
        "ylabel": "Centroid Offset [arcsec]",
        "unit": "arcsec",
        "col": ["dalt", "daz"],
        "title": "Alt/Az Centroid Offsets",
    },
    "centroidPixel": {
        "ylabel": "Centroid Offset [pixels]",
        "col": ["dx", "dy"],
        "unit": "pixels",
        "title": "CCD Pixel Centroid Offsets",
    },
    "flux": {
        "ylabel": "Magnitude Offset [mag]",
        "col": ["magoffset"],
        "unit": "mmag",
        "scale": 1e3,  # scale to mmag
        "title": "Flux Magnitude Offsets",
    },
    "ellip": {
        "ylabel": "Ellipticity",
        "col": ["e1_altaz", "e2_altaz"],
        "unit": "",
        "title": "",
    },
    "psf": {
        "ylabel": "PSF FWHM [arcsec]",
        "col": ["fwhm"],
        "scale": 1.0,
        "unit": "arcsec",
        "title": "PSF FWHM",
    },
}


class GuiderPlotter:
    # for plotting
    LAYOUT = 0
    # The MARKERS lists are used for plotting different detector
    MARKERS: list[str] = ["o", "x", "+", "*", "^", "v", "s", "p"]

    def __init__(self, starsDf: pd.DataFrame, guiderData: GuiderData) -> None:
        self.log = logging.getLogger(__name__)
        self.expId = guiderData.expid
        self.layout: MosaicLayout = MosaicLayout()

        if starsDf.empty:
            raise ValueError("starsDf is empty. No data to plot.")

        self.starsDf = starsDf[starsDf["expid"] == self.expId].reset_index(drop=True)
        self.guiderData = guiderData

        # Some metadata information
        self.expTime = self.guiderData.guiderDurationSec
        self.camRotAngle = self.guiderData.camRotAngle

        # assemble statistics
        self.statsDf = assembleStats(self.starsDf)

        # set seaborn style
        sns.set_style("white")
        sns.set_context("talk", font_scale=0.8)

    def setupFigure(self, figsize: Tuple[float, float] = (12, 12)) -> tuple[plt.Figure, dict[str, plt.Axes]]:
        """
        Setup a figure and axes for the guider mosaic layout.
        """
        fig, axs = self.layout.build(figsize=figsize)
        return fig, axs

    def printMetrics(self) -> None:
        """
        Print formatted metrics and statistics for the current exposure.
        """
        if self.statsDf.empty:
            self.log.warning("No statistics available for this exposure.")
            return

        filteredStatsDf = self.statsDf[self.statsDf["expid"] == self.expId]

        print(self.formatStatsSummary(filteredStatsDf))
        print(self.formatStdCentroidSummary(filteredStatsDf, self.expTime))
        print(self.formatPhotometricSummary(filteredStatsDf, self.expTime))
        print(self.formatPsfSummary(filteredStatsDf, self.expTime))

    def stripPlot(
        self, plotType: str = "centroidAltAz", saveAs: Optional[str] = None, coveragePct: int = 68
    ) -> plt.Figure:
        """
        Plot time-series strip plot for a chosen metric.

        This renders one or more panels vs elapsed time, fits a robust
        linear trend, annotates slope/significance/scatter, and draws
        reference zero lines or median PSF as needed.

        Parameters
        ----------
        plotType : str, optional
            Metric key in `stripPlotKwargs` (e.g., 'centroidAltAz').
        saveAs : str or None, optional
            If set, save the figure to this path.
        coveragePct : int, optional
            Central coverage percent for y-limits (e.g., 68 → 16–84).

        Returns
        -------
        stripFig : Matplotlib Figure.
        """
        cfg = stripPlotKwargs.get(plotType)
        if cfg is None:
            raise ValueError(f"Unknown plotType: {plotType}")
        # from here, tell mypy it’s a dict[str, Any]
        cfg = cast(dict[str, Any], cfg)

        n = len(cfg["col"])
        fig = make_figure(figsize=(8 * n, 6))
        axes = fig.subplots(nrows=1, ncols=n, sharex=True, sharey=True)
        axes = np.atleast_1d(axes)

        cols = cfg["col"]
        scale = float(cfg.get("scale", 1.0))
        unit = cfg.get("unit", "")
        expTime = float(self.expTime)

        df = self.starsDf.loc[self.starsDf["stamp"] > 0, ["elapsed_time", "detector", *cols]].copy()

        q1, q3 = (100 - coveragePct) / 2, 100 - (100 - coveragePct) / 2
        yvals = (df[cols].to_numpy(dtype=float) * scale).ravel()
        p16, p84 = np.nanpercentile(yvals, [q1, q3])
        sigma = mad_std(yvals, ignore_nan=True)
        ylims = (p16 - 2.5 * sigma, p84 + 2.5 * sigma)

        def _zero(ax, c):
            label = {
                "daz": "Az",
                "dalt": "Alt",
                "dx": "CCD X",
                "dy": "CCD Y",
                "e1_altaz": "e1",
                "e2_altaz": "e2",
                "magoffset": "Magnitude Offset",
            }.get(c, "")
            ax.axhline(0 if c != "fwhm" else np.nanmedian(df[c] * scale), color="grey", ls="--", label=label)

        for i, (ax, c) in enumerate(zip(axes, cols)):
            _zero(ax, c)
            fitter = RobustFitter(x=df["elapsed_time"].values, y=(df[c].values * scale))
            res = fitter.reportBestValues()
            txt = (
                f"Slope: {expTime * res.slope:.2f} {unit}/exposure\n"
                f"Significance: {abs(res.slope_tvalue):.1f} σ\n"
                f"scatter: {res.scatter:.3f} {unit}\n"
            )
            ax.text(
                0.02,
                0.98,
                txt,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", boxstyle="round,pad=0.3"),
            )
            fitter.plotBestFit(ax=ax, color=LIGHT_BLUE, label="", lw=2)

            for j, det in enumerate(df["detector"].unique()):
                m = self.MARKERS[j % len(self.MARKERS)]
                msk = df["detector"].eq(det)
                ax.scatter(
                    df.loc[msk, "elapsed_time"],
                    df.loc[msk, c] * scale,
                    color="lightgrey",
                    alpha=0.6,
                    marker=m,
                    label=f"{det}" if i == 0 else "",
                )
                out = msk & fitter.outlier_mask
                ax.scatter(
                    df.loc[out, "elapsed_time"], df.loc[out, c] * scale, color=LIGHT_BLUE, alpha=0.2, marker=m
                )

            if i == 0:
                ax.set_ylabel(cfg["ylabel"])
            ax.set_xlabel("Elapsed time [sec]")
            ax.set_ylim(*ylims)
            ax.legend(fontsize=10, ncol=4, loc="lower left")

        fig.suptitle(cfg["title"] or "", fontsize=14, fontweight="bold")
        if saveAs:
            fig.savefig(saveAs, dpi=120)
        return fig

    def starMosaic(
        self,
        stampNum: int = 2,
        fig: plt.Figure | None = None,
        axs: Optional[dict[str, plt.Axes]] = None,
        plo: float = 90.0,
        phi: float = 99.0,
        cutoutSize: int = 30,
        isAnimated: bool = False,
        saveAs: Optional[str] = None,
    ) -> list[Artist]:
        """
        Plot mosaic of guider stamps.

        Parameters
        ----------
        stampNum : int, optional
            Stamp index; &lt;0 uses stacked image.
        fig : Figure or None, optional
            Figure to draw on; creates one if None.
        axs : dict[str, Axes] or None, optional
            Axes dict for the mosaic; creates if None.
        plo : float, optional
            Lower percentile for scaling.
        phi : float, optional
            Upper percentile for scaling.
        cutoutSize : int, optional
            Cutout size per panel; -1 uses full frame.
        isAnimated : bool, optional
            If True, skip static overlays.
        saveAs : str or None, optional
            If set, save figure to this path.

        Returns
        -------
        artists : list
            Matplotlib artists added to the figure.
        """
        nStamps = len(self.guiderData)
        view = self.guiderData.view
        camAngle = self.guiderData.camRotAngle

        if fig is None or axs is None:
            fig, axs = self.setupFigure(figsize=(9, 9))

        artists: list[Artist] = []
        cutoutShapeList = []
        for detName in self.guiderData.guiderNames:
            # check if detector has data
            if not np.any(self.starsDf["detector"] == detName):
                shape = self.guiderData[detName, 0].shape
                refCenter = (float(shape[1] // 2), float(shape[0] // 2))
                centroidOffset = (0.0, 0.0)
            else:
                refCenter, centroidOffset = getReferenceCenter(self.starsDf, detName, stampNum)

            # Render the stamp panel
            imObj, centerCutout, shape, _ = renderStampPanel(
                axs[detName],
                self.guiderData,
                detName,
                stampNum,
                center=refCenter,
                cutoutSize=cutoutSize,
                plo=plo,
                phi=phi,
                annotate=True,
            )
            # Static overlays (when not animating)
            if not isAnimated:
                addStaticOverlays(axs[detName], detName, centerCutout, cutoutSize, camRotAngle=camAngle)

            starCross = plotStarCentroid(
                axs[detName],
                centerCutout,
                deltaXY=centroidOffset,
                markerSize=8 if cutoutSize > 0 else 0,
            )
            artists.append(imObj)
            artists.extend(starCross)
            cutoutShapeList.append(shape)

        jitter = getStdCentroid(self.statsDf, self.expId)
        stampInfo = annotateStampInfo(
            axs["center"], expid=self.expId, stampNum=stampNum, nStamps=nStamps, view=view, jitter=jitter
        )
        artists.append(stampInfo)

        cutoutSize = np.max(cutoutShapeList) if cutoutShapeList else 30

        if not isAnimated:
            drawArrows(axs["arrow"], cutoutSize, 90.0 + self.camRotAngle)

        for ax in axs.values():
            clearAxisTicks(ax, isSpine=cutoutSize < 0)

        if saveAs:
            fig.savefig(saveAs, dpi=120)

        return artists

    def makeGif(
        self,
        saveAs: str,
        nStampMax: int = 60,
        fps: int = 5,
        dpi: int = 100,
        plo: float = 90.0,
        phi: float = 99.0,
        cutoutSize: int = 30,
    ) -> animation.ArtistAnimation:
        """
        Create an animated GIF of the guider mosaic over stamps.

        Parameters
        ----------
        saveAs : str
            Output filepath for the GIF.
        nStampMax : int, optional
            Max number of stamps to animate.
        fps : int, optional
            Frames per second.
        dpi : int, optional
            Output dots per inch.
        plo : float, optional
            Lower percentile for image scaling.
        phi : float, optional
            Upper percentile for image scaling.
        cutoutSize : int, optional
            Cutout size per panel; -1 uses full frame.

        Returns
        -------
        matplotlib.animation.ArtistAnimation
            The created animation object.
        """
        # build canvas
        fig, axs = self.setupFigure(figsize=(10, 10))

        if self.starsDf.empty:
            self.log.warning("stars_df is empty. No data to plot.")
            return animation.ArtistAnimation(fig, [], interval=1000 / fps, blit=True, repeat_delay=1000)

        # number of frames
        nStamps = len(self.guiderData)
        total = min(nStamps, nStampMax)

        # initial (stacked) frame
        artists0 = self.starMosaic(
            stampNum=-1,
            fig=fig,
            axs=axs,
            plo=plo,
            phi=phi,
            cutoutSize=cutoutSize,
            isAnimated=False,
        )

        frames = 2 * [artists0]

        # sequential stamps
        for i in range(1, total):
            artists = self.starMosaic(
                stampNum=i,
                fig=fig,
                axs=axs,
                plo=plo,
                phi=phi,
                cutoutSize=cutoutSize,
                isAnimated=True,
            )
            frames.append(artists)
        frames += 2 * [artists0]

        # create animation
        ani = animation.ArtistAnimation(fig, frames, interval=1000 / fps, blit=True, repeat_delay=1000)

        ani.save(saveAs, fps=fps, dpi=dpi, writer="pillow")
        plt.close(fig)
        return ani

    @staticmethod
    def formatStdCentroidSummary(statsDf: pd.DataFrame, expTime: float = 1) -> str:
        """
        Pretty string summary of centroid stdev. stats from run_all_guiders.
        """
        # handle both dicts and DataFrames
        if (isinstance(statsDf, pd.DataFrame) and statsDf.empty) or (
            isinstance(statsDf, dict) and not statsDf
        ):
            return "No centroid stdev. statistics available."

        # if it's a DataFrame, extract the one-row dict
        if isinstance(statsDf, pd.DataFrame):
            stats = statsDf.iloc[0].to_dict()
        else:
            stats = statsDf

        js = stats
        summary = (
            f"\nGlobal centroid stdev. Summary Across All Guiders\n"
            f"{'-' * 45}\n"
            f"  - centroid stdev.  (AZ): {js['std_centroid_az']:.3f} arcsec (raw)\n"
            f"  - centroid stdev. (ALT): {js['std_centroid_alt']:.3f} arcsec (raw)\n"
            f"  - centroid stdev.  (AZ): {js['std_centroid_corr_az']:.3f} arcsec (linear corr)\n"
            f"  - centroid stdev. (ALT): {js['std_centroid_corr_alt']:.3f} arcsec (linear corr)\n"
            f"  - Drift Rate       (AZ): {expTime * js['drift_rate_az']:.2f} arcsec per exposure\n"
            f"  - Drift Rate      (ALT): {expTime * js['drift_rate_alt']:.2f} arcsec per exposure\n"
            f"  - Zero Offset      (AZ): {js['offset_zero_az']:.3f} arcsec\n"
            f"  - Zero Offset     (ALT): {js['offset_zero_alt']:.3f} arcsec"
        )
        return summary

    @staticmethod
    def formatPhotometricSummary(photStats: pd.DataFrame, expTime: float = 1) -> str:
        """
        Pretty-print summary of photometric variation statistics.
        """
        if (isinstance(photStats, pd.DataFrame) and photStats.empty) or (
            isinstance(photStats, dict) and not photStats
        ):
            return "No photometric statistics available."

        # if it's a DataFrame, extract the one-row dict
        if isinstance(photStats, pd.DataFrame):
            stats = photStats.iloc[0].to_dict()
        else:
            stats = photStats

        return (
            "\nPhotometric Variation Summary\n"
            "-----------------------------\n"
            f"  - Mag Drift Rate:      {expTime * stats['magoffset_rate'] * 1e3:.1f} mmag/exposure\n"
            f"  - Mag Zero Offset:     {stats['magoffset_zero'] * 1e3 :.1f} mmag\n"
            f"  - Mag RMS (detrended): {stats['magoffset_rms'] * 1e3:.1f} mmag"
        )

    @staticmethod
    def formatStatsSummary(summary: pd.DataFrame) -> str:
        """
        Pretty-print only the stats that are present in `summary`.
        Expects keys like:
          n_guiders, n_unique_stars, n_measurements, fraction_valid_stamps,
          N_<detector>, std_centroid_*, magoffset_*, etc.
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
        return "\n".join(lines)

    @staticmethod
    def formatPsfSummary(psfStats: pd.DataFrame, expTime: float = 1) -> str:
        """
        Pretty-print summary of PSF statistics.
        """
        if (isinstance(psfStats, pd.DataFrame) and psfStats.empty) or (
            isinstance(psfStats, dict) and not psfStats
        ):
            return "No PSF statistics available."

        if isinstance(psfStats, pd.DataFrame):
            psfStats = psfStats.iloc[0].to_dict()

        lines = ["", "-" * 50]
        lines.append("PSF Statistics Summary:")
        lines.append(f"  - FWHM slope   : {expTime * psfStats['fwhm_rate']:.3f} arcsec per exposure")
        lines.append(f"  - FWHM scatter : {psfStats['fwhm_rms']:.3f} arcsec")
        return "\n".join(lines)


class GuiderDataPlotter:
    """
    Class to plot guider stamp mosaics using a GuiderData container.
    Plot an animated gif of the CCD guider stamp.
    """

    def __init__(self, guiderData: GuiderData):
        """
        Initialize the GuiderMosaicPlotter with a GuiderData container.

        Parameters
        ----------
        guiderData : GuiderData
            GuiderData container with guider images and metadata.

        Returns
        -------
        None
        """
        self.guiderData = guiderData
        self.view = guiderData.view
        self.expId = guiderData.expid
        self.nStamps = len(guiderData)
        self.detNames = guiderData.guiderNames
        self.header = guiderData.header

        # get the last 5 digits of expId as seqNum
        self.seqNum = int(str(self.expId)[-5:])
        # get the dayObs from the first 8 digits of expId
        self.dayObs = int(str(self.expId)[:8])
        self.mosaic = MosaicLayout()

    def setupFigure(self, figsize: Tuple[float, float] = (12, 12)) -> tuple[plt.Figure, dict[str, plt.Axes]]:
        """
        Setup a figure and axes for the guider mosaic layout.
        """
        fig, axs = self.mosaic.build(figsize=figsize, hspace=0.0, wspace=0.0)
        return fig, axs

    def plotStampCcd(
        self,
        axs: plt.Axes,
        detName: str,
        stampNum: int = -1,
        plo: float = 50.0,
        phi: float = 99.0,
        is_ticks: bool = False,
    ) -> plt.AxesImage:
        """
        Plot a stamp or stacked image for a single CCD.
        """
        im, _, _, _ = renderStampPanel(
            axs, self.guiderData, detName, stampNum, plo=plo, phi=phi, annotate=True
        )
        if not is_ticks:
            clearAxisTicks(axs, isSpine=True)
        return im

    def plotStampArray(self, stampNum=0, fig=None, axs=None, plo=90.0, phi=99.0, isAnimated=False) -> list:
        """
        Plot the stamp array for all the guiders in a mosaic layout.

        Parameters
        ----------
        stampNum : int, optional
            Stamp number.
        fig : matplotlib.figure.Figure, optional
            Figure object.
        axs : dict, optional
            Dictionary of axes for the mosaic.
        plo : float, optional
            Lower percentile for image scaling.
        phi : float, optional
            Upper percentile for image scaling.
        isAnimated : bool, optional
            Whether this is part of an animation.

        Returns
        -------
        Artists : list
            List of matplotlib artist objects.
        """
        if fig is None:
            fig, axs = self.setupFigure(figsize=(9.5, 9.5))

        artists = []
        for detName in self.detNames:
            im, _, _, txt = renderStampPanel(
                axs[detName], self.guiderData, detName, stampNum, plo=plo, phi=phi, annotate=True
            )
            artists.extend([im, txt])

        nStamps, view = self.nStamps, self.view
        stampInfo = annotateStampInfo(
            axs["center"], expid=self.expId, stampNum=stampNum, nStamps=nStamps, view=view
        )
        artists.append(stampInfo)

        # add coordinate arrow
        cutoutSize = max(*self.guiderData[0].shape)
        if not isAnimated:
            camRotAngle = cast(float, self.guiderData.camRotAngle)
            drawArrows(axs["arrow"], cutoutSize, 90.0 + camRotAngle)

        for ax in axs.values():
            clearAxisTicks(ax, isSpine=True)
        clearAxisTicks(axs["center"], isSpine=False)
        return artists

    def plotStackedStampArray(self, fig=None, axs=None, plo=50, phi=99) -> list:
        """
        Plot the stacked stamp array for all the guiders.
        """
        artists = self.plotStampArray(stampNum=-1, fig=fig, axs=axs, plo=plo, phi=phi)
        return artists

    def makeGif(
        self, fps=5, dpi=80, saveAs=None, plo=50, phi=99, figsize=(9, 9)
    ) -> animation.ArtistAnimation:
        """
        Create an animated GIF of the guider CCD array
        over a sequence of stamps.

        Parameters
        ----------
        fps : int, optional
            Frames per second.
        dpi : int, optional
            Dots per inch for saved animation.
        saveAs : str or None, optional
            If provided, save the animation to this file.

        Returns
        -------
        matplotlib.animation.ArtistAnimation
            The resulting animation object.
        """
        # Create the animation
        fig, axs = self.setupFigure(figsize=figsize)
        nStamps = self.nStamps

        # plot the stacked image
        artists0 = self.plotStackedStampArray(fig=fig, axs=axs, plo=plo, phi=phi)
        frameList = 5 * [artists0]

        # loop over the stamps
        for i in range(1, nStamps - 1):
            artists = self.plotStampArray(stampNum=i, fig=fig, axs=axs, plo=plo, phi=phi, isAnimated=True)
            frameList.append(artists)

        frameList += 5 * [artists0]

        # update the stamps
        ani = animation.ArtistAnimation(fig, frameList, interval=1000 / fps, blit=True, repeat_delay=1000)

        ani.save(saveAs, fps=fps, dpi=dpi, writer="pillow")
        return ani

    def makeMp4(self, fps=5, dpi=80, saveAs="guider_ccd_array.mp4") -> animation.ArtistAnimation:
        """
        Create an MP4 animation of the guider CCD array
        over a sequence of stamps.

        Parameters
        ----------
        fps : int, optional
            Frames per second.
        dpi : int, optional
            Dots per inch for saved animation.
        saveAs : str, optional
            Output filename for the MP4 file.

        Returns
        -------
        matplotlib.animation.ArtistAnimation
            The resulting animation object.
        """
        # Create the animation
        fig, axs = self.setupFigure(figsize=(9, 9))
        nStamps = self.nStamps

        # plot the stacked image
        artists0 = self.plotStackedStampArray(fig=fig, axs=axs)
        frameList = 5 * [artists0]

        # loop over the stamps
        for i in range(nStamps):
            artists = self.plotStampArray(stampNum=i, fig=fig, axs=axs)
            frameList.append(artists)
        frameList += 5 * [artists0]

        # update the stamps
        ani = animation.ArtistAnimation(fig, frameList, interval=1000 / fps, blit=True, repeat_delay=1000)
        ani.save(saveAs, fps=fps, dpi=dpi)
        plt.close(fig)
        return ani


def getStdCentroid(statsDf: pd.DataFrame, expId: int) -> float:
    stdAz = statsDf.loc[statsDf["expid"] == expId, "std_centroid_corr_az"].values[0]
    stdAlt = statsDf.loc[statsDf["expid"] == expId, "std_centroid_corr_alt"].values[0]
    return np.hypot(stdAz, stdAlt)


def drawArrows(
    ax: plt.Axes,
    cutoutSize: int,
    rotAngle: float = 0.0,
    *,
    baseLabels: tuple[str, str] = ("Y DVCS", "X DVCS"),
    overlayLabels: tuple[str, str] = ("Alt", "Az"),
    baseColor: str = LIGHT_BLUE,
    overlayColor: str = "lightgrey",
    center: tuple[float, float] | None = None,
) -> None:
    """
    Draw DVCS (θ=0) and Alt/Az (θ=rotAngle) reference arrows.
    """
    x0, y0 = (cutoutSize // 2, cutoutSize // 2) if center is None else center
    L = cutoutSize / 3.0

    def _draw(theta_deg: float, color: str, labels: tuple[str, str]) -> np.ndarray:
        t = np.radians(theta_deg)
        # unit vectors rotated by t
        dx_az, dy_az = L * np.cos(t), L * np.sin(t)  # +X (Az)
        dx_alt, dy_alt = -L * np.sin(t), L * np.cos(t)  # +Y (Alt)

        for dx, dy, label in ((dx_az, dy_az, labels[1]), (dx_alt, dy_alt, labels[0])):
            coords = (x0, y0, dx, dy)
            ax.arrow(
                *coords,
                color=color,
                width=cutoutSize / 120,
                head_width=cutoutSize / 30,
                head_length=cutoutSize / 20,
                length_includes_head=True,
                zorder=10,
            )
            ax.text(x0 + dx + 0.5, y0 + dy, label, color=color, fontsize=6, fontweight="bold")
        return np.array([min(dx_az, dx_alt), min(dy_az, dy_alt)])

    # Base DVCS axes (θ=0), then rotated Alt/Az overlay
    _ = _draw(0.0, baseColor, baseLabels)
    deltas = _draw(rotAngle, overlayColor, overlayLabels)
    deltas = np.where(deltas > 0, 0, deltas)  # only negative shifts

    # Framing
    ax.set_aspect("equal", adjustable="box")
    border = cutoutSize * 0.15
    xmin = min(x0 - border, x0 + deltas[0] - border)
    ymin = min(y0 - border, y0 + deltas[1] - border)

    ax.set_xlim(xmin - border, xmin + cutoutSize + border)
    ax.set_ylim(ymin - border, ymin + cutoutSize + border)
    clearAxisTicks(ax, isSpine=True)
    ax.set_axis_off()


def getReferenceCenter(
    starsDf: pd.DataFrame,
    detName: str,
    stampNum: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Return ((refX, refY), (dX, dY)) for a detector/stamp.
    If no measurement for that stamp or stampNum < 0 (stacked),
    the offset is (0, 0).
    """
    mask1 = starsDf["detector"] == detName
    if not mask1.any():
        raise ValueError(f"No rows for detector {detName!r}")

    # Reference center (use .iloc[0] to avoid IndexError)
    refX = float(starsDf.loc[mask1, "xroi_ref"].iloc[0])
    refY = float(starsDf.loc[mask1, "yroi_ref"].iloc[0])

    mask2 = starsDf["stamp"] == stampNum
    mask = mask1 & mask2

    # Fallback: no row for that stamp or stacked request
    if (not mask.any()) or (stampNum < 0):
        return (refX, refY), (0.0, 0.0)

    row = starsDf.loc[mask, ["xroi", "yroi"]].iloc[0]
    dX = float(row["xroi"]) - refX
    dY = float(row["yroi"]) - refY
    return (refX, refY), (dX, dY)


def renderStampPanel(
    ax: plt.Axes,
    guiderData,
    detName: str,
    stampNum: int,
    *,
    center: tuple[float, float] | None = None,
    cutoutSize: int = -1,
    plo: float = 50.0,
    phi: float = 99.0,
    annotate: bool = True,
) -> tuple[plt.AxesImage, tuple[float, float], tuple[int, int], Optional[plt.Text]]:
    """
    1) Select stamp or coadd
    2) Optional crop around `center`
    3) Scale by percentiles
    4) imshow + equal aspect
    5) Optional detector label
    Returns: (imageArtist, centerCutout, cutoutShape, labelArtist)
    """
    # 1) image
    img = guiderData.getStampArrayCoadd(detName) if stampNum < 0 else guiderData[detName, stampNum]

    # 2) center + crop
    # Always produce concrete floats (fallback = image center)
    mx, my = img.shape[1] // 2, img.shape[0] // 2  # x=cols, y=rows
    if center is None:
        cx, cy = float(mx), float(my)
    else:
        cx_raw, cy_raw = center
        cx = float(cx_raw) if cx_raw is not None else float(mx)
        cy = float(cy_raw) if cy_raw is not None else float(my)

    if cutoutSize > 0:
        cutout, centerCutout = cropAroundCenter(img, (cx, cy), cutoutSize)
    else:
        cutout = img  # ndarray
        centerCutout = (cx, cy)  # tuple[float, float]

    # 3) scaling
    vmin, vmax = np.nanpercentile(cutout, [plo, phi])

    # 4) render
    im = ax.imshow(
        cutout,
        origin="lower",
        cmap="Greys",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        extent=(0, cutout.shape[1], 0, cutout.shape[0]),
        animated=True,
    )
    ax.set_aspect("equal", "box")

    # 5) optional label
    label = labelDetector(ax, detName) if annotate else None

    return im, centerCutout, cutout.shape, label


# ==== Axis / panel text annotations ====
def labelDetector(
    ax: plt.Axes,
    name: str,
    *,
    corner: str = "tl",  # "tl" | "tr" | "bl" | "br"
    color: str = "grey",
    fontsize: int = 9,
    weight: str = "bold",
    pad: tuple[float, float] = (0.025, 0.025),
) -> plt.Text:
    """Place the detector name on a CCD panel."""
    ha, va = ("left", "top") if corner[0] == "t" else ("left", "bottom")
    ha = "right" if corner[1] == "r" else ha
    va = "top" if corner[0] == "t" else va

    xpad, ypad = pad
    xpos = xpad if ha == "left" else 1 - xpad
    ypos = 1 - ypad if va == "top" else ypad

    txt = ax.text(
        xpos,
        ypos,
        name,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        weight=weight,
        color=color,
    )
    return txt


def annotateStampInfo(
    ax: plt.Axes,
    *,
    expid: int,
    stampNum: int,
    nStamps: int,
    view: str | None = None,
    jitter: float | None = None,  # arcsec
    extra: str | None = None,  # free-form extra text
    xy: tuple[float, float] = (1.085, -0.10),
) -> plt.Text:
    """Annotate center panel with exposure/stamp info text."""
    dayObs, seqNum = str(expid)[:8], int(str(expid)[-5:])

    text = ""
    if jitter is not None:
        text += f"Center Stdev.: {jitter:.2f} arcsec\n"
    if dayObs is not None and seqNum is not None and view is not None:
        text += f"day_obs: {dayObs}\nseq_num: {seqNum}\norientation: {view}\n"
    text += f"Stamp #: {stampNum + 1:02d}" if stampNum >= 0 else f"Stacked w/ {nStamps} stamps"
    if extra is not None:
        text += "\n" + extra

    txt = ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        color="grey",
    )
    clearAxisTicks(ax, isSpine=True)
    return txt


# ==== Overlays & guides ====


def addStaticOverlays(
    axsImg: plt.Axes,
    detName: str,
    centerCutout: tuple[float, float],
    cutoutSize: int,
    camRotAngle: float,
) -> None:
    """
    Add static overlays to a guider image axis, with detector annotation,
    crosshairs, and guide circles.
    """
    _ = labelDetector(axsImg, detName)
    axsImg.axvline(centerCutout[0], color=LIGHT_BLUE, lw=1.25, linestyle="--", alpha=0.75)
    axsImg.axhline(centerCutout[1], color=LIGHT_BLUE, lw=1.25, linestyle="--", alpha=0.75)
    plotCrosshairRotated(
        centerCutout,
        90 + camRotAngle,
        axs=axsImg,
        color="grey",
        size=cutoutSize,
    )
    radii = [10, 5] if cutoutSize > 0 else [10]
    _ = drawGuideCircles(
        axsImg,
        centerCutout,
        radii=radii,
        colors=[LIGHT_BLUE, LIGHT_BLUE],
        labels=["2″", "1″"],
        linewidth=2.0,
    )


def drawGuideCircles(
    ax: plt.Axes,
    center: tuple[float, float],
    radii: Sequence[float],
    colors: Sequence[str],
    labels: Optional[Sequence[str]] = None,
    text_offset: float = 1.0,
    **circle_kwargs: Any,
) -> list[plt.Text]:
    """
    Plot concentric guide circles on an axis. Each circle can have
    its own color and label, and all are centered at the specified position.
    """
    x0, y0 = center
    txt_list: list[plt.Text] = []
    for i, r in enumerate(radii):
        c = Circle(
            (x0, y0),
            r,
            edgecolor=colors[i],
            facecolor="none",
            linestyle="--",
            **circle_kwargs,
        )
        ax.add_patch(c)
        label_txt = labels[i] if labels is not None else ""
        txt = ax.text(
            x0 + r + text_offset,
            y0 - r / 4.0,
            label_txt,
            color=colors[i],
            va="center",
            fontsize=8,
        )
        txt_list.append(txt)
    return txt_list


def plotCrosshairRotated(
    center: tuple[float, float],
    angle: float,
    axs: plt.Axes | None = None,
    color: str = "grey",
    size: int = 30,
) -> None:
    """
    Plot a crosshair at a given center, rotated by the specified angle. The
    crosshair consists of two lines intersecting at the center.
    """
    if axs is None:
        axs = plt.gca()
    # make a cross rotated by the camera rotation angle
    cross_length = 1.5 * size if size > 0 else 30
    theta = np.radians(angle)

    # Cross center
    cx, cy = center
    # Horizontal line (rotated)
    dx = cross_length * np.cos(theta) / 2
    dy = cross_length * np.sin(theta) / 2
    axs.plot(
        [cx - dx, cx + dx],
        [cy - dy, cy + dy],
        color=color,
        ls="--",
        lw=1.0,
        alpha=0.5,
    )
    # Vertical line (rotated)
    dx_v = cross_length * np.cos(theta + np.pi / 2) / 2
    dy_v = cross_length * np.sin(theta + np.pi / 2) / 2
    axs.plot(
        [cx - dx_v, cx + dx_v],
        [cy - dy_v, cy + dy_v],
        color=color,
        ls="--",
        lw=1.0,
        alpha=0.5,
    )


def plotStarCentroid(
    ax: plt.Axes,
    centerCutout: tuple[float, float],
    *,
    deltaXY: tuple[float, float],
    markerSize: int = 8,
    errXY: tuple[float, float] | None = None,
    color: str = "firebrick",
) -> list[Line2D]:
    """
    Plot the star centroid on the cutout using explicit coordinates.
    """
    if markerSize <= 0:
        return []

    x_c, y_c = centerCutout
    dx, dy = deltaXY
    x_star, y_star = x_c + dx, y_c + dy
    xerr, yerr = errXY if errXY is not None else (0.0, 0.0)

    (marker,) = ax.plot(x_star, y_star, "o", color=color, markersize=markerSize)
    (hline,) = ax.plot([x_star - xerr, x_star + xerr], [y_star, y_star], color=color, lw=2.5)
    (vline,) = ax.plot([x_star, x_star], [y_star - yerr, y_star + yerr], color=color, lw=2.5)
    return [marker, hline, vline]


def clearAxisTicks(ax: plt.Axes, isSpine: bool = False) -> None:
    """
    Remove all ticks and tick labels from an axis.
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if not isSpine:
        for spine in ax.spines.values():
            spine.set_visible(False)


def cropAroundCenter(
    image: np.ndarray, center: tuple[float, float], size: int
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Crop a square region from the image centered at a specific point,
    returning both the cropped image and the new center coordinates.

    Returns
    -------
    cropped : np.ndarray
        Cropped image of shape (size, size).
    new_center : tuple[float, float]
        Coordinates of the original center in the cropped image.
    """
    x, y = center
    x, y = int(round(x)), int(round(y))
    half = size // 2

    y0 = max(0, y - half)
    y1 = min(image.shape[0], y + half)
    x0 = max(0, x - half)
    x1 = min(image.shape[1], x + half)

    cropped = image[y0:y1, x0:x1]

    # Calculate offsets for padding
    padTop = max(0, (half - y))
    padLeft = max(0, (half - x))
    padBottom = size - (padTop + (y1 - y0))
    padRight = size - (padLeft + (x1 - x0))

    if padTop > 0 or padBottom > 0 or padLeft > 0 or padRight > 0:
        cropped = np.pad(
            cropped,
            ((padTop, padBottom), (padLeft, padRight)),
            mode="constant",
            constant_values=np.nan,
        )

    # New center in cropped image
    newCenter = (x - x0 + padLeft, y - y0 + padTop)
    return cropped, newCenter


def assembleStats(stars) -> pd.DataFrame:
    """
    Assemble summary statistics from the stars dataframe
    for the current exposure.

    Returns
    -------
    pd.DataFrame
        DataFrame with a summary statistics for the current exposure.
    """
    nGuiders = stars["detector"].nunique()
    nUnique = stars["detid"].nunique()
    counts = stars.groupby("detector")["detid"].nunique().to_dict()
    guiderNames = stars["detector"].unique()
    starsPerGuiders = {f"N_{det}": counts.get(det, 0) for det in guiderNames}

    maskValid = (stars["stamp"] >= 0) & (stars["xccd"].notna())
    nMeas = int(maskValid.sum())

    stdCentroid = measureStdCentroidStats(stars, snr_th=5)
    phot = measurePhotometricVariation(stars, snr_th=5)
    psfStats = measurePsfStats(stars, snr_th=5)

    totalPossible = nUnique * stars["stamp"].nunique()
    fracValid = nMeas / totalPossible if totalPossible > 0 else np.nan

    summary = {
        "expid": stars["expid"].iloc[0],
        "n_guiders": nGuiders,
        "n_stars": nUnique,
        "n_measurements": nMeas,
        "fraction_valid_stamps": fracValid,
        "filter": stars["filter"].iloc[0],
        **starsPerGuiders,
        **stdCentroid,
        **phot,
        **psfStats,
    }
    return pd.DataFrame([summary])


def measureStdCentroidStats(stars: pd.DataFrame, snr_th: int = 5, flux_th: float = 10.0) -> pd.DataFrame:
    """
    Compute global std_centroid statistics across all guiders.

    Parameters
    ----------
    stars : pd.DataFrame
        Concatenated star table across all guider detectors.
    snr_th : int, optional
        SNR threshold for filtering stars.
    flux_th : float, optional
        Flux threshold for filtering stars.

    Returns
    -------
    dict
        Dictionary with std_centroid and related statistics.
    """
    # mask the objects w/ very low flux and SNR
    mask = (stars.snr > snr_th) & (stars.flux > flux_th)
    time = stars["elapsed_time"][mask].to_numpy()
    az = stars.daz[mask].to_numpy()
    alt = stars.dalt[mask].to_numpy()

    # Make a robust fit
    rf_alt = RobustFitter(time, alt)
    coefs_alt = rf_alt.reportBestValues()

    rf_az = RobustFitter(time, az)
    coefs_az = rf_az.reportBestValues()

    # Stats
    std_centroid_stats = {
        "std_centroid_az": mad_std(az),
        "std_centroid_alt": mad_std(alt),
        "std_centroid_corr_az": coefs_az.scatter,
        "std_centroid_corr_alt": coefs_alt.scatter,
        "drift_rate_az": coefs_az.slope,
        "drift_rate_alt": coefs_alt.slope,
        "drift_rate_signficance_az": coefs_az.slope_tvalue,
        "drift_rate_signficance_alt": coefs_alt.slope_tvalue,
        "offset_zero_az": np.nanmedian(az),
        "offset_zero_alt": np.nanmedian(alt),
    }
    return std_centroid_stats


def measurePsfStats(stars: pd.DataFrame, snr_th: int = 5, flux_th: float = 10) -> dict[str, float]:
    """
    Compute PSF statistics (FWHM and ellipticities) over all guiders.

    Parameters
    ----------
    stars : pd.DataFrame
        Concatenated star table across all guider detectors.
    snr_th : int, optional
        SNR threshold for filtering stars.
    flux_th : float, optional
        Flux threshold for filtering stars.

    Returns
    -------
    dict
        Dictionary with PSF and ellipticity statistics.
    """
    # mask the objects w/ very low flux and SNR
    mask = (stars.snr > snr_th) & (stars.flux > flux_th)
    time = stars["elapsed_time"][mask].to_numpy()
    psf = stars["fwhm"][mask].to_numpy()
    # Make a robust fit
    model = RobustFitter(time, psf)
    coefs = model.reportBestValues()

    # Elipiticities
    e1 = stars["e1_altaz"][mask].to_numpy()
    # Make a robust fit
    model = RobustFitter(time, e1)
    coefs_e1 = model.reportBestValues()

    e2 = stars["e2_altaz"][mask].to_numpy()
    # Make a robust fit
    model = RobustFitter(time, e2)
    coefs_e2 = model.reportBestValues()

    # Stats
    psf_stats = {
        "fwhm_rate": coefs.slope,
        "fwhm_signficance": coefs.slope_tvalue,
        "fwhm_zero": coefs.intercept,
        "fwhm_rms": coefs.scatter,
        "e1_rate": coefs_e1.slope,
        "e1_signficance": coefs_e1.slope_tvalue,
        "e1_zero": coefs.intercept,
        "e1_rms": coefs_e1.scatter,
        "e2_rate": coefs_e2.slope,
        "e2_signficance": coefs_e2.slope_tvalue,
        "e2_zero": coefs_e2.intercept,
        "e2_rms": coefs_e2.scatter,
    }
    return psf_stats


def measurePhotometricVariation(stars: pd.DataFrame, snr_th: int = 5) -> dict[str, float]:
    """
    Fit magoffset vs time across all rows, compute drift rate,
    zero-point, and RMS scatter.

    Parameters
    ----------
    stars : pd.DataFrame
        Star measurements across all guiders.
    snr_th : int, optional
        SNR threshold for filtering stars.

    Returns
    -------
    dict
        Dictionary with photometric variation statistics.
    """
    mo = stars["magoffset"].to_numpy()
    mask = np.isfinite(mo)
    mask &= stars["snr"] > snr_th
    if np.sum(mask) < 3:
        phot_stats = {
            "magoffset_rate": np.nan,
            "magoffset_zero": np.nan,
            "magoffset_rms": np.nan,
            "magoffset_signficance": np.nan,
        }
        return phot_stats

    time = stars["elapsed_time"][mask].to_numpy()
    rf_mag = RobustFitter(time, mo[mask])
    coefs_mag = rf_mag.reportBestValues()

    # Stats
    phot_stats = {
        "magoffset_rate": coefs_mag.slope,
        "magoffset_signficance": coefs_mag.slope_tvalue,
        "magoffset_zero": np.nanmedian(mo[mask]),
        "magoffset_rms": coefs_mag.scatter,
    }
    return phot_stats
