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
from typing import TYPE_CHECKING, Any, cast

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
    grid: list[tuple[str, ...]] = field(
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
        figsize: tuple[float, float] = (12, 12),
        hspace: float = 0.0,
        wspace: float = 0.0,
        constrained_layout: bool = False,
    ) -> tuple[plt.Figure, dict[str, plt.Axes]]:
        """
        Build the figure and axes dictionary for the predefined mosaic layout.

        Parameters
        ----------
        figsize : `tuple[float, float]`, optional
            Figure size in inches (width, height).
        hspace : `float`, optional
            Height space between subplots (gridspec hspace).
        wspace : `float`, optional
            Width space between subplots (gridspec wspace).
        constrained_layout : `bool`, optional
            Whether to enable Matplotlib constrained layout.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            The created figure.
        axs : `dict[str, matplotlib.axes.Axes]`
            Mapping from mosaic panel labels to axes.
        """
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

        # set seaborn style
        sns.set_style("white")
        sns.set_context("talk", font_scale=0.8)

    def setupFigure(self, figsize: tuple[float, float] = (12, 12)) -> tuple[plt.Figure, dict[str, plt.Axes]]:
        """
        Create a figure and axes using the guider mosaic layout.

        Parameters
        ----------
        figsize : `tuple[float, float]`, optional
            Figure size in inches (width, height).

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            The created figure.
        axs : `dict[str, matplotlib.axes.Axes]`
            Mapping of panel name to axes.
        """
        fig, axs = self.layout.build(figsize=figsize)
        return fig, axs

    def stripPlot(
        self, plotType: str = "centroidAltAz", saveAs: str | None = None, coveragePct: int = 68
    ) -> plt.Figure:
        """
        Plot time-series strip plot for a chosen metric.

        This renders one or more panels vs elapsed time, fits a robust
        linear trend, annotates slope/significance/scatter, and draws
        reference zero lines or median PSF as needed.

        Parameters
        ----------
        plotType : `str`, optional
            Metric key in `stripPlotKwargs` (e.g., 'centroidAltAz').
        saveAs : `str`, optional
            If not None, path to save the figure.
        coveragePct : `int`, optional
            Central percentile span used to derive y-limits (e.g. 68 -> 16–84).

        Returns
        -------
        stripFig : `matplotlib.figure.Figure`
            Figure containing the strip plot panels.
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
            fitter = RobustFitter()
            res = fitter.fit(x=df["elapsed_time"].values, y=(df[c].values * scale))
            txt = (
                f"Slope: {expTime * res.slope:.2f} {unit}/exposure\n"
                f"Significance: {abs(res.slopeTValue):.1f} σ\n"
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
                out = msk & fitter.outlierMask
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
        axs: dict[str, plt.Axes] | None = None,
        plo: float = 90.0,
        phi: float = 99.0,
        cutoutSize: int = 30,
        isAnimated: bool = False,
        saveAs: str | None = None,
    ) -> list[Artist]:
        """
        Plot a mosaic of guider stamps (a single stamp or a stacked image).

        Parameters
        ----------
        stampNum : `int`, optional
            Stamp index; values < 0 select the stacked (coadd) image.
        fig : `matplotlib.figure.Figure`, optional
            Existing figure to draw on; created if ``None``.
        axs : `dict[str, matplotlib.axes.Axes]`, optional
            Axes dictionary for the mosaic; created if ``None``.
        plo : `float`, optional
            Lower percentile for intensity scaling.
        phi : `float`, optional
            Upper percentile for intensity scaling.
        cutoutSize : `int`, optional
            Square cutout size around star center; -1 uses full frame.
        isAnimated : `bool`, optional
            If True, skip static overlays intended only for static displays.
        saveAs : `str`, optional
            If provided, path and filename to which the figure is saved.

        Returns
        -------
        artists : `list[matplotlib.artist.Artist]`
            Artists added for this mosaic frame.
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

        jitter = getStdCentroid(self.starsDf, self.expId)
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
        Create an animated GIF of the guider mosaic across sequential stamps.

        Parameters
        ----------
        saveAs : `str`
            Output filepath for the GIF.
        nStampMax : `int`, optional
            Maximum number of stamps to animate.
        fps : `int`, optional
            Frames per second.
        dpi : `int`, optional
            Output resolution in dots per inch.
        plo : `float`, optional
            Lower percentile for image scaling.
        phi : `float`, optional
            Upper percentile for image scaling.
        cutoutSize : `int`, optional
            Square cutout size; -1 means full frame.

        Returns
        -------
        ani : `matplotlib.animation.ArtistAnimation`
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


class GuiderDataPlotter:
    """
    Plot guider stamp mosaics and animations from a GuiderData instance.

    Parameters
    ----------
    guiderData : `GuiderData`
        Container with guider stamps and metadata.
    """

    def __init__(self, guiderData: GuiderData):
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

    def setupFigure(self, figsize: tuple[float, float] = (12, 12)) -> tuple[plt.Figure, dict[str, plt.Axes]]:
        """
        Create a figure and axes for the guider CCD mosaic.

        Parameters
        ----------
        figsize : `tuple[float, float]`, optional
            Figure size in inches.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Created figure.
        axs : `dict[str, matplotlib.axes.Axes]`
            Axes dictionary keyed by panel label.
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
        isTicks: bool = False,
    ) -> plt.AxesImage:
        """
        Plot a single CCD stamp (or stacked image) onto the provided axes.

        Parameters
        ----------
        axs : `matplotlib.axes.Axes`
            Target axes.
        detName : `str`
            Detector name.
        stampNum : `int`, optional
            Stamp index; values < 0 use stacked image.
        plo : `float`, optional
            Lower percentile for scaling.
        phi : `float`, optional
            Upper percentile for scaling.
        is_ticks : `bool`, optional
            If False, ticks are removed.

        Returns
        -------
        image : `matplotlib.image.AxesImage`
            Image artist.
        """
        im, _, _, _ = renderStampPanel(
            axs, self.guiderData, detName, stampNum, plo=plo, phi=phi, annotate=True
        )
        if not isTicks:
            clearAxisTicks(axs, isSpine=True)
        return im

    def plotStampArray(
        self,
        stampNum: int = 0,
        fig: plt.Figure | None = None,
        axs: dict[str, plt.Axes] | None = None,
        plo: float = 50,
        phi: float = 99,
        isAnimated: bool = False,
    ) -> list[Artist]:
        """
        Plot a mosaic of all guider stamps for a single stamp index.

        Parameters
        ----------
        stampNum : `int`, optional
            Stamp index; -1 for stacked image.
        fig : `matplotlib.figure.Figure`, optional
            Existing figure or None to create.
        axs : `dict`, optional
            Axes dictionary from mosaic builder.
        plo : `float`, optional
            Lower percentile for scaling.
        phi : `float`, optional
            Upper percentile for scaling.
        isAnimated : `bool`, optional
            If True, skip static overlays suited for static frames.

        Returns
        -------
        artists : `list`
            List of created artists.
        """
        if fig is None or axs is None:
            fig, axs = self.setupFigure(figsize=(9.5, 9.5))

        artists: list[Artist] = []
        for detName in self.detNames:
            im, _, _, txt = renderStampPanel(
                axs[detName], self.guiderData, detName, stampNum, plo=plo, phi=phi, annotate=True
            )
            if txt is not None:
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

    def plotStackedStampArray(
        self,
        fig: plt.Figure | None = None,
        axs: dict[str, plt.Axes] | None = None,
        plo: float = 50,
        phi: float = 99,
    ) -> list[Artist]:
        """
        Convenience wrapper to plot the stacked (coadded) stamp mosaic.

        Parameters
        ----------
        fig : `matplotlib.figure.Figure`, optional
            Existing figure or None to create.
        axs : `dict`, optional
            Axes dictionary or None to create.
        plo : `float`, optional
            Lower percentile for scaling.
        phi : `float`, optional
            Upper percentile for scaling.

        Returns
        -------
        artists : `list`
            List of created artists.
        """
        artists = self.plotStampArray(stampNum=-1, fig=fig, axs=axs, plo=plo, phi=phi)
        return artists

    def makeGif(
        self,
        saveAs: str,
        fps: int = 5,
        dpi: int = 80,
        plo: float = 50,
        phi: float = 99,
        figsize: tuple[float, float] = (9, 9),
    ) -> animation.ArtistAnimation:
        """
        Create an animated GIF over all stamps for this exposure.

        Parameters
        ----------
        fps : `int`, optional
            Frames per second.
        dpi : `int`, optional
            Output resolution.
        saveAs : `str`, optional
            Output file path (GIF); required to save.
        plo : `float`, optional
            Lower percentile for scaling.
        phi : `float`, optional
            Upper percentile for scaling.
        figsize : `tuple[int, int]`, optional
            Figure size passed to setup.

        Returns
        -------
        ani : `matplotlib.animation.ArtistAnimation`
            Created animation.
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

    def makeMp4(self, saveAs: str, fps: int = 5, dpi: int = 80) -> animation.ArtistAnimation:
        """
        Create an MP4 animation over all stamps for this exposure.

        Parameters
        ----------
        fps : `int`, optional
            Frames per second.
        dpi : `int`, optional
            Output resolution in dots per inch.
        saveAs : `str`, optional
            Output MP4 filename.

        Returns
        -------
        ani : `matplotlib.animation.ArtistAnimation`
            Created animation.
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
        return ani


def getStdCentroid(statsDf: pd.DataFrame, expId: int) -> float:
    """
    Compute combined (quadrature) corrected centroid scatter for an exposure.

    Parameters
    ----------
    statsDf : `pandas.DataFrame`
        Statistics DataFrame containing centroid scatter columns.
    expId : `int`
        Exposure identifier.

    Returns
    -------
    jitter : `float`
        Quadrature sum of corrected AZ and ALT centroid scatter (arcsec).
    """
    stdAz = mad_std(statsDf.loc[statsDf["expid"] == expId, "dalt"].to_numpy())
    stdAlt = mad_std(statsDf.loc[statsDf["expid"] == expId, "daz"].to_numpy())
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
    Draw reference arrows for instrument (DVCS) and rotated Alt/Az axes.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Target axes.
    cutoutSize : `int`
        Size scale used for arrow lengths and framing.
    rotAngle : `float`, optional
        Rotation angle (degrees) for overlay (Alt/Az) axes.
    baseLabels : `tuple[str, str]`, optional
        Labels for the base (unrotated) Y/X axes.
    overlayLabels : `tuple[str, str]`, optional
        Labels for the rotated Alt/Az axes.
    baseColor : `str`, optional
        Color for base axes arrows.
    overlayColor : `str`, optional
        Color for rotated axes arrows.
    center : `tuple[float, float]`, optional
        Arrow origin; defaults to cutout center when None.
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
    Determine reference (x,y) center and centroid offset for a detector/stamp.

    If there is no measurement for that stamp, or stampNum < 0 (a stacked
    image), the offset is (0, 0).

    Parameters
    ----------
    starsDf : `pandas.DataFrame`
        Star measurements table.
    detName : `str`
        Detector name.
    stampNum : `int`
        Stamp index; negative implies stacked image (zero offset).

    Returns
    -------
    center_ref : `tuple[float, float]`
        Reference center coordinates.
    delta : `tuple[float, float]`
        Centroid offset (dX, dY) relative to the reference center.
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
) -> tuple[plt.AxesImage, tuple[float, float], tuple[int, int], plt.Text | None]:
    """
    Render a single detector stamp (or stacked image) with optional cropping
    and annotation.

    Steps:
      1. Select stamp or coadd.
      2. Optionally crop around center.
      3. Apply percentile scaling.
      4. Display with imshow.
      5. Optionally annotate detector label.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Target axes.
    guiderData : `GuiderData`
        Guider data container.
    detName : `str`
        Detector name.
    stampNum : `int`
        Stamp index; negative for stacked/coadd.
    center : `tuple[float, float]`, optional
        Center for cropping; image center if None.
    cutoutSize : `int`, optional
        Square cutout size; -1 for full frame.
    plo : `float`, optional
        Lower percentile for scaling.
    phi : `float`, optional
        Upper percentile for scaling.
    annotate : `bool`, optional
        If True, add detector label text.

    Returns
    -------
    image : `matplotlib.image.AxesImage`
        Image artist.
    centerCutout : `tuple[float, float]`
        Center coordinates within the (possibly cropped) image.
    shape : `tuple[int, int]`
        Shape of the displayed (possibly cropped) image.
    label : `matplotlib.text.Text` or `None`
        Detector label artist if added.
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


def labelDetector(
    ax: plt.Axes,
    name: str,
    *,
    corner: str = "tl",
    color: str = "grey",
    fontsize: int = 9,
    weight: str = "bold",
    pad: tuple[float, float] = (0.025, 0.025),
) -> plt.Text:
    """Place the detector name text inside a panel.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Target axes.
    name : `str`
        Detector name to display.
    corner : `str`, optional
        Two-letter code: t/b (top/bottom) + l/r (left/right).
    color : `str`, optional
        Text color.
    fontsize : `int`, optional
        Font size.
    weight : `str`, optional
        Font weight.
    pad : `tuple[float, float]`, optional
        Relative (x,y) padding in axes fraction.

    Returns
    -------
    text : `matplotlib.text.Text`
        Created text artist.
    """
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
    jitter: float | None = None,
    extra: str | None = None,
    xy: tuple[float, float] = (1.085, -0.10),
) -> plt.Text:
    """
    Annotate exposure metadata and stamp index on the center panel.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Target axes.
    expid : `int`
        Exposure identifier.
    stampNum : `int`
        Current stamp index (0-based). Negative implies stacked.
    nStamps : `int`
        Total number of stamps available.
    view : `str`, optional
        Orientation descriptor.
    jitter : `float`, optional
        Combined centroid scatter (arcsec).
    extra : `str`, optional
        Additional free-form text.
    xy : `tuple[float, float]`, optional
        Annotation position in axes coordinates.

    Returns
    -------
    text : `matplotlib.text.Text`
        Created text artist.
    """
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


def addStaticOverlays(
    axsImg: plt.Axes,
    detName: str,
    centerCutout: tuple[float, float],
    cutoutSize: int,
    camRotAngle: float,
) -> None:
    """
    Add detector label, crosshairs, and guide circles to a guider panel.

    Parameters
    ----------
    axsImg : `matplotlib.axes.Axes`
        Target axes.
    detName : `str`
        Detector name.
    centerCutout : `tuple[float, float]`
        Center coordinates in the displayed cutout.
    cutoutSize : `int`
        Cutout size; influences overlay scaling.
    camRotAngle : `float`
        Camera rotation angle (degrees).
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
    labels: Sequence[str] | None = None,
    text_offset: float = 1.0,
    **circle_kwargs: Any,
) -> list[plt.Text]:
    """
    Draw concentric guide circles with optional labels.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Target axes.
    center : `tuple[float, float]`
        Center (x, y) of circles.
    radii : `Sequence[float]`
        Radii of circles.
    colors : `Sequence[str]`
        Edge colors per circle.
    labels : `Sequence[str]`, optional
        Labels per circle; None for no labels.
    text_offset : `float`, optional
        Offset for label placement along +x.
    **circle_kwargs : `dict`
        Extra keyword args passed to Circle.

    Returns
    -------
    texts : `list[matplotlib.text.Text]`
        List of label text artists.
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
    Plot a rotated crosshair centered on given coordinates.

    Parameters
    ----------
    center : `tuple[float, float]`
        Center (x, y) in image coordinates.
    angle : `float`
        Rotation angle in degrees.
    axs : `matplotlib.axes.Axes`, optional
        Target axes or None to use current.
    color : `str`, optional
        Line color.
    size : `int`, optional
        Size scaling factor.
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
    Plot the measured star centroid (with optional error bars) on a cutout.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Target axes.
    centerCutout : `tuple[float, float]`
        Reference center in the cutout.
    deltaXY : `tuple[float, float]`
        Offset (dx, dy) from center to centroid.
    markerSize : `int`, optional
        Marker size; <=0 disables plotting.
    errXY : `tuple[float, float]`, optional
        (xerr, yerr) half-lengths for error bars.
    color : `str`, optional
        Marker and line color.

    Returns
    -------
    artists : `list[matplotlib.lines.Line2D]`
        Marker and error bar line artists.
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
    Remove all ticks/labels; optionally keep spines.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Target axes.
    isSpine : `bool`, optional
        If True, retain spines; otherwise hide them.
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
    Extract a square crop centered on the specified coordinates.

    Parameters
    ----------
    image : `numpy.ndarray`
        Source 2D image.
    center : `tuple[float, float]`
        Center (x, y) in original image coordinates.
    size : `int`
        Desired square size in pixels.

    Returns
    -------
    cropped : `numpy.ndarray`
        Cropped (and possibly padded) image of shape (size, size).
    new_center : `tuple[float, float]`
        Center coordinates within the cropped image.
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
