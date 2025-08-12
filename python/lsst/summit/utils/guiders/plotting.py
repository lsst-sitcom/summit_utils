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
from typing import TYPE_CHECKING, Any, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.stats import mad_std
from matplotlib import animation
from matplotlib.patches import Circle
from scipy.ndimage import affine_transform

from lsst.summit.utils.utils import RobustFitter

if TYPE_CHECKING:
    from .reading import GuiderData

sns.set_context("talk", font_scale=0.8)

# Fine-tune fonts & lines from matplotlib side
plt.rcParams.update(
    {
        # --- font family & scaling ---
        "font.family": "sans-serif",
        # pick one that exists on most systems; change if you have a favourite
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
        "text.color": "#222222",
        "axes.labelcolor": "#222222",
        "axes.edgecolor": "#444444",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    }
)


__all__ = ["GuiderDataPlotter", "GuiderPlotter"]

LIGHT_BLUE = "#6495ED"


class GuiderPlotter:
    def addStaticOverlays(
        self,
        axsImg: plt.Axes,
        detName: str,
        centerCutout: tuple[float, float],
        cutoutSize: int,
    ) -> None:
        """
        Add static overlays to a guider image axis, with detector annotation,
        crosshairs, and guide circles.

        Parameters
        ----------
        axs_img : matplotlib.axes.Axes
            The axis to draw overlays on.
        detname : str
            Name of the detector.
        centerCutout : tuple
            Center of the cutout image.
        cutout_size : int
            Size of the cutout.

        Returns
        -------
        None
        """
        annotateDetector(detName, axsImg)
        axsImg.axvline(centerCutout[0], color=LIGHT_BLUE, lw=1.25, linestyle="--", alpha=0.75)
        axsImg.axhline(centerCutout[1], color=LIGHT_BLUE, lw=1.25, linestyle="--", alpha=0.75)
        plot_crosshair_rotated(
            centerCutout,
            90 + self.camRotAngle,
            axs=axsImg,
            color="grey",
            size=cutoutSize,
        )
        radii = [10, 5] if cutoutSize > 0 else [10]
        _ = plot_guide_circles(
            axsImg,
            centerCutout,
            radii=radii,
            colors=[LIGHT_BLUE, LIGHT_BLUE],
            labels=["2″", "1″"],
            linewidth=2.0,
        )

    def plotStarCentroid(
        self,
        axsImg: plt.Axes,
        detName: str,
        stampNum: int,
        centerCutout: tuple[float, float],
        xcen: float,
        ycen: float,
        markerSize: int = 8,
    ) -> list:
        """
        Plot the centroid of a star on the detector cutout image.

        Parameters
        ----------
        axs_img : matplotlib.axes.Axes
            Axis object to plot on.
        detname : str
            Detector name.
        stamp_num : int
            Stamp number, or -1 for stacked.
        centerCutout : tuple
            Center of the cutout image (x, y).
        xcen : float
            Reference x centroid.
        ycen : float
            Reference y centroid.
        markersize : int, optional
            Marker size for the centroid point.

        Returns
        -------
        list
            List of matplotlib artist objects for the centroid and error bars.
        """
        xroi, yroi = centerCutout[0], centerCutout[1]
        xroiErr, yroiErr = 0.0, 0.0

        star = self.starsDf[(self.starsDf["detector"] == detName)]
        if not star.empty:
            if stampNum >= 0:
                star = star[star["stamp"] == stampNum]
                if not star.empty:
                    xroi, yroi = star.iloc[0][["xroi", "yroi"]]
                    xroiErr, yroiErr = star.iloc[0][["xerr", "yerr"]]
            elif markerSize > 0:
                xroi, yroi = star.iloc[0][["xroi_ref", "yroi_ref"]]
                xroiErr, yroiErr = star[["xerr", "yerr"]].mean()

        xroi, yroi = (
            centerCutout[0] + xroi - xcen,
            centerCutout[1] + yroi - ycen,
        )

        (marker,) = axsImg.plot(xroi, yroi, "o", color="firebrick", markersize=markerSize)
        (hline,) = axsImg.plot([xroi - xroiErr, xroi + xroiErr], [yroi, yroi], color="firebrick", lw=2.5)
        (vline,) = axsImg.plot([xroi, xroi], [yroi - yroiErr, yroi + yroiErr], color="firebrick", lw=2.5)
        return [marker, hline, vline]

    UNIT_DICT = {
        "centroidAltAz": "arcsec",
        "centroidPixel": "pixels",
        "flux": "magnitudes",
        "secondMoments": "pixels²",
        "psf": "arcsec",
    }

    # for plotting
    LAYOUT: list[tuple[str, ...]] = [
        (".", "R40_SG1", "R44_SG0", "."),
        ("R40_SG0", "center", ".", "R44_SG1"),
        ("R00_SG1", ".", ".", "R04_SG0"),
        ("arrow", "R00_SG0", "R04_SG1", "."),
    ]
    DETNAMES = [cell for row in LAYOUT for cell in row if (cell[0] == "R")]

    # The COLOR_MAP and MARKERS lists are used for plotting different detector
    COLOR_MAP = [
        "black",
        "firebrick",
        "grey",
        "lightgrey",
        "blue",
        "green",
        "orange",
        "purple",
    ]
    MARKERS = ["o", "x", "+", "*", "^", "v", "s", "p"]

    def __init__(
        self,
        starsDf: pd.DataFrame,
        guiderData: GuiderData,
        isIsr: bool = True,
    ) -> None:
        self.log = logging.getLogger(__name__)
        self.expId = guiderData.header.get("expid", 0)

        if starsDf.empty:
            self.log.warning("stars_df is empty. No data to plot.")
            self.statsDf = pd.DataFrame(columns=["expid"])
            return

        self.starsDf = starsDf[starsDf["expid"] == self.expId]
        self.guiderData = guiderData
        self.isIsr = isIsr  # apply or not parrallel over scan bias correction

        # Some metadata information
        self.expTime = int(self.guiderData.header.get("SHUTTIME") or 0)
        self.seeing = float(self.guiderData.header.get("SEEING") or np.nan)
        self.camRotAngle = float(self.guiderData.header.get("CAM_ROT_ANGLE") or np.nan)
        elstart, elstop = (
            float(self.guiderData.header["el_start"] or np.nan),
            float(self.guiderData.header["el_end"] or np.nan),
        )
        azstart, azstop = (
            float(self.guiderData.header["az_start"] or np.nan),
            float(self.guiderData.header["az_end"] or np.nan),
        )
        self.el = 0.5 * (elstart + elstop)
        self.az = 0.5 * (azstart + azstop)

        # assemble statistics
        self.statsDf = self.assembleStats()

        sns.set_style("white")
        sns.set_context("talk", font_scale=0.8)

    def assembleStats(self) -> pd.DataFrame:
        """
        Assemble summary statistics from the stars dataframe
        for the current exposure.

        Returns
        -------
        pd.DataFrame
            DataFrame with a summary statistics for the current exposure.
        """
        stars = self.starsDf

        if stars.empty:
            cols = [
                "n_guiders",
                "n_stars",
                "fraction_valid_stamps",
                "n_measurements",
            ]
            exampleStdCentroid = [
                "std_centroid_az",
                "std_centroid_alt",
                "std_centroid_corr_az",
                "std_centroid_corr_alt",
                "drift_rate_az",
                "drift_rate_alt",
                "offset_zero_az",
                "offset_zero_alt",
            ]
            examplePhot = ["magoffset_rate", "magoffset_zero", "magoffset_rms"]
            guiderNames = stars["detector"].unique()
            cols += [f"N_{det}" for det in guiderNames]
            cols += exampleStdCentroid + examplePhot
            return pd.DataFrame(columns=cols)

        nGuiders = stars["detector"].nunique()
        nUnique = stars["starid"].nunique()
        counts = stars.groupby("detector")["starid"].nunique().to_dict()
        guiderNames = stars["detector"].unique()
        starsPerGuiders = {f"N_{det}": counts.get(det, 0) for det in guiderNames}

        maskValid = (stars["stamp"] >= 0) & (stars["xccd"].notna())
        nMeas = int(maskValid.sum())

        stdCentroid = measure_std_centroid_stats(stars, snr_th=5)
        phot = measure_photometric_variation(stars, snr_th=5)
        psfStats = measure_psf_stats(stars, snr_th=5)

        totalPossible = nUnique * stars["stamp"].nunique()
        fracValid = nMeas / totalPossible if totalPossible > 0 else np.nan

        summary = {
            "expid": self.expId,
            "n_guiders": nGuiders,
            "n_stars": nUnique,
            "n_measurements": nMeas,
            "fraction_valid_stamps": fracValid,
            "seeing": self.seeing,
            "filter": self.guiderData.header.get("filter", "unknown"),
            "alt": self.el,
            "az": self.az,
            **starsPerGuiders,
            **stdCentroid,
            **phot,
            **psfStats,
        }
        return pd.DataFrame([summary])

    def printMetrics(self) -> None:
        """
        Print formatted metrics and statistics for the current exposure.

        Returns
        -------
        None
        """
        if self.statsDf.empty:
            print("No statistics available for this exposure.")
            return

        filteredStatsDf = self.statsDf[self.statsDf["expid"] == self.expId].copy()

        print(self.formatStatsSummary(filteredStatsDf))
        print(self.formatStdCentroidSummary(filteredStatsDf, exptime=self.expTime))
        print(self.formatPhotometricSummary(filteredStatsDf, exptime=self.expTime))
        print(self.formatPsfSummary(filteredStatsDf, exptime=self.expTime))

    def stripPlot(self, plotType: str = "centroidAltAz", saveAs: Optional[str] = None) -> None:
        """
        Generate a strip plot (time series) for various metrics (e.g.,
        centroid, flux, PSF).

        Parameters
        ----------
        plot_type : str, optional
            Type of metric to plot. One of "centroidAltAz", "centroidPixel",
            "flux", "ellip", "psf".
        save_as : str or None, optional
            If provided, save the plot to this file.

        Returns
        -------
        None
        """
        if self.starsDf.empty:
            self.log.warning("stars_df is empty. No data to plot.")
            return

        # plot_kwargs dtype is dict[str, Any]
        plotKwargs: dict[str, dict] = {
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
        cfg = plotKwargs[plotType]  # type: dict[str, Any]
        scale = cfg.get("scale", 1.0)  # type: float
        cols = cfg["col"]
        unit = cfg.get("unit", "")
        exp = float(self.expTime)

        # filter and prepare
        df = self.starsDf[self.starsDf["elapsed_time"] > 0][["elapsed_time", "detector"] + cols].copy()

        # Compute overall mean and sigma for y-axis limits
        allData = df[cols].values.flatten()
        allData *= scale
        plow, phigh = np.nanpercentile(allData, [16, 84])
        sigmaVal = mad_std(allData, ignore_nan=True)
        ylims = (plow - 2.5 * sigmaVal, phigh + 2.5 * sigmaVal)

        # setup subplots
        n = len(cols)
        fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(8 * n, 6), sharex=True, sharey=True)
        fig.subplots_adjust(wspace=0.05)

        if n == 1:
            axes = [axes]

        count = 0
        for ax, col in zip(axes, cols):
            # horizontal line at zero
            if col == "daz":
                ax.axhline(0, color="grey", ls="--", label="Az")
            if col == "dalt":
                ax.axhline(0, color="grey", ls="--", label="Alt")

            if col == "dx":
                ax.axhline(0, color="grey", ls="--", label="CCD X")
            if col == "dy":
                ax.axhline(0, color="grey", ls="--", label="CCD Y")

            if col == "e1_altaz":
                ax.axhline(0, color="grey", ls="--", label="e1")
            if col == "e2_altaz":
                ax.axhline(0, color="grey", ls="--", label="e2")

            if col == "magoffset":
                ax.axhline(0, color="grey", ls="--", label="Magnitude Offset")

            if col == "fwhm":
                ax.axhline(np.nanmedian(df[col] * scale), color="grey", ls="--", label="PSF FWHM")

            model = RobustFitter(
                x=df["elapsed_time"].values,
                y=df[col].values * scale,
                residual_threshold=1.5 * sigmaVal,
            )

            results = model.report_best_values()
            # Format best-fit stats as text
            txt = (
                f"Slope: {exp * results.slope:.2f} {unit}/exposure\n"
                f"Significance: {np.abs(results.slope_tvalue):.1f} σ\n"
                f"scatter: {results.scatter:.3f} {unit}\n"
            )

            # Place text on the plot (top left, inside axes)
            ax.text(
                0.02,
                0.98,
                txt,
                transform=ax.transAxes,
                fontsize=11,
                color="black",
                ha="left",
                va="top",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", boxstyle="round,pad=0.3"),
            )
            model.plot_best_fit(ax=ax, color=LIGHT_BLUE, label="", lw=2, is_scatter=False)

            # scatter points
            # Use different markers for each guider detector
            uniqueStarIds = df["detector"].unique()
            for i, starid in enumerate(uniqueStarIds):
                marker = self.MARKERS[i % len(self.MARKERS)]
                mask = df["detector"] == starid
                mask2 = mask & (model.outlier_mask)

                ax.scatter(
                    df.loc[mask, "elapsed_time"],
                    df.loc[mask, col] * scale,
                    color="lightgrey",
                    alpha=0.6,
                    marker=marker,
                    label=f"{starid}" if count == 0 else "",
                )

                # plot outliers
                ax.scatter(
                    df.loc[mask2, "elapsed_time"],
                    df.loc[mask2, col] * scale,
                    color=LIGHT_BLUE,
                    alpha=0.2,
                    marker=marker,
                )

            if count == 0:
                ax.set_ylabel(cfg["ylabel"])
            ax.set_xlabel("Elapsed time [sec]")
            ax.set_ylim(*ylims)  # <-- Set y-axis limits here
            ax.legend(fontsize=10, ncol=4, loc="lower left")

            count += 1
        fig.suptitle(cfg["title"], fontsize=14, fontweight="bold")
        if count == 0:
            fig.tight_layout()

        if saveAs:
            fig.savefig(saveAs, dpi=120, bbox_inches="tight")
            print(f"Saved strip plot to {saveAs}")

    def getStarCentroidRef(self, guiderData: GuiderData) -> dict[str, tuple[float | None, float | None]]:
        """
        Convert the best star centroid in each guider to ROI coordinates.

        Parameters
        ----------
        guiderData : GuiderData
            Guider data object with star information.

        Returns
        -------
        dict of str to tuple of (float or None, float or None)
            Dictionary mapping detector name to (xroi, yroi) centroid.
        """
        centroids: dict[str, tuple[float | None, float | None]] = {}
        for det in self.DETNAMES:
            sub = self.starsDf[self.starsDf["detector"] == det]
            if len(sub) > 0:
                best = sub.loc[sub["snr"].idxmax()]
                xroi = np.nanmedian([best["xroi_ref"]])
                yroi = np.nanmedian([best["yroi_ref"]])
                centroids[det] = float(xroi), float(yroi)
            else:
                centroids[det] = (None, None)
        self.centroids = centroids
        return centroids

    def loadImage(
        self,
        guider: GuiderData,
        detName: str,
        stampNum: int = 2,
    ) -> np.ndarray:
        """
        Load a guider stamp image for a specific detector and stamp number.

        Parameters
        ----------
        guider : GuiderData
            Guider data object.
        detname : str
            Detector name.
        stamp_num : int, optional
            Stamp number to load; if negative, loads stacked image.

        Returns
        -------
        np.ndarray
            Image array for the specified stamp and detector.
        """
        # read full stamp
        if stampNum >= len(guider):
            raise ValueError(
                f"stamp_num {stampNum} is out of range for guider" + f"with {len(guider)} stamps."
            )
        elif stampNum < 0:
            return guider.getStampArrayCoadd(detName)
        else:
            img = guider[detName, stampNum]
            return img

    def starMosaic(
        self,
        stampNum: int = 2,
        fig: Optional[plt.Figure] = None,
        axs: Optional[dict[str, plt.Axes]] = None,
        plo: float = 90.0,
        phi: float = 99.0,
        cutoutSize: int = 30,
        isAnimated: bool = False,
        saveAs: Optional[str] = None,
    ) -> list:
        """
        Plot the stamp array for all the guiders in a mosaic layout.

        Parameters
        ----------
        stamp_num : int, optional
            Stamp number to plot; if negative, plot stacked image.
        fig : matplotlib.figure.Figure, optional
            Figure object to use.
        axs : dict, optional
            Dictionary of axes for the mosaic.
        plo : float, optional
            Lower percentile for image scaling.
        phi : float, optional
            Upper percentile for image scaling.
        cutout_size : int, optional
            Size of cutout for each detector panel.
        is_animated : bool, optional
            If True, do not add static overlays (for animation).
        save_as : str or None, optional
            If provided, save the figure to this file.

        Returns
        -------
        list
            List of matplotlib artist objects.
        """
        if self.starsDf.empty:
            self.log.warning("stars_df is empty. No data to plot.")
            return []

        if self.guiderData.view != "dvcs":
            raise ValueError("Guider view must be 'dvcs' for mosaic plotting.")

        if fig is None:
            gs = dict(hspace=0.0, wspace=0.0)
            fig, axs = plt.subplot_mosaic(
                cast(Any, self.LAYOUT),
                figsize=(9.5, 9.5),
                gridspec_kw=gs,
                constrained_layout=False,
            )

        assert axs is not None

        if not hasattr(self, "centroids"):
            self.getStarCentroidRef(self.guiderData)

        artists = []
        cutoutShapeList = []

        for detName in self.DETNAMES:
            imObject, starCross, cutoutShape = self._plotDetectorPanel(
                detName, stampNum, axs, plo, phi, cutoutSize, isAnimated
            )
            artists.append(imObject)
            artists.extend(starCross)
            cutoutShapeList.append(cutoutShape)

        stdAz = self.statsDf.loc[self.statsDf["expid"] == self.expId, "std_centroid_corr_az"].values[0]
        stdAlt = self.statsDf.loc[self.statsDf["expid"] == self.expId, "std_centroid_corr_alt"].values[0]
        std = np.hypot(stdAz, stdAlt)
        stampInfo = self.annotateCenter(stampNum, axs["center"], jitter=std)
        artists.append(stampInfo)

        cutoutSize = np.max(cutoutShapeList) if cutoutShapeList else 30
        if not isAnimated:
            drawArrows(axs["arrow"], cutoutSize, 90.0 + self.camRotAngle)

        for ax in axs.values():
            clearAxisTicks(ax, isSpine=cutoutSize < 0)

        if saveAs:
            fig.savefig(saveAs, dpi=120, bbox_inches="tight")
            print(f"Saved mosaic plot to {saveAs}")

        return artists

    def _plotDetectorPanel(
        self,
        detName: str,
        stampNum: int,
        axs: dict[str, plt.Axes],
        plo: float,
        phi: float,
        cutoutSize: int,
        isAnimated: bool,
    ) -> tuple[Any, list, tuple[int, int]]:
        """
        Plot a single detector panel in the mosaic.

        Parameters
        ----------
        detname : str
            Detector name.
        stamp_num : int
            Stamp number.
        axs : dict
            Mosaic axes dictionary.
        plo : float
            Lower percentile for image scaling.
        phi : float
            Upper percentile for image scaling.
        cutout_size : int
            Cutout size for the detector panel.
        is_animated : bool
            If True, do not add static overlays.

        Returns
        -------
        tuple
            (image object, star_cross artists, cutout shape)
        """
        img = self.loadImage(self.guiderData, detName, stampNum)
        mx, my = img.shape[0] // 2, img.shape[1] // 2
        xcen, ycen = self.centroids.get(detName, (mx, my))
        if xcen is None or ycen is None:
            xcen, ycen = mx, my
        center = (float(xcen), float(ycen))

        if cutoutSize > 0:
            cutout, centerCutout = crop_around_center(img, center, cutoutSize)
        else:
            cutout, centerCutout = img, center
            stampNum = -1  # do not plot star

        vmin, vmax = np.nanpercentile(cutout, plo), np.nanpercentile(cutout, phi)
        axsImg = axs[detName]
        imObject = axsImg.imshow(
            cutout,
            origin="lower",
            cmap="Greys",
            animated=True,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            extent=(0, cutout.shape[1], 0, cutout.shape[0]),
        )
        axsImg.set_aspect("equal", "box")

        if not isAnimated:
            self.addStaticOverlays(axsImg, detName, centerCutout, cutoutSize)

        starCross = self.plotStarCentroid(
            axsImg, detName, stampNum, centerCutout, xcen, ycen, markerSize=8 if cutoutSize > 0 else 0
        )

        axsImg.set_xlim(0, cutout.shape[1])
        axsImg.set_ylim(0, cutout.shape[0])
        return imObject, starCross, cutout.shape

    def mosaic(self, stampNum: int = 2, **kwargs) -> None:
        """
        Wrapper to plot a single guider mosaic for a specific stamp.

        Parameters
        ----------
        stamp_num : int, optional
            Stamp number to plot.
        **kwargs
            Additional keyword arguments passed to star_mosaic.

        Returns
        -------
        None
        """
        fig, axs = plt.subplot_mosaic(
            cast(Any, self.LAYOUT),
            figsize=(12, 12),
            gridspec_kw=dict(hspace=0.0, wspace=0.0),
            constrained_layout=False,
            sharey=False,
            sharex=False,
        )
        _ = self.starMosaic(stampNum=stampNum, cutoutSize=-1, fig=fig, axs=axs, **kwargs)

        plt.suptitle(f"Guider Mosaic for expid: {self.expId}\n stamp #: {stampNum + 1:02d}")
        plt.show()
        return None

    def annotateCenter(self, stampNum: int, ax: plt.Axes, jitter: float = -1) -> plt.Text:
        """
        Annotate the center panel with exposure and stamp info.

        Parameters
        ----------
        stamp_num : int
            Stamp number.
        ax : matplotlib.axes.Axes
            Axis to annotate.
        jitter : float, optional
            Jitter value to display.

        Returns
        -------
        matplotlib.text.Text
            The text artist added.
        """
        clearAxisTicks(ax, isSpine=True)
        text = f"Center Stdev.: {jitter:.2f} arcsec\n" if jitter > 0 else ""
        text += (
            f"expid: {self.expId}\nStamp #: {stampNum + 1:02d}"
            if stampNum >= 0
            else f"expid: {self.expId}\nStacked w/ {self.starsDf['stamp'].nunique()} stamps"
        )

        txt = ax.text(
            1.085,
            -0.10,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="grey",
        )
        return txt

    def plotCircle(
        self,
        ax: plt.Axes,
        xcen: float,
        ycen: float,
        radius: float = 5,
        color: str = "firebrick",
        lw: float = 1.0,
    ) -> Circle:
        """
        Add a circular patch at (xcen, ycen) with given radius on the axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axis to add the circle to.
        xcen : float
            X coordinate of center.
        ycen : float
            Y coordinate of center.
        radius : float, optional
            Radius of the circle.
        color : str, optional
            Color of the circle edge.
        lw : float, optional
            Line width.

        Returns
        -------
        matplotlib.patches.Circle
            The circle patch added to the axis.
        """
        circ = Circle(
            (xcen, ycen),
            radius=radius,
            edgecolor=color,
            facecolor="none",
            lw=lw,
            ls="--",
        )
        ax.add_patch(circ)
        return circ

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
        Create an animated GIF of the guider mosaic over a sequence of stamps.

        Parameters
        ----------
        save_as : str
            save the animation to this file.
        n_stamp_max : int, optional
            Maximum number of stamps to animate.
        fps : int, optional
            Frames per second.
        dpi : int, optional
            Dots per inch for saved animation.
        plo : float, optional
            Lower percentile for image scaling.
        phi : float, optional
            Upper percentile for image scaling.
        cutout_size : int, optional
            Size of cutout for each detector panel.

        Returns
        -------
        matplotlib.animation.ArtistAnimation
            The resulting animation object.
        """
        from matplotlib import animation

        # build canvas
        fig, axs = plt.subplot_mosaic(
            cast(Any, self.LAYOUT),
            figsize=(10, 10),
            gridspec_kw=dict(hspace=0.0, wspace=0.0),
            constrained_layout=False,
        )

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
    def formatStdCentroidSummary(statsDf: pd.DataFrame, exptime: float = 1) -> str:
        """
        Pretty string summary of centroid stdev. stats from run_all_guiders.

        Parameters
        ----------
        stats_df : pd.DataFrame
            DataFrame or dict with centroid statistics.
        exptime : float, optional
            Exposure time scaling factor.

        Returns
        -------
        str
            Formatted summary string.
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
            f"  - Drift Rate       (AZ): {exptime * js['drift_rate_az']:.2f} arcsec per exposure\n"
            f"  - Drift Rate      (ALT): {exptime * js['drift_rate_alt']:.2f} arcsec per exposure\n"
            f"  - Zero Offset      (AZ): {js['offset_zero_az']:.3f} arcsec\n"
            f"  - Zero Offset     (ALT): {js['offset_zero_alt']:.3f} arcsec"
        )
        return summary

    @staticmethod
    def formatPhotometricSummary(photStats: pd.DataFrame, exptime: float = 1) -> str:
        """
        Pretty-print summary of photometric variation statistics.

        Parameters
        ----------
        phot_stats : pd.DataFrame or dict
            DataFrame or dict with photometric statistics.
        exptime : float, optional
            Exposure time scaling factor.

        Returns
        -------
        str
            Formatted summary string.
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
            f"  - Mag Drift Rate:      {exptime * stats['magoffset_rate'] * 1e3:.1f} mmag/exposure\n"
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

        Parameters
        ----------
        summary : pd.DataFrame or dict
            DataFrame or dict with summary statistics.

        Returns
        -------
        str
            Formatted summary string.
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

        # Per-guider counts (keys begin with 'N_')
        guiderKeys = sorted(k for k in summary if k.startswith("N_"))
        if guiderKeys:
            lines.append("\nStars per Guider:")
            for k in guiderKeys:
                lines.append(f"  - {k[2:]}: {int(summary[k])}")
        return "\n".join(lines)

    @staticmethod
    def formatPsfSummary(psfStats: pd.DataFrame, exptime: float = 1) -> str:
        """
        Pretty-print summary of PSF statistics.
        Expects keys like:
          fwhm, e1, e2, etc.

        Parameters
        ----------
        psf_stats : pd.DataFrame or dict
            DataFrame or dict with PSF statistics.
        exptime : float, optional
            Exposure time scaling factor.

        Returns
        -------
        str
            Formatted summary string.
        """
        if (isinstance(psfStats, pd.DataFrame) and psfStats.empty) or (
            isinstance(psfStats, dict) and not psfStats
        ):
            return "No PSF statistics available."

        # if it's a DataFrame, extract the one-row dict
        if isinstance(psfStats, pd.DataFrame):
            psfStats = psfStats.iloc[0].to_dict()

        lines = ["", "-" * 50]
        lines.append("PSF Statistics Summary:")
        lines.append(f"  - FWHM slope   : {exptime * psfStats['fwhm_rate']:.3f} arcsec per exposure")
        lines.append(f"  - FWHM scatter : {psfStats['fwhm_rms']:.3f} arcsec")
        return "\n".join(lines)


class GuiderDataPlotter:
    """
    Class to plot guider stamp mosaics using a GuiderData container.
    Plot an animated gif of the CCD guider stamp.

    Example:
        from lsst.summit.utils.guiders.reading import GuiderReader
        from lsst.summit.utils.guiders.plotting import GuiderMosaicPlotter

        # Pick a seq number and dayObs
        seqNum, dayObs = 591, 20250425

        # Load the data from the butler
        guiderData = GuiderReader(seqNum, dayObs, view='dvcs')

        # Create the plotter object
        plotter = GuiderMosaicPlotter(guiderData)

        # Plot a stacked image of the stamps
        plotter.plotStackedStampArray()

        # Plot a single stamp
        plotter.plotStampArray(stampNum=9)

        # Make a gif of the stamps
        plotter.makeGif(nStampMax=50, fps=10)
    """

    # for plotting
    LAYOUT: list[tuple[str, ...]] = [
        (".", "R40_SG1", "R44_SG0", "."),
        ("R40_SG0", "center", ".", "R44_SG1"),
        ("R00_SG1", ".", ".", "R04_SG0"),
        ("arrow", "R00_SG0", "R04_SG1", "."),
    ]

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

        Parameters
        ----------
        detName : str
            Detector name.
        axs : matplotlib.axes.Axes
            Axis to plot on.
        stampNum : int, optional
            Stamp number; if negative, plot stacked image.
        plo : float, optional
            Lower percentile for image scaling.
        phi : float, optional
            Upper percentile for image scaling.
        is_ticks : bool, optional
            Whether to show ticks.

        Returns
        -------
        matplotlib.image.AxesImage
            Image artist.
        """
        if stampNum < 0:
            img = self.guiderData.getStampArrayCoadd(detName)
        else:
            img = self.guiderData[detName, stampNum]

        bias = np.median(img)
        imgIsr = img - bias
        lo, hi = np.nanpercentile(imgIsr, [plo, phi])

        im = axs.imshow(imgIsr, origin="lower", cmap="Greys", vmin=lo, vmax=hi, animated=True)

        if not is_ticks:
            axs.set_yticklabels([])
            axs.set_xticklabels([])
            axs.set_xticks([])
            axs.set_xticks([], minor=True)
            axs.set_yticks([])
            axs.set_yticks([], minor=True)
        return im

    def getStampNumberInfo(self, stampNum=0) -> str:
        """
        Generate a string with stamp number and observation info.

        Parameters
        ----------
        stampNum : int, optional
            Stamp number.

        Returns
        -------
        str
            Info string for the stamp.
        """
        text = f"day_obs: {self.dayObs}" + "\n" + f"seq_num: {self.seqNum}" + "\n"
        text += f"orientation: {self.view}" + "\n"
        if stampNum > 0:
            text += f"Stamp #: {stampNum + 1:02d}"
        else:
            text += f"Stacked Image w/ {self.nStamps} stamps"
        return text

    def plotStampInfo(self, stampNum=0, axs=None, moreText=None) -> plt.Text:
        """
        Plot stamp info text in the center panel.

        Parameters
        ----------
        stampNum : int, optional
            Stamp number.
        axs : matplotlib.axes.Axes, optional
            Axis to plot on.
        moreText : str or None, optional
            Additional text to append.

        Returns
        -------
        matplotlib.text.Text
            The text artist.
        """
        if axs is None:
            axs = plt.gca()

        text = self.getStampNumberInfo(stampNum)
        if moreText is not None:
            text += "\n" + moreText

        stampIdText = axs.text(
            1.085,
            -0.10,
            text,
            transform=axs.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="grey",
            animated=True,
        )
        axs.set_axis_off()
        clearAxisTicks(axs, isSpine=True)
        self.stampIdAxs = stampIdText
        self.stampIdMoreText = moreText
        return stampIdText

    def plotTextCcdName(self, detName, axs=None) -> plt.Text:
        """
        Annotate a CCD panel with its detector name.

        Parameters
        ----------
        detName : str
            Detector name.
        axs : matplotlib.axes.Axes, optional
            Axis to annotate.

        Returns
        -------
        matplotlib.text.Text
            The text artist.
        """
        if axs is None:
            axs = plt.gca()
        txt = axs.text(
            0.025,
            0.025,
            detName,
            transform=axs.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            weight="bold",
            color=LIGHT_BLUE,
        )
        return txt

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
        list
            List of matplotlib artist objects.
        """
        if fig is None:
            gs = dict(hspace=0.0, wspace=0.0)
            fig, axs = plt.subplot_mosaic(
                cast(Any, self.LAYOUT), figsize=(9.5, 9.5), gridspec_kw=gs, constrained_layout=False
            )

        artists = []
        for detName in self.detNames:
            im = self.plotStampCcd(axs[detName], detName, stampNum=stampNum, plo=plo, phi=phi)
            txt = self.plotTextCcdName(detName, axs=axs[detName])
            artists.extend([im, txt])
        stampInfo = self.plotStampInfo(axs=axs["center"], stampNum=stampNum)

        # add coordinate arrow
        cutoutSize = max(*self.guiderData[0].shape)
        if not isAnimated:
            cam_rot_angle = cast(float, self.guiderData.camRotAngle)
            drawArrows(axs["arrow"], cutoutSize, 90.0 + cam_rot_angle)

        artists.append(stampInfo)
        fig.tight_layout()
        return artists

    def plotStackedStampArray(self, fig=None, axs=None, plo=50, phi=99) -> list:
        """
        Plot the stacked stamp array for all the guiders.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Figure object.
        axs : dict, optional
            Dictionary of axes for the mosaic.

        Returns
        -------
        list
            List of matplotlib artist objects.
        """
        artists = self.plotStampArray(stampNum=-1, fig=fig, axs=axs, plo=plo, phi=phi)
        return artists

    def makeGif(
        self, nStampMax=1000, fps=5, dpi=80, saveAs=None, plo=50, phi=99, figsize=(9, 9)
    ) -> animation.ArtistAnimation:
        """
        Create an animated GIF of the guider CCD array
        over a sequence of stamps.

        Parameters
        ----------
        nStampMax : int, optional
            Maximum number of stamps to animate.
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
        fig, axs = plt.subplot_mosaic(
            cast(Any, self.LAYOUT),
            figsize=figsize,
            gridspec_kw=dict(hspace=0.0, wspace=0.0),
            constrained_layout=False,
        )

        nStamps = min(self.nStamps, nStampMax)
        # print("Number of stamps: ", nStamps)

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
        plt.close(fig)
        return ani

    def makeMp4(
        self, nStampMax=10, fps=5, dpi=80, saveAs="guider_ccd_array.mp4"
    ) -> animation.ArtistAnimation:
        """
        Create an MP4 animation of the guider CCD array
        over a sequence of stamps.

        Parameters
        ----------
        nStampMax : int, optional
            Maximum number of stamps to animate.
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
        fig, axs = plt.subplot_mosaic(
            cast(Any, self.LAYOUT),
            figsize=(9.5, 9.5),
            gridspec_kw=dict(hspace=0.0, wspace=0.0),
            constrained_layout=False,
        )

        nStamps = min(self.nStamps, nStampMax)
        print("Number of stamps: ", nStamps)

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


def measure_std_centroid_stats(stars: pd.DataFrame, snr_th: int = 5, flux_th: float = 10.0) -> pd.DataFrame:
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
    coefs_alt = rf_alt.report_best_values()

    rf_az = RobustFitter(time, az)
    coefs_az = rf_az.report_best_values()

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


def measure_psf_stats(stars: pd.DataFrame, snr_th: int = 5, flux_th: float = 10) -> dict[str, float]:
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
    coefs = model.report_best_values()

    # Elipiticities
    e1 = stars["e1_altaz"][mask].to_numpy()
    # Make a robust fit
    model = RobustFitter(time, e1)
    coefs_e1 = model.report_best_values()

    e2 = stars["e2_altaz"][mask].to_numpy()
    # Make a robust fit
    model = RobustFitter(time, e2)
    coefs_e2 = model.report_best_values()

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


def crop_around_center(
    image: np.ndarray, center: tuple[float, float], size: int
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Crop a square region from the image centered at a specific point,
    returning both the cropped image and the new center coordinates.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    center : tuple of float
        (x, y) coordinates of center.
    size : int
        Size of the square region to crop.

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
    pad_top = max(0, (half - y))
    pad_left = max(0, (half - x))
    pad_bottom = size - (pad_top + (y1 - y0))
    pad_right = size - (pad_left + (x1 - x0))

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        cropped = np.pad(
            cropped,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=np.nan,
        )

    # New center in cropped image
    new_center = (x - x0 + pad_left, y - y0 + pad_top)
    return cropped, new_center


def rotate_around_center(image: np.ndarray, angle_deg: float, center: tuple[float, float]) -> np.ndarray:
    """
    Rotate an image around a specified center by a given angle.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    angle_deg : float
        Rotation angle in degrees.
    center : tuple of float
        (x, y) coordinates of the rotation center.

    Returns
    -------
    np.ndarray
        Rotated image.
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Rotation matrix in (row, col) = (y, x) order
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])

    center_np = np.array(center)
    offset = center_np - R @ center_np

    rotated = affine_transform(image, R, offset=offset, order=3, mode="nearest")
    return rotated


def measure_photometric_variation(stars: pd.DataFrame, snr_th: int = 5) -> dict[str, float]:
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
    coefs_mag = rf_mag.report_best_values()

    # Stats
    phot_stats = {
        "magoffset_rate": coefs_mag.slope,
        "magoffset_signficance": coefs_mag.slope_tvalue,
        "magoffset_zero": np.nanmedian(mo[mask]),
        "magoffset_rms": coefs_mag.scatter,
    }
    return phot_stats


def annotateDetector(detName: str, ax: plt.Axes, color: str = "grey") -> plt.Text:
    """
    Annotate a detector panel with its name.

    Parameters
    ----------
    detName : str
        Name of the detector.
    ax : matplotlib.axes.Axes
        Axis to annotate.
    color : str, optional
        Color of the text.

    Returns
    -------
    matplotlib.text.Text
        The text artist added.
    """
    txt = ax.text(
        0.025,
        0.925,
        detName,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        weight="bold",
        color=color,
    )
    return txt


def clearAxisTicks(ax: plt.Axes, isSpine: bool = False) -> None:
    """
    Remove all ticks and tick labels from an axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to clear.
    is_spine : bool, optional
        If False, hide the axis spines as well.

    Returns
    -------
    None
    """
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if not isSpine:
        for spine in ax.spines.values():
            spine.set_visible(False)


def drawArrows(
    ax: plt.Axes,
    cutoutSize: int,
    rotAngle: float = 0,
) -> None:
    """
    Draw Alt/Az reference arrows on a single axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw arrows on.
    cutoutSize : int
        Size of the cutout.
    rotAngle : float, optional
        Rotation angle in degrees.

    Returns
    -------
    None
    """
    xmin1, ymin1 = draw_altaz_reference_arrow(ax, 0, cutout_size=cutoutSize)
    xmin2, ymin2 = draw_altaz_reference_arrow(
        ax,
        rot_angle=rotAngle,
        color="lightgrey",
        altlabel="Alt",
        azlabel="Az",
        cutout_size=cutoutSize,
    )

    ax.set_aspect("equal", adjustable="box")
    border = cutoutSize * 0.10
    xmin = np.min([xmin1, xmin2]) - border
    ymin = np.min([ymin1, ymin2]) - border
    ax.set_xlim(xmin, xmin + cutoutSize)
    ax.set_ylim(ymin, ymin + cutoutSize)
    clearAxisTicks(ax, isSpine=True)
    ax.set_axis_off()


def draw_altaz_reference_arrow(
    ax: plt.Axes,
    rot_angle: float = 0,
    altlabel: str = "Y DVCS",
    azlabel: str = "X DVCS",
    color: str = LIGHT_BLUE,
    length: Optional[float] = None,
    cutout_size: int = 30,
    center: Optional[tuple[float, float]] = None,
) -> tuple[float, float]:
    """
    Draw Alt/Az coordinate reference arrows in the bottom-left corner.
    The arrows indicate the Alt and Az directions,rotated by a specified angle.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on.
    rot_angle : float, optional
        Rotation angle in degrees.
    altlabel : str, optional
        Label for the Alt arrow.
    azlabel : str, optional
        Label for the Az arrow.
    color : str, optional
        Arrow color.
    length : float, optional
        Arrow length.
    cutout_size : int, optional
        Size of the cutout.
    center : tuple, optional
        Center coordinates for the arrows.

    Returns
    -------
    tuple of float
        Minimum x and y coordinates used for the arrows.
    """
    length = cutout_size / 4 if length is None else length

    theta = np.radians(rot_angle)
    dx_az, dy_az = length * np.cos(theta), length * np.sin(theta)
    dx_alt, dy_alt = -length * np.sin(theta), length * np.cos(theta)

    # Offset to lower-left corner
    if center is not None:
        x0, y0 = center
    else:
        x0, y0 = cutout_size // 2, cutout_size // 2

    # Az arrow
    ax.arrow(
        x0,
        y0,
        dx_az,
        dy_az,
        color=color,
        width=cutout_size / 120,
        head_width=cutout_size / 30,
        head_length=cutout_size / 20,
        length_includes_head=True,
        zorder=10,
    )
    ax.text(
        x0 + dx_az + 0.5,
        y0 + dy_az,
        azlabel,
        color=color,
        fontsize=9,
        fontweight="bold",
    )

    # Alt arrow
    ax.arrow(
        x0,
        y0,
        dx_alt,
        dy_alt,
        color=color,
        width=cutout_size / 120,
        head_width=cutout_size / 30,
        head_length=cutout_size / 20,
        length_includes_head=True,
        zorder=10,
    )
    ax.text(
        x0 + dx_alt + 0.5,
        y0 + dy_alt,
        altlabel,
        color=color,
        fontsize=9,
        fontweight="bold",
    )

    ax.set_xlim(x0, cutout_size)
    ax.set_ylim(y0, cutout_size)

    return np.nanmin([dx_alt + x0, dx_az + x0]), np.nanmin([dy_alt + y0, dy_az + y0])


def plot_crosshair_rotated(
    center: tuple[float, float],
    angle: float,
    axs: Optional[plt.Axes] = None,
    color: str = "grey",
    size: int = 30,
) -> None:
    """
    Plot a crosshair at a given center, rotated by the specified angle. The
    crosshair consists of two lines intersecting at the center.

    Parameters
    ----------
    center : tuple of float
        (x, y) coordinates of crosshair center.
    angle : float
        Rotation angle in degrees.
    axs : matplotlib.axes.Axes, optional
        Axis to plot on. If None, use current axis.
    color : str, optional
        Color of the crosshair.
    size : int, optional
        Size scaling for the crosshair.

    Returns
    -------
    None
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


def compute_camera_angle(df: pd.DataFrame, view: str = "ccd") -> float:
    """
    Compute the camera angle (degrees) from a DataFrame of star measurements.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of star measurements with columns for coordinates.
    view : str, optional
        Coordinate view ('ccd', 'dvcs', or 'fp').

    Returns
    -------
    float
        Camera rotation angle in degrees.
    """
    if len(df) < 3:
        raise ValueError("At least 3 stars are required to compute the camera angle.")

    if view == "ccd":
        xcol, ycol = "dx", "dy"
    elif view == "dvcs":
        # xcol, ycol = "dx_dvcs", "dy_dvcs"
        xcol, ycol = "dx", "dy"
    elif view == "fp":
        xcol, ycol = "dxfp", "dyfp"
    else:
        raise ValueError("view must be 'ccd', 'dvcs' of 'fp'.")

    dfn = df.dropna(subset=[xcol, ycol, "daz", "dalt"]).copy()
    if len(dfn) < 3:
        raise ValueError("At least 3 stars with valid coordinates are required.")
    xccd = dfn[xcol].values
    yccd = dfn[ycol].values
    az = dfn["daz"].values
    alt = dfn["dalt"].values

    angle_deg = compute_rotation_angle(xccd, yccd, az, alt)
    return angle_deg


def make_cutout(image, xcen, ycen, size=30) -> np.ndarray:
    """
    Create a square cutout from an image centered at (xcen, ycen).

    Parameters
    ----------
    image : np.ndarray
        Input image.
    xcen : float
        X center coordinate.
    ycen : float
        Y center coordinate.
    size : int, optional
        Size of the cutout (pixels).

    Returns
    -------
    np.ndarray
        Cutout image of shape (size, size).
    """
    if xcen is not None:
        x0, x1 = int(xcen - size / 2.0), int(xcen + size / 2.0)
        y0, y1 = int(ycen - size / 2.0), int(ycen + size / 2.0)
        cutout = image[y0:y1, x0:x1]
    else:
        cutout = np.zeros((size, size))
    return cutout


def plot_guide_circles(ax, center, radii, colors, labels=None, text_offset=1, **circle_kwargs) -> list:
    """
    Plot concentric guide circles on an axis. Each circle can have
    its own color and label, and all are centered at the specified position.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on.
    center : tuple
        Center coordinates (x, y).
    radii : list
        List of circle radii.
    colors : list
        List of colors for each circle.
    labels : list, optional
        Labels for each circle.
    text_offset : float, optional
        Offset for the label text.
    **circle_kwargs
        Additional keyword arguments for Circle.

    Returns
    -------
    list
        List of text artist objects for labels.
    """
    x0, y0 = center
    txt_list = []
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

        txt = ax.text(
            x0 + r + text_offset,
            y0 - r / 4.0,
            labels[i],
            color=colors[i],
            va="center",
            fontsize=8,
        )
        txt_list.append([txt])
    return txt_list


def compute_rotation_angle(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> float:
    """
    Compute the rotation angle (degrees) between two sets of coordinates.

    Parameters
    ----------
    x1, y1 : np.ndarray
        First set of coordinates.
    x2, y2 : np.ndarray
        Second set of coordinates.

    Returns
    -------
    float
        Rotation angle in degrees from (x1, y1) to (x2, y2).
    """
    # Centered coordinates
    Xa = np.vstack([x1 - np.mean(x1), y1 - np.mean(y1)])
    Xb = np.vstack([x2 - np.mean(x2), y2 - np.mean(y2)])

    # Rotation from 1 → 2
    try:
        U, _, Vt = np.linalg.svd(Xa @ Xb.T)
        R = U @ Vt
        theta = np.arctan2(R[1, 0], R[0, 0])
        angle_deg = np.degrees(theta) % 360.0
    except np.linalg.LinAlgError:
        angle_deg = np.nan
    return angle_deg
