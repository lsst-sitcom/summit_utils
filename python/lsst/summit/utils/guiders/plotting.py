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

from typing import Any, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.stats import mad_std
from matplotlib import animation
from matplotlib.patches import Circle

from lsst.summit.utils.guiders.reading import GuiderData
from lsst.summit.utils.guiders.transformation import convert_ccd_to_roi

__all__ = ["GuiderMosaicPlotter", "GuiderPlotter"]

LIGHT_BLUE = "#6495ED"


class GuiderPlotter:
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

    COLOR_MAP = ["black", "firebrick", "grey", "lightgrey"]
    MARKERS = [".", "x", "+", "s", "o", "^"]

    def __init__(self, stars_df: pd.DataFrame, expid: Optional[int] = None) -> None:
        self.exp_id = expid if expid else stars_df["expid"].iloc[0]
        self.stars_df = stars_df[stars_df["expid"] == self.exp_id]
        self.stats_df = self.assemble_stats()

        sns.set_style("white")
        sns.set_context("talk", font_scale=0.8)

    def assemble_stats(self) -> pd.DataFrame:
        stars = self.stars_df

        if stars.empty:
            cols = [
                "n_guiders",
                "n_stars",
                "fraction_valid_stamps",
                "n_measurements",
            ]
            example_std_centroid = [
                "std_centroid_az",
                "std_centroid_alt",
                "std_centroid_corr_az",
                "std_centroid_corr_alt",
                "offset_rate_az",
                "offset_rate_alt",
                "offset_zero_az",
                "offset_zero_alt",
            ]
            example_phot = ["magoffset_rate", "magoffset_zero", "magoffset_rms"]
            guider_names = stars["detector"].unique()
            cols += [f"N_{det}" for det in guider_names]
            cols += example_std_centroid + example_phot
            return pd.DataFrame(columns=cols)

        n_guiders = stars["detector"].nunique()
        n_unique = stars["starid"].nunique()
        counts = stars.groupby("detector")["starid"].nunique().to_dict()
        guider_names = stars["detector"].unique()
        stars_per_guiders = {f"N_{det}": counts.get(det, 0) for det in guider_names}

        mask_valid = (stars["stamp"] >= 0) & (stars["xccd"].notna())
        n_meas = int(mask_valid.sum())

        std_centroid = measure_std_centroid_stats(stars)
        phot = measure_photometric_variation(stars)

        total_possible = n_unique * stars["stamp"].nunique()
        frac_valid = n_meas / total_possible if total_possible > 0 else np.nan

        summary = {
            "expid": self.exp_id,
            "n_guiders": n_guiders,
            "n_stars": n_unique,
            "n_measurements": n_meas,
            "fraction_valid_stamps": frac_valid,
            **stars_per_guiders,
            **std_centroid,
            **phot,
        }
        return pd.DataFrame([summary])

    def print_metrics(self) -> None:
        filtered_stats_df = self.stats_df[self.stats_df["expid"] == self.exp_id].copy()

        print(self.format_stats_summary(filtered_stats_df))
        print(self.format_std_centroid_summary(filtered_stats_df))
        print(self.format_photometric_summary(filtered_stats_df))

    def strip_plot(self, plot_type: str = "centroidAltAz") -> None:
        # plot_kwargs dtype is dict[str, Any]
        plot_kwargs: dict[str, dict] = {
            "centroidAltAz": {
                "ylabel": "Centroid Offset [arcsec]",
                "col": ["dalt", "daz"],
                "title": "Alt/Az Centroid Offsets",
            },
            "centroidPixel": {
                "ylabel": "Centroid Offset [pixels]",
                "col": ["dx", "dy"],
                "title": "CCD Pixel Centroid Offsets",
            },
            "flux": {
                "ylabel": "Magnitude Offset [mag]",
                "col": ["magoffset"],
                "title": "Flux Magnitude Offsets",
            },
            "psf": {
                "ylabel": "PSF FWHM [arcsec]",
                "col": ["fwhm"],
                "scale": 0.2,
                "title": "PSF FWHM",
            },
        }
        cfg = plot_kwargs[plot_type]  # type: dict[str, Any]
        scale = cfg.get("scale", 1.0)  # type: float
        cols = cfg["col"]

        # filter and prepare
        df = self.stars_df[self.stars_df["stamp"] > 0][["stamp"] + cols].copy()

        # Compute overall mean and sigma for y-axis limits
        all_data = df[cols].values.flatten()
        all_data *= scale
        mean_val = np.nanmean(all_data)
        sigma_val = np.nanstd(all_data)
        ylims = (mean_val - 5 * sigma_val, mean_val + 5 * sigma_val)

        # setup subplots
        n = len(cols)
        fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(7 * n + 1, 6), sharex=True, sharey=True)
        if n == 1:
            axes = [axes]

        # define bins
        max_stamp = int(df["stamp"].max()) + 1
        bins = np.arange(1, max_stamp + 6, 5)
        bin_centers = bins[:-1] + 2.5

        for ax, col in zip(axes, cols):
            # scatter points
            ax.scatter(df["stamp"], df[col] * scale, color="lightgrey", alpha=0.7, label=col)

            # compute binned stats
            df["bin"] = pd.cut(df["stamp"], bins=bins, labels=False)
            stats = df.groupby("bin")[col].agg(["mean", "std"]).reset_index()

            valid = stats["bin"].notna()
            bin_idx = stats.loc[valid, "bin"].astype(int).values
            means = stats.loc[valid, "mean"].values * scale
            errs = stats.loc[valid, "std"].values * scale

            # plot error bars
            ax.errorbar(
                bin_centers[bin_idx],
                means,
                yerr=errs,
                fmt="o",
                color=LIGHT_BLUE,
                ecolor=LIGHT_BLUE,
                capsize=3,
            )
            ax.set_ylabel(cfg["ylabel"])
            ax.set_xlabel("# stamp")
            ax.set_ylim(*ylims)  # <-- Set y-axis limits here
            ax.legend(fontsize=12, loc="upper right")

            # top axis for elapsed time
            xstampers = np.arange(0, max_stamp + 10, 10)
            ax.set_xticks(xstampers)
            ax2 = ax.twiny()
            elapsed = 15.0 * xstampers / max_stamp
            ax2.set_xticks(xstampers)
            ax2.set_xticklabels([f"{e:.1f}" for e in elapsed])
            ax2.set_xlabel("Elapsed time [s]")

        fig.suptitle(cfg["title"], fontsize=14, fontweight="bold")
        fig.tight_layout()
        # return fig

    def select_best_star(self, guiderData: GuiderData) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Convert the best star centroid in each guider to ROI coordinates.

        The conversion depends on the guider view (dvcs, ccd, etc).

        Returns:
            centroids: dict of {detector: (xroi, yroi)}
        """
        centroids: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for det in self.DETNAMES:
            sub = self.stars_df[self.stars_df["detector"] == det]
            if len(sub) > 0:
                best = sub.loc[sub["snr"].idxmax()]
                xccd = np.array([best["xccd_ref"]])
                yccd = np.array([best["yccd_ref"]])

                args = (xccd, yccd, guiderData, det)
                centroids[det] = convert_ccd_to_roi(*args)
            else:
                centroids[det] = (np.array([]), np.array([]))
        self.centroids = centroids
        return centroids

    def load_image(self, guider: GuiderData, detname: str, stamp_num: int = 2) -> np.ndarray:
        # read full stamp
        if stamp_num >= len(guider.timestamps):
            raise ValueError(
                f"stamp_num {stamp_num} is out of range for guider with {len(guider.timestamps)} stamps."
            )
        elif stamp_num < 0:
            return guider.getStackedStampArray(detName=detname)
        else:
            img = guider.getStampArray(stampNum=stamp_num, detName=detname)
            return img - np.nanmedian(img, axis=0)

    def star_mosaic(
        self,
        guiderData: GuiderData,
        stamp_num: int = 2,
        fig: Optional[plt.Figure] = None,
        axs=None,
        plo=90.0,
        phi=99.0,
        cutout_size=30,
    ) -> list:
        """Plot the stamp array for all the guiders.
        Args:
            guider (GuiderData): guider data object
            stamp_num (int): stamp number
            fig (matplotlib.figure.Figure): figure object
            axs (matplotlib.axes.Axes): axes object
        """
        # the guider view should be 'dvcs'
        if guiderData.view != "dvcs":
            raise ValueError("Guider view must be 'dvcs' for mosaic plotting.")

        if fig is None:
            gs = dict(hspace=0.0, wspace=0.0)
            fig, axs = plt.subplot_mosaic(
                cast(Any, self.LAYOUT),
                figsize=(9.5, 9.5),
                gridspec_kw=gs,
                constrained_layout=False,
            )

        if not hasattr(self, "centroids"):
            self.select_best_star(guiderData)

        artists = []
        for detname in self.DETNAMES:
            xcen, ycen = self.centroids.get(detname, (None, None))

            img = self.load_image(guiderData, detname, stamp_num)
            cutout = make_cutout(img, xcen, ycen, size=cutout_size)
            vmin, vmax = np.nanpercentile(cutout, plo), np.nanpercentile(cutout, phi)

            axs_img = axs[detname]
            im_object = axs_img.imshow(
                cutout, origin="lower", cmap="Greys", animated=True, vmin=vmin, vmax=vmax
            )
            txt_object = self.annotate_detector(detname, axs_img)

            center = (cutout_size / 2.0, cutout_size / 2.0)
            # if xcen is not None:
            # crosshairs
            axs_img.axvline(cutout_size / 2.0, color="grey", linestyle="--", linewidth=1)
            axs_img.axhline(cutout_size / 2.0, color="grey", linestyle="--", linewidth=1)
            axs_img.set_aspect("equal", "box")
            _ = plot_guide_circles(
                axs_img,
                center,
                radii=[5, 10],
                colors=[LIGHT_BLUE, LIGHT_BLUE],
                labels=["1″", "2″"],
                linewidth=2.0,
            )

            artists.extend([im_object, txt_object])

        # Annotate the center
        std = np.nanstd(np.hypot(self.stars_df["dalt"], self.stars_df["daz"]))
        stamp_info = self.annotate_center(stamp_num, axs["center"], jitter=std)
        axs["center"].axis("off")
        artists.append(stamp_info)

        xmin1, ymin1 = draw_altaz_reference_arrow(axs["arrow"], cutout_size=cutout_size)
        xmin2, ymin2 = draw_altaz_reference_arrow(
            axs["arrow"],
            -self.rot_angle,
            color="lightgrey",
            altlabel=" y",
            azlabel="x",
            cutout_size=cutout_size,
        )
        axs["arrow"].axis("off")
        xmin = np.min([xmin1, xmin2]) - 3
        ymin = np.min([ymin1, ymin2]) - 3
        axs["arrow"].set_xlim(xmin, xmin + cutout_size)
        axs["arrow"].set_ylim(ymin, ymin + cutout_size)

        # Clear ticks, labels, and remove borders
        for ax in axs.values():
            self.clear_axis_ticks(ax)
            for spine in ax.spines.values():
                spine.set_visible(False)

        return artists

    def clear_axis_ticks(self, ax) -> None:
        """Remove all ticks and tick labels from an axis."""
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    def annotate_detector(self, detname, ax) -> plt.Text:
        """Annotate a detector panel with its name."""
        txt = ax.text(
            0.025,
            0.025,
            detname,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            weight="bold",
            color="grey",
        )
        return txt

    def annotate_center(self, stamp_num, ax, jitter=-1) -> plt.Text:
        """Annotate the center panel with exposure and stamp info."""
        self.clear_axis_ticks(ax)
        text = (
            f"expid: {self.exp_id}\nStamp #: {stamp_num + 1:02d}"
            if stamp_num >= 0
            else f"expid: {self.exp_id}\nStacked w/ {self.stars_df['stamp'].nunique()} stamps"
        )
        text += f"\nCenter Stdev.: {jitter:.2f} arcsec" if jitter > 0 else ""
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

    def plot_circle(self, ax, xcen, ycen, radius=5, color="firebrick", lw=1.0) -> Circle:
        """
        Add a circular patch at (xcen, ycen) with given radius on the axis.
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

    def make_gif(
        self,
        guider: GuiderData,
        n_stamp_max=60,
        fps=5,
        dpi=80,
        plo=90.0,
        phi=99.0,
        cutout_size=30,
    ) -> animation.ArtistAnimation:
        # the guider view should be 'dvcs'
        if guider.view != "dvcs":
            raise ValueError("Guider view must be 'dvcs' for mosaic GIF creation.")

        from matplotlib import animation

        # build canvas
        fig, axs = plt.subplot_mosaic(
            cast(Any, self.LAYOUT),
            figsize=(10, 10),
            gridspec_kw=dict(hspace=0.0, wspace=0.0),
            constrained_layout=False,
        )

        # number of frames
        n_stamps = len(guider.timestamps)
        total = min(n_stamps, n_stamp_max)

        print("Number of stamps: ", total)
        # initial (stacked) frame
        artists0 = self.star_mosaic(
            guider,
            stamp_num=-1,
            fig=fig,
            axs=axs,
            plo=plo,
            phi=phi,
            cutout_size=cutout_size,
        )

        frames = 2 * [artists0]

        # sequential stamps
        for i in range(1, total):
            artists = self.star_mosaic(
                guider,
                stamp_num=i,
                fig=fig,
                axs=axs,
                plo=plo,
                phi=phi,
                cutout_size=cutout_size,
            )
            frames.append(artists)
        frames += 2 * [artists0]

        # create animation
        ani = animation.ArtistAnimation(fig, frames, interval=1000 / fps, blit=True, repeat_delay=1000)
        filepath = f"guider_mosaic_{self.exp_id}.gif"
        ani.save(filepath, fps=fps, dpi=dpi, writer="pillow")
        plt.close(fig)
        return ani

    @staticmethod
    def format_std_centroid_summary(stats_df: pd.DataFrame) -> str:
        """
        Pretty string summary of centroid stdev. stats from run_all_guiders.
        """
        # handle both dicts and DataFrames
        if (isinstance(stats_df, pd.DataFrame) and stats_df.empty) or (
            isinstance(stats_df, dict) and not stats_df
        ):
            return "No centroid stdev. statistics available."

        # if it's a DataFrame, extract the one-row dict
        if isinstance(stats_df, pd.DataFrame):
            stats = stats_df.iloc[0].to_dict()
        else:
            stats = stats_df

        js = stats
        summary = (
            f"\nGlobal centroid stdev. Summary Across All Guiders\n"
            f"{'-' * 45}\n"
            f"  - centroid stdev.  (AZ): {js['std_centroid_az']:.3f} arcsec (raw)\n"
            f"  - centroid stdev. (ALT): {js['std_centroid_alt']:.3f} arcsec (raw)\n"
            f"  - centroid stdev.  (AZ): {js['std_centroid_corr_az']:.3f} arcsec (linear corr)\n"
            f"  - centroid stdev. (ALT): {js['std_centroid_corr_alt']:.3f} arcsec (linear corr)\n"
            f"  - Drift Rate       (AZ): {15 * js['drift_rate_az']:.3f} arcsec per exposure\n"
            f"  - Drift Rate      (ALT): {15 * js['drift_rate_alt']:.3f} arcsec per exposure\n"
            f"  - Zero Offset      (AZ): {js['offset_zero_az']:.3f} arcsec\n"
            f"  - Zero Offset     (ALT): {js['offset_zero_alt']:.3f} arcsec"
        )
        return summary

    @staticmethod
    def format_photometric_summary(phot_stats: pd.DataFrame) -> str:
        """
        Pretty-print summary of photometric variation statistics.
        """
        if (isinstance(phot_stats, pd.DataFrame) and phot_stats.empty) or (
            isinstance(phot_stats, dict) and not phot_stats
        ):
            return "No photometric statistics available."

        # if it's a DataFrame, extract the one-row dict
        if isinstance(phot_stats, pd.DataFrame):
            stats = phot_stats.iloc[0].to_dict()
        else:
            stats = phot_stats

        return (
            "\nPhotometric Variation Summary\n"
            "-------------------------------\n"
            f"  - Mag Drift Rate:      {stats['magoffset_rate']:.5f} mag/sec\n"
            f"  - Mag Zero Offset:     {stats['magoffset_zero']:.5f} mag\n"
            f"  - Mag RMS (detrended): {stats['magoffset_rms']:.5f} mag"
        )

    @staticmethod
    def format_stats_summary(summary: pd.DataFrame) -> str:
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

        # Per-guider counts (keys begin with 'N_')
        guider_keys = sorted(k for k in summary if k.startswith("N_"))
        if guider_keys:
            lines.append("\nStars per Guider:")
            for k in guider_keys:
                lines.append(f"  - {k[2:]}: {int(summary[k])}")
        return "\n".join(lines)


def make_cutout(image, xcen, ycen, size=30) -> np.ndarray:
    if xcen is not None:
        x0, x1 = int(xcen - size / 2.0), int(xcen + size / 2.0)
        y0, y1 = int(ycen - size / 2.0), int(ycen + size / 2.0)
        cutout = image[y0:y1, x0:x1]
    else:
        cutout = np.zeros((size, size))
    return cutout


def plot_guide_circles(ax, center, radii, colors, labels=None, text_offset=1, **circle_kwargs) -> list:
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


class GuiderMosaicPlotter:
    """Class to read and unpack the Guider data from Butler.
       Plot an animated gif of the CCD guider stamp.

    Example:
        from lsst.summit.utils.guiders.reading import readGuiderData
        from lsst.summit.utils.guiders.plotting import GuiderMosaicPlotter

        # Pick a seq number and dayObs
        seqNum, dayObs = 591, 20250425

        # Load the data from the butler
        reader = readGuiderData(seqNum, dayObs, view='dvcs', verbose=True)
        reader.init_guiders()
        reader.load_data()

        # Create the plotter object
        plotter = GuiderMosaicPlotter(reader)

        # Plot a stacked image of the stamps
        plotter.plot_stacked_stamp_array()

        # Plot a single stamp
        plotter.plot_stamp_array(stamp_num=9)

        # Make a gif of the stamps
        plotter.make_gif(n_stamp_max=50, fps=10)
    """

    def __init__(self, reader, butler=None, view="dvcs"):
        # reader object readGuiderData
        self.reader = reader
        self.view = reader.view
        self.dayObs = reader.dayObs
        self.seqNum = reader.seqNum
        self.exp_id = reader.expid
        self.n_stamps = reader.nStamps
        self.detnames = reader.getGuiderNames()

        # for plotting
        self.layout = [
            [".", "R40_SG1", "R44_SG0", "."],
            ["R40_SG0", "center", ".", "R44_SG1"],
            ["R00_SG1", ".", ".", "R04_SG0"],
            [".", "R00_SG0", "R04_SG1", "."],
        ]

    def plot_stamp_ccd(self, raft_ccd_key, stamp_num=-1, axs=None, plo=10.0, phi=99.0) -> plt.AxesImage:
        if axs is None:
            axs = plt.gca()
            plt.title(f"{self.exp_id}")

        if stamp_num < 0:
            img = self.reader.read_stacked(raft_ccd_key)
        else:
            img = self.reader.read(stamp_num, raft_ccd_key)

        bias = np.median(img)
        img_isr = img - bias
        lo, hi = np.nanpercentile(img_isr, [plo, phi])

        im = axs.imshow(img_isr, origin="lower", cmap="Greys", vmin=lo, vmax=hi, animated=True)
        axs.set_yticklabels([])
        axs.set_xticklabels([])
        axs.set_xticks([])
        axs.set_xticks([], minor=True)
        axs.set_yticks([])
        axs.set_yticks([], minor=True)
        return im

    def get_stamp_number_info(self, stamp_num=0) -> str:
        text = f"day_obs: {self.dayObs}" + "\n" + f"seq_num: {self.seqNum}" + "\n"
        text += f"orientation: {self.view}" + "\n"
        if stamp_num > 0:
            text += f"Stamp #: {stamp_num + 1:02d}"
        else:
            text += f"Stacked Image w/ {self.n_stamps} stamps"
        return text

    def plot_stamp_info(self, stamp_num=0, axs=None, more_text=None) -> plt.Text:
        if axs is None:
            axs = plt.gca()

        axs.set_xticks([])
        axs.set_yticks([])
        axs.set_axis_off()

        text = self.get_stamp_number_info(stamp_num)
        if more_text is not None:
            text += "\n" + more_text

        stamp_id_text = axs.text(
            1.085,
            -0.10,
            text,
            transform=axs.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="firebrick",
            animated=True,
        )
        axs.set_axis_off()
        self.stamp_id_axs = stamp_id_text
        self.stamp_id_more_text = more_text
        return stamp_id_text

    def plot_text_ccd_name(self, detname, axs=None) -> plt.Text:
        if axs is None:
            axs = plt.gca()
        txt = axs.text(
            0.025,
            0.025,
            detname,
            transform=axs.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            weight="bold",
            color="grey",
        )
        return txt

    def plot_stamp_array(self, stamp_num=0, fig=None, axs=None, plo=90.0, phi=99.0) -> list:
        """Plot the stamp array for all the guiders.
        Args:
            stamp_num (int): stamp number
            fig (matplotlib.figure.Figure): figure object
            axs (matplotlib.axes.Axes): axes object
        """
        if fig is None:
            gs = dict(hspace=0.0, wspace=0.0)
            fig, axs = plt.subplot_mosaic(
                self.layout, figsize=(9.5, 9.5), gridspec_kw=gs, constrained_layout=False
            )

        artists = []
        for detname in self.detnames:
            im = self.plot_stamp_ccd(detname, stamp_num=stamp_num, axs=axs[detname], plo=plo, phi=phi)
            txt = self.plot_text_ccd_name(detname, axs=axs[detname])
            artists.extend([im, txt])
        stamp_info = self.plot_stamp_info(axs=axs["center"], stamp_num=stamp_num)
        artists.append(stamp_info)
        return artists

    def plot_stacked_stamp_array(self, fig=None, axs=None) -> list:
        """Plot the stamp array for all the guiders.
        Args:
            stamp_num (int): stamp number
            fig (matplotlib.figure.Figure): figure object
            axs (matplotlib.axes.Axes): axes object
        """
        artists = self.plot_stamp_array(stamp_num=-1, fig=fig, axs=axs)
        return artists

    def make_gif(self, n_stamp_max=10, fps=5, dpi=80) -> animation.ArtistAnimation:
        # Create the animation
        fig, axs = plt.subplot_mosaic(
            self.layout,
            figsize=(9.5, 9.5),
            gridspec_kw=dict(hspace=0.0, wspace=0.0),
            constrained_layout=False,
        )

        nStamps = min(self.n_stamps, n_stamp_max)
        print("Number of stamps: ", nStamps)

        # plot the stacked image
        artists0 = self.plot_stacked_stamp_array(fig=fig, axs=axs)
        frame_list = 5 * [artists0]

        # loop over the stamps
        for i in range(nStamps):
            artists = self.plot_stamp_array(stamp_num=i, fig=fig, axs=axs)
            frame_list.append(artists)

        frame_list += 5 * [artists0]

        # update the stamps
        ani = animation.ArtistAnimation(fig, frame_list, interval=1000 / fps, blit=True, repeat_delay=1000)
        ani.save(f"guider_ccd_array_{self.exp_id}.gif", fps=fps, dpi=dpi, writer="pillow")
        plt.close(fig)
        return ani

    def make_mp4(self, n_stamp_max=10, fps=5, dpi=80):
        from matplotlib import animation

        # Create the animation
        fig, axs = plt.subplot_mosaic(
            self.layout,
            figsize=(9.5, 9.5),
            gridspec_kw=dict(hspace=0.0, wspace=0.0),
            constrained_layout=False,
        )

        nStamps = min(self.n_stamps, n_stamp_max)
        print("Number of stamps: ", nStamps)

        # plot the stacked image
        artists0 = self.plot_stacked_stamp_array(fig=fig, axs=axs)
        frame_list = 5 * [artists0]

        # loop over the stamps
        for i in range(nStamps):
            artists = self.plot_stamp_array(stamp_num=i, fig=fig, axs=axs)
            frame_list.append(artists)
        frame_list += 5 * [artists0]

        # update the stamps
        ani = animation.ArtistAnimation(fig, frame_list, interval=1000 / fps, blit=True, repeat_delay=1000)
        ani.save(f"guider_ccd_array_{self.exp_id}.mp4", fps=fps, dpi=dpi)
        plt.close(fig)
        return ani

    # TODO: add the Alt/Az orientation


def measure_std_centroid_stats(stars: pd.DataFrame) -> pd.DataFrame:
    """
    Compute global std_centroid statistics across all guiders.

    Parameters
    ----------
    stars : pd.DataFrame
        Concatenated star table across all guider detectors.

    Returns
    -------
    stars : pd.DataFrame
        DataFrame with new std_centroid statistic
        columns broadcasted to all rows.
    """
    time = (stars.stamp.to_numpy() + 0.5) * 0.3  # seconds
    time = time.astype(np.float64)
    az = stars.daz.to_numpy()
    alt = stars.dalt.to_numpy()

    # Linear fits
    coefs_az = np.polyfit(time, az, 1)
    coefs_alt = np.polyfit(time, alt, 1)

    # Stats
    std_centroid_stats = {
        "std_centroid_az": mad_std(az),
        "std_centroid_alt": mad_std(alt),
        "std_centroid_corr_az": mad_std(az - np.polyval(coefs_az, time)),
        "std_centroid_corr_alt": mad_std(alt - np.polyval(coefs_alt, time)),
        "drift_rate_az": coefs_az[0],
        "drift_rate_alt": coefs_alt[0],
        "offset_zero_az": coefs_az[1],
        "offset_zero_alt": coefs_alt[1],
    }
    return std_centroid_stats


def measure_photometric_variation(stars: pd.DataFrame) -> dict[str, float]:
    """
    Fit magoffset vs time across all rows, compute drift rate,
      zero-point, and RMS scatter,
    then add these as constant columns to `stars`.
    """
    mo = stars["magoffset"].to_numpy()
    mask = np.isfinite(mo)
    if np.sum(mask) < 3:
        phot_stats = {
            "magoffset_rate": np.nan,
            "magoffset_zero": np.nan,
            "magoffset_rms": np.nan,
        }
    else:
        time = (stars["stamp"].to_numpy()[mask] + 0.5) * 0.3  # seconds
        mo_valid = mo[mask]
        coef = np.polyfit(np.asarray(time, dtype=float), np.asarray(mo_valid, dtype=float), 1)
        rate, zero = coef
        resid = mo_valid - np.polyval(coef, time)
        rms = mad_std(resid)
        phot_stats = {
            "magoffset_rate": rate,
            "magoffset_zero": zero,
            "magoffset_rms": rms,
        }
    return phot_stats


def draw_altaz_reference_arrow(
    ax,
    rot_angle=0,
    altlabel="Alt",
    azlabel="az",
    color=LIGHT_BLUE,
    length=None,
    cutout_size=30,
    center=(None, None),
) -> tuple[float, float]:
    """Draw Alt/Az coordinate reference arrows in the bottom-left corner."""
    length = cutout_size / 4 if length is None else length
    # x0, y0 = cutout_size // 2, cutout_size // 2

    theta = np.radians(rot_angle)
    dx_az, dy_az = length * np.cos(theta), length * np.sin(theta)
    dx_alt, dy_alt = -length * np.sin(theta), length * np.cos(theta)

    # Offset to lower-left corner
    if center[0] is not None and center[1] is not None:
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
        width=0.25,
        head_width=1.0,
        head_length=1.5,
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
        width=0.25,
        head_width=1.0,
        head_length=1.5,
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

    return np.min([dx_alt + x0, dx_az + x0]), np.min([dy_alt + y0, dy_az + y0])
