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

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["PlotGuiderCCDStamps"]


class PlotGuiderCCDStamps:
    """Class to read and unpack the Guider data from Butler.
       Plot an animated gif of the CCD guider stamp.

    Example:
        from lsst.summit.utils.guiders.reading import readGuiderData
        from lsst.summit.utils.guiders.plotting import plotGuiderCCDStamps

        # Pick a seq number and dayObs
        seqNum, dayObs = 591, 20250425

        # Load the data from the butler
        reader = readGuiderData(seqNum, dayObs, view='dvcs', verbose=True)
        reader.init_guiders()
        reader.load_data()

        # Create the plotter object
        plotter = plotGuiderCCDStamps(reader)

        # Plot a stacked image of the stamps
        plotter.plot_stacked_stamp_array()

        # Plot a single stamp
        plotter.plot_stamp_array(stampNum=9)

        # Make a gif of the stamps
        plotter.make_gif(nStampsMax=50, fps=10)
    """

    def __init__(self, reader, butler=None, view="dvcs"):
        # reader object readGuiderData
        self.reader = reader
        self.view = reader.view
        self.dayObs = reader.dayObs
        self.seqNum = reader.seqNum
        self.expId = reader.expId
        self.nStamps = reader.nStamps
        self.detnames = reader.getGuiderNames()

        # for plotting
        self.layout = [
            [".", "R40_SG1", "R44_SG0", "."],
            ["R40_SG0", "center", ".", "R44_SG1"],
            ["R00_SG1", ".", ".", "R04_SG0"],
            [".", "R00_SG0", "R04_SG1", "."],
        ]

    def plot_stamp_ccd(self, raft_ccd_key, stampNum=-1, axs=None, plo=10.0, phi=99.0):
        if axs is None:
            axs = plt.gca()
            plt.title(f"{self.expId}")

        if stampNum < 0:
            img = self.reader.read_stacked(raft_ccd_key)
        else:
            img = self.reader.read(stampNum, raft_ccd_key)

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

    def get_stamp_number_info(self, stampNum=0):
        text = f"day_obs: {self.dayObs}" + "\n" + f"seq_num: {self.seqNum}" + "\n"
        text += f"orientation: {self.view}" + "\n"
        if stampNum > 0:
            text += f"Stamp #: {stampNum + 1:02d}"
        else:
            text += f"Stacked Image w/ {self.nStamps} stamps"
        return text

    def plot_stamp_info(self, stampNum=0, axs=None, more_text=None):
        if axs is None:
            axs = plt.gca()

        axs.set_xticks([])
        axs.set_yticks([])
        axs.set_axis_off()

        text = self.get_stamp_number_info(stampNum)
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

    def plot_text_ccd_name(self, detname, axs=None):
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

    def plot_stamp_array(self, stampNum=0, fig=None, axs=None, plo=90.0, phi=99.0):
        """Plot the stamp array for all the guiders.
        Args:
            stampNum (int): stamp number
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
            im = self.plot_stamp_ccd(detname, stampNum=stampNum, axs=axs[detname], plo=plo, phi=phi)
            txt = self.plot_text_ccd_name(detname, axs=axs[detname])
            artists.extend([im, txt])
        stamp_info = self.plot_stamp_info(axs=axs["center"], stampNum=stampNum)
        artists.append(stamp_info)
        return artists

    def plot_stacked_stamp_array(self, fig=None, axs=None):
        """Plot the stamp array for all the guiders.
        Args:
            stampNum (int): stamp number
            fig (matplotlib.figure.Figure): figure object
            axs (matplotlib.axes.Axes): axes object
        """
        artists = self.plot_stamp_array(stampNum=-1, fig=fig, axs=axs)
        return artists

    def make_gif(self, nStampsMax=10, fps=5, dpi=80):
        from matplotlib import animation

        # Create the animation
        fig, axs = plt.subplot_mosaic(
            self.layout,
            figsize=(9.5, 9.5),
            gridspec_kw=dict(hspace=0.0, wspace=0.0),
            constrained_layout=False,
        )

        nStamps = min(self.nStamps, nStampsMax)
        print("Number of stamps: ", nStamps)

        # plot the stacked image
        artists0 = self.plot_stacked_stamp_array(fig=fig, axs=axs)
        frame_list = 5 * [artists0]

        # loop over the stamps
        for i in range(nStamps):
            artists = self.plot_stamp_array(stampNum=i, fig=fig, axs=axs)
            frame_list.append(artists)

        frame_list += 5 * [artists0]

        # update the stamps
        ani = animation.ArtistAnimation(fig, frame_list, interval=1000 / fps, blit=True, repeat_delay=1000)
        ani.save(f"guider_ccd_array_{self.expId}.gif", fps=fps, dpi=dpi, writer="pillow")
        plt.close(fig)
        return ani

    def make_mp4(self, nStampsMax=10, fps=5, dpi=80):
        from matplotlib import animation

        # Create the animation
        fig, axs = plt.subplot_mosaic(
            self.layout,
            figsize=(9.5, 9.5),
            gridspec_kw=dict(hspace=0.0, wspace=0.0),
            constrained_layout=False,
        )

        nStamps = min(self.nStamps, nStampsMax)
        print("Number of stamps: ", nStamps)

        # plot the stacked image
        artists0 = self.plot_stacked_stamp_array(fig=fig, axs=axs)
        frame_list = 5 * [artists0]

        # loop over the stamps
        for i in range(nStamps):
            artists = self.plot_stamp_array(stampNum=i, fig=fig, axs=axs)
            frame_list.append(artists)
        frame_list += 5 * [artists0]

        # update the stamps
        ani = animation.ArtistAnimation(fig, frame_list, interval=1000 / fps, blit=True, repeat_delay=1000)
        ani.save(f"guider_ccd_array_{self.expId}.mp4", fps=fps, dpi=dpi)
        plt.close(fig)
        return ani

    # TODO: add the Alt/Az orientation
