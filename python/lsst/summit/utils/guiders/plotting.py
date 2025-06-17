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

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from astropy.stats import mad_std

import pandas as pd
import numpy as np

__all__ = ["plotGuiderCCDStamps", "GuiderPlotter"]

class GuiderPlotter:
    UNIT_DICT = {
        'centroidAltAz': 'arcsec',
        'centroidPixel': 'pixels',
        'flux': 'magnitudes',
        'secondMoments': 'pixels²',
        'psf': 'arcsec'
    }

    # for plotting
    LAYOUT =  [
               [      ".", "R40_SG1", "R44_SG0",       "."],
               ["R40_SG0",       "center",       ".", "R44_SG1"],
               ["R00_SG1",       ".",       ".", "R04_SG0"],
               [      ".", "R00_SG0", "R04_SG1",       "."],
    ]

    DETNAMES = [cell for row in LAYOUT for cell in row if (cell != ".")&(cell!="center")]
       
    COLOR_MAP = ['black', 'firebrick', 'grey', 'lightgrey']
    MARKERS = ['.', 'x', '+', 's', 'o', '^']

    def __init__(self, stars_df, stats_df=None, expId=None):
        self.expId = expId if expId else stars_df['expId'].iloc[0]
        self.stars_df = stars_df[stars_df['expId'] == self.expId]

        if stats_df is None:
            self.stats_df = self.assemble_stats()
        else:
            self.stats_df = stats_df

        sns.set_style('white')
        sns.set_context('talk', font_scale=0.8)

    def assemble_stats(self) -> pd.DataFrame:
        stars = self.stars_df

        if stars.empty:
            cols = [
                'n_guiders',
                'n_unique_stars',
                'fraction_valid_stamps',
                'n_measurements',
            ]
            example_std_centroid = [
                'std_centroid_az', 'std_centroid_alt', 'std_centroid_corr_az', 'std_centroid_corr_alt',
                'offset_rate_az', 'offset_rate_alt', 'offset_zero_az', 'offset_zero_alt'
            ]
            example_phot = ['mag_offset_rate', 'mag_offset_zero', 'mag_offset_rms']
            guider_names = stars['detector'].unique()
            cols += [f'N_{det}' for det in guider_names] + example_std_centroid + example_phot
            return pd.DataFrame(columns=cols)

        n_guiders = stars['detector'].nunique()
        n_unique = stars['star_id'].nunique()
        counts = stars.groupby('detector')['star_id'].nunique().to_dict()
        guider_names = stars['detector'].unique()
        stars_per_guiders = {f'N_{det}': counts.get(det, 0) for det in guider_names}

        mask_valid = (stars['stamp'] >= 0) & (stars['xpixel'].notna())
        n_meas = int(mask_valid.sum())

        std_centroid = measure_std_centroid_stats(stars)
        phot = measure_photometric_variation(stars)

        total_possible = n_unique * stars['stamp'].nunique()
        frac_valid = n_meas / total_possible if total_possible > 0 else np.nan

        summary = {
            'n_guiders': n_guiders,
            'n_unique_stars': n_unique,
            'n_measurements': n_meas,
            'fraction_valid_stamps': frac_valid,
            **stars_per_guiders,
            **std_centroid,
            **phot
        }
        return pd.DataFrame([summary])

    def print_metrics(self):
        num_stars = self.stats_df['n_unique_stars'].sum()
        print(f"Number of stars: {num_stars}\n")
        print("Metrics summary:")
        print(self.stats_df)

    def strip_plot(self, plot_type='centroidAltAz'):
        plot_columns = {
            'centroidAltAz': ['dalt', 'daz'],
            'centroidPixel': ['dx', 'dy'],
            'flux': ['mag_offset'],
            'secondMoments': ['ixx', 'iyy', 'ixy'],
            'psf': ['e1', 'e2', 'fwhm']
        }

        data = self.stars_df[self.stars_df['stamp'] > 0][['stamp'] + plot_columns[plot_type]].copy()
        # for col in plot_columns[plot_type]:
        #     data[col] -= data[col].median()
            
        melted = data.melt(id_vars='stamp', var_name='Measurement', value_name='value')

        fig, ax1 = plt.subplots(figsize=(8, 6))
        measurements = melted['Measurement'].unique()
        for i, measure in enumerate(measurements):
            subset = melted[melted['Measurement'] == measure]
            std = np.nanstd(subset['value'])
            ax1.scatter(subset['stamp'], subset['value'],
                        color=self.COLOR_MAP[i % len(self.COLOR_MAP)],
                        marker=self.MARKERS[i % len(self.MARKERS)],
                        alpha=0.7,
                        label=f"{measure} (rms: {std:.3f} {self.UNIT_DICT[plot_type]})")

        ax1.axhline(0, color='grey', ls='--')
        ax1.set_xlabel('# stamp')
        ax1.set_ylabel(f"value-median [{self.UNIT_DICT[plot_type]}]")

        stamp_unique = np.unique(melted['stamp'])
        ax1.set_xticks(stamp_unique[::5])

        # Second x-axis for elapsed time
        ax2 = ax1.twiny()
        ax2.set_xticks(ax1.get_xticks())
        elapsed_time = 15 * (ax1.get_xticks()+1) / (stamp_unique.max()+1)
        ax2.set_xticklabels([f'{et:.1f}' for et in elapsed_time])
        ax2.set_xlabel('Elapsed time [s]')

        plt.title(f"Strip plot: {plot_type}\nExpId: {self.expId}")
        ax1.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

    def select_best_star(self) -> dict:
        """
        Build a dict mapping each detector to the centroid of the star with highest SNR.
        If no stars for a detector, value is (None, None).
        Returns:
            centroids: dict of {detector: (xcentroid, ycentroid)}
        """
        centroids = {}
        detectors = self.DETNAMES
        for det in detectors:
            sub = self.stars_df[self.stars_df['detector'] == det]
            if len(sub)>0:
                best = sub.loc[sub['snr'].idxmax()]
                centroids[det] = (best['xcentroid_ref'], best['ycentroid_ref'])
            else:
                centroids[det] = (None, None)
        self.centroids = centroids
        pass

    def load_image(self, reader, detname, stampNum=2):
        # read full stamp
        img = reader.read(stampNum, detname) if stampNum >= 0 else reader.read_stacked(detname)
        return img - np.nanmedian(img, axis=0)

    def star_mosaic(self, reader, stampNum=2, fig=None, axs=None, plo=90., phi=99., cutout_size=30):
        """Plot the stamp array for all the guiders.
        Args:
            stampNum (int): stamp number
            fig (matplotlib.figure.Figure): figure object
            axs (matplotlib.axes.Axes): axes object
        """
        if fig is None:
            gs = dict(hspace=0.0, wspace=0.0)
            fig, axs = plt.subplot_mosaic(self.LAYOUT, figsize=(9.5, 9.5), gridspec_kw=gs, constrained_layout=False)
            
        if not hasattr(self, 'centroids'):
            self.select_best_star()

        artists = []
        for detname in self.DETNAMES:
            xcen, ycen = self.centroids.get(detname, (None, None))

            img = self.load_image(reader, detname, stampNum)
            cutout = make_cutout(img, xcen, ycen, size=cutout_size)
            vmin, vmax = np.nanpercentile(cutout, plo), np.nanpercentile(cutout, phi)

            axs_img = axs[detname]
            im_object = axs_img.imshow(cutout, origin='lower', cmap='Greys', animated=True, vmin=vmin, vmax=vmax)
            txt_object = self.annotate_detector(detname, axs_img)

            center  = (cutout_size/2., cutout_size/2.)
            # if xcen is not None:
            # crosshairs
            axs_img.axvline(cutout_size/2., color='grey', linestyle='--', linewidth=1)
            axs_img.axhline(cutout_size/2., color='grey', linestyle='--', linewidth=1)
            axs_img.set_aspect('equal', 'box')
            # self.plot_circle(axs_img, cutout_size/2, cutout_size/2, radius=5, color='grey')
            # self.plot_circle(axs_img, cutout_size/2, cutout_size/2, radius=10, color='grey')
            circle_object = plot_guide_circles(axs_img, center, radii=[5,10],
                colors=['firebrick','firebrick'],
                labels=['1″','2″'],
                linewidth=1)

            artists.extend([im_object, txt_object])

        # Annotate the center
        stamp_info = self.annotate_center(stampNum, axs['center'])
        axs['center'].axis('off')
        artists.append(stamp_info)
    
        # Clear ticks and labels
        for ax in axs.values():
            self.clear_axis_ticks(ax)
        return artists

    def clear_axis_ticks(self, ax):
        """Remove all ticks and tick labels from an axis."""
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    def annotate_detector(self, detname, ax):
        """Annotate a detector panel with its name."""
        txt = ax.text(
            0.025, 0.025, detname,
            transform=ax.transAxes,
            ha='left', va='bottom',
            fontsize=9, weight='bold', color='grey'
        )
        return txt

    def annotate_center(self, stampNum, ax):
        """Annotate the center panel with exposure and stamp info."""
        self.clear_axis_ticks(ax)
        text = f"ExpId: {self.expId}\nStamp #: {stampNum+1:02d}" if stampNum>=0 else f"ExpId: {self.expId}\nStacked w/ {self.stars_df['stamp'].nunique()} stamps"
        txt = ax.text(
            1.085, -0.10, text,
            transform=ax.transAxes,
            ha='center', va='center',
            fontsize=14, color='firebrick'
        )
        return txt
        
    def plot_circle(self, ax, xcen, ycen, radius=5, color='firebrick', lw=1.0):
        """
        Add a circular patch at (xcen, ycen) with given radius on the axis.
        """
        circ = Circle((xcen, ycen), radius=radius,
                      edgecolor=color, facecolor='none', lw=lw, ls='--')
        ax.add_patch(circ)
        return circ

    def make_gif(self, reader, nStampsMax=60, fps=5, dpi=80,
                 plo=90., phi=99., cutout_size=30):
        from matplotlib import animation
        # build canvas
        fig, axs = plt.subplot_mosaic(
            self.LAYOUT,
            figsize=(10, 10),
            gridspec_kw=dict(hspace=0.0, wspace=0.0),
            constrained_layout=False
        )
        # number of frames
        total = min(reader.nStamps, nStampsMax)
        print("Number of stamps: ", total)
        # initial (stacked) frame
        artists0 = self.star_mosaic(reader, stampNum=-1, fig=fig, axs=axs,
                                    plo=plo, phi=phi, cutout_size=cutout_size)

        frames = 2 * [artists0]
        
        # sequential stamps
        for i in range(1,total):
            artists = self.star_mosaic(reader, stampNum=i, fig=fig, axs=axs, 
                                       plo=plo, phi=phi, cutout_size=cutout_size)
            frames.append(artists)
        frames += 2 * [artists0]
        
        # create animation
        ani = animation.ArtistAnimation(
            fig, frames,
            interval=1000/fps, blit=True,
            repeat_delay=1000
        )
        filepath = f"guider_mosaic_{self.expId}.gif"
        ani.save(filepath, fps=fps, dpi=dpi, writer="pillow")
        plt.close(fig)
        return ani
    
def make_cutout(image, xcen, ycen, size=30):
if xcen is not None:
    x0, x1 = int(xcen - size/2.), int(xcen + size/2.)
    y0, y1 = int(ycen - size/2.), int(ycen + size/2.)
    cutout = image[y0:y1, x0:x1]
else:
    cutout = np.zeros((size, size))
return cutout

def plot_guide_circles(ax, center, radii, colors, labels=None,
               text_offset=1, **circle_kwargs):
x0, y0 = center
txt_list = []
for i, r in enumerate(radii):
c = Circle((x0, y0), r,
           edgecolor=colors[i],
           facecolor='none',
           linestyle='--',
           **circle_kwargs)
ax.add_patch(c)

txt = ax.text(x0 + r + text_offset, y0-r/4.,
        labels[i],
        color=colors[i],
        va='center',
        fontsize=8)
txt_list.append([txt])
return txt_list
                   
class plotGuiderCCDStamps:
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
    def __init__(self, reader, butler=None, view='dvcs'):
        # reader object readGuiderData
        self.reader = reader
        self.view = reader.view
        self.dayObs = reader.dayObs
        self.seqNum = reader.seqNum
        self.expId = reader.expId
        self.nStamps = reader.nStamps
        self.detnames = reader.getGuiderNames()

        # for plotting
        self.layout =  [
                        [      ".", "R40_SG1", "R44_SG0",       "."],
                        ["R40_SG0",       "center",       ".", "R44_SG1"],
                        ["R00_SG1",       ".",       ".", "R04_SG0"],
                        [      ".", "R00_SG0", "R04_SG1",       "."],
                    ]

    def plot_stamp_ccd(self, raft_ccd_key, stampNum=-1, axs=None, plo = 10.0, phi = 99.0):
        if axs is None:
            axs = plt.gca()
            plt.title(f"{self.expId}")
        
        if stampNum < 0:
            img = self.reader.read_stacked(raft_ccd_key)
        else:
            img = self.reader.read(stampNum, raft_ccd_key)
        
        bias = np.median(img)
        img_isr = img - bias
        lo,hi = np.nanpercentile(img_isr,[plo,phi])
    
        im = axs.imshow(img_isr,origin='lower',cmap='Greys',vmin=lo,vmax=hi, animated=True)
        axs.set_yticklabels([])
        axs.set_xticklabels([])
        axs.set_xticks([])
        axs.set_xticks([], minor=True)
        axs.set_yticks([])
        axs.set_yticks([], minor=True)
        return im

    def get_stamp_number_info(self, stampNum=0):
        text =  f"day_obs: {self.dayObs}"+ "\n"+f"seq_num: {self.seqNum}"+"\n"
        text += f"orientation: {self.view}"+"\n"
        if stampNum>0:
            text += f"Stamp #: {stampNum+1:02d}"
        else:
            text += f"Stacked Image w/ {self.nStamps} stamps"
        return text

    def plot_stamp_info(self, stampNum=0, axs=None, more_text=None):
        if axs is None:
            axs = plt.gca()
            
        axs.set_xticks([]); axs.set_yticks([])
        axs.set_axis_off()

        text = self.get_stamp_number_info(stampNum)
        if more_text is not None:
            text += "\n"+more_text
            
        stamp_id_text = axs.text(
            1.085, -0.10, text,
            transform=axs.transAxes,
            ha='center', va='center',
            fontsize=14, color='firebrick',
            animated=True
        )
        axs.set_axis_off()
        self.stamp_id_axs = stamp_id_text
        self.stamp_id_more_text = more_text
        return stamp_id_text

    def plot_text_ccd_name(self, detname, axs=None):
        if axs is None:
            axs = plt.gca()
        txt = axs.text(0.025, 0.025, detname,
                       transform=axs.transAxes,
                       ha='left', va='bottom',
                       fontsize=9,
                       weight='bold',
                       color='grey'
                       )
        return txt

    def plot_stamp_array(self, stampNum=0, fig=None, axs=None, plo=90., phi=99.):
        """Plot the stamp array for all the guiders.
        Args:
            stampNum (int): stamp number
            fig (matplotlib.figure.Figure): figure object
            axs (matplotlib.axes.Axes): axes object
        """
        if fig is None:
            gs = dict(hspace=0.0, wspace=0.0)
            fig, axs = plt.subplot_mosaic(self.layout, figsize=(9.5, 9.5), gridspec_kw=gs, constrained_layout=False)

        artists = []
        for detname in self.detnames:
            im = self.plot_stamp_ccd(detname, stampNum=stampNum, axs=axs[detname], plo=plo,phi=phi)
            txt = self.plot_text_ccd_name(detname, axs=axs[detname])
            artists.extend([im, txt])
        stamp_info = self.plot_stamp_info(axs=axs['center'], stampNum=stampNum)
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
        fig, axs = plt.subplot_mosaic(self.layout, figsize=(9.5, 9.5), gridspec_kw=dict(hspace=0.0, wspace=0.0), constrained_layout=False)

        nStamps = min( self.nStamps, nStampsMax )
        print("Number of stamps: ", nStamps)

        # plot the stacked image
        artists0 = self.plot_stacked_stamp_array(fig=fig, axs=axs)
        frame_list = 5*[artists0]

        #loop over the stamps
        for i in range(nStamps):
            artists = self.plot_stamp_array(stampNum=i, fig=fig, axs=axs)
            frame_list.append(artists)
        
        frame_list+= 5*[artists0]

        # update the stamps
        ani = animation.ArtistAnimation(
            fig, frame_list,
            interval=1000/fps, blit=True,
            repeat_delay=1000
        )
        ani.save(f"guider_ccd_array_{self.expId}.gif", fps=fps, dpi=dpi, writer="pillow")
        plt.close(fig)
        return ani

    def make_mp4(self, nStampsMax=10, fps=5, dpi=80):
        from matplotlib import animation
        # Create the animation
        fig, axs = plt.subplot_mosaic(self.layout, figsize=(9.5, 9.5), gridspec_kw=dict(hspace=0.0, wspace=0.0), constrained_layout=False)

        nStamps = min( self.nStamps, nStampsMax )
        print("Number of stamps: ", nStamps)

        # plot the stacked image
        artists0 = self.plot_stacked_stamp_array(fig=fig, axs=axs)
        frame_list = 5*[artists0]

        #loop over the stamps
        for i in range(nStamps):
            artists = self.plot_stamp_array(stampNum=i, fig=fig, axs=axs)
            frame_list.append(artists)
        frame_list+= 5*[artists0]

        # update the stamps
        ani = animation.ArtistAnimation(
            fig, frame_list,
            interval=1000/fps, blit=True,
            repeat_delay=1000
        )
        ani.save(f"guider_ccd_array_{self.expId}.mp4", fps=fps, dpi=dpi)
        plt.close(fig)
        return ani
    
    # TODO: add the Alt/Az orientation
