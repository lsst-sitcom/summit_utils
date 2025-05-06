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

__all__ = [
    "read_guider_data",
    "readGuiderData"
]

import numpy as np
from astropy.io import fits
from lsst.resources import ResourcePath

import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u

from lsst.obs.lsst import LsstCam
from lsst.daf.butler import Butler
from lsst.summit.utils.guiders.transformation import amp_to_ccdview

# TODO: Grab this information from a yaml file
guiderDetMap = {
    "R00_SG0": (189, 3),
    "R00_SG1": (190, 2),
    "R04_SG0": (193, 0),
    "R04_SG1": (194, 3),
    "R40_SG0": (197, 2),
    "R40_SG1": (198, 1),
    "R44_SG0": (201, 1),
    "R44_SG1": (202, 0),
}

guiderDetMap = {
    "R00_SG0": (189, 3),
    "R00_SG1": (190, 2),
    "R04_SG0": (193, 0),
    "R04_SG1": (194, 3),
    "R40_SG0": (197, 2),
    "R40_SG1": (198, 1),
    "R44_SG0": (201, 1),
    "R44_SG1": (202, 0),
}

class readGuiderData:
    """
    Butler class to read the guide data.

    The data is read from the LSSTCam butler, and the header information
    is extracted from the raw metadata. The data is in raw format and amplifier
    orientation. The data is then converted to the CCD coordinate system
    (DM view) using the amp_to_ccdview function.
    
    Parameters
    ----------
    seqNum : int
        The sequence number of the observation.
    dayObs : int
        The day of the observation in YYYYMMDD format.
    raft : str
        The raft name (e.g. "R00").
    ccd : str
        The CCD name (e.g. "SG0").
    butler : Butler, optional
        The butler object to use for reading the data. If None, a new
        butler object is created.

    Example
    -------
    seqNum, dayObs, raft, ccd = 591, 20250425, "R40", "SG1"

    # Instantiate the readGuiderData class
    reader = readGuiderData(seqNum, dayObs, raft, ccd)

    print("Header information used here")
    reader.print_header_info()

    print("Timestamp is based on the GDR Start Time and offset by the stamp index")
    reader.timestamp

    print("Read data for a given stamp. Array in CCD coordinates orientation.")
    reader.read(0)
    
    print("Plot Example")
    reader.plot_stamp_ccd(stampNum=20)        
    """
    def __init__(self, seqNum, dayObs, raft='R00', ccd='SG1', butler=None):
        self.butler = butler

        # Data Id information
        self.dayObs = dayObs
        self.seqNum = seqNum
        self.ccd = ccd

        # exposure
        self.expId = dayObs * 100000 + seqNum        

        # Query the detector number and segment number
        self.key = f"{raft}_{ccd}"
        self.det = guiderDetMap[self.key][0]
        self.seg = guiderDetMap[self.key][1]
        self.ampName = f"C{self.seg:02d}"

        # Define camera objects
        self.camera = LsstCam.getCamera()
        self.detector = self.camera[self.det]

        self.FREQ = 5 # Hz
        self.DELAY = 20/1000 # seconds

        # Initialize the attributes
        # Butler, header information, timestamp
        self.initialize()

    def initialize(self):
        if self.butler is None:
            # Initialize butler
            self.butler = Butler("LSSTCam", collections=["LSSTCam/raw/guider"], instrument="LSSTCam")
            
        # Initialize the attributes
        self.load_raw_exposure()
        # returns the raw (butler image object)

        self.get_header_info(self.raw)
        # returns the header information

        self.get_timestamp()
        # returns the timestamp
        pass

    def load_raw_exposure(self):
        """
        Load the raw exposure from the butler.
        This is a list of image exposure objects.        
        """
        self.raw = self.butler.get("guider_raw", exposure=self.expId, detector=self.det)
        pass
    
    def get_header_info(self, raw):
        """
        Get the header information from the raw exposure.        
        """
        m = raw.metadata.toDict()
        self.roiCol    = m['ROICOL']
        self.roiRow    = m['ROIROW']
        self.roiCols   = m['ROICOLS']
        self.roiRows   = m['ROIROWS']
        self.roiUnder  = m.get('ROIUNDER', m.get('ROIUNDRC', 6))
        self.nStamps   = m['N_STAMPS']
        self.start_time= m['GDSSTART']   # ISO‐formatted string
        self.FREQ      = 5  # 5 Hz
        pass
    
    # TODO: add a function to get the timestamp from the header
    def get_timestamp(self):
        """
        Get the timestamp from the header information.
        The timestamp is based on the GDS start time and the stamp index.

        TODO: add a function to get the timestamp from the header
        """
        # 1) stamp indices
        self.stamp = np.arange(self.nStamps, dtype=float)

        # 2) parse start time once
        t0 = Time(self.start_time, format='isot', scale='utc')

        # 3) build Time array
        dt = (1.0 / self.FREQ) * u.second
        dt+= self.DELAY * u.second
        self.timestamp = t0 + self.stamp * dt
        return self.timestamp
    
    def read_stamp(self, stamp):
        """
        Read the Guider data from Butler and return on Amplifier orientation.

        Returns:
            array: roi coordinates image array
        """
        # Unpack the data
        roiarr = self.raw[stamp].stamp_im.image.array
        return roiarr

    def read(self, stamp):
        """
        Read the Guider data from Butler and return ROI image flipped to be in CCD coordinate orientation

        Returns:
            array: ccd pixel coordinates image array
        """
        # Unpack the data
        roiarr = self.read_stamp(stamp)
        ccdarr = amp_to_ccdview(roiarr,self.detector,self.ampName)
        return ccdarr

    def print_header_info(self):
        print(f"Guider Data: {self.key}, expId: {self.expId}")
        print(f"ROI Row: {self.roiRow}, ROI Col: {self.roiCol}, ROI Rows: {self.roiRows}, ROI Cols: {self.roiCols}")
        print(f"Number of Stamps: {self.nStamps}")
        print(f"Start Time: {self.start_time}")
        pass
    
    def plot_stamp_ccd(self, stampNum=0, axs=None, plo = 10.0, phi = 99.0):
        """
        Plot the stamp in CCD coordinates.
        Parameters
        ----------
        stampNum : int
            Index of the stamp to plot.
        axs : matplotlib.axes.Axes
            The axes in which to draw.
        plo, phi : float
            Percentiles for display stretch.
        """
        if axs is None:
            axs = plt.gca()
            plt.title(f"{self.expId}")
        key = self.key
        key+= f": # {stampNum}"

        img = self.read(stampNum)
        bias = np.median(img)
        img_isr = img - bias
        lo,hi = np.nanpercentile(img_isr,[plo,phi])
    
        im = axs.imshow(img_isr,origin='lower',cmap='Greys',vmin=lo,vmax=hi,animated=True)
        axs.text(0.05, 0.9, key, fontsize=6, color="w")
        axs.set_yticklabels([])
        axs.set_xticklabels([])
        axs.set_xticks([])
        axs.set_xticks([], minor=True)
        axs.set_yticks([])
        axs.set_yticks([], minor=True)
        return im
    
    def plot_stamp(self,
                   img = None,
                   stampNum = 0,
                   ax = None,
                   im  = None,
                   txt = None,
                   plo = 10.0,
                   phi = 99.0):
        """
        Either create or update the image & text artists for a given stamp.
        
        Parameters
        ----------
        stampNum : int
            Index of the stamp to plot.
        ax : matplotlib.axes.Axes
            The axes in which to draw.
        im : AxesImage or None
            If None, a new imshow() is created; otherwise it's updated.
        txt : Text or None
            If None, a new Text is created; otherwise it's updated.
        plo, phi : float
            Percentiles for display stretch.
        
        Returns
        -------
        (im, txt) : tuple
            The image and text artists (new or updated).
        """
        # Read & normalise
        if img is None:
            img = self.read(stampNum)
        bias = np.median(img)
        img_isr = img - bias
        lo, hi = np.nanpercentile(img_isr, [plo, phi])

        # Image artist
        if im is None:
            im = ax.imshow(img_isr,
                           origin='lower',
                           cmap='Greys',
                           vmin=lo, vmax=hi,
                           animated=True)
        else:
            im.set_data(img_isr)
            im.set_clim(lo, hi)

        # Text artist
        key = f"{self.key}: #{stampNum}"
        if txt is None:
            txt = ax.text(0.15, 0.9,
                          key,
                          transform=ax.transAxes,
                          fontsize=14, color="w",
                          animated=True)
        else:
            ax.text(0.15, 0.9 , txt)
            # txt.set_position((0.15, 0.9))
            # ax.set_text(key)

        # Hide ticks once (if first call)
        if stampNum == 0 and txt is None:
            ax.set_xticks([]); ax.set_yticks([])

        return im, txt
                       
def read_guider_data(filename):
    rp = ResourcePath(filename)
    
    with rp.open(mode="rb") as f:
        
        hdu_list = fits.open(f)
        
        N_frames = hdu_list[0].header['N_STAMPS']
        x, y = hdu_list[0].header['ROIROWS'], hdu_list[0].header['ROICOLS']

        imgs = np.zeros((N_frames, x, y))
        timestamps = np.zeros(N_frames)

        if (len(hdu_list) - 1)/2 == N_frames:  ## if has rawstamps
            for i, j in zip(range(N_frames), range(2, len(hdu_list), 2)):
                imgs[i, :, :] = hdu_list[j].data
                timestamps[i] = hdu_list[j].header['STMPTMJD']

        elif len(hdu_list)-1 == N_frames:     ## if no rawstamps
            for i, j in zip(range(N_frames), range(1, len(hdu_list))):
                imgs[i, :, :] = hdu_list[j].data
                timestamps[i] = hdu_list[j].header['STMPTMJD']
    
    grand_hdr = hdu_list[0].header

    return imgs, timestamps, grand_hdr
