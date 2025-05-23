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
    "readGuiderData",
    "get_guider_stamps",
]

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
import astropy.units as u

from astropy.io import fits
from lsst.resources import ResourcePath

from lsst.obs.lsst import LsstCam
from lsst.daf.butler import Butler
from lsst.afw import cameraGeom
from lsst.summit.utils.utils import getSite
# Get the site and the camera object
site = getSite()
camera = LsstCam.getCamera()

from lsst.meas.algorithms.stamps import Stamp,Stamps
from lsst.afw.image import MaskedImageF

from lsst.summit.utils.guiders.transformation import mk_roi_bboxes, convert_roi

# Todo: put in summit utils guiders
#from utils import get_guider_stamps

from astropy.io import fits
from lsst.resources import ResourcePath

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
    
class readGuiderData:
    """Class to read and unpack the Guider data from Butler.
       Plot an animated gif of the CCD guider stamp.
    
    Works in the summit and usdf environments.
    
    Key Attributes:
        dataset (dict): Dictionary of guider data
        guiders (dict): Dictionary of guider detector information
    
    Example:
        from lsst.summit.utils.guiders.reading import readGuiderData

        seqNum, dayObs = 591, 20250425

        # Load the data from the butler
        reader = readGuiderData(seqNum, dayObs, view='dvcs', verbose=True)
        reader.init_guiders()
        reader.load_data()

        # reader makes a dictionary of the guider data
        # with the keys being the guider names 
        # and the values being the stamps object
        # the stamps orientation are defined by view, default 'dvcs'
        print("Guider names: ", reader.getGuiderNames())
        print("Guider ids: ", reader.getGuiderIds())
        print("Data orientation: ", reader.view)
        print("Guider data: ", reader.dataset)

        # Create the plotter object
        plotter = plotGuiderCCDStamps(reader)
        
        # Plot the initial stamp
        plotter.plot_stamp_array(stampNum=0, nStampsMax=10)

        # Make a gif of the stamps
        plotter.make_gif(nStampsMax=10, fps=5)
        
    """
    def __init__(self, seqNum, dayObs, verbose=False,
                 butler=None, view='dvcs', collections=['LSSTCam/raw/all', 'LSSTCam/raw/guider']):
        
        # data id
        self.butler = butler
        self.dayObs = dayObs
        self.seqNum = seqNum
        self.dataId = {'instrument': 'LSSTCam', 'day_obs': self.dayObs, 'seq_num': self.seqNum}

        # exposure
        self.expId = dayObs * 100000 + seqNum        

        # Define camera objects
        self.camera = LsstCam.getCamera()

        self.FREQ = 5 # Hz
        self.DELAY = 20/1000 # seconds

        self.view = view
        self.dataset = None
        self.verbose = verbose
        self.guiders = {}

        # Butler
        self.initialize_butler(collections)

    def load(self):
        # Initialize the attributes
        self.init_guiders()
        
        # Load the data
        self.load_data()
        self.get_header_info(self.dataset[self.detnames[0]])
        self.get_timestamp()

        # verbose
        if self.verbose:
            self.print_header_info()

    def initialize_butler(self, collections=None):
        if site=="summit":  
            repo = "LSSTCam"
        elif site=="staff-rsp":
            repo = "/repo/embargo"
        else:
            raise ValueError(f"Unknown butler repo for {site}")
        
        if self.butler is None:
            # Initialize butler
            self.butler = Butler(repo, collections=["LSSTCam/raw/guider"]+collections, instrument="LSSTCam")
        pass
    
    def init_guiders(self):
        """Load the guider detector information.
        """
        self.guiders = {}
        for detector in self.camera:
            if detector.getType()== cameraGeom.DetectorType.GUIDER :
                detName = detector.getName()
                self.guiders[detName] = detector.getId()

        self.detnames = list(self.guiders.keys())
        self.nGuiders = len(self.guiders)
        pass

    
    def load_data(self):
        """Load the data from the butler for all guider detectors.

        Args:
            butler (Butler): butler object
            dayObs (int): day of observation
            seqNum (int): sequence number
            view (str): view type ('roi', 'dvcs', 'ccd')
        """
        datas = {}
        dataId = self.dataId.copy()
        for detName, idet in self.guiders.items():
            det = self.camera[idet]
            if self.view == 'roi':
                dataId['detector'] = idet
                datas[detName] = self.butler.get('guider_raw', dataId)
            
            elif self.view == 'dvcs':
                datas[detName] = get_guider_stamps(idet,self.seqNum,self.dayObs,butler=self.butler,view='dvcs')
            
            elif self.view == 'ccd':
                datas[detName] = get_guider_stamps(idet,self.seqNum,self.dayObs,butler=self.butler,view='ccd')

        self.dataset = datas
        self.init_ampnames()
        #self.nStamps = max([len(data) for data in datas.values()])
        pass

    def init_ampnames(self):
        self.ampNames = {}
        for detname in self.guiders.keys():
            md = self.dataset[detname].metadata.toDict()
            # also get the ampName
            segment = md['ROISEG']
            ampName = 'C'+ segment[7:]        
            self.ampNames[detname] = ampName
        pass

    def get_timestamp(self):
        # 1) stamp indices
        self.stamp = np.arange(self.nStamps, dtype=float)

        # 2) parse start time once
        t0 = Time(self.start_time, format='isot', scale='utc')

        # 3) build Time array
        dt = (1.0 / self.FREQ) * u.second
        dt+= self.DELAY * u.second
        self.timestamp = t0 + self.stamp * dt
        return self.timestamp
    
    def get_header_info(self, raw):
        m = raw.metadata.toDict()
        self.roiCol    = m['ROICOL']
        self.roiRow    = m['ROIROW']
        self.roiCols   = m['ROICOLS']
        self.roiRows   = m['ROIROWS']
        self.roiUnder  = m.get('ROIUNDER', m.get('ROIUNDRC', 6))
        self.nStamps   = m['N_STAMPS']
        self.start_time= m['GDSSTART']
        self.FREQ      = 5  # 5 Hz
        pass

    def print_header_info(self):
        print(f"Data Id: {self.dataId}")
        print(f"ROI Row: {self.roiRow}, ROI Col: {self.roiCol}, ROI Rows: {self.roiRows}, ROI Cols: {self.roiCols}")
        print(f"Number of Stamps: {self.nStamps}")
        print(f"Acq. Start Time: {self.start_time}")
        pass

    def read(self, stamp, detname):
        """
        Read the Guider data from Butler and return image array.
        Orientation is defined by the view setting.
        Args:
            stamp (int): stamp number
            detname (str): detector name
        Returns:
            array: image array
        """
        # Unpack the data
        stamps = self.dataset[detname]

        if stamp >= len(stamps):
            return np.zeros((self.roiRows, self.roiCols), dtype=np.float32)

        roiarr = stamps[stamp].stamp_im.image.array
        return roiarr
    
    def read_stacked(self, detname):
        """
        Read the Guider data from Butler and return image array.
        Orientation is defined by the view setting.
        Args:
            detname (str): detector name
        Returns:
            array: image array
        """
        # Unpack the data
        stamps = self.dataset[detname]
        roiarr = []
        for stamp in stamps:
            roiarr.append(stamp.stamp_im.image.array)
        return np.median(roiarr, axis=0)
    
    def getGuiderNames(self):
        """Get the names of the guider detectors.
        """
        return list(self.guiders.keys())
    
    def getGuiderIds(self):
        """Get the ids of the guider detectors.
        """
        return list(self.guiders.values())
    
    def getGuiderAmpNames(self):
        """
        Get list of guider amp names
        """
        if hasattr(self, 'ampNames'):
            return list(self.ampNames.values())

        ampNames = {}
        for detname in self.guiders.keys():
            md = self.dataset[detname].metadata.toDict()
            # also get the ampName
            segment = md['ROISEG']
            ampName = 'C'+ segment[7:]        
            ampNames[detname] = ampName
        self.ampNames = ampNames
        return list(ampNames.values())

    def set_detector(self, detname):
        """
        Set the guider detector to be referenced.
        """
        if detname not in self.guiders.keys():
            raise ValueError(f"Guider {detname} not found.")
        
        self.detname = detname
        self.idet = self.guiders[detname]
        self.detector = self.camera[self.idet]
        self.ampName = self.ampNames[detname]
        pass

def get_guider_stamps(idet,seqNum,dayObs,repo='/repo/embargo',collections=['LSSTCam/raw/guider'],butler=None,view='dvcs'):
    """
    This class reads the stamp object from the Butler for one Guiders and 
    converts them to DVCS view, making a new Stamps object
    
    Parameters
    ----------
    idet : int
        Detector Id
        
    seqNum : int 
        Sequence Number

    dayObs : int
        Day Observation

    repo : str
        Butler repo

    collections : list of str
        Butler collections

    Returns
    -------
    stamps : lsst.meas.algorithms.stamps.Stamps
        Stamp images oriented in DVCS
    
    """
    # get Camera object
    camera = LsstCam.getCamera()
    detector = camera[idet]
    
    # Get a Butler if none is provided
    if butler==None:
        butler = Butler(repo, collections=collections)


    # for dayObs of 20250509 or before, the ROIs are swapped between SG0 and SG1.  Fix here
    if dayObs < 20250509:
        detName = camera[idet].getName()
        raft = detName[0:3]
        ccd = detName[4:7]
        if ccd=='SG0':
            ccd_swapped = 'SG1'
        elif ccd=='SG1':
            ccd_swapped = 'SG0'

        detName_swapped = raft + '_' + ccd_swapped
        detector_swapped = camera[detName_swapped]
        idet_swapped = detector_swapped.getId()

        dataId = {'instrument': 'LSSTCam', 'detector': idet_swapped, 'day_obs': dayObs,'seq_num':seqNum}

    else:
        dataId = {'instrument': 'LSSTCam', 'detector': idet, 'day_obs': dayObs,'seq_num':seqNum}


    # finally read from the Butler
    raw_stamps = butler.get('guider_raw', dataId)
    md = raw_stamps.metadata

    # fix CCD in the metadata
    if dayObs < 20250509:
        md['CCDSLOT'] = ccd_swapped        

    # also get the ampName
    segment = md['ROISEG']
    ampName = 'C'+ segment[7:]

    # build the CCD view and DVCS view Bounding Boxes
    ccd_view_bbox,dvcs_view_bbox = mk_roi_bboxes(md,camera)
    
    # now loop over the individual ROIs 
    stamp_list= []
    for i,masked_ims in enumerate(raw_stamps.getMaskedImages()):        

        # convert to DVCS view
        raw_roi = masked_ims.getImage().getArray()
        roi_dvcs = convert_roi(raw_roi,detector,ampName,camera,view=view)
        
        # build a Stamp Object
        output_masked_im = MaskedImageF(roi_dvcs)
        archive_element = [ccd_view_bbox,dvcs_view_bbox]
        stamp_list.append(Stamp(output_masked_im,archive_element))
    
    output_stamps = Stamps(stamp_list,md,use_archive=True)

    return output_stamps
