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
from typing import TYPE_CHECKING

__all__ = [
    "GuiderReader",
    "convertRawStampsToView",
    "GuiderData",
]

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from astropy.time import Time

from lsst.afw import cameraGeom
from lsst.afw.image import ExposureF, ImageF, MaskedImageF, VisitInfo
from lsst.daf.butler import Butler, DatasetNotFoundError
from lsst.meas.algorithms.stamps import Stamp, Stamps
from lsst.obs.lsst import LsstCam

from .exceptions import StampsNotFoundError
from .plotting import GuiderDataPlotter
from .transformation import (
    getCamRotAngle,
    makeCcdToDvcsTransform,
    makeInitGuiderWcs,
    makeRoiBbox,
    roiImageToDvcs,
)

if TYPE_CHECKING:
    from lsst.afw.cameraGeom import Camera, Detector
    from lsst.daf.base import PropertyList
    from lsst.geom import SkyWcs


@dataclass
class GuiderData:
    """
    LSST guider data container.

    Holds raw guider Stamps, WCS objects, and exposure metadata. Provides
    cached properties for timestamps, header information, and convenience
    (dict-like) access.

    Parameters
    ----------
    guiderNameMap : `dict[str, int]`
        Detector name to detector number map.
    rawStampsMap : `dict[str, Stamps]`
        Detector name to raw Stamps map.
    wcsMap : `dict[str, SkyWcs]`
        Detector name to SkyWcs map.
    camRotAngle : `float`
        Camera rotation angle in degrees.
    subtractMedian : `bool`, optional
        If True, subtract median row (bias) per stamp upon array extraction.
    view : `str`, optional
        Output view: 'dvcs', 'ccd', or 'roi'.

    Cached properties
    -----------------
    guiderNames        : sorted detector names.
    detNameMax         : detector with most stamps.
    metadata           : metadata dict for detNameMax.
    header             : exposure-level header summary.
    expid              : DAYOBS * 100000 + SEQNUM.
    timestamps         : masked astropy Time array (TAI scale).
    stampsMap          : Stamps for all detectors in requested view.
    axisRowMap         : detector → row axis map (0=Y, 1=X).
    roiAmpNames        : detector → amplifier name map.
    guiderFrequency    : stamp cadence in Hz.

    Public methods
    --------------
    printHeaderInfo()       : print header summary.
    getRowAxis(det)         : return row axis for detector.
    getWcs(det)             : return SkyWcs for detector.
    getStampArrayCoadd(det) : median stack of stamps.
    getGuiderAmpName(det)   : ROI amplifier name.
    getGuiderDetNum(det)    : detector number.
    plotMosaic(...)         : plot all detectors as a mosaic.
    plotStamp(...)          : plot one stamp from a single detector.
    makeGif(...)            : create a GIF animation of guider frames.

    Iteration and indexing
    ----------------------
    for det in guiderData        # iterates detector names
    guiderData.items()           # yields (det, Stamps)
    guiderData['R44_SG0']        # full Stamps object
    guiderData['R44_SG0', 3]     # stamp ndarray
    guiderData[3]                # stamp from detNameMax
    """

    guiderNameMap: dict[str, int]
    rawStampsMap: dict[str, Stamps]
    wcsMap: dict[str, "SkyWcs"]
    camRotAngle: float
    subtractMedian: bool = True
    view: str = "dvcs"  # view type, either 'dvcs' or 'ccd' or 'roi'

    def __repr__(self) -> str:
        """
        Return a compact one-line representation.

        Returns
        -------
        reprStr : `str`
            String summary with exposure id, number of stamps, view, and guider
            names.
        """
        return (
            f"GuiderData(expid={self.expid}, "
            f"nStamps={len(self)}, view='{self.view}', "
            f"guiders={self.guiderNames})"
        )

    def __len__(self) -> int:
        """
        Return number of stamps for the detector with the maximum count.

        Returns
        -------
        nStamps : `int`
            Number of stamps (time samples) in the exposure for the primary
            detector.
        """
        return len(self.rawStampsMap[self.detNameMax])

    @cached_property
    def guiderNames(self) -> list[str]:
        """
        Names of the guider detectors.

        Returns
        -------
        guiderNames : `list[str]`
            Sorted list of guider detector names.
        """
        return sorted(list(self.guiderNameMap.keys()))

    @cached_property
    def guiderIds(self) -> list[int]:
        """
        IDs of the guider detectors.

        Returns
        -------
        guiderIds : `list[int]`
            Sorted list of guider detector numeric IDs.
        """
        return sorted(list(self.guiderNameMap.values()))

    @cached_property
    def detNameMax(self) -> str:
        """
        Detector name with the maximum number of stamps.

        Returns
        -------
        detName : `str`
            Detector name possessing the most stamps.
        """
        return max(self.rawStampsMap, key=lambda k: len(self.rawStampsMap[k]))

    @cached_property
    def metadata(self) -> dict:
        """
        Metadata for the detector with the most stamps.

        Returns
        -------
        metadata : `dict`
            Copy of the detector-level metadata dictionary.
        """
        return self.rawStampsMap[self.detNameMax].metadata.toDict()

    @cached_property
    def header(self) -> dict[str, str | float | None]:
        """
        Dictionary of header metadata for this GuiderData.
        Fields: n_stamps, freq, expid, filter, cam_rot_angle, start_time,
        roi_cols, roi_rows, shuttime, az_start, el_start, az_end, el_end,
        seeing.

        Returns
        -------
        header : `dict[str, str | float | None]`
            The headers.
        """
        md = self.metadata
        info: dict[str, str | float | None] = {}
        info["n_stamps"] = len(self)
        info["freq"] = self.guiderFrequency
        # Ensure DAYOBS and SEQNUM are ints before math, keep ≤79 chars
        dayobs = int(md.get("DAYOBS", 0))
        seqnum = int(md.get("SEQNUM", 0))
        info["expid"] = dayobs * 100000 + seqnum
        info["filter"] = md.get("FILTBAND", "Unknown")
        info["cam_rot_angle"] = self.camRotAngle
        info["start_time"] = md.get("GDSSTART", None)
        info["roi_cols"] = int(md.get("ROICOLS", 0))
        info["roi_rows"] = int(md.get("ROIROWS", 0))
        info["shuttime"] = float(md["SHUTTIME"]) if "SHUTTIME" in md else np.nan
        info["guider_duration"] = self.guiderDurationSec
        info["az_start"] = float(md.get("AZSTART", np.nan))
        info["el_start"] = float(md.get("ELSTART", np.nan))
        info["az_end"] = md.get("AZEND", np.nan)
        info["el_end"] = md.get("ELEND", np.nan)
        info["seeing"] = md.get("SEEING", np.nan)
        return info

    @cached_property
    def expid(self) -> int:
        """
        Exposure identifier.

        Returns
        -------
        expid : `int`
            The exposure id.
        """
        if self.header["expid"] is None:
            raise ValueError("Missing expid in header.")
        return int(self.header["expid"])

    def printHeaderInfo(self) -> None:
        """Print a concise summary of key header fields."""
        print(f"Data Id: {self.expid}, filter-band: {self.header['filter']}")
        print(f"ROI Shape (row, col): {self.header['roi_rows']}, " f"{self.header['roi_cols']}")
        print(f"With nStamps {len(self)}" f" at {self.guiderFrequency} Hz")
        print(
            f"Acq. Start Time: {self.header['start_time']} \n"
            f"with readout duration: {self.header['guider_duration']:.2f} sec"
        )

    @cached_property
    def timestampMap(self) -> dict[str, Time]:
        """
        Aligned timestamp arrays for all detectors.

        Returns
        -------
        timestampMap : `dict[str, Time]`
            Mapping from detector name to masked Time array (TAI scale).
        """
        nStamps = len(self.rawStampsMap[self.detNameMax])
        timestampMap: dict[str, Time] = {}
        for detName, stamps in self.rawStampsMap.items():
            timestampMap[detName] = standardizeGuiderTimestamps(stamps, nStamps)
        return timestampMap

    @cached_property
    def guiderFrequency(self) -> float:
        """
        Guider stamp cadence.

        Returns
        -------
        frequency : `float`
            Median stamp frequency in Hz.
        """
        timestamps = self.timestampMap[self.detNameMax]
        # extract only the unmasked MJD values
        jd = timestamps[~timestamps.mask].jd
        period_sec = np.median(np.diff(jd)) * 86400.0
        return float(1.0 / period_sec)

    @cached_property
    def guiderDurationSec(self) -> float:
        """
        Total guider duration.

        Returns
        -------
        durationSec : `float`
            Total elapsed guider duration in seconds (including final period).
        """
        timestamps = self.timestampMap[self.detNameMax]
        jd = timestamps[~timestamps.mask].jd
        duration_sec = (jd[-1] - jd[0]) * 86400.0 + 1 / self.guiderFrequency
        return float(duration_sec)

    @cached_property
    def stampsMap(self) -> dict[str, Stamps]:
        """
        Stamps converted to requested view.

        Returns
        -------
        stampsMap : `dict[str, Stamps]`
            Mapping of detector name to converted Stamps object.
        """
        result: dict[str, Stamps] = {}
        if self.view == "roi":
            return self.rawStampsMap
        else:
            for detName, rawStamps in self.rawStampsMap.items():
                result[detName] = convertRawStampsToView(
                    rawStamps,
                    detName,
                    len(self),
                    view=self.view,
                )
            return result

    @cached_property
    def axisRowMap(self) -> dict[str, int]:
        """
        Axis mapping (row axis) for guider detectors.

        Returns
        -------
        axisMap : `dict[str, int]`
            Dictionary with detector names as keys and axis mapping as values.
            The axis mapping is 1 for X and 0 for Y.
        """
        camera = LsstCam.getCamera()
        axisMap: dict[str, int] = {}
        for detName in self.guiderNames:
            detNum = self.guiderNameMap[detName]
            detector = camera[detNum]
            nq = detector.getOrientation().getNQuarter()
            axisMap[detName] = 0 if nq % 2 == 0 else 1
        return axisMap

    @cached_property
    def roiAmpNames(self) -> dict[str, str]:
        """
        Amplifier names used in the ROI.

        Returns
        -------
        ampNames : `dict[str, str]`
            Mapping from detector name to amplifier name active in ROI.
        """
        ampNames: dict[str, str] = {}
        for detName in self.rawStampsMap.keys():
            md = self.rawStampsMap[detName].metadata.toDict()
            segment = md["ROISEG"]
            ampName = "C" + segment[7:]
            ampNames[detName] = ampName
        return ampNames

    def getRowAxis(self, detName: str) -> int:
        """
        Retrieve the index of the row axis for a detector.

        Parameters
        ----------
        detName : `str`
            Guider detector name.

        Returns
        -------
        axis : `int`
            0 if rows align with Y, 1 if rows align with X.
        """
        if detName not in self.axisRowMap:
            raise ValueError(f"Detector {detName} not found in axisRowMap.")
        return self.axisRowMap[detName]

    def getStampArrayCoadd(self, detName: str) -> np.ndarray:
        """
        Get the median-stacked stamp across time for the specified detector.

        The stack is computed across the time axis after optional median-row
        bias removal (controlled by ``self.subtractMedian``).

        Parameters
        ----------
        detName : `str`
            Guider detector name.

        Returns
        -------
        coadd : `np.ndarray`
            Median of all stamp arrays (time axis collapsed).
        """
        if detName not in self.stampsMap:
            raise KeyError(f"{detName!r} not present in stampsMap")

        stamps = self.stampsMap[detName]
        if len(stamps) == 0:
            raise StampsNotFoundError(f"No stamps found for detector {detName!r}")

        # Collect arrays, with optional bias subtraction
        arrList = [self[detName, idx] for idx in range(len(stamps))]
        stack = np.nanmedian(arrList, axis=0)
        return stack

    def getGuiderAmpName(self, detName: str) -> str:
        """
        Return amplifier name for a guider detector.

        Parameters
        ----------
        detName : `str`
            Guider detector name.

        Returns
        -------
        ampName : `str`
            Amplifier name string.
        """
        if detName not in self.roiAmpNames:
            raise ValueError(f"Detector {detName} not found in roiAmpNames.")
        return self.roiAmpNames[detName]

    def getGuiderDetNum(self, detName: str) -> int:
        """
        Return detector number for a guider detector name.

        Parameters
        ----------
        detName : `str`
            Guider detector name.

        Returns
        -------
        detNum : `int`
            The numeric detector id.
        """
        if detName not in self.guiderNameMap:
            raise ValueError(f"Detector {detName} not found in guiderNameMap.")
        return self.guiderNameMap[detName]

    def getWcs(self, detName: str) -> SkyWcs:
        """
        Return the wcs for a guider detector.

        Parameters
        ----------
        detName : `str`
            Guider detector name.

        Returns
        -------
        wcs : `SkyWcs`
            SkyWcs for the detector.
        """
        try:
            return self.wcsMap[detName]
        except KeyError as err:
            raise KeyError(f"{detName!r} not found in wcsMap") from err

    @cached_property
    def obsTime(self) -> Time:
        """
        Observation start time in TAI.

        Returns
        -------
        obsTime : `Time`
            Start time parsed from metadata (TAI scale).
        """
        gdstart = self.header["start_time"]
        return Time(gdstart, format="isot", scale="tai")

    # ------------------------------------------------------------------
    # Iterable / dict-like helpers
    # ------------------------------------------------------------------
    def __iter__(self):
        """Iterate over detector names in guiderNames order."""
        return iter(self.guiderNames)

    def items(self):
        """Yield (detName, stamps) pairs like dict.items()."""
        for det in self.guiderNames:
            yield det, self.stampsMap[det]

    def keys(self):
        """Iterate over detector names (dict-like .keys())."""
        return iter(self.guiderNames)

    def values(self):
        """Iterate over Stamps objects in guiderNames order."""
        for det in self.guiderNames:
            yield self.stampsMap[det]

    def __getitem__(self, key):
        """
        Direct stamp access helper.

        Example usage:
        gd["R44_SG0"]            -> full `Stamps` object
        gd["R44_SG0", 3]         -> stamp #3 ndarray
        gd[3]                    -> stamp #3 from `detNameMax`

        Parameters
        ----------
        key : `str | int | slice | tuple`
            Access pattern:
            - 'DET' -> Stamps
            - ('DET', i) -> single stamp ndarray
            - i (int) -> stamp i from primary detector
            - ('DET', slice) -> list of ndarrays

        Returns
        -------
        result : `Stamps | np.ndarray | list[np.ndarray]`
            Retrieved object according to key specification.
        """
        # Single detector name -> full Stamps
        if isinstance(key, str):
            return self.stampsMap[key]

        # idx only -> assume detNameMax
        if isinstance(key, (int, slice)):
            key = (self.detNameMax, key)

        if isinstance(key, tuple):
            if len(key) != 2:
                raise TypeError("Key must be (detName, idx) or (idx,)")
            detName, idx = key[0], key[1]

            if detName not in self.stampsMap:
                raise KeyError(f"{detName!r} not found in stampsMap")

            stamps = self.stampsMap[detName]
            # slice returns list of ndarrays
            if isinstance(idx, slice):
                arrays = [self._processStampArray(stamp, detName) for stamp in stamps[idx]]
                return arrays
            # int -> single ndarray
            return self._processStampArray(stamps[idx], detName)

        raise TypeError("Invalid key type for GuiderData indexing.")

    # helper -------------------------------------------------------------
    def _processStampArray(self, stamp: Stamp, detName: str) -> np.ndarray:
        """
        Convert a Stamp to an ndarray with optional median-row subtraction.

        Parameters
        ----------
        stamp : `Stamp`
            Input stamp object.
        detName : `str`
            Detector name (for row-axis logic).

        Returns
        -------
        array : `np.ndarray`
            2D image array (bias-corrected if configured).
        """
        arr = stamp.stamp_im.image.array
        if self.subtractMedian:
            axis = self.getRowAxis(detName)
            arr = arr - np.nanmedian(arr, axis=axis)
        return arr

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------
    @cached_property
    def plotter(self):
        return GuiderDataPlotter(self)

    def plotMosaic(self, stampNum: int = -1, plo: float = 50, phi: float = 99):
        """
        Plot mosaic of guider stamps.

        Parameters
        ----------
        stampNum : `int`, optional
            Stamp index; -1 for median stack.
        plo : `float`, optional
            Lower percentile stretch.
        phi : `float`, optional
            Upper percentile stretch.
        """
        self.plotter.plotStampArray(stampNum=stampNum, plo=plo, phi=phi)

    def plotStamp(self, detName: str, stampNum: int, plo: float = 50, phi: float = 99, figsize=(10, 8)):
        """
        Plot a single guider stamp.

        Parameters
        ----------
        detName : `str`
            Detector name.
        stampNum : `int`
            Stamp index.
        plo : `float`, optional
            Lower percentile stretch.
        phi : `float`, optional
            Upper percentile stretch.
        figsize : `tuple`, optional
            Figure size.
        """
        from matplotlib import pyplot as plt

        fig, axs = plt.subplots(1, 1, figsize=figsize)
        _ = self.plotter.plotStampCcd(axs, detName, stampNum=stampNum, plo=plo, phi=phi, is_ticks=True)
        axs.set_xlabel("X (pixels)", fontsize=11)
        axs.set_ylabel("Y (pixels)", fontsize=11)
        plt.title(f"{self.expid}")

    def makeGif(self, saveAs: str, fps: int = 5, plo: float = 50.0, phi: float = 99.0, figsize=(9, 9)):
        """
        Create a GIF animation of guider stamps over time.

        Parameters
        ----------
        saveAs : `str`
            Output filename.
        fps : `int`, optional
            Frames per second.
        plo : `float`, optional
            Lower percentile for display stretch.
        phi : `float`, optional
            Upper percentile for display stretch.
        figsize : `tuple`, optional
            Figure size.
        """
        self.plotter.makeGif(saveAs=saveAs, fps=fps, plo=plo, phi=phi, figsize=figsize)


class GuiderReader:
    """
    Utility to fetch LSST guider data via Butler.

    Example:
        from lsst.summit.utils.guiders.reading import GuiderReader
        from lsst.daf.butler import Butler
        butler = Butler("embargo", collections="LSSTCam/raw/guider")

        seqNum, dayObs = 461, 20250425
        reader = GuiderReader(butler, view="dvcs")
        guiderData = reader.get(dayObs=dayObs, seqNum=seqNum)
    """

    def __init__(self, butler: Butler, view: str = "dvcs"):
        self.butler = butler
        self.view = view
        self.log = logging.getLogger(__name__)
        # Define camera objects
        self.camera = LsstCam.getCamera()

        # Build guiderNameMap
        self.guiderNameMap: dict[str, int] = {}
        for detector in self.camera:
            if detector.getType() == cameraGeom.DetectorType.GUIDER:
                detName = detector.getName()
                self.guiderNameMap[detName] = detector.getId()

        self.guiderDetNames = list(self.guiderNameMap.keys())
        self.nGuiders = len(self.guiderNameMap)

    def get(
        self, dayObs: int, seqNum: int, subtractMedian: bool = True, scienceDetNum: int = 94
    ) -> GuiderData:
        """
        Retrieve guider data for a given dayObs / seqNum.

        Parameters
        ----------
        dayObs : `int`
            Day of observation in YYYYMMDD format.
        seqNum : `int`
            Sequence number.
        subtractMedian : `bool`, optional
            If True, subtract median row bias from each stamp upon access.
        scienceDetNum : `int`, optional
            Science detector number for WCS reference.

        Returns
        -------
        guiderData : `GuiderData`
            Assembled guider data object.
        """
        # check if the guider name is swaped (dayObs < 20250509)
        # modifies self.guiderNameMap in place if necessary
        self.applyGuiderNameSwapIfNeeded(dayObs)

        # 1. Get guider_raw stamps all guiders
        rawStampsDict = self.getGuiderRawStamps(dayObs, seqNum)

        # set the number of stamps
        # determine the maximum number of stamps among all guiders
        nStampsList = [len(stamps) for stamps in rawStampsDict.values()]
        self.nStamps = max(nStampsList)

        if self.nStamps <= 1:
            raise StampsNotFoundError(
                f"Only {self.nStamps} stamps found for dayObs {dayObs}, seqNum {seqNum}. "
                "At least 2 stamps are required to create GuiderData."
            )

        # Create a visitinfo dataId to get the WCS and camera rotation angle
        visitInfo = getVisitInfo(self.butler, dayObs, seqNum, scienceDetNum)

        # 2. Get WCS map for all guiders
        wcsMapDict = makeInitGuiderWcs(self.camera, visitInfo)

        # 3. Get camera rotation angle
        camRotAngle = getCamRotAngle(visitInfo)

        # 4. Pack everything into a GuiderData object
        guiderData = GuiderData(
            view=self.view,
            rawStampsMap=rawStampsDict,
            guiderNameMap=self.guiderNameMap,
            wcsMap=wcsMapDict,
            camRotAngle=camRotAngle,
            subtractMedian=subtractMedian,
        )
        return guiderData

    def getGuiderRawStamps(self, dayObs: int, seqNum: int) -> dict[str, Stamps]:
        """
        Fetch raw guider Stamps for all detectors.

        Parameters
        ----------
        dayObs : `int`
            Observation day (YYYYMMDD).
        seqNum : `int`
            Sequence number.

        Returns
        -------
        rawStamps : `dict[str, Stamps]`
            Mapping from detector name to raw Stamps object.
        """
        rawStamps: dict[str, Stamps] = {}
        for detName, detNum in self.guiderNameMap.items():
            try:
                rawStamps[detName] = self.butler.get(
                    "guider_raw",
                    day_obs=dayObs,
                    seq_num=seqNum,
                    detector=detNum,
                    instrument="LSSTCam",
                )
            except DatasetNotFoundError as e:
                raise StampsNotFoundError(f"No data for {detName} on {dayObs=} {seqNum=}") from e
        return rawStamps

    def applyGuiderNameSwapIfNeeded(self, dayObs: int) -> None:
        """
        Apply guider name swap (SG0/SG1) for early data if required.

        Parameters
        ----------
        dayObs : `int`
            Observation day (YYYYMMDD) used to decide whether to swap.
        """
        if getattr(self, "_guiderNameMapSwapped", False):
            return  # Already swapped; do nothing
        if dayObs < 20250509:
            newMap = {}
            for detName, detNum in self.guiderNameMap.items():
                if detName.endswith("SG0"):
                    swapped = detName.replace("SG0", "SG1")
                elif detName.endswith("SG1"):
                    swapped = detName.replace("SG1", "SG0")
                else:
                    swapped = detName
                swappedDetNum = self.camera[swapped].getId()
                newMap[swapped] = swappedDetNum
            self.guiderNameMap = newMap
            self._guiderNameMapSwapped = True


def getVisitInfo(butler: Butler, dayObs: int, seqNum: int, scienceDetNum: int = 94) -> VisitInfo:
    """
    Retrieve VisitInfo for a given dayObs / seqNum.

    Parameters
    ----------
    butler : `Butler`
        Active Butler instance.
    dayObs : `int`
        Observation day (YYYYMMDD).
    seqNum : `int`
        Sequence number.
    scienceDetNum : `int`, optional
        Science detector number.

    Returns
    -------
    visitInfo : `VisitInfo`
        VisitInfo object retrieved from butler.
    """
    dataId = {
        "instrument": "LSSTCam",
        "day_obs": dayObs,
        "seq_num": seqNum,
        "detector": scienceDetNum,
    }
    visitInfo = butler.get("raw.visitInfo", dataId)
    return visitInfo


# Helper functions for stamp conversion
def _makeRoiTransforms(metadata: dict, detector: Detector, camera: Camera, view: str) -> tuple[tuple, str]:
    """
    Construct ROI transforms and derive amplifier name.

    Parameters
    ----------
    metadata : `dict`
        Exposure-level metadata.
    detector : `Detector`
        The detector.
    camera : `Camera`
        The camera object.
    view : `str`
        Requested output view ('dvcs', 'ccd', or 'roi').

    Returns
    -------
    result : `tuple[tuple, str]`
        ((ccdViewBbox, fwd, back), ampName)
    """
    ampName = "C" + metadata["ROISEG"][7:]
    ccdViewBbox = makeRoiBbox(metadata, camera)
    fwd, back = makeCcdToDvcsTransform(ccdViewBbox, detector.getOrientation().getNQuarter())
    return (ccdViewBbox, fwd, back), ampName


def _convertMaskedImage(
    maskedImage: MaskedImageF,
    stampMetadata: dict,
    metadata: dict,
    transforms: tuple,
    detector: Detector,
    ampName: str,
    camera: Camera,
    view: str,
) -> Stamp:
    """
    Convert one masked image ROI and build a Stamp.

    Parameters
    ----------
    maskedImage : `MaskedImageF`
        Input masked image for the ROI.
    stampMetadata : `dict`
        Stamp-level metadata (modified in-place).
    metadata : `dict`
        Exposure-level metadata.
    transforms : `tuple`
        Tuple containing (ccdViewBbox, forwardTransform, inverseTransform).
    detector : `Detector`
        Camera detector.
    ampName : `str`
        Amplifier name.
    camera : `Camera`
        Camera object.
    view : `str`
        Output view ('dvcs', 'ccd', or 'roi').

    Returns
    -------
    stamp : `Stamp`
        Converted stamp object.
    """
    ccdViewBbox, fwd, back = transforms
    rawArray = maskedImage.getImage().getArray()
    dvcsArray = roiImageToDvcs(rawArray, metadata, detector, ampName, camera, view=view)
    outImg = MaskedImageF(dvcsArray)
    archiveElement = [ccdViewBbox, fwd, back]
    return Stamp(outImg, archiveElement, metadata=stampMetadata)


def _blankStamp(
    stampIdx: int,
    metadata: PropertyList,
    transforms: tuple,
) -> Stamp:
    """
    Create a blank (zero) stamp for a missing index.

    Parameters
    ----------
    stampIdx : `int`
        Missing stamp index.
    metadata : `PropertyList`
        Original metadata container.
    transforms : `tuple`
        (ccdViewBbox, forwardTransform, inverseTransform).

    Returns
    -------
    stamp : `Stamp`
        Placeholder blank stamp with NaN timestamp.
    """
    ccdViewBbox, fwd, back = transforms
    nRows, nCols = int(metadata["ROIROWS"]), int(metadata["ROICOLS"])
    blankArray = np.zeros((nRows, nCols), dtype=np.float32)
    blankImg = ExposureF(MaskedImageF(ImageF(array=blankArray)))
    missingMetadata = metadata.toDict().copy()
    missingMetadata["DAQSTAMP"] = stampIdx
    missingMetadata["STMPTMJD"] = np.nan
    archiveElement = [ccdViewBbox, fwd, back]
    return Stamp(blankImg, archiveElement, metadata=missingMetadata)


def convertRawStampsToView(
    rawStamps: Stamps,
    detName: str,
    nStamps: int,
    view: str = "dvcs",
) -> Stamps:
    """
    Convert guider stamps from raw ROI to a requested view ('dvcs' or 'ccd').
    Handles missing stamps and preserves metadata and order.

    Parameters
    ----------
    rawStamps : `Stamps`
        Input raw ROI stamps.
    detName : `str`
        Detector name.
    nStamps : `int`
        Target total number of stamps (after filling gaps).
    view : `str`, optional
        Output view: 'dvcs', 'ccd', or 'roi'.

    Returns
    -------
    stampsOut : `Stamps`
        Converted stamps with gaps filled by blank stamps.
    """
    camera = LsstCam.getCamera()
    detector = camera[detName]
    metadata = rawStamps.metadata

    # Ensure CCDSLOT matches swapped name if applicable
    metadata["CCDSLOT"] = detName[4:7]

    # Align timestamps and find valid/missing indices
    timestamps = standardizeGuiderTimestamps(rawStamps, nStamps)
    validIndices = np.where(~timestamps.mask)[0].tolist()
    missingIndices = np.where(timestamps.mask)[0].tolist()

    # Pre‑compute transforms once
    transforms, ampName = _makeRoiTransforms(metadata, detector, camera, view)

    stampsDict: dict[int, Stamp] = {}
    mIdx = 0  # index into masked images list

    for idx in validIndices:
        maskedImage = rawStamps.getMaskedImages()[mIdx]
        stampMeta = rawStamps[mIdx].metadata
        stampMeta["DAQSTAMP"] = stampMeta.get("DAQSTAMP", idx)
        stampsDict[idx] = _convertMaskedImage(
            maskedImage,
            stampMeta,
            metadata,
            transforms,
            detector,
            ampName,
            camera,
            view,
        )
        mIdx += 1

    # Fill gaps with blanks
    for idx in missingIndices:
        stampsDict[idx] = _blankStamp(idx, metadata, transforms)

    # Assemble in order
    stampList = [stampsDict[i] for i in range(nStamps)]
    return Stamps(stampList, metadata, use_archive=True)


def standardizeGuiderTimestamps(rawStamps: Stamps, nStamps: int) -> Time:
    """
    Return a masked `Time` array of length `nStamps` for one guider.

    Missing stamps are filled with `NaN` and masked so every detector shares a
    uniform, sortable timestamp vector. Result is in MJD, `scale="tai"`.

    Parameters
    ----------
    rawStamps : `Stamps`
        Input stamps (may have missing indices).
    nStamps : `int`
        Desired total length (maximum among guiders).

    Returns
    -------
    fullTimestamps : `Time`
        Masked Time array (scale='tai') with inferred cadence and gaps masked.
    """
    timestampsList = [stamp.metadata.get("STMPTMJD", np.nan) for stamp in rawStamps]
    mjdArray = np.ma.masked_invalid(timestampsList)
    timestamps = Time(mjdArray, format="mjd", scale="tai")

    # infer frequency from valid timestamps
    jdDiff = np.diff(timestamps.jd)
    freqDays = np.nanmedian(jdDiff)
    startJd = timestamps[0].jd

    if np.isnan(freqDays) or freqDays <= 0:
        raise ValueError(
            f"Invalid frequency {freqDays} derived from timestamps. "
            "Ensure that the timestamps are valid and evenly spaced."
        )

    timestampsIdeal = Time(startJd + np.arange(nStamps) * freqDays, format="jd", scale="tai")

    tolerance = 0.5 * freqDays
    actualJd = timestamps.jd

    # Build aligned timestamp list with np.nan where missing
    mjdList = []
    for idealJd in timestampsIdeal.jd:
        # check if any actual timestamp is close to this ideal timestamp
        if np.any(np.abs(actualJd - idealJd) < tolerance):
            # match found — find the closest
            idx = np.argmin(np.abs(actualJd - idealJd))
            mjdList.append(actualJd[idx])
        else:
            # no match — this is a missing timestamp
            mjdList.append(np.nan)

    mjdArray = np.ma.masked_invalid(mjdList)
    fullTimestamps = Time(mjdArray, format="mjd", scale="tai")
    fullTimestamps.freq = freqDays
    return fullTimestamps
