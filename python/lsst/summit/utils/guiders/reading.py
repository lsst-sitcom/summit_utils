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
from typing import TYPE_CHECKING, Optional

__all__ = [
    "GuiderReader",
    "getGuiderStamps",
    "GuiderData",
]


from dataclasses import dataclass

import numpy as np
from astropy.time import Time

import lsst.summit.utils.butlerUtils as butlerUtils
from lsst.afw import cameraGeom
from lsst.afw.image import ExposureF, ImageF, MaskedImageF
from lsst.daf.butler import Butler, DatasetNotFoundError
from lsst.meas.algorithms.stamps import Stamp, Stamps
from lsst.obs.lsst import LsstCam  # pylint: disable=unused-import
from lsst.summit.utils.guiders.exceptions import StampsNotFoundError
from lsst.summit.utils.guiders.guiderwcs import get_camera_rot_angle, make_init_guider_wcs
from lsst.summit.utils.guiders.transformation import convert_roi, mk_ccd_to_dvcs, mk_roi_bboxes

if TYPE_CHECKING:
    from lsst.geom import SkyWcs


@dataclass(slots=True)
class GuiderData:
    """Data class to hold guider data information."""

    dayObs: int
    seqNum: int
    timestamps: Time
    roiAmpNames: dict[str, str]
    guiderNameMap: dict[str, int]
    datasets: dict[str, Stamps]  # TODO: Consider renaming this & making private
    header: dict[str, str | float]
    wcs: dict[str, SkyWcs]
    axisRowMap: dict[str, int]  # 0 for Y, 1 for X
    view: str = "dvcs"  # view type, either 'dvcs' or 'ccd' or 'roi'
    # TODO: Add these properties back in if needed
    # filter_band: str

    @property
    def guiderNames(self) -> list[str]:
        """Get the names of the guider detectors."""
        return list(self.guiderNameMap.keys())

    def getRowAxis(self, detName: str) -> int:
        """Get the axis corresponding to the rows for a given guider detector.

        Parameters
        ----------
        detName : `str`
            The name of the detector.

        Returns
        -------
        axis : `int`
            The axis corresponding to the rows for a given detector.
        """
        if detName not in self.axisRowMap:
            raise ValueError(f"Detector {detName} not found in axisRowMap.")
        return self.axisRowMap[detName]

    def getStampArray(self, stampNum: int, detName: str, isIsr: bool = False) -> np.ndarray:
        """Get the stamp for a given stamp number and detector name.

        Parameters
        ----------
        stampNum : `int`
            The index of the stamp to retrieve.
        detName : `str`
            The name of the detector for which to retrieve the stamp.
        isIsr : `bool`, optional
            If True, subtract the median bias over the rows.

        Returns
        -------
        data : `np.ndarray`
            The stamp image array for the specified detector and stamp number.
        """
        if detName not in self.datasets:
            raise ValueError(f"Detector {detName} not found in datasets.")

        stamps = self.datasets[detName]
        array = stamps[stampNum].stamp_im.image.array
        if isIsr:
            whichaxis = self.getRowAxis(detName)
            array = array - np.median(array, axis=whichaxis)
        return array

    def getStackedStampArray(self, detName: str, isIsr: bool = False) -> np.ndarray:
        """Get the stacked stamp for a given detector name.

        Parameters
        ----------
        detName : `str`
            The name of the detector for which to retrieve the stacked stamp.
        isIsr : `bool`, optional
            If True, subtract the median bias over the rows.

        Returns
        -------
        stack : `np.ndarray`
            The median stack of all stamps for the specified detector.
        """
        if detName not in self.datasets:
            raise ValueError(f"Detector {detName} not found in datasets.")
        stamps = self.datasets[detName]
        roiarr = []
        for stamp in stamps[1:]:  # skip the first stamp (shutter opening)
            img = stamp.stamp_im.image.array
            # simple bias subtraction over the columns
            if isIsr:
                whichaxis = self.getRowAxis(detName)
                img = img - np.median(img, axis=whichaxis)
            roiarr.append(img)

        stack = np.nanmedian(roiarr, axis=0)
        return stack

    def getGuiderAmpName(self, detName: str) -> str:
        """Get the amplifier name for a given guider detector.

        Parameters
        ----------
        detName : `str`
            The name of the detector.

        Returns
        -------
        ampName : `str`
            The name of the amplifier used for the ROI
            for the specified guider detector.
        """
        if detName not in self.roiAmpNames:
            raise ValueError(f"Detector {detName} not found in roiAmpNames.")
        return self.roiAmpNames[detName]

    def getGuiderDetNum(self, detName: str) -> int:
        """Get the detector number for a given guider detector name.

        Parameters
        ----------
        detName : `str`
            The name of the detector.

        Returns
        -------
        detNum : `int`
            The detector number for the specified guider detector.
        """
        if detName not in self.guiderNameMap:
            raise ValueError(f"Detector {detName} not found in guiderNameMap.")
        return self.guiderNameMap[detName]


class GuiderReader:
    """Class to read and unpack the Guider data from Butler.

    Works in the summit and usdf environments.

    Key Attributes:
        dataset (dict): Dictionary of guider data
        guiders (dict): Dictionary of guider detector information

    Example:
        from lsst.summit.utils.guiders.reading import GuiderReader
        from lsst.daf.butler import Butler
        butler = Butler("embargo", collections="LSSTCam/raw/guider")

        seqNum, dayObs = 461, 20250425
        reader = GuiderReader(butler, view="dvcs", verbose=True)
        guider = reader.get(dayObs=dayObs, seqNum=seqNum)

        # The GuiderData class has all you need
        print(10*'-----')
        # The object now holds everything you need:
        print("Guider detectors available :", guider.guiderNames)
        print("Timestamp first value [MJD]:", guider.timestamps[0])
        print("Header fields              :", guider.header)
        print(10*'-----')
    """

    def __init__(
        self,
        butler: Optional[Butler] = None,
        view: str = "dvcs",
        verbose: bool = False,
    ):
        if butler is None:
            self.butler = butlerUtils.makeDefaultButler("LSSTCam")
        else:
            self.butler = butler

        assert self.butler is not None, "Butler must be provided or created."

        # Define camera objects
        self.camera = LsstCam.getCamera()

        self.view = view
        self.verbose = verbose
        self.guiderNameMap: dict[str, int] = {}
        self.log = logging.getLogger(__name__)

        for detector in self.camera:
            if detector.getType() == cameraGeom.DetectorType.GUIDER:
                detName = detector.getName()
                self.guiderNameMap[detName] = detector.getId()

        self.detNames = list(self.guiderNameMap.keys())
        self.nGuiders = len(self.guiderNameMap)
        self.axisRowMap = self.getAxisRowMapping()  # 0 for Y, 1 for X

    def get(self, dayObs: int, seqNum: int, detectors: list[int] | None = None) -> GuiderData:
        """Get the guider data for a given day of observation and sequence
        number.

        Parameters
        ----------
        dayObs : `int`
            Day of observation in YYYYMMDD format.
        seqNum : `int`
            Sequence number of the observation.
        detectors : `list[int]`, optional
            List of detector IDs to filter the data.
            If ``None``, all detectors will be included.

        Returns
        -------
        guiderData : `GuiderData`
            An instance of `GuiderData` containing the guider data.
        """
        if detectors is not None:
            # TODO: Add option to only get data for some detectors
            raise NotImplementedError("Filtering by specific detectors is not yet implemented.")
        # 0. Get the WCS map for all guiders
        wcs = self.getWCSMap(dayObs, seqNum)

        # 1. Get timestamps for the guider with the maximum number of stamps
        timestamps = self.getTimestamps(dayObs, seqNum)
        nstamps = len(timestamps)
        freq = timestamps.freq * 86400.0  # seconds

        if nstamps <= 1:
            raise StampsNotFoundError(
                f"Only {nstamps} stamps found for dayObs {dayObs}, seqNum {seqNum}. "
                "At least 2 stamps are required to create GuiderData."
            )

        # 2. Get data for all guiders, padding with empty stamps if necessary
        perDetectorData = self.getDataForAllDetectors(dayObs, seqNum, nstamps)

        # 3. Get header info, expId, filter, etc.
        header = self.getHeaderInfo(
            perDetectorData[self.detNames[0]], nstamps, freq, wcs["camera_rot_angle_deg"]
        )  # assume all the same for now

        # 4. Get the amplifier names for each guider
        roiAmpNames = self.getRoiAmpNames(perDetectorData)

        if self.verbose:
            self.printHeaderInfo(header)

        # 5. Pack everything into a GuiderData object
        guiderData = GuiderData(
            dayObs=dayObs,
            seqNum=seqNum,
            header=header,
            timestamps=timestamps,
            view=self.view,
            roiAmpNames=roiAmpNames,
            guiderNameMap=self.guiderNameMap,
            datasets=perDetectorData,
            wcs=wcs,
            axisRowMap=self.axisRowMap,
        )
        return guiderData

    def getTimestamps(self, dayObs: int, seqNum: int) -> Time:
        """
        Get timestamps for the guider with the maximum number of stamps.

        Parameters
        ----------
        dayObs : int
            Day of observation in YYYYMMDD format.
        seqNum : int
            Sequence number of the observation.

        Returns
        -------
        timestamps : astropy.time.Time
            Time array for each stamp (possibly masked for missing).
        """
        # find the guider with the maximum number of stamps
        nstamps = {}

        # loop over all guiders to find the max number of stamps
        for detName, detNum in self.guiderNameMap.items():
            try:
                n = self.butler.get(
                    "guider_raw",
                    day_obs=dayObs,
                    seq_num=seqNum,
                    detector=detNum,
                    instrument="LSSTCam",
                ).metadata["N_STAMPS"]
            except DatasetNotFoundError:
                self.log.warning(
                    f"No guider data found for dayObs {dayObs}, seqNum {seqNum}, detector {detName}."
                )
                n = 0
            nstamps[detName] = n
        nstamp = max(nstamps.values())

        if nstamp <= 1:
            self.log.warning(
                f"Only {nstamp} stamps found for dayObs {dayObs}, seqNum {seqNum}. "
                "Returning empty timestamps."
            )
            empty = Time(np.array([]), format="mjd", scale="utc")
            empty.freq = 0.0
            return empty

        # get the name of the guider witxh the max number of stamps
        detNameMax = [k for k, v in nstamps.items() if v == nstamp][0]

        # read the data for that guider
        raw_stamps = self.butler.get(
            "guider_raw",
            day_obs=dayObs,
            seq_num=seqNum,
            detector=self.guiderNameMap[detNameMax],
            instrument="LSSTCam",
        )

        # build the timestamps array
        timestamps = get_timestamps(raw_stamps, nstamp)
        return timestamps

    def getDataForAllDetectors(
        self, dayObs: int, seqNum: int, nstamps: int | None = None
    ) -> dict[str, Stamps]:
        """Load the data from the butler for all guider detectors.

        Parameters
        ----------
        dayObs : `int`
            Day of observation in YYYYMMDD format.
        seqNum : `int`
            Sequence number of the observation.

        Returns
        -------
        dataset : `dict[str, Stamps]`
            Dictionary with guider detector names as keys and Stamps objects as
            values.
        """
        assert self.butler is not None, "Butler must be provided or created."  # make mypy happy
        dataset: dict[str, Stamps] = {}

        # max(ntampsGuiders)
        for detName, detNum in self.guiderNameMap.items():
            if self.view == "roi":
                # TODO: Fix this!
                dataset[detName] = self.butler.get(
                    "guider_raw", day_obs=dayObs, seq_num=seqNum, detector=detNum
                )
            elif self.view == "dvcs":
                dataset[detName] = getGuiderStamps(
                    detNum,
                    seqNum,
                    dayObs,
                    butler=self.butler,
                    view="dvcs",
                    nstamps=nstamps,
                )
            elif self.view == "ccd":
                dataset[detName] = getGuiderStamps(
                    detNum,
                    seqNum,
                    dayObs,
                    butler=self.butler,
                    view="ccd",
                    nstamps=nstamps,
                )
            else:
                raise ValueError(f"Unknown view type: {self.view}. Use 'roi', 'dvcs', or 'ccd'.")

        return dataset

    def getWCSMap(self, dayObs: int, seqNum: int) -> dict[str, SkyWcs]:
        """Get the WCS map for all guider detectors.

        Parameters
        ----------
        dayObs : `int`
            Day of observation in YYYYMMDD format.
        seqNum : `int`
            Sequence number of the observation.

        Returns
        -------
        wcsMap : `dict[str, SkyWcs]`
            Dictionary for detector names as keys and WCS objects as values.
        """
        wcsGuideMap: dict[str, SkyWcs] = {}
        dataId = {
            "instrument": "LSSTCam",
            "day_obs": dayObs,
            "seq_num": seqNum,
            "detector": 94,  # any guider detector will do
        }
        # Get visitInfo from any detector
        visitInfo = self.butler.get("raw.visitInfo", dataId)
        # this is a dict with the WCS for each guider detector
        wcsGuideMap = make_init_guider_wcs(self.camera, visitInfo)

        # Add the camera rotation angle in degrees
        wcsGuideMap["camera_rot_angle_deg"] = get_camera_rot_angle(visitInfo)
        return wcsGuideMap

    def getAxisRowMapping(self) -> dict[str, int]:
        """Get the axis mapping for the rows for all guider detectors.

        Returns
        -------
        axisMap : `dict[str, int]`
            Dictionary with detector names as keys and axis mapping as values.
            The axis mapping is 1 for X and 0 for Y.
        """
        axisMap: dict[str, int] = {}
        for detName in self.guiderNameMap.keys():
            detNum = self.guiderNameMap[detName]
            detector = self.camera[detNum]
            nq = detector.getOrientation().getNQuarter()
            axisMap[detName] = 0 if nq % 2 == 0 else 1
        return axisMap

    def getRoiAmpNames(self, dataset: dict[str, Stamps]) -> dict[str, str]:
        """Get the name of the amplifier used for the ROI for each guider
        detector in the focal plane.

        Returns
        -------
        ampNames : dict[str, str]
            Dictionary with detector names as keys and amplifier names as
            values.
        """
        ampNames: dict[str, str] = {}
        for detName in dataset.keys():
            md = dataset[detName].metadata.toDict()
            # also get the ampName
            segment = md["ROISEG"]
            ampName = "C" + segment[7:]
            ampNames[detName] = ampName
        return ampNames

    def getHeaderInfo(
        self, raw: Stamps, nstamps: int, freq: float, cam_rot_angle: float = 0.0
    ) -> dict[str, str | float]:
        info: dict[str, str | float] = {}
        m = raw.metadata.toDict()
        info["n_stamps"] = nstamps
        info["start_time"] = m["GDSSTART"]
        info["roi_cols"] = m["ROICOLS"]
        info["roi_rows"] = m["ROIROWS"]
        info["FREQ"] = float(freq)
        info["expid"] = int(m["DAYOBS"]) * 100000 + int(m["SEQNUM"])
        info["filter"] = m["FILTBAND"]
        info["SHUTTIME"] = float(m.get("SHUTTIME", 30.0))
        info["AZSTART"] = float(m.get("AZSTART", 0))
        info["ELSTART"] = float(m.get("ELSTART", 0))
        info["AZEND"] = m.get("AZEND", None)
        info["ELEND"] = m.get("ELEND", None)
        info["SEEING"] = m.get("SEEING", None)
        info["CAM_ROT_ANGLE"] = cam_rot_angle
        return info

    def printHeaderInfo(self, header: dict[str, str | float]) -> None:
        # TODO: reinstate dataId and filter if necessary
        print(f"Data Id: {header['expid']}, filter-band: {header['filter']}")
        print(f"ROI Shape (row, col): {header['roi_rows']}, {header['roi_cols']}")
        mystr = f"With nstamps {int(header['n_stamps'])}"
        mystr = mystr + f" at {header['FREQ']} Hz"
        print(mystr)
        print(f"Acq. Start Time: {header['start_time']}")

    @property
    def guiderNames(self):
        """Get the names of the guider detectors."""
        return list(self.guiderNameMap.keys())

    @property
    def guiderIds(self):
        """Get the ids of the guider detectors."""
        return list(self.guiderNameMap.values())


# TODO: Check missing stamps
def getGuiderStamps(
    detNum: int,
    seqNum: int,
    dayObs: int,
    butler: Butler,
    view: str = "dvcs",
    nstamps: int | None = None,
    whichstamps: list[int] | None = None,
) -> Stamps:
    """
    This class reads the stamp object from the Butler for one Guider and
    converts them to DVCS view, making a new Stamps object

    Parameters
    ----------
    detNum : `int`
        Detector Id
    seqNum : `int`
        Sequence Number
    dayObs : `int`
        Day Observation
    repo : `str`
        Butler repo
    collections : `list` of `str`
        Butler collections
    butler : `lsst.daf.butler.Butler`, optional
        Butler object. If None, a new Butler will be created.
    view : `str`, optional
        View type, either 'dvcs' or 'ccd'. Default is 'dvcs'.
    whichstamps : `list` of `int`, optional
        List of stamp indices to read. If None, all stamps will be read.

    Returns
    -------
    stamps : `lsst.meas.algorithms.stamps.Stamps`
        Stamp images oriented in DVCS or CCD view, depending on the `view`
        parameter.
    """
    # get Camera object
    camera = LsstCam.getCamera()
    detector = camera[detNum]

    # for dayObs of 20250509 or before, the ROIs are swapped between SG0 and
    # SG1. Fix here
    ccd_swapped: str = ""
    if dayObs < 20250509:
        detName = camera[detNum].getName()
        raft = detName[0:3]
        ccd = detName[4:7]
        if ccd == "SG0":
            ccd_swapped = "SG1"
        elif ccd == "SG1":
            ccd_swapped = "SG0"

        detName_swapped = raft + "_" + ccd_swapped
        detector_swapped = camera[detName_swapped]
        detNum_swapped = detector_swapped.getId()

        dataId = {
            "instrument": "LSSTCam",
            "detector": detNum_swapped,
            "day_obs": dayObs,
            "seq_num": seqNum,
        }
    else:
        dataId = {
            "instrument": "LSSTCam",
            "detector": detNum,
            "day_obs": dayObs,
            "seq_num": seqNum,
        }

    # finally read from the Butler
    try:
        raw_stamps = butler.get("guider_raw", dataId)
    except DatasetNotFoundError:
        raise StampsNotFoundError(
            f"No guider data found for dayObs {dayObs}," f"seqNum {seqNum}, detector {detNum}."
        )

    md = raw_stamps.metadata

    # fix CCD in the metadata
    if dayObs < 20250509:
        md["CCDSLOT"] = ccd_swapped

    # also get the ampName
    segment = md["ROISEG"]
    ampName = "C" + segment[7:]

    # build the CCD view Bounding Boxes
    ccd_view_bbox = mk_roi_bboxes(md, camera)

    # also build the Translation methods
    # from CCD view -> DVCS view and the reverse
    ft, bt = mk_ccd_to_dvcs(ccd_view_bbox, detector.getOrientation().getNQuarter())

    if nstamps is None:
        nstamps = md["N_STAMPS"]

    timestamps = get_timestamps(raw_stamps, nstamps)  # to populate the metadata timestamps
    # freq = timestamps.freq * 86400.0  # seconds

    goodstamps = np.where(~timestamps.mask)[0].tolist()
    badstamps = np.where(timestamps.mask)[0].tolist()
    stampsDict = {}
    i = 0
    for index in goodstamps:
        masked_ims = raw_stamps.getMaskedImages()[i]
        rmd = raw_stamps[i].metadata
        rmd["DAQSTAMP"] = rmd.get("DAQSTAMP", index)

        # convert to DVCS or CCD view
        raw_roi = masked_ims.getImage().getArray()
        roi_dvcs = convert_roi(raw_roi, md, detector, ampName, camera, view=view)

        # build a Stamp Object
        output_masked_im = MaskedImageF(roi_dvcs)
        archive_element = [ccd_view_bbox, ft, bt]
        stampsDict[index] = Stamp(output_masked_im, archive_element, metadata=rmd)
        i += 1

    for index in badstamps:
        # print(
        #     f"Warning: The stamp {i} is missing ind detector {detNum},""
        #     "inserting empty stamp"
        # )
        nrows, ncols = int(md["ROIROWS"]), int(md["ROICOLS"])
        img0 = ImageF(array=np.zeros((nrows, ncols), dtype=np.float32))
        output_masked_im = ExposureF(MaskedImageF(img0))
        archive_element = [ccd_view_bbox, ft, bt]
        # create empty metadata
        md_empty = md.toDict().copy()
        md_empty["DAQSTAMP"] = index
        md_empty["STMPTMJD"] = np.nan
        stampsDict[index] = Stamp(output_masked_im, archive_element, metadata=md_empty)

    stamp_list = [stampsDict[i] for i in range(nstamps)]
    output_stamps = Stamps(stamp_list, md, use_archive=True)
    return output_stamps


def get_timestamps(raw_stamps: Stamps, nstamps: int = 50) -> Time:
    """
    Extract timestamps from a Stamps object and align them to an ideal
    sequence.

    Parameters
    ----------
    raw_stamps : Stamps
        Stamps object containing the guider stamp metadata.
    nstamps : int, optional
        Number of stamps expected (default is 50).

    Returns
    -------
    full_timestamps : astropy.time.Time
        Masked array of timestamps (MJD) for each stamp. Missing timestamps
        are masked. Has attribute 'freq' with the median interval in days.
    """
    timestamps_list = [stamp.metadata.get("STMPTMJD", np.nan) for stamp in raw_stamps]
    mjd_array = np.ma.masked_invalid(timestamps_list)
    timestamps = Time(mjd_array, format="mjd", scale="utc")

    # Infer frequency from valid timestamps
    dt = np.diff(timestamps.jd)
    freq = np.nanmedian(dt)
    start = timestamps[0].jd

    if np.isnan(freq) or freq <= 0:
        raise ValueError(
            f"Invalid frequency {freq} derived from timestamps. "
            "Ensure that the timestamps are valid and evenly spaced."
        )

    timestamps_ideal = Time(start + np.arange(nstamps) * freq, format="jd", scale="utc")

    tolerance = 0.5 * freq
    actual_jd = timestamps.jd

    # Build aligned timestamp list with np.nan where missing
    mjd_list = []
    for t_ideal in timestamps_ideal.jd:
        # Check if any actual timestamp is close to this ideal timestamp
        if np.any(np.abs(actual_jd - t_ideal) < tolerance):
            # Match found — find the closest
            idx = np.argmin(np.abs(actual_jd - t_ideal))
            mjd_list.append(actual_jd[idx])
        else:
            # No match — this is a missing timestamp
            mjd_list.append(np.nan)

    mjd_array = np.ma.masked_invalid(mjd_list)
    full_timestamps = Time(mjd_array, format="mjd", scale="utc")
    full_timestamps.freq = freq
    return full_timestamps


# dt = np.nanmedian(np.diff(timestamps.jd)) * 86400.0
# nmissing = np.count_nonzero(~timestamps.mask)


if __name__ == "__main__":
    seqNum, dayObs = 461, 20250425
    reader = GuiderReader(view="dvcs", verbose=True)
    guider = reader.get(dayObs=dayObs, seqNum=seqNum)

    # The GuiderData class has all you need
    print(10 * "-----")
    # The object now holds everything you need:
    print("Guider detectors available :", guider.guiderNames)
    print("Timestamp first value [MJD]:", guider.timestamps[0])
    print("Header fields              :", guider.header)
    print(10 * "-----")
