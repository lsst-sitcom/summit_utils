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

from typing import Optional

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
from lsst.daf.butler import Butler
from lsst.meas.algorithms.stamps import Stamp, Stamps
from lsst.obs.lsst import LsstCam  # pylint: disable=unused-import
from lsst.summit.utils.guiders.transformation import convert_roi, mk_ccd_to_dvcs, mk_roi_bboxes

FREQ = 5.0  # Hz, frequency of the guider data acquisition
DELAY = 20 / 1000  # seconds


@dataclass(slots=True)
class GuiderData:
    """Data class to hold guider data information."""

    dayObs: int
    seqNum: int
    timestamps: list[Time]
    roiAmpNames: dict[str, str]
    guiderNameMap: dict[str, int]
    datasets: dict[str, Stamps]  # TODO: Consider renaming this & making private
    header: dict[str, str | float]
    wcs: object | None = None  # WCS object, if available
    freq: float = FREQ  # TODO: Stop hard coding this, once that's possible from upstream
    view: str = "dvcs"  # view type, either 'dvcs' or 'ccd' or 'roi'
    # TODO: Add these properties back in if needed
    # filter_band: str

    def getStampArray(self, stampNum: int, detName: str) -> np.ndarray:
        """Get the stamp for a given stamp number and detector name.

        Parameters
        ----------
        stampNum : `int`
            The index of the stamp to retrieve.
        detName : `str`
            The name of the detector for which to retrieve the stamp.

        Returns
        -------
        data : `np.ndarray`
            The stamp image array for the specified detector and stamp number.
        """
        if detName not in self.datasets:
            raise ValueError(f"Detector {detName} not found in datasets.")

        stamps = self.datasets[detName]
        if stampNum >= len(stamps):
            print(f"Warning: Stamp number {stampNum} out of range for detector {detName}.")

            nrows, ncols = int(self.header["roi_rows"]), int(self.header["roi_cols"])
            return np.zeros((nrows, ncols), dtype=float)

        return stamps[stampNum].stamp_im.image.array

    def getStackedStampArray(self, detName: str, is_isr: bool = False) -> np.ndarray:
        """Get the stacked stamp for a given detector name.

        Parameters
        ----------
        detName : `str`
            The name of the detector for which to retrieve the stacked stamp.
        is_isr : `bool`, optional
            If True, subtract the median bias over the columns.

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
            if is_isr:
                img = img - np.median(img, axis=0)
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

    def getGuiderNames(self) -> list[str]:
        """Get the names of the guider detectors."""
        return list(self.guiderNameMap.keys())

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
        print("Guider detectors available :", guider.getGuiderNames())
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

        for detector in self.camera:
            if detector.getType() == cameraGeom.DetectorType.GUIDER:
                detName = detector.getName()
                self.guiderNameMap[detName] = detector.getId()

        self.detNames = list(self.guiderNameMap.keys())
        self.nGuiders = len(self.guiderNameMap)

    def get(
        self,
        dayObs: int,
        seqNum: int,
        detectors: list[int] | None = None,
        nstamps: int | None = 50,
    ) -> GuiderData:
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

        wcs = self.butler.get("raw.wcs", day_obs=dayObs, seq_num=seqNum, detector=23, instrument="LSSTCam")

        perDetectorData = self.getDataForAllDetectors(dayObs, seqNum, nstamps)
        header = self.getHeaderInfo(perDetectorData[self.detNames[0]])  # assume all the same for now
        roiAmpNames = self.getRoiAmpNames(perDetectorData)
        timestamps = self.getTimestamps(perDetectorData, header)

        if self.verbose:
            self.printHeaderInfo(header)

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
        )
        return guiderData

    def getTimestamps(self, perDetectorData, header: dict[str, str | float]) -> list[Time]:
        timestamps_all = []
        for detName in perDetectorData.keys():
            stamps = perDetectorData[detName]
            mjd_list = []
            for i in range(len(stamps)):
                mjd = stamps[i].metadata["STMPTMJD"]
                mjd_list.append(mjd if np.isfinite(mjd) else np.nan)

            mjd_array = np.ma.masked_invalid(mjd_list)
            tdet = Time(mjd_array, format="mjd", scale="utc")
            timestamps_all.append(tdet)
        # ascending array
        timestamps = np.unique(np.concatenate(timestamps_all)).tolist()
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

    def getHeaderInfo(self, raw: Stamps) -> dict[str, str | float]:
        info: dict[str, str | float] = {}
        m = raw.metadata.toDict()
        info["expid"] = int(m["DAYOBS"]) * 100000 + int(m["SEQNUM"])
        info["roi_col"] = m["ROICOL"]
        info["roi_row"] = m["ROIROW"]
        info["roi_cols"] = m["ROICOLS"]
        info["roi_rows"] = m["ROIROWS"]
        info["roiUnder"] = m.get("ROIUNDER", m.get("ROIUNDRC", 6))
        info["n_stamps"] = m["N_STAMPS"]
        info["start_time"] = m["GDSSTART"]
        info["filter"] = m["FILTBAND"]
        info["FREQ"] = FREQ
        return info

    def printHeaderInfo(self, header: dict[str, str | float]) -> None:
        # TODO: reinstate dataId and filter if necessary
        print(f"Data Id: {header['expid']}, filter-band: {header['filter']}")
        print(f"ROI Row: {header['roi_row']}, ROI Col: {header['roi_col']}")
        print(f"ROI Rows: {header['roi_rows']}, ROI Cols: {header['roi_cols']}")
        print(f"Number of Stamps: {header['n_stamps']}")
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
    raw_stamps = butler.get("guider_raw", dataId)
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
    timestamps_list = [stamp.metadata.get("STMPTMJD", np.nan) for stamp in raw_stamps]
    mjd_array = np.ma.masked_invalid(timestamps_list)
    timestamps = Time(mjd_array, format="mjd", scale="utc")

    # infer frequency from valid timestamps
    dt = np.diff(timestamps.jd)
    freq = np.nanmedian(dt)
    start = timestamps[0].jd
    timestamps_ideal = Time(start + np.arange(nstamps) * freq, format="jd", scale="utc")

    tolerance = 0.5 * freq
    actual_jd = timestamps.jd

    # Build aligned timestamp list with np.nan where missing
    mjd_list = []
    for t_ideal in timestamps_ideal.jd:
        # check if any actual timestamp is close to this ideal timestamp
        if np.any(np.abs(actual_jd - t_ideal) < tolerance):
            # match found — find the closest
            idx = np.argmin(np.abs(actual_jd - t_ideal))
            mjd_list.append(actual_jd[idx])
        else:
            # no match — this is a missing timestamp
            mjd_list.append(np.nan)

    mjd_array = np.ma.masked_invalid(mjd_list)
    full_timestamps = Time(mjd_array, format="mjd", scale="utc")
    return full_timestamps


# dt = np.nanmedian(np.diff(timestamps.jd)) * 86400.0
# nmissing = np.count_nonzero(timestamps.mask)


if __name__ == "__main__":
    seqNum, dayObs = 461, 20250425
    reader = GuiderReader(view="dvcs", verbose=True)
    guider = reader.get(dayObs=dayObs, seqNum=seqNum)

    # The GuiderData class has all you need
    print(10 * "-----")
    # The object now holds everything you need:
    print("Guider detectors available :", list(guider.datasets.keys()))
    print("Timestamp first value [MJD]:", guider.timestamps[0])
    print("Header fields              :", guider.header)
    print(10 * "-----")
