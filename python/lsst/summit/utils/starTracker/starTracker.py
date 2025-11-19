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

import datetime
import os
from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from PIL import Image

import lsst.afw.image as afwImage
import lsst.daf.base as dafBase
from lsst.afw.image import ExposureInfo, VisitInfo
from lsst.summit.utils.dateTime import dayObsIntToString

__all__ = (
    "KNOWN_CAMERAS",
    "narrowCam",
    "wideCam",
    "fastCam",
    "StarTrackerCamera",
    "tifToExp",
    "fitsToExp",
    "openFile",
    "dayObsToDateTime",
    "isStreamingModeFile",
    "dayObsSeqNumFromFilename",
    "dayObsSeqNumFrameNumFromFilename",
    "getRawDataDirForDayObs",
)

KNOWN_CAMERAS = ("narrow", "wide", "fast")


@dataclass(frozen=True, kw_only=True, slots=True)
class StarTrackerCamera:
    """A frozen dataclass for StarTracker camera configs"""

    cameraType: str
    suffix: str
    suffixWithSpace: str
    doAstrometry: bool
    cameraNumber: int
    snr: float
    minPix: int
    brightSourceFraction: float
    scaleError: float
    doSmoothPlot: bool


narrowCam = StarTrackerCamera(
    cameraType="narrow",
    suffix="_narrow",
    suffixWithSpace=" narrow",
    doAstrometry=True,
    cameraNumber=102,
    snr=5,
    minPix=25,
    brightSourceFraction=0.95,
    scaleError=5,
    doSmoothPlot=True,
)
wideCam = StarTrackerCamera(
    cameraType="wide",
    suffix="_wide",
    suffixWithSpace=" wide",
    doAstrometry=True,
    cameraNumber=101,
    snr=5,
    minPix=25,
    brightSourceFraction=0.8,
    scaleError=5,
    doSmoothPlot=True,
)
fastCam = StarTrackerCamera(
    cameraType="fast",
    suffix="_fast",
    suffixWithSpace=" fast",
    doAstrometry=True,
    cameraNumber=103,
    snr=2.5,
    minPix=10,
    brightSourceFraction=0.95,
    scaleError=60,
    doSmoothPlot=False,
)


def tifToExp(filename: str) -> afwImage.Exposure:
    """Open a tif image as an exposure.

    Opens the file, sets a blank mask plane, and converts the data to
    `np.float32` and returns an exposure, currently with no visitInfo.

    TODO: DM-38422 Once we have a way of getting the expTime, set that,
    and the frequency at which the images were taken.

    Parameters
    ----------
    filename : `str`
        The full path to the file to load.

    Returns
    -------
    exp : `lsst.afw.image.Exposure`
        The exposure.
    """
    im = Image.open(filename)
    imageData = im.getdata()
    data = np.asarray(imageData, dtype=np.float32)
    data = data.reshape(im.height, im.width)
    img = afwImage.ImageF(data)
    maskedIm = afwImage.MaskedImageF(img)
    exp = afwImage.ExposureF(maskedIm)
    return exp


def fitsToExp(filename: str) -> afwImage.Exposure:
    """Open a fits file as an exposure.

    Parameters
    ----------
    filename : `str`
        The full path to the file to load.

    Returns
    -------
    exp : `lsst.afw.image.Exposure`
        The exposure.
    """
    with fits.open(filename) as f:
        header = f[0].header
        data = f[1].data

    data = np.asarray(data, dtype=np.float32)
    img = afwImage.ImageF(data)
    maskedIm = afwImage.MaskedImageF(img)

    viDict = {}
    viDict["exposureTime"] = header.get("EXPTIME")

    # set the midpoint of BEG and END as the DATE
    begin = datetime.datetime.fromisoformat(header.get("DATE-BEG"))
    end = datetime.datetime.fromisoformat(header.get("DATE-END"))
    mid = begin + (end - begin) / 2
    newTime = dafBase.DateTime(mid.isoformat(), dafBase.DateTime.Timescale.TAI)
    viDict["date"] = newTime

    vi = VisitInfo(**viDict)
    expInfo = ExposureInfo(visitInfo=vi)
    exp = afwImage.ExposureF(maskedIm, exposureInfo=expInfo)
    return exp


def openFile(filename: str) -> afwImage.Exposure:
    """Open a file as an exposure, based on the file type.

    Parameters
    ----------
    filename : `str`
        The full path to the file to load.

    Returns
    -------
    exp : `lsst.afw.image.Exposure`
        The exposure.
    """
    if filename.endswith(".tif"):
        return tifToExp(filename)
    elif filename.endswith(".fits"):
        return fitsToExp(filename)
    else:
        raise ValueError("File type not recognized")


def dayObsToDateTime(dayObs: int) -> datetime.datetime:
    """Convert a dayObs to a datetime.

    Parameters
    ----------
    dayObs : `int`
        The dayObs.

    Returns
    -------
    datetime : `datetime`
        The datetime.
    """
    return datetime.datetime.strptime(dayObsIntToString(dayObs), "%Y-%m-%d")


def isStreamingModeFile(filename: str) -> bool:
    """Check if a filename is a streaming mode file.

    Parameters
    ----------
    filename : `str`
        The filename.

    Returns
    -------
    isStreaming : `bool`
        Whether the file is a streaming mode file.
    """
    # non-streaming filenames are like GC103_O_20240304_000009.fits
    # streaming filenames are like GC103_O_20240304_000007_0001316.fits
    # which is <camNum>_O_<dayObs>_<seqNum>_<streamSeqNum>.fits
    # so 5 sections means streaming, 4 means normal
    return os.path.basename(filename).count("_") == 4


def dayObsSeqNumFromFilename(filename: str) -> tuple[int, int] | tuple[None, None]:
    """Get the dayObs and seqNum from a filename.

    If the file is a streaming mode file (`None`, `None`) is returned.

    Parameters
    ----------
    filename : `str`
        The filename.

    Returns
    -------
    dayObs : `int` or `None`
        The dayObs.
    seqNum : `int` or `None`
        The seqNum.
    """
    # filenames are like GC101_O_20221114_000005.fits
    filename = os.path.basename(filename)  # in case we're passed a full path

    # these must not be processed like normal files as they're a part of a long
    # series, so return None, None even if that potentially causes problems
    # elsewhere, that code needs to deal with that.
    if isStreamingModeFile(filename):
        return None, None

    # this is a regular file
    parts = filename.split("_")
    _, _, dayObs, seqNumAndSuffix = parts
    seqNum = seqNumAndSuffix.removesuffix(".fits")

    return int(dayObs), int(seqNum)


def dayObsSeqNumFrameNumFromFilename(filename: str) -> tuple[int, int, int]:
    """Get the dayObs, seqNum and frameNum from a filename.

    If the file is not a streaming mode file then a `ValueError` is raised.

    Parameters
    ----------
    filename : `str`
        The filename.

    Returns
    -------
    dayObs : `int`
        The dayObs.
    seqNum : `int`
        The seqNum.
    frameNum : `int`
        The frameNum.

    Raises
    ------
    ValueError
        Raised if the file is not a streaming mode file.
    """
    # filenames are like GC103_O_20240308_000169_0000321.fits
    # which follows the pattern <camNum>_O_<dayObs>_<seqNum>_<frameNum>.fits
    filename = os.path.basename(filename)  # in case we're passed a full path

    if not isStreamingModeFile(filename):
        raise ValueError(f"{filename} is not a streaming mode file")

    # this is a regular file
    parts = filename.split("_")
    _, _, dayObs, seqNum, frameNumAndSuffix = parts
    frameNum = frameNumAndSuffix.removesuffix(".fits")

    return int(dayObs), int(seqNum), int(frameNum)


def getRawDataDirForDayObs(rootDataPath: str, camera: StarTrackerCamera, dayObs: int) -> str:
    """Get the raw data dir for a given dayObs.

    Parameters
    ----------
    rootDataPath : `str`
        The root data path.
    camera : `lsst.summit.utils.starTracker.StarTrackerCamera`
        The camera to get the raw data for.
    dayObs : `int`
        The dayObs.
    """
    camNum = camera.cameraNumber
    dayObsDateTime = datetime.datetime.strptime(str(dayObs), "%Y%m%d")
    dirSuffix = (
        f"GenericCamera/{camNum}/{dayObsDateTime.year}/" f"{dayObsDateTime.month:02}/{dayObsDateTime.day:02}/"
    )
    return os.path.join(rootDataPath, dirSuffix)
