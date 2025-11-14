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

import datetime
from typing import TYPE_CHECKING

import astropy
from astropy import units as u
from astropy.time import Time, TimeDelta

if TYPE_CHECKING:
    from lsst.daf.butler import DimensionRecord


__all__ = [
    "expRecordToTimespan",
    "efdTimestampToAstropy",
    "astropyToEfdTimestamp",
    "offsetDayObs",
    "calcNextDay",
    "calcPreviousDay",
    "calcDayOffset",
    "getDayObsStartTime",
    "getDayObsEndTime",
    "getDayObsForTime",
]


def expRecordToTimespan(expRecord: DimensionRecord) -> dict:
    """Get the timespan from an exposure record.

    Returns the timespan in a format where it can be used to directly unpack
    into a efdClient.select_time_series() call.

    Parameters
    ----------
    expRecord : `lsst.daf.butler.DimensionRecord`
        The exposure record.

    Returns
    -------
    timespanDict : `dict`
        The timespan in a format that can be used to directly unpack into a
        efdClient.select_time_series() call.
    """
    return {
        "begin": expRecord.timespan.begin.utc,
        "end": expRecord.timespan.end.utc,
    }


def efdTimestampToAstropy(timestamp: float) -> Time:
    """Get an efd timestamp as an astropy.time.Time object.

    Parameters
    ----------
    timestamp : `float`
        The timestamp, as a float.

    Returns
    -------
    time : `astropy.time.Time`
        The timestamp as an astropy.time.Time object.
    """
    return Time(timestamp, format="unix")


def astropyToEfdTimestamp(time: Time) -> float:
    """Get astropy Time object as an efd timestamp

    Parameters
    ----------
    time : `astropy.time.Time`
        The time as an astropy.time.Time object.

    Returns
    -------
    timestamp : `float`
        The timestamp, in UTC, in unix seconds.
    """

    return time.utc.unix


def offsetDayObs(dayObs: int, nDays: int) -> int:
    """Offset a dayObs by a given number of days.

    Parameters
    ----------
    dayObs : `int`
        The dayObs, as an integer, e.g. 20231225
    nDays : `int`
        The number of days to offset the dayObs by.

    Returns
    -------
    newDayObs : `int`
        The new dayObs, as an integer, e.g. 20231225
    """
    d1 = datetime.datetime.strptime(str(dayObs), "%Y%m%d")
    oneDay = datetime.timedelta(days=nDays)
    return int((d1 + oneDay).strftime("%Y%m%d"))


def calcNextDay(dayObs: int) -> int:
    """Given an integer dayObs, calculate the next integer dayObs.

    Integers are used for dayObs, but dayObs values are therefore not
    contiguous due to month/year ends etc, so this utility provides a robust
    way to get the integer dayObs which follows the one specified.

    Parameters
    ----------
    dayObs : `int`
        The dayObs, as an integer, e.g. 20231231

    Returns
    -------
    nextDayObs : `int`
        The next dayObs, as an integer, e.g. 20240101
    """
    return offsetDayObs(dayObs, 1)


def calcPreviousDay(dayObs: int) -> int:
    """Given an integer dayObs, calculate the next integer dayObs.

    Integers are used for dayObs, but dayObs values are therefore not
    contiguous due to month/year ends etc, so this utility provides a robust
    way to get the integer dayObs which follows the one specified.

    Parameters
    ----------
    dayObs : `int`
        The dayObs, as an integer, e.g. 20231231

    Returns
    -------
    nextDayObs : `int`
        The next dayObs, as an integer, e.g. 20240101
    """
    return offsetDayObs(dayObs, -1)


def calcDayOffset(startDay: int, endDay: int) -> int:
    """Calculate the number of days between two dayObs values.

    Positive if endDay is after startDay, negative if before, zero if equal.

    Parameters
    ----------
    startDay : `int`
        The starting dayObs, e.g. 20231225.
    endDay : `int`
        The ending dayObs, e.g. 20240101.

    Returns
    -------
    offset : `int`
        The number of days from startDay to endDay.
    """
    dStart = datetime.datetime.strptime(str(startDay), "%Y%m%d")
    dEnd = datetime.datetime.strptime(str(endDay), "%Y%m%d")
    return (dEnd - dStart).days


def getDayObsStartTime(dayObs: int) -> astropy.time.Time:
    """Get the start of the given dayObs as an astropy.time.Time object.

    The observatory rolls the date over at UTC-12.

    Parameters
    ----------
    dayObs : `int`
        The dayObs, as an integer, e.g. 20231225

    Returns
    -------
    time : `astropy.time.Time`
        The start of the dayObs as an astropy.time.Time object.
    """
    pythonDateTime = datetime.datetime.strptime(str(dayObs), "%Y%m%d")
    return Time(pythonDateTime) + 12 * u.hour


def getDayObsEndTime(dayObs: int) -> Time:
    """Get the end of the given dayObs as an astropy.time.Time object.

    Parameters
    ----------
    dayObs : `int`
        The dayObs, as an integer, e.g. 20231225

    Returns
    -------
    time : `astropy.time.Time`
        The end of the dayObs as an astropy.time.Time object.
    """
    return getDayObsStartTime(dayObs) + 24 * u.hour


def getDayObsForTime(time: Time) -> int:
    """Get the dayObs in which an astropy.time.Time object falls.

    Parameters
    ----------
    time : `astropy.time.Time`
        The time.

    Returns
    -------
    dayObs : `int`
        The dayObs, as an integer, e.g. 20231225
    """
    twelveHours = datetime.timedelta(hours=-12)
    offset = TimeDelta(twelveHours, format="datetime")
    return int((time + offset).utc.isot[:10].replace("-", ""))
