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

import asyncio
import nest_asyncio
from astropy.time import Time, TimeDelta
import datetime
import logging
import pandas as pd

from .utils import getSite

HAS_EFD_CLIENT = True
try:
    from lsst_efd_client import EfdClient
except ImportError:
    HAS_EFD_CLIENT = False

__all__ = [
    'makeEfdClient',
    'getEfdData',
    'expRecordToTimespan',
    'getDayObsStartTime',
    'getDayObsEndTime',
    'getSubTopics',
    'getStateAtTime',
]


TOPIC_ALIASES = {
    # the real alises with the database itself
    'MtAz': 'lsst.sal.MTMount.logevent_azimuthMotionState',
    'MtEl': 'lsst.sal.MTMount.logevent_elevationMotionState',
    'MtAzInPosition': 'lsst.sal.MTMount.logevent_azimuthInPosition',
    'MtElInPosition': 'lsst.sal.MTMount.logevent_elevationInPosition',
    'MtAzState': 'lsst.sal.MTMount.logevent_azimuthSystemState',
    'MtElState': 'lsst.sal.MTMount.logevent_elevationSystemState',
}

COMMAND_ALIASES = {
    'raDecTarget': 'lsst.sal.MTPtg.command_raDecTarget',
}

# setting aliases within the dictionary itself: making alt an alias for el
TOPIC_ALIASES['MtAlt'] = TOPIC_ALIASES['MtEl']
TOPIC_ALIASES['MtAltInPosition'] = TOPIC_ALIASES['MtElInPosition']

TIME_CHUNKING = datetime.timedelta(minutes=15)


def _getBeginEnd(dayObs, begin, end, timespan, event, expRecord):
    if expRecord is not None:
        forbiddenOpts = [event, begin, end, timespan, dayObs]
        if any([x is not None for x in forbiddenOpts]):
            raise ValueError("You can't specify both an expRecord and a begin/end or timespan or dayObs")
        begin = expRecord.timespan.begin
        end = expRecord.timespan.end
        return begin, end

    if event is not None:
        forbiddenOpts = [begin, end, timespan, dayObs]
        if any([x is not None for x in forbiddenOpts]):
            raise ValueError("You can't specify both an event and a begin/end or timespan or dayObs")
        begin = event.begin
        end = event.end
        return begin, end

    # check for dayObs, and that other options aren't inconsistently specified
    if dayObs is not None:
        forbiddenOpts = [begin, end, timespan]
        if any([x is not None for x in forbiddenOpts]):
            raise ValueError("You can't specify both a dayObs and a begin/end or timespan")
        begin = getDayObsStartTime(dayObs)
        end = getDayObsEndTime(dayObs)
        return begin, end
    # can now disregard dayObs entirely

    if begin is None:
        raise ValueError("You must specify either a dayObs or a begin/end or begin/timespan")
    # can now rely on begin, so just need to deal with end/timespan

    if end is None and timespan is None:
        raise ValueError("If you specify a begin, you must specify either a end or a timespan")
    if end is not None and timespan is not None:
        raise ValueError("You can't specify both a end and a timespan")
    if end is None:
        if timespan > datetime.timedelta(minutes=0):
            end = begin + timespan  # the normal case
        else:
            end = begin  # the case where timespan is negative
            begin = begin + timespan  # adding the negative to the start, i.e. subtracting it to bring back

    assert (begin is not None)
    assert (end is not None)
    return begin, end


def getEfdData(client, topic, *,
               columns=None,
               prePadding=0,
               postPadding=0,
               dayObs=None,
               begin=None,
               end=None,
               timespan=None,
               event=None,
               expRecord=None,
               noWarn=False,
               ):
    """Get one or more EFD topics over a time range in a non-blocking manner.

    The time range can be specified as either:
        * a dayObs, in which case the full 24 hour period is used,
        * a begin point and a end point,
        * a begin point and a timespan.
        * a mount event
        * an exposure record
    If it is desired to use an end time with a timespan, just specify it as the
    begin time and use a negative timespan.

    The results from all topics are merged into a single dataframe.

    Parameters
    ----------
    client : `lsst_efd_client.efd_helper.EfdClient`
        The EFD client to use.
    topic : `str`
        The topic to query.
    columns : `list` of `str`, optional
        The columns to query. If not specified, all columns are queried.
    prePadding : `float`
        The amount of time before the nominal start of the query to include.
    postPadding : `float`
        The amount of extra time after the nominal end of the query to include.
    dayObs : `int`, optional
        The dayObs to query. If specified, this is used to determine the begin
        and end times.
    begin : `astropy.Time`, optional
        The begin time for the query. If specified, either a end time or a
        timespan must be supplied.
    end : `astropy.Time`, optional
        The end time for the query. If specified, a begin time must also be
        supplied.
    timespan : `astropy.TimeDelta`, optional
        The timespan for the query. If specified, a begin time must also be
        supplied.
    event : `lsst.summit.utils.efdUtils.TmaEvent`, optional
        The event to query. If specified, this is used to determine the begin
        and end times, and all other options are disallowed.
    expRecord : `lsst.daf.butler.dimensions.DimensionRecord`, optional
        The exposure record containing the timespan to query. If specified, all
        other options are disallowed.
    noWarn : bool, optional
        If True, don't warn when no data is found. Useful for utility code
        which is checking for data.

    Returns
    -------
    data : `pandas.DataFrame`
        The merged data from all topics.

    Raises
    ------
    ValueError: If the topics are not in the EFD schema. ValueError: If both a
    dayObs and a begin/end or timespan are specified. ValueError: If a begin
    time is specified but no end time or timespan.

    """
    # ideally should calls mpts as necessary so that users needn't care if
    # things are packed

    # supports aliases so that you can query with them. If there is no entry in
    # the alias dict then it queries with the supplied key. The fact the schema
    # is now being checked means this shouldn't be a problem now.

    begin, end = _getBeginEnd(dayObs, begin, end, timespan, event, expRecord)
    begin -= TimeDelta(prePadding, format='sec')
    end += TimeDelta(postPadding, format='sec')

    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    ret = loop.run_until_complete(_getEfdData(client,
                                              topic,
                                              columns,
                                              begin.utc,
                                              end.utc))
    if ret.empty and not noWarn:
        log = logging.getLogger(__name__)
        log.warning(f"Topic {topic} is in the schema, but no data was returned by the query for the specified"
                    " time range")
    return ret


async def _getEfdData(client, topic, columns, begin, end):
    if columns is None:
        columns = ['*']

    availableTopics = await client.get_topics()

    topicToQuery = topic
    if topic in TOPIC_ALIASES:
        topicToQuery = TOPIC_ALIASES[topic]

    if topicToQuery not in availableTopics:
        raise ValueError(f"Topic {topicToQuery} not in EFD schema")

    data = await client.select_time_series(topicToQuery, columns, begin.utc, end.utc)

    return data


def getMostRecentRowWithDataBefore(client, topic, timeToLookBefore, warnStaleAfterNMinutes=60*12):
    """Get the most recent row of data for a topic before a given time.

    Parameters
    ----------
    client : `lsst_efd_client.efd_helper.EfdClient`
        The EFD client to use.
    topic : `str`
        The topic to query.
    timeToLookBefore : `astropy.Time`
        The time to look before.
    warnStaleAfterNMinutes : `float`, optional
        The number of minutes after which to consider the data stale and issue
        a warning.

    Returns
    -------
    row : `pandas.Series`
        The row of data.

    Raises
    ------
    ValueError: If the topic is not in the EFD schema.
    """
    staleAge = datetime.timedelta(warnStaleAfterNMinutes)

    df = pd.DataFrame()
    beginTime = timeToLookBefore
    while df.empty:
        df = getEfdData(client, topic, begin=beginTime, timespan=-TIME_CHUNKING, noWarn=True)
        beginTime -= TIME_CHUNKING

    lastRow = df.iloc[-1]
    commandTime = efdTimestampToAstropy(lastRow['private_sndStamp'])

    commandAge = timeToLookBefore - commandTime
    if commandAge > staleAge:
        log = logging.getLogger(__name__)
        log.warning(f"Component {topic} was last set {commandAge.sec/60:.1} minutes"
                    " before the requested time")

    return lastRow


def getStateAtTime(client, topic, timeToLookBefore, warnStaleAfterNMinutes=12*60):
    """Get the state of a component at a given time.

    Parameters
    ----------
    client : `lsst_efd_client.efd_helper.EfdClient`
        The EFD client to use.
    topic : `str`
        The topic to query.
    timeToLookBefore : `astropy.Time`
        The time to look before.
    warnStaleAfterNMinutes : `float`, optional
        The number of minutes after which to consider the data stale and issue
        a warning.

    Returns
    -------
    state : `Enum`
        The appropriate enumClass for the topic, holding the value it was set
        to at ``commandTime``.
    commandTime : `astropy.Time`
        The time the component was set to the returned state.

    Raises
    ------
    ValueError: If the topic is not in the EFD schema.
    """
    row = getMostRecentRowWithDataBefore(client=client,
                                         topic=topic,
                                         timeToLookBefore=timeToLookBefore,
                                         warnStaleAfterNMinutes=warnStaleAfterNMinutes)

    commandTime = efdTimestampToAstropy(row['private_sndStamp'])
    # TODO: need to a) know which column to get the value from, and
    # b) convert it to the appropriate enum class depending on topic.

    # example: for lsst.sal.MTMount.logevent_azimuthMotionState
    value = row['state']
    # state = AxisMotionState(value)

    return value, commandTime


def makeEfdClient():
    """Automatically create an EFD client based on the site.

    Returns
    -------
    efdClient : `lsst_efd_client.efd_helper.EfdClient`
        The EFD client to use for the current site.
    """
    if not HAS_EFD_CLIENT:
        raise RuntimeError("Could not create EFD client because importing lsst_efd_client failed.")

    try:
        site = getSite()
    except ValueError as e:
        raise RuntimeError("Could not create EFD client as the site could not be determined") from e

    if site == 'summit':
        return EfdClient('summit_efd')
    if site == 'base':
        return EfdClient('summit_efd_copy')
    if site in ['staff-rsp', 'rubin-devl']:
        return EfdClient('usdf_efd')

    raise RuntimeError(f"Could not create EFD client as the site {site} is not recognized")


def expRecordToTimespan(expRecord):
    """Get the timespan from an exposure record.

    Returns the timespan in a format where it can be used to directly unpack
    into a efdClient.select_time_series() call.

    Parameters
    ----------
    expRecord : `lsst.daf.butler.dimensions.ExposureRecord`
        The exposure record.

    Returns
    -------
    timespanDict : `dict`
        The timespan in a format that can be used to directly unpack into a
        efdClient.select_time_series() call.
    """
    return {'begin': expRecord.timespan.begin.utc,
            'end': expRecord.timespan.end.utc,
            }


def efdTimestampToAstropy(timestamp):
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
    return Time(timestamp, format='unix_tai')


def calcNextDay(dayObs):
    """Given an integer dayObs, calculate the next integer dayObs.

    Parameters
    ----------
    dayObs : `int`
        The dayObs, as an integer, e.g. 20231225
    """
    d1 = datetime.datetime.strptime(str(dayObs), '%Y%m%d')
    oneDay = datetime.timedelta(days=1)
    return (d1 + oneDay).strftime('%Y%m%d')


def getDayObsStartTime(dayObs):
    """Get the start of the given dayObs as an astropy.time.Time object.

    The observatory rolls the date over at UTC-12.
    """
    pythonDateTime = datetime.datetime.strptime(str(dayObs), "%Y%m%d")
    astroPyTime = Time(pythonDateTime)
    twelveHours = datetime.timedelta(hours=-12)
    oneDay = datetime.timedelta(hours=24)
    offset = TimeDelta(twelveHours, format='datetime')
    return astroPyTime + offset + oneDay


def getDayObsEndTime(dayObs):
    """Get the start of the given dayObs as an astropy.time.Time object.
    """
    return getDayObsStartTime(calcNextDay(dayObs))


def getSubTopics(client, topic):
    """Get all the sub topics within a given topic.

    Note that the topic need not be a complete one, for example, rather than
    doing `getSubTopics(client, 'lsst.sal.ATMCS')` to get all the topics for
    the AuxTel Mount Control System, you can do `getSubTopics(client,
    'lsst.sal.AT')` to get all which relate to the AuxTel in general.
    """
    loop = asyncio.get_event_loop()
    topics = loop.run_until_complete(client.get_topics())
    return sorted([t for t in topics if t.startswith(topic)])
