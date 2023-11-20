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
from astropy.time import Time, TimeDelta
from astropy import units as u
import datetime
import logging
import pandas as pd
import re
from deprecated.sphinx import deprecated

from lsst.utils.iteration import ensure_iterable

from .utils import getSite

HAS_EFD_CLIENT = True
try:
    from lsst_efd_client import EfdClient
except ImportError:
    HAS_EFD_CLIENT = False

__all__ = [
    'getEfdData',
    'getMostRecentRowWithDataBefore',
    'makeEfdClient',
    'expRecordToTimespan',
    'efdTimestampToAstropy',
    'astropyToEfdTimestamp',
    'clipDataToEvent',
    'calcNextDay',
    'getDayObsStartTime',
    'getDayObsEndTime',
    'getDayObsForTime',
    'getSubTopics',  # deprecated, being removed in w_2023_50
    'getTopics',
]


COMMAND_ALIASES = {
    'raDecTarget': 'lsst.sal.MTPtg.command_raDecTarget',
    'moveToTarget': 'lsst.sal.MTMount.command_moveToTarget',
    'startTracking': 'lsst.sal.MTMount.command_startTracking',
    'stopTracking': 'lsst.sal.MTMount.command_stopTracking',
    'trackTarget': 'lsst.sal.MTMount.command_trackTarget',  # issued at 20Hz - don't plot
}

# When looking backwards in time to find the most recent state event, look back
# in chunks of this size. Too small, and there will be too many queries, too
# large and there will be too much data returned unnecessarily, as we only need
# one row by definition. Will tune this parameters in consultation with SQuaRE.
TIME_CHUNKING = datetime.timedelta(minutes=15)


def _getBeginEnd(dayObs=None, begin=None, end=None, timespan=None, event=None, expRecord=None):
    """Calculate the begin and end times to pass to _getEfdData, given the
    kwargs passed to getEfdData.

    Parameters
    ----------
    dayObs : `int`
        The dayObs to query. If specified, this is used to determine the begin
        and end times.
    begin : `astropy.Time`
        The begin time for the query. If specified, either an end time or a
        timespan must be supplied.
    end : `astropy.Time`
        The end time for the query. If specified, a begin time must also be
        supplied.
    timespan : `astropy.TimeDelta`
        The timespan for the query. If specified, a begin time must also be
        supplied.
    event : `lsst.summit.utils.efdUtils.TmaEvent`
        The event to query. If specified, this is used to determine the begin
        and end times, and all other options are disallowed.
    expRecord : `lsst.daf.butler.dimensions.DimensionRecord`
        The exposure record containing the timespan to query. If specified, all
        other options are disallowed.

    Returns
    -------
    begin : `astropy.Time`
        The begin time for the query.
    end : `astropy.Time`
        The end time for the query.
    """
    if expRecord is not None:
        forbiddenOpts = [event, begin, end, timespan, dayObs]
        if any(x is not None for x in forbiddenOpts):
            raise ValueError("You can't specify both an expRecord and a begin/end or timespan or dayObs")
        begin = expRecord.timespan.begin
        end = expRecord.timespan.end
        return begin, end

    if event is not None:
        forbiddenOpts = [begin, end, timespan, dayObs]
        if any(x is not None for x in forbiddenOpts):
            raise ValueError("You can't specify both an event and a begin/end or timespan or dayObs")
        begin = event.begin
        end = event.end
        return begin, end

    # check for dayObs, and that other options aren't inconsistently specified
    if dayObs is not None:
        forbiddenOpts = [begin, end, timespan]
        if any(x is not None for x in forbiddenOpts):
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

    assert begin is not None
    assert end is not None
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
               warn=True,
               ):
    """Get one or more EFD topics over a time range, synchronously.

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
        The amount of time before the nominal start of the query to include, in
        seconds.
    postPadding : `float`
        The amount of extra time after the nominal end of the query to include,
        in seconds.
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
    warn : bool, optional
        If ``True``, warn when no data is found. Exists so that utility code
        can disable warnings when checking for data, and therefore defaults to
        ``True``.

    Returns
    -------
    data : `pd.DataFrame`
        The merged data from all topics.

    Raises
    ------
    ValueError:
        If the topics are not in the EFD schema.
    ValueError:
        If both a dayObs and a begin/end or timespan are specified.
    ValueError:
        If a begin time is specified but no end time or timespan.

    """
    # TODO: DM-40100 ideally should calls mpts as necessary so that users
    # needn't care if things are packed

    # supports aliases so that you can query with them. If there is no entry in
    # the alias dict then it queries with the supplied key. The fact the schema
    # is now being checked means this shouldn't be a problem now.

    # TODO: RFC-948 Move this import back to top of file once is implemented.
    import nest_asyncio

    begin, end = _getBeginEnd(dayObs, begin, end, timespan, event, expRecord)
    begin -= TimeDelta(prePadding, format='sec')
    end += TimeDelta(postPadding, format='sec')

    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    ret = loop.run_until_complete(_getEfdData(client=client,
                                              topic=topic,
                                              begin=begin,
                                              end=end,
                                              columns=columns))
    if ret.empty and warn:
        log = logging.getLogger(__name__)
        log.warning(f"Topic {topic} is in the schema, but no data was returned by the query for the specified"
                    " time range")
    return ret


async def _getEfdData(client, topic, begin, end, columns=None):
    """Get data for a topic from the EFD over the specified time range.

    Parameters
    ----------
    client : `lsst_efd_client.efd_helper.EfdClient`
        The EFD client to use.
    topic : `str`
        The topic to query.
    begin : `astropy.Time`
        The begin time for the query.
    end : `astropy.Time`
        The end time for the query.
    columns : `list` of `str`, optional
        The columns to query. If not specified, all columns are returned.

    Returns
    -------
    data : `pd.DataFrame`
        The data from the query.
    """
    if columns is None:
        columns = ['*']
    columns = list(ensure_iterable(columns))

    availableTopics = await client.get_topics()

    if topic not in availableTopics:
        raise ValueError(f"Topic {topic} not in EFD schema")

    data = await client.select_time_series(topic, columns, begin.utc, end.utc)

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
    row : `pd.Series`
        The row of data from the EFD containing the most recent data before the
        specified time.

    Raises
    ------
    ValueError:
        If the topic is not in the EFD schema.
    """
    staleAge = datetime.timedelta(warnStaleAfterNMinutes)

    firstDayPossible = getDayObsStartTime(20190101)

    if timeToLookBefore < firstDayPossible:
        raise ValueError(f"Requested time {timeToLookBefore} is before any data was put in the EFD")

    df = pd.DataFrame()
    beginTime = timeToLookBefore
    while df.empty and beginTime > firstDayPossible:
        df = getEfdData(client, topic, begin=beginTime, timespan=-TIME_CHUNKING, warn=False)
        beginTime -= TIME_CHUNKING

    if beginTime < firstDayPossible and df.empty:  # we ran all the way back to the beginning of time
        raise ValueError(f"The entire EFD was searched backwards from {timeToLookBefore} and no data was "
                         f"found in {topic=}")

    lastRow = df.iloc[-1]
    commandTime = efdTimestampToAstropy(lastRow['private_efdStamp'])

    commandAge = timeToLookBefore - commandTime
    if commandAge > staleAge:
        log = logging.getLogger(__name__)
        log.warning(f"Component {topic} was last set {commandAge.sec/60:.1} minutes"
                    " before the requested time")

    return lastRow


def makeEfdClient(testing=False):
    """Automatically create an EFD client based on the site.

    Parameters
    ----------
    testing : `bool`, optional
        Set to ``True`` if running in a test suite. This will default to using
        the USDF EFD, for which data has been recorded for replay by the ``vcr`
        package. Note data must be re-recorded to ``vcr`` from both inside and
        outside the USDF when the package/data changes, due to the use of a
        proxy meaning that the web requests are different depending on whether
        the EFD is being contacted from inside and outside the USDF.

    Returns
    -------
    efdClient : `lsst_efd_client.efd_helper.EfdClient`
        The EFD client to use for the current site.
    """
    if not HAS_EFD_CLIENT:
        raise RuntimeError("Could not create EFD client because importing lsst_efd_client failed.")

    if testing:
        return EfdClient('usdf_efd')

    try:
        site = getSite()
    except ValueError as e:
        raise RuntimeError("Could not create EFD client as the site could not be determined") from e

    if site == 'summit':
        return EfdClient('summit_efd')
    if site == 'tucson':
        return EfdClient('tucson_teststand_efd')
    if site == 'base':
        return EfdClient('base_efd')
    if site in ['staff-rsp', 'rubin-devl', 'usdf-k8s']:
        return EfdClient('usdf_efd')
    if site == 'jenkins':
        return EfdClient('usdf_efd')

    raise RuntimeError(f"Could not create EFD client as the {site=} is not recognized")


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
    return Time(timestamp, format='unix')


def astropyToEfdTimestamp(time):
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


def clipDataToEvent(df, event, prePadding=0, postPadding=0, logger=None):
    """Clip a padded dataframe to an event.

    Parameters
    ----------
    df : `pd.DataFrame`
        The dataframe to clip.
    event : `lsst.summit.utils.efdUtils.TmaEvent`
        The event to clip to.
    prePadding : `float`, optional
        The amount of time before the nominal start of the event to include, in
        seconds.
    postPadding : `float`, optional
        The amount of extra time after the nominal end of the event to include,
        in seconds.
    logger : `logging.Logger`, optional
        The logger to use. If not specified, a new one is created.

    Returns
    -------
    clipped : `pd.DataFrame`
        The clipped dataframe.
    """
    begin = event.begin.value - prePadding
    end = event.end.value + postPadding

    if logger is None:
        logger = logging.getLogger(__name__)

    if begin < df['private_efdStamp'].min():
        logger.warning(f"Requested begin time {begin} is before the start of the data")
    if end > df['private_efdStamp'].max():
        logger.warning(f"Requested end time {end} is after the end of the data")

    mask = (df['private_efdStamp'] >= begin) & (df['private_efdStamp'] <= end)
    clipped_df = df.loc[mask].copy()
    return clipped_df


def calcNextDay(dayObs):
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
    d1 = datetime.datetime.strptime(str(dayObs), '%Y%m%d')
    oneDay = datetime.timedelta(days=1)
    return int((d1 + oneDay).strftime('%Y%m%d'))


def getDayObsStartTime(dayObs):
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


def getDayObsEndTime(dayObs):
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


def getDayObsForTime(time):
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
    offset = TimeDelta(twelveHours, format='datetime')
    return int((time + offset).utc.isot[:10].replace('-', ''))


@deprecated(
    reason="getSubTopics() has been replaced by getTopics() and using wildcards. "
           "Will be removed after w_2023_50.",
    version="w_2023_40",
    category=FutureWarning,
)
def getSubTopics(client, topic):
    """Get all the sub topics within a given topic.

    Note that the topic need not be a complete one, for example, rather than
    doing `getSubTopics(client, 'lsst.sal.ATMCS')` to get all the topics for
    the AuxTel Mount Control System, you can do `getSubTopics(client,
    'lsst.sal.AT')` to get all which relate to the AuxTel in general.

    Parameters
    ----------
    client : `lsst_efd_client.efd_helper.EfdClient`
        The EFD client to use.
    topic : `str`
        The topic to query.

    Returns
    -------
    subTopics : `list` of `str`
        The sub topics.
    """
    loop = asyncio.get_event_loop()
    topics = loop.run_until_complete(client.get_topics())
    return sorted([t for t in topics if t.startswith(topic)])


def getTopics(client, toFind, caseSensitive=False):
    """Return all the strings in topics which match the topic query string.

    Supports wildcards, which are denoted as `*``, as per shell globs.

    Example:
    >>> # assume topics are ['apple', 'banana', 'grape']
    >>> getTopics(, 'a*p*')
    ['apple', 'grape']

    Parameters
    ----------
    client : `lsst_efd_client.efd_helper.EfdClient`
        The EFD client to use.
    toFind : `str`
        The query string, with optional wildcards denoted as *.
    caseSensitive : `bool`, optional
        If ``True``, the query is case sensitive. Defaults to ``False``.

    Returns
    -------
    matches : `list` of `str`
        The list of matching topics.
    """
    loop = asyncio.get_event_loop()
    topics = loop.run_until_complete(client.get_topics())

    # Replace wildcard with regex equivalent
    pattern = toFind.replace('*', '.*')
    flags = re.IGNORECASE if not caseSensitive else 0

    matches = []
    for topic in topics:
        if re.match(pattern, topic, flags):
            matches.append(topic)

    return matches
