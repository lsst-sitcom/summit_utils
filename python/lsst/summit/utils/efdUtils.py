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
import datetime
from lsst.summit.utils.utils import getSite  # XXX change back to relative import

# XXX Protect this import or file RFC and get that added to lsst_distrib
from lsst_efd_client import EfdClient
from lsst.utils.iteration import ensure_iterable

__all__ = [
    'makeEfdClient',
    'getEfdData',
    'expRecordToTimespan',
    'getDayObsStartTime',
    'getDayObsEndTime',
    'getSubTopics',
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
# setting aliases within the dictionary itself: making alt an alias for el
TOPIC_ALIASES['MtAlt'] = TOPIC_ALIASES['MtEl']
TOPIC_ALIASES['MtAltInPosition'] = TOPIC_ALIASES['MtElInPosition']


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
        end = begin + timespan

        # if isinstance(begin, datetime.datetime):
        #     begin = Time(begin)  # XXX handle UTC/TAI here
        # if isinstance(end, datetime.datetime):
        #     end = Time(end)  # XXX handle UTC/TAI here

    assert (begin is not None)
    assert (end is not None)
    return begin, end


def getEfdData(client, topics, *,
               prePadding=0,
               postPadding=0,
               dayObs=None,
               begin=None,
               end=None,
               timespan=None,
               event=None,
               expRecord=None,
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

    TODO: Add support for datetime objects?
    TODO: Add support for begin/end as strings?

    Parameters
    ----------
    client : `lsst_efd_client.efd_helper.EfdClient`
        The EFD client to use.
    topics : `str` or iterable of `str`
        The topics to query.
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
    # takes one of more topics and merges the results into one dataframe
    # ideally calls mpts as necessary so that users needn't care if things are
    # packed

    # supports aliases so that you can query with them. If there is no entry in
    # the alias dict then it queries with the supplied key. The fact the schema
    # is now being checked means this shouldn't be a problem now.

    begin, end = _getBeginEnd(dayObs, begin, end, timespan, event, expRecord)
    begin -= TimeDelta(prePadding, format='sec')
    end += TimeDelta(postPadding, format='sec')

    loop = asyncio.get_event_loop()
    ret = loop.run_until_complete(_getEfdData(client,
                                              topics,
                                              begin.utc,
                                              end.utc))
    return ret


async def _getEfdData(client, topics, begin, end):
    topics = list(ensure_iterable(topics))
    availableTopics = await client.get_topics()

    topicsToQuery = [t if t not in TOPIC_ALIASES else TOPIC_ALIASES[t] for t in topics]

    for topic in topicsToQuery:
        if topic not in availableTopics:
            raise ValueError(f"Topic {topic} not in EFD schema")

    data = await client.select_time_series(topicsToQuery[0], ['*'], begin.utc, end.utc)
    return data


def makeEfdClient(asychronous=True):  # XXX remove the async kwarg
    """Automatically create an EFD client based on the site.

    Returns
    -------
    efdClient : `lsst_efd_client.efd_helper.EfdClient`
        The EFD client to use for the current site.
    """
    try:
        site = getSite()
    except ValueError as e:
        raise RuntimeError("Could not create EFD client as the site could not be determined") from e

    # XXX remove non-async option
    if site == 'summit':
        return EfdClient('summit_efd', asychronous=asychronous)
    if site in ['staff-rsp', 'rubin-devl']:
        return EfdClient('usdf_efd', asychronous=asychronous)

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
    """
    pythonDateTime = datetime.datetime.strptime(str(dayObs), "%Y%m%d")
    astroPyTime = Time(pythonDateTime)
    twelveHours = datetime.timedelta(hours=-12)
    offset = TimeDelta(twelveHours, format='datetime')
    return astroPyTime + offset


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
