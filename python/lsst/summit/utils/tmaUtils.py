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

import re
import enum
import itertools
import logging
import pandas as pd
from dataclasses import dataclass
from astropy.time import Time
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from lsst.utils.iteration import ensure_iterable

from .utils import getCurrentDayObs_int, dayObsIntToString
from .efdUtils import (getEfdData,
                       makeEfdClient,
                       efdTimestampToAstropy,
                       COMMAND_ALIASES,
                       )

__all__ = (
    'TMA',
    'TMAEvent',
    'TMAEventMaker',
    'TMAState',
    'AxisMotionState',
    'PowerState',
    'getSlewsFromEventList',
    'getTracksFromEventList',
)

# we don't want to use `None` for a no data sentinel because dict.get('key')
# returns None if the key isn't present, and also we need to mark that the data
# was queried for and no data was found, whereas the key not being present
# means that we've not yet looked for the data.
NO_DATA_SENTINEL = "NODATA"


def getSlewsFromEventList(events):
    """Get the slew events from a list of TMAEvents.

    Parameters
    ----------
    events : `list` of `TMAEvent`
        The list of events to filter.

    Returns
    -------
    events : `list` of `TMAEvent`
        The filtered list of events.
    """
    return [e for e in events if e.type == TMAState.SLEWING]


def getTracksFromEventList(events):
    """Get the tracking events from a list of TMAEvents.

    Parameters
    ----------
    events : `list` of `TMAEvent`
        The list of events to filter.

    Returns
    -------
    events : `list` of `TMAEvent`
        The filtered list of events.
    """
    return [e for e in events if e.type == TMAState.TRACKING]


def getAzimuthElevationDataForEvent(client, event, prePadding=0, postPadding=0):
    """

    Parameters
    ----------
    client : `lsst_efd_client.efd_helper.EfdClient`
        The EFD client to use.
    event : `lsst.summit.utils.tmaUtils.TMAEvent`
        The event to plot.
    fig : `matplotlib.figure.Figure`, optional
        The figure to plot on. If not specified, a new figure will be created.
    prePadding : `float`, optional
        The amount of time to pad the event with before the start time, in
        seconds.
    postPadding : `float`, optional
        The amount of time to pad the event with after the end time, in
        seconds.
    """
    azimuthData = getEfdData(client,
                             'lsst.sal.MTMount.azimuth',
                             event=event,
                             prePadding=prePadding,
                             postPadding=postPadding)
    elevationData = getEfdData(client,
                               'lsst.sal.MTMount.elevation',
                               event=event,
                               prePadding=prePadding,
                               postPadding=postPadding)

    return azimuthData, elevationData


def plotEvent(client, event, fig=None, prePadding=0, postPadding=0, commands={},
              azimuthData=None, elevationData=None):
    """Plot the TMA axis states for a given TMAEvent.

    Parameters
    ----------
    client : `lsst_efd_client.efd_helper.EfdClient`
        The EFD client to use.
    event : `lsst.summit.utils.tmaUtils.TMAEvent`
        The event to plot.
    fig : `matplotlib.figure.Figure`, optional
        The figure to plot on. If not specified, a new figure will be created.
    prePadding : `float`, optional
        The amount of time to pad the event with before the start time, in
        seconds.
    postPadding : `float`, optional
        The amount of time to pad the event with after the end time, in
        seconds.
    commands : `dict` of `str` : `astropy.time.Time`, optional
        A dictionary of commands to plot on the figure. The keys are the
        topic names, and the values are the times at which the commands were
        sent.
    azimuthData : `pandas.DataFrame`, optional
        The azimuth data to plot. If not specified, it will be queried from the
        EFD.
    elevationData : `pandas.DataFrame`, optional
        The elevation data to plot. If not specified, it will be queried from
        the EFD.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        The figure on which the plot was made.
    """
    def tickFormatter(value, tick_number):
        # Convert the value to a string without subtracting large numbers
        # tick_number is unused.
        return f"{value:.2f}"

    if fig is None:
        fig = plt.figure(figsize=(10, 6))
        log = logging.getLogger(__name__)
        log.warning("Making new matplotlib figure - if this is in a loop you're going to have a bad time."
                    " Pass in a figure with fig = plt.figure(figsize=(10, 6)) to avoid this warning.")
    ax1 = fig.gca()

    if azimuthData is None or elevationData is None:
        azimuthData, elevationData = getAzimuthElevationDataForEvent(client,
                                                                     event,
                                                                     prePadding=prePadding,
                                                                     postPadding=postPadding)

    # Use the native color cycle for the lines. Because they're on different
    # axes they don't cycle by themselves
    linesColourCycle = [p['color'] for p in plt.rcParams['axes.prop_cycle']]
    colorCounter = 0

    ax1.plot(azimuthData['actualPosition'], label='Azimuth position', c=linesColourCycle[colorCounter])
    colorCounter += 1
    ax1.yaxis.set_major_formatter(FuncFormatter(tickFormatter))
    ax1.set_ylabel('Azimuth (degrees)')
    ax1.set_xlabel('Time (UTC)')  # yes, it really is UTC, matplotlib convert this automatically!

    ax2 = ax1.twinx()
    ax2.plot(elevationData['actualPosition'], label='Elevation position', c=linesColourCycle[colorCounter])
    colorCounter += 1
    ax2.yaxis.set_major_formatter(FuncFormatter(tickFormatter))
    ax2.set_ylabel('Elevation (degrees)')

    # put the ticks at an angle, and right align with the tick marks
    ax1.set_xticks(ax1.get_xticks())  # needed to supress a user warning
    xlabels = ax1.get_xticks()
    ax1.set_xticklabels(xlabels, rotation=40, ha='right')
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

    if prePadding or postPadding:
        # note the conversion to utc because the x-axis from the dataframe
        # already got automagically converted when plotting before, so this is
        # necessary for things to line up
        ax2.axvline(event.begin.utc.datetime, c='k', ls='--', alpha=0.5, label='Event begin/end')
        ax2.axvline(event.end.utc.datetime, c='k', ls='--', alpha=0.5)

    # plot any commands we might have
    if not isinstance(commands, dict):
        raise TypeError('commands must be a dict of command names with values as'
                        ' astropy.time.Time values')
    for command, commandTime in commands.items():
        # if commands weren't found, the item is set to None. This is common
        # for events so handle it gracefully and silently. The command finding
        # code logs about lack of commands found so no need to mention here.
        if commandTime is None:
            continue
        ax2.axvline(commandTime.utc.datetime, c=linesColourCycle[colorCounter],
                    ls='--', alpha=0.75, label=f'{command}')
        colorCounter += 1

    # combine the legends and put inside the plot
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    # ax2 is "in front" of ax1 because it has the vlines plotted on it, and
    # vlines are on ax2 so that they appear at the bottom of the legend, so
    # make sure to plot the legend on ax2, otherwise the vlines will go on top
    # of the otherwise-opaque legend.
    ax2.legend(handles, labels, facecolor='white', framealpha=1)

    # Add title with the event name, type etc
    dayObsStr = dayObsIntToString(event.dayObs)
    title = (f"{dayObsStr} - seqNum {event.seqNum} (version {event.version})"  # top line, rest below
             f"\nDuration = {event.duration:.2f}s"
             f" Event type: {event.type.name}"
             f" End reason: {event.endReason.name}"
             )
    ax2.set_title(title)
    return fig


def getCommandsDuringEvent(client, event, commands=['raDecTarget'], log=None, doLog=True):
    """Get the commands issued during an event.

    Get the times at which the specified commands were issued during the event.

    Parameters
    ----------
    client : `lsst_efd_client.efd_helper.EfdClient`
        The EFD client to use.
    event : `lsst.summit.utils.tmaUtils.TMAEvent`
        The event to plot.
    commands : `list` of `str`, optional
        The commands or command aliases to look for. Defaults to
        ['raDecTarget'].
    log : `logging.Logger`, optional
        The logger to use. If not specified, a new logger will be created if
        needed.
    doLog : `bool`, optional
        Whether to log messages. Defaults to True.
    """
    # TODO: Add support for padding the event here to allow looking for
    # triggering commands before the event
    if log is None and doLog:
        log = logging.getLogger(__name__)

    commands = ensure_iterable(commands)
    fullCommands = [c if c not in COMMAND_ALIASES else COMMAND_ALIASES[c] for c in commands]
    del commands  # make sure we always use their full names

    ret = {}
    for command in fullCommands:
        data = getEfdData(client, command, event=event, noWarn=True)
        if data.empty:
            if doLog:
                log.info(f'Found no command issued for {command} during event')
            ret[command] = None
        elif len(data) > 1:
            if doLog:
                log.warning(f'Found multiple commands issued for {command} during event, returning None')
            ret[command] = None
        else:
            assert len(data) == 1  # this must be true now
            commandTime = data.private_sndStamp
            ret[command] = Time(commandTime, format='unix_tai')

    return ret


def _initializeTma(tma):
    """Helper function to turn a TMA into a valid state for testing.

    Do not call directly in normal usage or code, as this just arbitrarily
    sets values to make the TMA valid.
    """
    tma._parts['azimuthInPosition'] = False
    tma._parts['azimuthMotionState'] = AxisMotionState.STOPPED
    tma._parts['azimuthSystemState'] = PowerState.ON
    tma._parts['elevationInPosition'] = False
    tma._parts['elevationMotionState'] = AxisMotionState.STOPPED
    tma._parts['elevationSystemState'] = PowerState.ON


def _turnOn(tma):
    """Helper function to turn TMA axes on for testing.

    Do not call directly in normal usage or code, as this just arbitrarily
    sets values to turn the axes on.
    """
    tma._parts['azimuthSystemState'] = PowerState.ON
    tma._parts['elevationSystemState'] = PowerState.ON


@dataclass(slots=True, kw_only=True)
class TMAEvent:
    dayObs: int
    seqNum: int
    type: str  # can be 'slew', 'track', 'pointToPoint'
    endReason: str  # can be 'slew', 'fault', 'stop', 'unfinished' - rare!
    duration: float  # seconds
    begin: Time
    end: Time
    beginFloat: float
    endFloat: float
    version: int = 0  # update this number any time a code change which could change event definitions is made

    def __lt__(self, other):
        if self.dayObs < other.dayObs:
            return True
        elif self.dayObs == other.dayObs:
            return self.seqNum < other.seqNum
        return False

    def __repr__(self):
        return (
            f"TMAEvent(dayObs={self.dayObs}, seqNum={self.seqNum}, type={self.type!r},"
            f" endReason={self.endReason!r}, duration={self.duration}, begin={self.begin!r},"
            f" end={self.end!r}, beginFloat={self.beginFloat}, endFloat={self.endFloat})"
        )

    def _ipython_display_(self):
        print(self.__str__())

    def __str__(self):
        return (
            f"dayObs: {self.dayObs}\nseqNum: {self.seqNum}\ntype: {self.type.name}"
            f"\nendReason: {self.endReason.name}\nduration: {self.duration}\nbegin: {self.begin!r},"
            f"\nend: {self.end!r}\nbeginFloat: {self.beginFloat}\nendFloat: {self.endFloat}"
        )


class TMAState(enum.IntEnum):
    """Overall state of the TMA"""
    UNINITIALIZED = -1
    STOPPED = 0
    TRACKING = 1
    SLEWING = 2
    FAULT = 3
    OFF = 4

    def __repr__(self):
        return f"TMAState.{self.name}"


class AxisMotionState(enum.IntEnum):
    """Motion state of azimuth elevation and camera cable wrap.

    Note: this is copied over from
    https://github.com/lsst-ts/ts_idl/blob/develop/python/lsst/ts/idl/enums/MTMount.py  # noqa: W505
    to save having to depend on T&S code directly. These enums are extremely
    static, so this is a reasonable thing to do, and much easier than setting
    up a dependency on ts_idl.
    """

    STOPPING = 0
    STOPPED = 1
    MOVING_POINT_TO_POINT = 2
    JOGGING = 3
    TRACKING = 4
    TRACKING_PAUSED = 5


class PowerState(enum.IntEnum):
    """Power state of a system or motion controller.

    Also used for motion controller state.

    Note that only a few systems (and no motion controllers)
    use TURNING_ON and TURNING_OFF. The oil supply system is one.

    Note: this is copied over from
    https://github.com/lsst-ts/ts_idl/blob/develop/python/lsst/ts/idl/enums/MTMount.py  # noqa: W505
    to save having to depend on T&S code directly. These enums are extremely
    static, so this is a reasonable thing to do, and much easier than setting
    up a dependency on ts_idl.
    """

    OFF = 0
    ON = 1
    FAULT = 2
    TURNING_ON = 3
    TURNING_OFF = 4
    UNKNOWN = 15


def getAxisAndType(rowFor):
    regex = r'(azimuth|elevation)(InPosition|MotionState|SystemState)'
    matches = re.search(regex, rowFor)
    axis = matches.group(1)
    rowType = matches.group(2)

    assert rowFor.endswith(f"{axis}{rowType}")
    return axis, rowType


class ReferenceList:
    """A class to allow making lists which contain references to a dictionary.

    Normally, making a list of items from a dictionary would make a copy of the
    items, but this class allows making a list which contains references to the
    underlying dictionary. This is useful for making a list of components, such
    that they can be manipulated in their logical sets.
    """
    def __init__(self, underlyingDictionary, keysToLink):
        self.dictionary = underlyingDictionary
        self.keys = keysToLink

    def __getitem__(self, index):
        return self.dictionary[self.keys[index]]

    def __setitem__(self, index, value):
        self.dictionary[self.keys[index]] = value

    def __len__(self):
        return len(self.keys)


class TMA:
    """A state machine model of the TMA.
    """
    _UNINITIALIZED_VALUE: int = -999

    def __init__(self, engineeringMode=True, debug=False):
        self.engineeringMode = engineeringMode
        self.log = logging.getLogger('lsst.summit.utils.tmaUtils.TMA')
        if debug:
            self.log.level = logging.DEBUG
        self._mostRecentRowTime = -1

        # the actual components of the TMA
        self._parts = {'azimuthInPosition': self._UNINITIALIZED_VALUE,
                       'azimuthMotionState': self._UNINITIALIZED_VALUE,
                       'azimuthSystemState': self._UNINITIALIZED_VALUE,
                       'elevationInPosition': self._UNINITIALIZED_VALUE,
                       'elevationMotionState': self._UNINITIALIZED_VALUE,
                       'elevationSystemState': self._UNINITIALIZED_VALUE,
                       }
        systemKeys = ['azimuthSystemState', 'elevationSystemState']
        positionKeys = ['azimuthInPosition', 'elevationInPosition']
        motionKeys = ['azimuthMotionState', 'elevationMotionState']

        # references to the _parts as conceptual groupings
        self.system = ReferenceList(self._parts, systemKeys)
        self.motion = ReferenceList(self._parts, motionKeys)
        self.inPosition = ReferenceList(self._parts, positionKeys)

        # tuples of states for state collapsing. Note that STOP_LIKE +
        # MOVING_LIKE must cover the full set of AxisMotionState enums
        self.STOP_LIKE = (AxisMotionState.STOPPING,
                          AxisMotionState.STOPPED,
                          AxisMotionState.TRACKING_PAUSED)
        self.MOVING_LIKE = (AxisMotionState.MOVING_POINT_TO_POINT,
                            AxisMotionState.JOGGING,
                            AxisMotionState.TRACKING)
        # Likewise, ON_LIKE + OFF_LIKE must cover the full set of PowerState
        # enums
        self.OFF_LIKE = (PowerState.OFF, PowerState.TURNING_OFF)
        self.ON_LIKE = (PowerState.ON, PowerState.TURNING_ON)
        self.FAULT_LIKE = (PowerState.FAULT,)  # note the trailing comma - this must be an iterable

    def apply(self, row):
        """Apply a row of data to the TMA state.

        Checks that the row contains data for a later time, and applies the
        relevant column entry to the relevant component.
        """
        timestamp = row['private_sndStamp']
        if timestamp < self._mostRecentRowTime:  # NB equals is OK, technically, though it never happens
            raise ValueError('TMA evolution must be monotonic increasing in time, tried to apply a row which'
                             ' predates the most previous one')
        self._mostRecentRowTime = timestamp

        rowFor = row['rowFor']  # e.g. elevationMotionState
        axis, rowType = getAxisAndType(rowFor)  # e.g. elevation, MotionState
        value = self._getRowPayload(row, rowType, rowFor)
        self.log.debug(f"Setting {rowFor} to {repr(value)}")
        self._parts[rowFor] = value
        try:
            # touch the state property as this executes the sieving, to make
            # sure we don't fall through the sieve at any point in time
            _ = self.state
        except RuntimeError as e:
            # improve error reporting, but always reraise this, as this is a
            # full-blown failure
            raise RuntimeError(f'Failed to apply {value} to {axis}{rowType} with state {self._parts}') from e

    def _getRowPayload(self, row, rowType, rowFor):
        """Get the correct column value from the row.

        Given the row, and which component it relates to, get the
        relevant value, as its enum or a bool, depending on the component.
        """
        match rowType:
            case 'MotionState':
                value = row[f'state_{rowFor}']
                return AxisMotionState(value)
            case 'SystemState':
                value = row[f'powerState_{rowFor}']
                return PowerState(value)
            case 'InPosition':
                value = row[f'inPosition_{rowFor}']
                return bool(value)
            case _:
                raise ValueError(f'Failed to get row payload with {rowType=} and {row=}')

    @property
    def _isValid(self):
        # TODO: probably need to init the inPositions to False? and then just
        # itertools.chain self.system and motion rather than checking all
        # parts? Really not sure about this though. If no days started during
        # TMA usage this would probably be easier - need to think and/or speak
        # to Russell & Tiago
        return not any([v == self._UNINITIALIZED_VALUE for v in self._parts.values()])

    # state inspection properties - a high level way of inspecting the state as
    # an API
    @property
    def isMoving(self):
        return self.state in [TMAState.TRACKING, TMAState.SLEWING]

    @property
    def isNotMoving(self):
        return not self.isMoving

    @property
    def isTracking(self):
        return self.state == TMAState.TRACKING

    @property
    def isSlewing(self):
        return self.state == TMAState.SLEWING

    @property
    def canMove(self):
        badStates = [PowerState.OFF, PowerState.TURNING_OFF, PowerState.FAULT, PowerState.UNKNOWN]
        return bool(
            self._isValid and
            self._parts['azimuthSystemState'] not in badStates and
            self._parts['elevationSystemState'] not in badStates
        )

    # axis inspection properties, for internal use
    @property
    def _axesInFault(self):
        """Returns a list of states for the axes, in order to call any() and
        all()"""
        return [x in self.FAULT_LIKE for x in self.system]

    @property
    def _axesOff(self):
        """Returns a list of states for the axes, in order to call any() and
        all()"""
        return [x in self.OFF_LIKE for x in self.system]

    @property
    def _axesOn(self):
        """Returns a list of states for the axes, in order to call any() and
        all()"""
        return [not x for x in self._axesOn]

    @property
    def _axesInMotion(self):
        """Returns a list of states for the axes, in order to call any() and
        all()"""
        return [x in self.MOVING_LIKE for x in self.motion]

    @property
    def _axesTRACKING(self):
        """Returns a list of states for the axes, in order to call any() and
        all()

        Note this is _axesTRACKING not _axesTracking to make sure it's clear
        that this is the AxisMotionState type of TRACKING and not the normal
        conceptual notion of tracking (the sky, i.e. as opposed to slewing)
        """
        return [x == AxisMotionState.TRACKING for x in self.motion]

    @property
    def _axesInPosition(self):
        """Returns a list of states for the axes, in order to call any() and
        all()"""
        return [x is True for x in self.inPosition]

    @property
    def state(self):
        # first, check we're valid, and if not, return UNINITIALIZED state, as
        # things are unknown
        if not self._isValid:
            return TMAState.UNINITIALIZED

        # if we're not in engineering mode, i.e. we're under normal CSC
        # control, then if anything is in fault, we're in fault. If we're
        # engineering then some axes will move when others are in fault
        if not self.engineeringMode:
            if any(self._axesInFault):
                return TMAState.FAULT
        else:
            # we're in engineering mode, so return fault state if ALL are in
            # fault
            if all(self._axesInFault):
                return TMAState.FAULT

        # if all axes are off, the TMA is OFF
        if all(self._axesOff):
            return TMAState.OFF

        # we know we're valid and at least some axes are not off, so see if
        # we're in motion if no axes are moving, we're stopped
        if not any(self._axesInMotion):
            return TMAState.STOPPED

        # now we know we're initialized, and that at least one axis is moving
        # so check axes for motion and in position. If all axes are tracking
        # and all are in position, we're tracking the sky
        if (all(self._axesTRACKING) and all(self._axesInPosition)):
            return TMAState.TRACKING

        # we now know explicitly that not everything is in position, so we no
        # longer need to check that. We do actually know that something is in
        # motion, but confirm that's the case and return SLEWING
        if (any(self._axesInMotion)):
            return TMAState.SLEWING

        # if we want to differentiate between MOVING_POINT_TO_POINT moves,
        # JOGGING moves and regular slews, the logic in the step above needs to
        # be changed and the new steps added here.

        raise RuntimeError('State error: fell through the state sieve - rewrite your logic!')


class TMAEventMaker:
    # TODO: turn this into an ABC with a TMA mount instance of it.

    # the topics which need logical combination to determine the overall mount
    # state. Will need updating as new components are added to the system.
    # relevant column: 'state'
    _movingComponents = [
        'lsst.sal.MTMount.logevent_azimuthMotionState',
        'lsst.sal.MTMount.logevent_elevationMotionState',
    ]
    # relevant column: 'inPosition'
    _inPositionComponents = [
        'lsst.sal.MTMount.logevent_azimuthInPosition',
        'lsst.sal.MTMount.logevent_elevationInPosition',
    ]
    # the components which, if in fault, put the TMA into fault
    # relevant column: 'powerState'
    _stateComponents = [
        'lsst.sal.MTMount.logevent_azimuthSystemState',
        'lsst.sal.MTMount.logevent_elevationSystemState',
    ]

    def __init__(self, client=None):
        if client is not None:
            self.client = client
        else:
            self.client = makeEfdClient()
        self.log = logging.getLogger(__name__)
        self._data = {}

    @staticmethod
    def isToday(dayObs):
        todayDayObs = getCurrentDayObs_int()
        if dayObs == todayDayObs:
            return True
        if dayObs > todayDayObs:
            raise ValueError("dayObs is in the future")
        return False

    @staticmethod
    def _shortName(topic):
        # get, for example 'azimuthInPosition' from
        # lsst.sal.MTMount.logevent_azimuthInPosition
        return topic.split('_')[-1]

    def _mergeData(self, data):
        """Merge a set of dataframes.

        data is a dict of dataframes, keyed by topic.
        """
        excludeColumns = ['private_sndStamp', 'rowFor']

        mergeArgs = {
            # 'on': 'private_sndStamp',
            'how': 'outer',
            'sort': True,
        }

        merged = None
        originalRowCounter = 0

        # Iterate over the keys and merge the corresponding DataFrames
        for key, df in data.items():
            originalRowCounter += len(df)
            component = self._shortName(key)  # Add suffix to column names to identify the source
            suffix = '_' + component

            df['rowFor'] = component

            columnsToSuffix = [col for col in df.columns if col not in excludeColumns]
            df_to_suffix = df[columnsToSuffix].add_suffix(suffix)
            df = pd.concat([df[excludeColumns], df_to_suffix], axis=1)

            if merged is None:
                merged = df.copy()
            else:
                merged = pd.merge(merged, df, **mergeArgs)

        merged = merged.loc[:, ~merged.columns.duplicated()]  # Remove duplicate columns after merge

        if len(merged) != originalRowCounter:
            self.log.warning("Merged data has a different number of rows to the original data, some"
                             " timestamps (rows) will contain more than one piece of actual information.")
        return merged

    def getEvents(self, dayObs):
        """Get the TMA seqNums for the specified dayObs.

        Gets the data from the cache or EFD as required, handling live data.
        Merges the EFD data streams, generates TmaEvents for the day's data.

        Parameters
        ----------
        dayObs : `int`
            The dayObs for which to get the events.

        Returns
        -------
        events : `list` of `lsst.summit.utils.tmaUtils.TMAState`
            The events for the specified dayObs.
        """
        workingLive = self.isToday(dayObs)
        data = None

        if dayObs in self._data and not workingLive:
            # data is in the cache and it's not being updated, so use it
            data = self._data[dayObs]
        elif dayObs not in self._data and not workingLive:
            # we don't have the data yet, but it's not growing, so put it in
            # the cache and use it from there
            self.log.info(f'Retrieving mount data for {dayObs} from the EFD')
            self._getEfdDataForDayObs(dayObs)
            data = self._data[dayObs]
        elif workingLive:
            # it's potentially updating data, so we must update the regarless
            # of whether we have it already or not
            self.log.info(f'Updating mount data for {dayObs} from the EFD')
            self._getEfdDataForDayObs(dayObs)
            data = self._data[dayObs]
        else:
            raise RuntimeError("This should never happen")

        if self._noDataFound(data):
            self.log.warning(f"No EFD data found for {dayObs=}")
            return []

        events = self._calculateEventsFromMergedData(data, dayObs)
        if not events:
            self.log.warning(f"Failed to calculate any events for {dayObs=} despite EFD data existing!")
        return events

    @staticmethod
    def _noDataFound(data):
        # can't just compare to with data == NO_DATA_SENTINEL because data
        # is usually a dataframe, and you can't compare a dataframe to a string
        # directly.
        return isinstance(data, str) and data == NO_DATA_SENTINEL

    def _getEfdDataForDayObs(self, dayObs):
        """Get the EFD data for the specified dayObs and store it in the cache.

        Gets the EFD data for all components, and stores it as a dict, keyed by
        the component topic name. If no data is found, the value is set to
        ``NO_DATA_SENTINEL`` to differentiate this from ``None``, as this is
        what you'd get if you queried the cache with `self._data.get(dayObs)`.
        It also marks that we have already queried this day.

        Parameters
        ----------
        dayObs : `int`
            The dayObs to query.

        Raises
        ------
        ValueError
            If the dayObs is in the future.
        """
        data = {}
        for component in itertools.chain(
            self._movingComponents,
            self._inPositionComponents,
            self._stateComponents
        ):
            data[component] = getEfdData(self.client, component, dayObs=dayObs)
            self.log.debug(f"Found {len(data[component])} for {component}")

        if all([dataframe.empty for dataframe in data.values()]):
            # if every single dataframe is empty, set the sentinel and don't
            # try to merge anything, otherwise merge all the data we found
            self.log.debug(f"No data found for {dayObs=}")
            # a sentinel value that's not None
            self._data[dayObs] = NO_DATA_SENTINEL
        else:
            merged = self._mergeData(data)
            self._data[dayObs] = merged

    def _calculateEventsFromMergedData(self, data, dayObs):
        """

        Runs the data, row by row, through the TMA state machine to get the
        state at each row, building a dict of these states. Then loops over the
        dict of states, building a list of events by chunking the states into
        blocks of the same state, and creating an event for each block.
        Off-type states are skipped over, with each event starting when the
        telescope next resumes motion.
        """
        engineering = True
        tma = TMA(engineeringMode=engineering)

        # For now, we assume that the TMA starts each day able to move, but
        # stationary. If this turns out to cause problems, we will need to
        # change to loading data from the previous day(s), and looking back
        # through it in time until a state change has been found for every
        # axis. For now though, Bruno et. al think this is acceptable and
        # preferable.
        _initializeTma(tma)

        tmaStates = {}
        for rowNum, row in data.iterrows():
            tma.apply(row)
            tmaStates[rowNum] = tma.state

        stateTuples = self._statesToEventTuples(tmaStates)
        events = self._makeEventsFromStateTuples(stateTuples, dayObs, data)

        return events

    def _statesToEventTuples(self, states):
        """
        """
        # Consider rewriting this with states as a list and using pop(0)?
        skipStates = (TMAState.STOPPED, TMAState.OFF, TMAState.FAULT)

        parsed_states = []
        eventStart = None
        rowNum = 0
        nRows = len(states)
        while rowNum < nRows:
            previousState = None
            state = states[rowNum]
            # if we're not in an event, fast forward through off-like rows
            # until a new event starts
            if eventStart is None and state in skipStates:
                rowNum += 1
                continue

            # we've started a new event, so walk through it and find the end
            eventStart = rowNum
            previousState = state
            rowNum += 1  # move to the next row before starting the while loop
            if rowNum == nRows:
                # we've reached the end of the data, and we're still in an
                # event, so don't return this presumably in-progress event
                self.log.warning('Reached the end of the data while still in an event')
                break
            state = states[rowNum]
            while state == previousState:
                rowNum += 1
                if rowNum == nRows:
                    break
                state = states[rowNum]
            parsed_states.append((eventStart, rowNum, previousState, state))
            if state in skipStates:
                eventStart = None

        return parsed_states

    def _makeEventsFromStateTuples(self, states, dayObs, data):
        """
        """
        seqNum = 0
        events = []
        for row in states:
            eventStart, eventEnd, eventType, endReason = row

            begin = data.iloc[eventStart]['private_sndStamp']
            end = data.iloc[eventEnd]['private_sndStamp']
            beginAstropy = efdTimestampToAstropy(begin)
            endAstropy = efdTimestampToAstropy(end)
            duration = end - begin
            event = TMAEvent(
                dayObs=dayObs,
                seqNum=seqNum,
                type=eventType,
                endReason=endReason,
                duration=duration,
                begin=beginAstropy,
                end=endAstropy,
                beginFloat=begin,
                endFloat=end,
            )
            events.append(event)
            seqNum += 1
        return events
