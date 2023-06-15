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
import datetime
import logging
import pandas as pd
from dataclasses import dataclass
from astropy.time import Time
from lsst.summit.utils.utils import getCurrentDayObs_int  # XXX change back to relative import
from lsst.summit.utils.efdUtils import (getEfdData,  # XXX change back to relative import
                                        makeEfdClient,
                                        )

__all__ = (
    'TMA',
    'TMAEvent',
    'TMAEventMaker',
    'TMAState',
    'AxisMotionState',
    'PowerState',
)

# we don't want to use `None` for a no data sentinel because dict.get('key')
# returns None if the key isn't present, and also we need to mark that the data
# was queried for and no data was found, whereas the key not being present
# means that we've not yet looked for the data.
NO_DATA_SENTINEL = "NODATA"


def _makeValid(tma):
    """Helper function to turn a TMA into a valid state for testing.

    Do not call directly in normal usage or code, as this just arbitrarily
    sets values to make the TMA valid.
    """
    tma._parts['azimuthInPosition'] = False
    tma._parts['azimuthMotionState'] = AxisMotionState.STOPPED
    tma._parts['azimuthSystemState'] = PowerState.OFF
    tma._parts['elevationInPosition'] = False
    tma._parts['elevationMotionState'] = AxisMotionState.STOPPED
    tma._parts['elevationSystemState'] = PowerState.OFF


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
    seqType: str  # can be 'slew', 'track', 'pointToPoint'
    endReason: str  # can be 'slew', 'fault', 'stop', 'unfinished' - rare!
    duration: float  # seconds
    beginTime: Time
    endTime: Time
    beginTimeFloat: float
    endTimeFloat: float

    def __lt__(self, other):
        if self.dayObs < other.dayObs:
            return True
        elif self.dayObs == other.dayObs:
            return self.seqNum < other.seqNum
        return False


class TMAState(enum.IntEnum):
    """Overall state of the TMA"""
    UNINITIALIZED = -1
    STOPPED = 0
    MOVING_POINT_TO_POINT = 1
    TRACKING = 2
    SLEWING = 3
    FAULT = 4
    OFF = 5


class AxisMotionState(enum.IntEnum):
    """Motion state of azimuth elevation and camera cable wrap.

    Note: this is copied over from
    https://github.com/lsst-ts/ts_idl/blob/develop/python/lsst/ts/idl/enums/MTMount.py  # noqa: E505
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
    https://github.com/lsst-ts/ts_idl/blob/develop/python/lsst/ts/idl/enums/MTMount.py  # noqa: E505
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
    _UNINITIALIZED_VALUE: int = -999

    def __init__(self):
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
        self.system = ReferenceList(self._parts, systemKeys)
        self.motion = ReferenceList(self._parts, motionKeys)
        self.inPosition = ReferenceList(self._parts, positionKeys)

        # tuples of states for state collapsing
        self.STOP_LIKE = (AxisMotionState.STOPPING, AxisMotionState.STOPPED)
        self.MOVING_LIKE = (AxisMotionState.MOVING_POINT_TO_POINT, AxisMotionState.JOGGING, AxisMotionState.TRACKING)
        self.OFF_LIKE = (PowerState.OFF, PowerState.TURNING_OFF)
        self.ON_LIKE = (PowerState.ON, PowerState.TURNING_ON)
        self.FAULT_LIKE = (PowerState.FAULT)

    @property
    def _isValid(self):
        # TODO: probably need to init the inPositions to False? and then just
        # itertools.chain self.system and motion rather than checking all
        # parts? Really not sure about this though. If no days started during
        # TMA usage this would probably be easier - need to think and/or speak
        # to Russell & Tiago
        return not any([v == self._UNINITIALIZED_VALUE for v in self._parts.values()])

    @property
    def isMoving(self):
        return self.state in [TMAState.MOVING_POINT_TO_POINT, TMAState.TRACKING, TMAState.SLEWING]

    @property
    def isNotMoving(self):
        return not self.isMoving

    @property
    def canMove(self):
        return self.state not in [TMAState.FAULT, TMAState.OFF]

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

    def apply(self, row):
        rowFor = row['rowFor']  # e.g. elevationMotionState
        axis, rowType = getAxisAndType(rowFor)  # e.g. elevation, MotionState
        self._parts[rowFor] = self._getRowPayload(row, rowType, rowFor)

    def _getRowPayload(self, row, rowType, rowFor):
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
    def state(self):
        # if anything is in fault, we're in fault, even if not initialized
        if any([x == PowerState.FAULT for x in self.system]):
            return TMAState.FAULT

        # next, check we're valid, and if not, return UNINITIALIZED state
        if not self._isValid:
            return TMAState.UNINITIALIZED

        # if any axis is off, the TMA is OFF
        if (any([x in (PowerState.OFF, PowerState.TURNING_OFF) for x in self.system])):
            return TMAState.OFF

        # we know we're valid and all axes are not off, so see if we're in motion
        # if all axes are stopped, we're stopped
        if (all([x in self.STOP_LIKE for x in self.motion])):
            return TMAState.STOPPED

        # now we know we're initialized, check axes for motion and in position
        # if all axes are tracking and all are in position, we're tracking the
        # sky
        if (all([x == AxisMotionState.TRACKING for x in self.motion]) and
            all([x is True for x in self.inPosition])):
            return TMAState.TRACKING

        # all axes in motion, but not all in position means slewing to target
        if (all([x == AxisMotionState.TRACKING for x in self.motion]) and
            not all([x is True for x in self.inPosition])):
            return TMAState.SLEWING

        # any axis moving point to point is moving point to point
        if (any([x == AxisMotionState.MOVING_POINT_TO_POINT for x in self.motion])):
            return TMAState.MOVING_POINT_TO_POINT

        # TODO: Add state reduction here for motions

        raise RuntimeError('State error: fell through the state seive - rewrite your logic!')


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

    # time keys in common between all:
    # 'private_efdStamp'
    # 'private_identity'
    # 'private_kafkaStamp'
    # 'private_origin'
    # 'private_rcvStamp'
    # 'private_revCode'
    # 'private_seqNum'
    # 'private_sndStamp'

    # XXX check this is compatible with astropy times
    TIME_CHUNKING = datetime.timedelta(minutes=15)

    def __init__(self, client=None):
        if client is not None:
            self.client = client
        else:
            self.client = makeEfdClient()
        self.log = logging.getLogger(__name__)
        self._data = {}

    def _combineFaultTypeStates(self, stateDict):
        """Combine the component states

        Parameters
        ----------
        stateDict : `dict` of `???`
            The state of each component.

        Returns
        -------
        ??? : ???
        """
        # PowerState:
        #     Off=0
        #     On=1
        #     Fault=2
        #     TurningOn=3
        #     TurningOff=4
        #     Unknown=15

        # if any in fault, return fault
        # if any off, return off
        # if all on, return on
        # raise RuntimeError for all other cases, stating the state of all
        # components

        # return the logical combination of all the states
        raise NotImplementedError("This function is not yet implemented")

    def _combineAxisStates(self, stateDict):
        """Combine the axis states

        Parameters
        ----------
        stateDict : `dict` of `???`
            The state of each component.

        Returns
        -------
        ??? : ???
        """
        # AxisMotionState:
        #     Stopping=0  # can be collapsed with stopping
        #     Stopped=1

        # unusual, used for balancing or parking etc, or hoisting etc
        #     MovingPointToPoint=2
        #     Jogging=3  # INGORE - we never do that
        #     Tracking=4  # most of the time but IT INCLUDES SLEWING
        #     TrackingPaused=5  # INGORE - Russell says "it's bizarre"
        # Jogging - can do on the hand paddle and the EUI (engineering user
        # interface) but the CSC doesn't support that

        # InPosition is False a lot of the time,

        # if AxisMotion is tracking then inPosition=True means you are tracking
        # and False means you're slewing
        # and it can go back and forth between the two!

        # if any MovingPointToPoint or jogging return slewing
        # if any stopped or stopping return stopped
        # if all tracking return tracking
        # raise RuntimeError for all other cases, state the state of all
        # components

        # return the logical combination of all the states
        raise NotImplementedError("This function is not yet implemented")

    def getState(self, time, detailed=False):
        """Get the mount state at the time specified.

        Parameters
        ----------
        time : `astropy.time.Time`
            The time at which to get the TMA state.
        detailed : `bool`, optional
            If detailed, return the state of all components at the specified
            time. Usually used to see why the particular overall system state
            was the case.

        Returns
        -------
        state : `???`
            The state of the TMA at the specified time.
        breakdown : `tuple` of `???`
            The state of all the subcomponents at the specified time, if
            ``detailed``, else ``None``.
        """
        # do the state-type ones first, because if we're in fault and not
        # detailed then we don't need to know anything else
        stateComponents = {}
        for component in self._stateComponents:
            stateComponents[component] = self._getComponentState(component, time)
        systemState = self._combineFaultTypeStates(stateComponents)

        # the early exit clause
        if systemState == 'fault' and not detailed:
            return "fault", None

        axisStates = {}
        for component in self._movingComponents:
            axisStates[component] = self._getComponentState(component, time)
        movingState = self._combineAxisStates(axisStates)

        breakdown = None
        if detailed:
            breakdown = axisStates.update(stateComponents)

        if systemState == 'fault':
            return 'fault', breakdown
        if systemState == 'off':
            return 'off', breakdown
        if systemState == 'on':
            return movingState, breakdown

        # all eventualities should have been covered by this point, so if we've
        # fallen all the way through the seive then raise
        raise RuntimeError('Overall state could not be determined')

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
        events : `list` of `TmaEvent`
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
            self.log.warning(f"No data found for {dayObs=}")
            return []

        events = self._calculateEventsFromMergedData(data)
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
            self.log.debug(f"No data found for {dayObs=}")
            # a sentinel value that's not None
            data = NO_DATA_SENTINEL

        merged = self._mergeData(data)
        self._data[dayObs] = merged

    def _calculateEventsFromMergedData(self, data):
        seqNums = []

        return seqNums
