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

import enum

__all__ = [
    'ScriptState',
    'AxisMotionState',
    'PowerState',
]

# TODO: Remove this file once RFC-942 passes and we can import from the source.


class ScriptState(enum.IntEnum):
    """ScriptState constants.

    Note: this is copied over from
    https://github.com/lsst-ts/ts_idl/blob/develop/python/lsst/ts/idl/enums/Script.py  # noqa: W505
    to save having to depend on T&S code directly. These enums are extremely
    static, so this is a reasonable thing to do, and much easier than setting
    up a dependency on ts_idl.
    """

    UNKNOWN = 0
    UNCONFIGURED = 1
    CONFIGURED = 2
    RUNNING = 3
    PAUSED = 4
    ENDING = 5
    STOPPING = 6
    FAILING = 7
    DONE = 8
    STOPPED = 9
    FAILED = 10
    CONFIGURE_FAILED = 11


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
