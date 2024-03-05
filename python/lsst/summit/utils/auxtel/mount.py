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

import logging

import numpy as np

from ..efdUtils import getEfdData


def hasTimebaseErrors(expRecord, client, maxDiff=1.05):
    """Check if an exposure has cRIO timebase errors.

    Data in the lsst.sal.ATMCS.mount_AzEl_Encoders topic is a packed
    time-series, and if the amount of data that is packed in isn't packed for a
    period of almost exactly one second then the unpacking code will
    incorrectly assign the intra-second timestamps, causing misalignment, as
    this breaks the fundamental assumption required to have a packed
    timeseries.

    Parameters
    ----------
    expRecord : `lsst.daf.butler.dimensions.DimensionRecord`
        The exposure record to query.
    client : `lsst_efd_client.efd_helper.EfdClient`
        The EFD client to use.
    maxDiff : `float`, optional
        The maximum difference in cRIO timestamps to consider as a timebase
        error, in seconds. The correct spacing is 1s, so 1.05 denotes a 50ms
        difference when the jitter should be on the order of microseconds.

    Returns
    -------
    containsErrors : `bool`
        `True` if the exposure has timebase errors, `False` otherwise.
    """
    log = logging.getLogger(__name__)
    mountPosition = getEfdData(
        client, "lsst.sal.ATMCS.mount_AzEl_Encoders", expRecord=expRecord, warn=False
    )

    if mountPosition.empty:
        log.warning(
            f"No mount data was found for {expRecord.obs_id}, so there is technically no"
            " timebase error present"
        )
        return False

    cRIOtimestamps = mountPosition["cRIO_timestamp"]
    if len(cRIOtimestamps) == 1:
        log.warning(
            f"cRIO_timestamp data had length 1 for {expRecord.obs_id}, so timebase errors are"
            " impossible"
        )
        return False

    diff = np.diff(cRIOtimestamps.values)
    if np.min(diff) < 0:
        raise ValueError("cRIO timestamps are not monotonically increasing - time is running backwards!")

    return np.max(diff) > maxDiff
