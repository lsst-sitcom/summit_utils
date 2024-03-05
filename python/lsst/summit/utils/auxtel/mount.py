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

import numpy as np
import logging
from ..efdUtils import getEfdData


def hasTimebaseErrors(expRecord, client, maxDiff=1.05):
    """Check if an exposure has cRIO timebase errors.

    Parameters
    ----------
    expRecord : `lsst.daf.butler.dimensions.DimensionRecord`
        The exposure record to query.
    client : `lsst_efd_client.efd_helper.EfdClient`
        The EFD client to use.
    maxDiff : `float`, optional
        The maximum difference in cRIO timestamps to consider as a timebase
        error.

    Returns
    -------
    containsErrors : `bool`
        `True` if the exposure has timebase errors, `False` otherwise.
    """
    mountPosition = getEfdData(client, "lsst.sal.ATMCS.mount_AzEl_Encoders", expRecord=expRecord, warn=False)
    if mountPosition.empty:
        log = logging.getLogger(__name__)
        log.warning(f"No mount data was found for {expRecord.obs_id}, so there is technically no"
                    " timebase error present")
        return False
    cRIO_ts = mountPosition["cRIO_timestamp"]
    return np.max(np.diff(cRIO_ts.values)) > maxDiff
