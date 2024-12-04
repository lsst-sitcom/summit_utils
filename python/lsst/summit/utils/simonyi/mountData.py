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

from dataclasses import dataclass
from typing import TYPE_CHECKING
import numpy as np
from scipy.optimize import minimize
from ..efdUtils import getEfdData

if TYPE_CHECKING:
    from astropy.time import Time
    from lsst_efd_client import EfdClient
    from pandas import DataFrame

    from lsst.daf.butler import DimensionRecord


@dataclass
class MountData:
    begin: Time
    end: Time
    azimuthData: DataFrame
    elevationData: DataFrame
    rotationData: DataFrame
    rotationTorques: DataFrame
    includedPrePadding: float
    includedPostPadding: float
    expRecord: DimensionRecord | None


def getAzElRotDataForPeriod(
        client: EfdClient, begin: Time, end: Time, prePadding: float = 0, postPadding: float = 0,
        maxDeltaT: float = 1.0e-3,
) -> MountData:
    def calcDeltaT(params, args):
        # This calculates the deltaT needed
        # to make the median(error) = 0
        [values, valTimes, demand, demTimes] = args
        [deltaT] = params
        demandInterp = np.interp(valTimes, demTimes + deltaT, demand)
        error = (values - demandInterp) * 3600
        value = abs(np.median(error))
        return value

    azimuthData = getEfdData(
        client,
        "lsst.sal.MTMount.azimuth",
        begin=begin,
        end=end,
        prePadding=prePadding,
        postPadding=postPadding,
    )
    elevationData = getEfdData(
        client,
        "lsst.sal.MTMount.elevation",
        begin=begin,
        end=end,
        prePadding=prePadding,
        postPadding=postPadding,
    )
    rotationData = getEfdData(
        client,
        "lsst.sal.MTRotator.rotation",
        begin=begin,
        end=end,
        prePadding=prePadding,
        postPadding=postPadding,
    )
    rotationTorques = getEfdData(
        client,
        "lsst.sal.MTRotator.motors",
        begin=begin,
        end=end,
        prePadding=prePadding,
        postPadding=postPadding,
    )

    azValues = azimuthData["actualPosition"].to_numpy()
    azValTimes = azimuthData["actualPositionTimestamp"].to_numpy()
    azDemand = azimuthData["demandPosition"].to_numpy()    
    azDemTimes = azimuthData["demandPositionTimestamp"].to_numpy()
    elValues = elevationData["actualPosition"].to_numpy()
    elValTimes = elevationData["actualPositionTimestamp"].to_numpy()
    elDemand = elevationData["demandPosition"].to_numpy()    
    elDemTimes = elevationData["demandPositionTimestamp"].to_numpy()
    rotValues = rotationData["actualPosition"].to_numpy()    
    rotDemand = rotationData["demandPosition"].to_numpy()

    # Calculate the deltaT needed to drive the median(error) to zero
    args = [azValues, azValTimes, azDemand, azDemTimes]
    x0 = [0.0]
    result = minimize(calcDeltaT, x0, args=args, method="Powell", bounds=[(-maxDeltaT, maxDeltaT)])
    deltaTAz = result.x[0]

    args = [elValues, elValTimes, elDemand, elDemTimes]
    x0 = [0.0]
    result = minimize(calcDeltaT, x0, args=args, method="Powell", bounds=[(-maxDeltaT, maxDeltaT)])
    deltaTEl = result.x[0]

    azDemandInterp = np.interp(azValTimes, azDemTimes + deltaTAz, azDemand)
    elDemandInterp = np.interp(elValTimes, elDemTimes + deltaTEl, elDemand)

    azError = (azValues - azDemandInterp) * 3600
    elError = (elValues - elDemandInterp) * 3600
    rotError = (rotValues - rotDemand) * 3600    

    azimuthData["azError"] = azError
    elevationData["elError"] = elError
    rotationData["rotError"] = rotError

    mountData = MountData(
        begin, end, azimuthData, elevationData, rotationData, rotationTorques, prePadding, postPadding, None
    )
    return mountData


def getAzElRotDataForExposure(
    client: EfdClient, expRecord: DimensionRecord, prePadding: float = 0, postPadding: float = 0
) -> MountData:

    begin = expRecord.timespan.begin
    end = expRecord.timespan.end
    mountData = getAzElRotDataForPeriod(client, begin, end, prePadding, postPadding)
    mountData.expRecord = expRecord
    return mountData
