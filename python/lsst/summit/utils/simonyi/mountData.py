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
    client: EfdClient, begin: Time, end: Time, prePadding: float = 0, postPadding: float = 0
) -> MountData:
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

    azValues = np.asarray(azimuthData["actualPosition"])
    elValues = np.asarray(elevationData["actualPosition"])
    rotValues = np.asarray(rotationData["actualPosition"])
    azDemand = np.asarray(azimuthData["demandPosition"])
    elDemand = np.asarray(elevationData["demandPosition"])
    rotDemand = np.asarray(rotationData["demandPosition"])

    azError = (azValues - azDemand) * 3600
    elError = (elValues - elDemand) * 3600
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
