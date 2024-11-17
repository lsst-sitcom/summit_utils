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

from typing import TYPE_CHECKING

from ..efdUtils import getEfdData

if TYPE_CHECKING:
    from astropy.time import Time
    from lsst_efd_client import EfdClient
    from pandas import DataFrame

    from lsst.daf.butler import DimensionRecord


def getAzElRotDataForPeriod(
    client: EfdClient, begin: Time, end: Time, prePadding: float = 0, postPadding: float = 0
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
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

    azValues = azimuthData["actualPosition"].values
    elValues = elevationData["actualPosition"].values
    rotValues = rotationData["actualPosition"].values
    azDemand = azimuthData["demandPosition"].values
    elDemand = elevationData["demandPosition"].values
    rotDemand = rotationData["demandPosition"].values

    azError = (azValues - azDemand) * 3600
    elError = (elValues - elDemand) * 3600
    rotError = (rotValues - rotDemand) * 3600

    azimuthData["azError"] = azError
    elevationData["elError"] = elError
    rotationData["rotError"] = rotError

    return azimuthData, elevationData, rotationData, rotationTorques


def getAzElRotDataForExposure(
    client: EfdClient, expRecord: DimensionRecord, prePadding: float = 0, postPadding: float = 0
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    begin = expRecord.timespan.begin
    end = expRecord.timespan.end
    return getAzElRotDataForPeriod(client, begin, end, prePadding, postPadding)
