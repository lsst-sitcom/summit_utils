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

import copy
import itertools
from collections.abc import Iterable, Mapping
from typing import Any

from deprecated.sphinx import deprecated

import lsst.daf.butler as dafButler
from lsst.summit.utils.utils import getSite

__all__ = [
    "makeDefaultLatissButler",
    "updateDataId",
    "sanitizeDayObs",
    "getMostRecentDayObs",
    "getSeqNumsForDayObs",
    "getMostRecentDataId",
    "getDatasetRefForDataId",
    "getDayObs",
    "getSeqNum",
    "getExpId",
    "datasetExists",
    "sortRecordsByDayObsThenSeqNum",
    "getDaysWithData",
    "getExpIdFromDayObsSeqNum",
    "updateDataIdOrDataCord",
    "fillDataId",
    "getExpRecordFromDataId",
    "getDayObsSeqNumFromExposureId",
    "removeDataProduct",
    "getLatissOnSkyDataIds",
    "getExpRecord",
]

_LATISS_DEFAULT_COLLECTIONS = ["LATISS/raw/all", "LATISS/calib", "LATISS/runs/quickLook"]

# RECENT_DAY must be in the past *and have data* (otherwise some tests are
# no-ops), to speed up queries by restricting them significantly,
# but data must definitely been taken since. Should
# also not be more than 2 months in the past due to 60 day lookback time on the
# summit. All this means it should be updated by an informed human.
RECENT_DAY = 20220503


def _configureForSite() -> None:
    try:
        site = getSite()
    except ValueError:
        # this method is run automatically on module import, so
        # don't fail for k8s where this cannot yet be determined
        print("WARNING: failed to automatically determine site")
        site = None
    if site == "tucson":
        global RECENT_DAY
        RECENT_DAY = 20211104  # TTS has limited data, so use this day


_configureForSite()


def getLatissDefaultCollections() -> list[str]:
    """Get the default set of LATISS collections, updated for the site at
    which the code is being run.

    Returns
    -------
    collections : `list` of `str`
        The default collections for the site.
    """
    collections = _LATISS_DEFAULT_COLLECTIONS
    try:
        site = getSite()
    except ValueError:
        site = ""

    if site == "tucson":
        collections.append("LATISS-test-data")
        return collections
    if site == "summit":
        collections.append("LATISS-test-data")
        return collections
    return collections


def _update_RECENT_DAY(day: int) -> None:
    """Update the value for RECENT_DAY once we have a value for free."""
    global RECENT_DAY
    RECENT_DAY = max(day - 1, RECENT_DAY)


def makeDefaultLatissButler(
    *,
    extraCollections: list[str] | None = None,
    writeable: bool = False,
    embargo: bool = False,
) -> dafButler.Butler:
    """Create a butler for LATISS using the default collections.

    Parameters
    ----------
    extraCollections : `list` of `str`
        Extra input collections to supply to the butler init.
    writable : `bool`, optional
        Whether to make a writable butler.
    embargo : `bool`, optional
        Use the embargo repo instead of the main one. Needed to access
        embargoed data.

    Returns
    -------
    butler : `lsst.daf.butler.Butler`
        The butler.
    """
    # TODO: Add logging to which collections are going in
    collections = getLatissDefaultCollections()
    if extraCollections:
        collections.extend(extraCollections)
    try:
        repoString = "LATISS" if not embargo else "/repo/embargo"
        butler = dafButler.Butler.from_config(
            repoString, collections=collections, writeable=writeable, instrument="LATISS"
        )
    except (FileNotFoundError, RuntimeError):
        # Depending on the value of DAF_BUTLER_REPOSITORY_INDEX and whether
        # it is present and blank, or just not set, both these exception
        # types can be raised, see tests/test_butlerUtils.py:ButlerInitTestCase
        # for details and tests which confirm these have not changed
        raise FileNotFoundError  # unify exception type
    return butler


@deprecated(
    reason="datasExists has been replaced by Butler.exists(). Will be removed after v26.0.",
    version="v26.0",
    category=FutureWarning,
)
def datasetExists(butler: dafButler.Butler, dataProduct: str, dataId: dict, **kwargs: Any) -> bool:
    """Collapse the tri-state behaviour of butler.datasetExists to a boolean.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        The butler
    dataProduct : `str`
        The type of data product to check for
    dataId : `dict`
        The dataId of the dataProduct to check for

    Returns
    -------
    exists : `bool`
        True if the dataProduct exists for the dataProduct and can be retreived
        else False.
    """
    return bool(butler.exists(dataProduct, dataId, **kwargs))


def updateDataId(dataId: dict | dafButler.DataCoordinate, **kwargs: Any) -> dict | dafButler.DataCoordinate:
    """Update a DataCoordinate or dataId dict with kwargs.

    Provides a single interface for adding the detector key (or others) to a
    dataId whether it's a DataCoordinate or a dict

    Parameters
    ----------
    dataId : `dict` or `lsst.daf.butler.DataCoordinate`
        The dataId to update.
    kwargs : `dict`
        The keys and values to add to the dataId.

    Returns
    -------
    dataId : `dict` or `lsst.daf.butler.DataCoordinate`
        The updated dataId, with the same type as the input.
    """

    match dataId:
        case dafButler.DataCoordinate():
            return dafButler.DataCoordinate.standardize(dataId, **kwargs)
        case dict() as dataId:
            return dict(dataId, **kwargs)
    raise ValueError(f"Unknown dataId type {type(dataId)}")


def sanitizeDayObs(day_obs: int | str) -> int:
    """Take string or int day_obs and turn it into the int version.

    Parameters
    ----------
    day_obs : `str` or `int`
        The day_obs to sanitize.

    Returns
    -------
    day_obs : `int`
        The sanitized day_obs.

    Raises
    ------
    ValueError
        Raised if the day_obs fails to translate for any reason.
    """
    if isinstance(day_obs, int):
        return day_obs
    elif isinstance(day_obs, str):
        try:
            return int(day_obs.replace("-", ""))
        except Exception:
            ValueError(f"Failed to sanitize {day_obs!r} to a day_obs")
    raise ValueError(f"Cannot sanitize {day_obs!r} to a day_obs")


def getMostRecentDayObs(butler: dafButler.Butler) -> int:
    """Get the most recent day_obs for which there is data.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler
        The butler to query.

    Returns
    -------
    day_obs : `int`
        The day_obs.
    """
    where = "exposure.day_obs>=RECENT_DAY"
    records = butler.registry.queryDimensionRecords(
        "exposure", where=where, datasets="raw", bind={"RECENT_DAY": RECENT_DAY}
    )
    recentDay = max(r.day_obs for r in records)
    _update_RECENT_DAY(recentDay)
    return recentDay


def getSeqNumsForDayObs(butler: dafButler.Butler, day_obs: int, extraWhere: str = "") -> list[int]:
    """Get a list of all seq_nums taken on a given day_obs.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler
        The butler to query.
    day_obs : `int` or `str`
        The day_obs for which the seq_nums are desired.
    extraWhere : `str`
        Any extra where conditions to add to the queryDimensionRecords call.

    Returns
    -------
    seq_nums : `iterable`
        The seq_nums taken on the corresponding day_obs in ascending numerical
        order.
    """
    day_obs = sanitizeDayObs(day_obs)
    where = "exposure.day_obs=dayObs"
    if extraWhere:
        extraWhere = extraWhere.replace('"', "'")
        where += f" and {extraWhere}"
    records = butler.registry.queryDimensionRecords(
        "exposure", where=where, bind={"dayObs": day_obs}, datasets="raw"
    )
    return sorted([r.seq_num for r in records])


def sortRecordsByDayObsThenSeqNum(
    records: dafButler.registry.queries.DimensionRecordQueryResults | list[dafButler.DimensionRecord],
) -> list[dafButler.DimensionRecord]:
    """Sort a set of records by dayObs, then seqNum to get the order in which
    they were taken.

    Parameters
    ----------
    records : `list` of `dafButler.DimensionRecord`
        The records to be sorted.

    Returns
    -------
    sortedRecords : `list` of `dafButler.DimensionRecord`
        The sorted records

    Raises
    ------
    ValueError
        Raised if the recordSet contains duplicate records, or if it contains
        (dayObs, seqNum) collisions.
    """
    records = list(records)  # must call list in case we have a generator
    recordSet = set(records)
    if len(records) != len(recordSet):
        raise ValueError("Record set contains duplicate records and therefore cannot be sorted unambiguously")

    daySeqTuples = [(r.day_obs, r.seq_num) for r in records]
    if len(daySeqTuples) != len(set(daySeqTuples)):
        raise ValueError(
            "Record set contains dayObs/seqNum collisions, and therefore cannot be sorted " "unambiguously"
        )

    records.sort(key=lambda r: (r.day_obs, r.seq_num))
    return records


def getDaysWithData(butler: dafButler.Butler, datasetType: str = "raw") -> list[int]:
    """Get all the days for which LATISS has taken data on the mountain.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler
        The butler to query.
    datasetType : `str`
        The datasetType to query.

    Returns
    -------
    days : `list` of `int`
        A sorted list of the day_obs values for which mountain-top data exists.
    """
    # 20200101 is a day between shipping LATISS and going on sky
    # We used to constrain on exposure.seq_num<50 to massively reduce the
    # number of returned records whilst being large enough to ensure that no
    # days are missed because early seq_nums were skipped. However, because
    # we have test datasets like LATISS-test-data-tts where we only kept
    # seqNums from 950 on one day, we can no longer assume this so don't be
    # tempted to add such a constraint back in here for speed.
    where = "exposure.day_obs>20200101"
    records = butler.registry.queryDimensionRecords("exposure", where=where, datasets=datasetType)
    return sorted(set([r.day_obs for r in records]))


def getMostRecentDataId(butler: dafButler.Butler) -> dict:
    """Get the dataId for the most recent observation.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler
        The butler to query.

    Returns
    -------
    dataId : `dict`
        The dataId of the most recent exposure.
    """
    lastDay = getMostRecentDayObs(butler)
    seqNum = getSeqNumsForDayObs(butler, lastDay)[-1]
    dataId = {"day_obs": lastDay, "seq_num": seqNum, "detector": 0}
    dataId.update(getExpIdFromDayObsSeqNum(butler, dataId))
    return dataId


def getExpIdFromDayObsSeqNum(butler: dafButler.Butler, dataId: dict) -> dict:
    """Get the exposure id for the dataId.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler
        The butler to query.
    dataId : `dict`
        The dataId for which to return the exposure id.

    Returns
    -------
    dataId : `dict`
        The dataId of the most recent exposure.
    """
    expRecord = getExpRecordFromDataId(butler, dataId)
    return {"exposure": expRecord.id}


def updateDataIdOrDataCord(dataId: dafButler.DataId, **updateKwargs: Any) -> Mapping[str, Any]:
    """Add key, value pairs to a dataId or data coordinate.

    Parameters
    ----------
    dataId : `dafButler.DataId`
        The dataId for which to return the exposure id.
    updateKwargs : `dict`
        The key value pairs add to the dataId or dataCoord.

    Returns
    -------
    dataId : `Mapping[str, Any]`
        The updated dataId.

    Notes
    -----
    Always returns a dict, so note that if a data coordinate is supplied, a
    dict is returned, changing the type.
    """
    newId = copy.copy(dataId)
    newId = _assureDict(newId)
    newId.update(updateKwargs)
    return newId


def fillDataId(butler: dafButler.direct_butler.DirectButler, dataId: dafButler.DataId) -> Mapping[str, Any]:
    """Given a dataId, fill it with values for all available dimensions.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        The butler.
    dataId : `dafButler.DataId`
        The dataId to fill.

    Returns
    -------
    dataId : `Mapping[str, Any]`
        The filled dataId.

    Notes
    -----
    This function is *slow*! Running this on 20,000 dataIds takes approximately
    7 minutes. Virtually all the slowdown is in the
    butler.registry.expandDataId() call though, so this wrapper is not to blame
    here, and might speed up in future with butler improvements.
    """
    # ensure it's a dict to deal with records etc
    dictDataId: Mapping[str, Any] = _assureDict(dataId)

    # this removes extraneous keys that would trip up the registry call
    # using _rewrite_data_id is perhaps ever so slightly slower than popping
    # the bad keys, or making a minimal dataId by hand, but is more
    # reliable/general, so we choose that over the other approach here
    filteredDataId, _ = butler._rewrite_data_id(dictDataId, butler.get_dataset_type("raw"))

    # now expand and turn back to a dict
    dataId = butler.registry.expandDataId(filteredDataId, detector=0).mapping  # this call is VERY slow
    dataId = _assureDict(dataId)

    missingExpId = getExpId(dataId) is None
    missingDayObs = getDayObs(dataId) is None
    missingSeqNum = getSeqNum(dataId) is None

    if missingDayObs or missingSeqNum:
        dayObsSeqNum = getDayObsSeqNumFromExposureId(butler, dataId)
        dataId.update(dayObsSeqNum)

    if missingExpId:
        expId = getExpIdFromDayObsSeqNum(butler, dataId)
        dataId.update(expId)

    return dataId


def _assureDict(
    dataId: Mapping | dafButler.dimensions.DataCoordinate | dafButler.DimensionRecord,
) -> dict[str, Any]:
    """Turn any data-identifier-like object into a dict.

    Parameters
    ----------
    dataId : `dict` or `lsst.daf.butler.dimensions.DataCoordinate` or
             `lsst.daf.butler.dimensions.DimensionRecord`
        The data identifier.

    Returns
    -------
    dataId : `dict`
        The data identifier as a dict.
    """
    if isinstance(dataId, dict):
        return dataId
    elif hasattr(dataId, "mapping"):  # dafButler.dimensions.DataCoordinate
        return {str(k): v for k, v in dataId.mapping.items()}
    elif hasattr(dataId, "items"):  # dafButler.dimensions.DataCoordinate
        return {str(k): v for k, v in dataId.items()}  # str() required due to full names
    elif hasattr(dataId, "dataId"):  # dafButler.DimensionRecord
        return {str(k): v for k, v in dataId.dataId.mapping.items()}
    else:
        raise RuntimeError(f"Failed to coerce {type(dataId)} to dict")


def getExpRecordFromDataId(butler: dafButler.Butler, dataId: dict) -> dafButler.DimensionRecord:
    """Get the exposure record for a given dataId.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        The butler.
    dataId : `dict`
        The dataId.

    Returns
    -------
    expRecord : `lsst.daf.butler.dimensions.DimensionRecord`
        The exposure record.
    """
    dataId = _assureDict(dataId)
    assert isinstance(dataId, dict), f"dataId must be a dict or DimensionRecord, got {type(dataId)}"

    if expId := getExpId(dataId):
        where = "exposure.id=expId"
        expRecords = butler.registry.queryDimensionRecords(
            "exposure", where=where, bind={"expId": expId}, datasets="raw"
        )

    else:
        dayObs = getDayObs(dataId)
        seqNum = getSeqNum(dataId)
        if not (dayObs and seqNum):
            raise RuntimeError(f"Failed to find either expId or day_obs and seq_num in dataId {dataId}")
        where = "exposure.day_obs=dayObs AND exposure.seq_num=seq_num"
        expRecords = butler.registry.queryDimensionRecords(
            "exposure", where=where, bind={"dayObs": dayObs, "seq_num": seqNum}, datasets="raw"
        )

    filteredExpRecords = set(expRecords)
    if not filteredExpRecords:
        raise LookupError(f"No exposure records found for {dataId}")
    assert len(filteredExpRecords) == 1, f"Found {len(filteredExpRecords)} exposure records for {dataId}"
    return filteredExpRecords.pop()


def getDayObsSeqNumFromExposureId(butler: dafButler.Butler, dataId: Mapping[str, Any]) -> Mapping[str, int]:
    """Get the day_obs and seq_num for an exposure id.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        The butler.
    dataId : `dict`
        The dataId containing the exposure id.

    Returns
    -------
    dataId : `dict`
        A dict containing only the day_obs and seq_num.
    """
    if (dayObs := getDayObs(dataId)) and (seqNum := getSeqNum(dataId)):
        return {"day_obs": dayObs, "seq_num": seqNum}

    if isinstance(dataId, int):
        dataId = {"exposure": dataId}
    else:
        dataId = _assureDict(dataId)
    assert isinstance(dataId, dict)

    if not (expId := getExpId(dataId)):
        raise RuntimeError(f"Failed to find exposure id in {dataId}")

    where = "exposure.id=expId"
    expRecords = butler.registry.queryDimensionRecords(
        "exposure", where=where, bind={"expId": expId}, datasets="raw"
    )
    filteredExpRecords = set(expRecords)
    if not filteredExpRecords:
        raise LookupError(f"No exposure records found for {dataId}")
    assert len(filteredExpRecords) == 1, f"Found {len(filteredExpRecords)} exposure records for {dataId}"
    record = filteredExpRecords.pop()
    return {"day_obs": record.day_obs, "seq_num": record.seq_num}


def getDatasetRefForDataId(
    butler: dafButler.Butler, datasetType: str | dafButler.DatasetType, dataId: dict
) -> dafButler.DatasetRef | None:
    """Get the datasetReference for a dataId.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        The butler.
    datasetType : `str` or `datasetType`
        The dataset type.
    dataId : `dict`
        The dataId.

    Returns
    -------
    datasetRef : `lsst.daf.butler.dimensions.DatasetReference`
        The dataset reference.
    """
    if not _expid_present(dataId):
        assert _dayobs_present(dataId) and _seqnum_present(dataId)
        dataId.update(getExpIdFromDayObsSeqNum(butler, dataId))

    dRef = butler.find_dataset(datasetType, dataId)
    return dRef


def removeDataProduct(
    butler: dafButler.Butler, datasetType: str | dafButler.DatasetType, dataId: dict
) -> None:
    """Remove a data prodcut from the registry. Use with caution.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        The butler.
    datasetType : `str` or `datasetType`
        The dataset type.
    dataId : `dict`
        The dataId.

    """
    if datasetType == "raw":
        raise RuntimeError("I'm sorry, Dave, I'm afraid I can't do that.")
    dRef = getDatasetRefForDataId(butler, datasetType, dataId)
    if dRef is None:
        return
    butler.pruneDatasets([dRef], disassociate=True, unstore=True, purge=True)
    return


def _dayobs_present(dataId: Mapping[str, Any]) -> bool:
    return _get_dayobs_key(dataId) is not None


def _seqnum_present(dataId: Mapping[str, Any]) -> bool:
    return _get_seqnum_key(dataId) is not None


def _expid_present(dataId: Mapping[str, Any]) -> bool:
    return _get_expid_key(dataId) is not None


def _get_dayobs_key(dataId: Mapping[str, Any]) -> str | None:
    """Return the key for day_obs if present, else None"""
    keys = [k for k in dataId.keys() if k.find("day_obs") != -1]
    if not keys:
        return None
    return keys[0]


def _get_seqnum_key(dataId: Mapping[str, Any]) -> str | None:
    """Return the key for seq_num if present, else None"""
    keys = [k for k in dataId.keys() if k.find("seq_num") != -1]
    if not keys:
        return None
    return keys[0]


def _get_expid_key(dataId: Mapping[str, Any]) -> str | None:
    """Return the key for expId if present, else None"""
    if "exposure.id" in dataId:
        return "exposure.id"
    elif "exposure" in dataId:
        return "exposure"
    return None


def getDayObs(dataId: Mapping[str, Any] | dafButler.DimensionRecord) -> int | None:
    """Get the day_obs from a dataId.

    Parameters
    ----------
    dataId : `dict` or `lsst.daf.butler.DimensionRecord`
        The dataId.

    Returns
    -------
    day_obs : `int` or `None`
        The day_obs value if present, else None.
    """
    if hasattr(dataId, "day_obs"):
        return getattr(dataId, "day_obs")
    if not _dayobs_present(dataId):
        return None
    return dataId["day_obs"] if "day_obs" in dataId else dataId["exposure.day_obs"]


def getSeqNum(dataId: Mapping[str, Any] | dafButler.DimensionRecord) -> int | None:
    """Get the seq_num from a dataId.

    Parameters
    ----------
    dataId : `dict` or `lsst.daf.butler.DimensionRecord`
        The dataId.

    Returns
    -------
    seq_num : `int` or `None`
        The seq_num value if present, else None.
    """
    if hasattr(dataId, "seq_num"):
        return getattr(dataId, "seq_num")
    if not _seqnum_present(dataId):
        return None
    return dataId["seq_num"] if "seq_num" in dataId else dataId["exposure.seq_num"]


def getExpId(dataId: Mapping[str, Any] | dafButler.DimensionRecord) -> int | None:
    """Get the expId from a dataId.

    Parameters
    ----------
    dataId : `dict` or `lsst.daf.butler.DimensionRecord`
        The dataId.

    Returns
    -------
    expId : `int` or `None`
        The expId value if present, else None.
    """
    if hasattr(dataId, "id"):
        return getattr(dataId, "id")
    if not _expid_present(dataId):
        return None
    return dataId["exposure"] if "exposure" in dataId else dataId["exposure.id"]


def getLatissOnSkyDataIds(
    butler: dafButler.direct_butler.DirectButler,
    skipTypes: Iterable[str] = ("bias", "dark", "flat"),
    checkObject: bool = True,
    full: bool = True,
    startDate: int | None = None,
    endDate: int | None = None,
) -> list[Mapping[str, int | str | None]]:
    """Get a list of all on-sky dataIds taken.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        The butler.
    skipTypes : `list` of `str`
        Image types to exclude.
    checkObject : `bool`
        Check if the value of target_name (formerly OBJECT) is set and exlude
        if it is not.
    full : `bool`
        Return filled dataIds. Required for some analyses, but runs much
        (~30x) slower.
    startDate : `int`
        The day_obs to start at, inclusive.
    endDate : `int`
        The day_obs to end at, inclusive.

    Returns
    -------
    dataIds : `list` or `dataIds`
        The dataIds.
    """

    def isOnSky(expRecord: dafButler.DimensionRecord) -> bool:
        imageType = expRecord.observation_type
        obj = expRecord.target_name
        if checkObject and obj == "NOTSET":
            return False
        if imageType not in skipTypes:
            return True
        return False

    recordSets = []
    days = getDaysWithData(butler)
    if startDate:
        days = [d for d in days if d >= startDate]
    if endDate:
        days = [d for d in days if d <= endDate]
    days = sorted(set(days))

    where = "exposure.day_obs=dayObs"
    for day in days:
        # queryDataIds would be better here, but it's then hard/impossible
        # to do the filtering for which is on sky, so just take the dataIds
        records = butler.registry.queryDimensionRecords(
            "exposure", where=where, bind={"dayObs": day}, datasets="raw"
        )
        recordSets.append(sortRecordsByDayObsThenSeqNum(records))

    dataIds = [r.dataId for r in filter(isOnSky, itertools.chain(*recordSets))]
    if full:
        expandedIds = [
            updateDataIdOrDataCord(butler.registry.expandDataId(dataId, detector=0).mapping)
            for dataId in dataIds
        ]
        filledIds = [fillDataId(butler, dataId) for dataId in expandedIds]
        return filledIds
    else:
        return [updateDataIdOrDataCord(dataId, detector=0) for dataId in dataIds]


def getExpRecord(
    butler: dafButler.Butler,
    instrument: str,
    expId: int | None = None,
    dayObs: int | None = None,
    seqNum: int | None = None,
) -> dafButler.DimensionRecord:
    """Get the exposure record for a given exposure ID or dayObs+seqNum.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        The butler.
    instrument : `str`
        The instrument name, e.g. 'LSSTCam'.
    expId : `int`
        The exposure ID.

    Returns
    -------
    expRecord : `lsst.daf.butler.DimensionRecord`
        The exposure record.
    """
    if expId is None and (dayObs is None or seqNum is None):
        raise ValueError("Must supply either expId or (dayObs AND seqNum)")

    where = "instrument=inst"  # Note you can't use =instrument as bind-strings can't clash with dimensions
    bind: "dict[str, str | int]" = {"inst": instrument}
    if expId:
        where += " AND exposure.id=expId"
        bind.update({"expId": expId})
    if dayObs and seqNum:
        where += " AND exposure.day_obs=dayObs AND exposure.seq_num=seqNum"
        bind.update({"dayObs": dayObs, "seqNum": seqNum})

    expRecords = butler.registry.queryDimensionRecords("exposure", where=where, bind=bind, datasets="raw")
    filteredExpRecords = list(set(expRecords))  # must call set as this may contain many duplicates
    if len(filteredExpRecords) != 1:
        raise RuntimeError(
            f"Failed to find unique exposure record for {instrument=} with"
            f" {expId=}, {dayObs=}, {seqNum=}, got {len(filteredExpRecords)} records"
        )
    return filteredExpRecords[0]
