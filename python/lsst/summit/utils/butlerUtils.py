# This file is part of rapid_analysis.
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

import lsst.daf.butler as dafButler
import itertools
import copy


__all__ = ["makeDefaultLatissButler",
           "sanitize_day_obs",
           "getMostRecentDayObs",
           "getSeqNumsForDayObs",
           "getMostRecentDataId",
           "getDatasetRefForDataId",
           "getDayObs",
           "getSeqNum",
           "getExpId",
           "datasetExists",
           "sortRecordsByAttribute",
           "getDaysWithData",
           "getExpIdFromDayObsSeqNum",
           "updateDataIdOrDataCord",
           "fillDataId",
           "getExpRecordFromDataId",
           "getDayObsSeqNumFromExposureId",
           "removeDataProduct",
           "getLatissOnSkyDataIds",
           "_repoDirToLocation"]

# TODO: DM-33864 Unify these now that DM-32742 is done.
LATISS_DEFAULT_COLLECTIONS = ['LATISS/raw/all', 'LATISS/calib', "LATISS/runs/quickLook"]
LATISS_SUPPLEMENTAL_COLLECTIONS = {'NCSA': ['LATISS/calib/DM-32209'],
                                   'summit': ['u/czw/DM-28920/calib.20210720']}

LATISS_REPO_LOCATION_MAP = {'NCSA': '/repo/main',
                            'NTS': '/readonly/repo/main',
                            'summit': '/repo/LATISS'}

_LOCATIONS = list(LATISS_REPO_LOCATION_MAP.keys())
_REPO_PATHS = list(LATISS_REPO_LOCATION_MAP.values())

# RECENT_DAY must be in the past, to speed up queries by restricting
# them significantly, but data must definitely been taken since. Should
# also not be more than 2 months in the past due to 60 day lookback time on the
# summit. All this means it should be updated by an informed human.
RECENT_DAY = 20220201


def _update_RECENT_DAY(day):
    """Update the value for RECENT_DAY once we have a value for free."""
    global RECENT_DAY
    RECENT_DAY = max(day-1, RECENT_DAY)


def makeDefaultLatissButler(location, *, extraCollections=None, writeable=False):
    """Create a butler for LATISS using the default collections.

    Parameters
    ----------
    location : `str`
        The location for which to create the default butler. Valid values are
        'NCSA', 'NTS' and 'summit'.
    extraCollections : `list` of `str`
        Extra input collections to supply to the butler init.
    writable : `bool`, optional
        Whether to make a writable butler.

    Returns
    -------
    butler : `lsst.daf.butler.Butler`
        The butler.
    """
    # TODO: DM-33849 remove this once we can use the butler API.
    # TODO: Add logging to which collections are going in
    if location not in _LOCATIONS:
        raise RuntimeError(f'Default butler location only supported for {_LOCATIONS}, got {location}')
    repodir = LATISS_REPO_LOCATION_MAP[location]
    LSC = LATISS_SUPPLEMENTAL_COLLECTIONS  # grrr, line lengths
    collections = (LSC[location] if location in LSC.keys() else []) + LATISS_DEFAULT_COLLECTIONS
    if extraCollections:
        collections.extend(extraCollections)
    return dafButler.Butler(repodir, collections=collections, writeable=writeable, instrument='LATISS')


def _repoDirToLocation(repoPath):
    assert(repoPath in _REPO_PATHS)
    # can only be in one place, will need changing if we ever have repo
    # paths that are the same in the repo map
    repoLocationInverse = {v: k for (k, v) in LATISS_REPO_LOCATION_MAP.items()}
    assert len(LATISS_REPO_LOCATION_MAP) == len(repoLocationInverse)  # make sure we didn't drop dupes
    location = repoLocationInverse[repoPath]
    return location


# TODO: DM-32940 can remove this whole function once this ticket merges.
def datasetExists(butler, dataProduct, dataId, **kwargs):
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
    try:
        exists = butler.datasetExists(dataProduct, dataId, **kwargs)
        return exists
    except (LookupError, RuntimeError):
        return False


def sanitize_day_obs(day_obs):
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
            return int(day_obs.replace('-', ''))
        except Exception:
            ValueError(f'Failed to sanitize {day_obs!r} to a day_obs')
    else:
        raise ValueError(f'Cannot sanitize {day_obs!r} to a day_obs')


def getMostRecentDayObs(butler):
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
    where = "exposure.day_obs>RECENT_DAY"
    records = butler.registry.queryDimensionRecords('exposure', where=where, datasets='raw',
                                                    bind={'RECENT_DAY': RECENT_DAY})
    recentDay = max(r.day_obs for r in records)
    _update_RECENT_DAY(recentDay)
    return recentDay


def getSeqNumsForDayObs(butler, day_obs, extraWhere=''):
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
    day_obs = sanitize_day_obs(day_obs)
    where = "exposure.day_obs=day_obs"
    if extraWhere:
        extraWhere = extraWhere.replace('"', '\'')
        where += f" and {extraWhere}"
    records = butler.registry.queryDimensionRecords("exposure",
                                                    where=where,
                                                    bind={'day_obs': day_obs},
                                                    datasets='raw')
    return sorted([r.seq_num for r in records])


def sortRecordsByAttribute(records, attribute):
    """Sort a set of records by a given attribute.

    Parameters
    ----------
    records : `list` of `dict`
        The records to be sorted.
    attribute : `str`
        The attribute to sort by.

    Returns
    -------
    sortedRecords : `list` of `dict`
        The sorted records

    Notes
    -----
    TODO: DM-34240 Does this even work?! What happens when you have several
    dayObs, and the seqNums therefore collide? The initial set() won't catch
    that, so how does this then behave?!
    """
    records = list(records)  # must call list, otherwise can't check length later
    recordSet = set(records)
    if len(records) != len(recordSet):  # must call set *before* sorting!
        raise RuntimeError("Record set contains duplicates, and therefore cannot be sorted unambiguously")

    sortedRecords = [r for (s, r) in sorted([(getattr(r, attribute), r) for r in recordSet])]

    if len(sortedRecords) != len(list(records)):
        raise RuntimeError(f'Ambiguous sort! Key {attribute} did not uniquely sort the records')
    return sortedRecords


def getDaysWithData(butler):
    """Get all the days for which LATISS has taken data on the mountain.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler
        The butler to query.

    Returns
    -------
    days : `list` of `int`
        A sorted list of the day_obs values for which mountain-top data exists.
    """
    # 20200101 is a day between shipping LATISS and going on sky
    # exposure.seq_num<50 massively reduces the number of returned records
    # whilst being large enough to ensure that no days are missed because early
    # seq_nums were skipped
    where = "exposure.day_obs>20200101 and exposure.seq_num<50"
    records = butler.registry.queryDimensionRecords("exposure", where=where, datasets='raw')
    return sorted(set([r.day_obs for r in records]))


def getMostRecentDataId(butler):
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
    dataId = {'day_obs': lastDay, 'seq_num': seqNum, 'detector': 0}
    dataId.update(getExpIdFromDayObsSeqNum(butler, dataId))
    return dataId


def getExpIdFromDayObsSeqNum(butler, dataId):
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
    return {'exposure': expRecord.id}


def updateDataIdOrDataCord(dataId, **updateKwargs):
    """Add key, value pairs to a dataId or data coordinate.

    Parameters
    ----------
    dataId : `dict`
        The dataId for which to return the exposure id.
    updateKwargs : `dict`
        The key value pairs add to the dataId or dataCoord.

    Returns
    -------
    dataId : `dict`
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


def fillDataId(butler, dataId):
    """Given a dataId, fill it with values for all available dimensions.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        The butler.
    dataId : `dict`
        The dataId to fill.

    Returns
    -------
    dataId : `dict`
        The filled dataId.

    Notes
    -----
    This function is *slow*! Running this on 20,000 dataIds takes approximately
    7 minutes. Virtually all the slowdown is in the
    butler.registry.expandDataId() call though, so this wrapper is not to blame
    here, and might speed up in future with butler improvements.
    """
    # ensure it's a dict to deal with records etc
    dataId = _assureDict(dataId)

    # this removes extraneous keys that would trip up the registry call
    # using _rewrite_data_id is perhaps ever so slightly slower than popping
    # the bad keys, or making a minimal dataId by hand, but is more
    # reliable/general, so we choose that over the other approach here
    dataId, _ = butler._rewrite_data_id(dataId, butler.registry.getDatasetType('raw'))

    # now expand and turn back to a dict
    dataId = butler.registry.expandDataId(dataId, detector=0).full  # this call is VERY slow
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


def _assureDict(dataId):
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
    elif hasattr(dataId, 'items'):  # dafButler.dimensions.DataCoordinate
        return {str(k): v for k, v in dataId.items()}  # str() required due to full names
    elif hasattr(dataId, 'dataId'):  # dafButler.dimensions.DimensionRecord
        return {str(k): v for k, v in dataId.dataId.items()}
    else:
        raise RuntimeError(f'Failed to coerce {type(dataId)} to dict')


def getExpRecordFromDataId(butler, dataId):
    """Get the exposure record for a given dataId.

    Parameters
    ----------
    butler : `lsst.daf.butler.Butler`
        The butler.
    dataId : `dict`
        The dataId.

    Returns
    -------
    expRecord : `lsst.daf.butler.dimensions.ExposureRecord`
        The exposure record.
    """
    dataId = _assureDict(dataId)
    assert isinstance(dataId, dict), f'dataId must be a dict or DimensionRecord, got {type(dataId)}'

    if expId := getExpId(dataId):
        where = "exposure.id=expId"
        expRecords = butler.registry.queryDimensionRecords("exposure",
                                                           where=where,
                                                           bind={'expId': expId},
                                                           datasets='raw')

    else:
        dayObs = getDayObs(dataId)
        seqNum = getSeqNum(dataId)
        if not (dayObs and seqNum):
            raise RuntimeError(f'Failed to find either expId or day_obs and seq_num in dataId {dataId}')
        where = "exposure.day_obs=day_obs AND exposure.seq_num=seq_num"
        expRecords = butler.registry.queryDimensionRecords("exposure",
                                                           where=where,
                                                           bind={'day_obs': dayObs, 'seq_num': seqNum},
                                                           datasets='raw')

    expRecords = list(expRecords)
    assert len(expRecords) == 1, f'Found more than one exposure record for {dataId}'
    return expRecords[0]


def getDayObsSeqNumFromExposureId(butler, dataId):
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
        return {'day_obs': dayObs, 'seq_num': seqNum}

    if isinstance(dataId, int):
        dataId = {'exposure': dataId}
    else:
        dataId = _assureDict(dataId)
    assert isinstance(dataId, dict)

    if not (expId := getExpId(dataId)):
        raise RuntimeError(f'Failed to find exposure id in {dataId}')

    where = "exposure.id=expId"
    expRecords = butler.registry.queryDimensionRecords("exposure",
                                                       where=where,
                                                       bind={'expId': expId},
                                                       datasets='raw')
    expRecords = list(expRecords)
    assert len(expRecords) == 1, f'Found more than one exposure record for {dataId}'
    return {'day_obs': expRecords[0].day_obs, 'seq_num': expRecords[0].seq_num}


def getDatasetRefForDataId(butler, datasetType, dataId):
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

    dRef = butler.registry.findDataset(datasetType, dataId)
    return dRef


def removeDataProduct(butler, datasetType, dataId):
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
    if datasetType == 'raw':
        raise RuntimeError("I'm sorry, Dave, I'm afraid I can't do that.")
    dRef = getDatasetRefForDataId(butler, datasetType, dataId)
    butler.pruneDatasets([dRef], disassociate=True, unstore=True, purge=True)
    return


def _dayobs_present(dataId):
    return _get_dayobs_key(dataId) is not None


def _seqnum_present(dataId):
    return _get_seqnum_key(dataId) is not None


def _expid_present(dataId):
    return _get_expid_key(dataId) is not None


def _get_dayobs_key(dataId):
    """Return the key for day_obs if present, else None
    """
    keys = [k for k in dataId.keys() if k.find('day_obs') != -1]
    if not keys:
        return None
    return keys[0]


def _get_seqnum_key(dataId):
    """Return the key for seq_num if present, else None
    """
    keys = [k for k in dataId.keys() if k.find('seq_num') != -1]
    if not keys:
        return None
    return keys[0]


def _get_expid_key(dataId):
    """Return the key for expId if present, else None
    """
    if 'exposure.id' in dataId:
        return 'exposure.id'
    elif 'exposure' in dataId:
        return 'exposure'
    return None


def getDayObs(dataId):
    """Get the day_obs from a dataId.

    Parameters
    ----------
    dataId : `dict`
        The dataId.

    Returns
    -------
    day_obs : `int` or `None`
        The day_obs value if present, else None.
    """
    if not _dayobs_present(dataId):
        return None
    return dataId['day_obs'] if 'day_obs' in dataId else dataId['exposure.day_obs']


def getSeqNum(dataId):
    """Get the seq_num from a dataId.

    Parameters
    ----------
    dataId : `dict`
        The dataId.

    Returns
    -------
    seq_num : `int` or `None`
        The seq_num value if present, else None.
    """
    if not _seqnum_present(dataId):
        return None
    return dataId['seq_num'] if 'seq_num' in dataId else dataId['exposure.seq_num']


def getExpId(dataId):
    """Get the expId from a dataId.

    Parameters
    ----------
    dataId : `dict`
        The dataId.

    Returns
    -------
    expId : `int` or `None`
        The expId value if present, else None.
    """
    if not _expid_present(dataId):
        return None
    return dataId['exposure'] if 'exposure' in dataId else dataId['exposure.id']


def getLatissOnSkyDataIds(butler, skipTypes=('bias', 'dark', 'flat'), checkObject=True, full=True,
                          startDate=None, endDate=None):
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
    def isOnSky(expRecord):
        imageType = expRecord.observation_type
        obj = expRecord.target_name
        if checkObject and obj == 'NOTSET':
            return False
        if imageType not in skipTypes:
            return True
        return False

    recordSets = []
    days = getDaysWithData(butler)
    if startDate:
        days = [d for d in days if d >= startDate]
    if endDate:
        days = [d for d in days if d <= startDate]
    days = sorted(set(days))

    where = "exposure.day_obs=day_obs"
    for day in days:
        # queryDataIds would be better here, but it's then hard/impossible
        # to do the filtering for which is on sky, so just take the dataIds
        records = butler.registry.queryDimensionRecords("exposure",
                                                        where=where,
                                                        bind={'day_obs': day},
                                                        datasets='raw')
        recordSets.append(sortRecordsByAttribute(records, 'seq_num'))

    dataIds = [r.dataId for r in filter(isOnSky, itertools.chain(*recordSets))]
    if full:
        expandedIds = [updateDataIdOrDataCord(butler.registry.expandDataId(dataId, detector=0).full)
                       for dataId in dataIds]
        filledIds = [fillDataId(butler, dataId) for dataId in expandedIds]
        return filledIds
    else:
        return [updateDataIdOrDataCord(dataId, detector=0) for dataId in dataIds]
