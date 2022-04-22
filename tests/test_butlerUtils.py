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

import unittest
from typing import Iterable
import datetime

import os
import lsst.utils.tests
from lsst.rapid.analysis.butlerUtils import (makeDefaultLatissButler,
                                             sanitize_day_obs,
                                             getMostRecentDayObs,
                                             getSeqNumsForDayObs,
                                             getMostRecentDataId,
                                             getDatasetRefForDataId,
                                             _dayobs_present,
                                             _seqnum_present,
                                             _expid_present,
                                             _get_dayobs_key,
                                             _get_seqnum_key,
                                             _get_expid_key,
                                             getDayObs,
                                             getSeqNum,
                                             getExpId,
                                             datasetExists,
                                             sortRecordsByAttribute,
                                             getDaysWithData,
                                             getExpIdFromDayObsSeqNum,
                                             updateDataIdOrDataCord,
                                             fillDataId,
                                             getExpRecordFromDataId,
                                             getDayObsSeqNumFromExposureId,
                                             getLatissOnSkyDataIds,
                                             _assureDict,
                                             LATISS_DEFAULT_COLLECTIONS,
                                             RECENT_DAY,
                                             LATISS_REPO_LOCATION_MAP,
                                             LATISS_SUPPLEMENTAL_COLLECTIONS)
from lsst.rapid.analysis.butlerUtils import removeDataProduct, _repoDirToLocation  # noqa: F401
import lsst.daf.butler as dafButler


class ButlerUtilsTestCase(lsst.utils.tests.TestCase):
    """A test case for testing sky position offsets for exposures."""

    def setUp(self):
        # this also functions as test_makeDefaultLatissButler(), but we may as
        # well catch the butler once it's made so it can be reused if needed,
        # given how hard it is to made it robustly

        # TODO: if/when RFC-811 passes, update this to use the env var
        possiblePaths = LATISS_REPO_LOCATION_MAP.values()
        paths = [path for path in possiblePaths if os.path.exists(path)]
        # can only be in one place, will need changing if we ever have repo
        # paths that are the same in the repo map
        assert len(paths) == 1
        butlerPath = paths[0]
        LATISS_REPO_LOCATION_MAP_INVERSE = {v: k for (k, v) in LATISS_REPO_LOCATION_MAP.items()}
        location = LATISS_REPO_LOCATION_MAP_INVERSE[butlerPath]

        with self.assertRaises(RuntimeError):
            makeDefaultLatissButler('')
            makeDefaultLatissButler('ThisIsNotAvalidLocation')

        # butler stuff
        butler = makeDefaultLatissButler(location)
        self.assertIsInstance(butler, dafButler.Butler)
        self.butler = butler

        # dict-like dataIds
        self.rawDataId = getMostRecentDataId(self.butler)
        self.fullId = fillDataId(butler, self.rawDataId)
        self.assertIn('exposure', self.fullId)
        self.assertIn('day_obs', self.fullId)
        self.assertIn('seq_num', self.fullId)
        self.expIdOnly = {'exposure': self.fullId['exposure'], 'detector': 0}
        self.dayObsSeqNumIdOnly = {'day_obs': getDayObs(self.fullId), 'seq_num': getSeqNum(self.fullId),
                                   'detector': 0}

        # expRecords
        self.expRecordNoDetector = getExpRecordFromDataId(self.butler, self.rawDataId)
        self.assertIsInstance(self.expRecordNoDetector, dafButler.dimensions.DimensionRecord)
        self.assertFalse(hasattr(self.expRecordNoDetector, 'detector'))
        # just a crosscheck on the above to make sure other things are correct
        self.assertTrue(hasattr(self.expRecordNoDetector, 'instrument'))

        # data coordinates
        # popping here because butler.registry.expandDataId cannot have
        # day_obs or seq_num present right now
        rawDataIdNoDayObSeqNum = _assureDict(self.rawDataId)
        if dayObsKey := _get_dayobs_key(rawDataIdNoDayObSeqNum):
            rawDataIdNoDayObSeqNum.pop(dayObsKey)
        if seqNumKey := _get_seqnum_key(rawDataIdNoDayObSeqNum):
            rawDataIdNoDayObSeqNum.pop(seqNumKey)
        self.rawDataIdNoDayObSeqNum = rawDataIdNoDayObSeqNum
        self.dataCoordMinimal = butler.registry.expandDataId(self.rawDataIdNoDayObSeqNum, detector=0)
        self.dataCoordFullView = butler.registry.expandDataId(self.rawDataIdNoDayObSeqNum, detector=0).full
        self.assertIsInstance(self.dataCoordMinimal, dafButler.dimensions.DataCoordinate)
        # NB the type check below is currently using a non-public API, but
        # at present there isn't a good alternative
        viewType = dafButler.core.dimensions._coordinate._DataCoordinateFullView
        self.assertIsInstance(self.dataCoordFullView, viewType)

    def test_LATISS_REPO_LOCATION_MAP(self):
        self.assertTrue(LATISS_REPO_LOCATION_MAP is not None)
        self.assertTrue(LATISS_REPO_LOCATION_MAP != [])
        self.assertTrue(len(LATISS_REPO_LOCATION_MAP) >= 1)
        self.assertTrue(len(LATISS_REPO_LOCATION_MAP) >= len(LATISS_SUPPLEMENTAL_COLLECTIONS))

    def test_LATISS_DEFAULT_COLLECTIONS(self):
        self.assertTrue(LATISS_DEFAULT_COLLECTIONS is not None)
        self.assertTrue(LATISS_DEFAULT_COLLECTIONS != [])
        self.assertTrue(len(LATISS_DEFAULT_COLLECTIONS) >= 1)

    def test_RECENT_DAY(self):
        todayInt = int(datetime.date.today().strftime("%Y%m%d"))
        self.assertTrue(RECENT_DAY <= todayInt)  # in the past
        self.assertTrue(RECENT_DAY >= 20200101)  # not too far in the past

        # no test here, but print a warning if it hasn't been updated recently
        recentDay_datetime = datetime.datetime.strptime(str(RECENT_DAY), "%Y%m%d")
        now = datetime.datetime.today()
        timeSinceUpdate = now - recentDay_datetime
        if timeSinceUpdate.days > 100:  # TODO:
            print(f"RECENT_DAY is now {timeSinceUpdate.days} days in the past. "
                  "You might want to consider updating this to speed up butler queries.")

    def test_sanitize_day_obs(self):
        dayObs = '2020-01-02'
        self.assertEqual(sanitize_day_obs(dayObs), 20200102)
        dayObs = 20210201
        self.assertEqual(sanitize_day_obs(dayObs), dayObs)

        with self.assertRaises(ValueError):
            sanitize_day_obs(1.234)
            sanitize_day_obs('Febuary 29th, 1970')

    def test_getMostRecentDayObs(self):
        # just a basic sanity check here as we can't know the value,
        # but at least check something is returned, and is plausible
        recentDay = getMostRecentDayObs(self.butler)
        self.assertIsInstance(recentDay, int)
        self.assertTrue(recentDay >= RECENT_DAY)
        # some test data might be set a millennium in the future, i.e.
        # the year wouldd be 2XXX+1000, so set to y4k just in case
        self.assertTrue(recentDay < 40000000)

    def test_getSeqNumsForDayObs(self):
        emptyDay = 19990101
        seqnums = getSeqNumsForDayObs(self.butler, emptyDay)
        self.assertIsInstance(seqnums, Iterable)
        self.assertEqual(len(list(seqnums)), 0)

        recentDay = getMostRecentDayObs(self.butler)
        seqnums = getSeqNumsForDayObs(self.butler, recentDay)
        self.assertIsInstance(seqnums, Iterable)
        self.assertTrue(len(list(seqnums)) >= 1)

    def test_getMostRecentDataId(self):
        # we can't know the values, but it should always return something
        # and the dict and int forms should always have certain keys and agree
        dataId = getMostRecentDataId(self.butler)
        self.assertIsInstance(dataId, dict)
        self.assertIn('day_obs', dataId)
        self.assertIn('seq_num', dataId)
        self.assertTrue('exposure' in dataId or 'exposure.id' in dataId)

    def test_getDatasetRefForDataId(self):
        dRef = getDatasetRefForDataId(self.butler, 'raw', self.rawDataId)
        self.assertIsInstance(dRef, lsst.daf.butler.core.datasets.ref.DatasetRef)

    def test__dayobs_present(self):
        goods = [{'day_obs': 123}, {'exposure.day_obs': 234}, {'day_obs': 345, 'otherkey': -1}]
        bads = [{'different_key': 123}]
        for good in goods:
            self.assertTrue(_dayobs_present(good))
        for bad in bads:
            self.assertFalse(_dayobs_present(bad))

    def test__seqnum_present(self):
        goods = [{'seq_num': 123}, {'exposure.seq_num': 234}, {'seq_num': 345, 'otherkey': -1}]
        bads = [{'different_key': 123}]
        for good in goods:
            self.assertTrue(_seqnum_present(good))
        for bad in bads:
            self.assertFalse(_seqnum_present(bad))

    def test__expid_present(self):
        goods = [{'exposure': 123}, {'exposure.id': 234}, {'exposure.id': 345, 'otherkey': -1}]
        bads = [{'different_key': 123}]
        for good in goods:
            self.assertTrue(_expid_present(good))
        for bad in bads:
            self.assertFalse(_expid_present(bad))

    def test_getDayObs(self):
        dayVal = 98765
        goods = [{'day_obs': dayVal}, {'exposure.day_obs': dayVal}, {'day_obs': dayVal, 'otherkey': -1}]
        bads = [{'different_key': 123}]
        for good in goods:
            self.assertTrue(getDayObs(good) == dayVal)
        for bad in bads:
            self.assertTrue(getDayObs(bad) is None)

    def test_getSeqNum(self):
        seqVal = 12345
        goods = [{'seq_num': seqVal}, {'exposure.seq_num': seqVal}, {'seq_num': seqVal, 'otherkey': -1}]
        bads = [{'different_key': 123}]
        for good in goods:
            self.assertTrue(getSeqNum(good) == seqVal)
        for bad in bads:
            self.assertTrue(getSeqNum(bad) is None)

    def test_getExpId(self):
        expIdVal = 12345
        goods = [{'exposure': expIdVal}, {'exposure.id': expIdVal}, {'exposure': expIdVal, 'otherkey': -1}]
        bads = [{'different_key': 123}]
        for good in goods:
            self.assertTrue(getExpId(good) == expIdVal)
        for bad in bads:
            self.assertTrue(getExpId(bad) is None)

    def test_datasetExists(self):
        self.assertTrue(datasetExists(self.butler, 'raw', self.rawDataId))
        self.assertTrue(datasetExists(self.butler, 'raw', self.expIdOnly))
        self.assertTrue(datasetExists(self.butler, 'raw', self.dayObsSeqNumIdOnly))
        return

    def test_sortRecordsByAttribute(self):
        where = "exposure.day_obs=day_obs"
        expRecords = self.butler.registry.queryDimensionRecords("exposure", where=where,
                                                                bind={'day_obs': RECENT_DAY})
        sortedIds = sortRecordsByAttribute(expRecords, 'seq_num')
        for i, _id in enumerate(sortedIds[:-1]):
            self.assertTrue(_id.seq_num < sortedIds[i+1].seq_num)

        # XXX deal with ambiguous sorts
        # with self.assertRaises(RuntimeError):
        #     expRecords = self.butler.registry.queryDimensionRecords
        # ("exposure", where=where, bind={'day_obs': RECENT_DAY})
        #     ids = itertools.chain(expRecords, expRecords)
        #     sortedIds = sortRecordsByAttribute(ids, 'seq_num')
        return

    def test_getDaysWithData(self):
        days = getDaysWithData(self.butler)
        self.assertTrue(len(days) >= 0)
        self.assertIsInstance(days[0], int)
        return

    def test_getExpIdFromDayObsSeqNum(self):
        expId = getExpIdFromDayObsSeqNum(self.butler, self.dayObsSeqNumIdOnly)
        self.assertTrue(_expid_present(expId))
        return

    def test_updateDataIdOrDataCord(self):
        updateVals = {'testKey': 'testValue'}

        ids = [self.rawDataId, self.expRecordNoDetector, self.dataCoordMinimal, self.dataCoordFullView]
        for originalId in ids:
            testId = updateDataIdOrDataCord(originalId, **updateVals)
            for k, v in updateVals.items():
                self.assertTrue(testId[k] == v)
        return

    def test_fillDataId(self):
        self.assertFalse(_dayobs_present(self.expIdOnly))
        self.assertFalse(_seqnum_present(self.expIdOnly))

        fullId = fillDataId(self.butler, self.expIdOnly)
        self.assertTrue(_dayobs_present(fullId))
        self.assertTrue(_seqnum_present(fullId))

        ids = [self.rawDataId, self.expRecordNoDetector, self.dataCoordMinimal, self.dataCoordFullView]
        for dataId in ids:
            fullId = fillDataId(self.butler, dataId)
            self.assertTrue(_dayobs_present(fullId))
            self.assertTrue(_seqnum_present(fullId))
            self.assertTrue(_expid_present(fullId))
        return

    def test_getExpRecordFromDataId(self):
        record = getExpRecordFromDataId(self.butler, self.rawDataId)
        self.assertIsInstance(record, dafButler.dimensions.DimensionRecord)
        return

    def test_getDayObsSeqNumFromExposureId(self):
        dayObsSeqNum = getDayObsSeqNumFromExposureId(self.butler, self.expIdOnly)
        self.assertTrue(_dayobs_present(dayObsSeqNum))
        self.assertTrue(_seqnum_present(dayObsSeqNum))
        return

    def test_removeDataProduct(self):
        # Can't think of an easy or safe test for this
        return

    def test_getLatissOnSkyDataIds(self):
        # This is very slow, consider removing as it's the least import of all
        # the util functions. However, restricting it to only the most recent
        # day does help a lot, so probably OK like that, and should speed up
        # with middleware improvements in the future, and we should ensure
        # that they don't break this, so inclined to leave for now
        dayToUse = getDaysWithData(self.butler)[-1]
        # the most recent day with data might only be biases or flats so make
        # sure to override the default of skipping biases, darks & flats
        skipTypes = ()
        ids = getLatissOnSkyDataIds(self.butler, skipTypes=skipTypes, startDate=dayToUse, endDate=dayToUse)
        self.assertTrue(len(ids) > 0)
        self.assertTrue(ids[0] is not None)

        ids = getLatissOnSkyDataIds(self.butler, skipTypes=skipTypes, startDate=dayToUse, endDate=dayToUse,
                                    full=True)
        self.assertTrue(len(ids) > 0)
        self.assertTrue(ids[0] is not None)
        testId = ids[0]
        self.assertTrue(_dayobs_present(testId))
        self.assertTrue(_seqnum_present(testId))
        self.assertTrue(_expid_present(testId))
        return

    def test__repoDirToLocation(self):
        # TODO: DM-34238 Remove this test and all mentions of repoDirToLocation
        # Actually pretty sure this whole method is going away
        # it's pretty gross, and only used by bestEffortIsr because I didn't
        # want to change its API in the middle of the last run.
        return

    def test__assureDict(self):
        for item in [self.rawDataId, self.fullId, self.expIdOnly,
                     self.expRecordNoDetector, self.dataCoordFullView,
                     self.dataCoordMinimal, self.rawDataIdNoDayObSeqNum]:
            testId = _assureDict(item)
            self.assertIsInstance(testId, dict)
        return

    def test__get_dayobs_key(self):
        dataId = {'a_random_key': 321, 'exposure.day_obs': 20200312, 'z_random_key': 'abc'}
        self.assertTrue(_get_dayobs_key(dataId) == 'exposure.day_obs')
        dataId = {'day_obs': 20200312}
        self.assertTrue(_get_dayobs_key(dataId) == 'day_obs')
        dataId = {'missing': 20200312}
        self.assertTrue(_get_dayobs_key(dataId) is None)
        return

    def test__get_seqnum_key(self):
        dataId = {'a_random_key': 321, 'exposure.seq_num': 123, 'z_random_key': 'abc'}
        self.assertTrue(_get_seqnum_key(dataId) == 'exposure.seq_num')
        dataId = {'seq_num': 123}
        self.assertTrue(_get_seqnum_key(dataId) == 'seq_num')
        dataId = {'missing': 123}
        self.assertTrue(_get_seqnum_key(dataId) is None)
        return

    def test__get_expid_key(self):
        dataId = {'a_random_key': 321, 'exposure.id': 123, 'z_random_key': 'abc'}
        self.assertTrue(_get_expid_key(dataId) == 'exposure.id')
        dataId = {'a_random_key': 321, 'exposure': 123, 'z_random_key': 'abc'}
        self.assertTrue(_get_expid_key(dataId) == 'exposure')
        dataId = {'missing': 123}
        self.assertTrue(_get_expid_key(dataId) is None)
        return


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
