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

"""Test cases for utils."""

import asyncio
import datetime
import unittest

import astropy
import pandas as pd
from astropy.time import Time

import lsst.utils.tests
from lsst.summit.utils.efdUtils import (
    astropyToEfdTimestamp,
    clipDataToEvent,
    efdTimestampToAstropy,
    getDayObsEndTime,
    getDayObsForTime,
    getDayObsStartTime,
    getEfdData,
    getMostRecentRowWithDataBefore,
    getTopics,
    makeEfdClient,
)
from lsst.summit.utils.tmaUtils import TMAEvent, TMAState

from .utils import getVcr

HAS_EFD_CLIENT = True
try:
    import lsst_efd_client
except ImportError:
    HAS_EFD_CLIENT = False

vcr = getVcr()


@unittest.skipIf(not HAS_EFD_CLIENT, "No EFD client available")
@vcr.use_cassette()
class EfdUtilsTestCase(lsst.utils.tests.TestCase):
    @classmethod
    @vcr.use_cassette()
    def setUpClass(cls):
        try:
            cls.client = makeEfdClient(testing=True)
        except RuntimeError:
            raise unittest.SkipTest("Could not instantiate an EFD client")
        cls.dayObs = 20230531
        # get a sample expRecord here to test expRecordToTimespan
        cls.axisTopic = "lsst.sal.MTMount.logevent_azimuthMotionState"
        cls.timeSeriesTopic = "lsst.sal.MTMount.azimuth"
        cls.event = TMAEvent(
            dayObs=20230531,
            seqNum=27,
            type=TMAState.TRACKING,
            endReason=TMAState.SLEWING,
            duration=0.47125244140625,
            begin=Time(1685578353.2265284, scale="utc", format="unix"),
            end=Time(1685578353.6977808, scale="utc", format="unix"),
            blockInfos=None,
            version=0,
            _startRow=254,
            _endRow=255,
        )

    @vcr.use_cassette()
    def tearDown(self):
        loop = asyncio.get_event_loop()
        if self.client.influx_client is not None:
            loop.run_until_complete(self.client.influx_client.close())

    @vcr.use_cassette()
    def test_makeEfdClient(self):
        self.assertIsInstance(self.client, lsst_efd_client.efd_helper.EfdClient)

    def test_getDayObsAsTimes(self):
        """This tests getDayObsStartTime and getDayObsEndTime explicitly,
        but the days we loop over are chosen to test calcNextDay() which is
        called by getDayObsEndTime().
        """
        for dayObs in (
            self.dayObs,  # the nominal value
            20200228,  # day before end of Feb on a leap year
            20200229,  # end of Feb on a leap year
            20210227,  # day before end of Feb on a non-leap year
            20200228,  # end of Feb on a non-leap year
            20200430,  # end of a month with 30 days
            20200530,  # end of a month with 31 days
            20201231,  # year rollover
        ):
            dayStart = getDayObsStartTime(dayObs)
            self.assertIsInstance(dayStart, astropy.time.Time)

            dayEnd = getDayObsEndTime(dayObs)
            self.assertIsInstance(dayStart, astropy.time.Time)

            self.assertGreater(dayEnd, dayStart)
            self.assertEqual(dayEnd.jd, dayStart.jd + 1)

    @vcr.use_cassette()
    def test_getTopics(self):
        topics = getTopics(self.client, "lsst.sal.MTMount*")
        self.assertIsInstance(topics, list)
        self.assertGreater(len(topics), 0)

        topics = getTopics(self.client, "*fake.topics.does.not.exist*")
        self.assertIsInstance(topics, list)
        self.assertEqual(len(topics), 0)

        # check we can find the mount with a preceding wildcard
        topics = getTopics(self.client, "*mTmoUnt*")
        self.assertIsInstance(topics, list)
        self.assertGreater(len(topics), 0)

        # check it fails if we don't allow case insensitivity
        topics = getTopics(self.client, "*mTmoUnt*", caseSensitive=True)
        self.assertIsInstance(topics, list)
        self.assertEqual(len(topics), 0)

    @vcr.use_cassette()
    def test_getEfdData(self):
        dayStart = getDayObsStartTime(self.dayObs)
        dayEnd = getDayObsEndTime(self.dayObs)
        oneDay = datetime.timedelta(hours=24)
        # twelveHours = datetime.timedelta(hours=12)

        # test the dayObs interface
        dayObsData = getEfdData(self.client, self.axisTopic, dayObs=self.dayObs)
        self.assertIsInstance(dayObsData, pd.DataFrame)

        # test the starttime interface
        dayStartData = getEfdData(self.client, self.axisTopic, begin=dayStart, timespan=oneDay)
        self.assertIsInstance(dayStartData, pd.DataFrame)

        # check they're equal
        self.assertTrue(dayObsData.equals(dayStartData))

        # test the starttime interface with an endtime
        dayEnd = getDayObsEndTime(self.dayObs)
        dayStartEndData = getEfdData(self.client, self.axisTopic, begin=dayStart, end=dayEnd)
        self.assertTrue(dayObsData.equals(dayStartEndData))

        # test event
        # note that here we're going to clip to an event and pad things, so
        # we want to use the timeSeriesTopic not the states, so that there's
        # plenty of rows to test the padding is actually working
        eventData = getEfdData(self.client, self.timeSeriesTopic, event=self.event)
        self.assertIsInstance(dayObsData, pd.DataFrame)

        # test padding options
        padded = getEfdData(self.client, self.timeSeriesTopic, event=self.event, prePadding=1, postPadding=2)
        self.assertGreater(len(padded), len(eventData))
        startTimeDiff = efdTimestampToAstropy(eventData.iloc[0]["private_efdStamp"]) - efdTimestampToAstropy(
            padded.iloc[0]["private_efdStamp"]
        )
        endTimeDiff = efdTimestampToAstropy(padded.iloc[-1]["private_efdStamp"]) - efdTimestampToAstropy(
            eventData.iloc[-1]["private_efdStamp"]
        )

        self.assertGreater(startTimeDiff.sec, 0)
        self.assertLess(startTimeDiff.sec, 1.1)  # padding isn't super exact, so give a little wiggle room
        self.assertGreater(endTimeDiff.sec, 0)
        self.assertLess(endTimeDiff.sec, 2.1)  # padding isn't super exact, so give a little wiggle room

        with self.assertRaises(ValueError):
            # not enough info to constrain
            _ = getEfdData(self.client, self.axisTopic)
            # dayObs supplied and a start time is not allowed
            _ = getEfdData(self.client, self.axisTopic, dayObs=self.dayObs, begin=dayStart)
            # dayObs supplied and a stop time is not allowed
            _ = getEfdData(self.client, self.axisTopic, dayObs=self.dayObs, end=dayEnd)
            # dayObs supplied and timespan is not allowed
            _ = getEfdData(self.client, self.axisTopic, dayObs=self.dayObs, timespan=oneDay)
            # being alone is not allowed
            _ = getEfdData(self.client, self.axisTopic, begin=self.dayObs)
            # good query, except the topic doesn't exist
            _ = getEfdData(self.client, "badTopic", begin=dayStart, end=dayEnd)

    @vcr.use_cassette()
    def test_getMostRecentRowWithDataBefore(self):
        time = Time(1687845854.736784, scale="utc", format="unix")
        rowData = getMostRecentRowWithDataBefore(
            self.client, "lsst.sal.MTM1M3.logevent_forceActuatorState", time
        )
        self.assertIsInstance(rowData, pd.Series)

        stateTime = efdTimestampToAstropy(rowData["private_efdStamp"])
        self.assertLess(stateTime, time)

    def test_efdTimestampToAstropy(self):
        time = efdTimestampToAstropy(1687845854.736784)
        self.assertIsInstance(time, astropy.time.Time)
        return

    def test_astropyToEfdTimestamp(self):
        time = Time(1687845854.736784, scale="utc", format="unix")
        efdTimestamp = astropyToEfdTimestamp(time)
        self.assertIsInstance(efdTimestamp, float)
        return

    @vcr.use_cassette()
    def test_clipDataToEvent(self):
        # get 10 mins of data either side of the event we'll clip to
        duration = datetime.timedelta(seconds=10 * 60)
        queryBegin = self.event.begin - duration
        queryEnd = self.event.end + duration
        dayObsData = getEfdData(self.client, "lsst.sal.MTMount.azimuth", begin=queryBegin, end=queryEnd)

        # clip the data, and check it's shorter, non-zero, and falls in the
        # right time range
        clippedData = clipDataToEvent(dayObsData, self.event)

        self.assertIsInstance(clippedData, pd.DataFrame)
        self.assertGreater(len(clippedData), 0)
        self.assertLess(len(clippedData), len(dayObsData))

        dataStart = efdTimestampToAstropy(clippedData.iloc[0]["private_efdStamp"])
        dataEnd = efdTimestampToAstropy(clippedData.iloc[-1]["private_efdStamp"])

        self.assertGreaterEqual(dataStart, self.event.begin)
        self.assertLessEqual(dataEnd, self.event.end)

        # test the pre/post padding options
        clippedPaddedData = clipDataToEvent(dayObsData, self.event, prePadding=1, postPadding=2)
        self.assertIsInstance(clippedPaddedData, pd.DataFrame)
        self.assertGreater(len(clippedPaddedData), 0)
        self.assertLess(len(clippedPaddedData), len(dayObsData))
        self.assertGreater(len(clippedPaddedData), len(clippedData))

        paddedDataStart = efdTimestampToAstropy(clippedPaddedData.iloc[0]["private_efdStamp"])
        paddedDataEnd = efdTimestampToAstropy(clippedPaddedData.iloc[-1]["private_efdStamp"])
        self.assertLessEqual(paddedDataStart, dataStart)
        self.assertGreaterEqual(paddedDataEnd, dataEnd)

        # Get the minimum and maximum timestamps before padding
        startTimeUnpadded = clippedData["private_efdStamp"].min()
        endTimeUnpadded = clippedData["private_efdStamp"].max()

        # Get the minimum and maximum timestamps after padding
        startTimePadded = clippedPaddedData["private_efdStamp"].min()
        endTimePadded = clippedPaddedData["private_efdStamp"].max()

        # Check that the difference between the min times and max times is
        # approximately equal to the padding. Not exact as data sampling is
        # not infinite.
        self.assertAlmostEqual(startTimeUnpadded - startTimePadded, 1, delta=0.1)
        self.assertAlmostEqual(endTimePadded - endTimeUnpadded, 2, delta=0.1)
        return

    def test_getDayObsForTime(self):
        pydate = datetime.datetime(2023, 2, 5, 13, 30, 1)
        time = Time(pydate)
        dayObs = getDayObsForTime(time)
        self.assertEqual(dayObs, 20230205)

        pydate = datetime.datetime(2023, 2, 5, 11, 30, 1)
        time = Time(pydate)
        dayObs = getDayObsForTime(time)
        self.assertEqual(dayObs, 20230204)
        return


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
