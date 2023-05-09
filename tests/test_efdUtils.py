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

import unittest
import lsst.utils.tests

# from astropy.time import Time

# XXX protect this import and set a SKIPIF variable to use in decorator
import lsst_efd_client
import astropy
import pandas as pd
import datetime
import asyncio

from lsst.summit.utils.efdUtils import (makeEfdClient,
                                        getEfdData,
                                        getDayObsStartTime,
                                        getDayObsEndTime,
                                        getSubTopics,
                                        )


class EfdUtilsTestCase(lsst.utils.tests.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.client = makeEfdClient()
        except RuntimeError:
            raise unittest.SkipTest("Could not instantiate an EFD client")
        # cls.assertIsInstance(cls.client, lsst_efd_client.efd_helper.EfdClient)
        cls.dayObs = 20230601
        # get a sample expRecord here to test expRecordToTimespan
        cls.axisTopic = 'lsst.sal.MTMount.logevent_azimuthMotionState'

    def tearDown(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.client.influx_client.close())

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

    def test_getSubTopics(self):
        return

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
        # test expRecord
        # test padding options

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
            _ = getEfdData(self.client, 'badTopic', begin=dayStart, end=dayEnd)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
