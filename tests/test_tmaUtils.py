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

from lsst.summit.utils.efdUtils import makeEfdClient
from lsst.summit.utils.tmaUtils import (TMA,
                                        TMAEvent,
                                        TMAEventMaker,
                                        TMAState,
                                        AxisMotionState,
                                        PowerState,
                                        getAxisAndType,
                                        )


class TmatilsTestCase(lsst.utils.tests.TestCase):

    # @classmethod
    # def setUpClass(cls):
    #     try:
    #         cls.client = makeEfdClient()
    #     except RuntimeError:
    #         raise unittest.SkipTest("Could not instantiate an EFD client")
    #     # cls.assertIsInstance(cls.client, lsst_efd_client.efd_helper.EfdClient)
    #     cls.dayObs = 20230601
    #     # get a sample expRecord here to test expRecordToTimespan
    #     cls.axisTopic = 'lsst.sal.MTMount.logevent_azimuthMotionState'

    # def tearDown(self):
    #     loop = asyncio.get_event_loop()
    #     loop.run_until_complete(self.client.influx_client.close())

    # def test_makeEfdClient(self):
    #     self.assertIsInstance(self.client, lsst_efd_client.efd_helper.EfdClient)

    def test_tmaInit(self):
        tma = TMA()
        self.assertFalse(tma._isValid)

        # setting one axis should not make things valid
        tma._parts['azimuthMotionState'] = 1
        self.assertFalse(tma._isValid)

        # setting all the other components should make things valid
        tma._parts['azimuthInPosition'] = 1
        tma._parts['azimuthSystemState'] = 1
        tma._parts['elevationInPosition'] = 1
        tma._parts['elevationMotionState'] = 1
        tma._parts['elevationSystemState'] = 1
        self.assertTrue(tma._isValid)

    def test_tmaReferences(self):
        """Check the linkage between the component lists and the _parts dict.
        """
        tma = TMA()

        # setting one axis should not make things valid
        self.assertEqual(tma._parts['azimuthMotionState'], tma._UNINITIALIZED_VALUE)
        self.assertEqual(tma._parts['elevationMotionState'], tma._UNINITIALIZED_VALUE)
        tma.motion[0] = AxisMotionState.TRACKING  # set azimuth to 0
        tma.motion[1] = AxisMotionState.TRACKING  # set azimuth to 0
        self.assertEqual(tma._parts['azimuthMotionState'], AxisMotionState.TRACKING)
        self.assertEqual(tma._parts['elevationMotionState'], AxisMotionState.TRACKING)

    def test_getAxisAndType(self):
        # check both the long and short form names work
        for s in ['azimuthMotionState', 'lsst.sal.MTMount.logevent_azimuthMotionState']:
            self.assertEqual(getAxisAndType(s), ('azimuth', 'MotionState'))

        # check in position, and use elevation instead of azimuth to test that
        for s in ['elevationInPosition', 'lsst.sal.MTMount.logevent_elevationInPosition']:
            self.assertEqual(getAxisAndType(s), ('elevation', 'InPosition'))

        for s in ['azimuthSystemState', 'lsst.sal.MTMount.logevent_azimuthSystemState']:
            self.assertEqual(getAxisAndType(s), ('azimuth', 'SystemState'))

def makeValid(tma):
    """Helper function to turn a TMA into a valid state.
    """
    for name, value in tma._parts.items():
        if value == tma._UNINITIALIZED_VALUE:
            tma._parts[name] = 1


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
