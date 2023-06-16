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

import pandas as pd
import numpy as np
import asyncio

from lsst.summit.utils.efdUtils import makeEfdClient
from lsst.summit.utils.tmaUtils import (TMA,
                                        TMAEvent,
                                        TMAEventMaker,
                                        TMAState,
                                        AxisMotionState,
                                        PowerState,
                                        getAxisAndType,
                                        _makeValid,  # move definition into here to discourage elsewhere?
                                        _turnOn,  # move definition into here to discourage elsewhere?
                                        )


def makeValid(tma):
    """Helper function to turn a TMA into a valid state.
    """
    for name, value in tma._parts.items():
        if value == tma._UNINITIALIZED_VALUE:
            tma._parts[name] = 1


class TmatilsTestCase(lsst.utils.tests.TestCase):

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

    def test_initStateLogic(self):
        tma = TMA()
        self.assertFalse(tma._isValid)
        self.assertFalse(tma.isMoving)
        self.assertFalse(tma.canMove)
        self.assertEqual(tma.state, TMAState.UNINITIALIZED)

        _makeValid(tma)  # we're valid, but still aren't moving and can't
        self.assertTrue(tma._isValid)
        self.assertNotEqual(tma.state, TMAState.UNINITIALIZED)
        self.assertFalse(tma.isMoving)
        self.assertFalse(tma.canMove)

        _turnOn(tma)  # can now move, still valid, but not in motion
        self.assertTrue(tma._isValid)
        self.assertTrue(tma.canMove)
        self.assertFalse(tma.isMoving)


class TMAEventMakerTestCase(lsst.utils.tests.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.client = makeEfdClient()
        except RuntimeError:
            raise unittest.SkipTest("Could not instantiate an EFD client")

        cls.dayObs = 20230601
        # get a sample expRecord here to test expRecordToTimespan
        cls.tmaEventMaker = TMAEventMaker(cls.client)
        cls.events = cls.tmaEventMaker.getEvents(cls.dayObs)  # does the fetch
        cls.sampleData = cls.tmaEventMaker._data[cls.dayObs]  # pull the data from the object and test length

    def tearDown(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.client.influx_client.close())

    def test_events(self):
        data = self.sampleData
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 993)

    def test_rowDataForValues(self):
        rowsFor = set(self.sampleData['rowFor'])
        self.assertEqual(len(rowsFor), 6)

        # hard coding these ensures that you can't extend the axes/model
        # without being explicit about it here.
        correct = {'azimuthInPosition',
                   'azimuthMotionState',
                   'azimuthSystemState',
                   'elevationInPosition',
                   'elevationMotionState',
                   'elevationSystemState'}
        self.assertSetEqual(rowsFor, correct)

    def test_monotonicTimeInDataframe(self):
        # ensure that each row is later than the previous
        times = self.sampleData['private_sndStamp']
        self.assertTrue(np.all(np.diff(times) > 0))

    def test_monotonicTimeApplicationOfRows(self):
        # ensure you can apply rows in the correct order
        tma = TMA()
        row1 = self.sampleData.iloc[0]
        row2 = self.sampleData.iloc[1]

        # just running this check it is OK
        tma.apply(row1)
        tma.apply(row2)

        # and that if you apply them in reverse order then things will raise
        tma = TMA()
        with self.assertRaises(ValueError):
            tma.apply(row2)
            tma.apply(row1)

    def test_fullDaySequence(self):
        # make sure we can apply all the data from the day without falling
        # through the logic sieve
        for engineering in (True, False):
            tma = TMA(engineeringMode=engineering)

            _makeValid(tma)  # XXX at some point this should be removed

            for rowNum, row in self.sampleData.iterrows():
                tma.apply(row)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
