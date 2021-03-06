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

import lsst.utils.tests

from lsst.summit.utils.bestEffort import BestEffortIsr
import lsst.afw.image as afwImage


class BestEffortIsrTestCase(lsst.utils.tests.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.bestEffortIsr = BestEffortIsr()
        except FileNotFoundError:
            raise unittest.SkipTest("Skipping tests that require the LATISS butler repo.")

        # chosen as this is available in the following locations - collections:
        # NCSA - LATISS/raw/all
        # TTS - LATISS-test-data-tts
        # summit - LATISS_test_data
        cls.dataId = {'day_obs': 20210121, 'seq_num': 743, 'detector': 0}

    def test_getExposure(self):
        exp = self.bestEffortIsr.getExposure(self.dataId)
        self.assertIsInstance(exp, afwImage.Exposure)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
