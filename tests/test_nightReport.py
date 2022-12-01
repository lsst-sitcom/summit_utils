# This file is part of summit_extras.
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
import tempfile
from unittest import mock

import lsst.utils.tests

import matplotlib as mpl
mpl.use('Agg')

from lsst.summit.extras.nightReport import NightReporter, loadReport, saveReport  # noqa: E402
import lsst.summit.utils.butlerUtils as butlerUtils  # noqa: E402


class NightReporterTestCase(lsst.utils.tests.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.butler = butlerUtils.makeDefaultLatissButler()
        except FileNotFoundError:
            raise unittest.SkipTest("Skipping tests that require the LATISS butler repo.")

        cls.dayObs = 20200314  # has 377 images and data also exists on the TTS & summit

        # Do the init in setUpClass because this takes about 35s for 20200314
        cls.reporter = NightReporter(cls.dayObs)
        # number of images isn't necessarily the same as the number for the
        # the dayObs in the registry becacuse of the test stands/summit
        # having partial data, so get the number of images from the length
        # of the scraped data. Not ideal, but best that can be done due to
        # only having partial days in the test datasets.
        cls.nImages = len(cls.reporter.data.keys())

    def test_saveAndLoad(self):
        """Test that a NightReporter can save itself, and be loaded back.
        """
        writeDir = tempfile.mkdtemp()
        saveReport(self.reporter, writeDir)
        loaded = loadReport(writeDir, self.dayObs)
        self.assertIsInstance(loaded, lsst.summit.extras.nightReport.NightReporter)
        self.assertGreaterEqual(len(loaded.data), 1)
        self.assertEqual(loaded.dayObs, self.dayObs)

    def test_printObsTable(self):
        """Test that a the printObsTable() method prints out the correct
        number of lines.
        """
        with mock.patch('sys.stdout') as fake_stdout:
            self.reporter.printObsTable()

        # newline for each row plus header
        self.assertEqual(len(fake_stdout.mock_calls), 2*(self.nImages+1))

        tailNumber = 20
        nLines = min(self.nImages, tailNumber)  # test stands have very few images on some days
        with mock.patch('sys.stdout') as fake_stdout:
            self.reporter.printObsTable(tailNumber=tailNumber)
        self.assertEqual(len(fake_stdout.mock_calls), 2*(nLines+1))

    def test_plotPerObjectAirMass(self):
        """Test that a the per-object airmass plots runs.
        """
        # We assume matplotlib is making plots, so just check that these
        # don't crash.

        # Default plotting:
        self.reporter.plotPerObjectAirMass()
        # plot with only one object as a str not a list of str
        self.reporter.plotPerObjectAirMass(objects=self.reporter.stars[0])
        # plot with first two objects as a list
        self.reporter.plotPerObjectAirMass(objects=self.reporter.stars[0:2])
        # flip y axis option
        self.reporter.plotPerObjectAirMass(airmassOneAtTop=True)
        # flip and select stars
        self.reporter.plotPerObjectAirMass(objects=self.reporter.stars[0], airmassOneAtTop=True)  # both

    def test_makePolarPlotForObjects(self):
        """Test that a the polar coverage plotting code runs.
        """
        # We assume matplotlib is making plots, so just check that these
        # don't crash.

        # test the default case
        self.reporter.makePolarPlotForObjects()
        # plot with only one object as a str not a list of str
        self.reporter.makePolarPlotForObjects(objects=self.reporter.stars[0])
        # plot with first two objects as a list
        self.reporter.makePolarPlotForObjects(objects=self.reporter.stars[0:2])
        # test turning lines off
        self.reporter.makePolarPlotForObjects(objects=self.reporter.stars[0:2], withLines=False)

    def test_calcShutterOpenEfficiency(self):
        efficiency = self.reporter.calcShutterOpenEfficiency()
        self.assertGreater(efficiency, 0)
        self.assertLessEqual(efficiency, 1)

    def test_internals(self):
        starsFromGetter = self.reporter.getObservedObjects()
        self.assertIsInstance(starsFromGetter, list)
        self.assertSetEqual(set(starsFromGetter), set(self.reporter.stars))

        # check the internal color map has the right number of items
        self.assertEqual(len(self.reporter.cMap), len(starsFromGetter))


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
