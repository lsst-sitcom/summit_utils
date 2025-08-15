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

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import lsst.utils.tests
from lsst.daf.butler import Butler
from lsst.summit.utils.butlerUtils import makeDefaultButler
from lsst.summit.utils.guiders.plotting import GuiderPlotter
from lsst.summit.utils.guiders.reading import GuiderReader
from lsst.summit.utils.guiders.tracking import GuiderStarTracker
from lsst.summit.utils.utils import getSite


class GuiderTestCase(unittest.TestCase):
    """Tests of the run method with fake data."""

    def setUp(self) -> None:
        try:
            if getSite() == "jenkins":
                raise unittest.SkipTest("Skip running butler-driven tests in Jenkins.")
            self.butler = makeDefaultButler("LSSTCam")
        except FileNotFoundError:
            raise unittest.SkipTest("Skipping tests that require the LSSTCam butler repo.")
        self.assertIsInstance(self.butler, Butler)

        self.dayObs = 20250629
        self.seqNum = 340

        self.reader = GuiderReader(self.butler, view="dvcs")
        self.guiderData = self.reader.get(dayObs=self.dayObs, seqNum=self.seqNum)
        self.tracker = GuiderStarTracker(self.guiderData)
        self.stars = self.tracker.trackGuiderStars(refCatalog=None)
        self.plotter = GuiderPlotter(self.stars, self.guiderData)

    def test_types(self) -> None:
        self.assertIsInstance(self.guiderData.header, dict)
        self.assertIsInstance(self.guiderData.stampsMap, dict)

        expectedKeys = (
            "R00_SG0",
            "R00_SG1",
            "R04_SG0",
            "R04_SG1",
            "R40_SG0",
            "R40_SG1",
            "R44_SG0",
            "R44_SG1",
        )
        self.assertTrue(
            all(key in self.guiderData.stampsMap for key in expectedKeys),
            "Not all expected guider datasets are present in the data.",
        )

        detName = "R00_SG0"
        single = self.guiderData[detName, 0]
        self.assertIsInstance(single, np.ndarray)
        stack = self.guiderData.getStampArrayCoadd(detName=detName)
        self.assertIsInstance(stack, np.ndarray)
        self.assertEqual(single.shape, (400, 400))

        return

    def test_detection(self) -> None:
        self.assertIsInstance(self.stars, pd.DataFrame)
        requiredColumns = (
            "xroi",
            "yroi",
            "dx",
            "dy",
            "dalt",
            "daz",
            "fwhm",
            "trackid",
            "expid",
        )
        self.assertTrue(
            all(col in self.stars.columns for col in requiredColumns),
            "Not all required columns are present in the stars DataFrame.",
        )

        maxStampIndex = max(self.stars["stamp"])
        nStamps = len(self.guiderData)  # we should make an attribute for this

        # we skip the first stamp, so the max index should be nStamps - 1
        self.assertEqual(maxStampIndex, nStamps - 1, "Did not get detections for all expected stamps")

    def testStarMosaicPlotFullView(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            self.plotter.starMosaic(stampNum=-1, cutoutSize=-1, plo=50, phi=98, saveAs=tmp.name)
            os.fsync(tmp.fileno())  # be strict
            size = os.path.getsize(tmp.name)
            self.assertGreater(size, 1000, f"{tmp.name} too small: {size} bytes")

    def testStarMosaicPlotZoomView(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            self.plotter.starMosaic(stampNum=-1, cutoutSize=12, plo=50, phi=98, saveAs=tmp.name)
            os.fsync(tmp.fileno())  # be strict
            size = os.path.getsize(tmp.name)
            self.assertGreater(size, 1000, f"{tmp.name} too small: {size} bytes")

    def testStarMosaicPlotStampZoomView(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            self.plotter.starMosaic(stampNum=4, cutoutSize=12, plo=50, phi=98, saveAs=tmp.name)
            os.fsync(tmp.fileno())  # be strict
            size = os.path.getsize(tmp.name)
            self.assertGreater(size, 1000, f"{tmp.name} too small: {size} bytes")

    def testStripPlotPsf(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            self.plotter.stripPlot(plotType="psf", saveAs=tmp.name)
            os.fsync(tmp.fileno())
            size = os.path.getsize(tmp.name)
            self.assertGreater(size, 1000, f"{tmp.name} too small: {size} bytes")

    def testStripPlotCentroidAltAz(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            self.plotter.stripPlot(plotType="centroidAltAz", saveAs=tmp.name)
            os.fsync(tmp.fileno())
            size = os.path.getsize(tmp.name)
            self.assertGreater(size, 1000, f"{tmp.name} too small: {size} bytes")

    def testStripPlotFlux(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            self.plotter.stripPlot(plotType="flux", saveAs=tmp.name)
            os.fsync(tmp.fileno())
            size = os.path.getsize(tmp.name)
            self.assertGreater(size, 1000, f"{tmp.name} too small: {size} bytes")

    def testStripPlotShape(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            self.plotter.stripPlot(plotType="ellip", saveAs=tmp.name)
            os.fsync(tmp.fileno())
            size = os.path.getsize(tmp.name)
            self.assertGreater(size, 1000, f"{tmp.name} too small: {size} bytes")

    def testMakeGif(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=True) as tmp:
            self.plotter.makeGif(cutoutSize=14, plo=50, phi=98, saveAs=tmp.name)
            os.fsync(tmp.fileno())
            size = os.path.getsize(tmp.name)
            self.assertGreater(size, 1000, f"{tmp.name} too small: {size} bytes")

    def test_plotting(self) -> None:
        # just to check that it runs without error
        self.plotter.printMetrics()

        # Check Star Mosaic Plot
        # Stacked and full stamp size
        self.testStarMosaicPlotFullView()
        # Stacked and zoomed in
        self.testStarMosaicPlotZoomView()
        # Single stamp and zoomed in
        self.testStarMosaicPlotStampZoomView()

        # Check Strip Plots
        self.testStripPlotPsf()
        self.testStripPlotCentroidAltAz()
        self.testStripPlotFlux()
        self.testStripPlotShape()

    def test_animation(self) -> None:
        self.testMakeGif()


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
