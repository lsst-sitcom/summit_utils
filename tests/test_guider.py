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

import numpy as np
import pandas as pd

import lsst.utils.tests
from lsst.daf.butler import Butler
from lsst.summit.utils.butlerUtils import makeDefaultButler
from lsst.summit.utils.guiders.detection import GuiderStarTracker
from lsst.summit.utils.guiders.plotting import GuiderPlotter
from lsst.summit.utils.guiders.reading import GuiderReader
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
        self.tracker = GuiderStarTracker(self.guiderData, min_snr=20, max_ellipticity=0.15, edge_margin=40)
        self.stars = self.tracker.track_guider_stars(ref_catalog=None)

    def test_types(self) -> None:
        self.assertIsInstance(self.guiderData.header, dict)
        self.assertIsInstance(self.guiderData.datasets, dict)

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
            all(key in self.guiderData.datasets for key in expectedKeys),
            "Not all expected guider datasets are present in the data.",
        )

        detName = "R00_SG0"
        single = self.guiderData.getStampArray(stampNum=0, detName=detName)
        self.assertIsInstance(single, np.ndarray)
        stack = self.guiderData.getStackedStampArray(detName=detName)
        self.assertIsInstance(stack, np.ndarray)
        self.assertEqual(single.shape, (400, 400))

        return

    def test_detection(self):
        self.assertIsInstance(self.stars, pd.DataFrame)
        requiredColumns = ("xroi", "yroi", "dx", "dy", "dalt", "daz", "fwhm", "trackid", "expid")
        self.assertTrue(
            all(col in self.stars.columns for col in requiredColumns),
            "Not all required columns are present in the stars DataFrame.",
        )

        maxStampIndex = max(self.stars["stamp"])
        nStamps = len(self.guiderData.timestamps)  # we should make an attribute for this
        # we skip the first stamp, so the max index should be nStamps - 1
        self.assertEqual(maxStampIndex, nStamps - 1, "Did not get detections for all expected stamps")

    def test_plotting(self):
        plotter = GuiderPlotter(self.stars, self.guiderData, isIsr=True)
        plotter.print_metrics()  # just to check that it runs without error

        # TODO: add saving code and check the file size
        # check the -1 plotting option
        _ = plotter.star_mosaic(stamp_num=-1, cutout_size=-1, plo=50, phi=98)

        # check zooming in works
        _ = plotter.star_mosaic(stamp_num=-1, cutout_size=30, plo=50, phi=98)

        # TODO: add saving code and check the file size
        _ = plotter.strip_plot()

        _ = plotter.strip_plot(plot_type="centroidPixel")

        _ = plotter.strip_plot(plot_type="flux")

        _ = plotter.strip_plot(plot_type="psf")

    def test_animation(self):
        plotter = GuiderPlotter(self.stars, self.guiderData, isIsr=True)

        # TODO: add saving code and check the file size
        plotter.make_gif(n_stamp_max=50, cutout_size=14, plo=50, phi=98)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
