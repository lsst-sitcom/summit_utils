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
import tempfile
import os

import lsst.utils.tests

from lsst.summit.utils.butlerUtils import makeDefaultLatissButler
from lsst.summit.utils.plotting import plot


class PlottingTestCase(lsst.utils.tests.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.butler = makeDefaultLatissButler()
        except FileNotFoundError:
            raise unittest.SkipTest("Skipping tests that require the LATISS butler repo.")

        # Chosen to work on the TTS, summit and NCSA
        cls.dataId = {'day_obs': 20200315, 'seq_num': 120, 'detector': 0}
        cls.outputDir = tempfile.mkdtemp()

    def test_plotting(self):
        """Test that the the plot is made and saved
        """
        exp = self.butler.get('raw', self.dataId)
        centroids = [(567, 746), (576, 599), (678, 989)]

        # Input is an exposure
        outputFilename = os.path.join(self.outputDir, 'testPlotting_exp.jpg')
        plot(exp,
             centroids=[centroids],
             showCompass=True,
             savePlotAs=outputFilename)
        self.assertTrue(os.path.isfile(outputFilename))
        self.assertTrue(os.path.getsize(outputFilename) > 10000)

        # Input is a numpy array
        outputFilename = os.path.join(self.outputDir, 'testPlotting_nparr.jpg')
        nparr = exp.getImage().array
        plot(nparr,
             showCompass=True,
             centroids=[centroids],
             savePlotAs=outputFilename)
        self.assertTrue(os.path.isfile(outputFilename))
        self.assertTrue(os.path.getsize(outputFilename) > 10000)

        # Input is an image
        outputFilename = os.path.join(self.outputDir, 'testPlotting_image.jpg')
        im = exp.getImage()
        plot(im,
             showCompass=True,
             centroids=[centroids],
             savePlotAs=outputFilename)
        self.assertTrue(os.path.isfile(outputFilename))
        self.assertTrue(os.path.getsize(outputFilename) > 10000)

        # Input is a masked image
#       outputFilename = os.path.join(self.outputDir, 'testPloting_mask.jpg')
#        plot(exp,
#             showCompass=True,
#             centroids=[centroids],
#             savePlotAs=outputFilename)
#        plot4.plot()
#        self.assertTrue(os.path.isfile(outputFilename))
#        self.assertTrue(os.path.getsize(outputFilename) > 10000)


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
