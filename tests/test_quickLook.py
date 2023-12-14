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

import lsst.afw.image as afwImage
import lsst.ip.isr.isrMock as isrMock
import lsst.ip.isr as ipIsr
import lsst.pex.exceptions
import lsst.pipe.base as pipeBase
from lsst.summit.utils.quickLook import QuickLookIsrTask, QuickLookIsrTaskConfig


class QuickLookIsrTaskTestCase(unittest.TestCase):
    def setUp(self):
        self.mockConfig = isrMock.IsrMockConfig()
        self.dataContainer = isrMock.MockDataContainer(config=self.mockConfig)
        self.camera = isrMock.IsrMock(config=self.mockConfig).getCamera()

        self.ccdExposure = isrMock.RawMock(config=self.mockConfig).run()
        detector = self.ccdExposure.getDetector()
        amps = detector.getAmplifiers()
        ampNames = [amp.getName() for amp in amps]

        # # Mock other optional parameters
        self.bias = self.dataContainer.get("bias")
        self.dark = self.dataContainer.get("dark")
        self.flat = self.dataContainer.get("flat")
        self.defects = self.dataContainer.get("defects")
        self.ptc = ipIsr.PhotonTransferCurveDataset(ampNames=ampNames)  # Mock PTC dataset
        self.bfKernel = self.dataContainer.get("bfKernel")
        self.bfGains = {}
        for amp_i, amp in enumerate(ampNames):
            self.bfGains[amp] = 0.9 + 0.1*amp_i
        self.task = QuickLookIsrTask(config=QuickLookIsrTaskConfig())

    def test_runQuickLook(self):
        # Execute the run method with the mock data
        result = self.task.run(self.ccdExposure,
                               camera=self.camera,
                               bias=self.bias,
                               dark=self.dark,
                               flat=self.flat,
                               defects=self.defects,
                               linearizer=None,
                               crosstalk=None,
                               bfKernel=self.bfKernel,
                               bfGains=self.bfGains,
                               ptc=self.ptc,
                               crosstalkSources=None
                               )
        self.assertIsNotNone(result, "Result of run method should not be None")
        self.assertIsInstance(result, pipeBase.Struct, "Result should be of type lsst.pipe.base.Struct")
        self.assertIsInstance(result.exposure, afwImage.Exposure,
                              "Resulting exposure should be an instance of lsst.afw.image.Exposure")

    def test_runQuickLookMissingData(self):
        # Test without any inputs other than the exposure
        result = self.task.run(self.ccdExposure)
        self.assertIsInstance(result.exposure, afwImage.Exposure)

    def test_runQuickLookBadDark(self):
        # Test with an incorrect dark frame
        bbox = self.ccdExposure.getBBox()
        bbox.grow(-20)
        with self.assertRaises(lsst.pex.exceptions.wrappers.LengthError):
            self.task.run(self.ccdExposure, camera=self.camera, bias=self.bias,
                          dark=self.dark[bbox], flat=self.flat, defects=self.defects)
