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

import tempfile
import unittest

import lsst.afw.image as afwImage
import lsst.daf.butler.tests as butlerTests
import lsst.ip.isr.isrMock as isrMock
import lsst.ip.isr as ipIsr
import lsst.pex.exceptions
import lsst.pipe.base as pipeBase
import lsst.pipe.base.testUtils
from lsst.summit.utils.quickLook import QuickLookIsrTask, QuickLookIsrTaskConfig


class QuickLookIsrTaskTestCase(unittest.TestCase):
    def setUp(self):
        self.mockConfig = isrMock.IsrMockConfig()
        self.dataContainer = isrMock.MockDataContainer(config=self.mockConfig)
        self.camera = isrMock.IsrMock(config=self.mockConfig).getCamera()

        self.ccdExposure = isrMock.RawMock(config=self.mockConfig).run()
        self.detector = self.ccdExposure.getDetector()
        amps = self.detector.getAmplifiers()
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


class CalibrateImageTaskRunQuantumTests(lsst.utils.tests.TestCase):
    """Tests of ``CalibrateImageTask.runQuantum``, which need a test butler,
    but do not need real images.

    Adapted from the unit tests of ``CalibrateImageTask.runQuantum``
    """
    def setUp(self):
        QuickLookIsrTaskTestCase.setUp(self)
        instrument = "testCam"
        exposure = 100
        visit = 100101
        detector = self.detector.getId()
        physical_filter = "testCam_filter"

        # Create a and populate a test butler for runQuantum tests.
        self.repo_path = tempfile.TemporaryDirectory()
        self.repo = butlerTests.makeTestRepo(self.repo_path.name)

        # dataIds for fake data
        butlerTests.addDataIdValue(self.repo, "instrument", instrument)
        butlerTests.addDataIdValue(self.repo, "exposure", exposure)
        butlerTests.addDataIdValue(self.repo, "physical_filter", physical_filter)
        butlerTests.addDataIdValue(self.repo, "detector", detector)
        butlerTests.addDataIdValue(self.repo, "visit", visit)

        # inputs
        butlerTests.addDatasetType(self.repo, "ccdExposure", {"instrument", "exposure", "detector"},
                                   "Exposure")
        butlerTests.addDatasetType(self.repo, "camera", {"instrument"}, "Camera")
        butlerTests.addDatasetType(self.repo, "bias", {"instrument", "detector"}, "Exposure")
        butlerTests.addDatasetType(self.repo, "dark", {"instrument", "detector"}, "Exposure")
        butlerTests.addDatasetType(self.repo, "flat", {"instrument", "physical_filter", "detector"},
                                   "Exposure")
        butlerTests.addDatasetType(self.repo, "defects", {"instrument", "detector"}, "Defects")
        butlerTests.addDatasetType(self.repo, "linearizer", {"instrument", "detector"}, "Linearizer")
        butlerTests.addDatasetType(self.repo, "crosstalk", {"instrument", "detector"}, "CrosstalkCalib")
        butlerTests.addDatasetType(self.repo, "bfKernel", {"instrument"}, "NumpyArray")
        butlerTests.addDatasetType(self.repo, "newBFKernel", {"instrument", "detector"},
                                   "BrighterFatterKernel")
        butlerTests.addDatasetType(self.repo, "ptc", {"instrument", "detector"}, "PhotonTransferCurveDataset")
        butlerTests.addDatasetType(self.repo, "crosstalkSources", {"instrument", "exposure", "detector"},
                                   "Exposure")

        # dataIds
        self.exposure_id = self.repo.registry.expandDataId(
            {"instrument": instrument, "exposure": exposure, "detector": detector})
        self.instrument_id = self.repo.registry.expandDataId(
            {"instrument": instrument})
        self.flat_id = self.repo.registry.expandDataId(
            {"instrument": instrument, "physical_filter": physical_filter, "detector": detector})
        self.detector_id = self.repo.registry.expandDataId(
            {"instrument": instrument, "detector": detector})
        self.visit_id = self.repo.registry.expandDataId(
            {"instrument": instrument, "visit": visit, "detector": detector})

        # put empty data
        self.butler = butlerTests.makeTestCollection(self.repo)
        self.butler.put(self.ccdExposure, "ccdExposure", self.exposure_id)
        self.butler.put(self.camera, "camera", self.instrument_id)
        self.butler.put(self.bias, "bias", self.detector_id)
        self.butler.put(self.dark, "dark", self.detector_id)
        self.butler.put(self.flat, "flat", self.flat_id)
        self.butler.put(self.defects, "defects", self.detector_id)
        # self.butler.put(self.ccdExposure, "linearizer", self.detector_id)
        # self.butler.put(self.ccdExposure, "crosstalk", self.detector_id)
        self.butler.put(self.bfKernel, "bfKernel", self.instrument_id)
        # self.butler.put(self.bfGains, "newBFKernel", self.detector_id)
        self.butler.put(self.ptc, "ptc", self.detector_id)
        # self.butler.put(self.ccdExposure, "crosstalkSources", self.exposure_id)

    def tearDown(self):
        del self.repo_path  # this removes the temporary directory

    def test_runQuantum(self):
        task = self.task
        lsst.pipe.base.testUtils.assertValidInitOutput(task)

        quantum = lsst.pipe.base.testUtils.makeQuantum(
            task, self.butler, self.exposure_id,
            {"blah": [self.exposure_id],
             "camera": [self.instrument_id],
             "bias": [self.detector_id],
             "dark": [self.detector_id],
             "flat": [self.flat_id],
             "defects": [self.detector_id],
             "bfKernel": [self.instrument_id],
             # "newBFKernel": [self.detector_id],
             "ptc": [self.detector_id],
             # outputs
             })
        mock_run = lsst.pipe.base.testUtils.runTestQuantum(task, self.butler, quantum)
        # Check that the proper kwargs are passed to run().
        self.assertEqual(mock_run.call_args.kwargs.keys(), {"exposures"})
