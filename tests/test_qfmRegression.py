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
import os

import lsst.utils
from lsst.summit.utils.qfmRegressionTools import (loadResults,
                                                  compareResults,
                                                  DataIdClass,
                                                  CentroidClass
                                                  )


class QfmRegressionTestCase(lsst.utils.tests.TestCase):
    """A test case for testing the QFM regression test code."""

    def setUp(self):
        packageDir = lsst.utils.getPackageDir('summit_utils')
        testDataDir = os.path.join(packageDir, 'tests', 'data')
        self.refFilename = os.path.join(testDataDir, 'test_data_reference.txt')
        self.compFilename = os.path.join(testDataDir, 'test_data_comparison.txt')

    def test_loadResults(self):
        # test basic instances in reference file
        refs = loadResults(self.refFilename)
        self.assertEqual(len(refs), 4)
        self.assertIsInstance(refs, dict)
        dataIds = list(refs.keys())
        centroids = list(refs.values())
        self.assertIsInstance(dataIds[0], DataIdClass)
        self.assertIsInstance(centroids[0], CentroidClass)

        # test filtering by instrument
        refs = loadResults(self.refFilename, instrument='LATISS')
        self.assertEqual(len(refs), 3)

        refs = loadResults(self.refFilename, instrument='LSSTComCam')
        self.assertEqual(len(refs), 1)

        # test comparison file loading
        refs = loadResults(self.compFilename)
        self.assertEqual(len(refs), 3)
        self.assertIsInstance(refs, dict)
        dataIds = list(refs.keys())
        centroids = list(refs.values())
        self.assertIsInstance(dataIds[0], DataIdClass)
        self.assertIsInstance(centroids[0], CentroidClass)

    def test_compareResults(self):
        moved = compareResults(self.refFilename, self.compFilename, tolerance=.1)
        self.assertGreaterEqual(len(moved), 1)
        for (dataId, distance) in moved:
            self.assertIsInstance(dataId, DataIdClass)
            self.assertIsInstance(distance, float)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
