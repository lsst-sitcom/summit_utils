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
import asyncio

from lsst.summit.utils.efdUtils import makeEfdClient
from lsst.summit.utils.blockUtils import (
    BlockParser,
)


# @unittest.skip("Skipping until DM-40101 is resolved.")
class BlockParserTestCase(lsst.utils.tests.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.client = makeEfdClient()
        except RuntimeError:
            raise unittest.SkipTest("Could not instantiate an EFD client")

        cls.dayObs = 20230615
        cls.dayObsNoBlocks = 20230531  # contains data but no blocks
        cls.blockParser = BlockParser(dayObs=cls.dayObs, client=cls.client)
        cls.blockNums = cls.blockParser.getBlockNums()
        cls.blockDict = {}
        for block in cls.blockNums:
            cls.blockDict[block] = cls.blockParser.getSeqNums(block)

    def tearDown(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.client.influx_client.close())

    def test_parsing(self):
        blockNums = self.blockParser.getBlockNums()
        self.assertTrue(all(isinstance(n, int) for n in blockNums))
        self.assertEqual(blockNums, list(self.blockDict.keys()))

        for block, seqNums in self.blockDict.items():
            self.assertIsInstance(block, int)
            self.assertIsInstance(seqNums, list)
            self.assertTrue(all(isinstance(s, int) for s in seqNums))

            found = self.blockParser.getSeqNums(block)
            self.assertTrue(all(isinstance(s, int) for s in found))
            self.assertEqual(found, seqNums)
            self.blockParser.printBlockEvolution(block)

            for seqNum in seqNums:
                data = self.blockParser.getRows(block, seqNum)
                self.assertIsInstance(data, pd.DataFrame)
                self.assertGreater(len(data), 0)
                self.blockParser.getBlockInfo(block=block, seqNum=seqNum)
                self.blockParser.printBlockEvolution(block, seqNum=seqNum)

    def test_notFoundBehavior(self):
        # no block data on this day so check init doesn't raise
        blockParser = BlockParser(dayObs=self.dayObsNoBlocks, client=self.client)
        self.assertIsInstance(blockParser, BlockParser)

        # check the queries which return nothing give nothing back gracefully
        blocks = blockParser.getBlockNums()
        self.assertIsInstance(blocks, list)
        self.assertEqual(len(blocks), 0)

        seqNums = blockParser.getSeqNums(block=123)
        self.assertIsInstance(seqNums, list)
        self.assertEqual(len(seqNums), 0)

        # just check this doesn't raise
        blockParser.getBlockInfo(block=1, seqNum=1)

        # now switch back to one with data, and make sure the same is true
        # when there is data present
        blockParser = self.blockParser
        seqNums = blockParser.getSeqNums(block=9999999)
        self.assertIsInstance(seqNums, list)
        self.assertEqual(len(seqNums), 0)

        # just check this doesn't raise
        blockParser.getBlockInfo(block=9999999, seqNum=9999999)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
