import unittest

import lsst.utils.tests
from lsst.summit.utils.coordinates import convertE1E2


class CoordinatesTestCase(lsst.utils.tests.TestCase):
    def test_convertE1E2(self) -> None:
        e1 = [0, 0, 1, 0.5]
        e2 = [0, 0, 0, 0.5]
        e1_out, e2_out = convertE1E2(e1, e2, 0, "ccs", "ocs")
        self.assertEqual(e1, e1_out)
        self.assertEqual(e2, e2_out)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module: object) -> None:
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
