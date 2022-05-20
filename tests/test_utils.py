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

import copy
import itertools
import unittest

import astropy.time
import astropy.units as u
import lsst.afw.image as afwImage
import lsst.geom as geom
import lsst.utils.tests
import numpy as np
from astro_metadata_translator import makeObservationInfo
from lsst.obs.base import createInitialSkyWcsFromBoresight
from lsst.obs.base.makeRawVisitInfoViaObsInfo import MakeRawVisitInfoViaObsInfo
from lsst.obs.lsst.latiss import Latiss
from lsst.summit.utils.utils import getExpPositionOffset
from lsst.obs.lsst.translators.latiss import AUXTEL_LOCATION


class ExpSkyPositionOffsetTestCase(lsst.utils.tests.TestCase):
    """A test case for testing sky position offsets for exposures."""

    def setUp(self):
        camera = Latiss.getCamera()
        self.assertTrue(len(camera) == 1)
        self.detector = camera[0]

        self.viMaker = MakeRawVisitInfoViaObsInfo()
        self.mi = afwImage.maskedImage.MaskedImageF(0, 0)
        self.baseHeader = dict(boresight_airmass=1.5,
                               temperature=15*u.deg_C,
                               observation_type="science",
                               exposure_time=5*u.ks,
                               detector_num=32,
                               location=AUXTEL_LOCATION,
                               )

    def test_getExpPositionOffset(self):
        epsilon = 0.0001
        ra1s = [0, 45, 90]
        ra2s = copy.copy(ra1s)
        ra2s.extend([r + epsilon for r in ra1s])
        ra1s = np.deg2rad(ra1s)
        ra2s = np.deg2rad(ra2s)

        epsilon = 0.0001
        dec1s = [0, 45, 90]
        dec2s = copy.copy(dec1s)
        dec2s.extend([d + epsilon for d in dec1s[:-1]])  # skip last point as >90 not allowed for dec

        rotAngle1 = geom.Angle(43.2, geom.degrees)  # arbitrary non-zero
        rotAngle2 = geom.Angle(56.7, geom.degrees)

        t1 = astropy.time.Time("2021-09-15T12:00:00", format="isot", scale="utc")
        t2 = astropy.time.Time("2021-09-15T12:01:00", format="isot", scale="utc")
        expTime = astropy.time.TimeDelta(20, format='sec')

        header1 = copy.copy(self.baseHeader)
        header2 = copy.copy(self.baseHeader)
        header1['datetime_begin'] = astropy.time.Time(t1, format="isot", scale="utc")
        header2['datetime_begin'] = astropy.time.Time(t2, format="isot", scale="utc")

        header1['datetime_end'] = astropy.time.Time(t1+expTime, format="isot", scale="utc")
        header2['datetime_end'] = astropy.time.Time(t2+expTime, format="isot", scale="utc")

        obsInfo1 = makeObservationInfo(**header1)
        obsInfo2 = makeObservationInfo(**header2)

        vi1 = self.viMaker.observationInfo2visitInfo(obsInfo1)
        vi2 = self.viMaker.observationInfo2visitInfo(obsInfo2)
        expInfo1 = afwImage.ExposureInfo()
        expInfo1.setVisitInfo(vi1)
        expInfo2 = afwImage.ExposureInfo()
        expInfo2.setVisitInfo(vi2)

        for ra1, dec1, ra2, dec2 in itertools.product(ra1s, dec1s, ra2s, dec2s):
            pos1 = geom.SpherePoint(ra1, dec1, geom.degrees)
            pos2 = geom.SpherePoint(ra2, dec2, geom.degrees)

            wcs1 = createInitialSkyWcsFromBoresight(pos1, rotAngle1, self.detector, flipX=True)
            wcs2 = createInitialSkyWcsFromBoresight(pos2, rotAngle2, self.detector, flipX=True)

            exp1 = afwImage.ExposureF(self.mi, expInfo1)
            exp2 = afwImage.ExposureF(self.mi, expInfo2)

            exp1.setWcs(wcs1)
            exp2.setWcs(wcs2)

            result = getExpPositionOffset(exp1, exp2)

            deltaRa = ra1 - ra2
            deltaDec = dec1 - dec2

            self.assertAlmostEqual(result.deltaRa.asDegrees(), deltaRa, 6)
            self.assertAlmostEqual(result.deltaDec.asDegrees(), deltaDec, 6)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
