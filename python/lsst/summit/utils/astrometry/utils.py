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

import numpy as np

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import AltAz, EarthLocation, SkyCoord

from lsst.afw.geom import SkyWcs
from lsst.daf.base import PropertySet
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask, CharacterizeImageConfig

__all__ = [
    'claverHeaderToWcs',
    'getAverageRaFromHeader',
    'getAverageDecFromHeader',
    'getAverageAzFromHeader',
    'getAverageElFromHeader',
    'genericCameraHeaderToWcs',
    'getIcrsAtZenith',
    'headerToWcs',
    'runCharactierizeImage',
    'filterSourceCatOnBrightest',
]


def claverHeaderToWcs(exp, nominalRa=None, nominalDec=None):
    """Given an exposure taken by Chuck Claver at his house, construct a wcs
    with the ra/dec set to zenith unless a better guess is supplied.

    Automatically sets the platescale depending on the lens.

    Parameters
    ----------
    exp : `lsst.afw.image.Exposure`
        The exposure to construct the wcs for.
    nominalRa : `float`, optional
        The guess for the ra.
    nominalDec : `float`, optional
        The guess for the Dec.

    Returns
    -------
    wcs : `lsst.afw.geom.SkyWcs`
        The constructed wcs.
    """
    header = exp.getMetadata().toDict()

    # set the plate scale depending on the lens and put into CD matrix
    # plate scale info from:
    # https://confluence.lsstcorp.org/pages/viewpage.action?pageId=191987725
    lens = header['INSTLEN']
    if '135mm' in lens:
        arcSecPerPix = 8.64
    elif '375mm' in lens:
        arcSecPerPix = 3.11
    elif '750mm' in lens:
        arcSecPerPix = 1.56
    else:
        raise ValueError(f'Unrecognised lens: {lens}')

    header['CD1_1'] = arcSecPerPix / 3600
    header['CD1_2'] = 0
    header['CD2_1'] = 0
    header['CD2_2'] = arcSecPerPix / 3600

    # calculate the ra/dec at zenith and assume Chuck pointed it vertically
    icrs = getIcrsAtZenith(float(header['OBSLON']),
                           float(header['OBSLAT']),
                           float(header['OBSHGT']),
                           header['UTC'])
    header['CRVAL1'] = nominalRa if nominalRa else icrs.ra.degree
    header['CRVAL2'] = nominalDec if nominalDec else icrs.dec.degree

    # just use the nomimal chip centre, not that it matters
    # given radec = zenith
    width, height = exp.image.array.shape
    header['CRPIX1'] = width/2
    header['CRPIX2'] = height/2

    header['CTYPE1'] = 'RA---TAN-SIP'
    header['CTYPE2'] = 'DEC--TAN-SIP'

    wcsPropSet = PropertySet.from_mapping(header)
    wcs = SkyWcs(wcsPropSet)
    return wcs


# don't be tempted to get cute and try to combine these 4 functions. It would
# be easy to do but it's not unlikley they will diverge in the future.
def getAverageRaFromHeader(header):
    raStart = header.get('RASTART')
    raEnd = header.get('RAEND')
    if not raStart or not raEnd:
        raise RuntimeError(f'Failed to get RA from header due to missing RASTART/END {raStart} {raEnd}')
    raStart = float(raStart)
    raEnd = float(raEnd)
    return (raStart + raEnd) / 2


def getAverageDecFromHeader(header):
    decStart = header.get('DECSTART')
    decEnd = header.get('DECEND')
    if not decStart or not decEnd:
        raise RuntimeError(f'Failed to get DEC from header due to missing DECSTART/END {decStart} {decEnd}')
    decStart = float(decStart)
    decEnd = float(decEnd)
    return (decStart + decEnd) / 2


def getAverageAzFromHeader(header):
    azStart = header.get('AZSTART')
    azEnd = header.get('AZEND')
    if not azStart or not azEnd:
        raise RuntimeError(f'Failed to get az from header due to missing AZSTART/END {azStart} {azEnd}')
    azStart = float(azStart)
    azEnd = float(azEnd)
    return (azStart + azEnd) / 2


def getAverageElFromHeader(header):
    elStart = header.get('ELSTART')
    elEnd = header.get('ELEND')
    if not elStart or not elEnd:
        raise RuntimeError(f'Failed to get el from header due to missing ELSTART/END {elStart} {elEnd}')
    elStart = float(elStart)
    elEnd = float(elEnd)
    return (elStart + elEnd) / 2


def patchHeader(header):
    """This is a TEMPORARY function to patch some info into the headers.
    """
    if header.get('SECPIX') == '3.11':
        # the narrow camera currently is wrong about its place scale by of ~2.2
        header['SECPIX'] = '1.44'
        # update the boresight locations until this goes into the header
        # service
        header['CRPIX1'] = 1898.10
        header['CRPIX2'] = 998.47
    if header.get('SECPIX') == '8.64':
        # update the boresight locations until this goes into the header
        # service
        header['CRPIX1'] = 1560.85
        header['CRPIX2'] = 1257.15
    if header.get('SECPIX') == '0.67':
        # use the fast camera chip centre until we know better
        header['CRPIX1'] = 329.5
        header['CRPIX2'] = 246.5
    return header


def genericCameraHeaderToWcs(exp):
    header = exp.getMetadata().toDict()
    header = patchHeader(header)

    header['CTYPE1'] = 'RA---TAN-SIP'
    header['CTYPE2'] = 'DEC--TAN-SIP'

    header['CRVAL1'] = getAverageRaFromHeader(header)
    header['CRVAL2'] = getAverageDecFromHeader(header)

    plateScale = header.get('SECPIX')
    if not plateScale:
        raise RuntimeError('Failed to find platescale in header')
    plateScale = float(plateScale)

    header['CD1_1'] = plateScale / 3600
    header['CD1_2'] = 0
    header['CD2_1'] = 0
    header['CD2_2'] = plateScale / 3600

    wcsPropSet = PropertySet.from_mapping(header)
    wcs = SkyWcs(wcsPropSet)
    return wcs


def getIcrsAtZenith(lon, lat, height, utc):
    """Get the icrs at zenith given a lat/long/height/time in UTC.

    Parameters
    ----------
    lon : `float`
        The longitude, in degrees.
    lat : `float`
        The latitude, in degrees.
    height : `float`
        The height above sea level in meters.
    utc : `str`
        The time in UTC as an ISO string, e.g. '2022-05-27 20:41:02'

    Returns
    -------
    skyCoordAtZenith : `astropy.coordinates.SkyCoord`
        The skyCoord at zenith.
    """
    location = EarthLocation.from_geodetic(lon=lon*u.deg,
                                           lat=lat*u.deg,
                                           height=height)
    obsTime = Time(utc, format='iso', scale='utc')
    skyCoord = SkyCoord(AltAz(obstime=obsTime,
                              alt=90.0*u.deg,
                              az=180.0*u.deg,
                              location=location))
    return skyCoord.transform_to('icrs')


def headerToWcs(header):
    """Convert an astrometry.net wcs header dict to a DM wcs object.

    Parameters
    ----------
    header : `dict`
        The wcs header, as returned from from the astrometry_net fit.

    Returns
    -------
    wcs : `lsst.afw.geom.SkyWcs`
        The wcs.
    """
    wcsPropSet = PropertySet.from_mapping(header)
    return SkyWcs(wcsPropSet)


def runCharactierizeImage(exp, snr, minPix):
    """Run the image characterization task, finding only bright sources.

    Parameters
    ----------
    exp : `lsst.afw.image.Exposure`
        The exposure to characterize.
    snr : `float`
        The SNR threshold for detection.
    minPix : `int`
        The minimum number of pixels to count as a source.

    Returns
    -------
    result : `lsst.pipe.base.Struct`
        The result from the image characterization task.
    """
    charConfig = CharacterizeImageConfig()
    charConfig.doMeasurePsf = False
    charConfig.doApCorr = False
    charConfig.doDeblend = False
    charConfig.repair.doCosmicRay = False

    charConfig.detection.minPixels = minPix
    charConfig.detection.thresholdValue = snr
    charConfig.detection.includeThresholdMultiplier = 1

    # fit background with the most simple thing possible as we don't need
    # much sophistication here. weighting=False is *required* for very
    # large binSizes.
    charConfig.background.algorithm = 'CONSTANT'
    charConfig.background.approxOrderX = 0
    charConfig.background.approxOrderY = -1
    charConfig.background.binSize = max(exp.getWidth(), exp.getHeight())
    charConfig.background.weighting = False

    # set this to use all the same minimal settings as those above
    charConfig.detection.background = charConfig.background

    charTask = CharacterizeImageTask(config=charConfig)

    charResult = charTask.run(exp)
    return charResult


def filterSourceCatOnBrightest(catalog, brightFraction, minSources=15, maxSources=200,
                               flux_field="base_CircularApertureFlux_3_0_instFlux"):
    """Filter a sourceCat on the brightness, leaving only the top fraction.

    Return a catalog containing the brightest sources in the input. Makes an
    initial coarse cut, keeping those above 0.1% of the maximum finite flux,
    and then returning the specified fraction of the remaining sources,
    or minSources, whichever is greater.

    Parameters
    ----------
    catalog : `lsst.afw.table.SourceCatalog`
        Catalog to be filtered.
    brightFraction : `float`
        Return this fraction of the brightest sources.
    minSources : `int`, optional
        Always return at least this many sources.
    maxSources : `int`, optional
        Never return more than this many sources.
    flux_field : `str`, optional
        Name of flux field to filter on.

    Returns
    -------
    result : `lsst.afw.table.SourceCatalog`
        Brightest sources in the input image, in ascending order of brightness.
    """
    assert minSources > 0
    assert brightFraction > 0 and brightFraction <= 1
    if not maxSources >= minSources:
        raise ValueError('maxSources must be greater than or equal to minSources, got '
                         f'{maxSources=}, {minSources=}')

    maxFlux = np.nanmax(catalog[flux_field])
    result = catalog.subset(catalog[flux_field] > maxFlux * 0.001)

    print(f"Catalog had {len(catalog)} sources, of which {len(result)} were above 0.1% of max")

    item = catalog.schema.find(flux_field)
    result = catalog.copy(deep=True)  # sort() is in place; copy so we don't modify the original
    result.sort(item.key)
    result = result.copy(deep=True)  # make it memory contiguous
    end = int(np.ceil(len(result)*brightFraction))
    return result[-min(maxSources, max(end, minSources)):]
