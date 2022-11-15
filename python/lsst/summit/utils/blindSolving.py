# This file is part of rapid_analysis.
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
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.coordinates import Angle
import astropy.units as u
from astroquery.astrometry_net import AstrometryNet
from astropy.time import Time
from astropy.coordinates import AltAz, EarthLocation, SkyCoord

import lsst.geom as geom
from lsst.afw.geom import SkyWcs
from lsst.daf.base import PropertySet
from lsst.pipe.tasks.characterizeImage import CharacterizeImageTask, CharacterizeImageConfig

from lsst.obs.lsst.translators.latiss import AUXTEL_LOCATION
from lsst.summit.utils.utils import quickSmooth


def getApiKey():
    """Get the astrometry.net API key if possible.

    Raises a RuntimeError if it isn't found.

    Returns
    -------
    apiKey : str
        The astrometry.net API key, if present.

    Raises
    ------
    RuntimeError
        Raised if the ASTROMETRY_NET_API_KEY is not set.
    """
    try:
        return os.environ['ASTROMETRY_NET_API_KEY']
    except KeyError as e:
        msg = "No AstrometryNet API key found. Sign up and get one, set it to $ASTROMETRY_NET_API_KEY"
        raise RuntimeError(msg) from e


def runImchar(exp, snr, minPix):
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
    charConfig.repair.doInterpolate = True
    charConfig.detection.minPixels = minPix
    charConfig.detection.thresholdValue = snr

    charTask = CharacterizeImageTask(config=charConfig)

    charResult = charTask.run(exp)
    return charResult


def plot(exp, icSrc=None, filteredSources=None, saveAs=None):
    """Plot an exposure, overlaying the selected sources and compass arrows.

    Plots the exposure on a logNorm scale, with the brightest sources, as
    selected by the configuration, overlaid with an x. Plots compass arrows
    for both north/east and az/el. Optionally saves the output to a file is
    ``saveAs`` is supplied.

    Parameters
    ----------
    exp : `lsst.afw.image.Exposure`
        The exposure to get the astrometry for.
    icSrc : `lsst.afw.table.SourceCatalog`
        The source catalog for the exposure.
    filteredSources : `lsst.afw.table.SourceCatalog`, optional
        The filtered source catalog. If supplied, shows which sources were
        selected.
    saveAs : `str`, optional
        Saves the plot to this filename if specified.
    """
    plt.figure(figsize=(16, 16))
    arr = exp.image.array
    clipVal = 1
    arr = np.clip(arr, clipVal, 1000000)  # This image has some negative values, and this removes them
    arr = quickSmooth(arr)
    plt.imshow(np.arcsinh(arr)/10,
               interpolation='None', cmap='gray', origin='lower')

    height, width = exp.image.array.shape
    leftFraction = .15  # fraction into the image to start the N/E compass
    rightFraction = .225  # fraction into the image to start the az/el compass
    fontsize = 20  # for the compass labels
    compassSize = 500
    textDistance = 650
    compassCenter = (leftFraction*width, leftFraction*height)
    compassAzElCent = ((1-rightFraction)*width, rightFraction*height)

    vi = exp.getInfo().getVisitInfo()
    az, _ = vi.boresightAzAlt
    _, dec = vi.boresightRaDec
    rotpa = vi.boresightRotAngle

    az = Angle(az.asDegrees(), u.deg)
    dec = Angle(dec.asDegrees(), u.deg)
    rotpa = Angle(rotpa.asDegrees(), u.deg)

    if icSrc:
        plt.scatter(icSrc['base_SdssCentroid_x'], icSrc['base_SdssCentroid_y'], color='red', marker='x')
    if filteredSources:
        markerStyle = dict(marker='o', linestyle='', markersize=20, linewidth=10, color='green',
                           markeredgecolor='green', fillstyle='none')
        plt.plot(filteredSources['base_SdssCentroid_x'],
                 filteredSources['base_SdssCentroid_y'],
                 **markerStyle)
    plt.arrow(compassCenter[0],
              compassCenter[1],
              -compassSize*np.sin(rotpa),
              compassSize*np.cos(rotpa),
              color='green', width=20)
    plt.text(compassCenter[0]-textDistance*np.sin(rotpa),
             compassCenter[1]+textDistance*np.cos(rotpa),
             'N',
             color='green', fontsize=fontsize, weight='bold')
    plt.arrow(compassCenter[0],
              compassCenter[1],
              compassSize*np.cos(rotpa),
              compassSize*np.sin(rotpa),
              color='green', width=20)
    plt.text(compassCenter[0]+textDistance*np.cos(rotpa),
             compassCenter[1]+textDistance*np.sin(rotpa),
             'E',
             color='green', fontsize=fontsize, weight='bold')

    sinTheta = np.cos(AUXTEL_LOCATION.lat) / np.cos(dec) * np.sin(az)
    theta = Angle(np.arcsin(sinTheta))
    rotAzEl = rotpa - theta - Angle(90.0 * u.deg)
    plt.arrow(compassAzElCent[0],
              compassAzElCent[1],
              -compassSize*np.sin(rotAzEl),
              compassSize*np.cos(rotAzEl),
              color='cyan', width=20)
    plt.text(compassAzElCent[0]-textDistance*np.sin(rotAzEl),
             compassAzElCent[1]+textDistance*np.cos(rotAzEl),
             'EL',
             color='cyan', fontsize=fontsize, weight='bold')
    plt.arrow(compassAzElCent[0],
              compassAzElCent[1],
              compassSize*np.cos(rotAzEl),
              compassSize*np.sin(rotAzEl),
              color='cyan', width=20)
    plt.text(compassAzElCent[0]+textDistance*np.cos(rotAzEl),
             compassAzElCent[1]+textDistance*np.sin(rotAzEl),
             'AZ',
             color='cyan', fontsize=fontsize, weight='bold')

    plt.ylim(0, height)
    plt.tight_layout()

    if saveAs:
        plt.savefig(saveAs)
    plt.show()


def _filterSourceCatalog(srcCat, brightSourceFraction, minSources=15):
    maxFlux = np.nanmax(srcCat['base_CircularApertureFlux_3_0_instFlux'])
    selectBrightestSource = srcCat['base_CircularApertureFlux_3_0_instFlux'] > maxFlux * 0.99
    brightestSource = srcCat.subset(selectBrightestSource)
    brightestCentroid = (brightestSource['base_SdssCentroid_x'][0],
                         brightestSource['base_SdssCentroid_y'][0])
    filteredCatalog = srcCat.subset(srcCat['base_CircularApertureFlux_3_0_instFlux'] > maxFlux * 0.001)

    # TODO: make these into log messages?
    print(f"Found {len(srcCat)} sources, {len(filteredCatalog)} bright sources")
    print(f"Brightest centroid at {brightestCentroid}")

    if not filteredCatalog.isContiguous():
        filteredCatalog = filteredCatalog.copy(deep=True)
    sources = filteredCatalog.asAstropy()
    sources.keep_columns(['base_SdssCentroid_x',
                          'base_SdssCentroid_y',
                          'base_CircularApertureFlux_3_0_instFlux'])
    sources.sort('base_CircularApertureFlux_3_0_instFlux', reverse=True)

    nSources = len(sources)
    if nSources <= minSources:
        return sources

    startPos = 0
    endPos = math.ceil(nSources*brightSourceFraction)
    sources.remove_rows([i for i in range(startPos, endPos+1)])
    return sources


def blindSolve(exp, *,
               radiusInDegrees=5,
               brightSourceFraction=0.1,
               snr=5,
               minPix=25,
               scaleEstimate=None,
               doPlot=False,
               savePlotAs=None):
    """Perform a blind astrometric solution for an image using astrometry.net

    Parameters
    ----------
    exp : `lsst.afw.image.Exposure`
        The exposure to get the astrometry for.
    radiusInDegrees : ``, optional
        The maximum radius to search out to from the nominal wcs.
    brightSourceFraction : ``, optional
        Take only this top-most fraction of sources from the image.
    snr : `float`
        The SNR threshold for detection.
    minPix : `int`
        The minimum number of pixels to count as a source.
    scaleEstimate : ``, optional
        An estimate of the platescale to use, in arcseconds per pixel. It is
        always used if supplied, but if not, the nominal platescale from the
        wcs is taken.
    doPlot : `bool`, optional
        Rending the image, the selected sources, and the N/E arrows and az/el
        arrows from the nominal wcs/boresight information.
    savePlotAs : `str`, optional
        Save the above plot as a png (for use when headless).

    Returns
    -------
    result : `dict`
        The results of the fit, with the following keys:
        ``nominalRa`` : `lsst.geom.Angle`
            The nominal ra from the exposure's boresight.
        ``nominalDec`` : `lsst.geom.Angle`
            The nominal dec from the exposure's boresight.
        ``calculatedRa`` : `lsst.geom.Angle`
            The fitted ra.
        ``calculatedDec`` : `lsst.geom.Angle`
            The fitted dec.
        ``deltaRa`` : `lsst.geom.Angle`,
            The change in ra, as an Angle.
        ``deltaDec`` : `lsst.geom.Angle`,
            The change in dec, as an Angle.
        ``deltaRaArcsec`` : `float``
            The change in ra in arcseconds, as a float.
        ``deltaDecArcsec`` : `float`
            The change in dec in arcseconds, as a float.
        ``astrometry_net_wcs_header`` : `dict`
            The fitted wcs, as a header dict.

    Raises
    ------
    RuntimeError
        Raised if the fit fails.
    """
    # fail early if this isn't present
    adn = AstrometryNet()
    adn.api_key = getApiKey()
    nominalWcs = exp.getWcs()
    if not nominalWcs:
        print('Trying to process image with None wcs - good luck!')

    imCharResult = runImchar(exp, snr, minPix)

    sourceCatalog = imCharResult.sourceCat
    if not sourceCatalog:
        raise RuntimeError('Failed to find any sources in image')
    filteredSources = _filterSourceCatalog(sourceCatalog, brightSourceFraction)

    if doPlot:
        plot(exp, sourceCatalog, filteredSources, saveAs=savePlotAs)

    vi = exp.getInfo().getVisitInfo()
    ra, dec = vi.boresightRaDec
    if np.isnan(ra.asDegrees()) or np.isnan(dec.asDegrees()):
        print('Got nan for ra/dec from visitInfo, using ra/dec from wcs...')
        if not nominalWcs:
            raise RuntimeError('No wcs for failing over to.')
        ra, dec = nominalWcs.getSkyOrigin()
        if np.isnan(ra.asDegrees()) or np.isnan(dec.asDegrees()):
            raise RuntimeError('Failed to get ra/dec from both visitInfo and wcs')

    if not scaleEstimate:
        nominalWcs = exp.getWcs()
        if nominalWcs:
            scaleEstimate = nominalWcs.getPixelScale().asArcseconds()
        else:
            raise RuntimeError('Got no kwarg for scaleEstimate and failed to find one in the nominal wcs.')

    image_height, image_width = exp.image.array.shape
    scale_units = 'arcsecperpix'
    scale_type = 'ev'  # ev means submit estimate and % error
    scale_err = 3.0  # error as percentage
    center_ra = ra.asDegrees()
    center_dec = dec.asDegrees()
    wcs_header = adn.solve_from_source_list(filteredSources['base_SdssCentroid_x'],
                                            filteredSources['base_SdssCentroid_y'],
                                            image_width, image_height,
                                            scale_units=scale_units,
                                            scale_type=scale_type,
                                            scale_est=scaleEstimate,
                                            scale_err=scale_err,
                                            center_ra=center_ra,
                                            center_dec=center_dec,
                                            radius=radiusInDegrees,
                                            solve_timeout=240)
    print('Finished solving!')

    nominalRa, nominalDec = exp.getInfo().getVisitInfo().getBoresightRaDec()

    if 'CRVAL1' not in wcs_header:
        raise RuntimeError("Astrometric fit failed.")
    calculatedRa = geom.Angle(wcs_header['CRVAL1'], geom.degrees)
    calculatedDec = geom.Angle(wcs_header['CRVAL2'], geom.degrees)

    deltaRa = geom.Angle(wcs_header['CRVAL1'] - nominalRa.asDegrees(), geom.degrees)
    deltaDec = geom.Angle(wcs_header['CRVAL2'] - nominalDec.asDegrees(), geom.degrees)

    result = {'nominalRa': nominalRa,
              'nominalDec': nominalDec,
              'calculatedRa': calculatedRa,
              'calculatedDec': calculatedDec,
              'deltaRa': deltaRa,
              'deltaDec': deltaDec,
              'deltaRaArcsec': deltaRa.asArcseconds(),
              'deltaDecArcsec': deltaDec.asArcseconds(),
              'astrometry_net_wcs_header': wcs_header,
              'nFilteredSources': len(filteredSources),
              'nRawSources': len(sourceCatalog),
              }

    return result


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


def chuckHeaderToWcs(exp, nominalRa=None, nominalDec=None):
    """Given an exposure taken by Chuck at his house, construct a wcs
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
