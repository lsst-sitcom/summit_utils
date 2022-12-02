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
import shutil
import subprocess
import tempfile
import numpy as np
from astropy.io import fits
import time
import uuid

import matplotlib.pyplot as plt
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


def filterOnBrightest(catalog, brightFraction, minSources=15,
                      flux_field="base_CircularApertureFlux_3_0_instFlux"):
    """Return a catalog containing the brightest sources in the input,
    making an initial coarse cut, keeping those above 0.1% of the maximum
    finite flux.

    Parameters
    ----------
    catalog : `lsst.afw.table.SourceCatalog`
        Catalog to be filtered.
    brightFraction : `float`
        Return this fraction of the brightest sources.
    minSources : `int`, optional
        Always return at least this many sources.
    flux_field : `str`, optional
        Name of flux field to filter on.

    Returns
    -------
    result : `lsst.afw.table.SourceCatalog`
        Brightest sources in the input image, in ascending order of brightness.
    """
    maxFlux = np.nanmax(catalog[flux_field])
    result = catalog.subset(catalog[flux_field] > maxFlux * 0.001)

    print(f"Catalog had {len(catalog)} sources, of which {len(result)} were above 0.1% of max")

    item = catalog.schema.find(flux_field)
    result = catalog.copy(deep=True)  # sort() is in place; copy so we don't modify the original
    result.sort(item.key)
    result = result.copy(deep=True)  # make it memory contiguous
    end = int(np.ceil(len(result)*brightFraction))
    return result[-max(end, minSources):]


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
    filteredSources = filterOnBrightest(sourceCatalog, brightSourceFraction)

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


# don't be tempted to get cute and try to combine these 4 functions. It would
# be a easy to do but it's not unlikley they will diverge in the future.
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


def genericCameraHeaderToWcs(exp):
    header = exp.getMetadata().toDict()
    width, height = exp.image.array.shape
    header['CRPIX1'] = width/2
    header['CRPIX2'] = height/2

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


class AstrometryNetResult():
    """Minimal wrapper class to construct and return results from the command
    line fitter.

    Constructs a DM wcs from the output of the command line fitter, and sets
    the plate scale on the class for use in the scatter measurement. This loads
    the corr files from the fit and calculates the scatter in pixels and
    arcseconds.

    Parameters
    ----------
    wcsFile : `str`
        The path to the .wcs file from the fit.
    corrFile : `str`
        The path to the .corr file from the fit.
    """
    wcs = None
    scatterPixels = None
    scatterArcseconds = None
    plateScale = None

    def __init__(self, wcsFile, corrFile=None):
        self.wcsFile = wcsFile
        self._buildWcs()  # create the DM wcs from the a.net wcs file. Sets self.plateScale from fitted wcs

        self.corrFile = corrFile
        self._calcScatter()  # sets self.scatterPixels and self.scatterArcseconds

    def _buildWcs(self):
        with fits.open(self.wcsFile) as f:
            header = f[0].header
        wcs = headerToWcs(header)
        self.wcs = wcs
        self.plateScale = wcs.getPixelScale().asArcseconds()

    def _calcScatter(self):
        if not self.corrFile:
            return None

        try:
            with fits.open(self.corrFile) as f:
                data = f[1].data

            meanSqErr = 0.0
            count = 0
            for i in range(data.shape[0]):
                row = data[i]
                count += 1
                error = (row[0] - row[4])**2 + (row[1] - row[5])**2  # square error in pixels
                error *= row[10]  # multiply by weight
                meanSqErr += error
            meanSqErr /= count  # divide by number of stars
            rmsErrorPixels = np.sqrt(meanSqErr)
            rmsErrorArsec = rmsErrorPixels * self.plateScale

            self.scatterPixels = rmsErrorPixels
            self.scatterArcseconds = rmsErrorArsec
        except Exception as e:
            print(f'Failed for calculate astrometric scatter: {repr(e)}')


class CommandLineSolver():
    """An interface for the solve-field command line tool from astrometry.net.

    Parameters
    ----------
    indexFilePath : `str`
        The path to the index files. Do not include the 4100 or 4200 etc. in
        the path. This is selected automatically depending on the `isWideField`
        flag when calling `run()`.
    checkInParallel : `bool`, optional
        Do the checks in parallel. Default is True.
    timeout : `float`, optional
        The timeout for the solve-field command. Default is 300 seconds.
    binary : `str`, optional
        The path to the solve-field binary. Default is 'solve-field', i.e. it
        is assumed to be on the path.
    """
    def __init__(self,
                 indexFilePath=None,
                 checkInParallel=True,
                 timeout=300,
                 binary='solve-field',
                 ):
        self.indexFilePath = indexFilePath
        self.checkInParallel = checkInParallel
        self.timeout = timeout
        self.binary = binary
        if not shutil.which(binary):
            raise RuntimeError(f"Could not find {binary} in path, please install 'solve-field' and either"
                               " put it on your PATH or specify the full path to it in the 'binary' argument")

    def _writeConfigFile(self, wide):
        """Write a temporary config file for astrometry.net.

        Parameters
        ----------
        wide : `bool`
            Is this a wide field image? Used to select the 4100 vs 4200 dir in
            the index file path.

        Returns
        -------
        filename : `str`
            The filename to which the config file was written.
        """
        indexFiles = os.path.join(self.indexFilePath, ('4100' if wide else '4200'))
        if not os.path.exists(indexFiles):
            raise RuntimeError(f"No index files found at {self.indexFilePath}, in {indexFiles} (you need a"
                               " 4100 dir for wide field and 4200 dir for narrow field images).")

        lines = []
        if self.checkInParallel:
            lines.append('inparallel')

        lines.append(f"cpulimit {self.timeout}")
        lines.append(f"add_path {indexFiles}")
        lines.append("autoindex")
        filename = tempfile.mktemp(suffix='.cfg')
        with open(filename, 'w') as f:
            f.writelines(line + '\n' for line in lines)
        return filename

    def _writeFitsTable(self, sourceCat):
        """Write the source table to a FITS file and return the filename.

        Parameters
        ----------
        sourceCat : `lsst.afw.table.SourceCatalog`
            The source catalog to write to a FITS file for the solver.

        Returns
        -------
        filename : `str`
            The filename to which the catalog was written.
        """
        fluxArray = sourceCat.columns.getGaussianInstFlux()
        fluxFinite = np.logical_and(np.isfinite(fluxArray), fluxArray > 0)
        fluxArray = fluxArray[fluxFinite]
        args = np.argsort(fluxArray)
        x = sourceCat.getColumnView().getX()[fluxFinite]
        y = sourceCat.getColumnView().getY()[fluxFinite]
        fluxArray = fluxArray[args][::-1]  # brightest finite flux
        xArray = x[args][::-1]
        yArray = y[args][::-1]
        x = fits.Column(name='X', format='D', array=xArray)
        y = fits.Column(name='Y', format='D', array=yArray)
        flux = fits.Column(name='FLUX', format='D', array=fluxArray)
        hdu = fits.BinTableHDU.from_columns([flux, x, y])

        filename = tempfile.mktemp(suffix='.fits')
        hdu.writeto(filename)
        return filename

    def run(self, exp, sourceCat, isWideField, percentageScaleError=10, radius=None, silent=True):
        """

        Parameters
        ----------
        exp : `lsst.afw.image.Exposure`
            The input exposure. Only used for its wcs.
        sourceCat : `lsst.afw.table.SourceCatalog`
            The detected source catalog for the exposure. One produced by a
            default run of CharacterizeImageTask is suitable.
        isWideField : `bool`
            Is this a wide field image? Used to select the correct index files.
        percentageScaleError : `float`, optional
            The percentage scale error to allow in the astrometric solution.
        radius : `float`, optional
            The search radius from the nominal wcs in degrees.
        silent : `bool`, optional
            Swallow the output from the command line? The solver is *very*
            chatty, so this is recommended.

        Returns
        -------
        result : `AstrometryNetResult` or `None`
            The result of the fit. If the fit was successful, the result will
            contain a valid DM wcs, a scatter in arcseconds and a scatter in
            pixels. If the fit failed, ``None`` is returned.
        """
        configFile = self._writeConfigFile(wide=isWideField)
        fitsFile = self._writeFitsTable(sourceCat)
        wcs = exp.getWcs()
        if not wcs:
            raise ValueError("No WCS in exposure")
        plateScale = wcs.getPixelScale().asArcseconds()
        scaleMin = plateScale*(1 - percentageScaleError/100)
        scaleMax = plateScale*(1 + percentageScaleError/100)

        ra, dec = wcs.getSkyOrigin()

        # do not use tempfile.TemporaryDirectory() because it must not exist,
        # it is made by the solve-field binary and barfs if it exists already!
        mainTempDir = tempfile.gettempdir()
        tempDirSuffix = str(uuid.uuid1()).split('-')[0]
        tempDir = os.path.join(mainTempDir, tempDirSuffix)

        cmd = (f"{self.binary} {fitsFile} "  # the data
               f"--width {exp.getWidth()} "  # image dimensions
               f"--height {exp.getHeight()} "  # image dimensions
               f"-3 {ra.asDegrees()} "
               f"-4 {dec.asDegrees()} "
               f"-5 {radius if radius else 180} "
               "-X X -Y Y -v -z 2 -t 2 "  # the parts of the bintable to use
               f"--scale-low {scaleMin:.3f} "  # the scale range
               f"--scale-high {scaleMax:.3f} "  # the scale range
               f"--scale-units arcsecperpix "
               "--crpix-center "  # the CRPIX is always the center of the image
               f"--config {configFile} "
               f"-D {tempDir} "
               "--overwrite "  # shouldn't matter as we're using temp files
               )

        t0 = time.time()
        with open(os.devnull, 'w') as devnull:
            result = subprocess.run(cmd, shell=True, check=True, stdout=devnull if silent else None)
        t1 = time.time()

        if result.returncode == 0:
            print(f"Fitting code ran in {(t1-t0):.2f} seconds")
            # output template is /tmpdirname/fitstempname + various suffixes
            # for each obj
            basename = os.path.basename(fitsFile).removesuffix('.fits')
            outputTemplate = os.path.join(tempDir, basename)
            wcsFile = outputTemplate + '.wcs'
            corrFile = outputTemplate + '.corr'

            if not os.path.exists(wcsFile):
                print("but failed to find a solution.")
                return None

            result = AstrometryNetResult(wcsFile, corrFile)
            return result
        else:
            print("Fit failed")
        return None
