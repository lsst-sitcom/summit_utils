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

import os
import shutil
import subprocess
import tempfile
import numpy as np
from astropy.io import fits
import time
import uuid
import warnings

from dataclasses import dataclass
from functools import cached_property

import lsst.geom as geom

from .utils import headerToWcs

__all__ = ['AstrometryNetResult', 'CommandLineSolver', 'OnlineSolver']


@dataclass(frozen=True)
class AstrometryNetResult:
    """Minimal wrapper class to construct and return results from the command
    line fitter.

    Constructs a DM wcs from the output of the command line fitter, and
    calculates the plate scale and astrometric scatter measurement in arcsec
    and pixels.

    Parameters
    ----------
    wcsFile : `str`
        The path to the .wcs file from the fit.
    corrFile : `str`, optional
        The path to the .corr file from the fit.
    """
    wcsFile: str
    corrFile: str = None

    def __post_init__(self):
        # touch these properties to ensure the files needed to calculate them
        # are read immediately, in case they are deleted from temp
        self.wcs
        self.rmsErrorArsec

    @cached_property
    def wcs(self):
        with fits.open(self.wcsFile) as f:
            header = f[0].header
        return headerToWcs(header)

    @cached_property
    def plateScale(self):
        return self.wcs.getPixelScale().asArcseconds()

    @cached_property
    def meanSqErr(self):
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
            return meanSqErr
        except Exception as e:
            print(f'Failed for calculate astrometric scatter: {repr(e)}')

    @cached_property
    def rmsErrorPixels(self):
        return np.sqrt(self.meanSqErr)

    @cached_property
    def rmsErrorArsec(self):
        return self.rmsErrorPixels * self.plateScale


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
                 fluxSlot='base_CircularApertureFlux_3_0_instFlux',
                 ):
        self.indexFilePath = indexFilePath
        self.checkInParallel = checkInParallel
        self.timeout = timeout
        self.binary = binary
        self.fluxSlot = fluxSlot
        if not shutil.which(binary):
            raise RuntimeError(f"Could not find {binary} in path, please install 'solve-field' and either"
                               " put it on your PATH or specify the full path to it in the 'binary' argument")

    def _writeConfigFile(self, wide, useGaia):
        """Write a temporary config file for astrometry.net.

        Parameters
        ----------
        wide : `bool`
            Is this a wide field image? Used to select the 4100 vs 4200 dir in
            the index file path. Ignored if ``useGaia`` is ``True``.
        useGaia : `bool`
            Use the 5200 Gaia catalog? If ``True``, ``wide`` is ignored.

        Returns
        -------
        filename : `str`
            The filename to which the config file was written.
        """
        fileSet = '4100' if wide else '4200'
        fileSet = '5200/LITE' if useGaia else fileSet
        indexFileDir = os.path.join(self.indexFilePath, fileSet)
        if not os.path.isdir(indexFileDir):
            raise RuntimeError(f"No index files found at {self.indexFilePath}, in {indexFileDir} (you need a"
                               " 4100 dir for wide field and 4200 dir for narrow field images).")

        lines = []
        if self.checkInParallel:
            lines.append('inparallel')

        lines.append(f"cpulimit {self.timeout}")
        lines.append(f"add_path {indexFileDir}")
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
        fluxArray = sourceCat[self.fluxSlot]
        fluxFinite = np.logical_and(np.isfinite(fluxArray), fluxArray > 0)
        fluxArray = fluxArray[fluxFinite]
        indices = np.argsort(fluxArray)
        x = sourceCat.getColumnView().getX()[fluxFinite]
        y = sourceCat.getColumnView().getY()[fluxFinite]
        fluxArray = fluxArray[indices][::-1]  # brightest finite flux
        xArray = x[indices][::-1]
        yArray = y[indices][::-1]
        x = fits.Column(name='X', format='D', array=xArray)
        y = fits.Column(name='Y', format='D', array=yArray)
        flux = fits.Column(name='FLUX', format='D', array=fluxArray)
        print(f' of which {len(fluxArray)} made it into the fit')
        hdu = fits.BinTableHDU.from_columns([flux, x, y])

        filename = tempfile.mktemp(suffix='.fits')
        hdu.writeto(filename)
        return filename

    # try to keep this call sig and the defaults as similar as possible
    # to the run method on the OnlineSolver
    def run(self, exp, sourceCat, isWideField, *,
            useGaia=False,
            percentageScaleError=10,
            radius=None,
            silent=True):
        """Get the astrometric solution for an image using astrometry.net using
        the binary ``solve-field`` and a set of index files.

        Parameters
        ----------
        exp : `lsst.afw.image.Exposure`
            The input exposure. Only used for its wcs and its dimensions.
        sourceCat : `lsst.afw.table.SourceCatalog`
            The detected source catalog for the exposure. One produced by a
            default run of CharacterizeImageTask is suitable.
        isWideField : `bool`
            Is this a wide field image? Used to select the correct index files.
            Ignored if ``useGaia`` is ``True``.
        useGaia : `bool`
            Use the Gaia 5200/LITE index files? If set, ``isWideField`` is
            ignored.
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
        wcs = exp.getWcs()
        if not wcs:
            raise ValueError("No WCS in exposure")

        configFile = self._writeConfigFile(wide=isWideField, useGaia=useGaia)
        print(f'Fitting image with {len(sourceCat)} sources', end='')
        fitsFile = self._writeFitsTable(sourceCat)

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
               f"--crpix-x {wcs.getPixelOrigin()[0]} "  # set the pixel origin
               f"--crpix-y {wcs.getPixelOrigin()[1]} "  # set the pixel origin
               f"--config {configFile} "
               f"-D {tempDir} "
               "--no-plots "  # don't make plots
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


class OnlineSolver():
    """A class to solve an image using the Astrometry.net online service.
    """

    def __init__(self):
        # import seems to spew warnings even if the required key is present
        # so we swallow them, and raise on init if the key is missing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from astroquery.astrometry_net import AstrometryNet

            self.apiKey = self.getApiKey()  # raises if not present so do first
            self.adn = AstrometryNet()
            self.adn.api_key = self.apiKey

    @staticmethod
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

    # try to keep this call sig and the defaults as similar as possible
    # to the run method on the CommandLineSolver
    def run(self, exp, sourceCat, *, percentageScaleError=10, radius=None, scaleEstimate=None):
        """Get the astrometric solution for an image using the astrometry.net
        online solver.

        Parameters
        ----------
        exp : `lsst.afw.image.Exposure`
            The input exposure. Only used for its wcs.
        sourceCat : `lsst.afw.table.SourceCatalog`
            The detected source catalog for the exposure. One produced by a
            default run of CharacterizeImageTask is suitable.
        percentageScaleError : `float`, optional
            The percentage scale error to allow in the astrometric solution.
        radius : `float`, optional
            The search radius from the nominal wcs in degrees.
        scaleEstimate : `float`, optional
            An estimate of the scale in arcseconds per pixel. Only used if
            (and required when) the exposure has no wcs.

        Returns
        -------
        result : `dict` or `None`
            The results of the fit, with the following keys, or ``None`` if
            the fit failed:
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
        """
        nominalWcs = exp.getWcs()
        if nominalWcs is not None:
            ra, dec = nominalWcs.getSkyOrigin()
            scaleEstimate = nominalWcs.getPixelScale().asArcseconds()
        else:
            print('Trying to process image with None wcs - good luck!')
            vi = exp.getInfo().getVisitInfo()
            ra, dec = vi.boresightRaDec
            if np.isnan(ra.asDegrees()) or np.isnan(dec.asDegrees()):
                raise RuntimeError('Exposure has no wcs and did not find nominal ra/dec in visitInfo')

        if not scaleEstimate:  # must either have a wcs or provide via kwarg
            raise RuntimeError('Got no kwarg for scaleEstimate and failed to find one in the nominal wcs.')

        image_height, image_width = exp.image.array.shape
        scale_units = 'arcsecperpix'
        scale_type = 'ev'  # ev means submit estimate and % error
        scale_err = percentageScaleError  # error as percentage
        center_ra = ra.asDegrees()
        center_dec = dec.asDegrees()
        radius = radius if radius else 180  # degrees
        try:
            wcs_header = self.adn.solve_from_source_list(sourceCat['base_SdssCentroid_x'],
                                                         sourceCat['base_SdssCentroid_y'],
                                                         image_width, image_height,
                                                         scale_units=scale_units,
                                                         scale_type=scale_type,
                                                         scale_est=scaleEstimate,
                                                         scale_err=scale_err,
                                                         center_ra=center_ra,
                                                         center_dec=center_dec,
                                                         radius=radius,
                                                         crpix_center=True,  # the CRPIX is always the center
                                                         solve_timeout=240)
        except RuntimeError:
            print('Failed to find a solution')
            return None

        print('Finished solving!')

        nominalRa, nominalDec = exp.getInfo().getVisitInfo().getBoresightRaDec()

        if 'CRVAL1' not in wcs_header:
            raise RuntimeError("Astrometric fit failed.")
        calculatedRa = geom.Angle(wcs_header['CRVAL1'], geom.degrees)
        calculatedDec = geom.Angle(wcs_header['CRVAL2'], geom.degrees)

        deltaRa = geom.Angle(wcs_header['CRVAL1'] - nominalRa.asDegrees(), geom.degrees)
        deltaDec = geom.Angle(wcs_header['CRVAL2'] - nominalDec.asDegrees(), geom.degrees)

        # TODO: DM-37213 change this to return an AstrometryNetResult class
        # like the CommandLineSolver does.

        result = {'nominalRa': nominalRa,
                  'nominalDec': nominalDec,
                  'calculatedRa': calculatedRa,
                  'calculatedDec': calculatedDec,
                  'deltaRa': deltaRa,
                  'deltaDec': deltaDec,
                  'deltaRaArcsec': deltaRa.asArcseconds(),
                  'deltaDecArcsec': deltaDec.asArcseconds(),
                  'astrometry_net_wcs_header': wcs_header,
                  'nSources': len(sourceCat),
                  }

        return result
