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

__all__ = [
    "PeekExposureTaskConfig",
    "PeekExposureTask",
]

from lsst.afw.detection import Psf
import lsst.afw.display as afwDisplay
import lsst.afw.math as afwMath
import lsst.daf.base as dafBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import numpy as np
from lsst.afw.geom.ellipses import Quadrupole
from lsst.afw.image import ImageD
from lsst.atmospec.utils import isDispersedExp
from lsst.geom import (
    Box2I,
    Extent2I,
    LinearTransform,
    Point2D,
    Point2I,
    SpherePoint,
    arcseconds,
    degrees,
)
from lsst.meas.algorithms.installGaussianPsf import InstallGaussianPsfTask
from lsst.meas.algorithms import (
    SubtractBackgroundTask,
    SourceDetectionTask,
)
from lsst.meas.base import (
    SingleFrameMeasurementTask,
    IdGenerator,
)
from lsst.afw.table import SourceTable


IDX_SENTINEL = -99999


def _estimateMode(data, frac=0.5):
    """Estimate the mode of a 1d distribution.

    Finds the smallest interval containing the fraction ``frac`` of the data,
    then takes the median of the values in that interval.

    Parameters
    ----------
    data : array-like
        1d array of data values
    frac : float, optional
        Fraction of data to include in the mode interval.  Default is 0.5.

    Returns
    -------
    mode : float
        Estimated mode of the data.
    """
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return np.nan
    elif len(data) == 1:
        return data[0]

    data = np.sort(data)
    interval = int(np.ceil(frac * len(data)))
    spans = data[interval:] - data[:-interval]
    start = np.argmin(spans)
    return np.median(data[start:start + interval])


def _bearingToUnitVector(wcs, bearing, imagePoint, skyPoint=None):
    """Compute unit vector along given bearing at given point in the sky.

    Parameters
    ----------
    wcs : `lsst.afw.geom.SkyWcs`
        World Coordinate System of image.
    bearing : `lsst.geom.Angle`
        Bearing (angle North of East) at which to compute unit vector.
    imagePoint : `lsst.geom.Point2D`
        Point in the image.
    skyPoint : `lsst.geom.SpherePoint`, optional
        Point in the sky.

    Returns
    -------
    unitVector : `lsst.geom.Extent2D`
        Unit vector in the direction of bearing.
    """
    if skyPoint is None:
        skyPoint = wcs.pixelToSky(imagePoint)
    dpt = wcs.skyToPixel(skyPoint.offset(bearing, 1e-4 * degrees)) - imagePoint
    return dpt / dpt.computeNorm()


def roseVectors(wcs, imagePoint, parAng=None):
    """Compute unit vectors in the N/W and optionally alt/az directions.

    Parameters
    ----------
    wcs : `lsst.afw.geom.SkyWcs`
        World Coordinate System of image.
    imagePoint : `lsst.geom.Point2D`
        Point in the image
    parAng : `lsst.geom.Angle`, optional
        Parallactic angle (position angle of zenith measured East from North)
        (default: None)

    Returns
    -------
    unitVectors : `dict` of `lsst.geom.Extent2D`
        Unit vectors in the N, W, alt, and az directions.
    """
    ncp = SpherePoint(0 * degrees, 90 * degrees)  # North Celestial Pole
    skyPoint = wcs.pixelToSky(imagePoint)
    bearing = skyPoint.bearingTo(ncp)

    out = dict()
    out["N"] = _bearingToUnitVector(
        wcs, bearing, imagePoint, skyPoint=skyPoint
    )
    out["W"] = _bearingToUnitVector(
        wcs, bearing + 90 * degrees, imagePoint, skyPoint=skyPoint
    )

    if parAng is not None:
        out["alt"] = _bearingToUnitVector(
            wcs, bearing - parAng, imagePoint, skyPoint=skyPoint
        )
        out["az"] = _bearingToUnitVector(
            wcs, bearing - parAng + 90 * degrees, imagePoint, skyPoint=skyPoint
        )

    return out


def plotRose(display, wcs, imagePoint, parAng=None, len=50):
    """Display unit vectors along N/W and optionally alt/az directions.

    Parameters
    ----------
    display : `lsst.afw.display.Display`
        Display on which to render rose.
    wcs : `lsst.afw.geom.SkyWcs`
        World Coordinate System of image.
    imagePoint : `lsst.geom.Point2D`
        Point in the image at which to render rose.
    parAng : `lsst.geom.Angle`, optional
        Parallactic angle (position angle of zenith measured East from North)
        (default: None)
    len : `float`, optional
        Length of the rose vectors (default: 50)
    """
    unitVectors = roseVectors(wcs, imagePoint, parAng=parAng)
    colors = dict(N="r", W="r", alt="g", az="g")
    for name, unitVector in unitVectors.items():
        display.line([imagePoint, imagePoint + len * unitVector], ctype=colors[name])
        display.dot(name, *(imagePoint + 1.6 * len * unitVector), ctype=colors[name])


class DonutPsf(Psf):
    def __init__(self, size, outerRad, innerRad):
        Psf.__init__(self, isFixed=True)
        self.size = size
        self.outerRad = outerRad
        self.innerRad = innerRad
        self.dimensions = Extent2I(size, size)

    def __deepcopy__(self, memo=None):
        return DonutPsf(self.size, self.outerRad, self.innerRad)

    def resized(self, width, height):
        assert width == height
        return DonutPsf(width, self.outerRad, self.innerRad)

    def _doComputeKernelImage(self, position=None, color=None):
        bbox = self.computeBBox(self.getAveragePosition())
        img = ImageD(bbox, 0.0)
        x, y = np.ogrid[bbox.minY:bbox.maxY + 1, bbox.minX:bbox.maxX + 1]
        rsqr = x**2 + y**2
        w = (rsqr < self.outerRad**2) & (rsqr > self.innerRad**2)
        img.array[w] = 1.0
        img.array /= np.sum(img.array)
        return img

    def _doComputeBBox(self, position=None, color=None):
        return Box2I(Point2I(-self.dimensions / 2), self.dimensions)

    def _doComputeShape(self, position=None, color=None):
        Ixx = self.outerRad**4 - self.innerRad**4
        Ixx /= self.outerRad**2 - self.innerRad**2
        return Quadrupole(Ixx, Ixx, 0.0)

    def _doComputeApertureFlux(self, radius, position=None, color=None):
        return 1 - np.exp(-0.5 * (radius / self.sigma) ** 2)

    def __eq__(self, rhs):
        if isinstance(rhs, DonutPsf):
            return (
                self.size == rhs.size and
                self.outerRad == rhs.outerRad and
                self.innerRad == rhs.innerRad
            )
        return False


class PeekTaskConfig(pexConfig.Config):
    """Config class for the PeekTask."""

    installPsf = pexConfig.ConfigurableField(
        target=InstallGaussianPsfTask,
        doc="Install a PSF model",
    )
    doInstallPsf = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Install a PSF model?",
    )
    background = pexConfig.ConfigurableField(
        target=SubtractBackgroundTask,
        doc="Estimate background",
    )
    detection = pexConfig.ConfigurableField(
        target=SourceDetectionTask,
        doc="Detect sources"
    )
    measurement = pexConfig.ConfigurableField(
        target=SingleFrameMeasurementTask,
        doc="Measure sources"
    )
    defaultBinSize = pexConfig.Field(
        dtype=int,
        default=1,
        doc="Default binning factor for exposure (often overridden).",
    )

    def setDefaults(self):
        super().setDefaults()
        # Configure to be aggressively fast.
        self.detection.thresholdValue = 5.0
        self.detection.includeThresholdMultiplier = 10.0
        self.detection.reEstimateBackground = False
        self.detection.doTempLocalBackground = False
        self.measurement.doReplaceWithNoise = False
        self.detection.minPixels = 40
        self.installPsf.fwhm = 5.0
        self.installPsf.width = 21
        # minimal set of measurements
        self.measurement.plugins.names = [
            "base_PixelFlags",
            "base_SdssCentroid",
            "ext_shapeHSM_HsmSourceMoments",
            "base_GaussianFlux",
            "base_PsfFlux",
            "base_CircularApertureFlux",
        ]
        self.measurement.slots.shape = "ext_shapeHSM_HsmSourceMoments"


class PeekTask(pipeBase.Task):
    """Peek at exposure to quickly detect and measure both the brightest source
    in the image, and a set of sources representative of the exposure's overall
    image quality.

    Optionally bins image and then:
        - installs a simple PSF model
        - measures and subtracts the background
        - detects sources
        - measures sources

    Designed to be quick at the expense of primarily completeness, but also to
    a lesser extent accuracy.
    """

    ConfigClass = PeekTaskConfig
    _DefaultName = "peek"

    def __init__(self, schema=None, **kwargs):
        super().__init__(**kwargs)

        if schema is None:
            schema = SourceTable.makeMinimalSchema()
        self.schema = schema

        self.makeSubtask("installPsf")
        self.makeSubtask("background")
        self.makeSubtask("detection", schema=self.schema)
        self.algMetadata = dafBase.PropertyList()
        self.makeSubtask('measurement', schema=self.schema, algMetadata=self.algMetadata)

    def run(self, exposure, binSize=None):
        """Peek at exposure.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure at which to peek.
        binSize : `int`, optional
            Binning factor for exposure.  Default is None, which will use the
            default binning factor from the config.

        Returns
        -------
        result : `pipeBase.Struct`
            Result of peeking.
            Struct containing:
                - sourceCat : `lsst.afw.table.SourceCatalog`
                    Source catalog from the binned exposure.
        """
        if binSize is None:
            binSize = self.config.defaultBinSize

        if binSize != 1:
            mi = exposure.getMaskedImage()
            binned = afwMath.binImage(mi, binSize)
            exposure.setMaskedImage(binned)

        if self.config.doInstallPsf:
            self.installPsf.run(exposure=exposure)

        self.background.run(exposure)

        idGenerator = IdGenerator()
        sourceIdFactory = idGenerator.make_table_id_factory()
        table = SourceTable.make(self.schema, sourceIdFactory)
        table.setMetadata(self.algMetadata)
        sourceCat = self.detection.run(table=table, exposure=exposure, doSmooth=True).sources

        self.measurement.run(measCat=sourceCat, exposure=exposure, exposureId=idGenerator.catalog_id)

        return pipeBase.Struct(
            sourceCat=sourceCat,
        )


class PeekDonutTaskConfig(pexConfig.Config):
    """Config class for the PeekDonutTask."""

    peek = pexConfig.ConfigurableField(
        target=PeekTask,
        doc="Peek configuration",
    )
    resolution = pexConfig.Field(
        dtype=float,
        default=16.0,
        doc="Target number of pixels for a binned donut",
    )
    binSizeMax = pexConfig.Field(
        dtype=int,
        default=10,
        doc="Maximum binning factor for donut mode",
    )

    def setDefaults(self):
        super().setDefaults()
        # Donuts are big even when binned.
        self.peek.installPsf.fwhm = 10.0
        self.peek.installPsf.width = 31
        # Use DonutPSF if not overridden
        self.peek.doInstallPsf = False


class PeekDonutTask(pipeBase.Task):
    """Peek at a donut exposure.

    The main modification for donuts is to aggressively bin the image to reduce
    the size of sources (donuts) from ~100 pixels or more to ~10 pixels.  This
    greatly increases the speed and detection capabilities of PeekTask with
    little loss of accuracy for centroids.
    """

    ConfigClass = PeekDonutTaskConfig
    _DefaultName = "peekDonut"

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.makeSubtask("peek")

    def run(self, exposure, donutDiameter, binSize=None):
        """Peek at donut exposure.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure at which to peek.
        donutDiameter : `float`
            Donut diameter in pixels.
        binSize : `int`, optional
            Binning factor for exposure.  Default is None, which will use the
            resolution config value to determine the binSize.

        Returns
        -------
        result : `pipeBase.Struct`
            Result of donut peeking.
            Struct containing:
                - mode : `str`
                    Peek mode that was run.
                - binSize : `int`
                    Binning factor used.
                - binnedSourceCat : `lsst.afw.table.SourceCatalog`
                    Source catalog from the binned exposure.
        """
        if binSize is None:
            binSize = int(
                np.floor(
                    np.clip(
                        donutDiameter / self.config.resolution,
                        1,
                        self.config.binSizeMax,
                    )
                )
            )
        binnedDonutDiameter = donutDiameter / binSize
        psf = DonutPsf(
            binnedDonutDiameter*1.5,
            binnedDonutDiameter*0.5,
            binnedDonutDiameter*0.5*0.3525
        )

        # Note that SourceDetectionTask will convolve with a _Gaussian
        # approximation to the PSF_ anyway, so we don't really need to be
        # precise with the PSF unless this changes.  PSFs that approach the
        # size of the image, however, can cause problems with the detection
        # convolution algorithm, so we limit the size.
        sigma = psf.computeShape(psf.getAveragePosition()).getDeterminantRadius()
        factor = 8*sigma / (min(exposure.getDimensions())/binSize)

        if factor > 1:
            psf = DonutPsf(
                binnedDonutDiameter*1.5/factor,
                binnedDonutDiameter*0.5/factor,
                binnedDonutDiameter*0.5*0.3525/factor
            )
        exposure.setPsf(psf)

        peekResult = self.peek.run(exposure, binSize)

        return pipeBase.Struct(
            mode="donut",
            binSize=binSize,
            binnedSourceCat=peekResult.sourceCat,
        )

    def getGoodSources(self, binnedSourceCat):
        """Perform any filtering on the source catalog.

        Parameters
        ----------
        binnedSourceCat : `lsst.afw.table.SourceCatalog`
            Source catalog from the binned exposure.

        Returns
        -------
        goodSourceMask : `numpy.ndarray`
            Boolean array indicating which sources are good.
        """
        # Perform any filtering on the source catalog
        goodSourceMask = np.ones(len(binnedSourceCat), dtype=bool)
        return goodSourceMask


class PeekPhotoTaskConfig(pexConfig.Config):
    """Config class for the PeekPhotoTask."""

    peek = pexConfig.ConfigurableField(
        target=PeekTask,
        doc="Peek configuration",
    )
    binSize = pexConfig.Field(
        dtype=int,
        default=2,
        doc="Binning factor for exposure",
    )

    def setDefaults(self):
        super().setDefaults()
        # Use a lower detection threshold in photo mode to go a bit fainter.
        self.peek.detection.includeThresholdMultiplier = 1.0
        self.peek.detection.thresholdValue = 10.0
        self.peek.detection.minPixels = 10


class PeekPhotoTask(pipeBase.Task):
    """Peek at a photo (imaging) exposure.

    For photo mode, we keep a relatively small detection threshold value, so we
    can detect faint sources to use for image quality assessment.
    """

    ConfigClass = PeekPhotoTaskConfig
    _DefaultName = "peekPhoto"

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.makeSubtask("peek")

    def run(self, exposure, binSize=None):
        """Peek at donut exposure.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure at which to peek.
        binSize : `int`, optional
            Binning factor for exposure.  Default is None, which will use the
            binning factor from the config.

        Returns
        -------
        result : `pipeBase.Struct`
            Result of photo peeking.
            Struct containing:
                - mode : `str`
                    Peek mode that was run.
                - binSize : `int`
                    Binning factor used.
                - binnedSourceCat : `lsst.afw.table.SourceCatalog`
                    Source catalog from the binned exposure.
        """
        if binSize is None:
            binSize = self.config.binSize

        peekResult = self.peek.run(exposure, binSize)

        return pipeBase.Struct(
            mode="photo",
            binSize=binSize,
            binnedSourceCat=peekResult.sourceCat,
        )

    def getGoodSources(self, binnedSourceCat):
        """Perform any filtering on the source catalog.

        Parameters
        ----------
        binnedSourceCat : `lsst.afw.table.SourceCatalog`
            Source catalog from the binned exposure.

        Returns
        -------
        goodSourceMask : `numpy.ndarray`
            Boolean array indicating which sources are good.
        """
        # Perform any filtering on the source catalog
        goodSourceMask = np.ones(len(binnedSourceCat), dtype=bool)
        return goodSourceMask


class PeekSpecTaskConfig(pexConfig.Config):
    """Config class for the PeekSpecTask."""

    peek = pexConfig.ConfigurableField(
        target=PeekTask,
        doc="Peek configuration",
    )
    binSize = pexConfig.Field(
        dtype=int,
        default=2,
        doc="binning factor for exposure",
    )
    maxFootprintAspectRatio = pexConfig.Field(
        dtype=float,
        default=10.0,
        doc="Maximum detection footprint aspect ratio to consider as 0th order"
            " (non-dispersed) light."
    )

    def setDefaults(self):
        super().setDefaults()
        # Use bright threshold
        self.peek.detection.includeThresholdMultiplier = 1.0
        self.peek.detection.thresholdValue = 500.0
        # Use a large radius aperture flux for spectra to better identify the
        # brightest source, which for spectra often has a saturated core.
        self.peek.measurement.slots.apFlux = "base_CircularApertureFlux_70_0"
        # Also allow a larger distance to peak for centroiding in case there's
        # a relatively large saturated region.
        self.peek.measurement.plugins['base_SdssCentroid'].maxDistToPeak = 15.0


class PeekSpecTask(pipeBase.Task):
    """Peek at a spectroscopic exposure.

    For spec mode, we dramatically increase the detection threshold to avoid
    creating blends with the long spectra objects that appear in these images.
    We also change the default aperture flux slot to a larger aperture, which
    helps overcome challenges with lost flux in the interpolated cores of
    saturated objects.
    """

    ConfigClass = PeekSpecTaskConfig
    _DefaultName = "peekSpec"

    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        self.makeSubtask("peek")

    def run(self, exposure, binSize=None):
        """Peek at spectroscopic exposure.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure at which to peek.
        binSize : `int`, optional
            Binning factor for exposure.  Default is None, which will use the
            binning factor from the config.

        Returns
        -------
        result : `pipeBase.Struct`
            Result of spec peeking.
            Struct containing:
                - mode : `str`
                    Peek mode that was run.
                - binSize : `int`
                    Binning factor used.
                - binnedSourceCat : `lsst.afw.table.SourceCatalog`
                    Source catalog from the binned exposure.
        """
        if binSize is None:
            binSize = self.config.binSize

        peekResult = self.peek.run(exposure, binSize)

        return pipeBase.Struct(
            mode="spec",
            binSize=binSize,
            binnedSourceCat=peekResult.sourceCat,
        )

    def getGoodSources(self, binnedSourceCat):
        """Perform any filtering on the source catalog.

        Parameters
        ----------
        binnedSourceCat : `lsst.afw.table.SourceCatalog`
            Source catalog from the binned exposure.

        Returns
        -------
        goodSourceMask : `numpy.ndarray`
            Boolean array indicating which sources are good.
        """
        # Perform any filtering on the source catalog
        goodSourceMask = np.ones(len(binnedSourceCat), dtype=bool)
        fpShapes = [src.getFootprint().getShape() for src in binnedSourceCat]
        # Filter out likely spectrum detections
        goodSourceMask &= np.array(
            [
                sh.getIyy() < self.config.maxFootprintAspectRatio * sh.getIxx()
                for sh in fpShapes
            ],
            dtype=bool
        )
        return goodSourceMask


class PeekExposureTaskConfig(pexConfig.Config):
    """Config class for the PeekExposureTask."""

    donutThreshold = pexConfig.Field(
        dtype=float,
        default=50.0,
        doc="Size threshold in pixels for when to switch to donut mode.",
    )
    doPhotoFallback = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="If True, fall back to photo mode if spec mode fails.",
    )
    donut = pexConfig.ConfigurableField(
        target=PeekDonutTask,
        doc="PeekDonut task",
    )
    photo = pexConfig.ConfigurableField(
        target=PeekPhotoTask,
        doc="PeekPhoto task",
    )
    spec = pexConfig.ConfigurableField(
        target=PeekSpecTask,
        doc="PeekSpec task",
    )


class PeekExposureTask(pipeBase.Task):
    """ Peek at exposure to quickly detect and measure both the brightest
    source in the image, and a set of sources representative of the
    exposure's overall image quality.

    Parameters
    ----------
    config : `lsst.summit.utils.peekExposure.PeekExposureTaskConfig`
        Configuration for the task.
    display : `lsst.afw.display.Display`, optional
        For displaying the exposure and sources.

    Notes
    -----
    The basic philosophy of PeekExposureTask is to:
    1) Classify exposures based on metadata into 'donut', 'spec', or 'photo'.
    2) Run PeekTask on the exposure through a wrapper with class specific
    modifications.
    3) Try only to branch in the code based on the metadata, and not on the
       data itself.  This avoids problematic discontinuities in measurements.

    The main knobs we fiddle with based on the classification are:
        - Detection threshold
        - Minimum number of pixels for a detection
        - Binning of the image
        - Installed PSF size
    """

    ConfigClass = PeekExposureTaskConfig
    _DefaultName = "peekExposureTask"

    def __init__(self, config, *, display=None, **kwargs):
        super().__init__(config=config, **kwargs)

        self.makeSubtask("donut")
        self.makeSubtask("photo")
        self.makeSubtask("spec")

        self._display = display

    def getDonutDiameter(self, exposure):
        """Estimate donut diameter from exposure metadata.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to estimate donut diameter for.

        Returns
        -------
        donutDiameter : `float`
            Estimated donut diameter in pixels.
        """
        visitInfo = exposure.getInfo().getVisitInfo()
        focusZ = visitInfo.focusZ
        instrumentLabel = visitInfo.instrumentLabel

        match instrumentLabel:
            case "LATISS":
                focusZ *= 41  # magnification factor
                fratio = 18.0
            case "LSSTCam" | "ComCam":
                fratio = 1.234
        # AuxTel/ComCam/LSSTCam all have 10 micron pixels (= 10e-3 mm)
        donutDiameter = abs(focusZ) / fratio / 10e-3
        self.log.info(f"{focusZ=} mm")
        self.log.info(f"donutDiameter = {donutDiameter} pixels")
        return donutDiameter

    def run(
        self,
        exposure,
        *,
        doDisplay=False,
        doDisplayIndices=False,
        mode="auto",
        binSize=None,
        donutDiameter=None,
    ):
        """
        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure at which to peek.
        doDisplay : `bool`, optional
            Display the exposure and sources?  Default False.  (Requires
            display to have been passed to task constructor)
        doDisplayIndices : `bool`, optional
            Display the source indices?  Default False.  (Requires display to
            have been passed to task constructor)
        mode : {'auto', 'donut', 'spec', 'photo'}, optional
            Mode to run in.  Default 'auto'.
        binSize : `int`, optional
            Binning factor for exposure.  Default is None, which let's subtasks
            control rebinning directly.
        donutDiameter : `float`, optional
            Donut diameter in pixels.  Default is None, which will estimate the
            donut diameter from the exposure metadata.

        Returns
        -------
        result : `pipeBase.Struct`
            Result of the peek.
            Struct containing:
                - mode : `str`
                    Peek mode that was run.
                - binSize : `int`
                    Binning factor used.
                - binnedSourceCat : `lsst.afw.table.SourceCatalog`
                    Source catalog from the binned exposure.
                - table : `astropy.table.Table`
                    Curated source table in unbinned coordinates.
                - brightestIdx : `int`
                    Index of brightest source in source catalog.
                - brightestCentroid : `lsst.geom.Point2D`
                    Brightest source centroid in unbinned pixel coords.
                - brightestPixelShape : `lsst.geom.Quadrupole`
                    Brightest source shape in unbinned pixel coords.
                - brightestEquatorialShape : `lsst.geom.Quadrupole`
                    Brightest source shape in equitorial coordinates (arcsec).
                - brightestAltAzShape : `lsst.geom.Quadrupole`
                    Brightest source shape in alt/az coordinates (arcsec).
                - psfPixelShape : `lsst.geom.Quadrupole`
                    Estimated PSF shape in unbinned pixel coords.
                - psfEquatorialShape : `lsst.geom.Quadrupole`
                    Estimated PSF shape in equitorial coordinates (arcsec).
                - psfAltAzShape : `lsst.geom.Quadrupole`
                    Estimated PSF shape in alt/az coordinates (arcsec).
                - pixelMedian : `float`
                    Median estimate of entire image.
                - pixelMode : `float`
                    Mode estimate of entire image.
        """
        # Make a copy so the original image is unmodified.
        exposure = exposure.clone()
        try:
            result = self._run(
                exposure, doDisplay, doDisplayIndices, mode, binSize, donutDiameter
            )
        except Exception as e:
            self.log.warning(f"Peek failed: {e}")
            result = pipeBase.Struct(
                mode="failed",
                binSize=0,
                binnedSourceCat=None,
                table=None,
                brightestIdx=0,
                brightestCentroid=Point2D(np.nan, np.nan),
                brightestPixelShape=Quadrupole(np.nan, np.nan, np.nan),
                brightestEquatorialShape=Quadrupole(np.nan, np.nan, np.nan),
                brightestAltAzShape=Quadrupole(np.nan, np.nan, np.nan),
                psfPixelShape=Quadrupole(np.nan, np.nan, np.nan),
                psfEquatorialShape=Quadrupole(np.nan, np.nan, np.nan),
                psfAltAzShape=Quadrupole(np.nan, np.nan, np.nan),
                pixelMedian=np.nan,
                pixelMode=np.nan,
            )
        return result

    def _run(self, exposure, doDisplay, doDisplayIndices, mode, binSize, donutDiameter):
        """ The actual run method, called by run().
        """
        # If image is ~large, then use a subsampling of the image for
        # speedy median/mode estimates.
        arr = exposure.getMaskedImage().getImage().array
        sampling = 1
        if arr.size > 250_000:
            sampling = int(np.floor(np.sqrt(arr.size / 250_000)))
        pixelMedian = np.nanmedian(arr[::sampling, ::sampling])
        pixelMode = _estimateMode(arr[::sampling, ::sampling])

        if donutDiameter is None:
            donutDiameter = self.getDonutDiameter(exposure)

        mode, binSize, binnedSourceCat = self.runPeek(
            exposure, mode, donutDiameter, binSize
        )

        table = self.transformTable(binSize, binnedSourceCat)

        match mode:
            case "donut":
                goodSourceMask = self.donut.getGoodSources(binnedSourceCat)
            case "spec":
                goodSourceMask = self.spec.getGoodSources(binnedSourceCat)
            case "photo":
                goodSourceMask = self.photo.getGoodSources(binnedSourceCat)

        # prepare output variables
        maxFluxIdx, brightCentroid, brightShape = self.getBrightest(
            binnedSourceCat, binSize, goodSourceMask
        )
        psfShape = self.getPsfShape(binnedSourceCat, binSize, goodSourceMask)

        equatorialShapes, altAzShapes = self.transformShapes(
            [brightShape, psfShape],
            exposure,
            binSize,
        )

        if doDisplay:
            self.updateDisplay(
                exposure, binSize, binnedSourceCat, maxFluxIdx, doDisplayIndices
            )

        return pipeBase.Struct(
            mode=mode,
            binSize=binSize,
            binnedSourceCat=binnedSourceCat,
            table=table,
            brightestIdx=maxFluxIdx,
            brightestCentroid=brightCentroid,
            brightestPixelShape=brightShape,
            brightestEquatorialShape=equatorialShapes[0],
            brightestAltAzShape=altAzShapes[0],
            psfPixelShape=psfShape,
            psfEquatorialShape=equatorialShapes[1],
            psfAltAzShape=altAzShapes[1],
            pixelMedian=pixelMedian,
            pixelMode=pixelMode,
        )

    def runPeek(self, exposure, mode, donutDiameter, binSize=None):
        """Classify exposure and run appropriate PeekTask wrapper.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to peek.
        mode : {'auto', 'donut', 'spec', 'photo'}
            Mode to run in.
        donutDiameter : `float`
            Donut diameter in pixels.
        binSize : `int`, optional
            Binning factor for exposure.  Default is None, which let's subtasks
            control rebinning directly.

        Returns
        -------
        result : `pipeBase.Struct`
            Result of the peek.
            Struct containing:
                - mode : `str`
                    Peek mode that was run.
                - binSize : `int`
                    Binning factor used.
                - binnedSourceCat : `lsst.afw.table.SourceCatalog`
                    Source catalog from the binned exposure.
        """
        if mode == "auto":
            # Note, no attempt to handle dispersed donuts.  They'll default to
            # donut mode.
            if donutDiameter > self.config.donutThreshold:
                mode = "donut"
            elif isDispersedExp(exposure):
                mode = "spec"
            else:
                mode = "photo"

        match mode:
            case "donut":
                result = self.donut.run(exposure, donutDiameter, binSize=binSize)
                binSizeOut = result.binSize
            case "spec":
                result = self.spec.run(exposure, binSize=binSize)
                binSizeOut = result.binSize
                if len(result.binnedSourceCat) == 0:
                    self.log.warn("No sources found in spec mode.")
                    if self.config.doPhotoFallback:
                        self.log.warn("Falling back to photo mode.")
                        # Note that spec.run already rebinned the image,
                        # so we don't need to do it again.
                        result = self.photo.run(exposure, binSize=1)
            case "photo":
                result = self.photo.run(exposure, binSize=binSize)
                binSizeOut = result.binSize
            case _:
                raise ValueError(f"Unknown mode {mode}")
        return result.mode, binSizeOut, result.binnedSourceCat

    def transformTable(self, binSize, binnedSourceCat):
        """Make an astropy table from the source catalog but with
        transformations back to the original unbinned coordinates.

        Since there's some ambiguity in the apFlux apertures when binning,
        we'll only populate the table with the slots columns (slot_apFlux
        doesn't indicate an aperture radius).  For simplicity, do the same for
        centroids and shapes too.

        And since we're only copying over the slots_* columns, we remove the
        "slots_" part of the column names and lowercase the first remaining
        letter.

        Parameters
        ----------
        binSize : `int`
            Binning factor used.
        binnedSourceCat : `lsst.afw.table.SourceCatalog`
            Source catalog from the binned exposure.

        Returns
        -------
        table : `astropy.table.Table`
            Curated source table in unbinned coordinates.
        """
        table = binnedSourceCat.asAstropy()
        cols = [n for n in table.colnames if n.startswith("slot")]
        table = table[cols]
        if "slot_Centroid_x" in cols:
            table["slot_Centroid_x"] = (
                binSize * table["slot_Centroid_x"] + (binSize - 1) / 2
            )
            table["slot_Centroid_y"] = (
                binSize * table["slot_Centroid_y"] + (binSize - 1) / 2
            )
        if "slot_Shape_x" in cols:
            table["slot_Shape_x"] = binSize * table["slot_Shape_x"] + (binSize - 1) / 2
            table["slot_Shape_y"] = binSize * table["slot_Shape_y"] + (binSize - 1) / 2
            table["slot_Shape_xx"] *= binSize**2
            table["slot_Shape_xy"] *= binSize**2
            table["slot_Shape_yy"] *= binSize**2
        # area and npixels are just confusing when binning, so remove.
        if "slot_PsfFlux_area" in cols:
            del table["slot_PsfFlux_area"]
        if "slot_PsfFlux_npixels" in cols:
            del table["slot_PsfFlux_npixels"]

        table.rename_columns(
            [n for n in table.colnames if n.startswith('slot_')],
            [n[5:6].lower()+n[6:] for n in table.colnames if n.startswith('slot_')]
        )

        return table

    def getBrightest(self, binnedSourceCat, binSize, goodSourceMask):
        """Find the brightest source in the catalog.

        Parameters
        ----------
        binnedSourceCat : `lsst.afw.table.SourceCatalog`
            Source catalog from the binned exposure.
        binSize : `int`
            Binning factor used.
        goodSourceMask : `numpy.ndarray`
            Boolean array indicating which sources are good.

        Returns
        -------
        maxFluxIdx : `int`
            Index of the brightest source in the catalog.
        brightCentroid : `lsst.geom.Point2D`
            Centroid of the brightest source (unbinned coords).
        brightShape : `lsst.geom.Quadrupole`
            Shape of the brightest source (unbinned coords).
        """
        fluxes = np.array([source.getApInstFlux() for source in binnedSourceCat])
        idxs = np.arange(len(binnedSourceCat))

        good = (goodSourceMask & np.isfinite(fluxes))

        if np.sum(good) == 0:
            maxFluxIdx = IDX_SENTINEL
            brightCentroid = Point2D(np.nan, np.nan)
            brightShape = Quadrupole(np.nan, np.nan, np.nan)
            return maxFluxIdx, brightCentroid, brightShape

        fluxes = fluxes[good]
        idxs = idxs[good]
        maxFluxIdx = idxs[np.nanargmax(fluxes)]
        brightest = binnedSourceCat[maxFluxIdx]

        # Convert binned coordinates back to original unbinned
        # coordinates
        brightX, brightY = brightest.getCentroid()
        brightX = binSize * brightX + (binSize - 1) / 2
        brightY = binSize * brightY + (binSize - 1) / 2
        brightCentroid = Point2D(brightX, brightY)
        brightIXX = brightest.getIxx() * binSize**2
        brightIXY = brightest.getIxy() * binSize**2
        brightIYY = brightest.getIyy() * binSize**2
        brightShape = Quadrupole(brightIXX, brightIYY, brightIXY)

        return maxFluxIdx, brightCentroid, brightShape

    def getPsfShape(self, binnedSourceCat, binSize, goodSourceMask):
        """Estimate the modal PSF shape from the sources.

        Parameters
        ----------
        binnedSourceCat : `lsst.afw.table.SourceCatalog`
            Source catalog from the binned exposure.
        binSize : `int`
            Binning factor used.
        goodSourceMask : `numpy.ndarray`
            Boolean array indicating which sources are good.

        Returns
        -------
        psfShape : `lsst.geom.Quadrupole`
            Estimated PSF shape (unbinned coords).
        """
        fluxes = np.array([source.getApInstFlux() for source in binnedSourceCat])
        idxs = np.arange(len(binnedSourceCat))

        good = (goodSourceMask & np.isfinite(fluxes))

        if np.sum(good) == 0:
            return Quadrupole(np.nan, np.nan, np.nan)

        fluxes = fluxes[good]
        idxs = idxs[good]

        psfIXX = _estimateMode(
            np.array([source.getIxx() for source in binnedSourceCat])[goodSourceMask]
        )
        psfIYY = _estimateMode(
            np.array([source.getIyy() for source in binnedSourceCat])[goodSourceMask]
        )
        psfIXY = _estimateMode(
            np.array([source.getIxy() for source in binnedSourceCat])[goodSourceMask]
        )

        return Quadrupole(
            psfIXX * binSize**2,
            psfIYY * binSize**2,
            psfIXY * binSize**2,
        )

    def transformShapes(self, shapes, exposure, binSize):
        """Transform shapes from x/y pixel coordinates to equitorial and
        horizon coordinates.

        Parameters
        ----------
        shapes : `list` of `lsst.geom.Quadrupole`
            List of shapes (in pixel coordinates) to transform.
        exposure : `lsst.afw.image.Exposure`
            Exposure containing WCS and VisitInfo for transformation.
        binSize : `int`
            Binning factor used.

        Returns
        -------
        equatorialShapes : `list` of `lsst.geom.Quadrupole`
            List of shapes transformed to equitorial (North and West)
            coordinates.  Units are arcseconds.
        altAzShapes : `list` of `lsst.geom.Quadrupole`
            List of shapes transformed to alt/az coordinates.  Units are
            arcseconds.
        """
        pt = Point2D(np.array([*exposure.getBBox().getCenter()]) / binSize)
        wcs = exposure.wcs
        visitInfo = exposure.info.getVisitInfo()
        parAngle = visitInfo.boresightParAngle

        equatorialShapes = []
        altAzShapes = []
        for shape in shapes:
            if wcs is None:
                equatorialShapes.append(Quadrupole(np.nan, np.nan, np.nan))
                altAzShapes.append(Quadrupole(np.nan, np.nan, np.nan))
                continue
            # The WCS transforms to N (dec) and E (ra), but we want N and W to
            # conform with weak-lensing conventions.  So we flip the [0]
            # component of the transformation.
            neTransform = wcs.linearizePixelToSky(pt, arcseconds).getLinear()
            nwTransform = LinearTransform(
                np.array([[-1, 0], [0, 1]]) @ neTransform.getMatrix()
            )
            equatorialShapes.append(shape.transform(nwTransform))

            # To get from N/W to alt/az, we need to additionally rotate by the
            # parallactic angle.
            rot = LinearTransform.makeRotation(parAngle).getMatrix()
            aaTransform = LinearTransform(nwTransform.getMatrix() @ rot)
            altAzShapes.append(shape.transform(aaTransform))

        return equatorialShapes, altAzShapes

    def updateDisplay(
        self, exposure, binSize, binnedSourceCat, maxFluxIdx, doDisplayIndices
    ):
        """Update the afwDisplay with the exposure and sources.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to peek.
        binSize : `int`
            Binning factor used.
        binnedSourceCat : `lsst.afw.table.SourceCatalog`
            Source catalog from the binned exposure.
        maxFluxIdx : `int`
            Index of the brightest source in the catalog.
        doDisplayIndices : `bool`
            Display the source indices?
        """
        if self._display is None:
            raise RuntimeError("Display failed as no display provided during init()")

        visitInfo = exposure.info.getVisitInfo()
        self._display.mtv(exposure)
        wcs = exposure.wcs
        if wcs is not None:
            plotRose(
                self._display,
                wcs,
                Point2D(200 / binSize, 200 / binSize),
                parAng=visitInfo.boresightParAngle,
                len=100 / binSize,
            )

        for idx, source in enumerate(binnedSourceCat):
            x, y = source.getCentroid()
            sh = source.getShape()
            self._display.dot(sh, x, y)
            if doDisplayIndices:
                self._display.dot(str(idx), x, y)

        if maxFluxIdx != IDX_SENTINEL:
            self._display.dot(
                "+",
                *binnedSourceCat[maxFluxIdx].getCentroid(),
                ctype=afwDisplay.RED,
                size=10,
            )