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

import logging
from lsst.ip.isr import IsrTask
import lsst.daf.butler as dafButler
from lsst.daf.butler.registry import ConflictingDefinitionError

from lsst.summit.utils.quickLook import QuickLookIsrTask
from lsst.summit.utils.butlerUtils import getLatissDefaultCollections, datasetExists

# TODO: add attempt for fringe once registry & templates are fixed

CURRENT_RUN = "LATISS/runs/quickLook/1"
ALLOWED_REPOS = ['/repo/main', '/repo/LATISS', '/readonly/repo/main']


class BestEffortIsr():
    """Class for getting an assembled image with the maximum amount of isr.

    BestEffortIsr.getExposure(dataId) returns an assembled image with as much
    isr performed as possible, dictated by the calibration products available,
    and optionally interpolates over cosmic rays. If an image image already
    exists in the butler, it is returned (for the sake of speed), otherwise it
    is generated and put(). Calibration products are loaded and cached to
    improve performance.

    This class uses the ``quickLookIsrTask``, see docs there for details.

    Acceptable repodir values are currently listed in ALLOWED_REPOS. This will
    be updated (removed) once DM-33849 is done.

    defaultExtraIsrOptions is a dict of options applied to all images.

    Parameters
    ----------
    repoDir : `str`
        The repo root. Will be removed after DM-33849.
    extraCollections : `list` of `str`, optional
        Extra collections to add to the butler init. Collections are prepended.
    defaultExtraIsrOptions : `dict`, optional
        A dict of extra isr config options to apply. Each key should be an
    attribute of an isrTaskConfigClass.
    doRepairCosmics : `bool`, optional
        Repair cosmic ray hits?
    doWrite : `bool`, optional
        Write the outputs to the quickLook rerun/collection?

    Raises
    ------
    FileNotFoundError:
        Raised when a butler cannot be automatically instantiated using
        the DAF_BUTLER_REPOSITORY_INDEX environment variable.
    """
    _datasetName = 'quickLookExp'

    def __init__(self, *,
                 extraCollections=[],
                 defaultExtraIsrOptions={},
                 doRepairCosmics=True,
                 doWrite=True,
                 embargo=False):
        self.log = logging.getLogger(__name__)

        collections = getLatissDefaultCollections()
        self.collections = extraCollections + collections
        self.log.info(f'Instantiating butler with collections={self.collections}')
        try:
            repoString = "LATISS" if not embargo else "/repo/embargo"
            self.butler = dafButler.Butler(repoString, collections=self.collections,
                                           instrument='LATISS',
                                           run=CURRENT_RUN if doWrite else None)
        except(FileNotFoundError, RuntimeError):
            # Depending on the value of DAF_BUTLER_REPOSITORY_INDEX and whether
            # it is present and blank, or just not set, both these exception
            # types can be raised, see
            # tests/test_butlerUtils.py:ButlerInitTestCase
            # for details and tests which confirm these have not changed
            raise FileNotFoundError  # unify exception type

        quickLookIsrConfig = QuickLookIsrTask.ConfigClass()
        quickLookIsrConfig.doRepairCosmics = doRepairCosmics
        self.doWrite = doWrite  # the task, as run by run() method, can't do the write, so we handle in here
        self.quickLookIsrTask = QuickLookIsrTask(config=quickLookIsrConfig)

        self.defaultExtraIsrOptions = defaultExtraIsrOptions

        self._cache = {}

    def _applyConfigOverrides(self, config, overrides):
        """Update a config class with a dict of options.

        Parameters
        ----------
        config : `lsst.pex.config.Config`
            The config class to update.
        overrides : `dict`
            The override options as a dict.

        Raises
        ------
        ValueError
            Raised if the override option isn't found in the config.
        """
        for option, value in overrides.items():
            if hasattr(config, option):
                setattr(config, option, value)
                self.log.info(f"Set isr config override {option} to {value}")
            else:
                raise ValueError(f"Override option {option} not found in isrConfig")

    @staticmethod
    def _parseExpIdOrDataId(expIdOrDataId, **kwargs):
        """Sanitize the expIdOrDataId to allow support both expIds and dataIds

        Supports expId as an integer, or a complete or partial dict. The dict
        is updated with the supplied kwargs.

        Parameters
        ----------
        expIdOrDataId : `int` or `dict
            The exposure id as an int or the dataId as as dict, or an expRecord
            or a dataCoordinate.

        Returns
        -------
        dataId : `dict`
            The sanitized dataId.
        """
        match expIdOrDataId:
            case int() as expId:
                return {"expId": expId}
            case dafButler.DataCoordinate() as dataId:
                return dataId
            case dafButler.DimensionRecord() as record:
                return record.dataId
            case dict() as dataId:
                return dataId
        raise RuntimeError(f"Invalid expId or dataId type {expIdOrDataId}: {type(expIdOrDataId)}")

    def clearCache(self):
        """Clear the internal cache of loaded calibration products.

        Only necessary if you want to use an existing bestEffortIsr object
        after adding new calibration products to the calibration collection.
        """
        self._cache = {}

    def getExposure(self, expIdOrDataId, extraIsrOptions={}, skipCosmics=False, forceRemake=False,
                    **kwargs):
        """Get the postIsr and cosmic-repaired image for this dataId.

        Note that when using the forceRemake option the image will not be
        written to the repo for reuse.

        Parameters
        ----------
        expIdOrDataId : `dict`
            The dataId
        extraIsrOptions : `dict`, optional
            extraIsrOptions is a dict of extra isr options applied to this
            image only.
        skipCosmics : `bool`, optional  # XXX THIS CURRENTLY DOESN'T WORK!
            Skip doing cosmic ray repair for this image?
        forceRemake : `bool`
            Remake the exposure even if there is a pre-existing one in the
            repo. Images that are force-remade are never written, as this is
            assumed to be used for testing/debug purposes, as opposed to normal
            operation. For updating individual images, removal from the
            registry can be used, and for bulk-updates the overall run number
            can be incremented.

        Returns
        -------
        exp : `lsst.afw.image.Exposure`
            The postIsr exposure
        """
        dataId = self._parseExpIdOrDataId(expIdOrDataId)

        if not forceRemake:
            try:
                exp = self.butler.get(self._datasetName, dataId, **kwargs)
                self.log.info("Found a ready-made quickLookExp in the repo. Returning that.")
                return exp
            except LookupError:
                pass

        try:
            raw = self.butler.get('raw', dataId, **kwargs)
        except LookupError:
            raise RuntimeError(f"Failed to retrieve raw for exp {dataId}") from None

        # default options that are probably good for most engineering time
        isrConfig = IsrTask.ConfigClass()
        isrConfig.doWrite = False  # this task writes separately, no need for this
        isrConfig.doSaturation = True  # saturation very important for roundness measurement in qfm
        isrConfig.doSaturationInterpolation = True
        isrConfig.overscan.leadingColumnsToSkip = 5
        isrConfig.overscan.fitType = 'MEDIAN_PER_ROW'

        # apply general overrides
        self._applyConfigOverrides(isrConfig, self.defaultExtraIsrOptions)
        # apply per-image overrides
        self._applyConfigOverrides(isrConfig, extraIsrOptions)

        isrParts = ['camera', 'bias', 'dark', 'flat', 'defects', 'linearizer', 'crosstalk', 'bfKernel',
                    'bfGains', 'ptc']

        isrDict = {}
        # we build a cache of all the isr components which will be used to save
        # the IO time on subsequent calls. This assumes people will not update
        # calibration products while this object lives, but this is a fringe
        # use case, and if they do, all they would need to do would be call
        # .clearCache() and this will rebuild with the new products.
        for component in isrParts:
            if component in self._cache and component != 'flat':
                self.log.info(f"Using {component} from cache...")
                isrDict[component] = self._cache[component]
                continue
            if datasetExists(self.butler, component, dataId):
                try:
                    # TODO: add caching for flats
                    item = self.butler.get(component, dataId=dataId)
                    self._cache[component] = item
                    isrDict[component] = self._cache[component]
                    self.log.info(f"Loaded {component} to cache")
                except Exception:  # now that we log the exception, we can catch all errors
                    # the product *should* exist but the get() failed, so log
                    # a very loud warning inc. the traceback as this is a sign
                    # of butler/database failures or something like that.
                    self.log.critical(f'Failed to find expected data product {component}!')
                    self.log.exception(f'Finding failure for {component}:')
            else:
                self.log.debug('No %s found for %s', component, dataId)

        quickLookExp = self.quickLookIsrTask.run(raw, **isrDict, isrBaseConfig=isrConfig).outputExposure

        if self.doWrite and not forceRemake:
            try:
                self.butler.put(quickLookExp, self._datasetName, dataId, **kwargs)
                self.log.info(f'Put {self._datasetName} for {dataId}')
            except ConflictingDefinitionError:
                # TODO: DM-34302 fix this message so that it's less scary for
                # users. Do this by having daemons know they're daemons.
                self.log.warning('Skipped putting existing exp into collection! (ignore if there was a race)')
                pass

        return quickLookExp
