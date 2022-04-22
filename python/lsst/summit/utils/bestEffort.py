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

from sqlite3 import OperationalError

import logging
from lsst.ip.isr import IsrTask
import lsst.daf.butler as dafButler
from lsst.daf.butler.registry import ConflictingDefinitionError

from lsst.rapid.analysis.quickLook import QuickLookIsrTask
from lsst.rapid.analysis.butlerUtils import (LATISS_DEFAULT_COLLECTIONS, LATISS_SUPPLEMENTAL_COLLECTIONS,
                                             _repoDirToLocation)

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
    """
    _datasetName = 'quickLookExp'

    def __init__(self, repodir, *,
                 extraCollections=[], defaultExtraIsrOptions={}, doRepairCosmics=True, doWrite=True):
        if repodir not in ALLOWED_REPOS:
            raise RuntimeError('Currently only NCSA and summit repos are supported')
        self.log = logging.getLogger(__name__)

        location = _repoDirToLocation(repodir)
        collections = (LATISS_SUPPLEMENTAL_COLLECTIONS[location] if location in
                       LATISS_SUPPLEMENTAL_COLLECTIONS.keys() else []) + LATISS_DEFAULT_COLLECTIONS
        self.collections = extraCollections + collections
        self.log.info(f'Instantiating butler with collections={self.collections}')
        self.butler = dafButler.Butler(repodir, collections=self.collections,
                                       instrument='LATISS',
                                       run=CURRENT_RUN if doWrite else None)

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
            The exposure id as an int or the dataId as as dict.

        Returns
        -------
        dataId : `dict`
            The sanitized dataId.
        """
        if type(expIdOrDataId) == int:
            _dataId = {'expId': expIdOrDataId}
        elif type(expIdOrDataId) == dict:
            _dataId = expIdOrDataId
            _dataId.update(kwargs)
        else:
            raise RuntimeError(f"Invalid expId or dataId type {expIdOrDataId}")
        return _dataId

    def clearCache(self):
        """Clear the internal cache of loaded calibration products.

        Only necessary if you want to use an existing bestEffortIsr object
        after adding new calibration products to the calibration collection.
        """
        self._cache = {}

    def getExposure(self, expIdOrDataId, extraIsrOptions={}, skipCosmics=False, **kwargs):
        """Get the postIsr and cosmic-repaired image for this dataId.

        Parameters
        ----------
        expIdOrDataId : `dict`
            The dataId
        extraIsrOptions : `dict`, optional
            extraIsrOptions is a dict of extra isr options applied to this
            image only.
        skipCosmics : `bool`, optional  # XXX THIS CURRENTLY DOESN'T WORK!
            Skip doing cosmic ray repair for this image?

        Returns
        -------
        exp : `lsst.afw.image.Exposure`
            The postIsr exposure
        """
        dataId = self._parseExpIdOrDataId(expIdOrDataId, **kwargs)

        try:
            exp = self.butler.get(self._datasetName, dataId=dataId)
            self.log.info("Found a ready-made quickLookExp in the repo. Returning that.")
            return exp
        except LookupError:
            pass

        try:
            raw = self.butler.get('raw', dataId=dataId)
        except LookupError:
            raise RuntimeError(f"Failed to retrieve raw for exp {dataId}") from None

        # default options that are probably good for most engineering time
        isrConfig = IsrTask.ConfigClass()
        isrConfig.doWrite = False  # this task writes separately, no need for this
        isrConfig.doSaturation = True  # saturation very important for roundness measurement in qfm
        isrConfig.doSaturationInterpolation = True
        isrConfig.overscanNumLeadingColumnsToSkip = 5
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
            try:
                # TODO: add caching for flats
                item = self.butler.get(component, dataId=dataId)
                self._cache[component] = item
                isrDict[component] = self._cache[component]
            except (RuntimeError, LookupError, OperationalError):
                pass

        quickLookExp = self.quickLookIsrTask.run(raw, **isrDict, isrBaseConfig=isrConfig).outputExposure

        if self.doWrite:
            try:
                self.butler.put(quickLookExp, self._datasetName, dataId)
                self.log.info(f'Put {self._datasetName} for {dataId}')
            except ConflictingDefinitionError:
                # TODO: DM-34302 fix this message so that it's less scary for
                # users. Do this by having daemons know they're daemons.
                self.log.warning('Skipped putting existing exp into collection! (ignore if there was a race)')
                pass

        return quickLookExp


if __name__ == '__main__':
    # TODO: DM-34239 Move this to be a butler-driven test
    repodir = '/repo/main'
    bestEffort = BestEffortIsr(repodir, doWrite=True)
    dataId = {'day_obs': 20200315, 'seq_num': 164, 'detector': 0}
    exp = bestEffort.getExposure(dataId)
