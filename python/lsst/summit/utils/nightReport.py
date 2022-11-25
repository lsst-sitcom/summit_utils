from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from astro_metadata_translator import ObservationInfo
from lsst.summit.utils.utils import obsInfoToDict  # change to .utils later XXX

__all__ = ['NightReport']

CALIB_VALUES = ['FlatField position', 'Park position', 'azel_target']
N_STARS_PER_SYMBOL = 6
MARKER_SEQUENCE = ['*', 'o', "D", 'P', 'v', "^", 's', '.', ',', 'o', 'v', '^',
                   '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h',
                   'H', '+', 'x', 'X', 'D', 'd', '|', '_']
SOUTHPOLESTAR = 'HD 185975'


@dataclass
class ColorAndMarker:
    '''Class for holding colors and marker symbols'''
    color: list
    marker: str = '*'


class NightReport():
    def __init__(self, butler, dayObs):
        self.butler = butler
        self.dayObs = dayObs
        self.data = dict()
        self._expRecordsLoaded = set()  # set of the expRecords loaded
        self._obsInfosLoaded = set()  # set of the seqNums loaded
        self.stars = None
        self.cMap = None
        self.rebuild()

    @staticmethod
    def getSortedData(data):
        if list(data.keys()) == sorted(data.keys()):
            return data
        else:
            return {k: data[k] for k in sorted(data.keys())}

    def getExpRecordDictForDayObs(self, dayObs):
        """Get all the exposureRecords as dicts for the current dayObs.

        Runs in ~0.05s for 1000 records.
        """
        expRecords = self.butler.registry.queryDimensionRecords("exposure",
                                                                where="exposure.day_obs=day_obs",
                                                                bind={'day_obs': dayObs},
                                                                datasets='raw')
        expRecords = list(expRecords)
        records = {e.seq_num: e.toDict() for e in expRecords}  # not guaranteed to be in order
        return self.getSortedData(records)

    def getObsInfoAndMetadataForSeqNum(self, seqNum):
        dataId = {'day_obs': self.dayObs, 'seq_num': seqNum, 'detector': 0}
        md = self.butler.get('raw.metadata', dataId)
        return ObservationInfo(md), md

    def rebuild(self):
        records = self.getExpRecordDictForDayObs(self.dayObs)
        if len(records) == len(self.data):  # nothing to do
            print('No new records found')
            return
        else:
            # still need to merge the new expRecordDicts into self.data
            # but only these, as the other items have obsInfos merged into them
            for seqNum in list(records.keys() - self._expRecordsLoaded):
                self.data[seqNum] = records[seqNum]
                self._expRecordsLoaded.add(seqNum)

        # now load all the obsInfos
        seqNums = list(records.keys())
        obsInfosToLoad = set(seqNums) - self._obsInfosLoaded
        if obsInfosToLoad:
            print(f"Loading {len(obsInfosToLoad)} obsInfo(s)")
        for seqNum in obsInfosToLoad:
            obsInfo, metadata = self.getObsInfoAndMetadataForSeqNum(seqNum)
            obsInfoDict = obsInfoToDict(obsInfo)
            records[seqNum].update(obsInfoDict)
            # _raw_metadata item will hopefully not be needed in the future
            # but add it while we have it for free, as it has DIMM seeing
            records[seqNum]['_raw_metadata'] = metadata
            self._obsInfosLoaded.add(seqNum)

        self.data = self.getSortedData(records)
        self.stars = self.getObservedObjects()
        self.cMap = self.makeStarColorAndMarkerMap(self.stars)

    def getObservedObjects(self):
        return sorted({record['target_name'] for record in self.data.values()})

    def getSeqNumsMatching(self, invert=False, subset=None, **kwargs):
        """Get seqNums which match/don't match all kwargs provided, e.g.

        report.getSeqNumsMatching(exposure_time=30,
                                  target_name='ETA1 DOR')

        Set invert=True to get all seqNums which don't match the provided
        args, e.g. to find all seqNums which are not calibs

        Subset allows for repeated filtering by passing in a set of seqNums
        """
        # make a copy data and restrict to subset if provided
        local = {seqNum: rec for seqNum, rec in self.data.items() if (subset is None or seqNum in subset)}

        # for each kwarg, filter out items which match/don't
        for filtAttr, filtVal in kwargs.items():
            toPop = []  # can't pop inside inner loop so collect
            for seqNum, record in local.items():
                v = record.get(filtAttr)
                if invert:
                    if v == filtVal:
                        toPop.append(seqNum)
                else:
                    if v != filtVal:
                        toPop.append(seqNum)
            [local.pop(seqNum) for seqNum in toPop]

        return sorted(local.keys())

    def printAvailableKeys(self, includeRaw=False):
        """Print all the keys available to query on, optionally including the
        full set of header keys.

        Note that there is a big mix of quantities, some are int/float/string
        but some are astropy quantities.
        """
        for seqNum, recordDict in self.data.items():  # loop + break because we don't know the first seqNum
            for k in recordDict.keys():
                print(k)
            if includeRaw:
                print("\nRaw header keys in _raw_metadata:")
                for k in recordDict['_raw_metadata']:
                    print(k)
            break

    @staticmethod
    def makeStarColorAndMarkerMap(stars):
        markerMap = {}
        colors = cm.rainbow(np.linspace(0, 1, N_STARS_PER_SYMBOL))
        for i, star in enumerate(stars):
            markerIndex = i//(N_STARS_PER_SYMBOL)
            colorIndex = i%(N_STARS_PER_SYMBOL)
            markerMap[star] = ColorAndMarker(colors[colorIndex], MARKER_SEQUENCE[markerIndex])
        return markerMap

    # def plotPerObjectAirMass(self, objects=None, airmassOneAtTop=True, filterFunc=None):
    #     """filterFunc is self as the first argument and seqNum as second."""
    #     if not objects:
    #         objects = self.stars

    #     objects = self._safeListArg(objects)

    #     # lazy to always recalculate but it's not *that* slow
    #     # and optionally passing around can be messy
    #     # TODO: keep some of this in class state
    #     airMasses = self._calcObjectAirmasses(objects, filterFunc=filterFunc)

    #     _ = plt.figure(figsize=(10, 6))
    #     for star in objects:
    #         if airMasses[star]:  # skip stars fully filtered out by callbacks
    #             ams, times = np.asarray(airMasses[star])[:, 0], np.asarray(airMasses[star])[:, 1]
    #         else:
    #             continue
    #         color = self.cMap[star].color
    #         marker = self.cMap[star].marker
    #         plt.plot(times, ams, color=color, marker=marker, label=star, ms=10, ls='')

    #     plt.ylabel('Airmass', fontsize=20)
    #     if airmassOneAtTop:
    #         ax = plt.gca()
    #         ax.set_ylim(ax.get_ylim()[::-1])
    #     _ = plt.legend(bbox_to_anchor=(1, 1.025), prop={'size': 15}, loc='upper left')
