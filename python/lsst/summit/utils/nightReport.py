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
        for i, seqNum in enumerate(obsInfosToLoad):
            if i+1 % 200 == 0:
                print(f"Loaded {i+1} obsInfos")
            obsInfo, metadata = self.getObsInfoAndMetadataForSeqNum(seqNum)
            obsInfoDict = obsInfoToDict(obsInfo)
            records[seqNum].update(obsInfoDict)
            # _raw_metadata item will hopefully not be needed in the future
            # but add it while we have it for free, as it has DIMM seeing
            records[seqNum]['_raw_metadata'] = metadata
            self._obsInfosLoaded.add(seqNum)

        self.data = self.getSortedData(self.data)  # make sure we stay sorted
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

    @staticmethod
    def _ensureList(arg):
        if type(arg) == str:
            return [arg]
        assert(type(arg) == list), f"Expect list, got {type(arg)}: {arg}"
        return arg

    def calcShutterOpenEfficiency(self, seqMin=0, seqMax=0):
        raise NotImplementedError("This is not yet implemented")
        # if seqMin == 0:
        #     seqMin = min(self.data.keys())
        # if seqMax == 0:
        #     seqMax = max(self.data.keys())
        # assert seqMax > seqMin
        # assert (seqMin in self.data.keys())
        # assert (seqMax in self.data.keys())

        # timeStart = self.data[seqMin]['ObservationInfo'].datetime_begin
        # timeEnd = self.data[seqMax]['ObservationInfo'].datetime_end
        # expTimeSum = 0
        # for seqNum in range(seqMin, seqMax+1):
        #     if seqNum not in self.data.keys():
        #         print(f"Warning! No data found for seqNum {seqNum}")
        #         continue
        #     expTimeSum += self.data[seqNum]['ObservationInfo'].exposure_time.value

        # timeOnSky = (timeEnd - timeStart).sec
        # efficiency = expTimeSum/timeOnSky
        # print(f"{100*efficiency:.2f}% shutter open in seqNum range {seqMin} and {seqMax}")
        # print(f"Total integration time = {expTimeSum:.1f}s")
        # return efficiency

    def printObsTable(self, imageType=None, tailNumber=0):
        """Print a table of the days observations.

        Parameters
        ----------
        imageType : str
            Only consider images with this image type
        tailNumber : int
            Only print out the last n entries in the night
        """
        raise NotImplementedError("This is not yet implemented")
        # lines = []
        # if not imageType:
        #     seqNums = self.data.keys()
        # else:
        #     seqNums = [s for s in self.data.keys()
        #                if self.data[s]['ObservationInfo'].observation_type == imageType]

        # seqNums = sorted(seqNums)
        # for i, seqNum in enumerate(seqNums):
        #     try:
        #         expTime = self.data[seqNum]['ObservationInfo'].exposure_time.value
        #         filt = self.data[seqNum]['ObservationInfo'].physical_filter
        #         imageType = self.data[seqNum]['ObservationInfo'].observation_type
        #         d1 = self.data[seqNum]['ObservationInfo'].datetime_begin
        #         obj = self.data[seqNum]['ObservationInfo'].object
        #         if i == 0:
        #             d0 = d1
        #         dt = (d1-d0)
        #         d0 = d1
        #         timeOfDay = d1.isot.split('T')[1]
        #         msg = f'{seqNum:4} {imageType:9} {obj:10} {timeOfDay} {filt:25} {dt.sec:6.1f}  {expTime:2.2f}'
        #     except KeyError:
        #         msg = f'{seqNum:4} - error parsing headers/observation info! Check the file'
        #     lines.append(msg)

        # print(r"{seqNum} {imageType} {obj} {timeOfDay} {filt} {timeSinceLastExp} {expTime}")
        # for line in lines[-tailNumber:]:
        #     print(line)

    def getExposureMidpoint(self, seqNum):
        """Return the midpoint of the exposure as a float in MJD.
        """
        timespan = self.data[seqNum]['timespan']
        return (timespan.begin.mjd + timespan.begin.mjd) / 2

    def plotPerObjectAirMass(self, objects=None, airmassOneAtTop=True):
        """XXX
        """
        if not objects:
            objects = self.stars

        objects = self._ensureList(objects)

        _ = plt.figure(figsize=(10, 6))
        for star in objects:
            seqNums = self.getSeqNumsMatching(target_name=star)
            airMasses = [self.data[seqNum]['boresight_airmass'] for seqNum in seqNums]
            obsTimes = [self.getExposureMidpoint(self, seqNum) for seqNum in seqNums]
            color = self.cMap[star].color
            marker = self.cMap[star].marker
            plt.plot(obsTimes, airMasses, color=color, marker=marker, label=star, ms=10, ls='')

        plt.ylabel('Airmass', fontsize=20)
        if airmassOneAtTop:
            ax = plt.gca()
            ax.set_ylim(ax.get_ylim()[::-1])
        _ = plt.legend(bbox_to_anchor=(1, 1.025), prop={'size': 15}, loc='upper left')
