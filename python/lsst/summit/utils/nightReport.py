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

import pickle
import logging

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from humanize.time import precisedelta

from astro_metadata_translator import ObservationInfo
from lsst.summit.utils.utils import (obsInfoToDict,  # change to .utils later XXX
                                     getFieldNameAndTileNumber
                                     )

__all__ = ['NightReport']

CALIB_VALUES = ['FlatField position', 'Park position', 'azel_target']
N_STARS_PER_SYMBOL = 6
MARKER_SEQUENCE = ['*', 'o', "D", 'P', 'v', "^", 's', '.', ',', 'o', 'v', '^',
                   '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h',
                   'H', '+', 'x', 'X', 'D', 'd', '|', '_']
SOUTHPOLESTAR = 'HD 185975'

CALIB_VALUES = ['FlatField position', 'Park position', 'azel_target']
# TODO: add skips for calib values


@dataclass
class ColorAndMarker:
    '''Class for holding colors and marker symbols'''
    color: list
    marker: str = '*'


class NightReport():
    def __init__(self, butler, dayObs, loadFromFile=None):
        self.log = logging.getLogger('lsst.summit.utils.NightReport')
        self.butler = butler
        self.dayObs = dayObs
        self.data = dict()
        self._expRecordsLoaded = set()  # set of the expRecords loaded
        self._obsInfosLoaded = set()  # set of the seqNums loaded
        self.stars = None
        self.cMap = None
        if loadFromFile is not None:
            self.load(loadFromFile)
        self.rebuild()

    def save(self, filename):
        toSave = (self.data, self._expRecordsLoaded, self._obsInfosLoaded, self.dayObs)
        with open(filename, "wb") as f:
            pickle.dump(toSave, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        with open(filename, "rb") as f:
            loaded = pickle.load(f)
        self.data, self._expRecordsLoaded, self._obsInfosLoaded, dayObs = loaded
        if dayObs != self.dayObs:
            raise RuntimeError(f"Loaded data is for {dayObs} but current dayObs is {self.dayObs}")
        assert len(self.data) == len(self._expRecordsLoaded)
        assert len(self.data) == len(self._obsInfosLoaded)
        self.log.info(f"Loaded {len(self.data)} records from {filename}")

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

    def rebuild(self, full=False):
        if full:
            self.data = dict()
            self._expRecordsLoaded = set()
            self._obsInfosLoaded = set()

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
            if (i+1) % 200 == 0:
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

    def getObservedObjects(self, ignoreTileNum=True):
        allTargets = sorted({record['target_name'] if record['target_name'] is not None else ''
                             for record in self.data.values()})
        if not ignoreTileNum:
            return allTargets
        # need to call set and sorted again here because what is unique now
        # wasn't before, because of the tile numbers
        return sorted(set([getFieldNameAndTileNumber(target, warn=False)[0] for target in allTargets]))

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

    def printAvailableKeys(self, sample=False, includeRaw=False):
        """Print all the keys available to query on, optionally including the
        full set of header keys.

        Note that there is a big mix of quantities, some are int/float/string
        but some are astropy quantities.
        """
        for seqNum, recordDict in self.data.items():  # loop + break because we don't know the first seqNum
            for k, v in recordDict.items():
                if sample:
                    print(f"{k}: {v}")
                else:
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

    def calcShutterTimes(self):
        result = {}

        firstObs = self.getNightStartSeqNum(method='safe')
        lastObs = max(self.data.keys())

        begin = self.data[firstObs]['datetime_begin']
        end = self.data[lastObs]['datetime_end']

        READOUT_TIME = 2.0
        shutterOpenTime = sum([self.data[s]['exposure_time'] for s in range(firstObs, lastObs+1)])
        readoutTime = sum([READOUT_TIME for _ in range(firstObs, lastObs+1)])

        sciSeqNums = self.getSeqNumsMatching(observation_type='science')
        scienceIntegration = sum([self.data[s]['exposure_time'] for s in sciSeqNums])
        scienceTimeTotal = scienceIntegration.value + (len(sciSeqNums) * READOUT_TIME)

        result['firstObs'] = firstObs
        result['lastObs'] = lastObs
        result['startTime'] = begin
        result['endTime'] = end
        result['nightLength'] = (end - begin).sec  # was a datetime.timedelta
        result['shutterOpenTime'] = shutterOpenTime.value  # was an Quantity
        result['readoutTime'] = readoutTime
        result['scienceIntegration'] = scienceIntegration.value  # was an Quantity
        result['scienceTimeTotal'] = scienceTimeTotal

        return result

    def printShutterTimes(self):
        """Print out the shutter efficiency stats in a human-readable format.
        """
        timings = self.calcShutterTimes()

        print(f"Observations started at: seqNum {timings['firstObs']:>3} at"
              f" {timings['startTime'].to_datetime().strftime('%H:%M:%S')} TAI")
        print(f"Observations ended at:   seqNum {timings['lastObs']:>3} at"
              f" {timings['endTime'].to_datetime().strftime('%H:%M:%S')} TAI")
        print(f"Total time on sky: {precisedelta(timings['nightLength'])}")
        print()
        print(f"Shutter open time: {precisedelta(timings['shutterOpenTime'])}")
        print(f"Readout time: {precisedelta(timings['readoutTime'])}")
        engEff = 100 * (timings['shutterOpenTime'] + timings['readoutTime']) / timings['nightLength']
        print(f"Engineering shutter efficiency = {engEff:.1f}%")
        print()
        print(f"Science integration: {precisedelta(timings['scienceIntegration'])}")
        sciEff = 100*(timings['scienceTimeTotal'] / timings['nightLength'])
        print(f"Science shutter efficiency = {sciEff:.1f}%")

    def getTimeDeltas(self):
        """Returns a dict, keyed by seqNum, of the time since the end of the last integration.
        """
        seqNums = list(self.data.keys())  # need a list not a generator, and NB it might not be contiguous!
        dts = [0]  # first item is zero by definition
        for i, seqNum in enumerate(seqNums[1:]):
            dt = self.data[seqNum]['datetime_begin'] - self.data[(seqNums[i])]['datetime_end']
            dts.append(dt.sec)

        return {s: dt for s, dt in zip(seqNums, dts)}

    def printObsGaps(self):
        """Print out the gaps between observations in a human-readable format.
        """
        THRESHOLD = 100
        dts = self.getTimeDeltas()

        # get the portion of the night we care about as there are and should
        # be gaps when taking calibs and waiting to go on sky.
        allSeqNums = list(self.data.keys())
        firstObs = self.getNightStartSeqNum(method='safe')
        startPoint = allSeqNums.index(firstObs) + 1  # there is always a big gap before firstObs by definition
        seqNums = allSeqNums[startPoint:]

        messages = []
        for seqNum in seqNums:
            dt = dts[seqNum]
            if dt > THRESHOLD:
                messages.append(f"seqNum {seqNum:3}: {precisedelta(dt)} gap")

        if messages:
            print(f"Gaps between observations greater than {THRESHOLD}s:")
            for line in messages:
                print(line)

    def getNightStartSeqNum(self, method='heuristic'):
        allowedMethods = ['heuristic', 'safe']
        if method not in allowedMethods:
            raise ValueError(f"Method must be one of {allowedMethods}")

        if method == 'safe':
            # take the first cwfs image and return that
            seqNums = self.getSeqNumsMatching(observation_type='cwfs')
            return min(seqNums)

    def printObsTable(self, **kwargs):
        """Print a table of the days observations.

        Parameters
        ----------
        imageType : str
            Only consider images with this image type
        tailNumber : int
            Only print out the last n entries in the night
        """
        seqNums = self.data.keys() if not kwargs else self.getSeqNumsMatching(**kwargs)
        seqNums = sorted(seqNums)  # should always be sorted, but is a totaly disaster here if not

        dts = self.getTimeDeltas()
        lines = []
        for seqNum in seqNums:
            try:
                expTime = self.data[seqNum]['exposure_time'].value
                imageType = self.data[seqNum]['observation_type']
                target = self.data[seqNum]['target_name']
                deadtime = dts[seqNum]
                filt = self.data[seqNum]['physical_filter']

                msg = f'{seqNum} {target} {expTime:.1f} {deadtime:.02f} {imageType} {filt}'
            except Exception:
                msg = f"Error parsing {seqNum}!"
            lines.append(msg)

        print(r"seqNum target expTime deadtime imageType filt")
        print(r"------ ------ ------- -------- --------- ----")
        for line in lines:
            print(line)

    def getExposureMidpoint(self, seqNum):
        """Return the midpoint of the exposure as a float in MJD.
        """
        timespan = self.data[seqNum]['timespan']
        return (timespan.begin.mjd + timespan.begin.mjd) / 2

    def plotPerObjectAirMass(self, objects=None, airmassOneAtTop=True):
        """XXX docs
        """
        if not objects:
            objects = self.stars

        objects = self._ensureList(objects)

        _ = plt.figure(figsize=(10, 6))
        for star in objects:
            seqNums = self.getSeqNumsMatching(target_name=star)
            airMasses = [self.data[seqNum]['boresight_airmass'] for seqNum in seqNums]
            obsTimes = [self.getExposureMidpoint(seqNum) for seqNum in seqNums]
            color = self.cMap[star].color
            marker = self.cMap[star].marker
            plt.plot(obsTimes, airMasses, color=color, marker=marker, label=star, ms=10, ls='')

        plt.ylabel('Airmass', fontsize=20)
        if airmassOneAtTop:
            ax = plt.gca()
            ax.set_ylim(ax.get_ylim()[::-1])
        _ = plt.legend(bbox_to_anchor=(1, 1.025), prop={'size': 15}, loc='upper left')

    def _makePolarPlot(self, azimuthsInDegrees, zenithAngles, marker="*-",
                       title=None, makeFig=True, color=None, objName=None):
        if makeFig:
            _ = plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)
        ax.plot([a*np.pi/180 for a in azimuthsInDegrees], zenithAngles, marker, c=color, label=objName)
        if title:
            ax.set_title(title, va='bottom')
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(0, 90)
        return ax

    def makePolarPlotForObjects(self, objects=None, withLines=False):
        if not objects:
            objects = self.stars
        objects = self._ensureList(objects)

        _ = plt.figure(figsize=(10, 10))

        for i, obj in enumerate(objects):
            seqNums = self.getSeqNumsMatching(target_name=obj)
            altAzes = [self.data[seqNum]['altaz_begin'] for seqNum in seqNums]
            alts = [altAz.alt.deg for altAz in altAzes if altAz is not None]
            azes = [altAz.az.deg for altAz in altAzes if altAz is not None]
            assert(len(alts) == len(azes))
            if len(azes) == 0:
                print(f"WARNING: found no alt/az data for {obj}")
            zens = [90 - alt for alt in alts]
            color = self.cMap[obj].color
            marker = self.cMap[obj].marker
            if withLines:
                marker += '-'

            ax = self._makePolarPlot(azes, zens, marker=marker, title=None, makeFig=False,
                                     color=color, objName=obj)
        lgnd = ax.legend(bbox_to_anchor=(1.05, 1), prop={'size': 15}, loc='upper left')
        for h in lgnd.legendHandles:
            size = 14
            if '-' in marker:
                size += 5
            h.set_markersize(size)
