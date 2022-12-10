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

from astro_metadata_translator import ObservationInfo
from lsst.summit.utils.utils import (obsInfoToDict,  # change to .utils later XXX
                                     getFieldNameAndTileNumber
                                     )

try:  # TODO: Remove post RFC-896: add humanize to rubin-env
    from humanize.time import precisedelta
    HAVE_HUMANIZE = True
except ImportError:
    # log a python warning about the lack of humanize
    logging.warning("humanize not available, install it to get better time printing")
    HAVE_HUMANIZE = False
    precisedelta = repr


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
            self._load(loadFromFile)
        self.rebuild()

    def save(self, filename):
        """Save the internal data to a file.

        Parameters
        ----------
        filename : `str`
            The full name and path of the file to save to.
        """
        toSave = (self.data, self._expRecordsLoaded, self._obsInfosLoaded, self.dayObs)
        with open(filename, "wb") as f:
            pickle.dump(toSave, f, pickle.HIGHEST_PROTOCOL)

    def _load(self, filename):
        """Load the report data from a file.

        Called on init if loadFromFile is not None. Should not be used directly
        as other things are populated on load in the __init__.

        Parameters
        ----------
        filename : `str`
            The full name and path of the file to load from.
        """
        with open(filename, "rb") as f:
            loaded = pickle.load(f)
        self.data, self._expRecordsLoaded, self._obsInfosLoaded, dayObs = loaded
        if dayObs != self.dayObs:
            raise RuntimeError(f"Loaded data is for {dayObs} but current dayObs is {self.dayObs}")
        assert len(self.data) == len(self._expRecordsLoaded)
        assert len(self.data) == len(self._obsInfosLoaded)
        self.log.info(f"Loaded {len(self.data)} records from {filename}")

    @staticmethod
    def _getSortedData(data):
        """Get a sorted copy of the internal data.
        """
        if list(data.keys()) == sorted(data.keys()):
            return data
        else:
            return {k: data[k] for k in sorted(data.keys())}

    def getExpRecordDictForDayObs(self, dayObs):
        """Get all the exposureRecords as dicts for the current dayObs.

        Notes
        -----
        Runs in ~0.05s for 1000 records.
        """
        expRecords = self.butler.registry.queryDimensionRecords("exposure",
                                                                where="exposure.day_obs=day_obs",
                                                                bind={'day_obs': dayObs},
                                                                datasets='raw')
        expRecords = list(expRecords)
        records = {e.seq_num: e.toDict() for e in expRecords}  # not guaranteed to be in order
        return self._getSortedData(records)

    def getObsInfoAndMetadataForSeqNum(self, seqNum):
        """Get the obsInfo and metadata for a given seqNum.

        Notes
        -----
        Very slow, as it has to load the whole file on object store repos
        and access the file on regular filesystem repos.
        """
        dataId = {'day_obs': self.dayObs, 'seq_num': seqNum, 'detector': 0}
        md = self.butler.get('raw.metadata', dataId)
        return ObservationInfo(md), md

    def rebuild(self, full=False):
        """Scrape new data if there is any, otherwise is a no-op.

        If full is True, then all data is reloaded.
        """
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

        self.data = self._getSortedData(self.data)  # make sure we stay sorted
        self.stars = self.getObservedObjects()
        self.cMap = self.makeStarColorAndMarkerMap(self.stars)

    def getObservedObjects(self, ignoreTileNum=True):
        """Get a list of the observed objects for the night.

        Repeated observations of individual imaging fields have _NNN appended
        to the field name. Use ``ignoreTileNum`` to remove these, collapsing
        the observations of the field to a single target name.

        Parameters
        ----------
        ignoreTileNum : `bool`, optional
            Remove the trailing _NNN tile number for imaging fields?
        """
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
        # copy data so we can pop, and restrict to subset if provided
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

        If sample is True, then a sample value for each key is printed too,
        which is useful for dealing with types and seeing what each item
        actually means.
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
        """Create a color/marker map for a list of observed objects.
        """
        markerMap = {}
        colors = cm.rainbow(np.linspace(0, 1, N_STARS_PER_SYMBOL))
        for i, star in enumerate(stars):
            markerIndex = i//(N_STARS_PER_SYMBOL)
            colorIndex = i%(N_STARS_PER_SYMBOL)
            markerMap[star] = ColorAndMarker(colors[colorIndex], MARKER_SEQUENCE[markerIndex])
        return markerMap

    @staticmethod
    def _ensureList(arg):
        """Ensure that if a string is passed rather than a list of strings,
        that we don't iterate over the letters in the string, but treat it as
        a list of length 1.
        """
        if type(arg) == str:
            return [arg]
        if type(arg) != list:
            raise ValueError(f"Expected a list or string, got {arg}")
        return arg

    def calcShutterTimes(self):
        """Calculate the total time spent on science, engineering and readout.

        Science and engineering time both include the time spent on readout,
        such that if images were taken all night with no downtime and no slews
        the efficiency would be 100%.

        Returns
        -------
        timings : `dict`
            Dictionary of the various calculated times, in seconds, and the
            seqNums of the first and last observations used in the calculation.
        """
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

        result = {}
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
        if not HAVE_HUMANIZE:
            self.log.warning('Please install humanize to use make this print as intended.')
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
        """Returns a dict, keyed by seqNum, of the time since the end of the
        last integration. The time since does include the readout, so is always
        greater than or equal to the readout time.

        Returns
        -------
        timeGaps : `dict`
            Dictionary of the time gaps, in seconds, keyed by seqNum.
        """
        seqNums = list(self.data.keys())  # need a list not a generator, and NB it might not be contiguous!
        dts = [0]  # first item is zero by definition
        for i, seqNum in enumerate(seqNums[1:]):
            dt = self.data[seqNum]['datetime_begin'] - self.data[(seqNums[i])]['datetime_end']
            dts.append(dt.sec)

        return {s: dt for s, dt in zip(seqNums, dts)}

    def printObsGaps(self, threshold=100):
        """Print out the gaps between observations in a human-readable format.

        Parameters
        ----------
        threshold : `float`
            The minimum time gap to print out, in seconds.
        """
        if not HAVE_HUMANIZE:
            self.log.warning('Please install humanize to use make this print as intended.')
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
            if dt > threshold:
                messages.append(f"seqNum {seqNum:3}: {precisedelta(dt)} gap")

        if messages:
            print(f"Gaps between observations greater than {threshold}s:")
            for line in messages:
                print(line)

    def getNightStartSeqNum(self, method='safe'):
        """Get the seqNum at which on-sky observations started.

        Parameters
        ----------
        method : `str`
            The calculation method to use. Options are:
            - 'safe': Use the first seqNum with an observation_type of
              'science'.
            - 'heuristic': Use a heuristic to find the first seqNum. Not yet
               implemented.

        Returns
        -------
        startSeqNum : `int`
            The seqNum of the start of the night's observing.
        """

        allowedMethods = ['heuristic', 'safe']
        if method not in allowedMethods:
            raise ValueError(f"Method must be one of {allowedMethods}")

        if method == 'heuristic':
            raise NotImplementedError("Heuristic method not yet implemented.")

        if method == 'safe':
            # take the first cwfs image and return that
            seqNums = self.getSeqNumsMatching(observation_type='cwfs')
            return min(seqNums)

    def printObsTable(self, **kwargs):
        """Print a table of the days observations.

        Parameters
        ----------
        **kwargs : `dict`
            Filter the observation table according to seqNums which match these
            {k: v} pairs. For example, to only print out science observations
            pass ``observation_type='science'``.
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

        Parameters
        ----------
        seqNum : `int`
            The seqNum to get the midpoint for.

        Returns
        -------
        midpointMjd : `float`
            The midpoint, as an mjd float.
        """
        timespan = self.data[seqNum]['timespan']
        return (timespan.begin.mjd + timespan.begin.mjd) / 2

    def plotPerObjectAirMass(self, objects=None, airmassOneAtTop=True):
        """Plot the airmass for objects observed over the course of the night.

        TODO: Add axis labels to the actual plot
        TODO: Add option to save to file

        Parameters
        ----------
        objects : `list` [`str`], optional
            The objects to plot. If not provided, all objects are plotted.
        airmassOneAtTop : `bool`, optional
            Put the airmass of 1 at the top of the plot, like astronomers
            expect.
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

    def makeAltAzCoveragePlot(self, objects=None, withLines=False):
        """Make a polar plot of the azimuth and zenith angle for each object.

        Plots the azimuth on the theta axis, and zenith angle (not altitude!)
        on the radius axis, such that 0 is at the centre, like you're looking
        top-down on the telescope.

        TODO: Add axis labels to the actual plot
        TODO: Add option to save to file

        Parameters
        ----------
        objects : `list` [`str`], optional
            The objects to plot. If not provided, all objects are plotted.
        withLines : `bool`, optional
            Connect the points with lines?
        """
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
