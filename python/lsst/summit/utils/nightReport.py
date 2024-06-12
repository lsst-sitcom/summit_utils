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

import datetime
import logging
import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astro_metadata_translator import ObservationInfo
from matplotlib.dates import date2num
from matplotlib.projections.polar import PolarAxes
from matplotlib.pyplot import cm

import lsst.daf.butler as dafButler
from lsst.utils.iteration import ensure_iterable

from .utils import getFieldNameAndTileNumber, obsInfoToDict

try:  # TODO: Remove post RFC-896: add humanize to rubin-env
    from humanize.time import precisedelta

    HAVE_HUMANIZE = True
except ImportError:
    # log a python warning about the lack of humanize
    logging.warning("humanize not available, install it to get better time printing")
    HAVE_HUMANIZE = False
    precisedelta = repr  # type: ignore


__all__ = ["NightReport"]

CALIB_VALUES = [
    "FlatField position",
    "Park position",
    "azel_target",
    "slew_icrs",
    "DaytimeCheckout001",
    "DaytimeCheckout002",
]
N_STARS_PER_SYMBOL = 6
MARKER_SEQUENCE = [
    "*",
    "o",
    "D",
    "P",
    "v",
    "^",
    "s",
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "8",
    "s",
    "p",
    "P",
    "*",
    "h",
    "H",
    "+",
    "x",
    "X",
    "D",
    "d",
    "|",
    "_",
]
SOUTHPOLESTAR = "HD 185975"


@dataclass
class ColorAndMarker:
    """Class for holding colors and marker symbols"""

    color: str
    marker: str = "*"


class NightReport:
    _version = 1

    def __init__(self, butler: dafButler.Butler, dayObs: int, loadFromFile: str | None = None):
        self._supressAstroMetadataTranslatorWarnings()  # call early
        self.log = logging.getLogger("lsst.summit.utils.NightReport")
        self.butler = butler
        self.dayObs = dayObs
        self.data: dict = dict()
        self._expRecordsLoaded: set = set()  # set of the expRecords loaded
        self._obsInfosLoaded: set = set()  # set of the seqNums loaded
        self.stars: list[str] = []
        self.cMap: dict[str, ColorAndMarker] = {}
        if loadFromFile is not None:
            self._load(loadFromFile)
        self.rebuild()  # sets stars and cMap

    def _supressAstroMetadataTranslatorWarnings(self) -> None:
        """NB: must be called early"""
        logging.basicConfig()
        logger = logging.getLogger("lsst.obs.lsst.translators.latiss")
        logger.setLevel(logging.ERROR)
        logger = logging.getLogger("astro_metadata_translator.observationInfo")
        logger.setLevel(logging.ERROR)

    def save(self, filename: str) -> None:
        """Save the internal data to a file.

        Parameters
        ----------
        filename : `str`
            The full name and path of the file to save to.
        """
        toSave = dict(
            data=self.data,
            _expRecordsLoaded=self._expRecordsLoaded,
            _obsInfosLoaded=self._obsInfosLoaded,
            dayObs=self.dayObs,
            version=self._version,
        )
        with open(filename, "wb") as f:
            pickle.dump(toSave, f, pickle.HIGHEST_PROTOCOL)

    def _load(self, filename: str) -> None:
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
        self.data = loaded["data"]
        self._expRecordsLoaded = loaded["_expRecordsLoaded"]
        self._obsInfosLoaded = loaded["_obsInfosLoaded"]
        dayObs = loaded["dayObs"]
        loadedVersion = loaded.get("version", 0)

        if dayObs != self.dayObs:
            raise RuntimeError(f"Loaded data is for {dayObs} but current dayObs is {self.dayObs}")
        if loadedVersion < self._version:
            self.log.critical(
                f"Loaded version is {loadedVersion} but current version is {self._version}."
                " Check carefully for compatibility issues/regenerate your saved report!"
            )
            # update to the version on the instance in case the report is
            # re-saved.
            self._version = loadedVersion
        assert len(self.data) == len(self._expRecordsLoaded)
        assert len(self.data) == len(self._obsInfosLoaded)
        self.log.info(f"Loaded {len(self.data)} records from {filename}")

    @staticmethod
    def _getSortedData(data: dict) -> dict:
        """Get a sorted copy of the internal data."""
        if list(data.keys()) == sorted(data.keys()):
            return data
        else:
            return {k: data[k] for k in sorted(data.keys())}

    def getExpRecordDictForDayObs(self, dayObs: int) -> dict:
        """Get all the exposureRecords as dicts for the current dayObs.

        Notes
        -----
        Runs in ~0.05s for 1000 records.
        """
        expRecords = self.butler.registry.queryDimensionRecords(
            "exposure", where="exposure.day_obs=dayObs", bind={"dayObs": dayObs}, datasets="raw"
        )
        listExpRecords = list(expRecords)
        records = {e.seq_num: e.toDict() for e in listExpRecords}  # not guaranteed to be in order
        for record in records.values():
            target = record["target_name"] if record["target_name"] is not None else ""
            if target:
                shortTarget, _ = getFieldNameAndTileNumber(target, warn=False)
            else:
                shortTarget = ""
            record["target_name_short"] = shortTarget
        return self._getSortedData(records)

    def getObsInfoAndMetadataForSeqNum(self, seqNum: int) -> tuple[ObservationInfo, dict]:
        """Get the obsInfo and metadata for a given seqNum.

        TODO: Once we have a summit repo containing all this info, remove this
        method and all scraping of headers! Probably also remove the save/load
        functionalty there too, as the whole init will go from many minutes to
        under a second.

        Parameters
        ----------
        seqNum : `int`
            The seqNum.

        Returns
        -------
        obsInfo : `astro_metadata_translator.ObservationInfo`
            The obsInfo.
        md : `dict`
            The raw metadata.

        Notes
        -----
        Very slow, as it has to load the whole file on object store repos
        and access the file on regular filesystem repos.
        """
        dataId = {"day_obs": self.dayObs, "seq_num": seqNum, "detector": 0}
        md = self.butler.get("raw.metadata", dataId)
        return ObservationInfo(md), md.toDict()

    def rebuild(self, full: bool = False) -> None:
        """Scrape new data if there is any, otherwise is a no-op.

        If full is True, then all data is reloaded.

        Parameters
        ----------
        full : `bool`, optional
            Do a full reload of all the data, removing any which is pre-loaded?
        """
        if full:
            self.data = dict()
            self._expRecordsLoaded = set()
            self._obsInfosLoaded = set()

        records = self.getExpRecordDictForDayObs(self.dayObs)
        if len(records) == len(self.data):  # nothing to do
            self.log.info("No new records found")
            # NB don't return here, because we need to rebuild the
            # star maps etc if we came from a file.
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
            self.log.info(f"Loading {len(obsInfosToLoad)} obsInfo(s)")
        for i, seqNum in enumerate(obsInfosToLoad):
            if (i + 1) % 200 == 0:
                self.log.info(f"Loaded {i+1} obsInfos")
            obsInfo, metadata = self.getObsInfoAndMetadataForSeqNum(seqNum)
            obsInfoDict = obsInfoToDict(obsInfo)
            records[seqNum].update(obsInfoDict)
            # _raw_metadata item will hopefully not be needed in the future
            # but add it while we have it for free, as it has DIMM seeing
            records[seqNum]["_raw_metadata"] = metadata
            self._obsInfosLoaded.add(seqNum)

        self.data = self._getSortedData(self.data)  # make sure we stay sorted
        self.stars = self.getObservedObjects()
        self.cMap = self.makeStarColorAndMarkerMap(self.stars)

    def getDatesForSeqNums(self) -> dict[int, datetime.datetime]:
        """Get a dict of {seqNum: date} for the report.

        Returns
        -------
        dates : `dict[int, datetime.datetime]`
            Dict of {seqNum: date} for the current report.
        """
        return {
            seqNum: self.data[seqNum]["timespan"].begin.to_datetime() for seqNum in sorted(self.data.keys())
        }

    def getObservedObjects(self, ignoreTileNum: bool = True) -> list[str]:
        """Get a list of the observed objects for the night.

        Repeated observations of individual imaging fields have _NNN appended
        to the field name. Use ``ignoreTileNum`` to remove these, collapsing
        the observations of the field to a single target name.

        Parameters
        ----------
        ignoreTileNum : `bool`, optional
            Remove the trailing _NNN tile number for imaging fields?
        """
        key = "target_name_short" if ignoreTileNum else "target_name"
        allTargets = sorted({record[key] if record[key] is not None else "" for record in self.data.values()})
        return allTargets

    def getSeqNumsMatching(
        self, invert: bool = False, subset: list[int] | None = None, **kwargs: str
    ) -> list[int]:
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

    def printAvailableKeys(self, sample: bool = False, includeRaw: bool = False) -> None:
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
                for k in recordDict["_raw_metadata"]:
                    print(k)
            break

    @staticmethod
    def makeStarColorAndMarkerMap(stars: list[str]) -> dict[str, ColorAndMarker]:
        """Create a color/marker map for a list of observed objects."""
        markerMap = {}
        # mypy doesn't recognize dynamically created colormap attributes
        colors = cm.rainbow(np.linspace(0, 1, N_STARS_PER_SYMBOL))  # type: ignore
        for i, star in enumerate(stars):
            markerIndex = i // (N_STARS_PER_SYMBOL)
            colorIndex = i % (N_STARS_PER_SYMBOL)
            markerMap[star] = ColorAndMarker(colors[colorIndex], MARKER_SEQUENCE[markerIndex])
        return markerMap

    def calcShutterTimes(self) -> dict | None:
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
        firstObs = self.getObservingStartSeqNum(method="heuristic")
        if not firstObs:
            self.log.warning("No on-sky observations found.")
            return None
        lastObs = max(self.data.keys())

        begin = self.data[firstObs]["datetime_begin"]
        end = self.data[lastObs]["datetime_end"]

        READOUT_TIME = 2.0
        shutterOpenTime = sum([self.data[s]["exposure_time"] for s in range(firstObs, lastObs + 1)])
        readoutTime = sum([READOUT_TIME for _ in range(firstObs, lastObs + 1)])

        sciSeqNums = self.getSeqNumsMatching(observation_type="science")
        scienceIntegration = sum([self.data[s]["exposure_time"] for s in sciSeqNums])
        scienceTimeTotal = scienceIntegration.value + (len(sciSeqNums) * READOUT_TIME)

        result: dict[str, float | int] = {}
        result["firstObs"] = firstObs
        result["lastObs"] = lastObs
        result["startTime"] = begin
        result["endTime"] = end
        result["nightLength"] = (end - begin).sec  # was a datetime.timedelta
        result["shutterOpenTime"] = shutterOpenTime.value  # was an Quantity
        result["readoutTime"] = readoutTime
        result["scienceIntegration"] = scienceIntegration.value  # was an Quantity
        result["scienceTimeTotal"] = scienceTimeTotal

        return result

    def printShutterTimes(self) -> None:
        """Print out the shutter efficiency stats in a human-readable
        format.
        """
        if not HAVE_HUMANIZE:
            self.log.warning("Please install humanize to make this print as intended.")
        timings = self.calcShutterTimes()
        if not timings:
            print("No on-sky observations found, so no shutter efficiency stats are available yet.")
            return

        print(
            f"Observations started at: seqNum {timings['firstObs']:>3} at"
            f" {timings['startTime'].to_datetime().strftime('%H:%M:%S')} TAI"
        )
        print(
            f"Observations ended at:   seqNum {timings['lastObs']:>3} at"
            f" {timings['endTime'].to_datetime().strftime('%H:%M:%S')} TAI"
        )
        print(f"Total time on sky: {precisedelta(timings['nightLength'])}")
        print()
        print(f"Shutter open time: {precisedelta(timings['shutterOpenTime'])}")
        print(f"Readout time: {precisedelta(timings['readoutTime'])}")
        engEff = 100 * (timings["shutterOpenTime"] + timings["readoutTime"]) / timings["nightLength"]
        print(f"Engineering shutter efficiency = {engEff:.1f}%")
        print()
        print(f"Science integration: {precisedelta(timings['scienceIntegration'])}")
        sciEff = 100 * (timings["scienceTimeTotal"] / timings["nightLength"])
        print(f"Science shutter efficiency = {sciEff:.1f}%")

    def getTimeDeltas(self) -> dict[int, int]:
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
            dt = self.data[seqNum]["datetime_begin"] - self.data[(seqNums[i])]["datetime_end"]
            dts.append(dt.sec)

        return {s: dt for s, dt in zip(seqNums, dts)}

    def printObsGaps(self, threshold: float | int = 100, includeCalibs: bool = False) -> None:
        """Print out the gaps between observations in a human-readable format.

        Prints the most recent gaps first.

        Parameters
        ----------
        threshold : `float`, optional
            The minimum time gap to print out, in seconds.
        includeCalibs : `bool`, optional
            If True, start at the lowest seqNum, otherwise start when the
            night's observing started.
        """
        if not HAVE_HUMANIZE:
            self.log.warning("Please install humanize to make this print as intended.")
        dts = self.getTimeDeltas()

        allSeqNums = list(self.data.keys())
        if includeCalibs:
            seqNums = allSeqNums
        else:
            firstObs = self.getObservingStartSeqNum(method="heuristic")
            if not firstObs:
                print("No on-sky observations found, so there can be no gaps in observing yet.")
                return
            # there is always a big gap before firstObs by definition so add 1
            startPoint = allSeqNums.index(firstObs) + 1
            seqNums = allSeqNums[startPoint:]

        messages = []
        for seqNum in reversed(seqNums):
            dt = dts[seqNum]
            if dt > threshold:
                messages.append(f"seqNum {seqNum:3}: {precisedelta(dt)} gap")

        if messages:
            print(f"Gaps between observations greater than {threshold}s:")
            for line in messages:
                print(line)

    def getObservingStartSeqNum(self, method: str = "safe") -> int | None:
        """Get the seqNum at which on-sky observations started.

        If no on-sky observations were taken ``None`` is returned.

        Parameters
        ----------
        method : `str`
            The calculation method to use. Options are:
            - 'safe': Use the first seqNum with an observation_type that is
            explicitly not a calibration or test. This is a safe way of
            excluding the calibs, but will include observations where we
            take some closed dome test images, or start observing too early,
            and go back to taking calibs for a while before the night starts.
            - 'heuristic': Use a heuristic to find the first seqNum. The
            current heuristic is to find the first seqNum with an observation
            type of CWFS, as we always do a CWFS focus before going on sky.
            This does not work well for old days, because this wasn't always
            the way data was taken. Note: may be updated in the future, at
            which point this will be renamed ``cwfs``.

        Returns
        -------
        startSeqNum : `int`
            The seqNum of the start of the night's observing.
        """
        allowedMethods = ["heuristic", "safe"]
        if method not in allowedMethods:
            raise ValueError(f"Method must be one of {allowedMethods}")

        if method == "safe":
            # as of 20221211, the full set of observation_types ever seen is:
            # acq, bias, cwfs, dark, engtest, flat, focus, science, stuttered,
            # test, unknown
            offSkyObsTypes = ["bias", "dark", "flat", "test", "unknown"]
            for seqNum in sorted(self.data.keys()):
                if self.data[seqNum]["observation_type"] not in offSkyObsTypes:
                    return seqNum
            return None

        if method == "heuristic":
            # take the first cwfs image and return that
            seqNums = self.getSeqNumsMatching(observation_type="cwfs")
            if not seqNums:
                self.log.warning("No cwfs images found, observing is assumed not to have started.")
                return None
            return min(seqNums)
        return None

    def printObsTable(self, **kwargs: Any) -> None:
        """Print a table of the days observations.

        Parameters
        ----------
        **kwargs : `dict`
            Filter the observation table according to seqNums which match these
            {k: v} pairs. For example, to only print out science observations
            pass ``observation_type='science'``.
        """
        seqNums = self.data.keys() if not kwargs else self.getSeqNumsMatching(**kwargs)
        seqNums = sorted(seqNums)  # should always be sorted, but is a total disaster here if not

        dts = self.getTimeDeltas()
        lines = []
        for seqNum in seqNums:
            try:
                expTime = self.data[seqNum]["exposure_time"].value
                imageType = self.data[seqNum]["observation_type"]
                target = self.data[seqNum]["target_name"]
                deadtime = dts[seqNum]
                filt = self.data[seqNum]["physical_filter"]

                msg = f"{seqNum} {target} {expTime:.1f} {deadtime:.02f} {imageType} {filt}"
            except Exception:
                msg = f"Error parsing {seqNum}!"
            lines.append(msg)

        print(r"seqNum target expTime deadtime imageType filt")
        print(r"------ ------ ------- -------- --------- ----")
        for line in lines:
            print(line)

    def getExposureMidpoint(self, seqNum: int) -> datetime.datetime:
        """Return the midpoint of the exposure as a float in MJD.

        Parameters
        ----------
        seqNum : `int`
            The seqNum to get the midpoint for.

        Returns
        -------
        midpoint : `datetime.datetime`
            The midpoint, as a python datetime object.
        """
        timespan = self.data[seqNum]["timespan"]
        expTime = self.data[seqNum]["exposure_time"]
        return ((timespan.begin) + expTime / 2).to_datetime()

    def plotPerObjectAirMass(
        self, objects: Iterable[str] | None = None, airmassOneAtTop: bool = True, saveFig: str = ""
    ) -> matplotlib.figure.Figure:
        """Plot the airmass for objects observed over the course of the night.

        Parameters
        ----------
        objects : `list` [`str`], optional
            The objects to plot. If not provided, all objects are plotted.
        airmassOneAtTop : `bool`, optional
            Put the airmass of 1 at the top of the plot, like astronomers
            expect.
        saveFig : `str`, optional
            Save the figure to this file path?

        Return
        ------
        fig : `matplotlib.figure.Figure`
            The figure object.
        """
        if not objects:
            objects = self.stars

        objects = ensure_iterable(objects)

        fig = plt.figure(figsize=(16, 12))
        for star in objects:
            if star in CALIB_VALUES:
                continue
            seqNums = self.getSeqNumsMatching(target_name_short=star)
            airMasses = [self.data[seqNum]["boresight_airmass"] for seqNum in seqNums]
            obsTimes = [self.getExposureMidpoint(seqNum) for seqNum in seqNums]
            color = self.cMap[star].color
            marker = self.cMap[star].marker
            obsDates = date2num(obsTimes)
            plt.plot(obsDates, airMasses, color=color, marker=marker, label=star, ms=10, ls="")

        plt.ylabel("Airmass", fontsize=20)
        plt.xlabel("Time (UTC)", fontsize=20)
        plt.xticks(rotation=25, horizontalalignment="right")

        ax = plt.gca()
        xfmt = matplotlib.dates.DateFormatter("%m-%d %H:%M:%S")
        ax.xaxis.set_major_formatter(xfmt)

        if airmassOneAtTop:
            ax.set_ylim(ax.get_ylim()[::-1])

        plt.legend(bbox_to_anchor=(1, 1.025), prop={"size": 15}, loc="upper left")

        plt.tight_layout()
        if saveFig:
            plt.savefig(saveFig)
        return fig

    def _makePolarPlot(
        self,
        azimuthsInDegrees: list[float],
        zenithAngles: list[float],
        marker: str = "*-",
        title: str | None = None,
        makeFig: bool = True,
        color: str | None = None,
        objName: str | None = None,
    ) -> matplotlib.axes.Axes:
        """Private method to actually do the polar plotting.

        azimuthsInDegrees : `list` [`float`]
            The azimuth values, in degrees.
        zenithAngles : `list` [`float`]
            The zenith angle values, but more generally, the values on the
            radial axis, so can be in whatever units you want.
        marker : `str`, optional
            The marker to use.
        title : `str`, optional
            The plot title.
        makeFig : `bool`, optional
            Make a new figure?
        color : `str`, optional
            The marker color.
        objName : `str`, optional
            The object name, for the legend.

        Returns
        -------
        ax : `matplotlib.axes.Axe`
            The axes on which the plot was made.
        """
        if makeFig:
            _ = plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)
        assert isinstance(ax, PolarAxes), "Expected a polar plot"
        ax.plot([a * np.pi / 180 for a in azimuthsInDegrees], zenithAngles, marker, c=color, label=objName)
        if title:
            ax.set_title(title, va="bottom")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlim(0, 90)
        return ax

    def makeAltAzCoveragePlot(
        self, objects: Iterable[str] | None = None, withLines: bool = False, saveFig: str = ""
    ) -> matplotlib.figure.Figure:
        """Make a polar plot of the azimuth and zenith angle for each object.

        Plots the azimuth on the theta axis, and zenith angle (not altitude!)
        on the radius axis, such that 0 is at the centre, like you're looking
        top-down on the telescope.

        Parameters
        ----------
        objects : `list` [`str`], optional
            The objects to plot. If not provided, all objects are plotted.
        withLines : `bool`, optional
            Connect the points with lines?
        saveFig : `str`, optional
            Save the figure to this file path?

        Return
        ------
        fig : `matplotlib.figure.Figure`
            The figure object.
        """
        if not objects:
            objects = self.stars
        objects = ensure_iterable(objects)

        fig = plt.figure(figsize=(16, 12))

        for obj in objects:
            if obj in CALIB_VALUES:
                continue
            seqNums = self.getSeqNumsMatching(target_name_short=obj)
            altAzes = [self.data[seqNum]["altaz_begin"] for seqNum in seqNums]
            alts = [altAz.alt.deg for altAz in altAzes if altAz is not None]
            azes = [altAz.az.deg for altAz in altAzes if altAz is not None]
            assert len(alts) == len(azes)
            if len(azes) == 0:
                self.log.warning(f"Found no alt/az data for {obj}")
            zens = [90 - alt for alt in alts]
            color = self.cMap[obj].color
            marker = self.cMap[obj].marker
            if withLines:
                marker += "-"

            ax = self._makePolarPlot(
                azes, zens, marker=marker, title=None, makeFig=False, color=color, objName=obj
            )
        lgnd = ax.legend(bbox_to_anchor=(1.05, 1), prop={"size": 15}, loc="upper left")
        ax.set_title("Axial coverage - azimuth (theta, deg) vs zenith angle (r, deg)", size=20)

        for h in lgnd.legend_handles:
            if h is None or not isinstance(h, matplotlib.lines.Line2D):
                continue
            size = 14
            if "-" in marker:
                size += 5
            h.set_markersize(size)

        plt.tight_layout()
        if saveFig:
            plt.savefig(saveFig)
        return fig
