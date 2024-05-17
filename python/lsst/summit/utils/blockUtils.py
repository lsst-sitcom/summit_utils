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
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from astropy.time import Time

from .efdUtils import efdTimestampToAstropy, getEfdData, makeEfdClient
from .enums import ScriptState

if TYPE_CHECKING:
    from .tmaUtils import TMAEvent

    try:
        from lsst_efd_client import EfdClient
    except ImportError:
        EfdClient = None  # this is currently just for mypy

__all__ = ("BlockParser", "BlockInfo", "ScriptStatePoint")


@dataclass(kw_only=True, frozen=True)
class BlockInfo:
    """Information about the execution of a "block".

    Each BlockInfo instance contains information about a single block
    execution. This is identified by the block number and sequence number,
    which, when combined with the dayObs, define the block ID.

    Each BlockInfo instance contains the following information:
        * The block ID - this is the primary identifier, as a string, for
          example "BL52_20230615_02", which is parsed into:
            * The block number, as an integer, for example 52, for "BLOCK-52".
            * The dayObs, as an integer, for example 20230615.
            * The seqNum - the execution number of that block on that day.
        * The begin and end times of the block execution, as astropy.time.Time
        * The SAL indices which were involved in the block execution, as a list
        * The SITCOM tickets which were involved in the block execution, as a
          list of strings, including the SITCOM- prefix.
        * The states of the script during the block execution, as a list of
          ``ScriptStatePoint`` instances.

    Parameters
    ----------
    blockNumber : `int`
        The block number, as an integer.
    blockId : `str`
        The block ID, as a string.
    dayObs : `int`
        The dayObs the block was run on.
    seqNum : `int`
        The sequence number of the block.
    begin : `astropy.time.Time`
        The time the block execution began.
    end : `astropy.time.Time`
        The time the block execution ended.
    salIndices : `list` of `int`
        One or more SAL indices, relating to the block.
    tickets : `list` of `str`
        One or more SITCOM tickets, relating to the block.
    states : `list` of `lsst.summit.utils.blockUtils.ScriptStatePoint`
        The states of the script during the block. Each element is a
        ``ScriptStatePoint`` which contains:
            - the time, as an astropy.time.Time
            - the state, as a ``ScriptState`` enum
            - the reason for state change, as a string, if present
    """

    blockNumber: int
    blockId: str
    dayObs: int
    seqNum: int
    begin: Time
    end: Time
    salIndices: list
    tickets: list
    states: list

    def __repr__(self) -> str:
        return (
            f"BlockInfo(blockNumber={self.blockNumber}, blockId={self.blockId}, salIndices={self.salIndices},"
            f" tickets={self.tickets}, states={self.states!r}"
        )

    def _ipython_display_(self) -> None:
        """This is the function which runs when someone executes a cell in a
        notebook with just the class instance on its own, without calling
        print() or str() on it.
        """
        print(self.__str__())

    def __str__(self) -> str:
        # no literal \n allowed inside {} portion of f-strings until python
        # 3.12, but it can go in via a variable
        newline = "  \n"
        return (
            f"dayObs: {self.dayObs}\n"
            f"seqNum: {self.seqNum}\n"
            f"blockNumber: {self.blockNumber}\n"
            f"blockId: {self.blockId}\n"
            f"begin: {self.begin.isot}\n"
            f"end: {self.end.isot}\n"
            f"salIndices: {self.salIndices}\n"
            f"tickets: {self.tickets}\n"
            f"states: \n{newline.join([str(state) for state in self.states])}"
        )


@dataclass(kw_only=True, frozen=True)
class ScriptStatePoint:
    """The execution state of a script at a point in time.

    Parameters
    ----------
    time : `astropy.time.Time`
        The time of the state change.
    state : `lsst.summit.utils.enums.ScriptState`
        The state of the script at this point in time.
    reason : `str`
        The reason for the state change, if given.
    """

    time: Time
    state: ScriptState
    reason: str

    def __repr__(self) -> str:
        return f"ScriptStatePoint(time={self.time!r}, state={self.state!r}, reason={self.reason!r})"

    def _ipython_display_(self) -> None:
        """This is the function which runs when someone executes a cell in a
        notebook with just the class instance on its own, without calling
        print() or str() on it.
        """
        print(self.__str__())

    def __str__(self) -> str:
        reasonStr = f" - {self.reason}" if self.reason else ""
        return f"{self.state.name:>10} @ {self.time.isot}{reasonStr}"


class BlockParser:
    """A class to parse BLOCK data from the EFD.

    Information on executed blocks is stored in the EFD (Electronic Facilities
    Database) in the ``lsst.sal.Script.logevent_state`` topic. This class
    parses that topic and provides methods to get information on the blocks
    which were run on a given dayObs. It also provides methods to get the
    events which occurred during a given block, and also to get the block in
    which a specified event occurred, if any.

    Parameters
    ----------
    dayObs : `int`
        The dayObs to get the block data for.
    client : `lsst_efd_client.efd_client.EfdClient`, optional
        The EFD client to use. If not specified, a new one is created.
    """

    def __init__(self, dayObs: int, client: EfdClient | None = None):
        self.log = logging.getLogger("lsst.summit.utils.blockUtils.BlockParser")
        self.dayObs = dayObs

        self.client = client
        if client is None:
            self.client = makeEfdClient()

        t0 = time.time()
        self.getDataForDayObs()
        self.log.debug(f"Getting data took {(time.time()-t0):.2f} seconds")
        t0 = time.time()
        self.augmentData()
        self.log.debug(f"Parsing data took {(time.time()-t0):.5f} seconds")

    def getDataForDayObs(self) -> None:
        """Retrieve the data for the specified dayObs from the EFD."""
        # Tiago thinks no individual block seqNums should take more than an
        # hour to run, so pad the dayObs by 1.5 hours to make sure we catch
        # any blocks which might span the end of the day.
        padding = 1.5 * 60 * 60
        data = getEfdData(
            self.client, "lsst.sal.Script.logevent_state", dayObs=self.dayObs, postPadding=padding
        )
        self.data = data

    def augmentDataSlow(self) -> None:
        """Parse each row in the data frame individually, pulling the
        information out into its own columns.
        """
        data = self.data
        blockPattern = r"BLOCK-(\d+)"
        blockIdPattern = r"BL\d+(?:_\w+)+"

        data["blockNum"] = pd.Series()
        data["blockId"] = pd.Series()
        data["blockDayObs"] = pd.Series()
        data["blockSeqNum"] = pd.Series()

        if "lastCheckpoint" not in self.data.columns:
            nRows = len(self.data)
            self.log.warning(
                f"Found {nRows} rows of data and no 'lastCheckpoint' column was in the data,"
                " so block data cannot be parsed."
            )

        for index, row in data.iterrows():
            rowStr = row["lastCheckpoint"]

            blockMatch = re.search(blockPattern, rowStr)
            blockNumber = int(blockMatch.group(1)) if blockMatch else None
            data.loc[index, "blockNum"] = blockNumber

            blockIdMatch = re.search(blockIdPattern, rowStr)
            blockId = blockIdMatch.group(0) if blockIdMatch else None
            data.loc[index, "blockId"] = blockId
            if blockId is not None:
                blockDayObs = int(blockId.split("_")[2])
                blockSeqNum = int(blockId.split("_")[3])
                data.loc[index, "blockDayObs"] = blockDayObs
                data.loc[index, "blockSeqNum"] = blockSeqNum

    def augmentData(self) -> None:
        """Parse the dataframe using vectorized methods, pulling the
        information out into its own columns.

        This method is much faster for large dataframes than augmentDataSlow,
        but is also much harder to maintain/debug, as the vectorized regexes
        are hard to work with, and to know which row is causing problems.
        """
        if "lastCheckpoint" not in self.data.columns:
            nRows = len(self.data)
            self.log.warning(
                f"Found {nRows} rows of data and no 'lastCheckpoint' column was in the data,"
                " so block data cannot be parsed."
            )
            # add the columns that would have been added for consistency
            self.data["blockNum"] = pd.Series()
            self.data["blockId"] = pd.Series()
            self.data["blockDayObs"] = pd.Series()
            self.data["blockSeqNum"] = pd.Series()
            return

        data = self.data
        blockPattern = r"BLOCK-(\d+)"
        blockIdPattern = r"(BL\d+(?:_\w+)+)"

        col = data["lastCheckpoint"]
        data["blockNum"] = col.str.extract(blockPattern, expand=False).astype(float).astype(pd.Int64Dtype())
        data["blockId"] = col.str.extract(blockIdPattern, expand=False)

        blockIdSplit = data["blockId"].str.split("_", expand=True)
        if blockIdSplit.columns.max() > 1:  # parsing the blockId succeeded
            data["blockDayObs"] = blockIdSplit[2].astype(float).astype(pd.Int64Dtype())
            data["blockSeqNum"] = blockIdSplit[3].astype(float).astype(pd.Int64Dtype())
        else:  # make nan filled columns for these
            nanSeries = pd.Series([np.nan] * len(data))
            data["blockDayObs"] = nanSeries
            data["blockSeqNum"] = nanSeries

    def _listColumnValues(self, column: str, removeNone: bool = True) -> list:
        """Get all the different values for the specified column, as a list.

        Parameters
        ----------
        column : `str`
            The column to get the values for.
        removeNone : `bool`
            Whether to remove None from the list of values.

        Returns
        -------
        values : `list`
            The values for the specified column.
        """
        values = set(self.data[column].dropna())
        if None in values and removeNone:
            values.remove(None)
        return sorted(values)

    def getBlockNums(self) -> list[int]:
        """Get the block numbers which were run on the specified dayObs.

        Returns
        -------
        blockNums : `list` of `int`
            The blocks which were run on the specified dayObs.
        """
        return self._listColumnValues("blockNum")

    def getSeqNums(self, block: int) -> list[int]:
        """Get the seqNums for the specified block.

        Parameters
        ----------
        block : `int`
            The block number to get the events for.

        Returns
        -------
        seqNums : `list` of `int`
            The sequence numbers for the specified block.
        """
        seqNums = self.data[self.data["blockNum"] == block]["blockSeqNum"]
        # block header rows have no blockId or seqNum, but do have a blockNum
        # so appear here, so drop the nans as they don't relate to an actual
        # run of a block
        seqNums = seqNums.dropna()
        return sorted(set(seqNums))

    def getRows(self, block: int, seqNum: int | None = None) -> pd.DataFrame:
        """Get all rows of data which relate to the specified block.

        If the seqNum is specified, only the rows for that sequence number are
        returned, otherwise all the rows relating to any block execution that
        day are returned. If the specified seqNum doesn't occur on the current
        day, an empty dataframe is returned.

        Parameters
        ----------
        block : `int`
            The block number to get the events for.
        seqNum : `int`, optional
            The sequence number, if specified, to get the row data for. If not
            specified, all data for the specified block is returned.

        Returns
        -------
        data : `pandas.DataFrame`
            The row data.
        """
        # Because we query for a whole dayObs, but BLOCKs can overlap the day
        # start/end, it's possible for the block's blockDayObs not to be the
        # same as self.dayObs around the beginning or end of the day, so filter
        # with an extra `& (self.data['blockDayObs'] == self.dayObs` when
        # getting the relevant rows.
        rowsForBlock = self.data[
            np.logical_and(self.data["blockNum"] == block, self.data["blockDayObs"] == self.dayObs)
        ]
        if rowsForBlock.empty:
            self.log.warning(f"No rows found for {block=} on dayObs={self.dayObs}")
        if seqNum is None:
            return rowsForBlock
        return rowsForBlock[rowsForBlock["blockSeqNum"] == seqNum]

    def printBlockEvolution(self, block: int, seqNum: int | None = None) -> None:
        """Display the evolution of the specified block.

        If the seqNum is specified, the evolution of that specific block
        exection is displayed, otherwise all executions of that block are
        printed.

        Parameters
        ----------
        block : `int`
            The block number to get the events for.
        seqNum : `int`, optional
            The sequence number, if specified, to print the evolution of. If
            not specified, all sequence numbers for the block are printed.
        """
        if seqNum is None:
            seqNums = self.getSeqNums(block)
        else:
            seqNums = [seqNum]
        print(f"Evolution of BLOCK {block} for dayObs={self.dayObs} {seqNum=}:")
        for seqNum in seqNums:
            blockInfo = self.getBlockInfo(block, seqNum)
            print(blockInfo, "\n")

    def getBlockInfo(self, block: int, seqNum: int) -> BlockInfo | None:
        """Get the block info for the specified block.

        Parses the rows relating to this block execution, and returns
        the information as a ``BlockInfo`` instance.

        Parameters
        ----------
        block : `int`
            The block number.
        seqNum : `int`
            The sequence number.

        Returns
        -------
        blockInfo : `lsst.summit.utils.blockUtils.BlockInfo`
            The block info.
        """
        rows = self.getRows(block, seqNum=seqNum)
        if rows.empty:
            print(f"No {seqNum=} on dayObs={self.dayObs} for {block=}")
            return None

        blockIds = set()
        tickets = set()
        salIndices = set()
        statePoints = []
        sitcomPattern = r"SITCOM-(\d+)"

        for index, row in rows.iterrows():
            salIndices.add(row["salIndex"])
            blockIds.add(row["blockId"])

            lastCheckpoint = row["lastCheckpoint"]
            sitcomMatches = re.findall(sitcomPattern, lastCheckpoint)
            tickets.update(sitcomMatches)

            time = efdTimestampToAstropy(row["private_efdStamp"])
            state = ScriptState(row["state"])
            reason = row["reason"]
            statePoint = ScriptStatePoint(time=time, state=state, reason=reason)
            statePoints.append(statePoint)

        # likewise for the blockIds
        if len(blockIds) > 1:
            raise RuntimeError(f"Found multiple blockIds ({blockIds}) for {seqNum=}")
        blockId = blockIds.pop()

        blockInfo = BlockInfo(
            blockNumber=block,
            blockId=blockId,
            dayObs=self.dayObs,
            seqNum=seqNum,
            begin=efdTimestampToAstropy(rows.iloc[0]["private_efdStamp"]),
            end=efdTimestampToAstropy(rows.iloc[-1]["private_efdStamp"]),
            salIndices=sorted(salIndices),
            tickets=[f"SITCOM-{ticket}" for ticket in sorted(tickets)],
            states=statePoints,
        )

        return blockInfo

    def getEventsForBlock(self, events: list[TMAEvent], block: int, seqNum: int) -> list[TMAEvent]:
        """Get the events which occurred during the specified block.

        Parameters
        ----------
        events : `list` of `lsst.summit.utils.tmaUtils.TMAEvent`
            The list of candidate events.
        block : `int`
            The block number to get the events for.
        seqNum : `int`
            The sequence number to get the events for.

        Returns
        -------
        events : `list` of `lsst.summit.utils.tmaUtils.TMAEvent`
            The events.
        """
        blockInfo = self.getBlockInfo(block, seqNum)
        if blockInfo is None:
            return []
        begin = blockInfo.begin
        end = blockInfo.end

        # each event's end being past the begin time and their
        # starts being before the end time means we get all the
        # events in the window and also those that overlap the
        # start/end too
        return [e for e in events if e.end >= begin and e.begin <= end]
