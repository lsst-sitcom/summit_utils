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

import re
import time
import logging
import pandas as pd
from dataclasses import dataclass
from astropy.time import Time

from .enums import ScriptState
from .efdUtils import (getEfdData,
                       makeEfdClient,
                       efdTimestampToAstropy,
                       )

__all__ = (
    'BlockParser',
    'BlockInfo',
    'ScriptStatePoint'
)


@dataclass(kw_only=True, frozen=True)
class BlockInfo:
    """Information about the execution of a "block".

    Each BlockInfo instance contains information about a single block
    execution. This is identified by the block number and sequence number,
    which, when combined with the dayObs, are exactly degenerate with the
    blockId.

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
    salIndices: int
    tickets: list
    states: list

    def __repr__(self):
        return (
            f"BlockInfo(blockNumber={self.blockNumber}, blockId={self.blockId}, salIndices={self.salIndices},"
            f" tickets={self.tickets}, states={self.states!r}"
        )

    def _ipython_display_(self):
        """This is the function which runs when someone executes a cell in a
        notebook with just the class instance on its own, without calling
        print() or str() on it.
        """
        print(self.__str__())

    def __str__(self):
        newline = '  \n'  # no \n allowed in f-strings until python 3.12
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

    def __repr__(self):
        return (
            f"ScriptStatePoint(time={self.time!r}, state={self.state!r}, reason={self.reason!r})"
        )

    def _ipython_display_(self):
        """This is the function which runs when someone executes a cell in a
        notebook with just the class instance on its own, without calling
        print() or str() on it.
        """
        print(self.__str__())

    def __str__(self):
        reasonStr = f" - {self.reason}" if self.reason else ""
        return (f"{self.state.name:>10} @ {self.time.isot}{reasonStr}")


class BlockParser:
    """A class to parse BLOCK data from the EFD.

    Information on executed blocks is stored in the EFD in the
    ``lsst.sal.Script.logevent_state`` topic. This class parses that topic and
    provides methods to get information on the blocks which were run on a given
    dayObs. It also provides methods to get the events which occurred during a
    given block, and also to get the block in which a specified event occurred,
    if any.

    Parameters
    ----------
    dayObs : `int`
        The dayObs to get the block data for.
    client : `lsst_efd_client.efd_client.EfdClient`, optional
        The EFD client to use. If not specified, a new one is created.
    """

    def __init__(self, dayObs, client=None):
        # TODO change mode of operation to not take dayObs on init, but instead
        # to work like the TMAEventMaker where the EFD data is cached as long
        # as the day isn't current.
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
        # self.augmentDataSlow()
        self.log.debug(f"Parsing data took {(time.time()-t0):.5f} seconds")

    def getDataForDayObs(self):
        """Retrieve the data for the specified dayObs from the EFD.
        """
        data = getEfdData(self.client, 'lsst.sal.Script.logevent_state', dayObs=self.dayObs)
        self.data = data

    def augmentDataSlow(self):
        """Parse each row in the data frame individually, pulling the
        information out into its own columns.
        """
        data = self.data
        blockPattern = r"BLOCK-(\d+)"
        blockIdPattern = r"BL\d+(?:_\w+)+"

        data['blockNum'] = pd.Series()
        data['blockId'] = pd.Series()
        data['blockDayObs'] = pd.Series()
        data['blockSeqNum'] = pd.Series()

        for index, row in data.iterrows():
            rowStr = row['lastCheckpoint']

            blockMatch = re.search(blockPattern, rowStr)
            blockNumber = int(blockMatch.group(1)) if blockMatch else None
            data.loc[index, 'blockNum'] = blockNumber

            blockIdMatch = re.search(blockIdPattern, rowStr)
            blockId = blockIdMatch.group(0) if blockIdMatch else None
            data.loc[index, 'blockId'] = blockId
            if blockId:
                blockDayObs = int(blockId.split('_')[2])
                blockSeqNum = int(blockId.split('_')[3])
                data.loc[index, 'blockDayObs'] = blockDayObs
                data.loc[index, 'blockSeqNum'] = blockSeqNum

    def augmentData(self):
        """Parse the dataframe using vectorized methods, pulling the
        information out into its own columns.

        This method is much faster for large dataframes than augmentDataSlow,
        but is also much harder to maintain/debug, as the vectorized regexes
        are hard to work with, and to know which row is causing problems.
        """
        data = self.data
        blockPattern = r"BLOCK-(\d+)"
        blockIdPattern = r"(BL\d+(?:_\w+)+)"

        col = data['lastCheckpoint']
        data['blockNum'] = col.str.extract(blockPattern, expand=False).astype(float).astype(pd.Int64Dtype())
        data['blockId'] = col.str.extract(blockIdPattern, expand=False)

        # TODO: add SITCOM tickets, making sure it does the equivalent of
        # re.findall and add them delimited somehow data['sitcomTickets'] =
        # data['lastCheckpoint'].str.extract(sitcomPattern, expand=False)

        blockIdSplit = data['blockId'].str.split('_', expand=True)
        data['blockDayObs'] = blockIdSplit[2].astype(float).astype(pd.Int64Dtype())
        data['blockSeqNum'] = blockIdSplit[3].astype(float).astype(pd.Int64Dtype())

    def _listColumnValues(self, column, removeNone=True):
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

    def getBlockNums(self):
        """Get the block numbers which were run on the specified dayObs.

        Returns
        -------
        blockNums : `list` of `int`
            The blocks which were run on the specified dayObs.
        """
        return self._listColumnValues('blockNum')

    def getSeqNums(self, block):
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
        return sorted(set(self.data[self.data['blockNum'] == block]['blockSeqNum']))

    def getRows(self, block, seqNum=None):
        """Get all rows of data which relate to the specified block.

        If the seqNum is specified, only the rows for that sequence number are
        returned, otherwise all the rows relating to any block execution that
        day are returned.

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
        rowsForBlock = self.data[self.data['blockNum'] == block]
        if seqNum is None:
            return rowsForBlock
        return rowsForBlock[rowsForBlock['blockSeqNum'] == seqNum]

    def printBlockEvolution(self, block, seqNum=None):
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
        print(f'Evolution of BLOCK {block} for dayObs={self.dayObs} {seqNum=}:')
        for seqNum in seqNums:
            blockInfo = self.getBlockInfo(block, seqNum)
            print(blockInfo, '\n')

    def getBlockInfo(self, block, seqNum):
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
            print(f'No {seqNum=} on dayObs={self.dayObs} for {block=}')
            return

        blockIds = set()
        tickets = set()
        salIndices = set()
        statePoints = []
        sitcomPattern = r"SITCOM-(\d+)"

        for index, row in rows.iterrows():
            salIndices.add(row['salIndex'])
            blockIds.add(row['blockId'])

            lastCheckpoint = row['lastCheckpoint']
            sitcomMatches = re.findall(sitcomPattern, lastCheckpoint)
            tickets.update(sitcomMatches)

            time = efdTimestampToAstropy(row['private_efdStamp'])
            state = ScriptState(row['state'])
            reason = row['reason']
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
            begin=efdTimestampToAstropy(rows.iloc[0]['private_efdStamp']),
            end=efdTimestampToAstropy(rows.iloc[-1]['private_efdStamp']),
            salIndices=sorted(salIndices),
            tickets=[f'SITCOM-{ticket}' for ticket in sorted(tickets)],
            states=statePoints,
        )

        return blockInfo

    def getEventsForBlock(self, events, block, seqNum):
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
        begin = blockInfo.begin
        end = blockInfo.end

        # each event's end being past the begin time and their
        # starts being before the end time means we get all the
        # events in the window and also those that overlap the
        # start/end too
        return [e for e in events if e.end >= begin and e.begin <= end]

    def getBlocksForEvent(self, event):
        return event.blockInfos
