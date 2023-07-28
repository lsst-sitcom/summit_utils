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


@dataclass(slots=True, kw_only=True, frozen=True)
class BlockInfo:
    """The block info relating to a TMAEvent.

    Parameters
    ----------
    blockNumber : `int`
        The block number, as an integer.
    blockId : `str`
        The block ID, as a string.
    salIndices : `list` of `int`
        One or more SAL indices, relating to the block.
    tickets : `list` of `str`
        One or more SITCOM tickets, relating to the block.
    states : `list` of `ScriptStatePoint`
        The states of the script during the block. Each element is a
        ``ScriptStatePoint`` which contains:
            - the time, as an astropy.time.Time
            - the state, as a ``ScriptState`` enum
            - the reason for state change, as a string
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


@dataclass(slots=True, kw_only=True, frozen=True)
class ScriptStatePoint:
    time: Time
    state: ScriptState
    reason: str

    def __repr__(self):
        return (
            f"ScriptStatePoint(time={self.time!r}, state={self.state!r}, reason={self.reason!r})"
        )

    def _ipython_display_(self):
        print(self.__str__())

    def __str__(self):
        reasonStr = f" - {self.reason}" if self.reason else ""
        return (f"{self.state.name:>10} @ {self.time.isot}{reasonStr}")


class BlockParser:
    def __init__(self, dayObs, client=None):
        t0 = time.time()
        self.client = client
        self.dayObs = dayObs
        if client is None:
            self.client = makeEfdClient()

        t0 = time.time()
        self.getDataForDayObs()
        print(f"Getting data took {(time.time()-t0):.2f} seconds")
        t0 = time.time()
        # self.augmentData()
        self.augmentDataSlow()
        print(f"Parsing data took {(time.time()-t0):.5f} seconds")

    def getDataForDayObs(self):
        data = getEfdData(self.client, 'lsst.sal.Script.logevent_state', dayObs=self.dayObs)  # , prePadding=86400*365)
        self.data = data

    def augmentDataSlow(self):
        data = self.data
        blockPattern = r"BLOCK-(\d+)"
        blockIdPattern = r"BL\d+(?:_\w+)+"
        sitcomPattern = r"SITCOM-(\d+)"

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
        data = self.data
        blockPattern = r"BLOCK-(\d+)"
        blockIdPattern = r"(BL\d+(?:_\w+)+)"
        sitcomPattern = r"SITCOM-(\d+)"

        data['blockNum'] = data['lastCheckpoint'].str.extract(blockPattern, expand=False).astype(float).astype(pd.Int64Dtype())
        data['blockId'] = data['lastCheckpoint'].str.extract(blockIdPattern, expand=False)

        # TODO: add SITCOM tickets, making sure it does the equivalent of re.findall and add them delimited somehow
        # data['sitcomTickets'] = data['lastCheckpoint'].str.extract(sitcomPattern, expand=False)

        blockIdSplit = data['blockId'].str.split('_', expand=True)
        data['blockDayObs'] = blockIdSplit[2].astype(float).astype(pd.Int64Dtype())
        data['blockSeqNum'] = blockIdSplit[3].astype(float).astype(pd.Int64Dtype())

    def _listColumnValues(self, column, removeNone=True):
        values = set(self.data[column].dropna())
        if None in values and removeNone:
            values.remove(None)
        return sorted(values)

    def getBlockNums(self,):
        return self._listColumnValues('blockNum')

    def getSeqNums(self, block):
        return sorted(set(self.data[self.data['blockNum'] == block]['blockSeqNum']))

    def getRows(self, block, seqNum=None):
        rowsForBlock = self.data[self.data['blockNum'] == block]
        if seqNum is None:
            return rowsForBlock
        return rowsForBlock[rowsForBlock['blockSeqNum'] == seqNum]

    def getBlockEvolution(self, block, seqNum=None):
        if seqNum is None:
            seqNums = self.getSeqNums(block)
        else:
            seqNums = [seqNum]
        print(f'Evolution of BLOCK {block} for dayObs={self.dayObs} {seqNum=}:')
        for seqNum in seqNums:
            blockInfo = self.getBlockInfo(block, seqNum)
            print(blockInfo, '\n')

    def getBlockInfo(self, block, seqNum):
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
            salIndices=sorted([i for i in salIndices]),
            tickets=[f'SITCOM-{ticket}' for ticket in tickets],
            states=statePoints,
        )

        return blockInfo

    def getEventsForBlock(self, events, block, seqNum):
        blockInfo = self.getBlockInfo(block, seqNum)
        begin = blockInfo.begin
        end = blockInfo.end

        # each event's end being past the begin time and their
        # starts being before the end time means we get all the
        # events in the window and also those that overlap the
        # start/end too
        return [e for e in events if e.end >= begin and e.begin <= end]

    def getBlockForEvent(self, event):

        raise NotImplementedError
