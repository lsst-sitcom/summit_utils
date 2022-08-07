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

__all__ = ['DataIdClass',
           'CentroidClass',
           'loadResults',
           'compareResults',
           ]

from dataclasses import dataclass


@dataclass
class DataIdClass:
    instrument: str
    day_obs: int
    seq_num: int

    def __hash__(self) -> int:
        return hash(self.instrument + str(self.day_obs) + str(self.seq_num))

    def __repr__(self) -> str:
        # TODO: find out how to print a { inside an f-string!
        return(f'{{instrument={self.instrument}, day_obs={self.day_obs}, seq_num={self.seq_num}}}')


@dataclass
class CentroidClass:
    x: float
    y: float

    def distanceTo(self, other):
        return ((self.x - other.x)**2 + (self.y - other.y)**2)**.5


def loadResults(filename, instrument=None):
    with open(filename, 'r') as f:
        resultData = f.readlines()

    results = {}
    for line in resultData:
        dataIdStr, centroidStr = line.split('~')
        dataIdStr.strip()
        centroidStr.strip()

        # TODO: replace with json safeload - this is fine but does json
        # provide a "safe eval" or is that only if it does the round-tripping?
        dataId = eval(dataIdStr)
        assert isinstance(dataId, dict)

        centroid = eval(centroidStr)
        assert isinstance(centroid, tuple)
        assert isinstance(centroid[0], float)
        assert isinstance(centroid[1], float)

        if instrument and dataId['instrument'] != instrument:
            continue
        dataId = DataIdClass(instrument=dataId['instrument'],
                             day_obs=dataId['day_obs'],
                             seq_num=dataId['seq_num'],
                             )
        c = CentroidClass(x=centroid[0],
                          y=centroid[1],
                          )
        results[dataId] = c

    print(f'Loaded {len(results)} dataIds for {filename}')
    return results


def compareResults(referenceFilename, comparisonFilename, tolerance, *,
                   printAll=False,
                   ignoreMissing=False):

    ref = loadResults(referenceFilename)
    comp = loadResults(comparisonFilename)
    refIds = list(ref.keys())
    compIds = list(comp.keys())

    notInComp = set(refIds) - set(compIds)
    notInRef = set(compIds) - set(refIds)
    commonIds = set(compIds).intersection(set(refIds))

    if not ignoreMissing and notInComp:
        print('\nDataIds in reference data not in comparison data:')
        for d in notInComp:
            print(d)

    # always display new ids as this is somewhat unexpected
    if notInRef:
        print('\nNew dataIds in comparison set without data in reference:')
        for d in notInRef:
            print(d)

    print(f'\nFound {len(commonIds)} dataIds in common between reference and comparison')
    moved = []  # really just so we can write tests for now
    for dataId in commonIds:
        refCentroid = ref[dataId]
        compCentroid = comp[dataId]
        dist = compCentroid.distanceTo(refCentroid)
        if printAll or dist > tolerance:
            print(f"{dataId} distance = {dist:.2f}")
        if dist > tolerance:
            moved.append((dataId, dist))
    return moved


if __name__ == '__main__':
    # TODO: Remove __main__ block and move to tests
    filename = '/Users/merlin/lsst/summit_utils/tests/data/test_data_reference.txt'
    compFile = '/Users/merlin/lsst/summit_utils/tests/data/test_data_comparison.txt'
    # result  = loadResults(filename)
    # print(result)
    tolerance = .10
    compareResults(filename, compFile, tolerance, printAll=True)
