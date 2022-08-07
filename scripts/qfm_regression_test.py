from lsst.summit.utils.qfmRegressionTools import compareResults

refFilename = '/Users/merlin/lsst/summit_utils/tests/data/test_data_reference.txt'
compFilename = '/Users/merlin/lsst/summit_utils/tests/data/test_data_comparison.txt'
compareResults(refFilename, compFilename, 0.5)
