# this task writes separately, no need for this
config.doWrite = False  # type: ignore
# saturation very important for roundness measurement in qfm
config.doSaturation = True  # type: ignore
config.doSaturationInterpolation = True  # type: ignore
config.overscan.fitType = "MEDIAN_PER_ROW"  # type: ignore
config.overscan.doParallelOverscan = True  # type: ignore
# Uncomment this to remove test warning
config.brighterFatterMaxIter = 2  # type: ignore
