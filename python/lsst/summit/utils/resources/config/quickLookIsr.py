# mypy: disable-error-code="name-defined"

config.doWrite = False  # this task writes separately, no need for this
config.doSaturation = True  # saturation very important for roundness measurement in qfm
config.doSaturationInterpolation = True
config.overscan.fitType = "MEDIAN_PER_ROW"
config.overscan.doParallelOverscan = True
config.brighterFatterMaxIter = 2  # Uncomment this to remove test warning
