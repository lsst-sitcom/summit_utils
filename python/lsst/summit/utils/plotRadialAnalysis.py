from collections.abc import Iterator

import matplotlib
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit  # type: ignore


def gaussianWithCentroid(
    xy: tuple[np.ndarray, np.ndarray],
    peak: float,
    fwhm: float,
    x0: float,
    y0: float,
    baseLine: float = 0.0,
) -> np.ndarray:
    """Gaussian distribution with centroid

    Parameters
    ----------
    xy: `np.ndarray`
        Points coordinates.
    peak: `float`
        Values of the intesity peak.
    fwhm: `float`
        Full Width at Half Maximum fo the distribution.
    x0, y0: `float`
        Center coordinates of the distribution.
    baseLine: `float`, optional
        Offset to apply. Default 0.

    Returns
    -------
    pdf: `np.ndarray`
        Probability density function of the distribution.
    """
    x, y = xy
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return baseLine + peak * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))


def moffatWithCentroid(
    xy: tuple[np.ndarray, np.ndarray],
    peak: float,
    alpha: float,
    beta: float,
    x0: float,
    y0: float,
    baseLine: float,
) -> np.ndarray:
    """Moffat distribution with centroid

    Parameters
    ----------
    xy: `np.ndarray`
        Points coordinates.
    peak: `float`
        Values of the intesity peak.
    alpha, beta: `float`
        Scale parameters.
    x0, y0: `float`
        Center coordinates of the distribution.
    baseline: `float`, optional
        Offset to apply. Default 0.

    Returns
    -------
    pdf: `np.ndarray`
        Probability density function of the distribution.
    """
    x, y = xy
    return baseLine + peak * (1 + (((x - x0) ** 2 + (y - y0) ** 2)) / alpha**2) ** (-beta)


def makeRadialPlot(
    axPlt: matplotlib.axes.Axes,
    axBkg: matplotlib.axes.Axes,
    axCtr: matplotlib.axes.Axes,
    data: np.ndarray,
    levels: np.ndarray | Iterator[float] | None = None,
):
    """Make a plot of radial analysis

    Create a plot with three layers:
        - Background image with the star cutout
        - The contour level of the star
        - The radial profile with Gaussia and Moffat fit
    The value of FWHM and Encircled Energy Radius (EE) at the
    50% and 80% are reported for both the fit methods.

    Parameters
    ----------
    axPlt: `matplotlib.axes.Axes`
        Axes used for the radial plot.
    axBkg: `matplotlib.axes.Axes`
        Axes used for the background image.
    axCtr: `matplotlib.axes.Axes`
        Axes used for the contours on the image.
    data: `np.ndarray`
        2D array containing the star cutout
    levels: `np.ndarray` or `Iterator` of `float` or `None`, optional
        The levels value for the contours.
        If None, is set to `np.linspace(1.5*np.std(data), data.max(), 5)`
    """
    # Create meshgrid for fitting with x and y positions
    xGrid, yGrid = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    xy = (xGrid.ravel(), yGrid.ravel())
    radialValues = data.ravel()

    # Initial guess for Gaussian fitting
    initialGuess = [np.max(radialValues), 10, data.shape[1] / 2, data.shape[0] / 2, np.median(radialValues)]
    gParams, _ = curve_fit(gaussianWithCentroid, xy, radialValues, p0=initialGuess)
    gPeakFit, gFwhmFit, gX0Fit, gY0Fit, gBaselineFit = gParams

    # Initial guess for Moffat fitting
    initialGuess = [
        np.max(radialValues),
        4,
        2,
        data.shape[1] / 2,
        data.shape[0] / 2,
        np.median(radialValues),
    ]
    mParams, _ = curve_fit(moffatWithCentroid, xy, radialValues, p0=initialGuess)
    mPeak, mAlphaFit, mBetaFit, mX0Fit, mY0Fit, mBaselineFit = mParams
    mFwhmFit = 0.2 * np.abs(2.0 * mAlphaFit * np.sqrt(2 ** (1 / mBetaFit) - 1.0))

    # Compute the curve of growth (cumulative energy)
    gRadii = np.sqrt((xGrid - gX0Fit) ** 2 + (yGrid - gY0Fit) ** 2).ravel()
    mRadii = np.sqrt((xGrid - mX0Fit) ** 2 + (yGrid - mY0Fit) ** 2).ravel()

    gSortedIndices = np.argsort(gRadii)
    mSortedIndices = np.argsort(mRadii)
    gSortedRadii = gRadii[gSortedIndices]
    mSortedRadii = mRadii[mSortedIndices]
    gSortedValues = radialValues[gSortedIndices] - gBaselineFit
    mSortedValues = radialValues[mSortedIndices] - mBaselineFit
    gCumulativeEnergy = np.cumsum(gSortedValues)
    mCumulativeEnergy = np.cumsum(mSortedValues)

    # Determine 50% and 80% encircled energy diameters from Gaussian
    gEE50Diameter = 2 * gSortedRadii[np.searchsorted(gCumulativeEnergy, 0.5 * gCumulativeEnergy[-1])]
    gEE80Diameter = 2 * gSortedRadii[np.searchsorted(gCumulativeEnergy, 0.8 * gCumulativeEnergy[-1])]

    # Determine 50% and 80% encircled energy diameters from Moffat
    mEE50Diamater = 2 * mSortedRadii[np.searchsorted(mCumulativeEnergy, 0.5 * mCumulativeEnergy[-1])]
    mEE80Diamater = 2 * mSortedRadii[np.searchsorted(mCumulativeEnergy, 0.8 * mCumulativeEnergy[-1])]

    x = 0.2 * mSortedRadii
    yScatter = mSortedValues + mBaselineFit  # added back the backgroud.
    gYFit = gaussianWithCentroid(xy, *gParams)[gSortedIndices]
    mYFit = moffatWithCentroid(xy, *mParams)[mSortedIndices]
    extent = (x.min(), x.max(), yScatter.min(), yScatter.max())

    # setting the layer zorder
    axPlt.set(zorder=3, facecolor=(1, 1, 1, 0))
    axCtr.set(zorder=2, facecolor=(1, 1, 1, 0))
    axBkg.set(zorder=1)

    # put the backgroud cutout image
    axBkg.imshow(
        data,
        cmap="gray",
        origin="lower",
        extent=extent,
        aspect="auto",
    )

    # Contour plot function
    levels = levels if levels is not None else np.linspace(1.5 * np.std(data), data.max(), 5)
    axCtr.contour(xGrid, yGrid, data, cmap="spring", levels=levels, alpha=0.7)

    # Radial profile plot
    axPlt.scatter(
        x,
        yScatter,
        marker="o",
        s=5,
        facecolor="none",
        edgecolor="cyan",
        label="Data",
    )
    axPlt.plot(
        x,
        gYFit,
        ls="-",
        color="r",
        label="Gaussian Fit",
    )
    axPlt.plot(
        x,
        mYFit,
        ls="-",
        color="y",
        label="Moffat Fit",
    )

    # Text with FWHM and EE50|80
    gText = (f"GFWHM: {gFwhmFit*0.2:.3f}\nGEE50|80: {gEE50Diameter*0.2:.3f}|{gEE80Diameter*0.2:.3f}",)
    mText = (f"MFWHM: {mFwhmFit:.3f}\nMEE50|80: {mEE50Diamater*0.2:.3f}|{mEE80Diamater*0.2:.3f}",)
    axPlt.text(0.5, 0.9, *gText, color="r", fontsize="small", transform=axPlt.transAxes)
    axPlt.text(0.5, 0.8, *mText, color="y", fontsize="small", transform=axPlt.transAxes)


def makePsfRadialPanel(
    cutOuts: list[np.ndarray],
    detType: str = "LSSTComCam",
    fig: matplotlib.figure.Figure | None = None,
    figSize: tuple[int, int] = (10, 10),
    maxCol: int = 3,
    **kwargs,
) -> matplotlib.figure.Figure:
    """Make a per-detector psf radial analysis.

    Each subplots shows for a detector a psf cutout, a radial analysis and the
    morphology contour.

    Parameters
    ----------
    cutOuts : `Iterator` of `np.ndarray`
        An iterator containing the 2D array of the star cutOuts.
    detType: `str`
        Detector type. Default 'LSSTComCam'.
    fig: `matplotlib.figure.Figure` or `None`, optional
        If provided, use this figure.  Default None.
    figSize: `tuple` of `float`, optional
        Figure size in inches.  Default (11, 12).
    maxCol: `int`, optional
        Maximum number of columns to use while creating the subplots grid.
        Default 3.
    **kwargs:
        Additional keyword arguments passed to matplotlib Figure constructor.

    Returns
    -------
    fig: `matplotlib.figure.Figure`
        The figure.
    """

    # generating figure if None
    if fig is None:
        fig = matplotlib.figure.Figure(figSize=figSize, **kwargs)

    # creating the gridspec grid
    if len(cutOuts) < maxCol:
        nRows = 1
        nCols = len(cutOuts)
    else:
        nRows = len(cutOuts) // maxCol + 1 if len(cutOuts) % maxCol != 0 else len(cutOuts) // maxCol
        nCols = maxCol

    gs = GridSpec(
        nrows=nRows,
        ncols=nCols,
        figure=fig,
        width_ratios=[1.0] * nCols,
        height_ratios=[1.0] * nRows,
    )

    # generate the three axes layers
    axsPlt = []
    axsBkg = []
    axsCtr = []
    for i in range(len(cutOuts)):
        axsPlt.append(fig.add_subplot(gs[i // nCols, i % nCols], xticks=[], yticks=[]))
        axsBkg.append(fig.add_subplot(gs[i // nCols, i % nCols], xticks=[], yticks=[]))
        axsCtr.append(fig.add_subplot(gs[i // nCols, i % nCols], xticks=[], yticks=[]))

    for axPlt, axBkg, axCtr, cutOut in zip(axsPlt, axsBkg, axsCtr, cutOuts):
        makeRadialPlot(axPlt, axBkg, axCtr, cutOut)

    return fig
