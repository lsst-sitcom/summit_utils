from collections.abc import Iterable

import matplotlib
import numpy as np
import pandas
from matplotlib.gridspec import GridSpec
from scipy.optimize import curve_fit  # type: ignore

import lsst


def gaussian2dFitFunction(
    xy: tuple[np.ndarray, np.ndarray],
    peak: float,
    fwhm: float,
    x0: float,
    y0: float,
    baseline: float = 0.0,
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
    x0: `float`
        The x position of the 2d Guassian function.
    y0: `float`
        The y position of the 2d Guassian function.
    baseline: `float`, optional
        Offset to apply. Default 0.

    Returns
    -------
    pdf: `np.ndarray`
        Probability density function of the distribution.
    """
    x, y = xy
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return baseline + peak * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2))


def moffat2dFitFunction(
    xy: tuple[np.ndarray, np.ndarray],
    peak: float,
    alpha: float,
    beta: float,
    x0: float,
    y0: float,
    baseline: float,
) -> np.ndarray:
    """Moffat distribution with centroid

    Parameters
    ----------
    xy: `np.ndarray`
        Points coordinates.
    peak: `float`
        Values of the intesity peak.
    alpha: `float`
        The alpha parameter for the Moffat distribution.
    beta: `float`
        The beta parameter for the Moffat distribution.
    x0: `float`
        x coordinate of the distribution.
    y0: `float`
        y coordinate of the distribution.
    baseline: `float`, optional
        Offset to apply. Default 0.

    Returns
    -------
    pdf: `np.ndarray`
        Probability density function of the distribution.
    """
    x, y = xy
    return baseline + peak * (1 + (((x - x0) ** 2 + (y - y0) ** 2)) / alpha**2) ** (-beta)


def doRadialAnalysis(data: np.ndarray):
    """Perform the radial analysis on a star cutout

    Parameters
    ----------
    data: `np.ndarray`
        2D array containing the star cutout

    Returns
    -------
    x: `np.ndarray`
        1d array with the radial from the centroid.
    y: `np.ndarray`
        1d array with the intensity value.
    gYFit: `np.ndarray`
        Gaussian Fit.
    mYFit: `np.ndarray`
        Moffat Fit.
    gStat: `list` of `float`
        A list containing the gaussian fit values of
        FWHM and Encircled Energ Fraction Radii (EE) at 50% and 80%
    mStat: `list` of `float`
        A list containing the moffat fit values of
        FWHM and Encircled Energ Fraction Radii (EE) at 50% and 80%
    """
    # Create meshgrid for fitting with x and y positions
    xGrid, yGrid = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    xy = (xGrid.ravel(), yGrid.ravel())
    radialValues = data.ravel()

    # Initial guess for Gaussian fitting
    initialGuess = [np.max(radialValues), 10, data.shape[1] / 2, data.shape[0] / 2, np.median(radialValues)]
    gParams, _ = curve_fit(gaussian2dFitFunction, xy, radialValues, p0=initialGuess)
    _, gFwhmFit, gX0Fit, gY0Fit, gBaselineFit = gParams

    # Initial guess for Moffat fitting
    initialGuess = [
        np.max(radialValues),
        4,
        2,
        data.shape[1] / 2,
        data.shape[0] / 2,
        np.median(radialValues),
    ]
    mParams, _ = curve_fit(moffat2dFitFunction, xy, radialValues, p0=initialGuess)
    _, mAlphaFit, mBetaFit, mX0Fit, mY0Fit, mBaselineFit = mParams
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
    mEE50Diameter = 2 * mSortedRadii[np.searchsorted(mCumulativeEnergy, 0.5 * mCumulativeEnergy[-1])]
    mEE80Diameter = 2 * mSortedRadii[np.searchsorted(mCumulativeEnergy, 0.8 * mCumulativeEnergy[-1])]

    x = 0.2 * mSortedRadii
    yScatter = mSortedValues + mBaselineFit  # added back the backgroud.
    gYFit = gaussian2dFitFunction(xy, *gParams)[gSortedIndices]
    mYFit = moffat2dFitFunction(xy, *mParams)[mSortedIndices]

    return (
        x,
        yScatter,
        gYFit,
        mYFit,
        [gFwhmFit, gEE50Diameter, gEE80Diameter],
        [mFwhmFit, mEE50Diameter, mEE80Diameter],
    )


def makeLayerPlot(
    axDict: dict[str, matplotlib.axes.Axes],
    data: np.ndarray,
    levels: np.ndarray | Iterable[float] | None = None,
):
    """Make per axes layer plot.

    Create a plot with three possible layers:
        - Background image with the star cutout
        - The contour level of the star
        - The radial profile with Gaussian and Moffat fit
    The value of FWHM and Encircled Energy Radii (EE)
    at 50% and 80% are reported if the 'radial' layer has been chosen.

    Parameters
    ----------
    axDict: `dict` of `matplotlib.axes.Axes`
        A Dictionary containing the axes layer to use
    data: `np.ndarray`
        2D array containing the star cutout
    levels: `np.ndarray` or `Iterable` of `float` or `None`, optional
        The levels value for the contour layer.
        If None, is set to `np.linspace(1.5*np.std(data), data.max(), 5)`
    """

    # setting the right z-order
    for lay, ax in axDict.items():
        if lay == "background":
            ax.set(zorder=1)
        elif lay == "contour":
            ax.set(zorder=2, facecolor=(1, 1, 1, 0))
        elif lay == "radial":
            ax.set(zorder=3, facecolor=(1, 1, 1, 0))

    # plot the background layer if present
    if "background" in axDict:
        # extent = (x.min(), x.max(), yScatter.min(), yScatter.max())
        axDict["background"].imshow(
            data,
            cmap="gray",
            origin="lower",
            # extent=extent,
            # aspect="auto",  # XXX equal?
        )

    # plot the contour layer if present
    if "contour" in axDict:
        xGrid, yGrid = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
        levels = levels if levels is not None else np.linspace(1.5 * np.std(data), data.max(), 5)
        axDict["contour"].contour(xGrid, yGrid, data, cmap="spring", levels=levels, alpha=0.7)

    # plot the radial analysis layer if present
    if "radial" in axDict:
        (
            x,
            yScatter,
            gYFit,
            mYFit,
            [gFwhmFit, gEE50Diameter, gEE80Diameter],
            [mFwhmFit, mEE50Diameter, mEE80Diameter],
        ) = doRadialAnalysis(data)

        axDict["radial"].scatter(
            x,
            yScatter,
            marker="o",
            s=5,
            facecolor="none",
            edgecolor="cyan",
            label="Data",
        )
        axDict["radial"].plot(
            x,
            gYFit,
            ls="-",
            color="r",
            label="Gaussian Fit",
        )
        axDict["radial"].plot(
            x,
            mYFit,
            ls="-",
            color="y",
            label="Moffat Fit",
        )

        # Text with FWHM and EE50|80
        gText = (
            f"FWHM: {gFwhmFit*0.2:.3f}  " f"EE50: {gEE50Diameter*0.2:.3f}  " f"EE80: {gEE80Diameter*0.2:.3f}"
        )

        mText = (
            f"FWHM: {mFwhmFit*0.2:.3f}  " f"EE50: {mEE50Diameter*0.2:.3f}  " f"EE80: {mEE80Diameter*0.2:.3f}"
        )

        axDict["radial"].text(
            0.15,
            0.9,
            gText,
            color="r",
            fontsize=9,
            fontweight="heavy",
            transform=axDict["radial"].transAxes,
        )

        axDict["radial"].text(
            0.15,
            0.85,
            mText,
            color="y",
            fontsize=9,
            fontweight="heavy",
            transform=axDict["radial"].transAxes,
        )


def makePsfPanel(
    cutOuts: dict[int, np.ndarray],
    detNameDict: dict[int, str],
    instrument: str = "LSSTComCam",
    layers: list[str] | str = "both",
    fig: matplotlib.figure.Figure | None = None,
    figSize: tuple[int, int] = (16, 12),
    **kwargs,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Make a per-detector PSF radial analysis.

    Each subplot shows for a detector a PSF cutout, a radial analysis and the
    morphology contour.

    Parameters
    ----------
    cutOuts: `dict[int, np.ndarray]`
        A dictionary containing the 2D array of the star cutOuts.
    detNameList: `dict[int, str]`
        A dictionary containing the detector names.
    instrument: `str`, optional
        Detector type. Default 'LSSTComCam'.
    layers: `str` or `list` of `str`, optional
        List of layers to be displayed ('background', 'radial', 'contour').
        It is possible to pass also the string 'both' as a shortcut for
        ['background', 'radial', 'contour']. Default 'both'.
    fig: `matplotlib.figure.Figure` or ``None``, optional
        If provided, use this figure.  Default ``None``.
    figSize: `tuple` of `float`, optional
        Figure size in inches.  Default (10, 10), but ignored if a `fig` is
        supplied.
    **kwargs:
        Additional keyword arguments passed to matplotlib Figure constructor.

    Returns
    -------
    fig: `matplotlib.figure.Figure`
        The figure.
    axLegend: `matplotlib.axes.Axes`
        The Axe to use for additional information aside the figure.
    """

    # generating figure if None
    # maybe is better to remove the figsize form user and
    # find a wise way to understand the w/h ratio automatically.
    if fig is None:
        fig = matplotlib.figure.Figure(figSize=figSize, **kwargs)

    # creating the intrument layout
    match instrument:
        case "LSSTComCam":
            maxCol = 3
            if len(cutOuts) < maxCol:
                nRows = 1
                nCols = len(cutOuts)
            else:
                nRows = len(cutOuts) // maxCol + 1 if len(cutOuts) % maxCol != 0 else len(cutOuts) // maxCol
                nCols = maxCol
        case _:
            print("So far only the LSSTComCam layout has been implemented. That layout will be used")
            maxCol = 3
            if len(cutOuts) < maxCol:
                nRows = 1
                nCols = len(cutOuts)
            else:
                nRows = len(cutOuts) // maxCol + 1 if len(cutOuts) % maxCol != 0 else len(cutOuts) // maxCol
                nCols = maxCol

    gs = GridSpec(
        nrows=nRows,
        ncols=nCols + 1,  # adding a colum for legend and statistc
        figure=fig,
        width_ratios=[1.0] * nCols + [0.05],
        height_ratios=[1.0] * nRows,
    )

    axLegend = fig.add_subplot(gs[:, -1])  # axe for title, legend and statistic
    axLegend.axis("off")

    # checking the layers parameter
    axs: dict[str, list] = dict()
    if layers == "both":
        for layer in ["background", "radial", "contour"]:
            axs[layer] = []

    elif (
        isinstance(layers, list)
        and "both" not in layers
        and all([el in ["background", "radial", "contour"] for el in layers])
    ):
        for layer in layers:
            axs[layer] = []
    else:
        raise ValueError("List of layers not valid.")

    # generate the axes layers
    # !!Very hardcode for ComCam!!
    # !!a wise method to swetup the axes layout for each insturment is needed!!
    for i in range(len(cutOuts)):
        for axType, axList in axs.items():
            if axType == "radial":  # keep ticks for radial plots
                axList.append(fig.add_subplot(gs[(gs.nrows - 1) - i // nCols, i % nCols]))
            else:
                axList.append(
                    fig.add_subplot(
                        gs[(gs.nrows - 1) - i // nCols, i % nCols],
                        xticks=[],
                        yticks=[],
                        aspect="equal",
                    )
                )
    for ax, title in zip(axs[list(axs.keys())[0]], detNameDict.values()):
        ax.set(title=title)

    for i, cutOut in enumerate(cutOuts.values()):
        makeLayerPlot({key: axs[key][i] for key in axs.keys()}, cutOut)

    # put fit legend if radial layer is request
    if "radial" in list(axs.keys()):

        # set legend
        gLabel = "Gaussian Fit"
        mLabel = "Moffat Fit"
        fitLabel = "Full Width at Half Maximum (FWHM)\nEncircled Energy Radius (EE)\nUnit: arcsecond"
        axLegend.axis("off")
        axLegend.text(
            0.1,
            0.95,
            gLabel,
            fontsize=20,
            fontweight="bold",
            ha="left",
            va="center",
            c="r",
        )
        axLegend.text(
            0.1,
            0.92,
            mLabel,
            fontsize=20,
            fontweight="bold",
            ha="left",
            va="center",
            c="y",
        )
        axLegend.text(
            0.1,
            0.87,
            fitLabel,
            fontsize=15,
            fontweight="medium",
            ha="left",
            va="center",
            c="black",
        )

        # sharing the axis limits
        axToShareX = np.argmax([ax.get_xlim()[1] for ax in axs["radial"]])
        # axToShareY = np.argmax([ax.get_ylim()[1] for ax in axs["radial"]])

        for ax in axs["radial"]:
            ax.sharex(axs["radial"][axToShareX])
            # ax.sharey(axs["radial"][axToShareY])

        # put shared x and y labels
        for i, ax in enumerate(axs["radial"]):
            if i <= maxCol:  # hardcoded for ComCam!
                ax.set(xlabel="r, arcsec")
            if i % maxCol == 0:
                ax.set(ylabel="counts")

    return fig, axLegend


def generateCutout(
    img: lsst.afw.image._exposure.ExposureF,
    center: np.ndarray | list[float] | tuple[float, float],
    pad: int = 10,
) -> np.ndarray:
    """Generate the cutout around a center position

    Parameters
    ----------
    img: `lsst.afw.image._exposure.ExposureF`
        The image from extract the cutouts
    center: `np.ndarray` or `list` of `float` or `tuple` of `float`
        The coordinates of the cutout center
    pad: `int`, optional
        Padding around the center, default 10.

    Returns
    -------
    cutout: `np.ndarray`
        The square cutout around the center position.
    """
    xlim = (center[0] - pad, center[0] + pad)
    ylim = (center[1] - pad, center[1] + pad)
    cutout = img.image.array[int(ylim[0]) : int(ylim[1]), int(xlim[0]) : int(xlim[1])]
    return cutout


def findNearestStarToTarget(
    tab: pandas.DataFrame, target: np.ndarray | list[float] | tuple[float, float]
) -> np.ndarray:
    """Find the nearest star w.r.t to a target coordinates
    N.B. The seacrh is done in PIXEL coordinates.

    Parameters
    ----------
    tab: `pandas.DataFrame`
        pandas.DataFram with the in focus stars positions.
    target: `np.ndarray` or `list` of `float` or `tuple` of `float`
        The target coordinates.
    """
    tab["center_sep"] = np.sqrt(
        (tab["slot_Centroid_x"] - target[0]) ** 2 + (tab["slot_Centroid_y"] - target[1]) ** 2
    )
    most_close = tab.sort_values(by=["center_sep"]).iloc[0].name
    nearest = tab.loc[most_close, ["slot_Centroid_x", "slot_Centroid_y"]].values
    return nearest


# change list in dictionary with det_num key
def makePanel(
    imgDict: dict[int, lsst.afw.image._exposure.ExposureF],
    sourceTableDict: dict[int, pandas.DataFrame],
    **kwargs,
) -> matplotlib.figure.Figure:
    """Create the panel with the in focus stars.
    See the documentation of `makePsfPanel` for more information.

    Parameters
    ----------
    imageDict: `dict[int, lsst.afw.image._exposure.ExposureF]`
        A dictionary containing the images from whose extract the cutouts.
    sourceTableDict: `dict[int, pandas.DataFrame]`
        A dictionary containing the source dataframe for each images.
    **kwargs:
        Parameters for the `makePsfPanel` method.

    Returns
    -------
    fig: `matplotlib.figure.Figure`
        The figure.
    """
    # can still be hardcoded? (ComCam and LSSTCam have the same detector size?)
    center = (2036.0, 2000.0)

    # check the key sorting in the dictionary
    imgDict = {detNum: imgDict[detNum] for detNum in sorted(list(imgDict.keys()))}
    sourceTableDict = {detNum: sourceTableDict[detNum] for detNum in sorted(list(sourceTableDict.keys()))}
    candidates = {detNum: findNearestStarToTarget(tab, center) for detNum, tab in sourceTableDict.items()}
    cutouts = {
        detNum: generateCutout(img, candidate)
        for (detNum, img), candidate in zip(imgDict.items(), candidates.values())
    }
    detName = {detNum: img.getInfo().getDetector().getName() for detNum, img in imgDict.items()}

    fig, axLegend = makePsfPanel(cutouts, detName, **kwargs)
    fig.suptitle("INFOCUS INSPECTION", fontsize=25)
    visitText = f"visit: {imgDict[0].getInfo().getVisitInfo().id}"
    axLegend.text(
        0.1,
        0.98,
        visitText,
        fontsize=20,
        fontweight="bold",
        ha="left",
        va="center",
        c="black",
    )

    return fig
