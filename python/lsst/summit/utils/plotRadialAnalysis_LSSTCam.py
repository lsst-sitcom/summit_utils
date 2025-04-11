from collections.abc import Iterable

import matplotlib
import numpy as np
import pandas
import yaml
from scipy.optimize import curve_fit  # type: ignore

import lsst
from lsst.utils.plotting.figures import make_figure


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


def doRadialAnalysis(data: np.ndarray, fitModel: str):
    """Perform the radial analysis on a star cutout

    Parameters
    ----------
    data: `np.ndarray`
        2D array containing the star cutout.
    fitModel: `str`
        Model used for the fit ('Moffat' or 'Gauss').

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

    match fitModel:
        case "Moffat":
            # Initial guess for Moffat fitting
            initialGuess = [
                np.max(radialValues),
                4,
                2,
                data.shape[1] / 2,
                data.shape[0] / 2,
                np.median(radialValues),
            ]
            Params, _ = curve_fit(moffat2dFitFunction, xy, radialValues, p0=initialGuess)
            _, AlphaFit, BetaFit, X0Fit, Y0Fit, BaselineFit = Params
            FwhmFit = np.abs(2.0 * AlphaFit * np.sqrt(2 ** (1 / BetaFit) - 1.0))
        case "Gauss":
            # Initial guess for Gaussian fitting
            initialGuess = [
                np.max(radialValues),
                10,
                data.shape[1] / 2,
                data.shape[0] / 2,
                np.median(radialValues),
            ]
            Params, _ = curve_fit(gaussian2dFitFunction, xy, radialValues, p0=initialGuess)
            _, FwhmFit, X0Fit, Y0Fit, BaselineFit = Params
        case _:
            raise ValueError(f"The model {fitModel} is not among the available ones (Gauss, Moffat)")

    # Compute the curve of growth (cumulative energy)
    Radii = np.sqrt((xGrid - X0Fit) ** 2 + (yGrid - Y0Fit) ** 2).ravel()

    SortedIndices = np.argsort(Radii)
    SortedRadii = Radii[SortedIndices]
    SortedValues = radialValues[SortedIndices] - BaselineFit
    CumulativeEnergy = np.cumsum(SortedValues)

    # Determine 50% and 80% encircled energy diameters
    EE50Diameter = 2 * SortedRadii[np.searchsorted(CumulativeEnergy, 0.5 * CumulativeEnergy[-1])]
    EE80Diameter = 2 * SortedRadii[np.searchsorted(CumulativeEnergy, 0.8 * CumulativeEnergy[-1])]

    x = 0.2 * SortedRadii
    yScatter = SortedValues + BaselineFit  # added back the backgroud.

    match fitModel:
        case "Moffat":
            YFit = moffat2dFitFunction(xy, *Params)[SortedIndices]
        case "Gauss":
            YFit = gaussian2dFitFunction(xy, *Params)[SortedIndices]
        case _:
            raise ValueError(f"The model {fitModel} is not among the available ones (Gauss, Moffat)")

    return (
        x,
        yScatter,
        YFit,
        [FwhmFit, EE50Diameter, EE80Diameter],
    )


def makeLayerPlot(
    ax: matplotlib.axes.Axes,
    data: np.ndarray,
    fitModel: str,
    layers: list[str] | str = "both",
    levels: np.ndarray | Iterable[float] | None = None,
) -> None | tuple[float]:
    """Make per axes layer plot.

    Create a plot with three possible layers:
        - Background image with the star cutout
        - The contour level of the star
        - The radial profile with Gaussian and Moffat fit
    The value of FWHM and Encircled Energy Radii (EE)
    at 50% and 80% are reported if the 'radial' layer has been chosen.

    Parameters
    ----------
    ax: `matplotlib.axes.Axes`
        The axes to use
    data: `np.ndarray`
        2D array containing the star cutout
    fitModel: `str`
        Model used for the fit ('Moffat' or 'Gauss')
    levels: `np.ndarray` or `Iterable` of `float` or `None`, optional
        The levels value for the contour layer.
        If None, is set to `np.linspace(1.5*np.std(data), data.max(), 5)`

    Returns
    -------
    If radial is a selected layer than returns the Fwhm EE50 and EE80
    for plotting purposes. Returns None otherwise.
    """

    (
        x,
        yScatter,
        YFit,
        [FwhmFit, EE50Diameter, EE80Diameter],
    ) = doRadialAnalysis(data, fitModel)

    # get figure and axes position on figure 
    # to create multiple axes on that position
    fig = ax.get_figure()
    bbox = ax.get_position()

    # plot the background layer if present
    if "background" in layers:
        axBkg = fig.add_axes(bbox, frameon=False)
        axBkg.imshow(data, cmap="gray", origin="lower", zorder=1)
        axBkg.set(zorder=1, xticks=[], yticks=[])

    # plot the contour layer if present
    if "contour" in layers:
        axCtr = fig.add_axes(bbox, frameon=False)
        xGrid, yGrid = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
        levels = levels if levels is not None else np.linspace(1.5 * np.std(data), data.max(), 5)
        axCtr.contour(xGrid, yGrid, data, cmap="spring", levels=levels, alpha=0.7)
        axCtr.set(zorder=2, facecolor=(1, 1, 1, 0), xticks=[], yticks=[])

    # plot the radial analysis layer if present
    if "radial" in layers:
        axRad = fig.add_axes(bbox, frameon=False)
        axRad.scatter(
            x,
            yScatter,
            marker="o",
            s=5,
            facecolor="none",
            edgecolor="cyan",
            label="Data",
        )

        axRad.plot(
            x,
            YFit,
            ls="-",
            color="y",
            label=f"{fitModel} Fit",
        )

        # add hline at zero level
        axRad.axhline(0, ls="--", color="silver", lw=1)
        axRad.set(zorder=3, facecolor=(1, 1, 1, 0), xticks=[], yticks=[])

    return FwhmFit, EE50Diameter, EE80Diameter


def createFigWithInstrumentLayout(
    fig: matplotlib.figure.Figure,
    inst: str,
    add_cbar: bool = False,
) -> matplotlib.figure.Figure:
    """Create a figure with the requested instrument layout"""
    with open("instrumentLayout.yaml") as file:
        layout = yaml.safe_load(file)[inst]

    if add_cbar:
        layout = [el + ["cbar"] for el in layout]

    axsDict = fig.subplot_mosaic(
        layout,
        subplot_kw={"xticks": [], "yticks": [], "aspect": "equal"},
        per_subplot_kw={"cbar": {"aspect": "auto"}},
        width_ratios=[1] * (len(layout[0]) - 1) + [0.15],
        gridspec_kw={"hspace": 0.0, "wspace": 0.05},
    )
    return axsDict


def makePsfPanel(
    cutouts: dict[str, np.ndarray],
    instrument: str = "LSSTComCam",
    layers: list[str] | str = "both",
    fitModel: str = "Moffat",
    levels: np.ndarray | Iterable[float] | None = None,
    **kwargs,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Make a per-detector PSF radial analysis.

    Each subplot shows for a detector a PSF cutout, a radial analysis and the
    morphology contour, a custom selection of this layer is possible.

    Parameters
    ----------
    cutouts: `dict[str, np.ndarray]`
        A detector's name key dictionary containing 
        the 2D array of the star cutouts.
    instrument: `str`, optional
        Detector type. Default 'LSSTComCam'.
    layers: `str` or `list` of `str`, optional
        List of layers to be displayed ('background', 'radial', 'contour').
        It is possible to pass also the string 'both' as a shortcut for
        ['background', 'radial', 'contour']. Default 'both'.
    fitModel: `str`, optional
        Model used for the radial fit ('Moffat' or 'Gauss').
        Default 'Moffat'.
    levels: `np.ndarray` or `Iterable` of `float` or `None`, optional
        The levels value for the contour layer.
        If None, is set to `np.linspace(1.5*np.std(data), data.max(), 5)`.
    **kwargs:
        Additional keyword arguments passed to matplotlib Figure constructor.

    Returns
    -------
    fig: `matplotlib.figure.Figure`
        The figure.
    """

    fig = make_figure(**kwargs)
    axsDict = createFigWithInstrumentLayout(fig, instrument, add_cbar=True)

    if layers == "both":
        layers = ["background", "radial", "contours"]

    fwhmDict = {}
    ee50Dict = {}
    ee80Dict = {}
    for detName, cutout in cutouts.items():
        #     axsDict[detName].text(
        #         0.525,
        #         0.95,
        #         detName,
        #         color="silver",
        #         fontsize=15,
        #         transform=axsDict[detName].transAxes,
        #         ha="right",
        #         va="top",
        #         zorder=5,
        #     )

        fwhm, ee50, ee80 = makeLayerPlot(axsDict[detName], cutout, fitModel, layers, levels)
        fwhmDict[detName] = fwhm
        ee50Dict[detName] = ee50
        ee80Dict[detName] = ee80

    cmap = matplotlib.colormaps["RdYlGn_r"]
    for detName in cutouts.keys():
        bbox = axsDict[detName].get_position()
        axText = fig.add_axes(bbox, frameon=True)
        axText.set(zorder=4, facecolor=(1, 1, 1, 0), xticks=[], yticks=[])

        if detName == "cbar":
            continue

        val = (fwhmDict[detName] - min(fwhmDict.values())) / (max(fwhmDict.values()) - min(fwhmDict.values()))
        color = cmap(val)
        axText.patch.set(lw=7, ec=color)
        for name, spine in axText.spines.items():
            spine.set_color(color)

        # Text with FWHM and EE50|80
        text = (
            f'{detName} FWHM: {fwhmDict[detName] * 0.2:.3f}"\n'
            f'EE50|80: {ee50Dict[detName] * 0.2:.3f}"|'
            f'{ee80Dict[detName] * 0.2:.3f}"',
        )
        axText.text(
            0.95,
            0.95,
            *text,
            color=color,
            fontsize=10,
            fontweight="bold",
            transform=axText.transAxes,
            ha="right",
            va="top",
        )

    # set colorbar
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(
            cmap=cmap,
            norm=matplotlib.pyplot.Normalize(
                vmin=min(fwhmDict.values()) * 0.2, vmax=max(fwhmDict.values()) * 0.2
            ),
        ),
        cax=axsDict["cbar"],
    )
    cbar.ax.set_ylabel(ylabel='FWHM"', fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    return fig


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
    tab: pandas.DataFrame,
    target: np.ndarray | list[float] | tuple[float, float],
    instrument: str,
) -> np.ndarray:
    """Find the nearest star w.r.t to a target coordinates
    N.B. The seacrh is done in PIXEL coordinates.

    Parameters
    ----------
    tab: `pandas.DataFrame`
        pandas.DataFram with the in focus stars positions.
    target: `np.ndarray` or `list` of `float` or `tuple` of `float`
        The target coordinates.
    instrument: `str`
        Instrument name. 
        Now needed to manage column name incosisntency 
        between ComCam and LSSTCam.
    """

    if instrument == "LSSTComCam":
        xCol = "slot_Centroid_x"
        yCol = "slot_Centroid_y"
    else:
        xCol = "x"
        yCol = "y"

    tab["center_sep"] = np.sqrt((tab[xCol] - target[0]) ** 2 + (tab[yCol] - target[1]) ** 2)
    most_close = tab.sort_values(by=["center_sep"]).iloc[0].name
    nearest = tab.loc[most_close, [xCol, yCol]].values
    return nearest


# change list in dictionary with det_num key
def makePanel(
    imgDict: dict[str, lsst.afw.image._exposure.ExposureF],
    sourceTableDict: dict[str, pandas.DataFrame],
    instrument: str,
    **kwargs,
) -> matplotlib.figure.Figure:
    """Create the panel with the in focus stars.
    See the documentation of `makePsfPanel` for more information.

    Parameters
    ----------
    imageDict: `dict[str, lsst.afw.image._exposure.ExposureF]`
        A detector's name key dictionary containing 
        the images from whose extract the cutouts.
    sourceTableDict: `dict[str, pandas.DataFrame]`
        A detector's name key dictionary containing 
        the source dataframe for each images.
    instrument: `str`
        Instrument name.
    **kwargs:
        Parameters for the `makePsfPanel` method.

    Returns
    -------
    fig: `matplotlib.figure.Figure`
        The figure.
    """
    # can still be hardcoded? (ComCam and LSSTCam have the same detector size?)
    center = (2036.0, 2000.0)

    candidates = {
        detName: findNearestStarToTarget(tab, center, instrument) for detName, tab in sourceTableDict.items()
    }
    cutouts = {detName: generateCutout(imgDict[detName], candidates[detName]) for detName in imgDict.keys()}

    fig = makePsfPanel(cutouts, instrument, **kwargs)
    fig.suptitle(f"visit: {imgDict[list(imgDict.keys())[0]].getInfo().getVisitInfo().id}", fontsize=25)
    return fig
