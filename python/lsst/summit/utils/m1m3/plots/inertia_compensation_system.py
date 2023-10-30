import logging

import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time
from lsst.summit.utils.type_utils import M1M3ICSAnalysis


def plot_hp_data(ax: plt.Axes, data: pd.Series | list, label: str) -> list[plt.Line2D]:
    """
    Plot hardpoint data on the given axes.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        The axes on which the data is plotted.
    topic : str
        The topic of the data.
    data : Series or list
        The data points to be plotted.
    label : str
        The label for the plotted data.

    Returns
    -------
    list
        A list containing the Line2D objects representing the plotted data
        lines.
    """
    line = ax.plot(data, "-", label=label, lw=0.5)
    return line


def mark_slew_begin_end(ax: plt.Axes, slew_begin: Time, slew_end: Time) -> plt.Line2D:
    """
    Mark the beginning and the end of a slew with vertical lines on the given
    axes.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        The axes where the vertical lines are drawn.
    slew_begin : astropy.time.Time
        The slew beginning time.
    slew_end : astropy.time.Time
        The slew ending time.

    Returns
    -------
    matplotlib.lines.Line2D
        The Line2D object representing the line drawn at the slew end.
    """
    _ = ax.axvline(slew_begin.datetime, lw=0.5, ls="--", c="k", zorder=-1)
    line = ax.axvline(
        slew_end.datetime, lw=0.5, ls="--", c="k", zorder=-1, label="Slew Start/Stop"
    )
    return line


def mark_padded_slew_begin_end(ax: plt.Axes, begin: Time, end: Time) -> plt.Line2D:
    """
    Mark the padded beginning and the end of a slew with vertical lines.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        The axes where the vertical lines are drawn.
    begin : astropy.time.Time
        The padded slew beginning time.
    end : astropy.time.Time
        The padded slew ending time.

    Returns
    -------
    matplotlib.lines.Line2D
        The Line2D object representing the line drawn at the padded slew end.
    """
    _ = ax.axvline(begin.datetime, alpha=0.5, lw=0.5, ls="-", c="k", zorder=-1)
    line = ax.axvline(
        end.datetime,
        alpha=0.5,
        lw=0.5,
        ls="-",
        c="k",
        zorder=-1,
        label="Padded Slew Start/Stop",
    )
    return line


def customize_hp_plot(
    ax: plt.Axes, dataset: M1M3ICSAnalysis, lines: list[plt.Line2D]
) -> None:
    """
    Customize the appearance of the hardpoint plot.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        The axes of the plot to be customized.
    dataset : M1M3ICSAnalysis
        The dataset object containing the data to be plotted and metadata.
    lines : list
        The list of Line2D objects representing the plotted data lines.
    """
    t_fmt = "%Y%m%d %H:%M:%S"
    ax.set_title(
        f"HP Measured Data\n "
        f"DayObs {dataset.event.dayObs} "
        f"SeqNum {dataset.event.seqNum} "
        f"v{dataset.event.version}\n "
        f"{dataset.df.index[0].strftime(t_fmt)} - "
        f"{dataset.df.index[-1].strftime(t_fmt)}"
    )
    ax.set_xlabel("Time [UTC]")
    ax.set_ylabel("HP Measured Forces [N]")
    ax.grid(":", alpha=0.2)
    ax.legend(ncol=4, handles=lines, fontsize="x-small")


def plot_velocity_data(ax: plt.Axes, dataset: M1M3ICSAnalysis) -> None:
    """
    Plot the azimuth and elevation velocities on the given axes.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        The axes on which the velocity data is plotted.
    dataset : M1M3ICSAnalysis
        The dataset object containing the data to be plotted and metadata.
    """
    ax.plot(dataset.df["az_actual_velocity"], color="royalblue", label="Az Velocity")
    ax.plot(dataset.df["el_actual_velocity"], color="teal", label="El Velocity")
    ax.grid(":", alpha=0.2)
    ax.set_ylabel("Actual Velocity\n [deg/s]")
    ax.legend(ncol=2, fontsize="x-small")


def plot_torque_data(ax: plt.Axes, dataset: M1M3ICSAnalysis) -> None:
    """
    Plot the azimuth and elevation torques on the given axes.

    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        The axes on which the torque data is plotted.
    dataset : M1M3ICSAnalysis
        The dataset object containing the data to be plotted and metadata.
    """
    ax.plot(dataset.df["az_actual_torque"], color="firebrick", label="Az Torque")
    ax.plot(dataset.df["el_actual_torque"], color="salmon", label="El Torque")
    ax.grid(":", alpha=0.2)
    ax.set_ylabel("Actual Torque\n [kN.m]")
    ax.legend(ncol=2, fontsize="x-small")


def plot_stable_region(
    fig: plt.figure, begin: Time, end: Time, label: str = "", color: str = "b"
) -> plt.Polygon:
    """
    Highlight a stable region on the plot with a colored span.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure containing the axes on which the stable region is
        highlighted.
    begin : astropy.time.Time
        The beginning time of the stable region.
    end : astropy.time.Time
        The ending time of the stable region.
    label : str, optional
        The label for the highlighted region.
    color : str, optional
        The color of the highlighted region.

    Returns
    -------
    matplotlib.patches.Polygon
        The Polygon object representing the highlighted region.
    """
    for ax in fig.axes:
        span = ax.axvspan(
            begin.datetime, end.datetime, fc=color, alpha=0.1, zorder=-2, label=label
        )
    return span


def plot_hp_measured_data(
    dataset: M1M3ICSAnalysis,
    fig: plt.figure,
    log: None | logging.Logger = None,
) -> None:
    """
    Create and plot hardpoint measured data, velocity, and torque on subplots.

    Parameters
    ----------
    dataset : M1M3ICSAnalysis
        The dataset object containing the data to be plotted and metadata.
    fig : matplotlib.figure.Figure
        The figure to be plotted on.
    log : logging.Logger, optional
        The logger object to log progress.
    """
    log = log.getChild(__name__) if log is not None else logging.getLogger(__name__)

    # Start clean
    fig.clear()

    # Add subplots
    ax_hp = fig.add_subplot(311)
    ax_tor = fig.add_subplot(312, sharex=ax_hp)
    ax_vel = fig.add_subplot(313, sharex=ax_hp)

    # Adjusting the height ratios
    fig.subplots_adjust(hspace=0.4)
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
    ax_hp.set_subplotspec(gs[0])
    ax_tor.set_subplotspec(gs[1])
    ax_vel.set_subplotspec(gs[2])

    # Plotting
    lines = []
    for hp in range(dataset.number_of_hardpoints):
        topic = dataset.measured_forces_topics[hp]
        line = plot_hp_data(ax_hp, dataset.df[topic], f"HP{hp+1}")
        lines.extend(line)

    slew_begin = Time(dataset.event.begin, scale="utc")
    slew_end = Time(dataset.event.end, scale="utc")

    mark_slew_begin_end(ax_hp, slew_begin, slew_end)
    mark_slew_begin_end(ax_vel, slew_begin, slew_end)
    line = mark_slew_begin_end(ax_tor, slew_begin, slew_end)
    lines.append(line)

    mark_padded_slew_begin_end(
        ax_hp, slew_begin - dataset.outer_pad, slew_end + dataset.outer_pad
    )
    mark_padded_slew_begin_end(
        ax_vel, slew_begin - dataset.outer_pad, slew_end + dataset.outer_pad
    )
    line = mark_padded_slew_begin_end(
        ax_tor, slew_begin - dataset.outer_pad, slew_end + dataset.outer_pad
    )
    lines.append(line)

    stable_begin, stable_end = dataset.find_stable_region()
    stat_begin, stat_end = (
        stable_begin + dataset.inner_pad,
        stable_end - dataset.inner_pad,
    )

    plot_velocity_data(ax_vel, dataset)
    plot_torque_data(ax_tor, dataset)
    span_stable = plot_stable_region(fig, stable_begin, stable_end, "Stable", color="k")
    span_with_padding = plot_stable_region(
        fig, stat_begin, stat_end, "Stable w/ Padding", color="b"
    )
    lines.extend([span_stable, span_with_padding])

    customize_hp_plot(ax_hp, dataset, lines)

    fig.subplots_adjust(hspace=0)

    return fig
