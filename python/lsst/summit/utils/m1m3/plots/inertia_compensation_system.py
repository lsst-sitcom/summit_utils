import logging
import pytz

import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time
from matplotlib.lines import Line2D

from lsst.summit.utils.type_utils import M1M3ICSAnalysis

__all__ = [
    "plot_hp_data",
    "mark_slew_begin_end",
    "mark_padded_slew_begin_end",
    "customize_fig",
    "customize_hp_plot",
    "add_hp_limits",
    "plot_velocity_data",
    "plot_torque_data",
    "plot_stable_region",
    "plot_hp_measured_data",
    "HP_BREAKAWAY_LIMIT",
    "HP_FATIGUE_LIMIT",
    "HP_OPERATIONAL_LIMIT",
    "FIGURE_WIDTH",
    "FIGURE_HEIGHT",
]

# Approximate value for breakaway
HP_BREAKAWAY_LIMIT: float = 3000  # [N]

# limit that can still damage the mirror with fatigue
HP_FATIGUE_LIMIT: float = 900  # [N]

# desired operational limit
HP_OPERATIONAL_LIMIT: float = 450  # [N]

FIGURE_WIDTH = 10
FIGURE_HEIGHT = 7


def plot_hp_data(ax: plt.Axes, data: pd.Series | list, label: str) -> plt.Line2D:
    """
    Plot hardpoint data on the given axes.

    Parameters
    ----------
    ax : `plt.Axes`
        The axes on which the data is plotted.
    data : `Series` or `list`
        The data points to be plotted.
    t0 : `astropy.time.Time`
        Start of a slew. Used as an offset to convert the x-axis to seconds.
    label : `str`
        The label for the plotted data.

    Returns
    -------
    lines : `plt.Line2D`
        The plotted data as a Line2D object.
    """
    line = ax.plot(data, linestyle="-", label=label, linewidth=0.5)
    #  Make this function consistent with others by returning single Line2D
    return line[0]


def mark_slew_begin_end(ax: plt.Axes, slew_begin: Time, slew_end: Time) -> plt.Line2D:
    """
    Mark the beginning and the end of a slew with vertical lines on the given
    axes.

    Parameters
    ----------
    ax : `matplotlib.axes._axes.Axes`
        The axes where the vertical lines are drawn.
    slew_begin : `astropy.time.Time`
        The slew beginning time.
    slew_end : `astropy.time.Time`
        The slew ending time.

    Returns
    -------
    line : `matplotlib.lines.Line2D`
        The Line2D object representing the line drawn at the slew end.
    """
    _ = ax.axvline(slew_begin, lw=0.5, ls="--", c="k", zorder=-1)
    line = ax.axvline(slew_end, lw=0.5, ls="--", c="k", zorder=-1, label="Slew Start/Stop")
    return line


def mark_padded_slew_begin_end(ax: plt.Axes, begin: Time, end: Time) -> plt.Line2D:
    """
    Mark the padded beginning and the end of a slew with vertical lines.

    Parameters
    ----------
    ax : `matplotlib.axes._axes.Axes`
        The axes where the vertical lines are drawn.
    begin : `astropy.time.Time`
        The padded slew beginning time.
    end : `astropy.time.Time`
        The padded slew ending time.

    Returns
    -------
    line : `matplotlib.lines.Line2D`
        The Line2D object representing the line drawn at the padded slew end.
    """
    print(">>>>>>", begin, end)
    _ = ax.axvline(begin, alpha=0.5, lw=0.5, ls="-", c="k", zorder=-1)
    line = ax.axvline(
        end,
        alpha=0.5,
        lw=0.5,
        ls="-",
        c="k",
        zorder=-1,
        label="Padded Slew Start/Stop",
    )
    return line


def customize_fig(fig: plt.Figure, dataset: M1M3ICSAnalysis):
    """
    Add a title to a figure and adjust its subplots spacing

    Paramters
    ---------
    fig : `matplotlib.pyplot.Figure`
        Figure to be custoized.
    dataset : `M1M3ICSAnalysis`
        The dataset object containing the data to be plotted and metadata.
    """
    t_fmt = "%Y-%m-%dT%H:%M:%S"
    fig.suptitle(
        f"HP Measured Data\n "
        f"DayObs {dataset.event.dayObs}, "
        f"SeqNum {dataset.event.seqNum}, "
        f"v{dataset.event.version}, "
        f"{dataset.df.index[0].strftime(t_fmt)} - "
        f"{dataset.df.index[-1].strftime(t_fmt)}\n"
        f"Az from {dataset.stats['az_start']:.2f} to {dataset.stats['az_end']:.2f} deg"
        f" El from {dataset.stats['el_start']:.2f} to {dataset.stats['el_end']:.2f} deg, ",
        y=1.00,
    )
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05)


def customize_hp_plot(ax: plt.Axes, lines: list[plt.Line2D]) -> None:
    """
    Customize the appearance of the hardpoint plot.

    Parameters
    ----------
    ax : `matplotlib.axes._axes.Axes`
        The axes of the plot to be customized.
    lines : `list`
        The list of Line2D objects representing the plotted data lines.
    """
    limit_lines = add_hp_limits(ax)
    lines.extend(limit_lines)

    ax.set_ylabel("HP Measured\n Forces [N]")
    ax.set_ylim(-3100, 3100)
    ax.grid(linestyle=":", alpha=0.5)


def add_hp_limits(ax: plt.Axes):
    """
    Add horizontal lines to represent the breakaway limits, the fatigue limits,
    and the operational limits.

    This was first discussed on Slack. From Doug Neil we got:

    > A fracture statistics estimate of the fatigue limit of a borosilicate
    > glass. The fatigue limit of borosilicate glass is 0.21 MPa (~30 psi).
    > This implies that repeated loads of 30% of our breakaway limit would
    > eventually produce failure. To ensure that the system is safe for the
    > life of the project we should provide a factor of safety of at least two.
    > I recommend a 30% repeated load limit, and a project goal to keep the
    > stress below 15% of the breakaway during normal operations.

    Parameters
    ----------
    ax : `plt.Axes`
        The axes on which the velocity data is plotted.
    """
    hp_limits = {
        "HP Breakaway Limit": {"pos_limit": HP_BREAKAWAY_LIMIT, "neg_limit": -HP_BREAKAWAY_LIMIT, "ls": "-"},
        "Repeated Load Limit (30% breakaway)": {
            "pos_limit": HP_FATIGUE_LIMIT,
            "neg_limit": -HP_FATIGUE_LIMIT,
            "ls": "--",
        },
        "Normal Ops Limit (15% breakaway)": {
            "pos_limit": HP_OPERATIONAL_LIMIT,
            "neg_limit": -HP_OPERATIONAL_LIMIT,
            "ls": ":",
        },
    }

    kwargs = dict(alpha=0.5, lw=1.0, c="r", zorder=-1)
    line_list = []

    for key, sub_dict in hp_limits.items():
        ax.axhline(sub_dict["pos_limit"], ls=sub_dict["ls"], **kwargs)
        line = ax.axhline(sub_dict["neg_limit"], ls=sub_dict["ls"], label=key, **kwargs)
        line_list.append(line)

    return line_list


def plot_velocity_data(ax: plt.Axes, dataset: M1M3ICSAnalysis) -> None:
    """
    Plot the azimuth and elevation velocities on the given axes.

    Parameters
    ----------
    ax : `matplotlib.axes._axes.Axes`
        The axes on which the velocity data is plotted.
    dataset : `M1M3ICSAnalysis`
        The dataset object containing the data to be plotted and metadata.
    """
    ax.plot(dataset.df["az_actual_velocity"], color="royalblue", label="Az Velocity")
    ax.plot(dataset.df["el_actual_velocity"], color="teal", label="El Velocity")
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_ylabel("Actual\n Velocity\n [deg/s]")

    l1 = Line2D([0], [0], color="royalblue", lw=1, label="Az Velocity")
    l2 = Line2D([0], [0], color="teal", lw=1, label="El Velocity")
    ax.legend(ncol=2, fontsize="x-small", handles=[l1, l2])


def plot_torque_data(ax: plt.Axes, dataset: M1M3ICSAnalysis) -> None:
    """
    Plot the azimuth and elevation torques on the given axes.

    Parameters
    ----------
    ax : `matplotlib.axes._axes.Axes`
        The axes on which the torque data is plotted.
    dataset : `M1M3ICSAnalysis`
        The dataset object containing the data to be plotted and metadata.
    """
    ax.plot(dataset.df["az_actual_torque"], color="firebrick", label="Az Torque")
    ax.plot(dataset.df["el_actual_torque"], color="salmon", label="El Torque")
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_ylabel("Actual\n Torque\n [kN.m]")

    l1 = Line2D([0], [0], color="firebrick", lw=1, label="Az Torque")
    l2 = Line2D([0], [0], color="salmon", lw=1, label="El Torque")
    ax.legend(ncol=2, fontsize="x-small", handles=[l1, l2])


def plot_stable_region(
    fig: plt.Figure, begin: Time, end: Time, label: str = "", color: str = "b"
) -> plt.Polygon:
    """
    Highlight a stable region on the plot with a colored span.

    Parameters
    ----------
    fig : `plt.Figure`
        The figure containing the axes on which the stable region is
        highlighted.
    begin : `astropy.time.Time`
        The beginning time of the stable region.
    end : `astropy.time.Time`
        The ending time of the stable region.
    label : `str`, optional
        The label for the highlighted region.
    color : `str`, optional
        The color of the highlighted region.

    Returns
    -------
    polygon : `matplotlib.patches.Polygon`
        The Polygon object representing the highlighted region.
    """
    for ax in fig.axes[1:]:
        span = ax.axvspan(begin.datetime, end.datetime, fc=color, alpha=0.1, zorder=-2, label=label)
    return span


def plot_hp_measured_data(
    dataset: M1M3ICSAnalysis,
    fig: plt.Figure,
    commands: dict[Time, str] | None = None,
    log: logging.Logger | None = None,
) -> plt.Figure:
    """
    Create and plot hardpoint measured data, velocity, and torque on subplots.
    This plot was designed for a figure with `figsize=(10, 7)` and `dpi=120`.

    Parameters
    ----------
    dataset : `M1M3ICSAnalysis`
        The dataset object containing the data to be plotted and metadata.
    fig : `plt.Figure`
        The figure to be plotted on.
    commands : `dict`, optional
        A dictionary times at which commands were issued, and with the values
        as the command strings themselves.
    log : `logging.Logger`, optional
        The logger object to log progress.
    """
    log = log.getChild(__name__) if log is not None else logging.getLogger(__name__)

    # Start clean
    fig.clear()

    # Add subplots
    gs = fig.add_gridspec(4, 1, height_ratios=[0.5, 2, 1, 1])

    ax_label = fig.add_subplot(gs[0])
    ax_hp = fig.add_subplot(gs[1])
    ax_tor = fig.add_subplot(gs[2], sharex=ax_hp)
    ax_vel = fig.add_subplot(gs[3], sharex=ax_hp)

    # Remove frame from axis dedicated to label
    ax_label.axis("off")

    # Convert the dataframe index to seconds using `slew_begin` as reference
    slew_begin = dataset.event.begin.to_datetime(timezone=pytz.timezone("UTC"))
    slew_end = (dataset.event.end.to_datetime(timezone=pytz.timezone("UTC")) - slew_begin).total_seconds()

    original_index = dataset.df.index.copy()
    dataset.df.index = (dataset.df.index - slew_begin).total_seconds()

    # Plotting
    line_list: list[plt.Line2D] = []
    for hp in range(dataset.number_of_hardpoints):
        topic = dataset.measured_forces_topics[hp]
        line = plot_hp_data(ax_hp, dataset.df[topic], f"HP{hp+1}")
        line_list.append(line)

    # Since slew_begin is our reference time, we can set it to zero
    mark_slew_begin_end(ax_hp, slew_begin=0, slew_end=slew_end)
    mark_slew_begin_end(ax_vel, slew_begin=0, slew_end=slew_end)
    line = mark_slew_begin_end(ax_tor, slew_begin=0, slew_end=slew_end)
    line_list.append(line)

    outer_pad = dataset.outer_pad.value
    ax_hp.set_xlim(-outer_pad, slew_end + outer_pad)

    plot_velocity_data(ax_vel, dataset)
    plot_torque_data(ax_tor, dataset)

    lineColors = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]  # cycle through the colors
    colorCounter = 0
    if commands is not None:
        for commandTime, command in commands.items():
            commandTime = (commandTime - slew_begin).total_seconds()
            command = command.replace("lsst.sal.", "")
            for ax in (ax_hp, ax_tor, ax_vel):  # so that the line spans all plots
                line = ax.axvline(
                    commandTime,
                    c=lineColors[colorCounter],
                    ls="--",
                    alpha=0.75,
                    label=f"{command}",
                )
            line_list.append(line)  # put it in the legend
            colorCounter += 1  # increment color so each line is different

    customize_hp_plot(ax_hp, line_list)
    # ax_hp.xaxis.set_ticklabels([])
    # ax_tor.xaxis.set_ticklabels([])
    ax_vel.set_xlabel("Time from slew start [s]")

    handles, labels = ax_hp.get_legend_handles_labels()
    ncol = 4 if len(handles) <= 10 else 5

    ax_label.legend(
        handles,
        labels,
        loc="lower center",
        frameon=False,
        ncol=ncol,
        fontsize="x-small",
        bbox_to_anchor=(0.5, -0.2),
    )

    # Ugly workaround to keep the index as timestamps
    dataset.df.index = original_index
    customize_fig(fig, dataset)

    return fig
