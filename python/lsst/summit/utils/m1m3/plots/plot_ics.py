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
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, TypedDict

import matplotlib.pyplot as plt
import pandas as pd
from astropy.time import Time
from matplotlib.patches import Patch

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ..inertia_compensation_system import M1M3ICSAnalysis

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
    "HP_FATIGUE_LIMIT_COMPRESSION",
    "HP_FATIGUE_LIMIT_TENSION",
    "HP_OPERATIONAL_LIMIT_COMPRESSION",
    "HP_OPERATIONAL_LIMIT_TENSION",
    "FIGURE_WIDTH",
    "FIGURE_HEIGHT",
]

# Approximate value for breakaway
HP_BREAKAWAY_LIMIT: float = 3000  # [N]

# limit that can still damage the mirror with fatigue
HP_FATIGUE_LIMIT_COMPRESSION: float = 1042  # [N]
HP_FATIGUE_LIMIT_TENSION: float = -1163.25  # [N]

# desired operational limit
HP_OPERATIONAL_LIMIT_COMPRESSION: float = 521  # [N]
HP_OPERATIONAL_LIMIT_TENSION: float = -581  # [N]

FIGURE_WIDTH = 10
FIGURE_HEIGHT = 7


class HPLimitsDict(TypedDict, total=False):
    pos_limit: float
    neg_limit: float
    ls: str


def plot_hp_data(ax: plt.Axes, data: pd.Series | list, label: str) -> plt.Line2D:
    """
    Plot hardpoint data on the given axes.

    Parameters
    ----------
    ax : `plt.Axes`
        The axes on which the data is plotted.
    topic : `str`
        The topic of the data.
    data : `Series` or `list`
        The data points to be plotted.
    label : `str`
        The label for the plotted data.

    Returns
    -------
    lines : `plt.Line2D`
        The plotted data as a Line2D object.
    """
    line = ax.plot(data, "-", label=label, lw=0.5)
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
    _ = ax.axvline(slew_begin.datetime, lw=0.5, ls="--", c="k", zorder=-1)
    line = ax.axvline(slew_end.datetime, lw=0.5, ls="--", c="k", zorder=-1, label="Slew Start/Stop")
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


def customize_fig(fig: plt.Figure, dataset: M1M3ICSAnalysis) -> None:
    """
    Add a title to a figure and adjust its subplots spacing

    Paramters
    ---------
    fig : `matplotlib.pyplot.Figure`
        Figure to be custoized.
    dataset : `M1M3ICSAnalysis`
        The dataset object containing the data to be plotted and metadata.
    """
    t_fmt = "%Y%m%d %H:%M:%S"
    fig.suptitle(
        f"HP Measured Data\n "
        f"DayObs {dataset.event.dayObs} "
        f"SeqNum {dataset.event.seqNum} "
        f"v{dataset.event.version}\n "
        f"{dataset.df.index[0].strftime(t_fmt)} - "
        f"{dataset.df.index[-1].strftime(t_fmt)}"
    )

    fig.subplots_adjust(hspace=0)


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

    ax.set_xlabel("Time [UTC]")
    ax.set_ylabel("HP Measured\n Forces [N]")
    ax.set_ylim(-3100, 3100)
    ax.grid(linestyle=":", alpha=0.2)


def add_hp_limits(ax: plt.Axes) -> list[plt.Line2D]:
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
    hp_limits: dict[str, HPLimitsDict] = {
        "HP Breakaway Limit": {
            "pos_limit": HP_BREAKAWAY_LIMIT,
            "neg_limit": -HP_BREAKAWAY_LIMIT,
            "ls": "-",
        },
        "Repeated Load Limit (30% breakaway)": {
            "pos_limit": HP_FATIGUE_LIMIT_COMPRESSION,
            "neg_limit": HP_FATIGUE_LIMIT_TENSION,
            "ls": "--",
        },
        "Normal Ops Limit (15% breakaway)": {
            "pos_limit": HP_OPERATIONAL_LIMIT_COMPRESSION,
            "neg_limit": HP_OPERATIONAL_LIMIT_TENSION,
            "ls": ":",
        },
    }

    kwargs: dict[str, Any] = dict(alpha=0.5, lw=1.0, c="r", zorder=-1)
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
    ax.grid(linestyle=":", alpha=0.2)
    ax.set_ylabel("Actual Velocity\n [deg/s]")
    ax.legend(ncol=2, fontsize="x-small")


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
    ax.grid(linestyle=":", alpha=0.2)
    ax.set_ylabel("Actual Torque\n [kN.m]")
    ax.legend(ncol=2, fontsize="x-small")


def plot_stable_region(
    fig: plt.Figure, begin: Time, end: Time, label: str = "", color: str = "b"
) -> Optional[Patch]:
    """Highlight a stable region on the plot with a colored span.

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
    patch : `matplotlib.patches.Patch`
        The patch object representing the highlighted region.
    """
    span = None  # Fixes mypy error about uninitialized variable
    for ax in fig.axes[1:]:
        span = ax.axvspan(begin.datetime, end.datetime, fc=color, alpha=0.1, zorder=-2, label=label)
    return span


def plot_hp_measured_data(
    dataset: M1M3ICSAnalysis,
    fig: plt.Figure,
    commands: dict[Time, str] | None = None,
    log: logging.Logger | None = None,
) -> Figure:
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
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 2, 1, 1])

    ax_label = fig.add_subplot(gs[0])
    ax_hp = fig.add_subplot(gs[1])
    ax_tor = fig.add_subplot(gs[2], sharex=ax_hp)
    ax_vel = fig.add_subplot(gs[3], sharex=ax_hp)

    # Remove frame from axis dedicated to label
    ax_label.axis("off")

    # Plotting
    line_list: list[plt.Line2D] = []
    for hp in range(dataset.number_of_hardpoints):
        topic = dataset.measured_forces_topics[hp]
        line = plot_hp_data(ax_hp, dataset.df[topic], f"HP{hp + 1}")
        line_list.append(line)

    slew_begin = Time(dataset.event.begin, scale="utc")
    slew_end = Time(dataset.event.end, scale="utc")

    mark_slew_begin_end(ax_hp, slew_begin, slew_end)
    mark_slew_begin_end(ax_vel, slew_begin, slew_end)
    line = mark_slew_begin_end(ax_tor, slew_begin, slew_end)
    line_list.append(line)

    mark_padded_slew_begin_end(ax_hp, slew_begin - dataset.outer_pad, slew_end + dataset.outer_pad)
    mark_padded_slew_begin_end(ax_vel, slew_begin - dataset.outer_pad, slew_end + dataset.outer_pad)
    line = mark_padded_slew_begin_end(ax_tor, slew_begin - dataset.outer_pad, slew_end + dataset.outer_pad)
    line_list.append(line)

    plot_velocity_data(ax_vel, dataset)
    plot_torque_data(ax_tor, dataset)

    lineColors = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]  # cycle through the colors
    colorCounter = 0
    if commands is not None:
        for commandTime, command in commands.items():
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

    handles, labels = ax_hp.get_legend_handles_labels()
    ax_label.legend(handles, labels, loc="center", frameon=False, ncol=4, fontsize="x-small")

    customize_fig(fig, dataset)

    return fig
