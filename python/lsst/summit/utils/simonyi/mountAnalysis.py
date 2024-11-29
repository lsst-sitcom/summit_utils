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

__all__ = ["calculateMountErrors", "plotMountErrors"]

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import num2date
from matplotlib.ticker import FuncFormatter

from lsst.summit.utils.tmaUtils import filterBadValues
from lsst.summit.utils.utils import dayObsIntToString

from .mountData import getAzElRotDataForExposure

if TYPE_CHECKING:
    from lsst_efd_client import EfdClient

    from lsst.daf.butler import DimensionRecord

    from .mountData import MountData


NON_TRACKING_IMAGE_TYPES = ["BIAS", "FLAT"]

COMCAM_ANGLE_TO_EDGE_OF_FIELD_ARCSEC = 1800.0
LSSTCAM_ANGLE_TO_EDGE_OF_FIELD_ARCSEC = 8500.0

# These levels determine the colouring of the cells in the RubinTV.
# Yellow for warning level, red for bad level
MOUNT_IMAGE_WARNING_LEVEL = 0.05
MOUNT_IMAGE_BAD_LEVEL = 0.10  # and red for this


@dataclass
class MountErrors:
    azRms: float
    elRms: float
    rotRms: float
    imageAzRms: float
    imageElRms: float
    imageRotRms: float
    imageImpactRms: float
    residualFiltering: bool
    nReplacedAz: int
    nReplacedEl: int


def tickFormatter(value: float, tick_number: float) -> str:
    # Convert the value to a string without subtracting large numbers
    # tick_number is unused.
    return f"{value:.2f}"


def calculateMountErrors(
    expRecord: DimensionRecord,
    client: EfdClient,
    maxDelta=0.1,
    doFilterResiduals=True,
) -> tuple[MountErrors, MountData] | False:
    """Queries EFD for a given exposure and calculates the RMS errors in the
    axes during the exposure, optionally plotting and saving the data.
    """
    logger = logging.getLogger(__name__)

    start = time.time()
    imgType = expRecord.observation_type.upper()
    if imgType in NON_TRACKING_IMAGE_TYPES:
        logger.info(f"Skipping mount torques for non-tracking image type {imgType} for {expRecord.id}")
        return False

    mountData = getAzElRotDataForExposure(client, expRecord)

    elevation = 90 - expRecord.zenith_angle

    azError = mountData.azimuthData["azError"].values
    elError = mountData.elevationData["elError"].values
    rotError = mountData.rotationData["rotError"].values
    if doFilterResiduals:
        # Filtering out bad values
        nReplacedAz = filterBadValues(azError, maxDelta)
        nReplacedEl = filterBadValues(elError, maxDelta)
        mountData.azimuthData["azError"] = azError
        mountData.elevationData["elError"] = elError
    azRms = np.sqrt(np.mean(azError * azError))
    elRms = np.sqrt(np.mean(elError * elError))
    rotRms = np.sqrt(np.mean(rotError * rotError))

    # Calculate Image impact RMS
    imageAzRms = azRms * np.cos(elevation * np.pi / 180.0)
    imageElRms = elRms
    imageRotRms = rotRms * COMCAM_ANGLE_TO_EDGE_OF_FIELD_ARCSEC * np.pi / 180.0 / 3600.0
    imageImpactRms = np.sqrt(imageAzRms**2 + imageElRms**2 + imageRotRms**2)

    end = time.time()
    elapsed = end - start
    logger.info(f"Elapsed time for EFD queries = {elapsed}")

    mountErrors = MountErrors(
        azRms=azRms,
        elRms=elRms,
        rotRms=rotRms,
        imageAzRms=imageAzRms,
        imageElRms=imageElRms,
        imageRotRms=imageRotRms,
        imageImpactRms=imageImpactRms,
        residualFiltering=doFilterResiduals,
        nReplacedAz=nReplacedAz,
        nReplacedEl=nReplacedEl,
    )

    return (mountErrors, mountData)


def plotMountErrors(
    mountData: MountData,
    mountErrors: MountErrors,
    figure=None,
    saveFilename: str = "",
):
    imageImpactRms = mountErrors.imageImpactRms
    expRecord = mountData.expRecord
    if expRecord is not None:
        dayObsString = dayObsIntToString(expRecord.day_obs)
        dataIdString = f"{expRecord.instrument} {dayObsString} - seqNum {expRecord.seq_num}"
        title = f"{dataIdString} - Exposure time = {expRecord.exposure_time:.1f}s"

    if figure is None:
        figure = plt.figure(figsize=(12, 8))

    utc = ZoneInfo("UTC")
    chile_tz = ZoneInfo("America/Santiago")

    # Function to convert UTC to Chilean time
    def offset_time_aware(utc_time):
        # Ensure the time is timezone-aware in UTC
        if utc_time.tzinfo is None:
            utc_time = utc.localize(utc_time)
        return utc_time.astimezone(chile_tz)

    [[ax1, ax4], [ax2, ax5], [ax3, ax6]] = figure.subplots(
        3,
        2,
        sharex="col",
        sharey=False,
        gridspec_kw={"wspace": 0.25, "hspace": 0, "height_ratios": [2.5, 1, 1], "width_ratios": [1.5, 1]},
    )
    # [ax1, ax4] = [azimuth, rotator]
    # [ax2, ax5] = [azError, rotError]
    # [ax3, ax6] = [azTorque, rotTorque]

    # Use the native color cycle for the lines. Because they're on
    # different axes they don't cycle by themselves
    axs = [ax1, ax2, ax3, ax4, ax5, ax6]
    lineColors = [p["color"] for p in plt.rcParams["axes.prop_cycle"]]
    nColors = len(lineColors)
    colorCounter = 0

    ax1.plot(
        mountData.azimuthData["actualPosition"],
        label="Azimuth position",
        c=lineColors[colorCounter % nColors],
    )
    colorCounter += 1
    ax1.yaxis.set_major_formatter(FuncFormatter(tickFormatter))
    ax1.set_ylabel("Azimuth (degrees)")

    ax1_twin = ax1.twinx()
    ax1_twin.plot(
        mountData.elevationData["actualPosition"],
        label="Elevation position",
        c=lineColors[colorCounter % nColors],
    )
    colorCounter += 1

    ax2.plot(
        mountData.azimuthData["azError"],
        label="Azimuth tracking error",
        c=lineColors[colorCounter % nColors],
    )
    colorCounter += 1
    ax2.plot(
        mountData.elevationData["elError"],
        label="Elevation tracking error",
        c=lineColors[colorCounter % nColors],
    )
    colorCounter += 1
    ax2.axhline(0.01, ls="-.", color="black")
    ax2.axhline(-0.01, ls="-.", color="black")
    ax2.yaxis.set_major_formatter(FuncFormatter(tickFormatter))
    ax2.set_ylabel("Tracking error (arcsec)")
    ax2.set_xticks([])  # remove x tick labels on the hidden upper x-axis
    ax2.set_ylim(-0.05, 0.05)
    ax2.set_yticks([-0.04, -0.02, 0.0, 0.02, 0.04])
    ax2.legend()
    ax2.text(0.1, 0.9, f"Image impact RMS = {imageImpactRms:.3f} arcsec (with rot).", transform=ax2.transAxes)
    if mountErrors.residualFiltering:
        ax2.text(
            0.1,
            0.8,
            (
                f"{mountErrors.nReplacedAz} bad az values and "
                f"{mountErrors.nReplacedEl} bad el values were replaced"
            ),
            transform=ax2.transAxes,
        )
    ax3_twin = ax3.twinx()
    ax3.plot(
        mountData.azimuthData["actualTorque"],
        label="Azimuth torque",
        c=lineColors[colorCounter % nColors],
    )
    colorCounter += 1
    ax3_twin.plot(
        mountData.elevationData["actualTorque"],
        label="Elevation torque",
        c=lineColors[colorCounter % nColors],
    )
    colorCounter += 1
    ax3.set_ylabel("Azimuth torque (Nm)")
    ax3_twin.set_ylabel("Elevation torque (Nm)")
    ax3.set_xlabel("Time (UTC)")  # yes, it really is UTC, matplotlib converts this automatically!

    # put the ticks at an angle, and right align with the tick marks
    ax3.set_xticks(ax3.get_xticks())  # needed to supress a user warning
    xlabels = ax3.get_xticks()
    ax3.set_xticklabels(xlabels)
    ax3.tick_params(axis="x", rotation=45)
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    ax4.plot(
        mountData.rotationData["actualPosition"],
        label="Rotator position",
        c=lineColors[colorCounter % nColors],
    )
    colorCounter += 1
    ax4.yaxis.set_major_formatter(FuncFormatter(tickFormatter))
    ax4.yaxis.tick_right()
    ax4.set_ylabel("Rotator angle (degrees)")
    ax4.yaxis.set_label_position("right")
    ax5.plot(
        mountData.rotationData["rotError"],
        c=lineColors[colorCounter % nColors],
    )

    colorCounter += 1
    ax5.axhline(0.1, ls="-.", color="black")
    ax5.axhline(-0.1, ls="-.", color="black")
    ax5.yaxis.set_major_formatter(FuncFormatter(tickFormatter))
    ax5.set_ylabel("Tracking error (arcsec)")
    ax5.tick_params(labelbottom=False)  # Hide x-axis tick labels without removing ticks
    ax5.set_ylim(-0.5, 0.5)
    ax5.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
    ax5.yaxis.tick_right()
    ax5.yaxis.set_label_position("right")

    ax6.plot(mountData.rotationTorques["torque0"], label="Torque0", c=lineColors[colorCounter % nColors])
    colorCounter += 1
    ax6.plot(mountData.rotationTorques["torque1"], label="Torque1", c=lineColors[colorCounter % nColors])
    ax6.set_ylabel("Rotator torque (Nm)")
    ax6.set_xlabel("Time (UTC)")  # yes, it really is UTC, matplotlib converts this automatically!
    # put the ticks at an angle, and right align with the tick marks
    ax6.set_xticks(ax6.get_xticks())  # needed to supress a user warning
    xlabels = ax6.get_xticks()
    ax6.set_xticklabels(xlabels)
    ax6.tick_params(axis="x", rotation=45)
    ax6.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax6.yaxis.tick_right()
    ax6.yaxis.set_label_position("right")
    ax6.legend()

    ax1_twin.yaxis.set_major_formatter(FuncFormatter(tickFormatter))
    ax1_twin.set_ylabel("Elevation (degrees)")
    ax1.tick_params(labelbottom=False)  # Hide x-axis tick labels without removing ticks
    # combine the legends and put inside the plot
    handles1a, labels1a = ax1.get_legend_handles_labels()
    handles1b, labels1b = ax1_twin.get_legend_handles_labels()
    handles2a, labels2a = ax3.get_legend_handles_labels()
    handles2b, labels2b = ax3_twin.get_legend_handles_labels()
    handles = handles1a + handles1b + handles2a + handles2b
    labels = labels1a + labels1b + labels2a + labels2b
    # ax2 is "in front" of ax1 because it has the vlines plotted on it, and
    # vlines are on ax2 so that they appear at the bottom of the legend, so
    # make sure to plot the legend on ax2, otherwise the vlines will go on
    # top of the otherwise-opaque legend.
    ax1_twin.legend(handles, labels, facecolor="white", framealpha=1)

    ax1.set_title("Azimuth and Elevation")
    ax4.set_title("Rotator")
    figure.suptitle(title, fontsize=14, y=1.01)  # Adjust y to move the title up

    # Create the upper axis for Chilean time
    ax1_twiny = ax1.twiny()
    ax1_twiny.set_xlim(ax1.get_xlim())  # Set the limits of the upper axis to match the lower axis
    utcTicks = ax1.get_xticks()  # Use the same ticks as the lower UTC axis
    utcTickLabels = [num2date(tick, tz=utc) for tick in utcTicks]
    chileTickLabels = [offset_time_aware(label) for label in utcTickLabels]
    # Set the same tick positions but with Chilean time labels
    ax1_twiny.set_xticks(utcTicks)
    ax1_twiny.set_xticklabels([tick.strftime("%H:%M:%S") for tick in chileTickLabels])
    ax1_twiny.set_xlabel("Time (Chilean Time)")

    ax4_twiny = ax4.twiny()
    ax4_twiny.set_xlim(ax4.get_xlim())  # Set the limits of the upper axis to match the lower axis
    utcTicks = ax4.get_xticks()  # Use the same ticks as the lower UTC axis
    utcTickLabels = [num2date(tick, tz=utc) for tick in utcTicks]
    chileTickLabels = [offset_time_aware(label) for label in utcTickLabels]
    # Set the same tick positions but with Chilean time labels
    ax4_twiny.set_xticks(utcTicks)
    ax4_twiny.set_xticklabels([tick.strftime("%H:%M:%S") for tick in chileTickLabels])
    ax4_twiny.set_xlabel("Time (Chilean Time)")

    # Add exposure start and end:
    for ax in axs:
        if expRecord is not None:
            ax.axvline(mountData.expRecord.timespan.begin.utc.datetime, ls="--", color="green")
            ax.axvline(mountData.expRecord.timespan.end.utc.datetime, ls="--", color="red")

    if saveFilename:
        plt.savefig(saveFilename)

    return figure
