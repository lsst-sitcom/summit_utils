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

__all__ = [
    "PeekExposureTaskConfig",
    "PeekExposureTask",
]

from typing import Any

from deprecated.sphinx import deprecated

from lsst.pipe.tasks.peekExposure import PeekExposureTask, PeekExposureTaskConfig


@deprecated(
    reason="PeekExposureTaskConfig has been moved to pipe_tasks. Will be removed after v28.0.",
    version="v28.0",
    category=FutureWarning,
)
class PeekExposureTaskConfig(PeekExposureTaskConfig):
    """Config class for the PeekExposureTask."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)


@deprecated(
    reason="PeekExposureTask has been moved to pipe_tasks. Will be removed after v28.0.",
    version="v28.0",
    category=FutureWarning,
)
class PeekExposureTask(PeekExposureTask):
    """ """

    def __init__(self, config: Any, *args: Any, **kwargs: Any):
        super().__init__(config=config, *args, **kwargs)
