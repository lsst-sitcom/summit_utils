from typing import Protocol

import pandas as pd
from astropy import units as u
from lsst.summit.utils.efdUtils import EfdClient


class Event(Protocol):
    """TMA Event."""

    @property
    def dayObs(self) -> int:
        """Day of the observation."""

    @property
    def seqNum(self) -> int:
        """Day of the observation."""

    @property
    def version(self) -> int:
        """Version of the TMAEventMaker."""


class M1M3ICSAnalysis(Protocol):
    """M1M3ICSAnalysis."""

    @property
    def event(self) -> Event:
        """Event."""

    @property
    def inner_pad(self) -> u.Quantity:
        """Inner pad."""

    @property
    def outer_pad(self) -> u.Quantity:
        """Outer pad."""

    @property
    def n_sigma(self) -> int:
        """Number of sigma."""

    @property
    def client(self) -> EfdClient:
        """EFD client."""

    @property
    def number_of_hardpoints(self) -> int:
        """Number of hardpoints."""

    @property
    def measured_forces_topics(self) -> list:
        """Measured forces topics."""

    @property
    def df(self) -> pd.DataFrame:
        """Dataframe."""

    @property
    def stats(self) -> pd.DataFrame:
        """Statistics."""
