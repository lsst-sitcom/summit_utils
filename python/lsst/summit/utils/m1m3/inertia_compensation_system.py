#!/usr/bin/python
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.time import Time
from lsst.summit.utils.m1m3.plots import inertia_compensation_system
from lsst.summit.utils.efdUtils import EfdClient, getEfdData
from lsst.summit.utils.tmaUtils import TMAEvent, TMAEventMaker
from lsst.ts.xml.tables.m1m3 import FATABLE_XFA, FATABLE_YFA, FATABLE_ZFA, HP_COUNT

__all__ = ["M1M3ICSAnalysis"]


class M1M3ICSAnalysis:
    """
    Evaluate the M1M3 Inertia Compensation System's performance by
    calculating the minima, maxima and peak-to-peak values during a
    slew. In addition, calculates the mean, median and standard deviation
    when the slew has contant velocity or zero acceleration.

    Parameters
    ----------
    event : `lsst.summit.utils.tmaUtils.TMAEvent`
        Abtract representation of a slew event.
    efd_client : EfdClient
        Client to access the EFD.
    inner_pad : float, optional
        Time padding inside the stable time window of the slew.
    outer_pad : float, optional
        Time padding outside the slew time window.
    n_sigma : float, optional
        Number of standard deviations to use for the stable region.
    log : logging.Logger, optional
        Logger object to use for logging messages.
    """

    def __init__(
        self,
        event: TMAEvent,
        efd_client: EfdClient,
        inner_pad: float = 1.0,
        outer_pad: float = 1.0,
        n_sigma: float = 1.0,
        log: logging.Logger | None = None,
    ):
        self.log = (
            log.getChild(type(self).__name__)
            if log is not None
            else logging.getLogger(type(self).__name__)
        )

        self.event = event
        self.inner_pad = inner_pad * u.second
        self.outer_pad = outer_pad * u.second
        self.n_sigma = n_sigma
        self.client = efd_client

        self.number_of_hardpoints = HP_COUNT
        self.measured_forces_topics = [
            f"measuredForce{i}" for i in range(self.number_of_hardpoints)
        ]

        self.applied_forces_topics = (
            [f"xForces{actuator}" for actuator in range(FATABLE_XFA)]
            + [f"yForces{actuator}" for actuator in range(FATABLE_YFA)]
            + [f"zForces{actuator}" for actuator in range(FATABLE_ZFA)]
        )

        self.log.info(f"Query datasets for {event.dayObs} {event.seqNum}")
        self.df = self.query_dataset()

        self.log.info("Calculate statistics")
        self.stats = self.get_stats()

        self.log.info("Pack results into a Series")
        self.stats = self.pack_stats_series()

    def find_stable_region(self) -> tuple[Time, Time]:
        """
        Find the stable region of the dataset. By stable, we mean the region
        where the torque is within n_sigma of the mean.

        Returns
        -------
        tuple[Time, Time]:
            The begin and end times of the stable region.
        """
        az_torque = self.df["az_actual_torque"]
        az_torque_regions = find_adjacent_true_regions(
            np.abs(az_torque - az_torque.mean()) < self.n_sigma * az_torque.std()
        )

        el_torque = self.df["el_actual_torque"]
        el_torque_regions = find_adjacent_true_regions(
            np.abs(el_torque - el_torque.mean()) < self.n_sigma * el_torque.std()
        )

        stable_begin = max([reg[0] for reg in az_torque_regions + el_torque_regions])
        stable_begin = Time(stable_begin, scale="utc")

        stable_end = min([reg[-1] for reg in az_torque_regions + el_torque_regions])
        stable_end = Time(stable_end, scale="utc")

        return stable_begin, stable_end

    def query_dataset(self) -> pd.DataFrame:
        """
        Query all the relevant data, resample them to have the same requency
        and merge them in a single dataframe.

        Returns
        -------
        pd.DataFrame
        """
        evt = self.event
        query_config = {
            "hp_measured_forces": {
                "topic": "lsst.sal.MTM1M3.hardpointActuatorData",
                "columns": self.measured_forces_topics,
                "err_msg": f"No hard-point data found for event {evt.seqNum} on {evt.dayObs}",
            },
            "m1m3_applied_velocity_forces": {
                "topic": "lsst.sal.MTM1M3.appliedVelocityForces",
                "columns": self.applied_forces_topics,
                "err_msg": None,
                "rename_columns": {col: f"avf_{col}" for col in self.applied_forces_topics},
            },
            "m1m3_applied_acceleration_forces": {
                "topic": "lsst.sal.MTM1M3.appliedAccelerationForces",
                "columns": self.applied_forces_topics,
                "err_msg": None,
                "rename_columns": {col: f"aaf_{col}" for col in self.applied_forces_topics},
            },
            "tma_az": {
                "topic": "lsst.sal.MTMount.azimuth",
                "columns": ["timestamp", "actualPosition", "actualVelocity", "actualTorque"],
                "err_msg": f"No TMA azimuth data found for event {evt.seqNum} on {evt.dayObs}",
                "reset_index": True,
                "rename_columns": {
                    "actualTorque": "az_actual_torque",
                    "actualVelocity": "az_actual_velocity",
                    "actualPosition": "az_actual_position",
                },
            },
            "tma_el": {
                "topic": "lsst.sal.MTMount.elevation",
                "columns": ["timestamp", "actualPosition", "actualVelocity", "actualTorque"],
                "err_msg": f"No TMA elevation data found for event {evt.seqNum} on {evt.dayObs}",
                "reset_index": True,
                "rename_columns": {
                    "actualPosition": "el_actual_position",
                    "actualTorque": "el_actual_torque",
                    "actualVelocity": "el_actual_velocity",
                },
            },
        }

        # Query datasets
        queries = {key: self.query_efd_data(**cfg) for key, cfg in query_config.items()}  # type: ignore

        # Merge datasets
        df = self.merge_datasets(queries)

        # Convert torque from Nm to kNm
        cols = ["az_actual_torque", "el_actual_torque"]
        df.loc[:, cols] *= 1e-3

        return df

    def merge_datasets(self, queries: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple datasets based on their timestamps.

        Parameters
        ----------
        queries (dict[str, pd.DataFrame]):
            A dictionary of dataframes to be merged.

        Returns
        -------
        df : pd.DataFrame
            A merged dataframe.
        """
        merge_cfg = {
            "left_index": True,
            "right_index": True,
            "tolerance": timedelta(seconds=1),
            "direction": "nearest",
        }

        self.log.info("Merging datasets")
        df_list = [df for _, df in queries.items()]
        merged_df = df_list[0]

        for df in df_list[1:]:
            merged_df = pd.merge_asof(merged_df, df, **merge_cfg)

        return merged_df

    def query_efd_data(
        self,
        topic: str,
        columns: list[str],
        err_msg: str | None = None,
        reset_index: bool = False,
        rename_columns: dict | None = None,
        resample: float | None = None,
    ) -> pd.DataFrame:
        """
        Query the EFD data for a given topic and return a dataframe.

        Parameters
        ----------
        topic : str
            The topic to query.
        columns : List[str]
            The columns to query.
        err_msg : str, optional
            The error message to raise if no data is found.
            If None, it creates a dataframe padded with zeros.
        reset_index : bool, optional
            Whether to reset the index of the dataframe.
        rename_columns : dict, optional
            A dictionary of column names to rename.
        resample : float, optional
            The resampling frequency in seconds.

        Returns
        -------
        df : pd.DataFrame
            A dataframe containing the queried data.
            If no data is found and `err_msg` is None,
            returns a dataframe padded with zeros.
        """
        self.log.info(f"Querying dataset: {topic}")
        df = getEfdData(
            self.client,
            topic,
            columns=columns,
            event=self.event,
            prePadding=self.outer_pad,
            postPadding=self.outer_pad,
            warn=False,
        )

        self.log.debug(f"Queried {df.index.size} rows from {topic}")
        if df.index.size == 0:
            if err_msg is not None:
                self.log.error(err_msg)
                raise ValueError(err_msg)
            else:
                self.log.warning(
                    f"Empty dataset for {topic}. Returning a zero-padded dataframe."
                )
                begin_timestamp = pd.Timestamp(self.event.begin.unix, unit='s')
                end_timestamp = pd.Timestamp(self.event.end.unix, unit='s')
                index = pd.DatetimeIndex(
                    pd.date_range(begin_timestamp, end_timestamp, freq="1S")
                )
                df = pd.DataFrame(
                    columns=columns,
                    index=index,
                    data=np.zeros((index.size, len(columns))),
                )

        if rename_columns is not None:
            df = df.rename(columns=rename_columns)

        if reset_index:
            df["timestamp"] = Time(
                df["timestamp"], format="unix_tai", scale="utc"
            ).datetime
            df.set_index("timestamp", inplace=True)
            df.index = df.index.tz_localize("UTC")

        return df

    def get_midppoint(self) -> Time:
        """Return the halfway point between begin and end."""
        return self.df.index[len(self.df.index) // 2]

    def get_stats(self) -> pd.DataFrame:
        """
        Calculate statistics for each column in a given dataset.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing calculated statistics for each column in the
            dataset. For each column, the statistics include minimum, maximum,
            and peak-to-peak values.

        Notes
        -----
        This function computes statistics for each column in the provided
        dataset. It utilizes the `get_minmax` function to calculate minimum,
        maximum, and peak-to-peak values for each column's data.
        """
        cols = self.measured_forces_topics
        full_slew_stats = pd.DataFrame(
            data=[self.get_slew_minmax(self.df[col]) for col in cols], index=cols
        )
        self.log.info("Find stable time window")
        begin, end = self.find_stable_region()

        self.log.debug("Update begin and end times")
        begin = begin + self.inner_pad
        end = end - self.inner_pad

        self.log.debug("Calculate statistics in stable time window from M1M3")
        stable_slew_stats = pd.DataFrame(
            data=[
                self.get_stats_in_torqueless_interval(
                    self.df[col].loc[begin.isot: end.isot]
                )
                for col in cols
            ],
            index=cols,
        )

        self.log.debug("Concatenate statistics")
        stats = pd.concat((full_slew_stats, stable_slew_stats), axis=1)

        return stats

    @staticmethod
    def get_stats_in_torqueless_interval(s: pd.Series) -> pd.Series:
        """
        Calculate statistical measures within a torqueless interval.

        This static method computes descriptive statistics for a given pandas
        Series within a torqueless interval. The torqueless interval represents
        a period of the data analysis when no external torque is applied.

        Parameters
        ----------
        s : pandas.Series
            A pandas Series containing data values for analysis.

        Returns
        -------
        pandas.Series
            A pandas Series containing the following statistical measures:
            - Mean: The arithmetic mean of the data.
            - Median: The median value of the data.
            - Standard Deviation (Std): The standard deviation of the data.
        """
        result = pd.Series(
            data=[s.mean(), s.median(), s.std()],
            index=["mean", "median", "std"],
            name=s.name,
        )
        return result

    @staticmethod
    def get_slew_minmax(s: pd.Series) -> pd.Series:
        """
        Calculate minimum, maximum, and peak-to-peak values for a data-series.

        Parameters
        ----------
        s : pandas.Series
            The input pandas Series containing data.

        Returns
        -------
        pandas.Series
            A Series containing the following calculated values for the two
            halves of the input Series:
            - min: Minimum value of the Series.
            - max: Maximum value of the Series.
            - ptp: Peak-to-peak (ptp) value of the Series (abs(max - min)).
        """
        result = pd.Series(
            data=[s.min(), s.max(), np.ptp(s)],
            index=["min", "max", "ptp"],
            name=s.name,
        )
        return result

    def pack_stats_series(self) -> pd.Series:
        """
        Pack the stats DataFrame into a Series with custom index labels.

        This method takes the DataFrame of statistics stored in the 'stats'
        attribute of the current object and reshapes it into a Series where the
        indexes are generated using custom labels based on the column names and
        index positions. The resulting Series combines values from all columns
        of the DataFrame.

        Returns
        -------
        pandas.Series
            A Series with custom index labels based on the column names and
            index positions. The Series contains values from all columns of the
            DataFrame.
        """
        if isinstance(self.stats, pd.Series):
            self.log.info("Stats are already packed into a Series.")
            return self.stats

        self.log.info("Packing stats into a Series.")
        df = self.stats.transpose()

        # Define the prefix patterns
        column_prefixes = df.columns
        index_positions = df.index

        # Generate all combinations of prefixes and positions
        index_prefixes = [
            f"measuredForce{stat.capitalize()}{position}"
            for stat in index_positions
            for position, _ in enumerate(column_prefixes)
        ]

        # Flatten the DataFrame and set the new index
        result_series = df.stack().reset_index(drop=True)
        result_series.index = index_prefixes

        # Append the event information to the Series
        event_keys = [
            "dayObs",
            "seqNum",
            "version",
            "begin",
            "end",
            "duration",
            "type",
            "endReason",
        ]
        event_dict = vars(self.event)
        event_dict = {key: val for key, val in event_dict.items() if key in event_keys}

        # Create a pandas Series from the dictionary
        event_series = pd.Series(event_dict)

        # Create a new Pandas Series correlating event and system information
        system_series = pd.Series(
            {
                "az_start": self.get_nearest_value("az_actual_torque", self.event.begin),
                "az_end": self.get_nearest_value("az_actual_torque", self.event.end),
                "az_extreme_vel": self.get_extreme_value("az_actual_velocity"),
                "az_extreme_torque": self.get_extreme_value("az_actual_torque"),
                "el_start": self.get_nearest_value("el_actual_torque", self.event.begin),
                "el_end": self.get_nearest_value("el_actual_torque", self.event.end),
                "el_extreme_vel": self.get_extreme_value("el_actual_velocity"),
                "el_extreme_torque": self.get_extreme_value("el_actual_torque"),
                "ics_enabled": self.get_ics_status()
            }
        )

        system_series["az_diff"] = system_series["az_end"] - system_series["az_start"]
        system_series["el_diff"] = system_series["el_end"] - system_series["el_start"]

        # Concatenate the two Series
        result_series = pd.concat([event_series, system_series, result_series])

        # Rename the series columns
        result_series = result_series.rename(
            {
                "dayObs": "day_obs",
                "seqNum": "seq_num",
                "version": "version",
                "begin": "time_begin",
                "end": "time_end",
                "duration": "time_duration",
                "type": "slew_type",
                "endReason": "end_reason",
            }
        )

        # Display the resulting Series
        return result_series

    def get_extreme_value(self, column):
        """
        Return the most extreme (either max or min) value from a given column.

        Parameters
        ----------
        column : str
            The column to query.

        Returns
        -------
        extreme_val : float
            The most extreme value from the given column.
        """
        index_of_extreme = self.df[column].abs().idxmax()
        extreme_value = self.df.loc[index_of_extreme, column]
        return extreme_value

    def get_nearest_value(self, column, timestamp):
        """
        Return the nearest value to a given timestamp from a given column.

        Parameters
        ----------
        column : str
            The column to query.
        timestamp : astropy.time.Time
            The timestamp to query.
        """
        timestamp = pd.Timestamp(timestamp.iso, tz="UTC")
        time_diff = abs(self.df.index - timestamp)
        idx = time_diff.argmin()
        return self.df[column].iloc[idx]

    def get_ics_status(self, threshold: float = 1e-6) -> bool:
        """
        Evaluate the values of the applied velocity and acceleration forces
        inside the padded stable time window.
        If the values are all zero,
        then this function will return False as the ICS is not enabled.
        Otherwise, it will return True.

        Parameters
        ----------
        threshold : float, optional
            Threshold value used to determine if the ICS is enabled or not.
            If all the values of the applied velocity and acceleration forces
            are below this threshold, then the ICS is considered to be
            disabled.

        Returns
        -------
        bool
            True if the ICS is enabled, False otherwise.
        """
        avf0 = (self.df[[c for c in self.df.columns if "avf" in c]].abs() < threshold).all().eq(True).all()
        aaf0 = (self.df[[c for c in self.df.columns if "aaf" in c]].abs() < threshold).all().eq(True).all()
        return not (avf0 and aaf0)


def find_adjacent_true_regions(
    series: pd.Series, min_adjacent: None | int = None
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Find regions in a boolean Series containing adjacent True values.

    Parameters
    ----------
    series : pd.Series
        The boolean Series to search for regions.

    min_adjacent : int, optional
        Minimum number of adjacent True values in a region.
        Defaults to half size of the series.

    Returns
    -------
    list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]
        A list of tuples representing the start and end indices of regions
        containing more than or equal to min_adjacent adjacent True values.
    """
    min_adjacent = min_adjacent if min_adjacent else 0.5 * series.size
    regions = []
    for key, group in series.groupby((series != series.shift()).cumsum()):
        if key and len(group) >= min_adjacent:
            region_indices = group.index
            regions.append((region_indices.min(), region_indices.max()))
    return regions


def evaluate_m1m3_ics_single_slew(
    day_obs: int,
    seq_number: int,
    event_maker: TMAEventMaker,
    inner_pad: float = 1.0,
    outer_pad: float = 1.0,
    n_sigma: float = 1.0,
    log: logging.Logger | None = None,
) -> M1M3ICSAnalysis:
    """
    Evaluate the M1M3 Inertia Compensation System in a single slew with a
    `seqNumber` sequence number and observed during `dayObs`.

    Parameters
    ----------
    day_obs : int
        Observation day in the YYYYMMDD format.
    seq_number : int
        Sequence number associated with the slew event.
    event_maker : TMAEventMaker
        Object to retrieve TMA events.
    inner_pad : float, optional
        Time padding inside the stable time window of the slew.
    outer_pad : float, optional
        Time padding outside the slew time window.
    n_sigma : float, optional
        Number of standard deviations to use for the stable region.
    log : logging.Logger, optional
        Logger object to use for logging messages.

    Returns
    -------
    result : `lsst.summit.utils.m1m3.inertia_compensation_system.M1M3ICSAnalysis`
        The results of the analysis.

    Raises
    ------
    ValueError
        Raised if there is no hardpoint data for the specified event.
    """
    log = log.getChild(__name__) if log is not None else logging.getLogger(__name__)

    log.info("Retrieving TMA slew event.")
    event = event_maker.getEvent(day_obs, seq_number)
    if event is None:
        raise ValueError(f"Could not find event with {seq_number} in {day_obs}")

    log.info("Start inertia compensation system analysis.")
    performance_analysis = M1M3ICSAnalysis(
        event,
        event_maker.client,
        inner_pad=inner_pad,
        outer_pad=outer_pad,
        n_sigma=n_sigma,
        log=log,
    )

    return performance_analysis


def evaluate_m1m3_ics_day_obs(
    day_obs: int,
    event_maker: TMAEventMaker,
    inner_pad: float = 1.0,
    outer_pad: float = 1.0,
    n_sigma: float = 1.0,
    log: logging.Logger | None = None,
) -> pd.DataFrame:
    """
    Evaluate the M1M3 Inertia Compensation System in every slew event during a
    `dayObs`.

    Parameters
    ----------
    day_obs : int
        Observation day in the YYYYMMDD format.
    event_maker : TMAEventMaker
        Object to retrieve TMA events.
    inner_pad : float, optional
        Time padding inside the stable time window of the slew.
    outer_pad : float, optional
        Time padding outside the slew time window.
    n_sigma : float, optional
        Number of standard deviations to use for the stable region.
    log : logging.Logger, optional
        Logger object to use for logging messages.

    Returns
    -------
    pd.DataFrame
        Data-frame containing statistical summary of the analysis.
    """
    log = log.getChild(__name__) if log is not None else logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)

    log.info("Retrieving TMA slew events.")
    events = event_maker.getEvents(day_obs)
    log.info(f"Found {len(events)} events for day {day_obs}")

    stats = None
    for event in events:
        log.info(f"Start inertia compensation system analysis on {event.seqNum}.")

        try:
            performance_analysis = M1M3ICSAnalysis(
                event,
                event_maker.client,
                inner_pad=inner_pad,
                outer_pad=outer_pad,
                n_sigma=n_sigma,
                log=log,
            )
            log.info(
                f"Complete inertia compensation system analysis on {event.seqNum}."
            )
        except ValueError:
            log.warning(f"Missing data for {event.seqNum} on {event.dayObs}")
            continue

        if stats is None:
            stats = performance_analysis.stats
        else:
            stats = pd.concat((stats.T, performance_analysis.stats), axis=1).T

    assert isinstance(stats, pd.DataFrame)
    stats = stats.set_index("seq_num", drop=False)
    return stats


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    # Setup 
    day_obs = 20230728
    seq_num = 38

    print("Start - Single Slew")
    event_maker = TMAEventMaker()

    # Start analysis
    t0 = time.time()
    results = evaluate_m1m3_ics_single_slew(day_obs, seq_num, event_maker)
    t1 = time.time()
    print(f"That took {(t1-t0):.2f} seconds")

    # Save Results
    results.stats.to_csv(f"m1m3_ics_stats_{day_obs}_{seq_num}.csv")
    results.df.to_csv(f"m1m3_ics_df_{day_obs}_{seq_num}.csv")
    print(f"Result Series:\n{results.stats}")
    
    # Start plotting
    dpi = 180
    fig_size = (15, 10)
    filename = "m1m3_ics_hp_measured_data.png"
    fig = plt.figure(figsize=fig_size, dpi=dpi)

    fig = inertia_compensation_system.plot_hp_measured_data(results, fig)
    fig.savefig(filename)
    print("End - Single Slew")
