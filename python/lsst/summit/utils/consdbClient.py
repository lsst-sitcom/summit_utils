# This file is part of summit_utils.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import logging
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlparse

import requests
from astropy.table import Table

__all__ = ["ConsDbClient", "FlexibleMetadataInfo"]


logger = logging.getLogger(__name__)


def _urljoin(*args: str) -> str:
    """Join parts of a URL with slashes.

    Does not do any quoting.  Mostly to remove a level of list-making.

    Parameters
    ----------
    *args : `str`
        Each parameter is a URL part.

    Returns
    -------
    url : `str`
        The joined URL.
    """
    return "/".join(args)


def _redact(url: str) -> str:
    """Redact authentication information from a URL.

    Parameters
    ----------
    url : `str`
        A URL.

    Returns
    -------
    safe_url : `str`
        The URL with any password component removed.
    """
    parsed = urlparse(url)
    if parsed.password:
        parsed = parsed._replace(
            netloc=f"{parsed.username}:***@{parsed.hostname}",
        )
    return parsed.geturl()


def _check_status(r: requests.Response) -> None:
    """Check the status of an HTTP response and raise if an error.

    Adds additional response information to the raise_for_status exception.

    Parameters
    ----------
    r : `requests.Response`
        The response to check.

    Raises
    ------
    requests.HTTPError
        Raised if a non-successful status is returned.
    """
    r.url = _redact(r.url)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        try:
            e.add_note(str(e.response.json()))
        except requests.JSONDecodeError:
            pass
        raise e


@dataclass
class FlexibleMetadataInfo:
    """Description of a flexible metadata value.

    Parameters
    ----------
    dtype : `str`
        Data type of the flexible metadata value.
        One of ``bool``, ``int``, ``float``, or ``str``.
    doc : `str`
        Documentation string.
    unit : `str`, optional
        Unit of the value.
    ucd : `str`, optional
        IVOA Unified Content Descriptor
        (https://www.ivoa.net/documents/UCD1+/).
    """

    dtype: str
    doc: str
    unit: str | None = None
    ucd: str | None = None


class ConsDbClient:
    """A client library for accessing the Consolidated Database.

    This library provides a basic interface for using flexible metadata
    (key/value pairs associated with observation ids from an observation
    type table), determining the schema of ConsDB tables, querying the
    ConsDB using a general SQL SELECT statement, and inserting into
    ConsDB tables.

    Parameters
    ----------
    url : `str`, optional
        Base URL of the Web service, defaults to the value of environment
        variable ``LSST_CONSDB_PQ_URL`` (the location of the publish/query
        service).

    Notes
    -----
    This client is a thin layer over the publish/query Web service, which
    avoids having a dependency on database drivers.

    It enforces the return of query results as Astropy Tables.
    """

    def __init__(self, url: str | None = None):
        self.session = requests.Session()
        if url is None:
            self.url = os.environ["LSST_CONSDB_PQ_URL"]
        else:
            self.url = url
        self.url = self.url.rstrip("/")

    def _handle_get(self, url: str, query: dict[str, str | list[str]] | None = None) -> Any:
        """Submit GET requests to the server.

        Parameters
        ----------
        url : `str`
            URL to GET.
        query : `dict` [`str`, `str` | `list` [`str`]], optional
            Query parameters to attach to the URL.

        Raises
        ------
        requests.RequestException
            Raised if any kind of connection error occurs.
        requests.HTTPError
            Raised if a non-successful status is returned.
        requests.JSONDecodeError
            Raised if the result does not decode as JSON.

        Returns
        -------
        result : `Any`
            Result of decoding the Web service result content as JSON.
        """
        logger.debug(f"GET {url}")
        response = self.session.get(url, params=query)
        _check_status(response)
        return response.json()

    def _handle_post(self, url: str, data: dict[str, Any]) -> requests.Response:
        """Submit POST requests to the server.

        Parameters
        ----------
        url : `str`
            URL to POST.
        data : `dict` [`str`, `Any`]
            Key/value pairs of data to POST.

        Raises
        ------
        requests.RequestException
            Raised if any kind of connection error occurs.
        requests.HTTPError
            Raised if a non-successful status is returned.

        Returns
        -------
        result : `requests.Response`
            The raw Web service result object.
        """
        logger.debug(f"POST {url}: {data}")
        response = self.session.post(url, json=data)
        _check_status(response)
        return response

    @staticmethod
    def compute_flexible_metadata_table_name(instrument: str, obs_type: str) -> str:
        """Compute the name of a flexible metadata table.

        Each instrument and observation type made with that instrument can
        have a flexible metadata table.  This function is useful when
        issuing SQL queries, and it avoids a round-trip to the server.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument (e.g. ``LATISS``).
        obs_type : `str`
            Name of the observation type (e.g. ``Exposure``).

        Returns
        -------
        table_name : `str`
            Name of the appropriate flexible metadata table.
        """
        return f"cdb_{instrument}.{obs_type}_flexdata"

    @staticmethod
    def compute_fixed_metadata_namespace(instrument: str) -> str:
        """Compute the namespace for a fixed metadata table.

        Each instrument has its own namespace in the ConsDB.
        This function is useful when issuing SQL queries, and it avoids a
        round-trip to the server.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument (e.g. ``LATISS``).

        Returns
        -------
        namespace_name : `str`
            Name of the appropriate namespace
        """
        return f"cdb_{instrument}"

    def add_flexible_metadata_key(
        self,
        instrument: str,
        obs_type: str,
        key: str,
        dtype: str,
        doc: str,
        unit: str | None = None,
        ucd: str | None = None,
    ) -> requests.Response:
        """Add a key to a flexible metadata table.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument (e.g. ``LATISS``).
        obs_type : `str`
            Name of the observation type (e.g. ``Exposure``).
        key : `str`
            Name of the key to be added (must not already exist).
        dtype : `str`
            One of ``bool``, ``int``, ``float``, or ``str``.
        doc : `str`
            Documentation string for the key.
        unit : `str`, optional
            Unit for the value. Should be from the IVOA
            (https://www.ivoa.net/documents/VOUnits/) or astropy.
        ucd : `str`, optional
            IVOA Unified Content Descriptor
            (https://www.ivoa.net/documents/UCD1+/).

        Returns
        -------
        response : `requests.Response`
            HTTP response from the server, with 200 status for success.

        Raises
        ------
        requests.RequestException
            Raised if any kind of connection error occurs.
        requests.HTTPError
            Raised if a non-successful status is returned.
        """
        data = {"key": key, "dtype": dtype, "doc": doc}
        if unit is not None:
            data["unit"] = unit
        if ucd is not None:
            data["ucd"] = ucd
        url = _urljoin(self.url, "flex", quote(instrument), quote(obs_type), "addkey")
        return self._handle_post(url, data)

    def get_flexible_metadata_keys(self, instrument: str, obs_type: str) -> dict[str, FlexibleMetadataInfo]:
        """Retrieve the valid keys for a flexible metadata table.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument (e.g. ``LATISS``).
        obs_type : `str`
            Name of the observation type (e.g. ``Exposure``).

        Returns
        -------
        key_info : `dict` [ `str`, `FlexibleMetadataInfo` ]
            Dict of keys and information values.

        Raises
        ------
        requests.RequestException
            Raised if any kind of connection error occurs.
        requests.HTTPError
            Raised if a non-successful status is returned.
        """
        url = _urljoin(self.url, "flex", quote(instrument), quote(obs_type), "schema")
        result = self._handle_get(url)
        return {key: FlexibleMetadataInfo(*value) for key, value in result.items()}

    def get_flexible_metadata(
        self, instrument: str, obs_type: str, obs_id: int, keys: list[str] | None = None
    ) -> dict[str, Any]:
        """Get the flexible metadata for an observation.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument (e.g. ``LATISS``).
        obs_type : `str`
            Name of the observation type (e.g. ``Exposure``).
        obs_id : `int`
            Unique observation id.
        keys : `list` [ `str` ], optional
            List of keys to be retrieved; all if not specified.

        Returns
        -------
        result_dict : `dict` [ `str`, `Any` ]
            Dictionary of key/value pairs for the observation.

        Raises
        ------
        requests.RequestException
            Raised if any kind of connection error occurs.
        requests.HTTPError
            Raised if a non-successful status is returned.
        """
        url = _urljoin(
            self.url,
            "flex",
            quote(instrument),
            quote(obs_type),
            "obs",
            quote(str(obs_id)),
        )
        return self._handle_get(url, {"k": keys} if keys else None)

    def get_all_metadata(
        self, instrument: str, obs_type: str, obs_id: int, flex: bool = False
    ) -> dict[str, Any]:
        """Get all metadata for an observation.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument (e.g. ``LATISS``).
        obs_type : `str`
            Name of the observation type (e.g. ``Exposure``).
        obs_id : `int`
            Unique observation id.
        flex : `bool`
            Include flexible metadata.

        Returns
        -------
        result_dict : `dict` [ `str`, `Any` ]
            Dictionary of key/value pairs for the observation.

        Raises
        ------
        requests.RequestException
            Raised if any kind of connection error occurs.
        requests.HTTPError
            Raised if a non-successful status is returned.
        """
        url = _urljoin(
            self.url,
            "query",
            quote(instrument),
            quote(obs_type),
            "obs",
            quote(str(obs_id)),
        )
        return self._handle_get(url, {"flex": "1"} if flex else None)

    def insert_flexible_metadata(
        self,
        instrument: str,
        obs_type: str,
        obs_id: int,
        values: dict[str, Any] | None = None,
        *,
        allow_update: bool = False,
        **kwargs,
    ) -> requests.Response:
        """Set flexible metadata values for an observation.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument (e.g. ``LATISS``).
        obs_type : `str`
            Name of the observation type (e.g. ``Exposure``).
        obs_id : `int`
            Unique observation id.
        values : `dict` [ `str`, `Any` ], optional
            Dictionary of key/value pairs to add for the observation.
        allow_update : `bool`, optional
            If ``True``, allow replacement of values of existing keys.
        **kwargs : `dict`
            Additional key/value pairs, overriding ``values``.

        Returns
        -------
        response : `requests.Response`
            HTTP response from the server, with 200 status for success.

        Raises
        ------
        ValueError
            Raised if no values are provided in ``values`` or kwargs.
        requests.RequestException
            Raised if any kind of connection error occurs.
        requests.HTTPError
            Raised if a non-successful status is returned.
        """
        if values:
            values.update(kwargs)
        else:
            values = kwargs
        if not values:
            raise ValueError(f"No values to set for {instrument} {obs_type} {obs_id}")
        data = {"values": values}
        url = _urljoin(
            self.url,
            "flex",
            quote(instrument),
            quote(obs_type),
            "obs",
            quote(str(obs_id)),
        )
        if allow_update:
            url += "?u=1"
        return self._handle_post(url, data)

    def insert(
        self,
        instrument: str,
        table: str,
        obs_id: int,
        values: dict[str, Any],
        *,
        allow_update=False,
        **kwargs,
    ) -> requests.Response:
        """Insert values into a single ConsDB fixed metadata table.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument (e.g. ``LATISS``).
        table : `str`
            Name of the table to insert into.
        obs_id : `int`
            Unique observation id.
        values : `dict` [ `str`, `Any` ], optional
            Dictionary of column/value pairs to add for the observation.
        allow_update : `bool`, optional
            If ``True``, allow replacement of values of existing columns.
        **kwargs : `dict`
            Additional column/value pairs, overriding ``values``.

        Returns
        -------
        response : `requests.Response`
            HTTP response from the server, with 200 status for success.

        Raises
        ------
        ValueError
            Raised if no values are provided in ``values`` or kwargs.
        requests.RequestException
            Raised if any kind of connection error occurs.
        requests.HTTPError
            Raised if a non-successful status is returned.
        """
        if values:
            values.update(kwargs)
        else:
            values = kwargs
        if not values:
            raise ValueError(f"No values to insert for {instrument} {table} {obs_id}")
        data = {"table": table, "obs_id": obs_id, "values": values}
        url = _urljoin(
            self.url,
            "insert",
            quote(instrument),
            quote(table),
            "obs",
            quote(str(obs_id)),
        )
        if allow_update:
            url += "?u=1"
        return self._handle_post(url, data)

    def insert_multiple(
        self,
        instrument: str,
        table: str,
        obs_dict: dict[int, dict[str, Any]],
        *,
        allow_update=False,
    ) -> requests.Response:
        """Insert values into a single ConsDB fixed metadata table.

        Parameters
        ----------
        instrument : `str`
            Name of the instrument (e.g. ``LATISS``).
        table : `str`
            Name of the table to insert into.
        obs_dict : `dict` [ `int`, `dict` [ `str`, `Any` ] ]
            Dictionary of observation ids, each with a dictionary of
            column/value pairs to add for each observation.
        allow_update : `bool`, optional
            If ``True``, allow replacement of values of existing columns.

        Returns
        -------
        response : `requests.Response`
            HTTP response from the server, with 200 status for success.

        Raises
        ------
        ValueError
            Raised if no values are provided in ``obs_dict``.
        requests.RequestException
            Raised if any kind of connection error occurs.
        requests.HTTPError
            Raised if a non-successful status is returned.
        """
        if not obs_dict:
            raise ValueError(f"No values to insert for {instrument} {table}")
        data = {"table": table, "obs_dict": obs_dict}
        url = _urljoin(
            self.url,
            "insert",
            quote(instrument),
            quote(table),
        )
        if allow_update:
            url += "?u=1"
        return self._handle_post(url, data)

    def query(self, query: str) -> Table:
        """Query the ConsDB database.

        Parameters
        ----------
        query : `str`
            A SQL query (currently) to the database.

        Returns
        -------
        result : `Table`
            An ``astropy.Table`` containing the query results.

        Raises
        ------
        requests.RequestException
            Raised if any kind of connection error occurs.
        requests.HTTPError
            Raised if a non-successful status is returned.

        Notes
        -----
        This is a very general query interface because it is expected that
        a wide variety of types of queries will be needed. If some types prove
        to be common, syntactic sugar could be added to make them simpler.
        """
        url = _urljoin(self.url, "query")
        data = {"query": query}
        result = self._handle_post(url, data).json()
        if "columns" not in result:
            # No result rows
            return Table(rows=[])
        return Table(rows=result["data"], names=result["columns"])

    def schema(
        self, instrument: str | None = None, table: str | None = None
    ) -> dict[str, tuple[str, str]] | list[str]:
        """Retrieve information about ConsDB.

        If ``instrument`` and ``table`` are given, return the schema of a
        fixed metadata table in ConsDB.

        If only ``instrument`` is given, return the names of all tables
        for that instrument.

        If no arguments are given, return the names of all instruments.

        Parameters
        ----------
        instrument : `str`, optional
            Name of the instrument (e.g. ``LATISS``).
        table : `str`, optional
            Name of the table to insert into.

        Returns
        -------
        info : `list` [ `str` ] or `dict` [ `str`, `tuple` [ `str`, `str` ] ]
            A list of instrument strings or table names, or else a dict of
            columns with values that are tuples containing a data type string
            and a documentation string.

        Raises
        ------
        ValueError
            Raised if only ``table`` is given.
        requests.RequestException
            Raised if any kind of connection error occurs.
        requests.HTTPError
            Raised if a non-successful status is returned.

        Notes
        -----
        Fixed metadata data types may use the full database vocabulary,
        unlike flexible metadata data types.
        """
        if instrument is None:
            if table is not None:
                raise ValueError("Must specify instrument if table is given")
            url = _urljoin(self.url, "schema")
        elif table is None:
            url = _urljoin(self.url, "schema", quote(instrument))
        else:
            url = _urljoin(self.url, "schema", quote(instrument), quote(table))
        result = self._handle_get(url)
        if instrument is not None and table is not None:
            return {key: (str(value[0]), str(value[1])) for key, value in result.items()}
        else:
            return [str(value) for value in result]
