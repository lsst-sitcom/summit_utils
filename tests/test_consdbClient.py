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

import pytest
import responses
from requests import HTTPError

from lsst.summit.utils import ConsDbClient, FlexibleMetadataInfo


@pytest.fixture
def client():
    return ConsDbClient("http://example.com/consdb")


def test_table_name():
    instrument = "latiss"
    obs_type = "exposure"
    assert (
        ConsDbClient.compute_flexible_metadata_table_name(instrument, obs_type)
        == "cdb_latiss.exposure_flexdata"
    )


def test_exposure_id():
    assert ConsDbClient.compute_exposure_id("latiss", "O", 20240402, 35) == 2024040200035
    assert ConsDbClient.compute_exposure_id("LATISS", "C", 20240402, 35) == 2024040200035
    assert ConsDbClient.compute_exposure_id("LSSTComCamSim", "S", 20240402, 35) == 7024040200035
    with pytest.raises(ValueError):
        ConsDbClient.compute_exposure_id("bad_instrument", "O", 20240402, 35)


def test_ccdexposure_id():
    assert ConsDbClient.compute_ccdexposure_id("latiss", 2024040200035, 0) == 5205 * (2**23) + 35 * 256 + 0
    with pytest.raises(ValueError):
        ConsDbClient.compute_ccdexposure_id("latiss", 20240402000035, 1)
    with pytest.raises(ValueError):
        ConsDbClient.compute_ccdexposure_id("LsstComCam", 20240402000035, 9)
    with pytest.raises(ValueError):
        ConsDbClient.compute_ccdexposure_id("LsstCam", 20240402000035, 205)
    assert (
        ConsDbClient.compute_ccdexposure_id("LSSTComCamSim", 7024040200035, 2)
        == 5 * (2**37) + 5205 * (2**23) + 35 * 256 + 2
    )


@responses.activate
def test_add_flexible_metadata_key(client):
    instrument = "latiss"
    obs_type = "exposure"
    responses.post(
        "http://example.com/consdb/flex/latiss/exposure/addkey",
        json={
            "message": "Key added to flexible metadata",
            "key": "foo",
            "instrument": "latiss",
            "obs_type": "exposure",
        },
        match=[
            responses.matchers.json_params_matcher({"key": "foo", "dtype": "bool", "doc": "bool key"}),
        ],
    )
    responses.post(
        "http://example.com/consdb/flex/latiss/exposure/addkey",
        json={
            "message": "Key added to flexible metadata",
            "key": "bar",
            "instrument": "latiss",
            "obs_type": "exposure",
        },
        match=[
            responses.matchers.json_params_matcher({"key": "bar", "dtype": "int", "doc": "int key"}),
        ],
    )
    responses.post(
        "http://example.com/consdb/flex/bad_instrument/exposure/addkey",
        status=404,
        json={"message": "Unknown instrument", "value": "bad_instrument", "valid": ["latiss"]},
    )
    responses.post(
        "http://example.com/consdb/flex/latiss/bad_obs_type/addkey",
        status=404,
        json={"message": "Unknown observation type", "value": "bad_obs_type", "valid": ["exposure"]},
    )

    assert (
        client.add_flexible_metadata_key(instrument, obs_type, "foo", "bool", "bool key").json()["key"]
        == "foo"
    )
    assert (
        client.add_flexible_metadata_key(instrument, obs_type, "bar", "int", "int key").json()["instrument"]
        == "latiss"
    )
    with pytest.raises(HTTPError, match="404") as e:
        client.add_flexible_metadata_key("bad_instrument", obs_type, "error", "int", "instrument error")
    json_data = e.value.response.json()
    assert json_data["message"] == "Unknown instrument"
    assert json_data["value"] == "bad_instrument"
    assert json_data["valid"] == ["latiss"]
    with pytest.raises(HTTPError, match="404"):
        client.add_flexible_metadata_key(instrument, "bad_obs_type", "error", "int", "obs_type error")


@responses.activate
def test_get_flexible_metadata_keys(client):
    description = {"foo": ["bool", "a", None, None], "bar": ["float", "b", "deg", "pos.eq.ra"]}
    responses.get(
        "http://example.com/consdb/flex/latiss/exposure/schema",
        json=description,
    )
    instrument = "latiss"
    obs_type = "exposure"
    assert client.get_flexible_metadata_keys(instrument, obs_type) == {
        "foo": FlexibleMetadataInfo("bool", "a"),
        "bar": FlexibleMetadataInfo("float", "b", "deg", "pos.eq.ra"),
    }


@responses.activate
def test_get_flexible_metadata(client):
    results = {"bool_key": True, "int_key": 42, "float_key": 3.14159, "str_key": "foo"}
    responses.get(
        "http://example.com/consdb/flex/latiss/exposure/obs/271828",
        json=results,
    )
    responses.get(
        "http://example.com/consdb/flex/latiss/exposure/obs/271828?k=float_key", json={"float_key": 3.14159}
    )
    responses.get(
        "http://example.com/consdb/flex/latiss/exposure/obs/271828?k=int_key&k=float_key",
        json={"float_key": 3.14159, "int_key": 42},
    )
    instrument = "latiss"
    obs_type = "exposure"
    obs_id = 271828
    assert client.get_flexible_metadata(instrument, obs_type, obs_id) == results
    assert client.get_flexible_metadata(instrument, obs_type, obs_id, ["float_key"]) == {
        "float_key": results["float_key"]
    }
    assert client.get_flexible_metadata(instrument, obs_type, obs_id, ["int_key", "float_key"]) == {
        "int_key": results["int_key"],
        "float_key": results["float_key"],
    }


@responses.activate
def test_insert_flexible_metadata(client):
    instrument = "latiss"
    obs_type = "exposure"
    with pytest.raises(ValueError):
        client.insert_flexible_metadata(instrument, obs_type, 271828)
    # TODO: more POST tests


@responses.activate
def test_schema(client):
    description = {"foo": ("bool", "a"), "bar": ("int", "b")}
    responses.get(
        "http://example.com/consdb/schema/latiss/misc_table",
        json=description,
    )
    instrument = "latiss"
    table = "misc_table"
    assert client.schema(instrument, table) == description


# TODO: more POST tests
#    client.insert(instrument, table, obs_id, values, allow_update)
#    client.insert_multiple(instrument, table, obs_dict, allow_update)
#    client.query(query)
