"""Opt-in Ray and S3-compatible integration coverage for native reader merges.

Run through ``scripts/run_fast_path_merge_integration.sh``. The regular test
suite skips this module unless ``DAFT_LANCE_RUN_FAST_PATH_INTEGRATION=1``.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import lance
import pyarrow as pa
import pytest

import daft
import daft_lance
from daft.io import IOConfig, S3Config

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("DAFT_LANCE_RUN_FAST_PATH_INTEGRATION") != "1",
        reason="Run scripts/run_fast_path_merge_integration.sh to provision MinIO and Ray.",
    ),
]


def _storage_options() -> dict[str, str]:
    endpoint = os.environ["DAFT_LANCE_TEST_S3_ENDPOINT"]
    return {
        "aws_endpoint": endpoint,
        "aws_access_key_id": os.environ["DAFT_LANCE_TEST_S3_ACCESS_KEY"],
        "aws_secret_access_key": os.environ["DAFT_LANCE_TEST_S3_SECRET_KEY"],
        "aws_region": os.environ.get("DAFT_LANCE_TEST_S3_REGION", "us-east-1"),
        "allow_http": "true",
    }


def _io_config() -> IOConfig:
    return IOConfig(
        s3=S3Config(
            endpoint_url=os.environ["DAFT_LANCE_TEST_S3_ENDPOINT"],
            key_id=os.environ["DAFT_LANCE_TEST_S3_ACCESS_KEY"],
            access_key=os.environ["DAFT_LANCE_TEST_S3_SECRET_KEY"],
            region_name=os.environ.get("DAFT_LANCE_TEST_S3_REGION", "us-east-1"),
            use_ssl=False,
            force_virtual_addressing=False,
        )
    )


@pytest.fixture(params=["native", "ray"])
def daft_runner(request: pytest.FixtureRequest) -> Iterator[str]:
    if request.param == "ray":
        ray = pytest.importorskip("ray")
        if not ray.is_initialized():
            ray.init(num_cpus=4, include_dashboard=False)
        daft.set_runner_ray()
    else:
        ray = None
        daft.set_runner_native(num_threads=4)

    yield request.param

    daft.set_runner_native(num_threads=4)
    if ray is not None and ray.is_initialized():
        ray.shutdown()


def test_native_reader_merge_on_minio_with_deletions(daft_runner: str) -> None:
    storage_options = _storage_options()
    io_config = _io_config()
    root = os.environ["DAFT_LANCE_TEST_S3_URI"].rstrip("/")
    uri = f"{root}/{daft_runner}-{uuid.uuid4().hex}"

    for fragment_id in range(8):
        start = fragment_id * 8
        lance.write_dataset(
            pa.table({"id": list(range(start, start + 8))}),
            uri,
            mode="create" if fragment_id == 0 else "append",
            storage_options=storage_options,
            data_storage_version="2.2",
        )

    dataset = lance.dataset(uri, storage_options=storage_options)
    dataset.delete("id IN (1, 14, 37, 62)")
    dataset = lance.dataset(uri, storage_options=storage_options)
    assert any(fragment.metadata.deletion_file is not None for fragment in dataset.get_fragments())

    source = daft.read_lance(
        uri,
        io_config=io_config,
        include_fragment_id=True,
        default_scan_options={"with_row_address": True},
    ).with_column("score", daft.col("id") * 10)

    daft_lance.merge_columns_df(
        source,
        uri,
        io_config=io_config,
        storage_options=storage_options,
    )

    reopened = lance.dataset(uri, storage_options=storage_options)
    result = reopened.to_table().sort_by("id").to_pydict()
    expected_ids = [value for value in range(64) if value not in {1, 14, 37, 62}]
    assert result["id"] == expected_ids
    assert result["score"] == [value * 10 for value in expected_ids]
    assert any(fragment.metadata.deletion_file is not None for fragment in reopened.get_fragments())
