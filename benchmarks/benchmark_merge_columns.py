"""Reproducible merge_columns_df benchmark harness.

Run ``--mode auto`` at commit 1b1fa3c for the manual-writer baseline and on
the candidate branch for the native-reader result. Run ``--mode slow`` on the
candidate branch for the keyed-merge comparison. Each invocation prints one
JSON document suitable for attaching to a PR.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import lance
import pyarrow as pa

import daft
from daft_lance.lance_merge_column import _merge_slow_path, merge_columns_from_df
from daft_lance.namespace import DatasetOpenContext


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", "slow"], required=True)
    parser.add_argument("--scenario", choices=["narrow", "vector", "deletion"], required=True)
    parser.add_argument("--fragments", type=int, default=10)
    parser.add_argument("--rows-per-fragment", type=int, default=1_000)
    parser.add_argument("--vector-width", type=int, default=768)
    parser.add_argument("--iterations", type=int, default=3)
    return parser.parse_args()


def _write_dataset(path: str, fragments: int, rows_per_fragment: int) -> lance.LanceDataset:
    for fragment_id in range(fragments):
        start = fragment_id * rows_per_fragment
        lance.write_dataset(
            pa.table({"id": range(start, start + rows_per_fragment)}),
            path,
            mode="create" if fragment_id == 0 else "append",
            data_storage_version="2.2",
        )
    return lance.dataset(path)


def _input_dataframe(path: str, scenario: str, vector_width: int) -> tuple[daft.DataFrame, str]:
    dataframe = daft.read_lance(
        path,
        include_fragment_id=True,
        default_scan_options={"with_row_address": True},
    )
    if scenario == "vector":

        @daft.func.batch(return_dtype=daft.DataType.fixed_size_list(daft.DataType.float32(), vector_width))
        def make_vector(ids):  # type: ignore[no-untyped-def]
            import numpy as np

            return [np.full(vector_width, float(value), dtype=np.float32) for value in ids.to_pylist()]

        return dataframe.with_column("embedding", make_vector(dataframe["id"])), "embedding"
    return dataframe.with_column("score", daft.col("id") * 10), "score"


def _dataset_size(dataset: lance.LanceDataset) -> tuple[int, int]:
    files = [data_file for fragment in dataset.get_fragments() for data_file in fragment.metadata.files]
    return len(files), sum(data_file.file_size_bytes or 0 for data_file in files)


def _run_once(args: argparse.Namespace) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="daft-lance-merge-benchmark-") as tmp_dir:
        path = str(Path(tmp_dir) / "dataset")
        dataset = _write_dataset(path, args.fragments, args.rows_per_fragment)
        if args.scenario == "deletion":
            dataset.delete("id % 10 = 0")
            dataset = lance.dataset(path)

        dataframe, new_column = _input_dataframe(path, args.scenario, args.vector_width)
        open_context = DatasetOpenContext.from_dataset(dataset, path)
        start = time.perf_counter()
        if args.mode == "auto":
            merged = merge_columns_from_df(dataframe, dataset, open_context)
        else:
            merged = _merge_slow_path(
                dataframe,
                dataset,
                open_context,
                read_columns=["_rowaddr", new_column],
                left_on="_rowaddr",
                right_on="_rowaddr",
                reader_schema=None,
                batch_size=None,
            )
        elapsed = time.perf_counter() - start

        column = merged.to_table(columns=[new_column]).column(new_column)
        assert len(column) == merged.count_rows()
        assert column.null_count == 0
        file_count, size_bytes = _dataset_size(merged)
        return {
            "seconds": elapsed,
            "rows": merged.count_rows(),
            "data_files": file_count,
            "size_bytes": size_bytes,
        }


def main() -> None:
    args = _parse_args()
    samples = [_run_once(args) for _ in range(args.iterations)]
    seconds = [sample["seconds"] for sample in samples]
    print(
        json.dumps(
            {
                "mode": args.mode,
                "scenario": args.scenario,
                "fragments": args.fragments,
                "rows_per_fragment": args.rows_per_fragment,
                "vector_width": args.vector_width if args.scenario == "vector" else None,
                "iterations": args.iterations,
                "median_seconds": statistics.median(seconds),
                "min_seconds": min(seconds),
                "max_seconds": max(seconds),
                "samples": samples,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
