"""Conditional overwrite: ``mode="overwrite_where"`` replaces a predicate's rows in one commit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lance
import pyarrow as pa
import pytest

import daft
import daft_lance
from daft.recordbatch import MicroPartition
from daft_lance.lance_data_sink import LanceDataSink, _compile_predicate
from daft_lance.lance_overwrite_where import _SCALAR_INDEX_PLAN_MARKER, _candidate_fragment_ids


def _seed(uri: str) -> None:
    """Three fragments; the first and second each mix two ``dt`` values."""
    daft_lance.write_lance(
        daft.from_pydict({"dt": ["d1", "d2"], "id": [1, 2]}), uri, mode="create", max_rows_per_file=2
    ).collect()
    daft_lance.write_lance(
        daft.from_pydict({"dt": ["d2", "d3"], "id": [3, 4]}), uri, mode="append", max_rows_per_file=2
    ).collect()
    daft_lance.write_lance(
        daft.from_pydict({"dt": ["d2", "d2"], "id": [5, 6]}), uri, mode="append", max_rows_per_file=2
    ).collect()


def _rows(uri: str) -> list[tuple[str, int]]:
    table = lance.dataset(uri).to_table().to_pydict()
    return sorted(zip(table["dt"], table["id"]))


def _overwrite(uri: str, dts: list[str | None], ids: list[int], predicate: str, **kwargs: Any) -> dict[str, list[Any]]:
    return daft_lance.write_lance(
        daft.from_pydict({"dt": dts, "id": ids}), uri, mode="overwrite_where", predicate=predicate, **kwargs
    ).to_pydict()


def test_replaces_only_matching_rows_in_one_version(tmp_path: Path) -> None:
    uri = str(tmp_path / "tbl")
    _seed(uri)
    before = lance.dataset(uri).version

    stats = _overwrite(uri, ["d2", "d2"], [100, 101], "dt = 'd2'")

    assert _rows(uri) == [("d1", 1), ("d2", 100), ("d2", 101), ("d3", 4)]
    # One commit, not a delete followed by an append: readers never see the gap.
    assert lance.dataset(uri).version == before + 1
    assert stats["version"] == [before + 1]


def test_rerunning_the_same_overwrite_is_idempotent(tmp_path: Path) -> None:
    uri = str(tmp_path / "tbl")
    _seed(uri)

    _overwrite(uri, ["d2", "d2"], [100, 101], "dt = 'd2'")
    first = _rows(uri)
    _overwrite(uri, ["d2", "d2"], [100, 101], "dt = 'd2'")

    assert _rows(uri) == first


def test_predicate_matching_nothing_only_appends(tmp_path: Path) -> None:
    uri = str(tmp_path / "tbl")
    _seed(uri)
    before = lance.dataset(uri).version

    _overwrite(uri, ["d9"], [42], "dt = 'd9'")

    assert _rows(uri) == [("d1", 1), ("d2", 2), ("d2", 3), ("d2", 5), ("d2", 6), ("d3", 4), ("d9", 42)]
    assert lance.dataset(uri).version == before + 1


def test_empty_input_deletes_the_matched_rows(tmp_path: Path) -> None:
    uri = str(tmp_path / "tbl")
    _seed(uri)
    before = lance.dataset(uri).version
    empty = daft.from_pydict({"dt": ["d2"], "id": [1]}).limit(0)

    daft_lance.write_lance(empty, uri, mode="overwrite_where", predicate="dt = 'd2'").collect()

    assert _rows(uri) == [("d1", 1), ("d3", 4)]
    assert lance.dataset(uri).version == before + 1


def test_fully_matched_fragment_is_removed_not_just_emptied(tmp_path: Path) -> None:
    uri = str(tmp_path / "tbl")
    _seed(uri)
    # The third seeded fragment is all "d2", so the overwrite drops it entirely
    # while the mixed fragments only gain deletion files.
    _overwrite(uri, ["d2"], [100], "dt = 'd2'")

    fragments = lance.dataset(uri).get_fragments()
    assert sum(fragment.count_rows() for fragment in fragments) == 3
    assert all(fragment.count_rows() > 0 for fragment in fragments)


def test_rows_outside_the_predicate_are_rejected(tmp_path: Path) -> None:
    uri = str(tmp_path / "tbl")
    _seed(uri)
    before = _rows(uri)

    with pytest.raises(Exception, match="do not satisfy"):
        _overwrite(uri, ["d2", "d9"], [100, 101], "dt = 'd2'")

    assert _rows(uri) == before


def test_null_rows_count_as_not_satisfying_the_predicate(tmp_path: Path) -> None:
    uri = str(tmp_path / "tbl")
    _seed(uri)

    # ``dt = 'd2'`` is NULL, not False, for a NULL dt: a naive "keep rows where
    # NOT predicate" check would let this row through.
    with pytest.raises(Exception, match="do not satisfy"):
        _overwrite(uri, ["d2", None], [100, 101], "dt = 'd2'")


def test_validate_predicate_false_appends_rows_outside_the_predicate(tmp_path: Path) -> None:
    uri = str(tmp_path / "tbl")
    _seed(uri)

    _overwrite(uri, ["d2", "d9"], [100, 101], "dt = 'd2'", validate_predicate=False)

    assert _rows(uri) == [("d1", 1), ("d2", 100), ("d3", 4), ("d9", 101)]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mode": "overwrite_where"}, "requires a non-empty SQL predicate"),
        ({"mode": "overwrite_where", "predicate": "   "}, "requires a non-empty SQL predicate"),
        ({"mode": "append", "predicate": "dt = 'd2'"}, 'only supported with mode="overwrite_where"'),
        ({"mode": "overwrite_where", "predicate": "dt = 'd2'", "use_mem_wal": True}, "not supported with use_mem_wal"),
    ],
)
def test_argument_validation(tmp_path: Path, kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        daft_lance.write_lance(daft.from_pydict({"dt": ["d2"], "id": [1]}), str(tmp_path / "tbl"), **kwargs)


def test_requires_an_existing_table(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Cannot overwrite_where to non-existent Lance dataset"):
        _overwrite(str(tmp_path / "missing"), ["d2"], [1], "dt = 'd2'")


def test_schema_must_match_like_append(tmp_path: Path) -> None:
    uri = str(tmp_path / "tbl")
    _seed(uri)

    with pytest.raises(ValueError, match="Schema of data does not match table schema"):
        daft_lance.write_lance(
            daft.from_pydict({"dt": ["d2"]}), uri, mode="overwrite_where", predicate="dt = 'd2'"
        ).collect()


def test_storage_version_conflict_is_detected_like_append(tmp_path: Path) -> None:
    """Regression guard for the mode normalization.

    ``resolve_storage_version`` only checks the "append" mode; before
    ``overwrite_where`` was normalized to it, a conflicting version was accepted
    silently.
    """
    uri = str(tmp_path / "tbl")
    daft_lance.write_lance(
        daft.from_pydict({"dt": ["d2"], "id": [1]}), uri, mode="create", data_storage_version="2.1"
    ).collect()

    with pytest.raises(ValueError, match="does not match existing dataset version"):
        _overwrite(uri, ["d2"], [2], "dt = 'd2'", data_storage_version="2.0")


@pytest.mark.parametrize("predicate", ["nosuchcol = 1", "dt ==== 'x'"])
def test_predicate_must_be_a_valid_lance_filter(tmp_path: Path, predicate: str) -> None:
    """Bad predicates fail on the driver, before any data is written."""
    uri = str(tmp_path / "tbl")
    _seed(uri)

    with pytest.raises(ValueError, match="is not a valid Lance filter"):
        _overwrite(uri, ["d2"], [100], predicate)


def test_unparseable_predicate_points_at_the_escape_hatch() -> None:
    with pytest.raises(ValueError, match="validate_predicate=False"):
        _compile_predicate("dt ==== 'x'")


def test_pruning_uses_a_scalar_index_when_one_covers_the_predicate(tmp_path: Path) -> None:
    uri = str(tmp_path / "tbl")
    _seed(uri)
    dataset = lance.dataset(uri)

    # No index: every fragment has to be visited, and the planner says so.
    assert _candidate_fragment_ids(dataset, "id = 4") is None

    dataset.create_scalar_index("id", "BTREE")
    dataset = lance.dataset(uri)

    # Pinned plan-node name: pruning silently stops working if Lance renames it.
    plan = dataset.scanner(columns=[], filter="id = 4", with_row_address=True).explain_plan(True)
    assert _SCALAR_INDEX_PLAN_MARKER in plan

    # id 4 lives only in the second seeded fragment.
    assert _candidate_fragment_ids(dataset, "id = 4") == {1}
    assert _candidate_fragment_ids(dataset, "id > 2") == {1, 2}
    assert _candidate_fragment_ids(dataset, "id = 999") == set()


def test_indexed_table_stays_queryable_after_overwrite(tmp_path: Path) -> None:
    uri = str(tmp_path / "tbl")
    n = 300
    vector_type = pa.list_(pa.float32(), 2)
    seed = pa.table(
        {
            "id": pa.array(range(n), pa.int64()),
            "dt": pa.array(["d1" if i % 2 else "d2" for i in range(n)]),
            "vector": pa.array([[float(i % 3), 0.0] for i in range(n)], type=vector_type),
        }
    )
    lance.write_dataset(seed, uri, max_rows_per_file=100)
    dataset = lance.dataset(uri)
    dataset.create_scalar_index("id", "BTREE")
    try:
        dataset.create_index("vector", "IVF_PQ", num_partitions=2, num_sub_vectors=1)
    except Exception:
        pytest.skip("Could not create vector index (lance version or dataset size issue)")

    new_rows = pa.table(
        {
            "id": pa.array([1000, 1001], pa.int64()),
            "dt": pa.array(["d2", "d2"]),
            "vector": pa.array([[7.0, 7.0], [8.0, 8.0]], type=vector_type),
        }
    )
    daft_lance.write_lance(daft.from_arrow(new_rows), uri, mode="overwrite_where", predicate="dt = 'd2'").collect()

    dataset = lance.dataset(uri)
    # Deleted rows are invisible through the scalar index that still covers them.
    assert dataset.to_table(filter="id = 0").num_rows == 0
    assert dataset.to_table(filter="id = 1").num_rows == 1
    assert sorted(dataset.to_table(filter="dt = 'd2'").to_pydict()["id"]) == [1000, 1001]

    # New fragments are not in the index; the search must still find them.
    nearest = {"column": "vector", "q": pa.array([8.0, 8.0], type=pa.float32()), "k": 1, "use_index": True}
    assert daft.read_lance(uri, default_scan_options={"nearest": nearest}).select("id").to_pydict()["id"] == [1001]


def test_namespace_addressed_table(tmp_path: Path) -> None:
    ns: dict[str, Any] = {"namespace_impl": "dir", "namespace_properties": {"root": str(tmp_path)}}
    table_id = ["events"]

    daft_lance.write_lance(
        daft.from_pydict({"dt": ["d1", "d2"], "id": [1, 2]}), table_id=table_id, mode="create", **ns
    ).collect()
    daft_lance.write_lance(
        daft.from_pydict({"dt": ["d2"], "id": [100]}),
        table_id=table_id,
        mode="overwrite_where",
        predicate="dt = 'd2'",
        **ns,
    ).collect()

    result = daft_lance.read_lance(table_id=table_id, **ns).to_pydict()
    assert sorted(zip(result["dt"], result["id"])) == [("d1", 1), ("d2", 100)]


def test_concurrent_append_survives_the_overwrite(tmp_path: Path) -> None:
    """Documents a known gap, so a future Lance change surfaces here.

    Lance does not treat a concurrent append as conflicting with the Update this
    mode commits, so rows another writer adds while the overwrite runs stay in
    the table even when they match the predicate -- and nothing raises. The
    docstring on ``write_lance`` warns about it; this test pins the behavior.
    """
    uri = str(tmp_path / "tbl")
    _seed(uri)

    sink = LanceDataSink(
        uri=uri,
        schema=daft.from_pydict({"dt": ["d2"], "id": [1]}).schema(),
        mode="overwrite_where",
        predicate="dt = 'd2'",
    )
    sink.start()  # pins the read version

    concurrent = pa.table({"dt": pa.array(["d2"], pa.large_string()), "id": pa.array([900], pa.int64())})
    lance.write_dataset(concurrent, uri, mode="append")

    results = list(sink.write(iter([MicroPartition.from_pydict({"dt": ["d2"], "id": [100]})])))
    sink.finalize(results)

    assert ("d2", 900) in _rows(uri)
