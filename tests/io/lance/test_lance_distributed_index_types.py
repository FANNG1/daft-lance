from __future__ import annotations

import warnings
from pathlib import Path

import lance
import pytest

from daft.dependencies import pa
from daft_lance import create_scalar_index

warnings.filterwarnings("ignore", category=DeprecationWarning, module="lance")


def _make_dataset(path: Path) -> str:
    table = pa.table(
        {
            "id": list(range(80)),
            "doc": [f"document number {i % 8}" for i in range(80)],
            "tags": [[f"tag-{i % 4}", "common"] for i in range(80)],
        }
    )
    lance.write_dataset(table, str(path), mode="create", max_rows_per_file=20)
    return str(path)


@pytest.mark.parametrize(
    ("index_type", "column", "expected_type", "predicate", "expected_rows"),
    [
        ("ZONEMAP", "id", "ZoneMap", "id >= 40", 40),
        ("NGRAM", "doc", "NGram", "contains(doc, 'number 3')", 10),
        ("LABEL_LIST", "tags", "LabelList", "array_contains(tags, 'tag-1')", 20),
        ("BLOOMFILTER", "id", "BloomFilter", "id = 42", 1),
    ],
)
def test_new_type_builds_distributed_with_full_metadata(
    tmp_path: Path,
    index_type: str,
    column: str,
    expected_type: str,
    predicate: str,
    expected_rows: int,
) -> None:
    """ZONEMAP/NGRAM/LABEL_LIST/BLOOMFILTER now use the distributed segment workflow."""
    uri = _make_dataset(tmp_path / f"{index_type.lower()}.lance")

    # fragment_group_size=1 forces one segment per worker build.
    create_scalar_index(uri, column=column, index_type=index_type, fragment_group_size=1)

    described = lance.dataset(uri).describe_indices()
    assert len(described) == 1
    desc = described[0]
    assert desc.name == f"{column}_{index_type.lower()}_idx"
    # The whole point of the segment workflow: metadata must be complete, not "Unknown".
    assert desc.index_type == expected_type
    assert desc.type_url != ""
    assert desc.num_rows_indexed == 80
    assert len(desc.segments) >= 1

    results = lance.dataset(uri).scanner(filter=predicate).to_table()
    assert results.num_rows == expected_rows


def test_new_types_replace_rebuilds_in_place(tmp_path: Path) -> None:
    uri = _make_dataset(tmp_path / "replace.lance")
    create_scalar_index(uri, column="id", index_type="ZONEMAP", name="z_idx")
    create_scalar_index(uri, column="id", index_type="ZONEMAP", name="z_idx")

    described = lance.dataset(uri).describe_indices()
    assert len(described) == 1
    assert described[0].name == "z_idx"
    assert lance.dataset(uri).scanner(filter="id < 10").to_table().num_rows == 10


def test_new_types_reject_existing_name_without_replace(tmp_path: Path) -> None:
    uri = _make_dataset(tmp_path / "noreplace.lance")
    create_scalar_index(uri, column="id", index_type="ZONEMAP", name="z_idx")
    with pytest.raises(ValueError, match="already exists. Set replace=True"):
        create_scalar_index(uri, column="id", index_type="ZONEMAP", name="z_idx", replace=False)


def test_all_supported_types_match_pylance_segment_native_set(tmp_path: Path) -> None:
    """DISTRIBUTED_INDEX_TYPES must stay in sync with what Lance can build as segments."""
    from daft_lance.lance_scalar_index import DISTRIBUTED_INDEX_TYPES

    ds = lance.dataset(_make_dataset(tmp_path / "probe.lance"))
    segment_native = {
        t
        for t in ("BTREE", "BITMAP", "INVERTED", "NGRAM", "ZONEMAP", "LABEL_LIST", "BLOOMFILTER", "RTREE")
        if ds._is_segment_native_scalar_index_type(t)
    }
    # Everything we route must be segment-native; RTREE stays excluded until
    # GeoArrow columns are testable.
    assert DISTRIBUTED_INDEX_TYPES <= segment_native
    assert "RTREE" not in DISTRIBUTED_INDEX_TYPES
