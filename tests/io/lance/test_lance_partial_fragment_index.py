from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, cast

import lance
import pytest

from daft.dependencies import pa
from daft_lance import create_scalar_index
from daft_lance.lance_scalar_index import SegmentedFragmentIndexHandler

warnings.filterwarnings("ignore", category=DeprecationWarning, module="lance")


def _make_dataset(path: Path, n_rows: int = 80, rows_per_file: int = 20) -> str:
    table = pa.table(
        {
            "id": list(range(n_rows)),
            "name": [f"name-{i % 8}" for i in range(n_rows)],
        }
    )
    lance.write_dataset(table, str(path), mode="create", max_rows_per_file=rows_per_file)
    return str(path)


def _fragment_ids(uri: str) -> list[int]:
    return sorted(f.fragment_id for f in lance.dataset(uri).get_fragments())


def _covered_fragments(uri: str, index_name: str) -> set[int]:
    for desc in lance.dataset(uri).describe_indices():
        if desc.name == index_name:
            covered: set[int] = set()
            for segment in desc.segments:
                covered.update(segment.fragment_ids)
            return covered
    raise AssertionError(f"index {index_name} not found")


def _sorted_rows(uri: str, predicate: str) -> dict[str, list[Any]]:
    table = lance.dataset(uri).scanner(filter=predicate).to_table().sort_by("id")
    return table.to_pydict()


def test_partial_build_then_backfill_matches_full_build(tmp_path: Path) -> None:
    uri = _make_dataset(tmp_path / "incremental.lance")
    full_uri = _make_dataset(tmp_path / "full.lance")

    # Index the first half of the fragments.
    create_scalar_index(uri, column="name", index_type="INVERTED", fragment_ids=[0, 1])
    assert _covered_fragments(uri, "name_idx") == {0, 1}
    # The partial index is already query-correct: uncovered fragments fall
    # back to scans, including predicates that hit only uncovered fragments
    # and zero-hit predicates.
    for predicate in ["name = 'name-3'", "id >= 40", "name = 'nope'"]:
        assert _sorted_rows(uri, predicate) == _sorted_rows(full_uri, predicate), predicate

    # Backfill the remaining fragments under the same index name.
    create_scalar_index(uri, column="name", index_type="INVERTED", fragment_ids=[0, 1, 2, 3])
    assert _covered_fragments(uri, "name_idx") == {0, 1, 2, 3}
    # Backfill appends: the original {0,1} segment is preserved, not rebuilt.
    segments = [sorted(s.fragment_ids) for d in lance.dataset(uri).describe_indices() for s in d.segments]
    assert segments == [[0, 1], [2, 3]]

    # Reference: one-shot full build on an identical dataset.
    create_scalar_index(full_uri, column="name", index_type="INVERTED")

    for predicate in ["name = 'name-3'", "name in ('name-0', 'name-7')", "id >= 40"]:
        assert _sorted_rows(uri, predicate) == _sorted_rows(full_uri, predicate), predicate


def test_partial_build_btree(tmp_path: Path) -> None:
    uri = _make_dataset(tmp_path / "btree.lance")

    create_scalar_index(uri, column="id", index_type="BTREE", fragment_ids=[0, 2])
    assert _covered_fragments(uri, "id_idx") == {0, 2}

    create_scalar_index(uri, column="id", index_type="BTREE", fragment_ids=_fragment_ids(uri))
    assert _covered_fragments(uri, "id_idx") == {0, 1, 2, 3}

    assert lance.dataset(uri).scanner(filter="id = 42").to_table().num_rows == 1


def test_backfill_after_appending_new_fragments(tmp_path: Path) -> None:
    uri = _make_dataset(tmp_path / "append.lance", n_rows=40)
    create_scalar_index(uri, column="name", index_type="INVERTED", fragment_ids=[0, 1])
    covered_before = _covered_fragments(uri, "name_idx")

    # Append data; Lance mints new fragment IDs for it.
    extra = pa.table({"id": list(range(100, 140)), "name": [f"name-{100 + i}" for i in range(40)]})
    lance.write_dataset(extra, uri, mode="append", max_rows_per_file=20)

    new_fragments = set(_fragment_ids(uri)) - covered_before
    assert new_fragments, "append must add fragments"

    create_scalar_index(uri, column="name", index_type="INVERTED", fragment_ids=sorted(new_fragments))
    assert _covered_fragments(uri, "name_idx") == set(_fragment_ids(uri))

    found = lance.dataset(uri).scanner(filter="name = 'name-103'").to_table().num_rows
    assert found == 1


def test_duplicate_fragment_ids_are_deduplicated(tmp_path: Path) -> None:
    uri = _make_dataset(tmp_path / "dupes.lance")
    create_scalar_index(uri, column="name", index_type="INVERTED", fragment_ids=[0, 0, 1, 1, 1])
    assert _covered_fragments(uri, "name_idx") == {0, 1}


def test_unknown_fragment_ids_raise(tmp_path: Path) -> None:
    uri = _make_dataset(tmp_path / "oob.lance")
    with pytest.raises(ValueError, match=r"\[99\].*do not exist"):
        create_scalar_index(uri, column="name", index_type="INVERTED", fragment_ids=[0, 99])


def test_empty_fragment_ids_raise(tmp_path: Path) -> None:
    uri = _make_dataset(tmp_path / "empty.lance")
    with pytest.raises(ValueError, match="non-empty"):
        create_scalar_index(uri, column="name", index_type="INVERTED", fragment_ids=[])


def test_backfill_with_fully_covered_fragments_is_noop(tmp_path: Path) -> None:
    uri = _make_dataset(tmp_path / "noop.lance")
    create_scalar_index(uri, column="name", index_type="INVERTED", fragment_ids=[0, 1])
    version_before = lance.dataset(uri).version

    create_scalar_index(uri, column="name", index_type="INVERTED", fragment_ids=[0, 1])

    assert lance.dataset(uri).version == version_before
    assert _covered_fragments(uri, "name_idx") == {0, 1}


def test_name_reuse_without_fragment_ids_replaces_atomically(tmp_path: Path) -> None:
    """Default replace=True rebuilds the whole index atomically (no error)."""
    uri = _make_dataset(tmp_path / "reuse.lance")
    create_scalar_index(uri, column="name", index_type="INVERTED", fragment_ids=[0])
    version_before = lance.dataset(uri).version

    create_scalar_index(uri, column="name", index_type="INVERTED")

    latest = lance.dataset(uri)
    assert latest.version == version_before + 1
    segments = [sorted(s.fragment_ids) for d in latest.describe_indices() for s in d.segments]
    assert segments == [[0, 1, 2, 3]]  # old partial segment retired by the atomic swap


def test_backfill_different_column_rejected(tmp_path: Path) -> None:
    """A different column under the same name is rejected by Lance's build API.

    daft-lance keeps no duplicated pre-check; the error surfaces from the
    build API itself.
    """
    uri = _make_dataset(tmp_path / "other_column.lance")
    create_scalar_index(uri, column="name", index_type="INVERTED", fragment_ids=[0])
    with pytest.raises(Exception, match="(?i)different fields|already exists"):
        create_scalar_index(uri, column="id", index_type="BTREE", name="name_idx", fragment_ids=[1])


def test_backfill_mixed_index_type_rejected(tmp_path: Path) -> None:
    """Appending segments of a different type is rejected by Lance's commit API.

    daft-lance keeps no duplicated pre-check; the manifest stays intact.
    """
    uri = _make_dataset(tmp_path / "mixed_type.lance")
    create_scalar_index(uri, column="name", index_type="INVERTED", name="shared_idx", fragment_ids=[0])
    with pytest.raises(ValueError, match="cannot change index 'shared_idx' from type"):
        create_scalar_index(uri, column="name", index_type="BITMAP", name="shared_idx", fragment_ids=[2])
    # The manifest must still be describable and the original type intact.
    ds = lance.dataset(uri)
    assert [idx.name for idx in ds.describe_indices()] == ["shared_idx"]


def test_fragment_ids_rejected_for_unsupported_type(tmp_path: Path) -> None:
    uri = _make_dataset(tmp_path / "rtree.lance")
    with pytest.raises(ValueError, match="Unsupported distributed index type 'RTREE'"):
        create_scalar_index(uri, column="id", index_type="RTREE", fragment_ids=[0])
    assert lance.dataset(uri).describe_indices() == []


def test_bitmap_partial_backfill_matches_full_build(tmp_path: Path) -> None:
    uri = _make_dataset(tmp_path / "bitmap_partial.lance")
    full_uri = _make_dataset(tmp_path / "bitmap_full.lance")

    create_scalar_index(uri, column="id", index_type="BITMAP", fragment_ids=[0, 1])
    create_scalar_index(uri, column="id", index_type="BITMAP", fragment_ids=_fragment_ids(uri))
    create_scalar_index(full_uri, column="id", index_type="BITMAP")

    assert _covered_fragments(uri, "id_idx") == {0, 1, 2, 3}
    for predicate in ["id = 42", "id in (0, 39, 79)", "id >= 60"]:
        assert _sorted_rows(uri, predicate) == _sorted_rows(full_uri, predicate), predicate


def test_btree_partial_backfill_matches_full_build(tmp_path: Path) -> None:
    uri = _make_dataset(tmp_path / "btree_equiv.lance")
    full_uri = _make_dataset(tmp_path / "btree_full.lance")

    create_scalar_index(uri, column="id", index_type="BTREE", fragment_ids=[0, 1])
    create_scalar_index(uri, column="id", index_type="BTREE", fragment_ids=_fragment_ids(uri))
    create_scalar_index(full_uri, column="id", index_type="BTREE")

    for predicate in ["id = 7", "id in (20, 59)", "id >= 40"]:
        assert _sorted_rows(uri, predicate) == _sorted_rows(full_uri, predicate), predicate


def test_compacted_fragment_conflict_surfaces_as_error(tmp_path: Path) -> None:
    """The coordinator must refuse to commit segments whose fragments died.

    If a compaction rewrites the fragments while workers are building, Lance
    would still accept the commit and the index would reference dead fragment
    IDs forever.
    """
    import pickle

    from daft_lance.lance_scalar_index import _validate_segments_against_manifest

    uri = _make_dataset(tmp_path / "conflict.lance")
    ds = lance.dataset(uri)
    open_context = type(
        "Ctx",
        (),
        {
            "uri": uri,
            "open_pinned": lambda self: ds,
            "open_latest": lambda self: lance.dataset(uri),
        },
    )()

    handler = SegmentedFragmentIndexHandler(
        open_context=open_context,
        column="name",
        index_type="INVERTED",
        name="name_idx",
    )
    segment = pickle.loads(handler([0, 1]))

    # Compact away the fragments the segment was built against.
    lance.dataset(uri).optimize.compact_files()

    latest = lance.dataset(uri)
    with pytest.raises(RuntimeError, match="no longer exist"):
        _validate_segments_against_manifest(latest, [segment])


def test_label_list_backfill_and_rebuild_roundtrip(tmp_path: Path) -> None:
    """Regression: pylance reports LABEL_LIST as 'LabelList'.

    The type guard must normalize separators so backfill and plain rebuild
    both work.
    """
    uri = str(tmp_path / "labels.lance")
    lance.write_dataset(
        pa.table({"tags": [[f"t{i % 4}", "common"] for i in range(80)]}),
        uri,
        mode="create",
        max_rows_per_file=20,
    )

    create_scalar_index(uri, column="tags", index_type="LABEL_LIST", fragment_ids=[0, 1])
    create_scalar_index(uri, column="tags", index_type="LABEL_LIST", fragment_ids=[2, 3])
    assert _covered_fragments(uri, "tags_idx") == {0, 1, 2, 3}

    # Plain rebuild (default replace=True) also still works.
    create_scalar_index(uri, column="tags", index_type="LABEL_LIST")
    assert _covered_fragments(uri, "tags_idx") == {0, 1, 2, 3}


def test_full_rebuild_may_change_type(tmp_path: Path) -> None:
    """A full rebuild may change the index type via the atomic swap.

    A different column under the same name is rejected by Lance's build API.
    """
    uri = _make_dataset(tmp_path / "swap.lance")
    create_scalar_index(uri, column="name", index_type="INVERTED", name="shared_idx", fragment_ids=[0])

    create_scalar_index(uri, column="name", index_type="BTREE", name="shared_idx")

    described = lance.dataset(uri).describe_indices()
    assert len(described) == 1
    assert described[0].index_type == "BTree"
    assert lance.dataset(uri).scanner(filter="name = 'name-3'").to_table().num_rows == 10

    with pytest.raises(Exception, match="(?i)different fields|already exists"):
        create_scalar_index(uri, column="id", index_type="BTREE", name="shared_idx")


def test_replace_false_with_fragment_ids_still_backfills(tmp_path: Path) -> None:
    """replace=False does not apply to backfill.

    Backfill appends and never replaces, so the flag is irrelevant on that
    path (documented interaction).
    """
    uri = _make_dataset(tmp_path / "rf.lance")
    create_scalar_index(uri, column="name", index_type="INVERTED", fragment_ids=[0, 1])
    version_before = lance.dataset(uri).version

    create_scalar_index(uri, column="name", index_type="INVERTED", fragment_ids=[2, 3], replace=False)

    latest = lance.dataset(uri)
    assert latest.version == version_before + 1
    segments = [sorted(s.fragment_ids) for d in latest.describe_indices() for s in d.segments]
    assert segments == [[0, 1], [2, 3]]


def test_backfill_with_multiple_segments_per_task(tmp_path: Path) -> None:
    """fragment_group_size=1 splits the backfill across workers.

    Multiple new segments are appended alongside the preserved old one in a
    single commit.
    """
    uri = _make_dataset(tmp_path / "multi.lance")
    create_scalar_index(uri, column="name", index_type="INVERTED", fragment_ids=[0, 1])

    create_scalar_index(uri, column="name", index_type="INVERTED", fragment_ids=[2, 3], fragment_group_size=1)

    segments = [sorted(s.fragment_ids) for d in lance.dataset(uri).describe_indices() for s in d.segments]
    assert segments == [[0, 1], [2], [3]]
    assert lance.dataset(uri).scanner(filter="name = 'name-3'").to_table().num_rows == 10


def test_duplicate_fragment_ids_warn(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    """The de-duplication is user-visible via a log warning."""
    import logging

    uri = _make_dataset(tmp_path / "dupwarn.lance")
    with caplog.at_level(logging.WARNING, logger="daft_lance.lance_scalar_index"):
        create_scalar_index(uri, column="name", index_type="INVERTED", fragment_ids=[0, 0, 1])
    assert any("Duplicate fragment_ids" in r.message for r in caplog.records)
    assert _covered_fragments(uri, "name_idx") == {0, 1}


def _guard_recorder(order: list[str], real_guard: Any, ds: Any, metas: Any, expected: Any) -> Any:
    order.append("guard")
    return real_guard(ds, metas, expected)


def _commit_recorder(order: list[str], real_commit: Any, ds: Any, *args: Any, **kwargs: Any) -> Any:
    order.append("commit")
    return real_commit(ds, *args, **kwargs)


def test_guard_runs_before_commit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The manifest guard is on the commit path (before the commit call)."""
    import daft_lance.lance_scalar_index as lsi

    order: list[str] = []
    real_guard = lsi._validate_segments_against_manifest
    monkeypatch.setattr(
        lsi,
        "_validate_segments_against_manifest",
        lambda ds, metas, expected=None: _guard_recorder(order, real_guard, ds, metas, expected),
    )
    real_commit = lance.LanceDataset.commit_existing_index_segments
    monkeypatch.setattr(
        lance.LanceDataset,
        "commit_existing_index_segments",
        lambda self, *a, **k: _commit_recorder(order, real_commit, self, *a, **k),
    )

    uri = _make_dataset(tmp_path / "order.lance")
    create_scalar_index(uri, column="name", index_type="INVERTED", fragment_ids=[0])

    assert order == ["guard", "commit"]


def test_guard_rejects_overlapping_and_incomplete_coverage() -> None:
    """The commit guard catches duplicate coverage and missing fragments."""
    from daft_lance.lance_scalar_index import _validate_segments_against_manifest

    class FakeSegment:
        def __init__(self, fragment_ids: set[int]) -> None:
            self.fragment_ids = fragment_ids

    class FakeFragments:
        def __init__(self, ids: list[int]) -> None:
            self._ids = ids

        def get_fragments(self) -> list[FakeFragment]:
            return [FakeFragment(i) for i in self._ids]

    class FakeFragment:
        def __init__(self, fragment_id: int) -> None:
            self.fragment_id = fragment_id

    ds = FakeFragments([0, 1, 2, 3])

    # Overlapping coverage: fragment 1 built by two workers.
    with pytest.raises(RuntimeError, match="more than one segment"):
        _validate_segments_against_manifest(
            cast(Any, ds), [cast(Any, FakeSegment({0, 1})), cast(Any, FakeSegment({1, 2}))], [0, 1, 2]
        )

    # Incomplete coverage: fragment 3 scheduled but no segment covers it.
    with pytest.raises(RuntimeError, match="not covered by any built segment"):
        _validate_segments_against_manifest(cast(Any, ds), [cast(Any, FakeSegment({0, 1}))], [0, 1, 2, 3])

    # Exact coverage passes.
    _validate_segments_against_manifest(
        cast(Any, ds), [cast(Any, FakeSegment({0, 1})), cast(Any, FakeSegment({2, 3}))], [0, 1, 2, 3]
    )
