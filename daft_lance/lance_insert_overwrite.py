from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import lance
import pyarrow as pa
import pyarrow.compute as pc

import daft.pickle
from daft import from_pylist
from daft.datatype import DataType
from daft.runners import get_or_create_runner
from daft.udf import cls as daft_cls
from daft.udf import method

if TYPE_CHECKING:
    from lance.fragment import FragmentMetadata

    from daft_lance.namespace import DatasetOpenContext

logger = logging.getLogger(__name__)

_FRAGMENT_DELETE_RETURN_DTYPE = DataType.struct(
    {
        "fragment_id": DataType.int64(),
        "fragment_meta": DataType.binary(),
        "removed": DataType.bool(),
    }
)

# Pinned to the plan node Lance emits when a scalar index answers the filter. A
# rename would silently cost us fragment pruning, so
# test_pruning_uses_a_scalar_index_when_one_covers_the_predicate asserts on it.
_SCALAR_INDEX_PLAN_MARKER = "ScalarIndexQuery"

# A Lance row address packs the fragment id into its high 32 bits.
_FRAGMENT_ID_SHIFT = pa.scalar(32, type=pa.uint64())

# Each partition builds its own handler and reopens the pinned snapshot once, so
# this bounds the manifest reads a wide table pays for the extra parallelism.
_MAX_DELETE_PARTITIONS = 64


@daft_cls
class FragmentDeleteHandler:
    """Applies one delete predicate to a fragment and reports what changed.

    Runs as a Daft UDF: the driver ships fragment ids, each task reopens the
    pinned snapshot and writes a deletion file for the rows the predicate
    matches. Data files are never rewritten, so row addresses -- and every index
    built on them -- stay valid.
    """

    def __init__(self, open_context: DatasetOpenContext, predicate: str) -> None:
        self.open_context = open_context
        self.predicate = predicate
        self._lance_ds: lance.LanceDataset | None = None

    def _dataset(self) -> lance.LanceDataset:
        # Opened once per instance, not per fragment: the reopen costs a pinned
        # manifest read and must not sit on the per-row path.
        if self._lance_ds is None:
            self._lance_ds = self.open_context.open_pinned()
        return self._lance_ds

    @method.batch(return_dtype=_FRAGMENT_DELETE_RETURN_DTYPE)
    def __call__(self, fragment_ids: Any) -> list[dict[str, Any]]:
        lance_ds = self._dataset()
        results: list[dict[str, Any]] = []
        for fragment_id in fragment_ids:
            fragment = lance_ds.get_fragment(fragment_id)
            if fragment is None:
                raise ValueError(f"Fragment {fragment_id} not found in dataset")
            deletions_before = fragment.metadata.num_deletions
            updated = fragment.delete(self.predicate)
            if updated is None:
                # Every row matched: the fragment leaves the dataset entirely.
                results.append({"fragment_id": int(fragment_id), "fragment_meta": None, "removed": True})
                continue
            # A fragment the predicate missed comes back unchanged; committing it
            # as "updated" would only add noise to the transaction.
            changed = updated.num_deletions != deletions_before
            results.append(
                {
                    "fragment_id": int(fragment_id),
                    "fragment_meta": daft.pickle.dumps(updated) if changed else None,
                    "removed": False,
                }
            )
        return results


def _candidate_fragment_ids(dataset: lance.LanceDataset, predicate: str) -> set[int] | None:
    """Fragment ids that hold rows matching ``predicate``, or None when unknown.

    Only worth doing when a scalar index can answer the filter: then this is an
    index lookup that skips most fragments. Without an index the scan costs the
    same full pass the delete step already pays, so we return None and let the
    delete visit every fragment rather than paying for both.
    """
    scanner = dataset.scanner(columns=[], filter=predicate, with_row_address=True)
    if _SCALAR_INDEX_PLAN_MARKER not in scanner.explain_plan(True):
        return None

    fragment_ids: set[int] = set()
    # Streamed, not to_table(): one overwritten partition can be hundreds of
    # millions of row addresses, and we only need the ids they live in.
    for batch in scanner.to_batches():
        batch_ids = pc.unique(pc.shift_right(batch.column("_rowaddr"), _FRAGMENT_ID_SHIFT))
        fragment_ids.update(cast("list[int]", batch_ids.to_pylist()))
    return fragment_ids


def _delete_matching_rows(
    open_context: DatasetOpenContext,
    predicate: str,
    fragment_ids: list[int],
) -> tuple[list[FragmentMetadata], list[int]]:
    """Run the per-fragment delete as a Daft job; return (updated, removed)."""
    if not fragment_ids:
        return [], []

    df = from_pylist([{"fragment_id": fragment_id} for fragment_id in fragment_ids])
    partitions = min(len(fragment_ids), _MAX_DELETE_PARTITIONS)
    # from_pylist lands everything in one partition, which would pin the whole
    # delete to a single task on a distributed runner. The native runner has no
    # partitions to spread -- repartition there is a no-op that only warns.
    if partitions > 1 and get_or_create_runner().name != "native":
        df = df.repartition(partitions, "fragment_id")
    handler = FragmentDeleteHandler(open_context, predicate)
    df = df.with_column("delete_result", handler(df["fragment_id"]))  # type: ignore[arg-type]

    updated_fragments: list[FragmentMetadata] = []
    removed_fragment_ids: list[int] = []
    for result in df.collect().to_pydict()["delete_result"]:
        if result["removed"]:
            removed_fragment_ids.append(int(result["fragment_id"]))
        elif result["fragment_meta"] is not None:
            updated_fragments.append(daft.pickle.loads(result["fragment_meta"]))
    return updated_fragments, removed_fragment_ids


def apply_conditional_overwrite(
    *,
    open_context: DatasetOpenContext,
    predicate: str,
    new_fragments: list[FragmentMetadata],
) -> lance.LanceDataset | None:
    """Delete the rows matching ``predicate`` and add ``new_fragments`` in one commit.

    ``open_context`` must be pinned to the version the write started from; that
    version is what the commit declares as its read version, so the deletions
    describe the snapshot they were computed against.

    Returns the committed dataset, or None when there was nothing to do (no
    matching rows and no new data), in which case no version is created.
    """
    pinned = open_context.open_pinned()
    candidates = _candidate_fragment_ids(pinned, predicate)
    if candidates is None:
        fragment_ids = [fragment.fragment_id for fragment in pinned.get_fragments()]
        logger.info("No scalar index covers %r; running delete over all %d fragments", predicate, len(fragment_ids))
    else:
        fragment_ids = sorted(candidates)
        logger.info("Scalar index pruned delete for %r down to %d fragments", predicate, len(fragment_ids))

    updated_fragments, removed_fragment_ids = _delete_matching_rows(open_context, predicate, fragment_ids)

    if not updated_fragments and not removed_fragment_ids and not new_fragments:
        return None

    operation = lance.LanceOperation.Update(
        removed_fragment_ids=removed_fragment_ids,
        updated_fragments=updated_fragments,
        new_fragments=list(new_fragments),
        # Deletions do not change any field's values, so no index needs to be
        # dropped from the fragments that survive.
        fields_modified=[],
    )
    return lance.LanceDataset.commit(
        open_context.uri,
        operation,
        read_version=open_context.version,
        storage_options=open_context.storage_options,
        **open_context.commit_kwargs,
    )
