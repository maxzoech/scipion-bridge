"""Item ids: the lineage policies of the stages (in-process, see lineage.py)."""

import numpy as np

import scipion_bridge as B
from scipion_bridge.core.streaming import lineage
from scipion_bridge.core.streaming.lineage import (
    AccumulatorLineage,
    PassThrough,
    SourceLineage,
    element_digest,
    fill_created,
    map_id,
)
from scipion_bridge.core.struct import concat, identity
from scipion_bridge.core.struct.id_column import IdColumn

F = b"F" * 16
A, B_, C, D = (bytes([k]) * 16 for k in range(4))


class Value(B.Struct):
    value: float


class Group(B.Struct):
    members: B.Set[Value]
    group_id: int


def _values(n, namespace=None, start=0):
    s = B.Set[Value](capacity=n)
    s["value"] = np.arange(start, start + n, dtype=np.float64)
    if namespace is not None:
        identity.set_ids(s, IdColumn.of(namespace, range(start, start + n)))
    return s


def _ids(obj):
    column = identity.element_ids(obj)
    assert column is not None
    return column.elements()


# -- Sources -------------------------------------------------------------------
def test_sources_name_items_by_their_element_ids_however_they_were_chunked():
    whole = _values(6, F)
    joined = concat([_values(3, F), _values(3, F, start=3)])

    assert SourceLineage("source").receive(whole, None) == SourceLineage(
        "source",
    ).receive(joined, None)


def test_items_with_other_element_ids_get_other_ids():
    source = SourceLineage("source")

    assert source.receive(_values(3, F), None) != source.receive(
        _values(3, F, start=3),
        None,
    )


def test_an_input_sent_twice_gives_two_items():
    source = SourceLineage("source")

    first = source.receive(_values(3, F), None)
    second = source.receive(_values(3, F), None)

    assert first != second


def test_items_without_element_ids_are_counted():
    ids = [SourceLineage("source").receive(value, None) for value in [1, 2]]
    counted = SourceLineage("source")

    assert [counted.receive(value, None) for value in [1, 2]][0] == ids[0]
    assert counted.receive(3, None) not in ids


def test_sources_of_other_stages_give_other_ids():
    assert SourceLineage("a").receive(_values(3, F), None) != SourceLineage(
        "b",
    ).receive(_values(3, F), None)


def test_items_routed_into_a_child_keep_their_id():
    assert SourceLineage("source").receive(_values(3, F), A) == A


def test_element_digest_covers_the_sets_nested_in_an_item():
    group = Group(members=_values(2, F), group_id=0)
    other = Group(members=_values(2, F, start=2), group_id=0)

    assert element_digest((1, 2)) is None
    assert element_digest(_values(2)) is None
    assert element_digest(group) is not None
    assert element_digest(group) != element_digest(other)


# -- Maps and pass-through stages -------------------------------------------------
def test_map_results_are_named_by_the_map_and_their_parent():
    assert map_id("map", A) == map_id("map", A)
    assert map_id("map", A) != map_id("map", B_)
    assert map_id("map", A) != map_id("other", A)


def test_pass_through_stages_keep_the_id():
    stage = PassThrough("merge")

    assert stage.emit(stage.receive(1, A), 1) == [A]
    assert stage.emit(stage.receive(1, A), 0) == []


# -- Accumulators ----------------------------------------------------------------
def _emissions(policy, received):
    """The ids an accumulator emits for ``received``: (id, emissions) pairs."""
    return [policy.emit(lid, count) for lid, count in received]


def test_an_emission_names_the_items_received_since_the_last_one():
    policy = AccumulatorLineage("chunk")
    _emissions(policy, [(B_, 0), (A, 0)])

    assert policy.parents() == [A, B_]
    assert (
        _emissions(AccumulatorLineage("chunk"), [(A, 0), (B_, 0), (C, 1)])[2]
        == _emissions(AccumulatorLineage("chunk"), [(B_, 0), (A, 0), (C, 1)])[2]
    )


def test_an_emission_names_the_item_that_left_a_rest_in_the_state():
    """chunk(4) of items of 3 rows: C's last rows go into the next chunk."""
    policy = AccumulatorLineage("chunk")
    _emissions(policy, [(A, 0), (C, 1), (D, 0)])

    assert policy.parents() == [C, D]


def test_emissions_of_one_call_and_of_later_calls_differ():
    policy = AccumulatorLineage("flatten")

    first = policy.emit(A, 3)
    later = policy.emit(A, 1)

    assert len(set(first)) == 3
    assert later[0] not in first


def test_the_previous_emission_chains_the_history_of_a_stage():
    """A scan emitting for every item: the same item after another history
    gives another id."""
    after_a = _emissions(AccumulatorLineage("scan"), [(A, 1), (C, 1)])
    after_b = _emissions(AccumulatorLineage("scan"), [(B_, 1), (C, 1)])

    assert after_a[1] != after_b[1]


def test_flush_names_the_items_since_the_last_emission():
    collected = AccumulatorLineage("collect")
    other = AccumulatorLineage("collect")
    _emissions(collected, [(A, 0), (B_, 0)])
    _emissions(other, [(A, 0)])

    assert collected.flush(1) != other.flush(1)
    assert collected.flush(0) == []


def test_runs_with_the_same_inputs_give_the_same_ids():
    received = [(A, 0), (B_, 1), (C, 2), (D, 0)]

    first, second = AccumulatorLineage("s"), AccumulatorLineage("s")

    assert _emissions(first, received) == _emissions(second, received)
    assert first.flush(1) == second.flush(1)


# -- Rows a stage creates ----------------------------------------------------------
def test_created_rows_get_ids_from_the_item_they_are_emitted_in():
    created = _values(3)

    fill_created(created, A)

    namespace = lineage.item_id(A, b"")
    assert _ids(created) == [(namespace, row) for row in range(3)]


def test_rows_with_ids_keep_them():
    s = concat([_values(2, F), _values(2)])

    fill_created(s, A)

    elements = _ids(s)
    assert elements[:2] == [(F, 0), (F, 1)]
    assert elements[2:] == [(lineage.item_id(A, b""), 2), (lineage.item_id(A, b""), 3)]


def test_the_sets_of_an_item_get_ids_of_their_own():
    collection = B.Collection[Group](size=2)
    collection[0] = Group(members=_values(2), group_id=0)
    collection[1] = Group(members=_values(2), group_id=1)
    item = (collection, _values(2))

    fill_created(item, A)

    namespaces = {
        namespace
        for obj in [collection[0].members, collection[1].members, item[1]]
        for namespace, _ in _ids(obj)
    }
    assert len(namespaces) == 3
    assert _ids(collection[1])[0][1] == 0


def test_members_below_two_axes_get_no_ids():
    groups = B.Set[Group]([Group(members=_values(2), group_id=0)])

    fill_created(groups, A)

    assert _ids(groups) == [(lineage.item_id(A, b""), 0)]
    assert set(identity.hidden_ids(groups)) == {()}
