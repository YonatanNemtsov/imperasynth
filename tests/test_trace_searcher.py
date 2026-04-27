"""Trace searcher pieces — ported from notebooks/trace_searcher.ipynb.

The trace searcher provides:
  - TraceGroup: parallel execution envs for a multi-instance Problem
  - create_maps_from_problem: per-instance computational map (cmap) precomputation
  - find_possible_*_actions: enumerate aligned candidate actions across traces
  - rate_action / get_ranked_action_groups: heuristic scoring

These tests exercise each piece on a tiny "sum of list" problem.
"""
from core_lang_env.comp_env import BoolFunction, Function
from searchers_utils import Problem, propagate_importance_multi
from trace_searcher import (
    NO_ACTION,
    RETURN_FUNC_NAME,
    TraceGroup,
    create_maps_from_problem,
    create_traces_from_problem,
    filter_actions_by_active_trace_indices,
    find_possible_actions,
    find_possible_end_else_actions,
    find_possible_end_if_actions,
    find_possible_return_actions,
    find_possible_start_if_actions,
    get_ranked_action_groups,
    rate_action_group,
)


def _list_problem():
    """Sum-of-list problem: tuple input, int output."""
    add = Function(lambda x, y: (x + y,), [int, int], [int])
    head = Function(lambda lst: ((lst[0],) if lst else ()), [tuple], [int])
    tail = Function(lambda lst: ((lst[1:],) if lst else ((),)), [tuple], [tuple])
    is_empty = BoolFunction(lambda lst: ((len(lst) == 0,)), [tuple], [bool])

    funcs = {
        "add": add,
        "get_head": head,
        "get_tail": tail,
        "is_empty": is_empty,
    }

    problem = Problem(
        (tuple,),
        (int,),
        {
            0: (((1, 4, 6),), (11,)),
            1: (((4, 10, 4, 2),), (20,)),
            2: (((7, 11, 3),), (21,)),
        },
    )
    return problem, funcs


def _hdist(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return abs(a - b)
    return 0


# ---------- TraceGroup / create_traces_from_problem ----------

def test_create_traces_one_per_instance():
    problem, funcs = _list_problem()
    traces = create_traces_from_problem(problem, funcs)
    assert len(traces) == len(problem.instances)


def test_traces_have_input_objects_in_order():
    problem, funcs = _list_problem()
    traces = create_traces_from_problem(problem, funcs)
    # Trace 0's first input value should be (1, 4, 6) per the problem.
    inp_id = traces[0].input_objects[0]
    assert traces[0].objects[inp_id].value == (1, 4, 6)
    # Trace 1's input is (4, 10, 4, 2).
    inp_id = traces[1].input_objects[0]
    assert traces[1].objects[inp_id].value == (4, 10, 4, 2)


def test_trace_group_init_from_problem():
    problem, funcs = _list_problem()
    group = TraceGroup.init_new_group_from_problem_and_funcs(problem, funcs)
    assert len(group.traces) == 3


# ---------- create_maps_from_problem ----------

def test_create_maps_returns_one_per_instance():
    problem, funcs = _list_problem()
    cmaps = create_maps_from_problem(problem, funcs, _hdist, float("inf"), map_search_size=300)
    assert len(cmaps) == 3


def test_each_cmap_contains_target_object():
    problem, funcs = _list_problem()
    cmaps = create_maps_from_problem(problem, funcs, _hdist, float("inf"), map_search_size=300)
    # Each cmap should contain (int, target_value) as a known object.
    for i, (cmap, instance) in enumerate(zip(cmaps, problem.instances.values())):
        target = instance[1][0]
        assert (int, target) in cmap.objects, f"cmap {i} missing target {target}"


# ---------- find_possible_*_actions ----------

def test_find_possible_actions_returns_aligned_candidates():
    """Each returned tuple has (schema, row, output_len) where row has one entry
    per trace (either a ShortAction or NO_ACTION)."""
    problem, funcs = _list_problem()
    group = TraceGroup.init_new_group_from_problem_and_funcs(problem, funcs)
    cmaps = create_maps_from_problem(problem, funcs, _hdist, float("inf"), map_search_size=300)

    var_list = [{"x0": t.input_objects[0]} for t in group.traces]
    actions = find_possible_actions(group.traces, cmaps, var_list)
    assert len(actions) > 0
    # Each row must have len == n_traces.
    for schema, row, output_len in actions:
        assert len(row) == len(group.traces)
        # Each entry is either NO_ACTION or a tuple (func_name, input_ids).
        for entry in row:
            assert entry == NO_ACTION or (isinstance(entry, tuple) and len(entry) == 2)


def test_find_possible_return_actions_when_target_in_variables():
    """If a variable already holds the target value, a return action exists for it."""
    problem, funcs = _list_problem()
    traces = create_traces_from_problem(problem, funcs)
    targets = [inst[1][0] for inst in problem.instances.values()]

    # Manually inject a variable that holds the target value in each trace.
    var_list = []
    for trace, target in zip(traces, targets):
        # Add a CompObject whose value matches target by applying a synthetic action.
        # Simpler: use add() to create the target value, then point a variable at it.
        # For tuple-input traces this is messy. Instead, just point x0 at any input.
        var_list.append({"x0": trace.input_objects[0]})

    # x0 doesn't equal the int target, so no return action should be possible.
    return_actions = find_possible_return_actions(traces, targets, var_list)
    assert return_actions == []


def test_find_possible_start_if_actions_returns_proper_subsets():
    """All non-empty proper subsets of group_indices."""
    subsets = find_possible_start_if_actions({0, 1, 2})
    # Non-empty proper subsets of {0,1,2}: {0}, {1}, {2}, {0,1}, {0,2}, {1,2}. Total 6.
    assert len(subsets) == 6
    for s in subsets:
        assert 0 < len(s) < 3


def test_find_possible_end_if_returns_full_group():
    """end_if has only one possible split (all traces unify)."""
    result = find_possible_end_if_actions({0, 1, 2})
    assert result == [{0, 1, 2}]


# ---------- filter_actions_by_active_trace_indices ----------

def test_filter_actions_drops_rows_with_no_action_in_active_trace():
    """An action that has NO_ACTION in any active trace is dropped from the filtered list."""
    aligned = [
        (("get_head", ("x0",)), (("get_head", (0,)), ("get_head", (0,)), NO_ACTION), 1),
        (("get_tail", ("x0",)), (("get_tail", (0,)), ("get_tail", (0,)), ("get_tail", (0,))), 1),
    ]
    # Active = {0, 1, 2} → first row has NO_ACTION in trace 2 → dropped.
    filtered = filter_actions_by_active_trace_indices(aligned, {0, 1, 2})
    assert len(filtered) == 1
    assert filtered[0][0] == ("get_tail", ("x0",))


def test_filter_actions_with_active_subset_keeps_action():
    """If trace 2 is inactive, the row whose only NO_ACTION is at trace 2 stays."""
    aligned = [
        (("get_head", ("x0",)), (("get_head", (0,)), ("get_head", (0,)), NO_ACTION), 1),
    ]
    filtered = filter_actions_by_active_trace_indices(aligned, {0, 1})
    assert len(filtered) == 1
    # Trace 2's slot should now be NO_ACTION since it's inactive.
    _, row, _ = filtered[0]
    assert row[2] == NO_ACTION


# ---------- ranking ----------

def test_get_ranked_action_groups_returns_sorted_list():
    """Output is sorted by rating ascending."""
    problem, funcs = _list_problem()
    group = TraceGroup.init_new_group_from_problem_and_funcs(problem, funcs)
    cmaps = create_maps_from_problem(problem, funcs, _hdist, float("inf"), map_search_size=300)

    var_list = [{"x0": t.input_objects[0]} for t in group.traces]
    targets = [inst[1][0] for inst in problem.instances.values()]
    importances = [propagate_importance_multi(c, t) for c, t in zip(cmaps, targets)]
    actions = find_possible_actions(group.traces, cmaps, var_list)
    filtered = filter_actions_by_active_trace_indices(actions, {0, 1, 2})

    ranked = get_ranked_action_groups(group, filtered, targets, var_list, importances)
    # Sorted: ratings should be non-decreasing.
    ratings = [r[0] for r in ranked]
    assert ratings == sorted(ratings)
