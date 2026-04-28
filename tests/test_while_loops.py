"""While-loop search support — incremental tests as Phase 1/2/3 land.

Phase 1 (this file, currently): the orchestrator with `enable_while_loops=True`
generates StartWhile / EndWhile candidates and produces ASTs containing
WhileNode skeletons (with TBD condition + body) without crashing.

Phase 2/3 will add execute-phase replay and boolean condition learning.
"""
from core_lang_env.comp_env import BoolFunction, Function
from core_lang_env.syntax_tree import WhileNode
from searchers.search_orchestrator import (
    AugmentationRequestEndWhile,
    AugmentationRequestStartWhile,
    BASE_REQUEST_TYPES,
    FuncCallCandidate,
    SearchOrchestrator,
    SearchState,
    StartWhileCandidate,
    WHILE_REQUEST_TYPES,
    generate_augmentation_requests_from_state,
    generate_end_while_candidates,
    generate_start_while_candidates,
    get_cls_map,
)
from searchers.searchers_utils import Problem
from searchers.trace_searcher import (
    find_possible_end_while_actions,
    find_possible_start_while_actions,
)


# ---------- find_possible_start_while_actions ----------

def test_find_possible_start_while_actions_includes_full_set():
    """Unlike start_if which excludes the full set, while includes it
    (a while can apply to all traces — no else branch needed)."""
    subsets = find_possible_start_while_actions({0, 1, 2})
    assert {0, 1, 2} in subsets
    # Total: all non-empty subsets of {0,1,2} = 7 (singletons + pairs + full).
    assert len(subsets) == 7


def test_find_possible_start_while_singleton_input():
    subsets = find_possible_start_while_actions({0})
    assert subsets == [{0}]


# ---------- get_cls_map flag behaviour ----------

def test_get_cls_map_default_excludes_while():
    cls_map = get_cls_map()
    assert all(t in cls_map for t in BASE_REQUEST_TYPES)
    assert not any(t in cls_map for t in WHILE_REQUEST_TYPES)


def test_get_cls_map_enabled_includes_while():
    cls_map = get_cls_map(enable_while_loops=True)
    assert all(t in cls_map for t in BASE_REQUEST_TYPES)
    assert all(t in cls_map for t in WHILE_REQUEST_TYPES)


# ---------- generate_augmentation_requests respects cls_map ----------

def _trivial_problem():
    funcs = {"identity": Function(lambda x: (x,), [int], [int])}
    bools = {"gt": BoolFunction(lambda x, y: (x > y,), [int, int], [bool])}
    problem = Problem(
        (int,),
        (int,),
        instances={0: ((1,), (1,)), 1: ((2,), (2,))},
    )
    return problem, funcs, bools


def test_initial_state_with_while_enabled_yields_start_while_request():
    problem, funcs, _ = _trivial_problem()
    state = SearchState.init_new_search_state_from_problem_and_funcs(problem, funcs)
    requests = generate_augmentation_requests_from_state(state, get_cls_map(True))
    assert any(isinstance(r, AugmentationRequestStartWhile) for r in requests)


def test_initial_state_with_while_disabled_omits_start_while_request():
    problem, funcs, _ = _trivial_problem()
    state = SearchState.init_new_search_state_from_problem_and_funcs(problem, funcs)
    requests = generate_augmentation_requests_from_state(state, get_cls_map(False))
    assert not any(isinstance(r, AugmentationRequestStartWhile) for r in requests)


# ---------- generate_start_while_candidates ----------

def test_generate_start_while_candidates_returns_one_per_subset():
    """For 2 traces, 3 candidates should exist: {0}, {1}, {0,1}."""
    problem, funcs, _ = _trivial_problem()
    state = SearchState.init_new_search_state_from_problem_and_funcs(problem, funcs)
    requests = generate_augmentation_requests_from_state(state, get_cls_map(True))
    while_req = next(r for r in requests if isinstance(r, AugmentationRequestStartWhile))
    candidates = generate_start_while_candidates(state, while_req, cmaps=None)
    assert len(candidates) == 3
    while_indices_sets = {tuple(sorted(c.while_indices)) for c in candidates}
    assert while_indices_sets == {(0,), (1,), (0, 1)}


# ---------- apply_start_while_candidate ----------

def test_apply_start_while_inserts_while_node():
    problem, funcs, _ = _trivial_problem()
    state = SearchState.init_new_search_state_from_problem_and_funcs(problem, funcs)
    requests = generate_augmentation_requests_from_state(state, get_cls_map(True))
    while_req = next(r for r in requests if isinstance(r, AugmentationRequestStartWhile))
    candidates = generate_start_while_candidates(state, while_req, cmaps=None)
    # Pick the candidate where all traces enter the loop.
    full_subset_cand = next(c for c in candidates if set(c.while_indices) == {0, 1})
    new_state = state.apply_start_while_candidate(full_subset_cand)
    assert isinstance(new_state.ast_group.ast.statements[0], WhileNode)


# ---------- End-to-end smoke: orchestrator step() with while enabled doesn't crash ----------

def _hdist(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return abs(a - b)
    return 0


def test_orchestrator_step_with_while_enabled_doesnt_crash():
    """Run a few orchestrator steps with while enabled; no exceptions should escape."""
    problem, funcs, bools = _trivial_problem()
    orch = SearchOrchestrator.create_new_orchestrator_from_problem(
        problem, funcs, bools, _hdist, 50, map_size=10, enable_while_loops=True
    )
    for _ in range(50):
        if orch.search_queue.empty():
            break
        orch.step(trace_length_limit=4, max_ast_len=10)
    # The mere fact that we reached this assertion means no exception escaped.
    assert orch.last_processed_state is not None


def test_find_possible_end_while_actions_enumerates_rebindings():
    """With one pre-loop var (x0) and one body var (x1), both ints in scope,
    find_possible_end_while_actions returns the (x0,) <- (x1,) rebinding."""
    increment = Function(lambda x: (x + 1,), [int], [int])
    funcs = {"increment": increment}
    problem = Problem(
        (int,), (int,),
        instances={0: ((1,), (3,)), 1: ((1,), (4,))},
    )
    state = SearchState.init_new_search_state_from_problem_and_funcs(problem, funcs)

    # Drive: start_while + one body func call so var states have x0 (pre-loop) and x1 (body).
    state = state.apply_start_while_candidate(StartWhileCandidate(state.aug_stack.peek()[0], (0, 1)))
    body_pos = state.aug_stack.peek()[0]
    state = state.apply_func_call_candidate(FuncCallCandidate(
        exec_position=body_pos,
        group_indices=(0, 1),
        var_action=("increment", ("x0",)),
        short_actions=(("increment", (state.variable_states[0]["x0"],)),
                       ("increment", (state.variable_states[1]["x0"],))),
        output_length=1,
    ))

    rebinds = find_possible_end_while_actions(
        state.trace_group.traces,
        state.variable_states,
        body_var_names={"x1"},
        while_indices={0, 1},
    )
    assert (("x0",), ("x1",)) in rebinds


def test_generate_end_while_candidates_includes_rebinding():
    """End-of-while candidate generation should now include the (x0,) <- (x1,) rebinding,
    not just the trivial no-rebinding fallback."""
    increment = Function(lambda x: (x + 1,), [int], [int])
    funcs = {"increment": increment}
    problem = Problem(
        (int,), (int,),
        instances={0: ((1,), (3,)), 1: ((1,), (4,))},
    )
    state = SearchState.init_new_search_state_from_problem_and_funcs(problem, funcs)
    state = state.apply_start_while_candidate(StartWhileCandidate(state.aug_stack.peek()[0], (0, 1)))
    body_pos = state.aug_stack.peek()[0]
    state = state.apply_func_call_candidate(FuncCallCandidate(
        exec_position=body_pos,
        group_indices=(0, 1),
        var_action=("increment", ("x0",)),
        short_actions=(("increment", (state.variable_states[0]["x0"],)),
                       ("increment", (state.variable_states[1]["x0"],))),
        output_length=1,
    ))

    end_frontier = state.aug_stack.peek()
    request = AugmentationRequestEndWhile(
        exec_position=end_frontier[0],
        parent_indices=set(end_frontier[2]),
        group_indices=set(end_frontier[3]),
    )
    candidates = generate_end_while_candidates(state, request, cmaps=None)
    rebinding = [c for c in candidates if c.target_vars == ("x0",) and c.source_vars == ("x1",)]
    assert len(rebinding) == 1


def test_orchestrator_with_while_disabled_matches_prior_behavior():
    """With the flag off, the orchestrator behaves exactly as before — no
    StartWhile candidates ever appear in the search."""
    problem, funcs, bools = _trivial_problem()
    orch = SearchOrchestrator.create_new_orchestrator_from_problem(
        problem, funcs, bools, _hdist, 50, map_size=10, enable_while_loops=False
    )
    for _ in range(50):
        if orch.search_queue.empty():
            break
        orch.step(trace_length_limit=4, max_ast_len=10)
    # No state's AST should contain a WhileNode when the feature is disabled.
    def has_while(node):
        if isinstance(node, WhileNode):
            return True
        return any(has_while(c) for c in getattr(node, "children", []))

    for skel in orch.program_skeleton_candidates:
        assert not has_while(skel.ast_group.ast)
