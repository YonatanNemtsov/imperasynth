"""End-to-end orchestrator tests — ported from notebooks/search_orchestrator.ipynb.

These run a real search loop on small problems and verify the orchestrator
gets through without crashing and produces at least one program skeleton.

Note: the while-loop construction path is currently incomplete in the orchestrator
(no `apply_start_while_candidate` etc. yet). These tests use problems whose
solutions only need func calls + if/else, which is the path that does work.
"""
from core_lang_env.comp_env import BoolFunction, Function
from search_orchestrator import (
    EndElseCandidate,
    EndIfCandidate,
    FuncCallCandidate,
    ReturnCandidate,
    SearchOrchestrator,
    SearchState,
    StartIfCandidate,
    generate_augmentation_requests_from_state,
)
from searchers_utils import Problem


def _max_of_two_problem():
    """Return-the-larger-of-two-ints. Solvable by a single if/else with return on each branch."""
    funcs = {
        "add": Function(lambda x, y: (x + y,), [int, int], [int]),
        "sub": Function(lambda x, y: (x - y,), [int, int], [int]),
        "identity": Function(lambda x: (x,), [int], [int]),
    }
    bools = {
        "gt": BoolFunction(lambda x, y: (x > y,), [int, int], [bool]),
        "equal": BoolFunction(lambda x, y: (x == y,), [int, int], [bool]),
        "is_even": BoolFunction(lambda x: (x % 2 == 0,), [int], [bool]),
        "not": BoolFunction(lambda x: (not x,), [bool], [bool]),
    }
    problem = Problem(
        (int, int),
        (int,),
        instances={
            0: ((3, 7), (7,)),
            1: ((10, 4), (10,)),
            2: ((11, 1), (11,)),
            3: ((18, 23), (23,)),
        },
    )
    return problem, funcs, bools


def _hdist(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return abs(a - b)
    return 0


# ---------- Initial-state construction ----------

def test_init_search_state_has_n_traces_and_empty_ast():
    problem, funcs, _ = _max_of_two_problem()
    state = SearchState.init_new_search_state_from_problem_and_funcs(problem, funcs)
    assert len(state.trace_group.traces) == 4
    assert state.ast_group.ast.statements == []
    assert state.search_concluded is False


def test_initial_state_yields_func_call_and_start_if_requests():
    """At the root of an empty program, the legal augmentations include
    func call, start-if, and return."""
    problem, funcs, _ = _max_of_two_problem()
    state = SearchState.init_new_search_state_from_problem_and_funcs(problem, funcs)
    requests = generate_augmentation_requests_from_state(state)
    request_types = {type(r).__name__ for r in requests}
    assert "AugmentationRequestFuncCall" in request_types
    assert "AugmentationRequestStartIf" in request_types
    assert "AugmentationRequestReturn" in request_types


# ---------- Orchestrator setup ----------

def test_create_orchestrator_initializes_queue_with_one_state():
    problem, funcs, bools = _max_of_two_problem()
    orch = SearchOrchestrator.create_new_orchestrator_from_problem(
        problem, funcs, bools, _hdist, 50, map_size=20
    )
    assert orch.search_queue.qsize() == 1
    assert orch.cmaps  # not empty
    assert len(orch.cmaps) == 4
    assert orch.completed_programs == []


# ---------- Step-level smoke ----------

def test_orchestrator_step_doesnt_crash_on_initial_state():
    """One step from the initial empty state should not crash and should
    produce at least one new state in the queue."""
    problem, funcs, bools = _max_of_two_problem()
    orch = SearchOrchestrator.create_new_orchestrator_from_problem(
        problem, funcs, bools, _hdist, 50, map_size=20
    )
    initial_qsize = orch.search_queue.qsize()
    result = orch.step(trace_length_limit=4, max_ast_len=10)
    # The step processes one state; new states should be enqueued.
    assert orch.search_queue.qsize() >= initial_qsize
    # And no exception escaped.
    assert result is not None or orch.search_queue.empty()


# ---------- Full search ----------

def test_search_finds_at_least_one_program_skeleton():
    """Running the full search on the max-of-two problem should produce
    at least one program skeleton candidate within a reasonable budget."""
    problem, funcs, bools = _max_of_two_problem()
    orch = SearchOrchestrator.create_new_orchestrator_from_problem(
        problem, funcs, bools, _hdist, 50, map_size=20
    )

    MAX_STEPS = 5000
    for _ in range(MAX_STEPS):
        if orch.search_queue.empty():
            break
        orch.step(trace_length_limit=4, max_ast_len=10)
        if orch.program_skeleton_candidates:
            break

    assert orch.program_skeleton_candidates, (
        f"no program skeleton found in {MAX_STEPS} steps; "
        f"queue size at end: {orch.search_queue.qsize()}"
    )


def test_program_skeleton_has_concluded_search_and_nonempty_ast():
    problem, funcs, bools = _max_of_two_problem()
    orch = SearchOrchestrator.create_new_orchestrator_from_problem(
        problem, funcs, bools, _hdist, 50, map_size=20
    )
    for _ in range(5000):
        if orch.search_queue.empty():
            break
        orch.step(trace_length_limit=4, max_ast_len=10)
        if orch.program_skeleton_candidates:
            break

    skel = orch.program_skeleton_candidates[0]
    assert skel.search_concluded
    assert skel.ast_group.ast.statements  # not empty


# ---------- Candidate type sanity ----------

def test_candidate_dataclass_field_shapes():
    """Quick structural check on candidate types."""
    fc = FuncCallCandidate(
        exec_position=((0,), (0, 0)),
        group_indices=(0, 1),
        var_action=("f", ("x0",)),
        short_actions=(("f", (0,)), ("f", (0,))),
        output_length=1,
    )
    assert fc.output_length == 1

    sif = StartIfCandidate(exec_position=((0,), (0, 0)), if_indices=(0,), else_indices=(1,))
    assert sif.if_indices == (0,)

    eif = EndIfCandidate(group_indices=(0, 1))
    assert eif.group_indices == (0, 1)

    eelse = EndElseCandidate(
        exec_position=((0,), (0, 0)),
        group_indices=(0,),
        target_vars=("x1",),
        source_vars=("x2",),
    )
    assert eelse.target_vars == ("x1",)

    ret = ReturnCandidate(
        exec_position=((0,), (0, 0)),
        group_indices=(0, 1),
        return_var="x1",
        short_actions=(("RETURN_FUNC_NAME", (1,)), ("RETURN_FUNC_NAME", (1,))),
    )
    assert ret.return_var == "x1"
