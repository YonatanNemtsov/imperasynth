"""Seeded-search regression tests.

The orchestrator's `search_queue` is public, so callers can hand-build a partial
SearchState and seed the queue with it. These tests verify that the search can
finish from a hand-built halfway state — particularly important for while loops,
since the heuristic alone wouldn't reliably steer the unseeded search to a
while-shaped skeleton.

Loadbearing bug fixes that this test depends on:
  - SimpleCompEnv.copy now copies action_history_short and signature.
  - generate_return_candidates reads state.variable_states (not the frozen
    AnnotatedAST annotations).
  - generate_enter_while_candidates filters out traces whose body wouldn't
    successfully execute on their current state.
"""
import contextlib
import io
from queue import PriorityQueue

from core_lang_env.comp_env import BoolFunction, Function
from core_lang_env.syntax_tree import WhileNode
from searchers.search_orchestrator import (
    EndElseCandidate,
    EndIfCandidate,
    EndWhileCandidate,
    FuncCallCandidate,
    SearchOrchestrator,
    SearchState,
    StartIfCandidate,
    StartWhileCandidate,
)
from searchers.searchers_utils import Problem


def _hdist(a, b):
    return abs(a - b) if isinstance(a, int) and isinstance(b, int) else 0


def _sum_of_list_problem():
    funcs = {
        "add": Function(lambda x, y: (x + y,), [int, int], [int]),
        "get_head": Function(lambda lst: ((lst[0],) if lst else ()), [tuple], [int]),
        "get_tail": Function(lambda lst: ((lst[1:],) if lst else ((),)), [tuple], [tuple]),
        "zero": Function(lambda: (0,), [], [int]),
    }
    bools = {
        "is_empty": BoolFunction(lambda lst: ((len(lst) == 0,)), [tuple], [bool]),
        "not": BoolFunction(lambda b: ((not b,)), [bool], [bool]),
    }
    problem = Problem(
        (tuple,), (int,),
        instances={
            0: (((1,),), (1,)),       # 1 iter
            1: (((1, 2),), (3,)),     # 2 iters
        },
    )
    return problem, funcs, bools


def _build_seed_state(problem, funcs):
    """Hand-build the partial state:
        x1 = zero();
        while (TBD) { x2 = get_head(x0); x3 = add(x1, x2); x4 = get_tail(x0); x0,x1 <- x4,x3; }
    Build phase complete (first iteration ran), executing_while_frontier on top.
    """
    state = SearchState.init_new_search_state_from_problem_and_funcs(problem, funcs)

    fc_pos = state.aug_stack.peek()[0]
    state = state.apply_func_call_candidate(FuncCallCandidate(
        exec_position=fc_pos, group_indices=(0, 1),
        var_action=("zero", ()),
        short_actions=(("zero", ()), ("zero", ())),
        output_length=1,
    ))

    sw_pos = state.aug_stack.peek()[0]
    state = state.apply_start_while_candidate(StartWhileCandidate(sw_pos, (0, 1)))

    fc_pos = state.aug_stack.peek()[0]
    state = state.apply_func_call_candidate(FuncCallCandidate(
        exec_position=fc_pos, group_indices=(0, 1),
        var_action=("get_head", ("x0",)),
        short_actions=(("get_head", (state.variable_states[0]["x0"],)),
                       ("get_head", (state.variable_states[1]["x0"],))),
        output_length=1,
    ))

    fc_pos = state.aug_stack.peek()[0]
    state = state.apply_func_call_candidate(FuncCallCandidate(
        exec_position=fc_pos, group_indices=(0, 1),
        var_action=("add", ("x1", "x2")),
        short_actions=(
            ("add", (state.variable_states[0]["x1"], state.variable_states[0]["x2"])),
            ("add", (state.variable_states[1]["x1"], state.variable_states[1]["x2"])),
        ),
        output_length=1,
    ))

    fc_pos = state.aug_stack.peek()[0]
    state = state.apply_func_call_candidate(FuncCallCandidate(
        exec_position=fc_pos, group_indices=(0, 1),
        var_action=("get_tail", ("x0",)),
        short_actions=(("get_tail", (state.variable_states[0]["x0"],)),
                       ("get_tail", (state.variable_states[1]["x0"],))),
        output_length=1,
    ))

    ew_pos = state.aug_stack.peek()[0]
    state = state.apply_end_while_candidate(EndWhileCandidate(
        exec_position=ew_pos, group_indices=(0, 1),
        target_vars=("x0", "x1"),
        source_vars=("x4", "x3"),
    ))
    return state


def _has_while(node):
    if isinstance(node, WhileNode):
        return True
    return any(_has_while(c) for c in getattr(node, "children", []))


def test_seeded_sum_of_list_search_finds_while_skeleton():
    """Hand-build a partial state with the sum-of-list while-loop shape, seed
    the orchestrator's queue, and verify the search drains to a completed
    skeleton whose trace solutions match each instance's target."""
    problem, funcs, bools = _sum_of_list_problem()

    with contextlib.redirect_stdout(io.StringIO()):
        orch = SearchOrchestrator.create_new_orchestrator_from_problem(
            problem, funcs, bools, _hdist, 50, map_size=50, enable_while_loops=True
        )
    # Replace queue with our seed.
    orch.search_queue = PriorityQueue()
    orch.tie_counter = 0
    orch.visited_states = set()
    orch.enqueue(_build_seed_state(problem, funcs))

    MAX_STEPS = 200
    for _ in range(MAX_STEPS):
        if orch.search_queue.empty():
            break
        with contextlib.redirect_stdout(io.StringIO()):
            orch.step(trace_length_limit=20, max_ast_len=30)
        if orch.program_skeleton_candidates:
            break

    # At least one skeleton found.
    assert orch.program_skeleton_candidates, "search did not find any skeleton from the seed"

    # The skeleton should contain a WhileNode (we seeded one).
    skeleton = orch.program_skeleton_candidates[0]
    assert _has_while(skeleton.ast_group.ast), "skeleton lost its while loop"

    # Each trace should have a solution_object_id pointing to a value matching its target.
    for i, trace in enumerate(skeleton.trace_group.traces):
        assert trace.solution_object_id is not None, f"trace {i} has no solution"
        target = problem.instances[i][1][0]
        actual = trace.objects[trace.solution_object_id].value
        assert actual == target, f"trace {i}: returned {actual}, expected {target}"


def test_seeded_sum_of_list_phase3_fills_in_while_condition():
    """Phase 3: extract_while_conditional_problem_for_group + integrate_while
    fills in the TBD condition with a learned bool expression. Verify the
    completed program actually executes correctly on training inputs AND on
    inputs not seen during training (generalization)."""
    from core_lang_env.comp_env import CompObject, SimpleCompEnv
    from core_lang_env.exec_code_v2 import (
        ExecutionContext, ExecutionPositionV2, execute_step,
    )

    problem, funcs, bools = _sum_of_list_problem()

    with contextlib.redirect_stdout(io.StringIO()):
        orch = SearchOrchestrator.create_new_orchestrator_from_problem(
            problem, funcs, bools, _hdist, 50, map_size=50, enable_while_loops=True
        )
    orch.search_queue = PriorityQueue()
    orch.tie_counter = 0
    orch.visited_states = set()
    orch.enqueue(_build_seed_state(problem, funcs))

    for _ in range(200):
        if orch.search_queue.empty() or orch.completed_programs:
            break
        with contextlib.redirect_stdout(io.StringIO()):
            orch.step(trace_length_limit=20, max_ast_len=30)

    assert orch.completed_programs, "Phase 3 did not produce a completed program"
    program = orch.completed_programs[0]

    # The completed AST must NOT contain TBD_CONDITIONAL anywhere.
    code = repr(program)
    assert "TBD_CONDITIONAL" not in code, "completed program still has TBD condition"

    # Execute the synthesized program on training + held-out inputs.
    def run_on(lst):
        env = SimpleCompEnv()
        env.add_input_object(CompObject(tuple, lst))
        for name, f in funcs.items():
            env.add_function(name, f)
        for name, f in bools.items():
            env.add_function(name, f)
        ctx = ExecutionContext(program, ExecutionPositionV2.start_position(), env, {"x0": 0}, False)
        for _ in range(1000):
            if ctx.completed:
                break
            execute_step(ctx)
        return env.objects[env.solution_object_id].value if env.solution_object_id is not None else None

    # Training instances.
    assert run_on((1,)) == 1
    assert run_on((1, 2)) == 3
    # Generalization: empty list and longer lists.
    assert run_on(()) == 0
    assert run_on((1, 2, 3, 4, 5)) == 15
    assert run_on((10, 20, 30)) == 60


def _sum_of_evens_problem():
    funcs = {
        "add": Function(lambda x, y: (x + y,), [int, int], [int]),
        "get_head": Function(lambda lst: ((lst[0],) if lst else ()), [tuple], [int]),
        "get_tail": Function(lambda lst: ((lst[1:],) if lst else ((),)), [tuple], [tuple]),
        "zero": Function(lambda: (0,), [], [int]),
        "identity": Function(lambda x: (x,), [int], [int]),
    }
    bools = {
        "is_empty": BoolFunction(lambda lst: ((len(lst) == 0,)), [tuple], [bool]),
        "is_even": BoolFunction(lambda x: (x % 2 == 0,), [int], [bool]),
        "not": BoolFunction(lambda b: ((not b,)), [bool], [bool]),
    }
    problem = Problem(
        (tuple,), (int,),
        instances={
            0: (((2,),), (2,)),       # one even
            1: (((1,),), (0,)),       # one odd
            2: (((1, 2),), (2,)),     # mix, len 2
            3: (((2, 3, 4),), (6,)),  # mix, len 3
        },
    )
    return problem, funcs, bools


def _build_sum_of_evens_seed(problem, funcs):
    """Hand-build the full sum-of-evens skeleton:
        x1 = zero();
        while (TBD) {
            x2 = get_head(x0);
            x3 = get_tail(x0);
            if (TBD) { x4 = add(x1, x2); }
            else     { x5 = identity(x1); x4 <- x5; }
            x0, x1 <- x3, x4;
        }
    Build phase complete; executing_while_frontier on top.
    """
    state = SearchState.init_new_search_state_from_problem_and_funcs(problem, funcs)
    n = len(state.trace_group.traces)
    all_idx = tuple(range(n))

    def fc(s, var_action):
        func_name, arg_names = var_action
        sa = tuple(
            (func_name, tuple(s.variable_states[i][a] for a in arg_names))
            for i in range(n)
        )
        return s.apply_func_call_candidate(FuncCallCandidate(
            s.aug_stack.peek()[0], all_idx, var_action, sa, 1,
        ))

    state = fc(state, ("zero", ()))
    state = state.apply_start_while_candidate(StartWhileCandidate(
        state.aug_stack.peek()[0], all_idx,
    ))
    state = fc(state, ("get_head", ("x0",)))
    state = fc(state, ("get_tail", ("x0",)))

    # Iter-1 split by is_even(head).
    if_idx = tuple(i for i in range(n) if problem.instances[i][0][0][0] % 2 == 0)
    else_idx = tuple(i for i in range(n) if i not in if_idx)
    state = state.apply_start_if_candidate(StartIfCandidate(
        state.aug_stack.peek()[0], if_idx, else_idx,
    ))

    # if-branch: x4 = add(x1, x2)
    sa = tuple(
        ("add", (state.variable_states[i]["x1"], state.variable_states[i]["x2"]))
        if i in if_idx else "NO_ACTION"
        for i in range(n)
    )
    state = state.apply_func_call_candidate(FuncCallCandidate(
        state.aug_stack.peek()[0], if_idx, ("add", ("x1", "x2")), sa, 1,
    ))
    state = state.apply_end_if_candidate(EndIfCandidate(group_indices=if_idx))

    # else-branch: x5 = identity(x1)
    sa = tuple(
        ("identity", (state.variable_states[i]["x1"],))
        if i in else_idx else "NO_ACTION"
        for i in range(n)
    )
    state = state.apply_func_call_candidate(FuncCallCandidate(
        state.aug_stack.peek()[0], else_idx, ("identity", ("x1",)), sa, 1,
    ))
    state = state.apply_end_else_candidate(EndElseCandidate(
        state.aug_stack.peek()[0], else_idx,
        target_vars=("x4",), source_vars=("x5",),
    ))

    state = state.apply_end_while_candidate(EndWhileCandidate(
        state.aug_stack.peek()[0], all_idx,
        target_vars=("x0", "x1"), source_vars=("x3", "x4"),
    ))
    return state


def test_seeded_sum_of_evens_phase3_completes_correctly():
    """Sum-of-evens has a while loop with an if/else inside its body. This
    exercises three pieces simultaneously: (1) the body-validity simulator
    handling vars defined by end_else, (2) Phase 3's while-condition
    extractor simulating IfElseNode children correctly, (3) the if/else
    Phase 3 fill being threaded through to the while-condition stage."""
    from core_lang_env.comp_env import CompObject, SimpleCompEnv
    from core_lang_env.exec_code_v2 import (
        ExecutionContext, ExecutionPositionV2, execute_step,
    )

    problem, funcs, bools = _sum_of_evens_problem()

    with contextlib.redirect_stdout(io.StringIO()):
        orch = SearchOrchestrator.create_new_orchestrator_from_problem(
            problem, funcs, bools, _hdist, 50, map_size=50, enable_while_loops=True
        )
    orch.search_queue = PriorityQueue()
    orch.tie_counter = 0
    orch.visited_states = set()
    orch.enqueue(_build_sum_of_evens_seed(problem, funcs))

    for _ in range(500):
        if orch.search_queue.empty() or orch.completed_programs:
            break
        with contextlib.redirect_stdout(io.StringIO()):
            orch.step(trace_length_limit=30, max_ast_len=40)

    assert orch.completed_programs, "Phase 3 did not produce a completed program"
    program = orch.completed_programs[0]
    assert "TBD_CONDITIONAL" not in repr(program), "completed program still has TBD condition"

    def run_on(lst):
        env = SimpleCompEnv()
        env.add_input_object(CompObject(tuple, lst))
        for name, f in {**funcs, **bools}.items():
            env.add_function(name, f)
        ctx = ExecutionContext(program, ExecutionPositionV2.start_position(), env, {"x0": 0}, False)
        for _ in range(1000):
            if ctx.completed:
                break
            execute_step(ctx)
        return env.objects[env.solution_object_id].value if env.solution_object_id is not None else None

    # Training instances.
    assert run_on((2,)) == 2
    assert run_on((1,)) == 0
    assert run_on((1, 2)) == 2
    assert run_on((2, 3, 4)) == 6
    # Generalization: empty list, all-odd, all-even, longer mixes.
    assert run_on(()) == 0
    assert run_on((1, 3, 5)) == 0
    assert run_on((2, 4, 6)) == 12
    assert run_on((1, 2, 3, 4, 5, 6)) == 12
    assert run_on((7, 8, 9, 10)) == 18


def test_simple_comp_env_copy_does_not_share_action_history_short():
    """Regression test for the bug where SimpleCompEnv.copy left
    action_history_short and signature shared between original and copy.
    """
    from core_lang_env.comp_env import CompObject, SimpleCompEnv

    add = Function(lambda x, y: (x + y,), [int, int], [int])
    env = SimpleCompEnv()
    env.add_input_object(CompObject(int, 1))
    env.add_input_object(CompObject(int, 2))
    env.add_function("add", add)

    env_copy = env.copy()
    # Both start with empty action history.
    assert env.action_history_short == []
    assert env_copy.action_history_short == []

    # Apply on the copy only.
    env_copy.apply_function("add", (0, 1))

    # The original must NOT have been mutated.
    assert env.action_history_short == [], (
        "SimpleCompEnv.copy is leaking action_history_short between "
        "original and copy"
    )
    assert env.signature.sig == [], (
        "SimpleCompEnv.copy is leaking signature between original and copy"
    )
    # The copy itself does have the new action.
    assert len(env_copy.action_history_short) == 1
