"""Seeded search for sum-of-list — a problem that genuinely requires a while loop.

We hand-build a partial SearchState that already contains:
    x1 = zero();
    while (TBD) {
        x2 = get_head(x0);
        x3 = add(x1, x2);
        x4 = get_tail(x0);
        x0, x1 <- x4, x3;
    }
…with the build phase complete (first iteration ran), then seed the orchestrator's
search queue with that state. The remaining work for the search:
  - For each iteration of each trace, choose ENTER vs SKIP.
  - Eventually fire RETURN on x1 — should match each trace's target sum.

If the machinery is correct, the search should drain to a completed skeleton.
Run from v1/:
    python3 scripts/seed_sum_of_list.py
"""
import contextlib
import io
import os
import sys
from queue import PriorityQueue

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core_lang_env.comp_env import BoolFunction, Function
from core_lang_env.parser import ast_to_code_str
from searchers.search_orchestrator import (
    EndWhileCandidate,
    FuncCallCandidate,
    SearchOrchestrator,
    SearchState,
    StartWhileCandidate,
)
from searchers.searchers_utils import Problem


def hdist(a, b):
    return abs(a - b) if isinstance(a, int) and isinstance(b, int) else 0


def build_sum_of_list_problem():
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


def build_seed_state(problem, funcs):
    """Drive the orchestrator's apply methods to construct the halfway state."""
    state = SearchState.init_new_search_state_from_problem_and_funcs(problem, funcs)

    # x1 = zero();
    fc_pos = state.aug_stack.peek()[0]
    state = state.apply_func_call_candidate(FuncCallCandidate(
        exec_position=fc_pos, group_indices=(0, 1),
        var_action=("zero", ()),
        short_actions=(("zero", ()), ("zero", ())),
        output_length=1,
    ))

    # start_while, both traces enter
    sw_pos = state.aug_stack.peek()[0]
    state = state.apply_start_while_candidate(StartWhileCandidate(sw_pos, (0, 1)))

    # body: x2 = get_head(x0)
    fc_pos = state.aug_stack.peek()[0]
    state = state.apply_func_call_candidate(FuncCallCandidate(
        exec_position=fc_pos, group_indices=(0, 1),
        var_action=("get_head", ("x0",)),
        short_actions=(("get_head", (state.variable_states[0]["x0"],)),
                       ("get_head", (state.variable_states[1]["x0"],))),
        output_length=1,
    ))

    # body: x3 = add(x1, x2)
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

    # body: x4 = get_tail(x0)
    fc_pos = state.aug_stack.peek()[0]
    state = state.apply_func_call_candidate(FuncCallCandidate(
        exec_position=fc_pos, group_indices=(0, 1),
        var_action=("get_tail", ("x0",)),
        short_actions=(("get_tail", (state.variable_states[0]["x0"],)),
                       ("get_tail", (state.variable_states[1]["x0"],))),
        output_length=1,
    ))

    # end_while: rebind x0 <- x4, x1 <- x3
    ew_pos = state.aug_stack.peek()[0]
    state = state.apply_end_while_candidate(EndWhileCandidate(
        exec_position=ew_pos, group_indices=(0, 1),
        target_vars=("x0", "x1"),
        source_vars=("x4", "x3"),
    ))
    return state


def show_state(label, state):
    print(label)
    print("-" * 70)
    print("AST:")
    print(ast_to_code_str(state.ast_group.ast))
    print()
    for i, vs in enumerate(state.variable_states):
        deref = {k: state.trace_group.traces[i].objects[v].value for k, v in vs.items()}
        print(f"  trace {i}: {deref}")
    print(f"  stack depth: {len(state.aug_stack.stack)}")
    # Frontier = (exec_pos, options, parent_indices, group_indices)
    # Stack is bottom-up; print top-down (top = next to fire).
    for i, frontier in enumerate(reversed(state.aug_stack.stack)):
        exec_pos, options, parent, group = frontier
        marker = "TOP" if i == 0 else f" -{i}"
        print(f"  {marker}: ast_pos={exec_pos.ast_position} temporal={exec_pos.temporal_stack}")
        print(f"        options={sorted(options)}")
        print(f"        parent={sorted(parent)}, group={sorted(group)}")
    print(f"  search_concluded: {state.search_concluded}")
    print()


def main():
    problem, funcs, bools = build_sum_of_list_problem()

    with contextlib.redirect_stdout(io.StringIO()):
        orch = SearchOrchestrator.create_new_orchestrator_from_problem(
            problem, funcs, bools, hdist, 50, map_size=50, enable_while_loops=True
        )

    orch.search_queue = PriorityQueue()
    orch.tie_counter = 0
    orch.visited_states = set()

    seed = build_seed_state(problem, funcs)
    show_state("=== SEED STATE ===", seed)

    # ===========================================================================
    # PHASE A: WALK THE EXPECTED SUCCESS PATH BY HAND
    # ===========================================================================
    # ENTER {1} → execute body for trace 1 → SKIP {1} → RETURN x1
    # At each step, verify the state advances correctly and print key info.
    print("=" * 75)
    print("=== PHASE A: walking the expected success path manually ===")
    print("=" * 75)

    from searchers.search_orchestrator import (
        EnterWhileCandidate, SkipWhileCandidate, ExecuteBlockCandidate,
        ExecuteFuncCallCandidate, ExecuteDirectAssignCandidate,
        AugmentationRequestReturn, generate_return_candidates,
    )
    from core_lang_env.syntax_tree import (
        FunctionCallAssignNode, DirectAssignNode,
    )

    state = seed

    enter_pos = state.aug_stack.peek()[0]
    state = state.apply_enter_while_candidate(EnterWhileCandidate(enter_pos, (1,)))
    show_state("after ENTER {1}", state)

    block_pos = state.aug_stack.peek()[0]
    state = state.apply_execute_block_candidate(ExecuteBlockCandidate(
        block_pos, (1,),
        (FunctionCallAssignNode, FunctionCallAssignNode, FunctionCallAssignNode, DirectAssignNode),
    ))
    show_state("after EXECUTE_BLOCK", state)

    fc_pos = state.aug_stack.peek()[0]
    state = state.apply_execute_func_call_candidate(ExecuteFuncCallCandidate(fc_pos, (1,)))
    show_state("after EXECUTE_FUNC_CALL (x2 = get_head)", state)

    fc_pos = state.aug_stack.peek()[0]
    state = state.apply_execute_func_call_candidate(ExecuteFuncCallCandidate(fc_pos, (1,)))
    show_state("after EXECUTE_FUNC_CALL (x3 = add)", state)

    fc_pos = state.aug_stack.peek()[0]
    state = state.apply_execute_func_call_candidate(ExecuteFuncCallCandidate(fc_pos, (1,)))
    show_state("after EXECUTE_FUNC_CALL (x4 = get_tail)", state)

    da_pos = state.aug_stack.peek()[0]
    state = state.apply_execute_direct_assign_candidate(ExecuteDirectAssignCandidate(da_pos, (1,)))
    show_state("after EXECUTE_DIRECT_ASSIGN (x0,x1 <- x4,x3)", state)

    skip_pos = state.aug_stack.peek()[0]
    state = state.apply_skip_while_candidate(SkipWhileCandidate(skip_pos, (1,)))
    show_state("after SKIP {1}", state)

    # Now ask the orchestrator: what RETURN candidates would be generated here?
    parent_frontier = state.aug_stack.peek()
    targets_list = [problem.instances[i][1][0] for i in range(len(problem.instances))]
    return_request = AugmentationRequestReturn(
        exec_position=parent_frontier[0],
        parent_indices=set(parent_frontier[2]),
        group_indices=set(parent_frontier[3]),
    )
    return_cands = generate_return_candidates(state, return_request, orch.cmaps, targets_list)
    print(f"return candidates at parent frontier: {len(return_cands)}")
    for c in return_cands:
        print(f"  return_var={c.return_var}, group_indices={c.group_indices}")
    print()

    if return_cands:
        state_final = state.apply_return_candidate(return_cands[0])
        show_state("after RETURN (success path complete?)", state_final)
        print(f"search_concluded: {state_final.search_concluded}")
        print(f"trace solution values: {[t.objects[t.solution_object_id].value if t.solution_object_id is not None else None for t in state_final.trace_group.traces]}")
        print()

    # ===========================================================================
    # PHASE B: ACTUAL SEARCH from the seed
    # ===========================================================================
    print("=" * 75)
    print("=== PHASE B: running search from the seed ===")
    print("=" * 75)

    # Reset orch state for clean run
    orch.search_queue = PriorityQueue()
    orch.tie_counter = 0
    orch.visited_states = set()
    orch.program_skeleton_candidates = []
    orch.completed_programs = []
    orch.enqueue(seed)

    # Wrap step() to log what's being processed.
    from searchers.search_orchestrator import score_state
    orig_step = orch.step
    log = []

    parent_frontier_states_seen = []

    def logged_step(*args, **kwargs):
        if not orch.search_queue.empty():
            peek = orch.search_queue.queue[0]
            score, st = peek
            top = st.aug_stack.peek() if st.aug_stack.stack else None
            log.append((score, len(st.aug_stack.stack),
                        sorted(top[1]) if top else None))
            # Record info for stack==1 (parent frontier) states.
            if len(st.aug_stack.stack) == 1 and top and "AUG_RETURN" in top[1]:
                # For each, what RETURN candidates would be generated?
                from searchers.search_orchestrator import (
                    AugmentationRequestReturn, generate_return_candidates,
                )
                tg = [problem.instances[i][1][0] for i in range(len(problem.instances))]
                req = AugmentationRequestReturn(
                    exec_position=top[0],
                    parent_indices=set(top[2]),
                    group_indices=set(top[3]),
                )
                rcands = generate_return_candidates(st, req, orch.cmaps, tg)
                vs_summary = [{k: st.trace_group.traces[i].objects[v].value for k, v in vs.items()}
                              for i, vs in enumerate(st.variable_states)]
                parent_frontier_states_seen.append((score, len(rcands), vs_summary))
        return orig_step(*args, **kwargs)
    orch.step = logged_step

    MAX_STEPS = 300000
    n_exc = 0
    first_exc = None
    for step in range(MAX_STEPS):
        if orch.search_queue.empty():
            break
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                orch.step(trace_length_limit=30, max_ast_len=50)
        except Exception as e:
            n_exc += 1
            if first_exc is None:
                first_exc = (step, type(e).__name__, str(e)[:120])
        if orch.completed_programs:
            break

    print("first 8 states processed (score, stack_depth, top_options):")
    for entry in log[:8]:
        print(f"  {entry}")
    print(f"... ({len(log)} total)")
    print("last 8 states processed:")
    for entry in log[-8:]:
        print(f"  {entry}")
    enter_path = [e for e in log if e[1] >= 3]
    print(f"states with stack >= 3 processed: {len(enter_path)} / {len(log)}")
    print(f"parent-frontier (stack=1, RETURN-eligible) states processed: {len(parent_frontier_states_seen)}")
    for score, n_rc, vs in parent_frontier_states_seen[:8]:
        print(f"  score={score} return_candidates={n_rc} vars={vs}")
    print()

    print(f"steps run:           {step + 1}")
    print(f"queue size at end:   {orch.search_queue.qsize()}")
    print(f"visited states:      {len(orch.visited_states)}")
    print(f"exceptions:          {n_exc}")
    print(f"program skeletons:   {len(orch.program_skeleton_candidates)}")
    print(f"completed programs:  {len(orch.completed_programs)}")
    if first_exc:
        print(f"first exception:     step {first_exc[0]}: {first_exc[1]}: {first_exc[2]}")
    print()

    for i, sk in enumerate(orch.program_skeleton_candidates[:3]):
        print(f"--- skeleton {i} ---")
        print(ast_to_code_str(sk.ast_group.ast))
        sols = [t.objects[t.solution_object_id].value if t.solution_object_id is not None else None
                for t in sk.trace_group.traces]
        print(f"trace solutions: {sols}")
        print()
    for i, prog in enumerate(orch.completed_programs[:3]):
        print(f"--- completed program {i} (with learned bool conditions) ---")
        print(ast_to_code_str(prog))
        print()


if __name__ == "__main__":
    main()
