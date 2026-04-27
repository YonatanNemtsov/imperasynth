"""Benchmark while-loop synthesis on sum-of-list, comparing from-scratch and
seeded modes.

From-scratch: empty initial state, search must navigate to a while-shaped
skeleton on its own. With the current heuristic this is currently expected
to fail in any reasonable budget — the search prefers smaller-AST options
that explore the if/else and func-call space first, and the while-loop
branching factor is enormous.

Seeded: a hand-built partial state with the loop body already constructed.
The search just has to navigate the execute phase + return + condition fill.

This benchmark is a measuring stick: any future heuristic / pruning
improvements should reduce the from-scratch step count toward the seeded
count.

Run from v1/:
    python3 benchmarks/bench_while_search.py
    python3 benchmarks/bench_while_search.py --max-steps 200000
"""
import argparse
import contextlib
import io
import os
import sys
import time
from queue import PriorityQueue

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core_lang_env.parser import ast_to_code_str
from searchers.search_orchestrator import (
    EndWhileCandidate,
    FuncCallCandidate,
    SearchOrchestrator,
    SearchState,
    StartWhileCandidate,
)

from benchmarks.problems import sum_of_list


def _hdist(a, b):
    return abs(a - b) if isinstance(a, int) and isinstance(b, int) else 0


def _build_sum_of_list_seed(problem, funcs):
    """Hand-build a partial state that has the entire loop body constructed
    (build phase complete, executing_while frontier on top)."""
    state = SearchState.init_new_search_state_from_problem_and_funcs(problem, funcs)

    fc_pos = state.aug_stack.peek()[0]
    state = state.apply_func_call_candidate(FuncCallCandidate(
        fc_pos, (0, 1), ("zero", ()), (("zero", ()), ("zero", ())), 1,
    ))
    sw_pos = state.aug_stack.peek()[0]
    state = state.apply_start_while_candidate(StartWhileCandidate(sw_pos, (0, 1)))
    fc_pos = state.aug_stack.peek()[0]
    state = state.apply_func_call_candidate(FuncCallCandidate(
        fc_pos, (0, 1), ("get_head", ("x0",)),
        (("get_head", (state.variable_states[0]["x0"],)),
         ("get_head", (state.variable_states[1]["x0"],))), 1,
    ))
    fc_pos = state.aug_stack.peek()[0]
    state = state.apply_func_call_candidate(FuncCallCandidate(
        fc_pos, (0, 1), ("add", ("x1", "x2")),
        (("add", (state.variable_states[0]["x1"], state.variable_states[0]["x2"])),
         ("add", (state.variable_states[1]["x1"], state.variable_states[1]["x2"]))), 1,
    ))
    fc_pos = state.aug_stack.peek()[0]
    state = state.apply_func_call_candidate(FuncCallCandidate(
        fc_pos, (0, 1), ("get_tail", ("x0",)),
        (("get_tail", (state.variable_states[0]["x0"],)),
         ("get_tail", (state.variable_states[1]["x0"],))), 1,
    ))
    ew_pos = state.aug_stack.peek()[0]
    state = state.apply_end_while_candidate(EndWhileCandidate(
        ew_pos, (0, 1), target_vars=("x0", "x1"), source_vars=("x4", "x3"),
    ))
    return state


def run_search(problem, funcs, bools, *, seeded: bool, max_steps: int,
               trace_length_limit: int, max_ast_len: int, map_size: int = 50):
    with contextlib.redirect_stdout(io.StringIO()):
        orch = SearchOrchestrator.create_new_orchestrator_from_problem(
            problem, funcs, bools, _hdist, 50, map_size=map_size, enable_while_loops=True
        )
    if seeded:
        orch.search_queue = PriorityQueue()
        orch.tie_counter = 0
        orch.visited_states = set()
        orch.enqueue(_build_sum_of_list_seed(problem, funcs))

    t0 = time.perf_counter()
    steps = 0
    for steps in range(1, max_steps + 1):
        if orch.search_queue.empty():
            break
        with contextlib.redirect_stdout(io.StringIO()):
            orch.step(trace_length_limit=trace_length_limit, max_ast_len=max_ast_len)
        if orch.completed_programs:
            break
    elapsed = time.perf_counter() - t0
    return orch, steps, elapsed


def report(label, orch, steps, elapsed):
    print(f"--- {label} ---")
    print(f"  steps:              {steps}")
    print(f"  elapsed:            {elapsed:.2f} s")
    print(f"  queue at end:       {orch.search_queue.qsize()}")
    print(f"  visited:            {len(orch.visited_states)}")
    print(f"  skeletons:          {len(orch.program_skeleton_candidates)}")
    print(f"  completed programs: {len(orch.completed_programs)}")
    if orch.completed_programs:
        print(f"  first completed program:")
        for line in ast_to_code_str(orch.completed_programs[0]).splitlines():
            print(f"    {line}")
    elif orch.program_skeleton_candidates:
        print(f"  first skeleton (no condition fill):")
        for line in ast_to_code_str(orch.program_skeleton_candidates[0].ast_group.ast).splitlines():
            print(f"    {line}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scratch", "seeded", "both"], default="both")
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--trace-length-limit", type=int, default=20)
    parser.add_argument("--max-ast-len", type=int, default=30)
    args = parser.parse_args()

    problem, funcs, bools = sum_of_list()

    if args.mode in ("seeded", "both"):
        orch, steps, elapsed = run_search(
            problem, funcs, bools, seeded=True,
            max_steps=args.max_steps,
            trace_length_limit=args.trace_length_limit,
            max_ast_len=args.max_ast_len,
        )
        report("SEEDED (build phase pre-built)", orch, steps, elapsed)

    if args.mode in ("scratch", "both"):
        orch, steps, elapsed = run_search(
            problem, funcs, bools, seeded=False,
            max_steps=args.max_steps,
            trace_length_limit=args.trace_length_limit,
            max_ast_len=args.max_ast_len,
        )
        report("FROM-SCRATCH (empty initial AST)", orch, steps, elapsed)


if __name__ == "__main__":
    main()
