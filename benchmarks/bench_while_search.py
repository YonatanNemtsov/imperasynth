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


def _build_sum_of_list_seed(problem, funcs, level: str = "full"):
    """Hand-build a partial state at the requested level of detail.

    Levels (each strictly less pre-built than the prev):
      - "full":   x1 = zero(); start_while; full body; end_while with rebind.
                  Search just navigates execute phase + return.
      - "heavy":  x1 = zero(); start_while; full body (head + add + tail)
                  but NO end_while. Search must pick the right rebinding
                  pair (end_while), execute phase, return.
      - "medium": x1 = zero(); start_while; only x2 = get_head(x0) in body.
                  Search must finish the body + end_while + execute + return.
      - "tiny":   x1 = zero(); start_while applied. Empty body.
                  Search must build the entire body + end_while + execute + return.
    """
    state = SearchState.init_new_search_state_from_problem_and_funcs(problem, funcs)

    # x1 = zero();
    fc_pos = state.aug_stack.peek()[0]
    state = state.apply_func_call_candidate(FuncCallCandidate(
        fc_pos, (0, 1), ("zero", ()), (("zero", ()), ("zero", ())), 1,
    ))
    # start_while
    sw_pos = state.aug_stack.peek()[0]
    state = state.apply_start_while_candidate(StartWhileCandidate(sw_pos, (0, 1)))

    if level == "tiny":
        return state

    # Body: x2 = get_head(x0)
    fc_pos = state.aug_stack.peek()[0]
    state = state.apply_func_call_candidate(FuncCallCandidate(
        fc_pos, (0, 1), ("get_head", ("x0",)),
        (("get_head", (state.variable_states[0]["x0"],)),
         ("get_head", (state.variable_states[1]["x0"],))), 1,
    ))

    if level == "medium":
        return state

    # Body: x3 = add(x1, x2)
    fc_pos = state.aug_stack.peek()[0]
    state = state.apply_func_call_candidate(FuncCallCandidate(
        fc_pos, (0, 1), ("add", ("x1", "x2")),
        (("add", (state.variable_states[0]["x1"], state.variable_states[0]["x2"])),
         ("add", (state.variable_states[1]["x1"], state.variable_states[1]["x2"]))), 1,
    ))
    # Body: x4 = get_tail(x0)
    fc_pos = state.aug_stack.peek()[0]
    state = state.apply_func_call_candidate(FuncCallCandidate(
        fc_pos, (0, 1), ("get_tail", ("x0",)),
        (("get_tail", (state.variable_states[0]["x0"],)),
         ("get_tail", (state.variable_states[1]["x0"],))), 1,
    ))

    if level == "heavy":
        return state

    # end_while with rebind x0 <- x4, x1 <- x3
    ew_pos = state.aug_stack.peek()[0]
    state = state.apply_end_while_candidate(EndWhileCandidate(
        ew_pos, (0, 1), target_vars=("x0", "x1"), source_vars=("x4", "x3"),
    ))
    return state


def _queue_stats(orch):
    """Snapshot queue size and ast_size (score[0]) distribution."""
    sizes = [score[0] for score, _ in list(orch.search_queue.queue)]
    if not sizes:
        return {"qsize": 0, "min_ast": None, "max_ast": None, "median_ast": None}
    sizes.sort()
    return {
        "qsize": len(sizes),
        "min_ast": sizes[0],
        "max_ast": sizes[-1],
        "median_ast": sizes[len(sizes) // 2],
    }


def run_search(problem, funcs, bools, *, seeded: bool, max_steps: int,
               trace_length_limit: int, max_ast_len: int, map_size: int = 50,
               checkpoints: list[int] | None = None,
               max_while: int | None = None, max_if: int | None = None,
               seed_level: str = "full"):
    with contextlib.redirect_stdout(io.StringIO()):
        orch = SearchOrchestrator.create_new_orchestrator_from_problem(
            problem, funcs, bools, _hdist, 50, map_size=map_size, enable_while_loops=True
        )
    if seeded:
        orch.search_queue = PriorityQueue()
        orch.tie_counter = 0
        orch.visited_states = set()
        orch.enqueue(_build_sum_of_list_seed(problem, funcs, level=seed_level))

    checkpoints = sorted(checkpoints or [])
    cp_idx = 0
    cp_log = []

    t0 = time.perf_counter()
    steps = 0
    for steps in range(1, max_steps + 1):
        if orch.search_queue.empty():
            break
        with contextlib.redirect_stdout(io.StringIO()):
            orch.step(trace_length_limit=trace_length_limit, max_ast_len=max_ast_len,
                      max_while=max_while, max_if=max_if)
        if cp_idx < len(checkpoints) and steps == checkpoints[cp_idx]:
            cp_log.append((steps, time.perf_counter() - t0, _queue_stats(orch)))
            cp_idx += 1
        if orch.completed_programs:
            break
    elapsed = time.perf_counter() - t0
    return orch, steps, elapsed, cp_log


def report(label, orch, steps, elapsed, cp_log=None):
    from core_lang_env.syntax_tree import IfElseNode, WhileNode

    def _count(node, t):
        n = 1 if isinstance(node, t) else 0
        for c in getattr(node, "children", []):
            n += _count(c, t)
        return n

    n_with_while = sum(1 for _, st in list(orch.search_queue.queue)
                       if _count(st.ast_group.ast, WhileNode) > 0)

    print(f"--- {label} ---")
    print(f"  steps:              {steps}")
    print(f"  elapsed:            {elapsed:.2f} s")
    print(f"  queue at end:       {orch.search_queue.qsize()}")
    print(f"  queue with While:   {n_with_while}")
    print(f"  visited:            {len(orch.visited_states)}")
    print(f"  skeletons:          {len(orch.program_skeleton_candidates)}")
    print(f"  completed programs: {len(orch.completed_programs)}")

    # Show a sample of queued while-containing states to see what they look like.
    samples = []
    for score, st in list(orch.search_queue.queue):
        if _count(st.ast_group.ast, WhileNode) > 0:
            samples.append((score, st))
        if len(samples) >= 3:
            break
    if samples:
        print(f"  sample queued while-states:")
        for score, st in samples:
            top = st.aug_stack.peek() if st.aug_stack.stack else None
            top_opts = sorted(top[1]) if top else None
            print(f"    score={score} stack_depth={len(st.aug_stack.stack)} top_opts={top_opts}")
            for line in ast_to_code_str(st.ast_group.ast).splitlines():
                print(f"      {line}")
    if cp_log:
        print(f"  checkpoints:")
        print(f"    {'step':>10}  {'time':>6}  {'qsize':>8}  {'min_ast':>8}  {'med_ast':>8}  {'max_ast':>8}")
        for step, t, s in cp_log:
            print(f"    {step:>10}  {t:>6.1f}  {s['qsize']:>8}  {str(s['min_ast']):>8}  {str(s['median_ast']):>8}  {str(s['max_ast']):>8}")
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
    parser.add_argument("--max-while", type=int, default=None,
                        help="Cap on number of WhileNodes in any explored AST.")
    parser.add_argument("--max-if", type=int, default=None,
                        help="Cap on number of IfElseNodes in any explored AST.")
    parser.add_argument("--seed-level", choices=["tiny", "medium", "heavy", "full"], default="full",
                        help="How much of the program is pre-built in the seed.")
    args = parser.parse_args()

    problem, funcs, bools = sum_of_list()

    cps = sorted({n for n in (10000, 50000, 100000, 250000, 500000, 1_000_000)
                  if n <= args.max_steps})

    if args.mode in ("seeded", "both"):
        orch, steps, elapsed, cp_log = run_search(
            problem, funcs, bools, seeded=True,
            max_steps=args.max_steps,
            trace_length_limit=args.trace_length_limit,
            max_ast_len=args.max_ast_len,
            max_while=args.max_while, max_if=args.max_if,
            checkpoints=cps,
            seed_level=args.seed_level,
        )
        report(f"SEEDED ({args.seed_level} seed)", orch, steps, elapsed, cp_log)

    if args.mode in ("scratch", "both"):
        orch, steps, elapsed, cp_log = run_search(
            problem, funcs, bools, seeded=False,
            max_steps=args.max_steps,
            trace_length_limit=args.trace_length_limit,
            max_ast_len=args.max_ast_len,
            max_while=args.max_while, max_if=args.max_if,
            checkpoints=cps,
        )
        report("FROM-SCRATCH (empty initial AST)", orch, steps, elapsed, cp_log)


if __name__ == "__main__":
    main()
