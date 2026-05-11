"""Benchmark predecessor: return x0 - 1 given x0 and x1 = 0.

Tests while-loop synthesis with a function-helper composition in the
condition. Target program shape:

    while (not(eq(succ(x1), x0))) {
        x2 = succ(x1)
        x1 <- x2
    }
    return x1

For input (8, 0): iters 1-7 advance x1: 0→1→...→7. Iter 8 cond
not(eq(succ(7), 8)) = not(eq(8, 8)) = False → exit. Return x1 = 7.

The body is trivial (single succ + 1-target rebind). The hard part is
finding the right while condition — which requires `succ` available
inside the bool searcher so it can compose `not(eq(succ(x1), x0))`.
For this benchmark, `succ` is included in the bools dict (in addition
to funcs).

Run from v1/:
    python3 benchmarks/bench_predecessor.py
    python3 benchmarks/bench_predecessor.py --seed-level full --max-steps 50000
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

from benchmarks.problems import predecessor
from benchmarks._progress import print_progress


def _hdist(a, b):
    return abs(a - b) if isinstance(a, int) and isinstance(b, int) else 0


def _build_seed(problem, funcs, level: str = "full"):
    """Hand-build partial state. Levels:
      - "full":  start_while applied; full body (x2 = succ(x1)) and
                 end_while rebind x1 <- x2. Search just navigates execute
                 phase + return + Phase 3 condition fill.
      - "body":  start_while + x2 = succ(x1) only. Search must add the
                 end_while rebind, execute phase, return.
      - "tiny":  start_while applied. Empty body. Search builds it all.
    """
    state = SearchState.init_new_search_state_from_problem_and_funcs(problem, funcs)
    n = len(state.trace_group.traces)
    all_idx = tuple(range(n))

    sw_pos = state.aug_stack.peek()[0]
    state = state.apply_start_while_candidate(StartWhileCandidate(sw_pos, all_idx))

    if level == "tiny":
        return state

    # Body: x2 = succ(x1)
    fc_pos = state.aug_stack.peek()[0]
    short_actions = tuple(
        ("succ", (state.variable_states[i]["x1"],)) for i in range(n)
    )
    state = state.apply_func_call_candidate(FuncCallCandidate(
        fc_pos, all_idx, ("succ", ("x1",)), short_actions, 1,
    ))

    if level == "body":
        return state

    # end_while: x1 <- x2
    ew_pos = state.aug_stack.peek()[0]
    state = state.apply_end_while_candidate(EndWhileCandidate(
        ew_pos, all_idx, target_vars=("x1",), source_vars=("x2",),
    ))
    return state


def _queue_stats(orch):
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
               seed_level: str = "full",
               progress_every: int = 1000):
    with contextlib.redirect_stdout(io.StringIO()):
        orch = SearchOrchestrator.create_new_orchestrator_from_problem(
            problem, funcs, bools, _hdist, 50, map_size=map_size, enable_while_loops=True
        )
    # The right while condition is `not(eq(succ(x1), x0))` — depth 3.
    # Default max_depth=2 doesn't capture it; partition {1} (only one trace
    # continues at iter 3) becomes unrealizable and the search dead-ends.
    orch.bool_env.max_depth = 3
    if seeded:
        orch.search_queue = PriorityQueue()
        orch.tie_counter = 0
        orch.visited_states = set()
        orch.enqueue(_build_seed(problem, funcs, level=seed_level))

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
        if progress_every and steps % progress_every == 0:
            print_progress(orch, steps, t0)
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
    parser.add_argument("--mode", choices=["scratch", "seeded", "both"], default="seeded")
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--trace-length-limit", type=int, default=20)
    parser.add_argument("--max-ast-len", type=int, default=20)
    parser.add_argument("--max-while", type=int, default=None)
    parser.add_argument("--max-if", type=int, default=None)
    parser.add_argument("--seed-level", choices=["tiny", "body", "full"], default="full")
    parser.add_argument("--progress-every", type=int, default=1000,
                        help="Print queue stats + sample queued AST every N steps. 0 disables.")
    args = parser.parse_args()

    problem, funcs, bools = predecessor()

    cps = sorted({n for n in (10000, 50000, 100000, 250000) if n <= args.max_steps})

    if args.mode in ("seeded", "both"):
        orch, steps, elapsed, cp_log = run_search(
            problem, funcs, bools, seeded=True,
            max_steps=args.max_steps,
            trace_length_limit=args.trace_length_limit,
            max_ast_len=args.max_ast_len,
            max_while=args.max_while, max_if=args.max_if,
            checkpoints=cps,
            seed_level=args.seed_level,
            progress_every=args.progress_every,
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
            progress_every=args.progress_every,
        )
        report("FROM-SCRATCH (empty initial AST)", orch, steps, elapsed, cp_log)


if __name__ == "__main__":
    main()
