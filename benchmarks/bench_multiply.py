"""Benchmark multiply: (a, b) -> a*b via repeated addition.

Tests the interdependent bool search — the loop condition needs an INVENTED
counter that doesn't exist in the problem inputs. The bool/scheme searcher
should hypothesize:
    init:    c0 = zero()
    update:  c0 = succ(c0)
    cond:    not(eq(c0, x1))

Target program shape:
    x2 = zero()
    c0 = zero()
    while (not(eq(c0, x1))) {
        x3 = add(x2, x0)
        x2 <- x3
        c0 = succ(c0)
    }
    return x2

Run from v1/:
    python3 benchmarks/bench_multiply.py --seed-level full
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

from benchmarks.problems import multiply
from benchmarks._progress import print_progress


def _hdist(a, b):
    return abs(a - b) if isinstance(a, int) and isinstance(b, int) else 0


def _all_indices(state):
    return tuple(range(len(state.trace_group.traces)))


def _fc(state, var_action, output_length=1):
    fc_pos = state.aug_stack.peek()[0]
    func_name, arg_names = var_action
    short_actions = tuple(
        (func_name, tuple(state.variable_states[i][a] for a in arg_names))
        for i in range(len(state.trace_group.traces))
    )
    return state.apply_func_call_candidate(FuncCallCandidate(
        fc_pos, _all_indices(state), var_action, short_actions, output_length,
    ))


def _build_seed(problem, funcs, level: str = "full"):
    """Levels:
      - "full":  x2=zero(); start_while; x3=add(x2,x0); end_while x2<-x3.
                 Search fills bool condition (interdependent search invents c0).
      - "body":  x2=zero(); start_while; x3=add(x2,x0). Search adds end_while.
      - "tiny":  x2=zero(); start_while. Search builds it all.
    """
    state = SearchState.init_new_search_state_from_problem_and_funcs(problem, funcs)

    # x2 = zero()
    state = _fc(state, ("zero", ()))

    # start_while — all traces enter (b > 0 in all instances)
    sw_pos = state.aug_stack.peek()[0]
    state = state.apply_start_while_candidate(StartWhileCandidate(
        sw_pos, _all_indices(state),
    ))

    if level == "tiny":
        return state

    # x3 = add(x2, x0)
    state = _fc(state, ("add", ("x2", "x0")))

    if level == "body":
        return state

    # end_while: x2 <- x3
    ew_pos = state.aug_stack.peek()[0]
    state = state.apply_end_while_candidate(EndWhileCandidate(
        ew_pos, _all_indices(state), target_vars=("x2",), source_vars=("x3",),
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
               trace_length_limit: int, max_ast_len: int, map_size: int = 200,
               checkpoints: list[int] | None = None,
               max_while: int | None = None, max_if: int | None = None,
               seed_level: str = "full",
               progress_every: int = 1000):
    with contextlib.redirect_stdout(io.StringIO()):
        orch = SearchOrchestrator.create_new_orchestrator_from_problem(
            problem, funcs, bools, _hdist, 50, map_size=map_size, enable_while_loops=True
        )
    # Bool searcher needs to find `not(eq(c0, x1))` once c0 is augmented in.
    # In BoolSearchEnvironment depth terms: eq(c0, x1) is depth 1 (two leaves),
    # not(eq(c0, x1)) is depth 2.
    orch.bool_env.max_depth = 2
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
    parser.add_argument("--trace-length-limit", type=int, default=30)
    parser.add_argument("--max-ast-len", type=int, default=30)
    parser.add_argument("--max-while", type=int, default=None)
    parser.add_argument("--max-if", type=int, default=None)
    parser.add_argument("--seed-level", choices=["tiny", "body", "full"], default="full")
    parser.add_argument("--progress-every", type=int, default=1000,
                        help="Print queue stats + a sample queued AST every N steps. 0 disables.")
    args = parser.parse_args()

    problem, funcs, bools = multiply()

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
