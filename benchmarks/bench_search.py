"""Search benchmarks — run the orchestrator on each problem and report metrics.

Usage (from v1/):
    python3 benchmarks/bench_search.py
    python3 benchmarks/bench_search.py --problem max_of_two --max-steps 5000

Reports per-problem:
    - wall time
    - steps taken
    - queue size at end
    - # program skeletons found
    - # completed programs (with boolean conditions filled in)
"""
import argparse
import sys
import time

# Make sibling modules importable when run as a script.
sys.path.insert(0, __file__.rsplit("/", 2)[0])

from search_orchestrator import SearchOrchestrator

from benchmarks.problems import ALL


def _hdist(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return abs(a - b)
    return 0


def run_one(name, problem_factory, max_steps, trace_length_limit, max_ast_len, map_size):
    problem, funcs, bools = problem_factory()
    print(f"\n=== {name} ({len(problem.instances)} instances) ===")

    t0 = time.perf_counter()
    orch = SearchOrchestrator.create_new_orchestrator_from_problem(
        problem, funcs, bools, _hdist, 50, map_size=map_size
    )
    setup_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    steps = 0
    for steps in range(1, max_steps + 1):
        if orch.search_queue.empty():
            break
        orch.step(trace_length_limit=trace_length_limit, max_ast_len=max_ast_len)
    search_time = time.perf_counter() - t0

    print(f"  setup:        {setup_time*1000:.1f} ms")
    print(f"  search:       {search_time*1000:.1f} ms ({steps} steps)")
    print(f"  queue size:   {orch.search_queue.qsize()}")
    print(f"  skeletons:    {len(orch.program_skeleton_candidates)}")
    print(f"  completed:    {len(orch.completed_programs)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=list(ALL) + ["all"], default="all")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--trace-length-limit", type=int, default=4)
    parser.add_argument("--max-ast-len", type=int, default=10)
    parser.add_argument("--map-size", type=int, default=20)
    args = parser.parse_args()

    names = list(ALL) if args.problem == "all" else [args.problem]
    for name in names:
        run_one(
            name,
            ALL[name],
            args.max_steps,
            args.trace_length_limit,
            args.max_ast_len,
            args.map_size,
        )


if __name__ == "__main__":
    main()
