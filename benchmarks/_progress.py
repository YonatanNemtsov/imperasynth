"""Shared progress-print helper for benchmarks.

Every bench's run_search loop calls `print_progress(orch, steps, t0)` at
some interval; this module centralizes the formatting so each bench
doesn't duplicate the snippet.
"""
import time

from core_lang_env.parser import ast_to_code_str


def queue_stats(orch):
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


def print_progress(orch, steps: int, t0: float, preview_lines: int = 8):
    """Print queue + skeleton stats and a sample queued AST."""
    stats = queue_stats(orch)
    print(f"[step {steps:>6}] t={time.perf_counter()-t0:>5.1f}s  q={stats['qsize']:>5}  "
          f"visited={len(orch.visited_states):>6}  "
          f"sk={len(orch.program_skeleton_candidates)}  "
          f"cp={len(orch.completed_programs)}  "
          f"ast(min/med/max)={stats['min_ast']}/{stats['median_ast']}/{stats['max_ast']}",
          flush=True)
    qitems = list(orch.search_queue.queue)
    if not qitems:
        return
    _, sample_state = min(qitems)
    lines = ast_to_code_str(sample_state.ast_group.ast).splitlines()
    preview = lines[:preview_lines]
    if len(lines) > preview_lines:
        preview.append(f"    ... [{len(lines) - preview_lines} more lines]")
    for line in preview:
        print(f"    {line}", flush=True)
