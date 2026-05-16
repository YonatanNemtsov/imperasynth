# Imperative Program Synthesis from Examples


SSA-like imperative program synthesizer. Given input/output pairs, and a set of primitive functions, the system searches for a program that takes each input to its corresponding output. The programs are constructed in a small imperative language with if-else, while loops, function-call assignments, and direct assignments (variable-to-variable).

## Examples

Each problem is defined by:
- a set of primitive functions for program skeleton construction, i.e excluding conditions (typed; e.g. `add: int×int → int`),
- a set of primitive functions for condition construction (e.g. `eq`, `not`, `is_empty`) 
- a handful of input/output pairs.


### Max of two

Given `identity` for the body and `gt`, `not` for conditions, with pairs `{(x0, x1) -> output}: (3,7)→7, (10,4)→10, (11,1)→11, (18,23)→23`, the synthesizer produces:

```
inputs: x0, x1

{
    if (gt(x1, x0)) {
        x2 = identity(x1);
    } else {
        x3 = identity(x0);
        x2 <- x3;
    }
    return x2;
}
```

A single if-else returning the larger input. ~1.4K steps, ~8 seconds from an empty initial AST.

### Multiplication

Given `add` and `zero` for the body plus `zero`, `succ`, `eq`, `not` for conditions, with pairs `{(x0, x1) -> output}: (3,2)→6, (4,3)→12, (2,5)→10`, the synthesizer produces:

```
inputs: x0, x1

{
    x2 = zero();
    c0 = zero();                        ← invented persistent state
    b0 = eq(x0, c0);
    while (not(b0)) {
        x3 = add(x1, x2);
        x2 <- x3;
        b0 = eq(x0, c0);
        s0 = succ(c0);
        c0 <- s0;                       ← invented update step
    }
    return x2;
}
```

~3 minutes from an empty initial AST.

## Benchmark numbers

Numbers below are from a single CPU core. **Seed** indicates how much of the target program is pre-built before search starts: `scratch` is an empty AST, `tiny`/`body`/`full` are progressively pre-built partial programs (defined per bench, see `benchmarks/bench_*.py`). **Steps** is the number of states the orchestrator processed (popped from the priority queue) before finding a solution. Rows with `—` mean the search exhausted the step budget without finding one.

| Problem | Seed | Steps | Time | Notes |
|---|---|---|---|---|
| predecessor | scratch | 60 | 0.1s | ✓ |
| predecessor | full | 50 | 0.1s | ✓ |
| sum_of_evens | full | 71 | 0.03s | ✓ |
| sum_of_evens | body | 40,463 | 27s | ✓ |
| sum_of_evens | tiny | — | — | doesn't converge |
| sum_of_evens | scratch | — | — | doesn't converge |
| multiply | full | 125 | 0.55s | ✓ |
| multiply | scratch | ~32K | ~3 min | ✓ |
| sum_of_list | full | 9 | <0.1s | ✓ |
| sum_of_list | scratch | 21,631 | 6.3s | ✓, `--max-while 1 --max-if 0` |

## How it works

Upfront, for each input/output pair the system builds a directed graph of values reachable from the inputs by applying primitive functions (computational maps). These graphs are used to prune the search later — function calls that can't reach the target output get dropped.

Search proceeds in three phases per candidate program:

1. **Build the program shape.** Add one statement at a time — a function call, the start or end of a `while` / `if-else`, a variable rebind. The reachability graphs prune statements that can't lead to the target. When opening a branch, the proposed split (which input/output pairs enter the `if` vs `else`, or take another iteration vs skip) is checked: if no bool expression over the visible variables can realize that split, the split is dropped.
2. **Decide per-pair branching.** The conditions are still placeholders, so at each `while` check (does this pair iterate again, or exit?) and each `if/else` (which branch?) the search has to choose a split. Same realizability check as in phase 1: splits no bool expression could realize are dropped. Chosen splits get recorded so phase 3 can later find conditions that fit them.
3. **Fill in the conditions.** Once a program shape is complete (with placeholders where the bool conditions go), search for boolean expressions that fit all the recorded enter/skip and branch decisions. If no expression over the visible variables fits, fall back to **interdependent search**: hypothesize an auxiliary state variable (with its own initializer and per-iteration update), simulate it forward, and try again over the variables plus the hypothesized one.

A priority queue picks which partial program to expand next. Smaller programs go first; among equal-size programs, those whose replayed trace gets closer to the target. Ties broken depth-first (more recent state wins) so the search dives into a promising shape instead of jumping between siblings.

## Defining your own problem

Here's a problem where the output depends on parity: if `x0 > x1` return `x0 + x1`; else, return `x0 - x1`. Four examples, no seed.

```python
from core_lang_env.comp_env import Function, BoolFunction
from core_lang_env.parser import ast_to_code_str
from searchers.searchers_utils import Problem
from searchers.search_orchestrator import SearchOrchestrator

# Primitives the synthesizer can use in the body.
funcs = {
    "add": Function(lambda x, y: (x + y,), [int, int], [int]),
    "sub": Function(lambda x, y: (x - y,), [int, int], [int]),
}

# Primitives the synthesizer can use inside boolean conditions.
bools = {
    "gt": BoolFunction(lambda x, y: (x > y,), [int, int], [bool]),
    "not":     BoolFunction(lambda b: (not b,), [bool], [bool]),
}

problem = Problem(
    input_types=(int, int),
    output_types=(int,),
    instances={
        0: ((4, 3), (7,)),    # 4 > 3 → 4 + 3 = 7
        1: ((8, 2), (10,)),   # 8 > 2 → 8 + 2 = 10
        2: ((3, 5), (-2,)),   # 3 < 5 → 3 − 5 = −2
        3: ((1, 4), (-3,)),   # 1 < 4 → 1 − 4 = −3
    },
)

# Heuristic distance from current value to target.
def hdist(a, b):
    return abs(a - b) if isinstance(a, int) and isinstance(b, int) else 0

orch = SearchOrchestrator.create_new_orchestrator_from_problem(
    problem, funcs, bools, hdist, heuristic_cutoff=50,
    map_size=50, enable_while_loops=True,
)

for _ in range(50_000):
    if orch.search_queue.empty() or orch.completed_programs:
        break
    orch.step(trace_length_limit=10, max_ast_len=10)

if orch.completed_programs:
    print(ast_to_code_str(orch.completed_programs[0]))
```

A few notes:
- `Function`/`BoolFunction` wrap a Python function with its input/output types. Outputs are always tuples (so `lambda: (0,)` for a constant `zero`).
- `Problem.instances` maps an instance id to `(inputs_tuple, outputs_tuple)`.
- `hdist` is your heuristic for "how far is value `a` from target `b`". For ints, absolute difference works; for other types, return `0` and the search relies on reachability-graph distance alone.
- `bool_env.max_depth` caps how deep boolean expressions can nest. Default 2; bump higher when conditions need composed helpers like `not(eq(succ(x1), x0))`.
- `max_while` / `max_if` (kwargs to `orch.step`) cap how many `while` / `if-else` nodes the AST may contain. Useful for constraining the shape when you know it.
- Completed programs are full ASTs; `ast_to_code_str` pretty-prints them.

The `benchmarks/` directory has full driver scripts showing the same pattern with progress prints, seeded starts, and reporting.


## Quick start

```bash
pip install -r requirements.txt
python3 -m pytest                                            # 96 tests, ~5s
python3 benchmarks/bench_predecessor.py --mode scratch       # x − 1, 60 steps, 0.1s
python3 benchmarks/bench_multiply.py --mode scratch          # ~3 min, invents a counter
```

Each bench accepts `--mode {scratch,seeded,both}`, `--seed-level {tiny,body,full}`, `--max-steps N`, `--max-while N`, `--max-if N`, `--progress-every N`.


## Repository layout

```
core_lang_env/   AST types, parser, executor, basic computational primitives
searchers/      Orchestrator + search state + condition searcher (interdep search)
benchmarks/     Per-problem benchmark scripts (multiply, predecessor, sum_of_*)
tests/          Pytest suite (96 tests covering executor, AST, search, condition fill)
scripts/        One-off scripts (currently just a seeded-search example)
```

## Status

Research prototype. Active areas for future work: multi-dep-var schemes, learned priority and scheme priors, library learning across problems.

## License

Apache 2.0.
