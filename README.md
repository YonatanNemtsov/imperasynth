# Imperative Program Synthesis from Examples

Synthesizes SSA-like imperative programs from input/output pairs and a set of primitive functions. [Examples below.](#examples)

## Examples

Each problem is defined by:
- a set of primitive functions (typed; e.g. `add: int×int → int`),
- a set of boolean primitives for conditions (e.g. `eq`, `not`, `is_empty`),
- a handful of input/output pairs.

### Predecessor

Given `succ` and a small number of `(x0, 0) → (x0 − 1)` pairs, the synthesizer produces:

```
{
    b0 = succ(x1);
    b1 = eq(b0, x0);
    while (not(b1)) {
        x2 = succ(x1);
        x1 <- x2;
        b0 = succ(x1);
        b1 = eq(b0, x0);
    }
    return x1;
}
```

50 steps, 0.1s.

### Sum of evens

Given `add`, `get_head`, `get_tail`, `zero`, `identity` plus `is_empty`, `is_even`, `not`, the synthesizer produces:

```
{
    x1 = zero();
    b0 = is_empty(x0);
    while (not(b0)) {
        x2 = get_head(x0);
        x3 = get_tail(x0);
        if (is_even(x2)) {
            x4 = add(x1, x2);
        } else {
            x4 <- x1;
        }
        x0, x1 <- x3, x4;
        b0 = is_empty(x0);
    }
    return x1;
}
```

40K steps from a partial body seed, 27s. Shows `if/else` nested inside `while`.

### Multiplication

Given `add` and `zero` for the body plus `zero`, `succ`, `eq`, `not` for conditions, with pairs `(3,2)→6, (4,3)→12, (2,5)→10`, the synthesizer produces:

```
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

`c0` is not among the problem inputs and isn't reachable from them through `funcs` alone — the synthesizer hypothesized that a state variable initialized with `zero()` and updated each iteration with `succ` would let it express the exit condition. ~32K search steps, ~3 minutes from an empty initial AST.

## What it can / can't do

**Can:**
- Counter-driven `while` loops (multiplication, predecessor by `succ` increment).
- List traversals (sum-of-list, sum-of-evens with `if/else` inside `while`).
- Single `if/else` conditionals (max-of-two, double-or-diff).
- Invent a single auxiliary state variable as part of bool search (n=1 dep var).

**Can't (yet):**
- Recursion as a primitive — programs are iterative only.
- Multiple invented state variables (n=2+ schemes are designed but not implemented).
- Neural / learned priors (the search is pure cost-based enumeration with cmap pruning).
- Library learning across problems.

## Quick start

```bash
pip install -r requirements.txt
python3 -m pytest                                            # 96 tests, ~5s
python3 benchmarks/bench_predecessor.py --mode scratch       # x − 1, 60 steps, 0.1s
python3 benchmarks/bench_multiply.py --mode scratch          # ~3 min, invents a counter
```

Each bench accepts `--mode {scratch,seeded,both}`, `--seed-level {tiny,body,full}`, `--max-steps N`, `--max-while N`, `--max-if N`, `--progress-every N`.

## Benchmark numbers

| Problem | Seed | Steps | Time | Notes |
|---|---|---|---|---|
| predecessor | scratch | 60 | 0.1s | ✓ |
| predecessor | full | 50 | 0.1s | ✓ |
| sum_of_evens | full | 71 | 0.03s | ✓ |
| sum_of_evens | body | 40,463 | 27s | ✓ |
| sum_of_evens | tiny | — | — | doesn't converge |
| sum_of_evens | scratch | — | — | doesn't converge |
| multiply | full | 125 | 0.55s | ✓, `--max-while 1 --max-if 0` |
| multiply | scratch | ~32K | ~3 min | ✓, same constraints |
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

Here's a problem where the output depends on parity: if `x0` is even, return `x0 + x1`; if odd, return `x0 - x1`. Four examples, no seed.

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
    "is_even": BoolFunction(lambda x: (x % 2 == 0,), [int], [bool]),
    "not":     BoolFunction(lambda b: (not b,), [bool], [bool]),
}

problem = Problem(
    input_types=(int, int),
    output_types=(int,),
    instances={
        0: ((4, 3), (7,)),    # 4 is even → 4 + 3
        1: ((2, 5), (7,)),    # 2 is even → 2 + 5
        2: ((3, 8), (-5,)),   # 3 is odd  → 3 − 8
        3: ((7, 1), (6,)),    # 7 is odd  → 7 − 1
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
