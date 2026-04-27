"""Utilities used across the searcher layer — ported from notebooks/searchers_utils.ipynb.

Covers:
  - compute_block_hash / compute_func_call_hash: deterministic AST hashing
  - get_unused_variables_in_ast: simple liveness analysis
  - get_variables_defined_in_node
  - find_minimal_steps_in_comp_map: A* heuristic distance
  - get_*_exec_positions: position helpers (require an executed AnnotatedAST)
"""
from core_lang_env.comp_env import (
    BoolFunction,
    CompObject,
    CompSignature,
    Function,
    SimpleCompEnv,
)
from core_lang_env.exec_code_v2 import (
    AnnotatedAST,
    ExecutionContext,
    ExecutionPositionV2,
    execute_step_with_annotation,
)
from core_lang_env.parser import parse_code_str
from searchers_utils import (
    SimpleMapper,
    compute_block_hash,
    compute_func_call_hash,
    compute_func_call_output_hash,
    extract_minimal_subgraph,
    find_minimal_steps_in_comp_map,
    get_if_else_node_exec_positions,
    get_node_exec_positions,
    get_unused_variables_in_ast,
    get_variables_defined_in_node,
    get_while_node_exec_positions,
)


# ---------- AST hashing ----------

def test_compute_block_hash_is_deterministic():
    code = """
    {
        x2 = f(x0, x1);
        return x2;
    }
    """
    ast1 = parse_code_str(code)
    ast2 = parse_code_str(code)
    h1, _ = compute_block_hash(ast1, {"x0": 0, "x1": 1})
    h2, _ = compute_block_hash(ast2, {"x0": 0, "x1": 1})
    assert h1 == h2


def test_compute_block_hash_changes_with_program():
    h1, _ = compute_block_hash(parse_code_str("{ x2 = f(x0, x1); return x2; }"), {"x0": 0, "x1": 1})
    h2, _ = compute_block_hash(parse_code_str("{ x2 = g(x0, x1); return x2; }"), {"x0": 0, "x1": 1})
    assert h1 != h2


def test_compute_block_hash_changes_with_input_hashes():
    """Same code, different initial variable hashes ⇒ different block hash."""
    ast = parse_code_str("{ x2 = f(x0, x1); return x2; }")
    h1, _ = compute_block_hash(ast, {"x0": 0, "x1": 1})
    h2, _ = compute_block_hash(ast, {"x0": 99, "x1": 100})
    assert h1 != h2


def test_compute_func_call_output_hash_changes_with_index():
    """Different output indices for the same call yield distinct hashes."""
    inp_hashes = {"x0": 0, "x1": 1}
    h0 = compute_func_call_output_hash(0, "f", ("x0", "x1"), inp_hashes)
    h1 = compute_func_call_output_hash(1, "f", ("x0", "x1"), inp_hashes)
    assert h0 != h1


def test_compute_func_call_hash_independent_of_output_index():
    """compute_func_call_hash hashes the call (func + inputs); it does NOT include
    output index, while compute_func_call_output_hash does. So the two hashes
    are necessarily different."""
    ast = parse_code_str("{ x2 = f(x0, x1); return x2; }")
    func_call = ast.children[0]
    inp_hashes = {"x0": 0, "x1": 1}

    call_hash = compute_func_call_hash(func_call, inp_hashes)
    output_hash = compute_func_call_output_hash(0, "f", ("x0", "x1"), inp_hashes)
    assert call_hash != output_hash  # different by construction
    # But each is deterministic.
    assert compute_func_call_hash(func_call, inp_hashes) == call_hash


# ---------- Variable analysis ----------

def test_get_unused_variables_finds_dead_definitions():
    """x7 is computed but never read."""
    ast = parse_code_str("""
    {
        x2 = f(x0, x1);
        if (TBD()) {
            x3 = g(x2);
            x10 = f(x3);
        } else {
            x4 = g(x1);
            x7 = h(x3);
            x3 <- x4;
        }
        return x10;
    }
    """)
    unused = get_unused_variables_in_ast(ast)
    assert "x7" in unused


def test_get_unused_variables_empty_when_all_live():
    ast = parse_code_str("""
    {
        x2 = add(x0, x1);
        return x2;
    }
    """)
    assert get_unused_variables_in_ast(ast) == []


def test_get_variables_defined_in_block():
    ast = parse_code_str("""
    {
        x2 = f(x0, x1);
        x3 = g(x2);
        return x3;
    }
    """)
    defined = get_variables_defined_in_node(ast, ())
    # x2, x3 are defined inside this block (return doesn't define).
    assert "x2" in defined
    assert "x3" in defined


# ---------- Heuristic distance over a comp_map ----------

def _make_comp_map_for_target(inp_types, inputs, target_type, target, funcs):
    """Build a ComputationalMap by running SimpleMapper to reach target.

    Note: SimpleMapper.search has a buggy default `heuristic_cutoff='inf'` (a string),
    so every caller must pass `float('inf')` explicitly to disable pruning.
    """
    hdist = lambda a, b: abs(a - b) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else 0
    mapper = SimpleMapper(inp_types, inputs, target_type, target, funcs, hdist)
    mapper.search(max_steps=200, heuristic_cutoff=float("inf"))
    return extract_minimal_subgraph(mapper.comp_map, target)


def test_find_minimal_steps_zero_when_target_is_input():
    add = Function(lambda x, y: [x + y], [int, int], [int])
    cmap = _make_comp_map_for_target((int, int), (3, 4), int, 7, {"add": add})
    # Target is already among inputs?
    assert find_minimal_steps_in_comp_map(cmap, [7, 0], 7) == 0


def test_find_minimal_steps_one_for_single_action():
    """add(3, 4) reaches 7 in one step."""
    add = Function(lambda x, y: [x + y], [int, int], [int])
    cmap = _make_comp_map_for_target((int, int), (3, 4), int, 7, {"add": add})
    assert find_minimal_steps_in_comp_map(cmap, [3, 4], 7) == 1


def test_find_minimal_steps_inf_when_unreachable():
    """No way to reach 100 from {3, 4} using only `add`."""
    add = Function(lambda x, y: [x + y], [int, int], [int])
    cmap = _make_comp_map_for_target((int, int), (3, 4), int, 7, {"add": add})
    # Target 100 isn't even in the cmap — should be infinity.
    assert find_minimal_steps_in_comp_map(cmap, [3, 4], 100) == float("inf")


# ---------- Execution position helpers ----------
# These need an executed AnnotatedAST. Build one fresh per test.

def _build_executed_annot(code_str, var_names, input_values, step_limit=2000):
    increment = Function(lambda x: [x + 1], [int], [int])
    add = Function(lambda x, y: [x + y], [int, int], [int])
    gt = BoolFunction(lambda x, y: [x > y], [int, int], [bool])
    is_even = BoolFunction(lambda x: [x % 2 == 0], [int], [bool])

    env = SimpleCompEnv()
    for v in input_values:
        env.add_input_object(CompObject(int, v))
    for name, f in [("increment", increment), ("add", add), ("gt", gt), ("is_even", is_even)]:
        env.add_function(name, f)

    ast = parse_code_str(code_str)
    context = ExecutionContext(ast, ExecutionPositionV2.start_position(), env, var_names.copy(), False)
    annot = AnnotatedAST(ast, CompSignature([]), {}, var_names.copy())

    count = 0
    while not context.completed and count < step_limit:
        execute_step_with_annotation(context, annot)
        count += 1
    return ast, annot


def test_get_node_exec_positions_records_loop_iterations():
    """Each iteration of a while loop body produces one entry."""
    ast, annot = _build_executed_annot(
        """
        {
            while (gt(x1, x0)) {
                x0 = increment(x0);
            }
            return x0;
        }
        """,
        var_names={"x0": 0, "x1": 1},
        input_values=[3, 7],
    )
    # The while loop body is at position (0, 1) — child 1 of the while node at (0,).
    body_positions = get_node_exec_positions(annot, (0, 1))
    # 4 iterations: 3→4→5→6→7. So 4 entries into the body.
    assert len(body_positions["entries"]) == 4


def test_get_while_node_exec_positions_has_one_skip_per_loop_run():
    """For a single top-level while loop run, exactly one skip event is recorded
    (the iteration where the condition becomes false)."""
    ast, annot = _build_executed_annot(
        """
        {
            while (gt(x1, x0)) {
                x0 = increment(x0);
            }
            return x0;
        }
        """,
        var_names={"x0": 0, "x1": 1},
        input_values=[3, 7],
    )
    while_positions = get_while_node_exec_positions(annot, (0,))
    assert len(while_positions["skips"]) == 1


def test_get_if_else_node_exec_positions_distinguishes_branches():
    """When the condition is false, the IF block has zero entries and one skip."""
    ast, annot = _build_executed_annot(
        """
        {
            if (gt(x0, x1)) {
                x0 = increment(x0);
            } else {}
            return x0;
        }
        """,
        var_names={"x0": 0, "x1": 1},
        input_values=[3, 7],  # 3 > 7 is false → else branch
    )
    if_else_positions = get_if_else_node_exec_positions(annot, (0,))
    assert if_else_positions["entries"] == []  # IF block never entered
    assert len(if_else_positions["skips"]) == 1
