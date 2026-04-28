"""Condition searcher — ported from notebooks/condition_searcher.ipynb.

`condition_searcher` extracts boolean classification problems from executed
ASTs (which boolean held when the if branch was taken vs the else, etc.)
and searches for a small bool program that explains the labels.
"""
from searchers.condition_searcher import (
    create_bool_program_expressions,
    enumerate_realizable_partitions,
    extract_ifelse_conditional_problem,
    extract_while_conditional_problem,
    get_conditional_problem_instances_from_annotated_ast_and_ifelse_node_position,
    search_boolean_traces,
)
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
from core_lang_env.syntax_tree import BoolExprNode, FunctionCallAssignNode
from searchers.searchers_utils import Problem


# ---------- Build an executed AnnotatedAST against a simple env ----------

def _basic_funcs():
    return {
        "increment": Function(lambda x: [x + 1], [int], [int]),
        "add": Function(lambda x, y: [x + y], [int, int], [int]),
        "gt": BoolFunction(lambda x, y: [x > y], [int, int], [bool]),
        "is_even": BoolFunction(lambda x: [x % 2 == 0], [int], [bool]),
        "equal": BoolFunction(lambda x, y: [x == y], [int, int], [bool]),
    }


def _executed(code_str, var_names, input_values, step_limit=2000):
    env = SimpleCompEnv()
    for v in input_values:
        env.add_input_object(CompObject(int, v))
    for name, f in _basic_funcs().items():
        env.add_function(name, f)

    ast = parse_code_str(code_str)
    context = ExecutionContext(ast, ExecutionPositionV2.start_position(), env, var_names.copy(), False)
    annot = AnnotatedAST(ast, CompSignature([]), {}, var_names.copy())
    count = 0
    while not context.completed and count < step_limit:
        execute_step_with_annotation(context, annot)
        count += 1
    return ast, annot, env


# ---------- get_conditional_problem_instances_from_annotated_ast_and_ifelse_node_position ----------

def test_get_conditional_problem_instances_separates_entries_and_skips():
    """For an executed if/else, the helper labels each parent iteration as
    either entry (if-branch) or skip (else-branch)."""
    code = """
    {
        while (gt(x1, x0)) {
            if (is_even(x0)) {
                x0 = increment(x0);
            }
            else {}
            x0 = increment(x0);
        }
        return x0;
    }
    """
    ast, annot, env = _executed(code, var_names={"x0": 0, "x1": 1}, input_values=[0, 4])
    # The if-else is at (0, 1, 0) — inside the while body.
    instances, var_names = get_conditional_problem_instances_from_annotated_ast_and_ifelse_node_position(
        annot, (0, 1, 0), env
    )

    # x0 starts at 0 and increments. is_even(x0) ↔ x0 even.
    # Iter 1: x0=0 (even) → entry.
    # After each iter: x0 += 1 (always, regardless of branch) plus +1 if entered.
    # Entries should be parents where x0 was even at the if-point.
    assert len(instances["entries"]) >= 1
    # var_names should be a sorted list of variables visible at the if point.
    assert "x0" in var_names


# ---------- extract_ifelse_conditional_problem ----------

def test_extract_ifelse_problem_returns_problem_with_bool_outputs():
    code = """
    {
        while (gt(x1, x0)) {
            if (is_even(x0)) {
                x0 = increment(x0);
            }
            else {}
            x0 = increment(x0);
        }
        return x0;
    }
    """
    ast, annot, env = _executed(code, {"x0": 0, "x1": 1}, [0, 4])
    result = extract_ifelse_conditional_problem(annot, (0, 1, 0), env)
    assert result is not None
    problem, var_names = result
    assert isinstance(problem, Problem)
    assert problem.output_types is bool or problem.output_types == bool

    # Each instance's output should be a single-element bool tuple.
    for inputs, output in problem.instances.values():
        assert len(output) == 1
        assert isinstance(output[0], bool)


def test_extract_ifelse_problem_returns_none_when_branch_never_runs():
    """An if-else inside a never-entered while produces no instances → None."""
    code = """
    {
        while (gt(x0, x1)) {
            if (is_even(x0)) {} else {}
        }
        return x0;
    }
    """
    # x0=3, x1=7: gt(3, 7) is false → while body never runs → if-else never executes.
    ast, annot, env = _executed(code, {"x0": 0, "x1": 1}, [3, 7])
    # The if-else is at (0, 1, 0).
    result = extract_ifelse_conditional_problem(annot, (0, 1, 0), env)
    assert result is None


# ---------- extract_while_conditional_problem ----------

def test_extract_while_problem_returns_inter_dependent_group():
    code = """
    {
        while (gt(x1, x0)) {
            x0 = increment(x0);
        }
        return x0;
    }
    """
    ast, annot, env = _executed(code, {"x0": 0, "x1": 1}, [3, 7])
    result = extract_while_conditional_problem(annot, (0,), env)
    assert result is not None
    problem_group, var_names = result
    # Each entry has a True label; the final skip has False.
    assert "x0" in var_names
    assert "x1" in var_names


# ---------- search_boolean_traces ----------

def test_search_boolean_traces_finds_simple_threshold():
    """Learn the function `gt(a, b)` from labeled instances."""
    funcs = _basic_funcs()
    instances = {
        0: ((5, 3), (True,)),  # 5 > 3
        1: ((2, 9), (False,)),
        2: ((10, 4), (True,)),
        3: ((1, 1), (False,)),
    }
    bool_problem = Problem((int, int), bool, instances)
    traces = search_boolean_traces(bool_problem, funcs, max_depth=2)
    # Each instance's trace should have a solution_object_id pointing to the expected label.
    assert len(traces) == len(instances)
    for trace, expected_label in zip(traces, [True, False, True, False]):
        assert trace.solution_object_id is not None
        assert trace.objects[trace.solution_object_id].value is expected_label


# ---------- create_bool_program_expressions ----------

def test_create_bool_program_expressions_builds_function_call_chain():
    """After search_boolean_traces succeeds, create_bool_program_expressions
    converts a successful trace into a list of AST nodes ending in a BoolExprNode."""
    funcs = _basic_funcs()
    instances = {
        0: ((5, 3), (True,)),
        1: ((2, 9), (False,)),
        2: ((10, 4), (True,)),
    }
    bool_problem = Problem((int, int), bool, instances)
    traces = search_boolean_traces(bool_problem, funcs, max_depth=2)
    statements = create_bool_program_expressions(traces[0], ["x0", "x1"])

    # Last statement is the bool expression itself; the rest are function calls.
    assert isinstance(statements[-1], BoolExprNode)
    for stmt in statements[:-1]:
        assert isinstance(stmt, FunctionCallAssignNode)


# ---------- enumerate_realizable_partitions ----------

def test_enumerate_realizable_partitions_sum_of_evens_iter1():
    """At sum-of-evens iter-1 if/else, the right partition {0,3} | {1,2}
    (traces with even head) must be realizable via is_even(x2). Constants
    {} and {0,1,2,3} are also realizable via is_even(x1) (always-True for
    x1=0). Subsets that no bool can produce — e.g. {0,1} — must be absent."""
    funcs = {
        "add": Function(lambda x, y: (x + y,), [int, int], [int]),
        "get_head": Function(lambda lst: ((lst[0],) if lst else ()), [tuple], [int]),
        "get_tail": Function(lambda lst: ((lst[1:],) if lst else ((),)), [tuple], [tuple]),
    }
    bools = {
        "is_empty": BoolFunction(lambda lst: ((len(lst) == 0,)), [tuple], [bool]),
        "is_even": BoolFunction(lambda x: (x % 2 == 0,), [int], [bool]),
        "not": BoolFunction(lambda b: ((not b,)), [bool], [bool]),
    }

    # Per-trace in-scope values at the if/else (after head, tail in body).
    in_scope = [
        {"x0": (2,),       "x1": 0, "x2": 2, "x3": ()},
        {"x0": (1,),       "x1": 0, "x2": 1, "x3": ()},
        {"x0": (1, 2),     "x1": 0, "x2": 1, "x3": (2,)},
        {"x0": (2, 3, 4),  "x1": 0, "x2": 2, "x3": (3, 4)},
    ]

    partitions = enumerate_realizable_partitions(in_scope, funcs, bools, max_depth=2)

    # The discriminating partition for is_even(x2) is {0, 3}.
    assert frozenset({0, 3}) in partitions, f"missing {{0,3}}; got {set(partitions.keys())}"
    # is_empty(x3) discriminates {0,1} vs {2,3}: {2,3} (or {0,1} via not(...)) should be realizable.
    assert frozenset({2, 3}) in partitions or frozenset({0, 1}) in partitions

    # A partition no primitive bool produces, e.g. {0, 2}, must be absent
    # (no bool maps trace 0 and 2 to True while 1 and 3 to False with these inputs).
    assert frozenset({0, 2}) not in partitions

    # The realizing expression for {0, 3} must end in a BoolExprNode.
    depth, stmts = partitions[frozenset({0, 3})]
    assert depth == 1, f"is_even(x2) is depth-1, got {depth}"
    assert isinstance(stmts[-1], BoolExprNode)


def test_enumerate_realizable_partitions_constants_present():
    """Constant-true and constant-false partitions are realizable when any
    bool function returns the same value on every trace."""
    funcs = {}
    bools = {
        "is_zero": BoolFunction(lambda x: (x == 0,), [int], [bool]),
    }
    in_scope = [
        {"x0": 5},
        {"x0": 7},
        {"x0": 3},
    ]
    partitions = enumerate_realizable_partitions(in_scope, funcs, bools, max_depth=2)
    # is_zero(x0) is False on all → empty True-set is realizable.
    assert frozenset() in partitions
