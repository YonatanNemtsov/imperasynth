"""End-to-end execution tests — ported from notebooks/exec_example.ipynb.

`exec_code_v2` drives the synthesized program. These tests parse small code
blocks, execute them step-by-step on a `SimpleCompEnv`, and verify the final
return value matches what the program logically computes. They also check
that `AnnotatedAST` records the same action history as the env.
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
    execute_step,
    execute_step_with_annotation,
)
from core_lang_env.parser import parse_code_str

STEP_LIMIT = 1000  # guard against infinite loops if a test code is buggy


def _make_basic_env(*input_values):
    """Build a SimpleCompEnv with increment/add/another + gt/equal/is_even, given input ints."""
    increment = Function(lambda x: [x + 1], [int], [int])
    add = Function(lambda x, y: [x + y], [int, int], [int])
    gt = BoolFunction(lambda x, y: [x > y], [int, int], [bool])
    equal = BoolFunction(lambda x, y: [x == y], [int, int], [bool])
    is_even = BoolFunction(lambda x: [x % 2 == 0], [int], [bool])

    env = SimpleCompEnv()
    for v in input_values:
        env.add_input_object(CompObject(int, v))
    for name, f in [("increment", increment), ("add", add),
                    ("gt", gt), ("equal", equal), ("is_even", is_even)]:
        env.add_function(name, f)
    return env


def _run_to_completion(context: ExecutionContext, annot: AnnotatedAST | None = None):
    count = 0
    while not context.completed and count < STEP_LIMIT:
        if annot is None:
            execute_step(context)
        else:
            execute_step_with_annotation(context, annot)
        count += 1
    assert context.completed, f"execution did not complete within {STEP_LIMIT} steps"


def _solution_value(env: SimpleCompEnv):
    assert env.solution_object_id is not None, "no return statement was reached"
    return env.objects[env.solution_object_id].value


def test_simple_function_call_sequence():
    """x0 + 1 then return."""
    env = _make_basic_env(10)
    ast = parse_code_str("""
    {
        x1 = increment(x0);
        return x1;
    }
    """)
    var_names = {"x0": 0}
    context = ExecutionContext(ast, ExecutionPositionV2.start_position(), env, var_names, False)
    _run_to_completion(context)
    assert _solution_value(env) == 11


def test_if_branch_taken_when_condition_true():
    env = _make_basic_env(10, 3)  # x0=10, x1=3
    ast = parse_code_str("""
    {
        if (gt(x0, x1)) {
            x2 = add(x0, x1);
        } else {
            x2 = increment(x0);
        }
        return x2;
    }
    """)
    var_names = {"x0": 0, "x1": 1}
    context = ExecutionContext(ast, ExecutionPositionV2.start_position(), env, var_names, False)
    _run_to_completion(context)
    assert _solution_value(env) == 13  # 10 + 3 (if branch)


def test_else_branch_taken_when_condition_false():
    env = _make_basic_env(3, 10)  # x0=3, x1=10
    ast = parse_code_str("""
    {
        if (gt(x0, x1)) {
            x2 = add(x0, x1);
        } else {
            x2 = increment(x0);
        }
        return x2;
    }
    """)
    var_names = {"x0": 0, "x1": 1}
    context = ExecutionContext(ast, ExecutionPositionV2.start_position(), env, var_names, False)
    _run_to_completion(context)
    assert _solution_value(env) == 4  # increment(3)


def test_while_loop_runs_until_condition_false():
    """Increment x0 until it equals x1.

    Starting state: x0=3, x1=7. Loop: while gt(x1, x0) { x0 = increment(x0); }
    After loop: x0=7, x1=7 (loop exits because gt(7, 7) is false). Return x0.
    """
    env = _make_basic_env(3, 7)
    ast = parse_code_str("""
    {
        while (gt(x1, x0)) {
            x0 = increment(x0);
        }
        return x0;
    }
    """)
    var_names = {"x0": 0, "x1": 1}
    context = ExecutionContext(ast, ExecutionPositionV2.start_position(), env, var_names, False)
    _run_to_completion(context)
    assert _solution_value(env) == 7


def test_notebook_example_mixed_while_if():
    """The first example from notebooks/exec_example.ipynb.

    Inputs: x0=10, x1=3.
      x0 = increment(x0);              -> x0 = 11
      while (gt(x0, x1)) { x1 = increment(x1); }   -> x1 grows to 11
      if (gt(x1, x0)) { ... } else { x2 = add(x0, x1); return x2; }
        -> gt(11, 11) is false, take else: x2 = 11 + 11 = 22, return 22.
    """
    env = _make_basic_env(10, 3)
    ast = parse_code_str("""
    {
        x0 = increment(x0);
        while (gt(x0, x1)) {
            x1 = increment(x1);
        }
        if (gt(x1, x0)) {
            x0 = increment(x0);
        }
        else {
            x2 = add(x0, x1);
            return x2;
        }
        return x0;
    }
    """)
    var_names = {"x0": 0, "x1": 1}
    context = ExecutionContext(ast, ExecutionPositionV2.start_position(), env, var_names, False)
    _run_to_completion(context)
    assert _solution_value(env) == 22


def test_direct_assign_swap():
    """x0, x1 <- x1, x0; reverses the two."""
    env = _make_basic_env(7, 13)
    ast = parse_code_str("""
    {
        x0, x1 <- x1, x0;
        return x0;
    }
    """)
    var_names = {"x0": 0, "x1": 1}
    context = ExecutionContext(ast, ExecutionPositionV2.start_position(), env, var_names, False)
    _run_to_completion(context)
    assert _solution_value(env) == 13  # was originally x1


def test_annotated_ast_records_actions():
    """After running with annotation, env.action_history matches annot.get_action_history()."""
    env = _make_basic_env(3, 7)
    ast = parse_code_str("""
    {
        while (gt(x1, x0)) {
            x0 = increment(x0);
        }
        x2 = add(x0, x1);
        return x2;
    }
    """)
    var_names = {"x0": 0, "x1": 1}
    context = ExecutionContext(ast, ExecutionPositionV2.start_position(), env, var_names, False)
    annot = AnnotatedAST(ast, CompSignature([]), {}, var_names.copy())
    _run_to_completion(context, annot)

    # Both views should agree on the function-call actions in order. Annot also
    # records the synthetic RETURN action at the end, so trim that for comparison.
    annot_actions_no_return = [
        a for a in annot.get_action_history() if a[0] != "RETURN_FUNC_NAME"
    ]
    assert annot_actions_no_return == env.action_history_short
    assert _solution_value(env) == 14  # 7 + 7


def test_empty_block_in_else_does_not_crash():
    """If/else with an empty else block executes cleanly."""
    env = _make_basic_env(5)
    ast = parse_code_str("""
    {
        if (is_even(x0)) {
            x0 = increment(x0);
        } else {}
        return x0;
    }
    """)
    var_names = {"x0": 0}
    context = ExecutionContext(ast, ExecutionPositionV2.start_position(), env, var_names, False)
    _run_to_completion(context)
    assert _solution_value(env) == 5  # 5 is odd, else branch (empty), x0 unchanged
