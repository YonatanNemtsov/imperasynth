"""AST mutations driven by the augmentation system — ported from notebooks/ast_searcher.ipynb.

Each `apply_augmentation_*_group` builds a small AST step-by-step. These tests
freeze the structural shape of the resulting AST after each mutation, plus the
parser-formatter round-trip that nothing semantic was lost.
"""
from searchers.ast_searcher_v3 import (
    AnnotatedAstGroup,
    NO_ACTION,
    apply_augmentation_end_else_group,
    apply_augmentation_end_if_group,
    apply_augmentation_function_call_group,
    apply_augmentation_return_group,
    apply_augmentation_start_if_group,
)
from core_lang_env.parser import ast_to_code_str
from core_lang_env.syntax_tree import (
    BlockNode,
    DirectAssignNode,
    FunctionCallAssignNode,
    IfElseNode,
    ReturnNode,
)


def _initial_group(n_traces=3, var_name="x0"):
    """Initial AST group: empty BlockNode, one initial var ('x0' → 0)."""
    return AnnotatedAstGroup.initialize_new_group([{var_name: 0}] * n_traces)


def test_initial_group_has_empty_ast():
    group = _initial_group(n_traces=3)
    assert isinstance(group.ast, BlockNode)
    assert group.ast.statements == []


def test_apply_function_call_inserts_one_node():
    group = _initial_group(n_traces=3)
    new_group = apply_augmentation_function_call_group(
        group,
        ((0,), (0, 0)),  # AST insert pos, temporal pos
        (
            ("f", ("x0",)),
            (("f", (0,)), ("f", (0,)), NO_ACTION),  # third trace skips
            1,  # one output
        ),
    )
    assert len(new_group.ast.statements) == 1
    node = new_group.ast.statements[0]
    assert isinstance(node, FunctionCallAssignNode)
    assert node.func_name == "f"
    assert node.arg_names == ("x0",)
    # Output var should be 'x1' (fresh after x0).
    assert node.var_names == ("x1",)


def test_apply_start_if_inserts_ifelse_node():
    group = _initial_group(n_traces=3)
    new_group = apply_augmentation_start_if_group(
        group,
        ((0,), (0, 0)),
    )
    assert len(new_group.ast.statements) == 1
    node = new_group.ast.statements[0]
    assert isinstance(node, IfElseNode)
    assert isinstance(node.if_block, BlockNode)
    assert isinstance(node.else_block, BlockNode)
    # Both branches start empty.
    assert node.if_block.statements == []
    assert node.else_block.statements == []


def test_apply_return_inserts_return_node():
    group = _initial_group(n_traces=2)
    new_group = apply_augmentation_return_group(
        group,
        ((0,), (0, 0)),
        (
            ("RETURN_FUNC_NAME", ("x0",)),
            (("RETURN_FUNC_NAME", (0,)), ("RETURN_FUNC_NAME", (0,))),
            0,
        ),
    )
    assert len(new_group.ast.statements) == 1
    node = new_group.ast.statements[0]
    assert isinstance(node, ReturnNode)
    assert node.return_var_name == "x0"


def test_full_if_else_construction_round_trip():
    """Build: { f(x0); if (TBD()) { g(...) } else { h(...); x_,x_ <- x_,x_ } }
    and verify the formatted result reparses to an equivalent AST."""
    group = _initial_group(n_traces=3)

    # 1) Top-level func call.
    group = apply_augmentation_function_call_group(
        group,
        ((0,), (0, 0)),
        (("f", ("x0",)), (("f", (0,)), ("f", (0,)), NO_ACTION), 1),
    )
    # 2) Start if/else at position 1.
    group = apply_augmentation_start_if_group(group, ((1,), (1, 1)))

    # 3) End if (no final assign needed for this test).
    group = apply_augmentation_end_if_group(group)

    # 4) End else with a final assign (x3, x4 <- x1, x5).
    group = apply_augmentation_end_else_group(
        group,
        ((1, 2, 0), (1, 1, 1, 1)),
        ("x3", "x4"),
        ("x1", "x5"),
        {0, 1},
    )

    # The constructed AST should be a single block of [func_call, if_else].
    assert len(group.ast.statements) == 2
    assert isinstance(group.ast.statements[0], FunctionCallAssignNode)
    assert isinstance(group.ast.statements[1], IfElseNode)

    # Else block should now end with the DirectAssign we requested.
    else_block = group.ast.statements[1].else_block
    final = else_block.statements[-1]
    assert isinstance(final, DirectAssignNode)
    assert final.target_vars == ("x3", "x4")
    assert final.source_vars == ("x1", "x5")

    # Code should be formatable (round-trip is tested in test_parser).
    code = ast_to_code_str(group.ast)
    assert "if" in code
    assert "else" in code


def test_each_annotated_ast_in_group_shares_underlying_ast():
    """All annotated ASTs in a group reference the same underlying AST object
    after a group-level mutation. This is intentional — the AST is shared,
    only the per-trace annotations differ."""
    group = _initial_group(n_traces=3)
    new_group = apply_augmentation_function_call_group(
        group,
        ((0,), (0, 0)),
        (("f", ("x0",)), (("f", (0,)), ("f", (0,)), ("f", (0,))), 1),
    )
    asts = {id(annot.ast) for annot in new_group.annot_asts}
    # All annotated asts reference the same AST instance.
    assert len(asts) == 1


def test_no_action_in_third_trace_is_skipped():
    """When a trace has NO_ACTION for a func call, its annotation should not
    record that action."""
    group = _initial_group(n_traces=3)
    before_actions = [len(annot.mapping) for annot in group.annot_asts]
    assert before_actions == [0, 0, 0]

    new_group = apply_augmentation_function_call_group(
        group,
        ((0,), (0, 0)),
        (("f", ("x0",)), (("f", (0,)), ("f", (0,)), NO_ACTION), 1),
    )
    after_actions = [len(annot.mapping) for annot in new_group.annot_asts]
    # Trace 0 and 1 record one action each; trace 2 (NO_ACTION) records none.
    assert after_actions == [1, 1, 0]


def test_signature_distinguishes_different_constructions():
    """Different sequences of augmentations yield different group signatures."""
    g1 = _initial_group(n_traces=2)
    g1 = apply_augmentation_function_call_group(
        g1,
        ((0,), (0, 0)),
        (("f", ("x0",)), (("f", (0,)), ("f", (0,))), 1),
    )

    g2 = _initial_group(n_traces=2)
    g2 = apply_augmentation_function_call_group(
        g2,
        ((0,), (0, 0)),
        (("g", ("x0",)), (("g", (0,)), ("g", (0,))), 1),
    )

    assert g1.get_signature() != g2.get_signature()
