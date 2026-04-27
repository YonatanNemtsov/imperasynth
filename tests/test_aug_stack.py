"""Augmentation stack transitions — ported from notebooks/ast_searcher.ipynb.

The stack drives AST construction. Each transition pops one frontier and
pushes zero-or-more new ones. These tests freeze the stack-length deltas
expected for each transition kind, plus the full sequences for if/else
and while-loop construction.
"""
from ast_searcher_v3 import (
    AUG_END_ELSE,
    AUG_END_IF,
    AUG_END_WHILE,
    AUG_ENTER_WHILE,
    AUG_EXECUTE_BLOCK,
    AUG_EXECUTE_DIRECT_ASSIGN,
    AUG_EXECUTE_FUNC_CALL,
    AUG_FUNC_CALL,
    AUG_RETURN,
    AUG_SKIP_WHILE,
    AUG_START_IF,
    AUG_START_WHILE,
    AugmentationStack,
)
from core_lang_env.syntax_tree import DirectAssignNode, FunctionCallAssignNode


def test_init_stack_has_one_frontier():
    stack = AugmentationStack.init_new_stack(all_indices={0, 1, 2})
    assert len(stack.stack) == 1


def test_func_call_keeps_stack_length():
    stack = AugmentationStack.init_new_stack(all_indices={0, 1, 2})
    stack.apply_transition(AUG_FUNC_CALL, {0, 1, 2}, None)
    assert len(stack.stack) == 1


def test_start_if_pushes_three_frontiers():
    """START_IF pops one frontier and pushes three (parent-after, else, if)."""
    stack = AugmentationStack.init_new_stack(all_indices={0, 1, 2})
    stack.apply_transition(AUG_START_IF, {0, 1}, None)
    assert len(stack.stack) == 3


def test_full_if_else_sequence_empties_stack():
    """A complete if/else/return program drains the stack to zero."""
    stack = AugmentationStack.init_new_stack(all_indices={0, 1, 2})
    stack.apply_transition(AUG_FUNC_CALL, {0, 1, 2}, None)
    stack.apply_transition(AUG_START_IF, {0, 1}, None)
    stack.apply_transition(AUG_FUNC_CALL, {0, 1}, None)
    stack.apply_transition(AUG_END_IF, {0, 1}, None)
    stack.apply_transition(AUG_FUNC_CALL, {2}, None)
    stack.apply_transition(AUG_END_ELSE, {2}, None)
    stack.apply_transition(AUG_RETURN, {0, 1, 2}, None)
    assert stack.stack == []


def test_while_construction_phase():
    """START_WHILE → FUNC_CALL → END_WHILE: building the loop body."""
    stack = AugmentationStack.init_new_stack(all_indices={0, 1, 2})
    stack.apply_transition(AUG_START_WHILE, {0, 1}, None)
    # START_WHILE pushes parent, executing-while, building-while frontiers.
    assert len(stack.stack) == 3
    stack.apply_transition(AUG_FUNC_CALL, {0, 1}, None)  # inside building-while
    assert len(stack.stack) == 3
    stack.apply_transition(AUG_END_WHILE, {0, 1}, None)
    # END_WHILE pops the building frontier, pushes nothing.
    assert len(stack.stack) == 2


def test_while_execution_phase():
    """ENTER_WHILE → EXECUTE_BLOCK → execute body nodes → SKIP_WHILE."""
    stack = AugmentationStack.init_new_stack(all_indices={0, 1, 2})
    stack.apply_transition(AUG_START_WHILE, {0, 1}, None)
    stack.apply_transition(AUG_FUNC_CALL, {0, 1}, None)
    stack.apply_transition(AUG_END_WHILE, {0, 1}, None)

    # Now in the execution phase. Top frontier offers ENTER_WHILE / SKIP_WHILE.
    stack.apply_transition(AUG_ENTER_WHILE, {0, 1}, None)
    block_nodes = [FunctionCallAssignNode, DirectAssignNode]
    stack.apply_transition(AUG_EXECUTE_BLOCK, {0, 1}, block_nodes)
    stack.apply_transition(AUG_EXECUTE_FUNC_CALL, {0, 1}, None)
    stack.apply_transition(AUG_EXECUTE_DIRECT_ASSIGN, {0, 1}, None)

    # After executing the body, we're back to ENTER/SKIP. Skip out for {0, 1}.
    stack.apply_transition(AUG_SKIP_WHILE, {0, 1}, None)

    # Parent frontier remains for the rest of the program.
    assert len(stack.stack) == 1


def test_nested_while_with_if_construction():
    """A while containing an if/else builds without errors and ends in a clean stack."""
    stack = AugmentationStack.init_new_stack(all_indices={0, 1})
    stack.apply_transition(AUG_START_WHILE, {0, 1}, None)
    stack.apply_transition(AUG_START_IF, {0}, None)
    stack.apply_transition(AUG_FUNC_CALL, {0}, None)
    stack.apply_transition(AUG_END_IF, {0}, None)
    stack.apply_transition(AUG_FUNC_CALL, {1}, None)
    stack.apply_transition(AUG_END_ELSE, {1}, None)
    stack.apply_transition(AUG_END_WHILE, {0, 1}, None)
    # Building phase complete; execution phase + parent frontier remain.
    assert len(stack.stack) == 2


def test_signature_is_stable_across_copies():
    """Two stacks built from the same transition sequence have equal signatures."""
    def build():
        s = AugmentationStack.init_new_stack(all_indices={0, 1})
        s.apply_transition(AUG_FUNC_CALL, {0, 1}, None)
        s.apply_transition(AUG_START_IF, {0}, None)
        return s

    a = build()
    b = build()
    assert a.get_signature() == b.get_signature()


def test_copy_independence():
    """Mutating the copy doesn't change the original."""
    s = AugmentationStack.init_new_stack(all_indices={0, 1})
    s.apply_transition(AUG_FUNC_CALL, {0, 1}, None)
    snap_len = len(s.stack)

    c = s.copy()
    c.apply_transition(AUG_START_IF, {0}, None)

    assert len(s.stack) == snap_len  # original untouched
    assert len(c.stack) == snap_len + 2
