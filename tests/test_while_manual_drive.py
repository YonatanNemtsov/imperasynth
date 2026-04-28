"""Manual drive of a while loop — bypasses search, hand-applies candidates,
verifies trace state advances iteration-by-iteration.

The point: prove the build phase + execute phase actually iterate.

Setup: 2 traces, both starting with x0=1. We construct a loop whose body
increments x0:
    while (TBD) {
        x1 = increment(x0);
        x0 <- x1;          // end-while DirectAssign rebinds
    }
The build phase runs the first iteration (x0: 1 → 2). The execute phase
runs a second iteration (x0: 2 → 3). After exiting, x0 = 3.
"""
from core_lang_env.comp_env import Function
from core_lang_env.syntax_tree import (
    DirectAssignNode,
    FunctionCallAssignNode,
    WhileNode,
)
from searchers.search_orchestrator import (
    EndWhileCandidate,
    EnterWhileCandidate,
    ExecuteBlockCandidate,
    ExecuteDirectAssignCandidate,
    ExecuteFuncCallCandidate,
    FuncCallCandidate,
    SearchState,
    SkipWhileCandidate,
    StartWhileCandidate,
)
from searchers.searchers_utils import Problem


def test_manual_while_loop_two_iterations():
    increment = Function(lambda x: (x + 1,), [int], [int])
    funcs = {"increment": increment}

    problem = Problem(
        (int,), (int,),
        instances={0: ((1,), (3,)), 1: ((1,), (3,))},
    )

    state = SearchState.init_new_search_state_from_problem_and_funcs(problem, funcs)

    # Sanity: x0 = 1 in both traces.
    assert all(t.objects[state.variable_states[i]["x0"]].value == 1
               for i, t in enumerate(state.trace_group.traces))

    # ============================================================
    # BUILD PHASE = first iteration
    # ============================================================

    # 1) Start while; both traces enter the loop.
    root_pos = state.aug_stack.peek()[0]
    state = state.apply_start_while_candidate(StartWhileCandidate(root_pos, (0, 1)))

    assert isinstance(state.ast_group.ast.statements[0], WhileNode)

    # 2) Inside the body, fire FUNC_CALL `x1 = increment(x0)` on both traces.
    body_frontier = state.aug_stack.peek()
    fc_pos = body_frontier[0]
    state = state.apply_func_call_candidate(FuncCallCandidate(
        exec_position=fc_pos,
        group_indices=(0, 1),
        var_action=("increment", ("x0",)),
        short_actions=(("increment", (state.variable_states[0]["x0"],)),
                       ("increment", (state.variable_states[1]["x0"],))),
        output_length=1,
    ))

    # First-iteration result: x0 still 1, x1 = increment(1) = 2.
    for i, t in enumerate(state.trace_group.traces):
        assert t.objects[state.variable_states[i]["x0"]].value == 1
        assert t.objects[state.variable_states[i]["x1"]].value == 2

    # 3) End while with rebinding x0 <- x1 — inserts the DirectAssign at end of body
    #    and rebinds x0 in variable_states.
    end_frontier = state.aug_stack.peek()
    end_pos = end_frontier[0]
    state = state.apply_end_while_candidate(EndWhileCandidate(
        exec_position=end_pos,
        group_indices=(0, 1),
        target_vars=("x0",),
        source_vars=("x1",),
    ))

    # After end_while: x0 should be the value-2 object, x1 cleaned up.
    for i, t in enumerate(state.trace_group.traces):
        assert t.objects[state.variable_states[i]["x0"]].value == 2
        assert "x1" not in state.variable_states[i]

    # AST should have a WhileNode whose body contains [FuncCall, DirectAssign].
    while_node = state.ast_group.ast.statements[0]
    assert isinstance(while_node, WhileNode)
    assert len(while_node.block.statements) == 2
    assert isinstance(while_node.block.statements[0], FunctionCallAssignNode)
    assert isinstance(while_node.block.statements[1], DirectAssignNode)
    assert while_node.block.statements[1].target_vars == ("x0",)
    assert while_node.block.statements[1].source_vars == ("x1",)

    # ============================================================
    # EXECUTE PHASE = second iteration
    # ============================================================

    # 4) Enter while again — both traces take a second iteration.
    enter_frontier = state.aug_stack.peek()
    enter_pos = enter_frontier[0]
    state = state.apply_enter_while_candidate(
        EnterWhileCandidate(enter_pos, (0, 1))
    )

    # 5) Execute the body block. subnodes = (FunctionCallAssignNode, DirectAssignNode).
    block_frontier = state.aug_stack.peek()
    block_pos = block_frontier[0]
    state = state.apply_execute_block_candidate(ExecuteBlockCandidate(
        exec_position=block_pos,
        group_indices=(0, 1),
        subnodes=(FunctionCallAssignNode, DirectAssignNode),
    ))

    # 6) Execute the func call: x1 = increment(x0). x0=2 → x1=3.
    fc_frontier = state.aug_stack.peek()
    fc_pos = fc_frontier[0]
    state = state.apply_execute_func_call_candidate(
        ExecuteFuncCallCandidate(fc_pos, (0, 1))
    )

    for i, t in enumerate(state.trace_group.traces):
        assert t.objects[state.variable_states[i]["x0"]].value == 2
        assert t.objects[state.variable_states[i]["x1"]].value == 3

    # 7) Execute the direct assign: x0 <- x1.
    da_frontier = state.aug_stack.peek()
    da_pos = da_frontier[0]
    state = state.apply_execute_direct_assign_candidate(
        ExecuteDirectAssignCandidate(da_pos, (0, 1))
    )

    for i, t in enumerate(state.trace_group.traces):
        assert t.objects[state.variable_states[i]["x0"]].value == 3
        assert "x1" not in state.variable_states[i]

    # 8) Skip while — exit the loop.
    skip_frontier = state.aug_stack.peek()
    skip_pos = skip_frontier[0]
    state = state.apply_skip_while_candidate(
        SkipWhileCandidate(skip_pos, (0, 1))
    )

    # Stack should now have only the parent frontier (the post-while position).
    assert len(state.aug_stack.stack) == 1
    # Final trace state: x0 = 3 in both traces (incremented twice from 1).
    for i, t in enumerate(state.trace_group.traces):
        assert t.objects[state.variable_states[i]["x0"]].value == 3

    # And both traces should have recorded 2 increment actions.
    for t in state.trace_group.traces:
        assert len(t.action_history) == 2
        assert all(a[0] == "increment" for a in t.action_history)
