"""Data flow graph — ported from notebooks/data_flow_graph.ipynb.

`DataFlowGraph.from_ast` walks an AST and builds a graph where nodes are
variables / function calls / direct assigns, and edges encode dependencies.
"""
from core_lang_env.parser import parse_code_str
from searchers.data_flow_graph import (
    DataFlowGraph,
    DataFlowGraphNodeDirectAssign,
    DataFlowGraphNodeFunctionCall,
    DataFlowGraphNodeVar,
)


def test_from_ast_includes_input_nodes():
    ast = parse_code_str("""
    {
        x2 = add(x0, x1);
        return x2;
    }
    """)
    graph = DataFlowGraph.from_ast(ast, {"x0": 0, "x1": 1})
    # x0 (id=0) and x1 (id=1) are inputs.
    assert 0 in graph.input_nodes
    assert 1 in graph.input_nodes


def test_from_ast_creates_func_call_node_for_each_call():
    ast = parse_code_str("""
    {
        x2 = add(x0, x1);
        x3 = add(x2, x1);
        return x3;
    }
    """)
    graph = DataFlowGraph.from_ast(ast, {"x0": 0, "x1": 1})
    func_call_nodes = [n for n in graph.nodes.values() if isinstance(n, DataFlowGraphNodeFunctionCall)]
    assert len(func_call_nodes) == 2


def test_from_ast_func_call_inputs_point_to_var_nodes():
    ast = parse_code_str("""
    {
        x2 = add(x0, x1);
        return x2;
    }
    """)
    graph = DataFlowGraph.from_ast(ast, {"x0": 0, "x1": 1})
    fc_node = next(n for n in graph.nodes.values() if isinstance(n, DataFlowGraphNodeFunctionCall))
    fc_inputs = graph.graph[fc_node.node_id]
    assert len(fc_inputs) == 2
    # Both should be DataFlowGraphNodeVar (x0 and x1).
    for inp in fc_inputs:
        assert isinstance(inp, DataFlowGraphNodeVar)


def test_from_ast_direct_assign_creates_direct_assign_node():
    """Swap assignment: both x0 and x1 already exist, so no fresh-var bug is hit."""
    ast = parse_code_str("""
    {
        x0, x1 <- x1, x0;
        return x0;
    }
    """)
    graph = DataFlowGraph.from_ast(ast, {"x0": 0, "x1": 1})
    da_nodes = [n for n in graph.nodes.values() if isinstance(n, DataFlowGraphNodeDirectAssign)]
    # One DirectAssignNode with two pairs → two DataFlowGraphNodeDirectAssign nodes.
    assert len(da_nodes) == 2


def test_from_ast_direct_assign_with_fresh_target_vars():
    """End-of-else assignments often introduce fresh target variables;
    DataFlowGraph creates new var nodes for them."""
    ast = parse_code_str("""
    {
        x2 = add(x0, x1);
        x3, x4 <- x2, x0;
        return x3;
    }
    """)
    graph = DataFlowGraph.from_ast(ast, {"x0": 0, "x1": 1})
    da_nodes = [n for n in graph.nodes.values() if isinstance(n, DataFlowGraphNodeDirectAssign)]
    assert len(da_nodes) == 2  # one assign per (target, source) pair


def test_from_ast_descends_into_if_else_blocks():
    """DataFlowGraph.from_ast should also process the bodies of if/else."""
    ast = parse_code_str("""
    {
        if (cond()) {
            x2 = add(x0, x1);
        } else {
            x3 = add(x1, x0);
        }
        return x0;
    }
    """)
    graph = DataFlowGraph.from_ast(ast, {"x0": 0, "x1": 1})
    func_call_nodes = [n for n in graph.nodes.values() if isinstance(n, DataFlowGraphNodeFunctionCall)]
    # One add() in if block, one in else block.
    assert len(func_call_nodes) == 2


def test_from_ast_descends_into_while_block():
    ast = parse_code_str("""
    {
        while (cond()) {
            x0 = increment(x0);
        }
        return x0;
    }
    """)
    graph = DataFlowGraph.from_ast(ast, {"x0": 0})
    func_call_nodes = [n for n in graph.nodes.values() if isinstance(n, DataFlowGraphNodeFunctionCall)]
    assert len(func_call_nodes) == 1
