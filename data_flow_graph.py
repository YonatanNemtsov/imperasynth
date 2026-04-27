from core_lang_env.comp_env import *
from core_lang_env.syntax_tree import *
import hashlib


class DataFlowGraphNode:
    def __init__(self, node_id):
        self.node_id = node_id

    def __hash__(self):
        return self.node_id
    
    def __eq__(self, other):
        return self.node_id == other.node_id

class DataFlowGraphNodeVar(DataFlowGraphNode):
    def __init__(self, node_id):
        self.node_id = node_id

    @staticmethod
    def from_func_call(func_call_node: 'DataFlowGraphNodeFunctionCall', output_index: int):
        node_id = hash((func_call_node.func_name, func_call_node.node_id, output_index))
        return DataFlowGraphNodeVar(node_id)
    
    def __repr__(self):
        return f'DataFlowGraphNodeVar(node_id={self.node_id%1000}...)'

class DataFlowGraphNodeFunctionCall(DataFlowGraphNode):
    def __init__(self, func_name, node_id):
        self.func_name = func_name
        self.node_id = node_id

    @staticmethod
    def from_inputs(func_name, input_nodes: tuple[DataFlowGraphNodeVar]):
        node_id = hash((func_name, *tuple(inp.node_id for inp in input_nodes)))
        return DataFlowGraphNodeFunctionCall(func_name, node_id)
    
    def __repr__(self):
        return f'DataFlowGraphNodeFunctionCall({self.func_name}, node_id={self.node_id%1000}...)'

class DataFlowGraphNodeDirectAssign(DataFlowGraphNode):
    def __init__(self, node_id, target_node: DataFlowGraphNodeVar, source_node: DataFlowGraphNodeVar):
        self.node_id = node_id
        self.target = target_node
        self.source = source_node
    
    @staticmethod
    def from_source_and_target(target_node: DataFlowGraphNodeVar, source_node: DataFlowGraphNodeVar):
        node_id = hash(('assign', target_node.node_id, source_node.node_id))
        return DataFlowGraphNodeDirectAssign(node_id, target_node, source_node)

    def __repr__(self):
        return f'DataFlowGraphNodeDirectAssign(target_id={self.target.node_id%1000}, source_id={self.source.node_id%1000}...)'

class DataFlowGraph:
    def __init__(self, graph: dict[int, list[DataFlowGraphNode]], nodes: dict[int, DataFlowGraphNode], input_nodes: list[int], return_var: DataFlowGraphNodeVar = None):
        self.graph = graph
        self.nodes = nodes
        self.input_nodes = input_nodes
        self.return_var = return_var

    @staticmethod
    def from_ast(ast: BlockNode, input_ids: dict[str: int]):
        nodes = {}
        graph = {}
        var_ids = input_ids.copy()
        input_nodes = []
        for var, id in var_ids.items():
            node = DataFlowGraphNodeVar(id)
            graph[id] = []
            nodes[id] = node
            input_nodes.append(id)
        
        DataFlowGraph._prosses_block(nodes, graph, var_ids, ast)
        return DataFlowGraph(graph, nodes, input_nodes)

    @staticmethod
    def _process_func_call(nodes, graph, var_ids, stmt: FunctionCallAssignNode):
        func_name = stmt.func_name
        # print(nodes, var_ids, stmt.arg_names, stmt)
        inputs = [nodes[var_ids[arg]] for arg in stmt.arg_names]
        new_func_call_node = DataFlowGraphNodeFunctionCall.from_inputs(func_name, inputs)
        nodes[new_func_call_node.node_id] = new_func_call_node
        graph[new_func_call_node.node_id] = inputs
        for idx, out_name in enumerate(stmt.var_names):
            out_node = DataFlowGraphNodeVar.from_func_call(new_func_call_node, idx)
            var_ids[out_name] = out_node.node_id
            graph[out_node.node_id] = [new_func_call_node]
            nodes[out_node.node_id] = out_node
    
    @staticmethod
    def _prosses_direct_assign(nodes, graph, var_ids, stmt: DirectAssignNode):
        sources = [nodes[var_ids[arg]] for arg in stmt.source_vars]
        targets = [nodes[var_ids[arg]] for arg in stmt.target_vars]
        
        for t, s in zip(targets, sources):
            new_direct_assign_node = DataFlowGraphNodeDirectAssign.from_source_and_target(t, s)
            nodes[new_direct_assign_node.node_id] = new_direct_assign_node
            graph[new_direct_assign_node.node_id] = [s]
            graph[t.node_id].append(new_direct_assign_node)

    
    @staticmethod
    def _prosses_if_else(nodes, graph, var_ids, stmt: IfElseNode):
        DataFlowGraph._prosses_block(nodes, graph, var_ids, stmt.if_block)
        DataFlowGraph._prosses_block(nodes, graph, var_ids, stmt.else_block)

    @staticmethod
    def _prosses_while(nodes, graph, var_ids, stmt: WhileNode):
        DataFlowGraph._prosses_block(nodes, graph, var_ids, stmt.block)

    @staticmethod
    def _prosses_block(nodes, graph, var_ids, block: BlockNode):
        for stmt in block.statements:
            if isinstance(stmt, FunctionCallAssignNode):
                DataFlowGraph._process_func_call(nodes, graph, var_ids, stmt)

            elif isinstance(stmt, DirectAssignNode):
                DataFlowGraph._prosses_direct_assign(nodes, graph, var_ids, stmt)
            
            elif isinstance(stmt, IfElseNode):
                DataFlowGraph._prosses_if_else(nodes, graph, var_ids, stmt)
            
            elif isinstance(stmt, WhileNode):
                DataFlowGraph._prosses_while(nodes, graph, var_ids, stmt)




# --------------- Visualizer -------------

from graphviz import Digraph

def visualize_data_flow_graph(dfg: DataFlowGraph, filename='data_flow_graph'):
    """
    Visualizes the Data Flow Graph using Graphviz.
    Variable nodes are squares, function calls are circles, assignments are diamonds.
    """
    # Create a directed graph
    dot = Digraph(comment="Data Flow Graph", filename=filename)
    dot.attr(rankdir='LR', splines='true')  # Left-to-right for data flow
    
    # Color scheme
    colors = {
        'var': '#c5e1a5',      # light green for variables
        'input': '#fff59d',    # light yellow for inputs
        'func': '#90caf9',     # light blue for functions
        'assign': '#ffcc80',   # light orange for assignments
    }
    
    # First, let's understand the structure
    print(f"Number of nodes: {len(dfg.nodes)}")
    print(f"Number of edges: {sum(len(v) for v in dfg.graph.values())}")
    
    # Create mapping from node ID to node object for easy lookup
    id_to_node = {node_id: node_obj for node_id, node_obj in dfg.nodes.items()}
    
    # Add all nodes
    for node_id, node_obj in dfg.nodes.items():
        # Create a display ID (use the actual node ID for consistency)
        display_id = str(node_id)
        
        # Determine node type and create label
        if isinstance(node_obj, DataFlowGraphNodeVar):
            # Check if it's an input node
            is_input = node_id in dfg.input_nodes
            
            # Simple label using the node ID
            label = f"Var\n{node_id % 10000:04d}"
            fillcolor = colors['input'] if is_input else colors['var']
            
            dot.node(display_id, label,
                    shape="box",
                    style="filled,rounded",
                    fillcolor=fillcolor,
                    fontname="Arial",
                    fontsize="10")
            
        elif isinstance(node_obj, DataFlowGraphNodeFunctionCall):
            label = f"{node_obj.func_name}\n{node_id % 10000:04d}"
            dot.node(display_id, label,
                    shape="circle",
                    style="filled",
                    fillcolor=colors['func'],
                    fontname="Arial",
                    fontsize="10")
            
        elif isinstance(node_obj, DataFlowGraphNodeDirectAssign):
            label = f"Assign\n{node_id % 10000:04d}"
            dot.node(display_id, label,
                    shape="diamond",
                    style="filled",
                    fillcolor=colors['assign'],
                    fontname="Arial",
                    fontsize="9")
            
        else:
            # Unknown node type
            label = f"Node\n{node_id % 10000:04d}"
            dot.node(display_id, label,
                    shape="ellipse",
                    style="filled",
                    fillcolor="lightgray",
                    fontname="Arial",
                    fontsize="9")
    
    # Add edges
    for from_id, to_nodes in dfg.graph.items():
        from_display_id = str(from_id)
        
        # Skip if from node doesn't exist (shouldn't happen)
        if from_id not in id_to_node:
            continue
            
        from_node = id_to_node[from_id]
        
        for to_node in to_nodes:
            to_display_id = str(to_node.node_id)
            
            # Skip if to node doesn't exist
            if to_node.node_id not in id_to_node:
                continue
                
            # Style edges based on node types
            if isinstance(from_node, DataFlowGraphNodeVar) and isinstance(to_node, DataFlowGraphNodeFunctionCall):
                # Variable to function (data input)
                dot.edge(to_display_id, from_display_id, 
                        arrowsize="0.8",
                        penwidth="1.5",
                        color="blue")
            elif isinstance(from_node, DataFlowGraphNodeFunctionCall) and isinstance(to_node, DataFlowGraphNodeVar):
                # Function to variable (data output)
                dot.edge(to_display_id, from_display_id, 
                        arrowsize="0.8",
                        penwidth="1.5",
                        color="green")
            elif isinstance(from_node, DataFlowGraphNodeDirectAssign):
                # Assignment
                dot.edge(to_display_id, from_display_id, 
                        arrowsize="0.8",
                        penwidth="1.2",
                        color="orange",
                        style="dashed")
            else:
                # Default
                dot.edge(to_display_id, from_display_id, 
                        arrowsize="0.7",
                        penwidth="1.0",
                        color="black")
    
    with dot.subgraph(name='cluster_legend') as legend:
        legend.attr(label='Legend', style='rounded', color='lightgray', fontsize='10')
        
        # Create legend nodes in a row
        legend.node('L1', 'Variable', shape="box", style="filled,rounded", 
                   fillcolor=colors['var'], fontname="Arial", fontsize="8")
        legend.node('L2', 'Input', shape="box", style="filled,rounded", 
                   fillcolor=colors['input'], fontname="Arial", fontsize="8")
        legend.node('L3', 'Function', shape="circle", style="filled", 
                   fillcolor=colors['func'], fontname="Arial", fontsize="8")
        legend.node('L4', 'Assign', shape="diamond", style="filled", 
                   fillcolor=colors['assign'], fontname="Arial", fontsize="8")
        
        # Force them to be on the same rank (horizontal)
        legend.attr(rank='same')
        # Connect with invisible edges to force left-to-right order
        legend.edge('L1', 'L2', style='invis')
        legend.edge('L2', 'L3', style='invis')
        legend.edge('L3', 'L4', style='invis')
    
    # Render the graph
    try:
        output_path = dot.render(filename, format='png', cleanup=True)
        print(f"Graph saved to: {output_path}")
        return dot
    except Exception as e:
        print(f"Error rendering graph: {e}")
        # Save DOT source for debugging
        #dot_filename = f"{filename}.dot"
        #with open(dot_filename, 'w') as f:
        #    f.write(dot.source)
        #print(f"DOT source saved to: {dot_filename}")
        #return dot
