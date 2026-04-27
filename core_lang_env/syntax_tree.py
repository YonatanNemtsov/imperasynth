# TODO: Maybe add an "InputNode", something like INPUTS: x1, x2; 
# This will take some refactoring, but not too much, maybe 2-3 hours. 
# I think it might be worth it. 

ASTNodePosition = tuple[int]

class ASTSignature:
    pass

class ASTNode:
    def copy(self) -> 'ASTNode':
        raise NotImplementedError("Subclasses must implement copy()")
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ASTNode):
            return NotImplemented
        return self.__dict__ == other.__dict__
    
    def get_child(self, index) -> 'ASTNode':
        raise NotImplementedError("Subclasses must implement get_child()")
    
    @property
    def children(self) -> list['ASTNode']:
        raise NotImplementedError("Subclasses must implement get_child()")
    
    def get_node_at_position(self, position: ASTNodePosition):
        if len(position) == 0:
            return self
        if len(position) == 1:
            return self.get_child(position[0])
        return self.get_child(position[0]).get_node_at_position(position[1:])
    
    def replace_child(self, index, new_node):
        raise NotImplementedError("Subclasses must implement replace_child()")
    
    def replace_node(self, position, new_node):
        parent = self.get_node_at_position(position[:-1])
        parent.replace_child(position[-1], new_node)

    def insert_node(self, position, node):
        insertion_node = self.get_node_at_position(position[:-1])
        if not isinstance(insertion_node, BlockNode):
            raise TypeError("can't insert into non BlockNode node.")
        insertion_node.statements.insert(position[-1], node)



class BlockNode(ASTNode):
    """ Represents a block of statements enclosed in curly braces { ... } """
    def __init__(self, statements: list[ASTNode]):
        self.statements = statements  # List of ASTNode
    def copy(self) -> 'BlockNode':
        return BlockNode([stmt.copy() for stmt in self.statements])
    
    def __eq__(self, other: ASTNode):
        if not isinstance(other, BlockNode):
            return False
        if len(self.statements) != len(other.statements):
            return False
        return all((s1 == s2 for s1, s2 in zip(self.statements, other.statements)))
    
    def get_child(self, index: int) -> ASTNode:
        if 0 <= index < len(self.statements):
            return self.statements[index]
        raise IndexError(f"BlockNode has no child at index {index}")
    
    @property
    def children(self) -> list['ASTNode']:
        return self.statements
    
    def replace_child(self, index, new_node):
        if 0 <= index < len(self.statements):
            self.statements[index] = new_node
            return
        raise IndexError(f"BlockNode has no child at index {index}")
    
    def __repr__(self):
        return f'BlockNode({self.statements})'

class FunctionCallAssignNode(ASTNode):
    """Expressions of the form: 
    x3, x4 = basic_func(x1, x2);
    """
    def __init__(self, var_names: list[str] , func_name: str, arg_names: list[str]):
        self.var_names = var_names
        self.func_name = func_name
        self.arg_names = arg_names
    
    def copy(self) -> 'FunctionCallAssignNode':
        return FunctionCallAssignNode(self.var_names, self.func_name, self.arg_names)
    
    def __eq__(self, other: ASTNode):
        return self.__dict__ == other.__dict__
    
    def get_child(self, index: int) -> ASTNode:
        raise IndexError("FunctionCallAssignNode has no children")
    
    @property
    def children(self) -> list['ASTNode']:
        return []
    
    def replace_child(self, index, new_node):
        raise IndexError("FunctionCallAssignNode has no children")
    
    def __repr__(self):
        return f'FunctionCallAssignNode({self.var_names}, {self.func_name}, {self.arg_names})'

class DirectAssignNode(ASTNode):
    """Represents direct variable assignment (single or multiple): 
       x <- y  OR  x1,x2 <- y1,y2"""
    def __init__(self, target_vars: tuple[str], source_vars: tuple[str]):
        if len(target_vars) != len(source_vars):
            raise ValueError("Number of target and source variables must match")
        self.target_vars = target_vars  # List of variables being assigned to
        self.source_vars = source_vars  # List of source variables

    def copy(self) -> 'DirectAssignNode':
        return DirectAssignNode(self.target_vars, self.source_vars)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DirectAssignNode):
            return NotImplemented
        return (self.target_vars == other.target_vars and 
                self.source_vars == other.source_vars)
    
    def get_child(self, index: int) -> ASTNode:
        raise IndexError("DirectAssignNode has no children")
    
    @property
    def children(self) -> list['ASTNode']:
        return []
    
    def replace_child(self, index: int, new_node: ASTNode):
        raise IndexError("DirectAssignNode has no children")
    
    def __repr__(self):
        targets = ','.join(self.target_vars)
        sources = ','.join(self.source_vars)
        return f'DirectAssignNode([{targets}], [{sources}])'

    @property
    def is_swap(self) -> bool:
        """Returns True if this represents a variable swap (x,y = y,x)"""
        return (len(self.target_vars) > 1 and 
                set(self.target_vars) == set(self.source_vars))

class BoolExprNode(ASTNode):
    """ Expressions of the form: 
    bool_func(x1,x3) 
    
    """
    def __init__(self, bool_func: str, arg_names: tuple[str]):
        self.bool_func = bool_func
        self.arg_names = arg_names
    
    def copy(self) -> 'BoolExprNode':
        return BoolExprNode(self.bool_func, self.arg_names.copy())
    
    def __eq__(self, other):
        return self.__dict__ == other.__dict__
    
    def get_child(self, index: int) -> ASTNode:
        raise IndexError("BoolExprNode has no children")
    
    @property
    def children(self) -> list['ASTNode']:
        return []
    
    def __repr__(self):
        return f'BoolExprNode({self.bool_func}, {self.arg_names})'
    
class IfNode(ASTNode):
    """ DEPRECATED if (bool expression) {code block} """
    def __init__(self, bool_expr: BoolExprNode, block: BlockNode):
        self.bool_expr = bool_expr  # (var, op, value)
        self.block = block  # List of ASTNode
    
    def copy(self) -> 'IfNode':
        return IfNode(self.bool_expr.copy(), self.block.copy())

class ElseNode(ASTNode):
    """DEPRECATED"""
    def __init__(self, block: BlockNode):
        self.block = block
    def copy(self) -> 'ElseNode':
        return ElseNode(self.block.copy())

class IfElseNode(ASTNode):
    def __init__(self, bool_expr: BoolExprNode, if_block: BlockNode, else_block: BlockNode):
        self.bool_expr = bool_expr
        self.if_block = if_block
        self.else_block = else_block
    
    def copy(self) -> 'IfElseNode':
        return IfElseNode(self.bool_expr.copy(), self.if_block.copy(), self.else_block.copy())
    
    def __eq__(self, other: ASTNode):
        if not isinstance(other, IfElseNode):
            return False
        return all((
            self.bool_expr == other.bool_expr,
            self.if_block == other.if_block,
            self.else_block == other.else_block
        ))
    
    def get_child(self, index: int) -> ASTNode:
        if index == 0:
            return self.bool_expr
        if index == 1:
            return self.if_block
        if index == 2 and self.else_block:
            return self.else_block
        raise IndexError(f"IfElseNode has no child at index {index}")
    
    @property
    def children(self) -> list['ASTNode']:
        return [self.bool_expr, self.if_block, self.else_block]
    
    def replace_child(self, index: int, new_node: ASTNode) -> ASTNode:
        if index not in (0,1,2):
            raise IndexError(f"IfElseNode has no child at index {index}")
        if index == 0:
            if not isinstance(new_node, BoolExprNode):
                raise TypeError("IfElseNode condition must be BoolExprNode")
            self.bool_expr = new_node
        
        elif index == 1:
            if not isinstance(new_node, BlockNode):
                raise TypeError("IfElseNode if_block must be BlockNode")
            self.if_block = new_node
        
        elif index == 2:
            self.copy()
            if not isinstance(new_node, BlockNode):
                raise TypeError("IfElseNode else_block must be BlockNode")
            if self.else_block is None:
                raise IndexError("IfElseNode has no else_block at index 2")
            self.else_block = new_node
    
    def __repr__(self):
        return f'IfElseNode({self.bool_expr, self.if_block, self.else_block})'
    
class WhileNode(ASTNode):
    """ while (bool expression) {code block} """
    def __init__(self, bool_expr: BoolExprNode, block: BlockNode):
        self.bool_expr = bool_expr  # (var, op, value)
        self.block = block  # List of ASTNode
    
    def copy(self) -> 'WhileNode':
        return WhileNode(self.bool_expr.copy(), self.block.copy())
    
    def __eq__(self, other: ASTNode):
        if not isinstance(other, WhileNode):
            return False
        return all((
            self.bool_expr == other.bool_expr,
            self.block == other.block,
        ))
    
    def get_child(self, index: int) -> ASTNode:
        if index == 0:
            return self.bool_expr
        if index == 1:
            return self.block
        raise IndexError(f"WhileNode has no child at index {index}")
    
    @property
    def children(self) -> list['ASTNode']:
        return [self.bool_expr, self.block]
    
    def replace_child(self, index: int, new_node: ASTNode) -> ASTNode:
        if index not in (0,1):
            raise IndexError(f"WhileNode has no child at index {index}")
        if index == 0:
            if not isinstance(new_node, BoolExprNode):
                raise TypeError("WhileNode condition must be BoolExprNode")
            self.bool_expr = new_node
        
        elif index == 1:
            if not isinstance(new_node, BlockNode):
                raise TypeError("WhileNode block must be BlockNode")
            self.block = new_node
        
    def __repr__(self):
        return f'WhileNode({self.bool_expr}, {self.block})'


class RepeatNode(ASTNode):
    """ 
    Represents a repeat loop: repeat (count) {code block} 
    Executes the block a fixed number of times
    """
    def __init__(self, count_var: str, block: BlockNode):
        self.count_var = count_var
        self.block = block          # Code block to repeat
        
    def copy(self) -> 'RepeatNode':
        return RepeatNode(self.count_var, self.block.copy())
    
    def __eq__(self, other: ASTNode):
        if not isinstance(other, RepeatNode):
            return False
        return self.count_var == other.count_var and self.block == other.block
    
    def get_child(self, index: int) -> ASTNode:
        if index == 0:
            return self.block
        raise IndexError(f"RepeatNode has no child at index {index}")
    
    @property
    def children(self) -> list['ASTNode']:
        return [self.block]
    
    def replace_child(self, index: int, new_node: ASTNode):
        if index != 0:
            raise IndexError(f"RepeatNode has no child at index {index}")
        if not isinstance(new_node, BlockNode):
            raise TypeError("RepeatNode block must be BlockNode")
        self.block = new_node
    
    def __repr__(self):
        return f'RepeatNode({self.count_var}, {self.block})'


class ReturnNode(ASTNode):
    def __init__(self, return_var_name):
        self.return_var_name = return_var_name
    
    def copy(self) -> 'ReturnNode':
        return ReturnNode(self.return_var_name)

    def get_child(self, index: int) -> ASTNode:
        raise IndexError("ReturnNode has no children")
    
    @property
    def children(self) -> list['ASTNode']:
        return []

    def replace_child(self, index, new_node):
        raise IndexError("ReturnNode has no children")
    
    def __repr__(self):
        return f'ReturnNode("{self.return_var_name}")'








############# Visualizer #######################

from graphviz import Digraph

def visualize_ast(node: 'ASTNode', graph=None, parent_id=None, edge_label='') -> Digraph:
    """
    Visualizes the Abstract Syntax Tree using Graphviz
    Returns a Digraph object that can be rendered
    """
    if graph is None:
        graph = Digraph(filename='./images/ast')
        graph.attr('node', shape='box', style='rounded', fontname='Courier')
    
    # Create unique node ID
    node_id = str(id(node))
    
    # Determine node label based on type
    if isinstance(node, BlockNode):
        label = 'Block'
    elif isinstance(node, DirectAssignNode):
        label = f'Assign: {node.target_var} <- {node.source_var}'
    elif isinstance(node, FunctionCallAssignNode):
        label = f'FuncAssign: {", ".join(node.var_names)}\n{node.func_name}({", ".join(node.arg_names)})'
    elif isinstance(node, BoolExprNode):
        label = f'BoolExpr\n{node.bool_func}({", ".join(node.arg_names)})'
    elif isinstance(node, IfNode):
        label = 'If'
    elif isinstance(node, IfElseNode):
        label = 'IfElse'
    elif isinstance(node, WhileNode):
        label = 'While'
    elif isinstance(node, RepeatNode):
        label = f'Repeat: {node.count_var}'
    elif isinstance(node, ReturnNode):
        label = f'Return {node.return_var_name}'
    else:
        label = 'ASTNode'
    
    # Add the node to the graph
    graph.node(node_id, label)
    
    # Connect to parent if exists
    if parent_id is not None:
        graph.edge(parent_id, node_id, label=edge_label)
    
    # Recursively add children
    if isinstance(node, BlockNode):
        for i, stmt in enumerate(node.statements):
            visualize_ast(stmt, graph, node_id, f'stmt {i}')
    
    elif isinstance(node, IfNode):
        visualize_ast(node.bool_expr, graph, node_id, 'condition')
        visualize_ast(node.block, graph, node_id, 'body')
    
    elif isinstance(node, IfElseNode):
        visualize_ast(node.bool_expr, graph, node_id, 'condition')
        visualize_ast(node.if_block, graph, node_id, 'if-body')
        visualize_ast(node.else_block, graph, node_id, 'else-body')
    
    elif isinstance(node, WhileNode):
        visualize_ast(node.bool_expr, graph, node_id, 'condition')
        visualize_ast(node.block, graph, node_id, 'body')
    
    elif isinstance(node, RepeatNode):
        visualize_ast(node.block, graph, node_id, 'body')
    
    return graph