from .comp_env import * # SimpleCompEnv, Function, BoolFunction, CompSignature
from .syntax_tree import ASTNode, BlockNode, FunctionCallAssignNode, BoolExprNode, IfElseNode, WhileNode, ReturnNode, DirectAssignNode

ASTNodePosition = tuple[int]
ExecutionPositionTuple = tuple[tuple[int], tuple[int]]

class ExecutionPosition:
    # TODO: Replace this class with simple tuples. 
    def __init__(self, ast_position: list[int], temporal_stack: list[int]):
        self.ast_position = list(ast_position)
        self.temporal_stack = list(temporal_stack) # "How many times" BEFORE now, has this line in this position been read.

    def copy(self):
        return ExecutionPosition(self.ast_position.copy(), self.temporal_stack.copy())
    
    def next_line(self):
        self.ast_position[-1] += 1
        self.temporal_stack[-1] = 0

    def enter_context(self):
        self.ast_position.append(0)
        self.temporal_stack[-1] += 1
        self.temporal_stack.append(0)
    
    def exit_context(self):
        self.ast_position.pop()
        self.temporal_stack.pop()
    
    def __repr__(self):
        return f'ExecutionPosition({self.ast_position}, {self.temporal_stack})'
    
    def __eq__(self, other):
        return (self.ast_position, self.temporal_stack) == (other.ast_position, other.temporal_stack)
    
    def lexico_position(self):
        return (self.temporal_stack[0],) + tuple((zip(self.ast_position, self.temporal_stack[1:])))
         
    def __lt__(self, other):
        return self.lexico_position() < other.lexico_position()

    def start_position():
        return ExecutionPosition([],[0])
    
    def to_tuple(self) -> ExecutionPositionTuple:
        return (tuple(self.ast_position), tuple(self.temporal_stack))

def lexico_position(position_tuple):
    ast_position, temporal_stack = position_tuple
    return (temporal_stack[0],) + tuple((zip(ast_position, temporal_stack[1:])))
    
class ExecutionPositionV2(tuple):
    __slots__ = ()  # no __dict__

    def __new__(cls, ast_position, temporal_stack):
        # convert to tuples to ensure immutability
        return super().__new__(cls, (tuple(ast_position), tuple(temporal_stack)))

    @property
    def ast_position(self):
        return self[0]

    @property
    def temporal_stack(self):
        return self[1]

    def copy(self):
        # tuples are already immutable, copy not strictly necessary
        return ExecutionPositionV2(self.ast_position, self.temporal_stack)

    def next_line(self):
        # return a new instance with last ast_position incremented, temporal_stack[-1] reset
        new_ast = self.ast_position[:-1] + (self.ast_position[-1] + 1,)
        new_temp = self.temporal_stack[:-1] + (0,)
        return ExecutionPositionV2(new_ast, new_temp)

    def enter_context(self):
        # push a new 0 to ast_position and temporal_stack, increment previous temporal_stack[-1]
        new_ast = self.ast_position + (0,)
        new_temp = self.temporal_stack[:-1] + (self.temporal_stack[-1] + 1, 0)
        return ExecutionPositionV2(new_ast, new_temp)

    def exit_context(self):
        # remove last element of ast_position and temporal_stack
        if len(self.ast_position) == 0:
            raise IndexError("Cannot exit context from empty position")
        new_ast = self.ast_position[:-1]
        new_temp = self.temporal_stack[:-1]
        return ExecutionPositionV2(new_ast, new_temp)

    def __repr__(self):
        return f'ExecutionPositionV2({self.ast_position}, {self.temporal_stack})'

    def lexico_position(self):
        return (self.temporal_stack[0],) + tuple(zip(self.ast_position, self.temporal_stack[1:]))

    def __lt__(self, other):
        return self.lexico_position() < other.lexico_position()

    @staticmethod
    def start_position():
        return ExecutionPositionV2((), (0,))



class ExecutionResult:
    def __init__(self, exec_type: str, exec_position: ExecutionPosition, result_data: Action | bool | None):
        self.exec_type = exec_type
        self.exec_position = exec_position
        self.result_data = result_data

class ExecutionContext:
    def __init__(self, ast: BlockNode, execution_position: ExecutionPositionV2, trace: SimpleCompEnv, variables: dict[str, ObjId], completed: bool):
        self.ast = ast
        self.execution_position = execution_position
        self.trace = trace
        self.variables = variables
        self.completed = completed

    def get_current_node(self) -> ASTNode:
        return self.ast.get_node_at_position(self.execution_position.ast_position)

def execute_step(context: ExecutionContext):
    if context.completed:
        print("program execution completed.")
        return
    
    node = context.get_current_node()

    if isinstance(node, BlockNode):
        execute_block_node(context, node)
    elif isinstance(node, FunctionCallAssignNode):
        execute_function_call_assign_node(context, node)
    elif isinstance(node, BoolExprNode):
        execute_bool_expr_node(context, node)
    elif isinstance(node, IfElseNode):
        execute_ifelse_node(context, node)
    elif isinstance(node, WhileNode):
        execute_while_node(context, node)
    elif isinstance(node, DirectAssignNode):
        execute_direct_assign_node(context, node)
    elif isinstance(node, ReturnNode):
        execute_return_node(context, node)
    
    return context

def go_to_next_statement(context: ExecutionContext):
    parent_block = context.ast.get_node_at_position(context.execution_position.ast_position[:-1])
    if len(parent_block.children) == context.execution_position.ast_position[-1] + 1:
        context.execution_position = context.execution_position.exit_context()
    else:
        context.execution_position = context.execution_position.next_line() 

def statement_seen(context: ExecutionContext) -> bool:
    return context.execution_position.temporal_stack[-1] > 0

def execute_block_node(context: ExecutionContext, node: BlockNode):
    if statement_seen(context): # if block has already been completed:
        if context.execution_position.ast_position == (): # if entire program completed, mark as such. 
            context.completed = True
            return
        context.execution_position = context.execution_position.exit_context()
    else:
        context.execution_position = context.execution_position.enter_context()
        if node.children == []:
            context.execution_position = context.execution_position.exit_context()

def execute_function_call_assign_node(context: ExecutionContext, node: FunctionCallAssignNode):
    func_name = node.func_name
    input_ids = tuple(context.variables[name] for name in node.arg_names)
    output_ids = context.trace.apply_function(func_name, input_ids)

    for name, output_id in zip(node.var_names, output_ids):
        context.variables[name] = output_id

    go_to_next_statement(context)

def execute_bool_expr_node(context: ExecutionContext, node: BoolExprNode):
    bool_func_name = node.bool_func
    input_ids = tuple(context.variables[name] for name in node.arg_names)
    bool_func: BoolFunction = context.trace.known_functions[bool_func_name]
    
    return bool_func.try_run([context.trace.objects[obj_id] for obj_id in input_ids])[0]

def execute_ifelse_node(context: ExecutionContext, node: IfElseNode):
    if statement_seen(context):
        go_to_next_statement(context)
        return
    
    bool_expr_node = node.bool_expr
    conditional_value = execute_bool_expr_node(context, bool_expr_node)
    
    context.execution_position = context.execution_position.enter_context()
    context.execution_position = context.execution_position.next_line() # go to if block
    if not conditional_value: # go to else block
        context.execution_position = context.execution_position.next_line()

def execute_while_node(context: ExecutionContext, node: WhileNode):
    bool_expr_node = node.bool_expr
    conditional_value = execute_bool_expr_node(context, bool_expr_node)
    if conditional_value:
        # go to block node.
        context.execution_position = context.execution_position.enter_context()
        context.execution_position = context.execution_position.next_line()
    else:
        go_to_next_statement(context)   

def execute_direct_assign_node(context: ExecutionContext, node: DirectAssignNode):
    obj_ids = [context.variables.pop(var_name) for var_name in node.source_vars]
    for target_var, obj_id in zip(node.target_vars, obj_ids):
        context.variables[target_var] = obj_id
    
    go_to_next_statement(context)

def execute_return_node(context: ExecutionContext, node: ReturnNode):
    context.trace.solution_object_id = context.variables[node.return_var_name]
    context.completed = True



class AnnotatedAST:
    def __init__(self, ast: ASTNode, signature: CompSignature, mapping: dict[ShortAction, ExecutionPositionTuple], initial_vars: dict[str, ObjId]):
        self.ast = ast
        self.signature = signature
        self.mapping = mapping
        self.initial_vars = initial_vars

        self.reverse_mapping: dict[ExecutionPositionTuple, ShortAction] = {}

        self.action_node_positions = set()
        self.ast_node_pos_to_action = {}
        
        for action, exec_position in mapping.items():
            self.action_node_positions.add(exec_position[0])
            self.ast_node_pos_to_action.setdefault(tuple(exec_position[0]), []).append(action)
            self.reverse_mapping[exec_position] = action


    def add_action(self, short_action: ShortAction, exec_position: ExecutionPositionTuple):
        self.signature.append_action(short_action)
        self.mapping[short_action] = exec_position
        self.ast_node_pos_to_action.setdefault(exec_position[0], []).append(short_action)
        # self.ast_node_pos_to_action[exec_position[0]]
        self.action_node_positions.add(exec_position[0])
        self.reverse_mapping[exec_position] = short_action
        
    
    def get_actions_by_ast_position(self, ast_position: ASTNodePosition):
        return self.reverse_mapping[ast_position]
    
    def get_action_history(self):
        return [self.reverse_mapping[pos] for pos in sorted(self.mapping.values(), key=lexico_position)]
    
    def get_action_history_up_to_position(self, exec_position_tuple: tuple):
        """Return all actions that occur up to and including the given execution position."""
        # Sort by lexicographic order of ExecutionPosition tuples
        sorted_positions = sorted(self.mapping.values(), key=lexico_position)
        return [
            self.reverse_mapping[pos]
            for pos in sorted_positions
            if lexico_position(pos) < lexico_position(exec_position_tuple)  # keep all earlier or equal
        ]

    
    def get_variable_state_at_position(self, exec_position_tuple: tuple):
        """
        Reconstruct the variable bindings as they were right before the given execution position.
        Returns a dict {var_name: obj_id}.
        """
        history = self.get_action_history_up_to_position(exec_position_tuple)

        var_dict = self.initial_vars.copy()
        for short_action in history:
            pos = self.mapping[short_action]
            
            node = self.ast.get_node_at_position(pos[0])
            if isinstance(node, FunctionCallAssignNode):
                output_vars = node.var_names
                for idx, var in enumerate(output_vars):
                    output_id = create_canonical_id(*short_action, idx)
                    var_dict[var] = output_id
            if isinstance(node, DirectAssignNode):
                for target, source in zip(node.target_vars, node.source_vars):
                    if source not in var_dict:
                        continue
                    var_dict[target] = var_dict[source]


        return var_dict
    
    @staticmethod
    def create_new_annotated_ast(ast: ASTNode, initial_vars: dict[str, ObjId]):
        return AnnotatedAST(ast, CompSignature([]), dict(), initial_vars)
    
    def copy(self):
        return AnnotatedAST(self.ast.copy(), self.signature.copy(), self.mapping.copy(), self.initial_vars.copy())
    
    def __repr__(self):
        return f"AnnotatedAST(history_len={len(self.mapping)})"

RETURN_FUNC_NAME = "RETURN_FUNC_NAME"
def execute_step_with_annotation(context: ExecutionContext , annotation: AnnotatedAST):
    node = context.get_current_node()
    if isinstance(node, FunctionCallAssignNode):
        exec_position_tuple = context.execution_position
        execute_step(context)
        short_action = context.trace.action_history_short[-1]
        annotation.add_action(short_action, exec_position_tuple)
    if isinstance(node, ReturnNode):
        exec_position_tuple = context.execution_position
        execute_step(context)
        return_obj_id = context.trace.solution_object_id
        short_action = (RETURN_FUNC_NAME, (return_obj_id, ))
        annotation.add_action(short_action, exec_position_tuple)

    else:
        execute_step(context)


def main():
    pass

if __name__ == '__main__':
    main()