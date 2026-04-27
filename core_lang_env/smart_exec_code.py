from .comp_env import SimpleCompEnv, Function, BoolFunction, ActionDependencyGraph
from .syntax_tree import ASTNode, BlockNode, FunctionCallAssignNode, BoolExprNode, IfElseNode, WhileNode, ReturnNode, DirectAssignNode, RepeatNode

class ExecutionState:
    def __init__(self, block: BlockNode, env: SimpleCompEnv, var_dict: dict):
        self.block = block
        self.env = env
        self.var_dict = var_dict.copy()  # Ensure isolated copy
        self.current_stmt_index = 0
        self.call_stack = []  # Stack of (block, index) tuples
        self.done = False
        self.last_action = None  # Track last executed action type

    def clone(self):
        """Create a safe copy for stepping"""
        new_state = ExecutionState(self.block, self.env, self.var_dict)
        new_state.current_stmt_index = self.current_stmt_index
        new_state.call_stack = self.call_stack.copy()
        new_state.done = self.done
        new_state.last_action = self.last_action
        return new_state

    def get_current_statement(self):
        if self.current_stmt_index < len(self.block.statements):
            return self.block.statements[self.current_stmt_index]
        return None

def execute_step(state: ExecutionState, 
                basic_function_dict: dict,
                basic_bool_func_dict: dict,
                save_bool_objects=False) -> tuple[ExecutionState, str]:
    """
    Execute one step and return (new_state, action_result)
    
    Returns:
        tuple: (new_state, action_type)
        action_type can be:
            - "assignment"
            - "if_entered"/"else_entered"
            - "while_entered"/"while_skipped"
            - "repeat_entered"/"repeat_skipped"
            - "return"
            - "block_completed"
            - None (when done)
    """
    if state.done:
        return state, None

    current_stmt = state.get_current_statement()
    if current_stmt is None:
        # End of current block
        if state.call_stack:
            # Pop the call stack
            state.block, state.current_stmt_index = state.call_stack.pop()
            return state, "block_completed"
        else:
            state.done = True
            return state, None

    new_state = state.clone()
    new_state.current_stmt_index += 1  # Advance to next statement by default
    action_result = None

    try:
        if isinstance(current_stmt, DirectAssignNode):
            # Handle multiple assignments
            if len(current_stmt.target_vars) != len(current_stmt.source_vars):
                raise ValueError(f"Assignment count mismatch: {len(current_stmt.target_vars)} targets vs {len(current_stmt.source_vars)} sources")
            
            # Get all source values first (important for swaps)
            source_values = [state.var_dict[src] for src in current_stmt.source_vars]
            
            # Perform all assignments
            for target, src_value in zip(current_stmt.target_vars, source_values):
                new_state.var_dict[target] = src_value
                
            action_result = "direct_assign"

        elif isinstance(current_stmt, FunctionCallAssignNode):
            execute_function_call_assign(current_stmt, new_state.env, new_state.var_dict, 
                                    basic_function_dict, basic_bool_func_dict)
            action_result = "assignment"

        elif isinstance(current_stmt, IfElseNode):
            cond = execute_bool_expr(current_stmt.bool_expr, new_state.env,
                                    new_state.var_dict, basic_function_dict,
                                    basic_bool_func_dict,
                                    save_bool_objects=save_bool_objects)
            if cond:
                new_state.call_stack.append((new_state.block, new_state.current_stmt_index))
                new_state.block = current_stmt.if_block
                new_state.current_stmt_index = 0
                action_result = "if_entered"
            else:
                if current_stmt.else_block:
                    new_state.call_stack.append((new_state.block, new_state.current_stmt_index))
                    new_state.block = current_stmt.else_block
                    new_state.current_stmt_index = 0
                    action_result = "else_entered"
                else:
                    action_result = "if_skipped"

        elif isinstance(current_stmt, WhileNode):
            cond = execute_bool_expr(current_stmt.bool_expr, new_state.env,
                                   new_state.var_dict, basic_function_dict,
                                   basic_bool_func_dict,
                                   save_bool_objects=save_bool_objects)
            if cond:
                # Push current position before entering loop
                new_state.call_stack.append((new_state.block, new_state.current_stmt_index - 1))  # Will re-evaluate condition
                new_state.block = current_stmt.block
                new_state.current_stmt_index = 0
                action_result = "while_entered"
            else:
                action_result = "while_skipped"
        
        elif isinstance(current_stmt, RepeatNode):
            repeat_count = int(current_stmt.count_var)
            if repeat_count > 0:
                # Push the continuation point first (what to do after all iterations)
                new_state.call_stack.append((new_state.block, new_state.current_stmt_index))
                
                # Then push N copies of the block to execute
                for _ in range(repeat_count - 1):
                    new_state.call_stack.append((current_stmt.block, 0))
                
                # Move to the first iteration
                new_state.block = current_stmt.block
                new_state.current_stmt_index = 0
                action_result = f"repeat_entered_{repeat_count}"
            else:
                action_result = "repeat_skipped"
            
            # Cancel the automatic statement advance since we're jumping to the block
            # new_state.current_stmt_index -= 1
        
        elif isinstance(current_stmt, ReturnNode):
            execute_return(current_stmt, new_state.env, new_state.var_dict,
                         basic_function_dict, basic_bool_func_dict)
            new_state.done = True
            action_result = "return"

        else:
            raise ValueError(f"Unsupported statement type: {type(current_stmt)}")

    except Exception as e:
        new_state.done = True
        raise RuntimeError(f"Error executing statement at index {state.current_stmt_index}: {e}")

    new_state.last_action = action_result
    return new_state, action_result


# Helper execution functions (optimized versions)
def execute_function_call_assign(node: FunctionCallAssignNode, environment: SimpleCompEnv, var_dict: dict[str, int], basic_function_dict: dict[str, Function], basic_bool_func_dict: dict[str, BoolFunction]):
    """Execute function call assignment with error handling"""
    try:
        arg_env_ids = [var_dict[arg] for arg in node.arg_names]
        out_obj_ids = environment.apply_function(node.func_name, arg_env_ids)
        for var_name, var_id in zip(node.var_names, out_obj_ids):
            var_dict[var_name] = var_id
    except KeyError as e:
        raise ValueError(f"Undefined variable or function in assignment: {e}")

def execute_direct_assign(node: DirectAssignNode,
                          environment: SimpleCompEnv,
                          var_dict: dict[str, int],
                          basic_function_dict: dict[str, Function],
                          basic_bool_func_dict: dict[str, BoolFunction]):
    pass

def execute_bool_expr(node: BoolExprNode, 
                     environment: SimpleCompEnv, 
                     var_dict: dict[str, int], 
                     basic_function_dict: dict[str, Function], 
                     basic_bool_func_dict: dict[str, BoolFunction],
                     save_bool_objects=False) -> bool:
    """Evaluate boolean expression with error handling"""
    try:
        arg_ids = [var_dict[name] for name in node.arg_names]
        if save_bool_objects:
            condition_obj_id = environment.apply_function(node.bool_func, arg_ids)[0]
            return environment.objects[condition_obj_id].value
        
        return environment.known_functions[node.bool_func].try_run([environment.objects[i] for i in arg_ids])[0]
        
    except KeyError as e:
        raise ValueError(f"Undefined variable or function in boolean expression: {e}")

def execute_return(node: ReturnNode, 
                  environment: SimpleCompEnv, 
                  var_dict: dict[str, int], 
                  basic_function_dict: dict[str, Function], 
                  basic_bool_func_dict: dict[str, BoolFunction]):
    """Execute return statement with error handling"""
    try:
        environment.assign_solution_object(var_dict[node.return_var_name])
    except KeyError as e:
        raise ValueError(f"Undefined return variable: {e}")


class ProgramController:
    def __init__(self, ast: BlockNode, env: SimpleCompEnv, var_dict: dict, funcs: dict, bool_funcs: dict):
        self.initial_state = ExecutionState(ast, env, var_dict.copy())
        self.state = self.initial_state.clone()
        self.funcs = funcs
        self.bool_funcs = bool_funcs
        self.history = []
        self.breakpoints = set()
        self.watched_vars = set()

    def reset(self):
        """Reset execution to initial state"""
        self.state = self.initial_state.clone()
        self.history = []

    def step(self, save_bool_objects=False, print_info=True) -> tuple[bool, str, dict]:
        """
        Execute one step
        
        Returns:
            tuple: (done, action_result, state_snapshot)
            - done: True if execution completed
            - action_result: Description of what happened ("assignment", "while_entered", etc.)
            - state_snapshot: Current variable values and position
        """
        if self.state.done:
            return True, None, self._get_state_snapshot()

        # Store previous state in history
        self.history.append(self.state.clone())
        
        try:
            # Execute next step
            self.state, action_result = execute_step(
                self.state, 
                self.funcs, 
                self.bool_funcs,
                save_bool_objects=save_bool_objects
            )
            
            # Create state snapshot
            snapshot = self._get_state_snapshot()
            
            return self.state.done, action_result, snapshot
            
        except Exception as e:
            self.state.done = True
            raise RuntimeError(f"Execution error at step {len(self.history)}: {e}")

    def _get_state_snapshot(self) -> dict:
        """Get current execution state snapshot"""
        return {
            'vars': {
                name: self.state.env.objects[obj_id].value
                for name, obj_id in self.state.var_dict.items()
            },
            'position': self.state.current_stmt_index,
            'call_stack_depth': len(self.state.call_stack),
            'watched': {
                var: self.state.env.objects[self.state.var_dict[var]].value
                for var in self.watched_vars
                if var in self.state.var_dict
            }
        }

    def add_breakpoint(self, line_num: int):
        """Add breakpoint at specific line number"""
        self.breakpoints.add(line_num)

    def remove_breakpoint(self, line_num: int):
        """Remove breakpoint"""
        self.breakpoints.discard(line_num)

    def watch_variable(self, var_name: str):
        """Add variable to watch list"""
        self.watched_vars.add(var_name)

    def unwatch_variable(self, var_name: str):
        """Remove variable from watch list"""
        self.watched_vars.discard(var_name)

    def run_to_breakpoint(self, save_bool_objects=False) -> tuple[bool, str, dict]:
        """
        Execute until next breakpoint or completion
        
        Returns same tuple as step()
        """
        while True:
            done, action, snapshot = self.step(save_bool_objects=save_bool_objects)
            if done or snapshot['position'] in self.breakpoints:
                return done, action, snapshot

    def get_current_statement(self) -> ASTNode:
        """Get current AST node being executed"""
        if self.state.done:
            return None
        if self.state.current_stmt_index < len(self.state.block.statements):
            return self.state.block.statements[self.state.current_stmt_index]
        return None