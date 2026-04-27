
from .comp_env import SimpleCompEnv, Function, BoolFunction
from .syntax_tree import BlockNode, FunctionCallAssignNode, BoolExprNode, IfElseNode, WhileNode, ReturnNode

def execute_code_block(block: BlockNode, environment: SimpleCompEnv, var_dict: dict[str, int], basic_function_dict: dict[str, Function], basic_bool_func_dict: dict[str, BoolFunction]):
    for statement in block.statements:
        if isinstance(statement, FunctionCallAssignNode):
            execute_function_call_assign(statement, environment, var_dict, basic_function_dict, basic_bool_func_dict)
        #elif isinstance(statement, IfNode):
        #    execute_if(statement, environment, var_dict, basic_function_dict, basic_bool_func_dict)
        #elif isinstance(statement, ElseNode):
        #   execute_else(statement, environment, var_dict, basic_function_dict, basic_bool_func_dict)"""
        elif isinstance(statement, IfElseNode):
            execute_ifelse(statement, environment, var_dict, basic_function_dict, basic_bool_func_dict)
        elif isinstance(statement, WhileNode):
            execute_while(statement, environment, var_dict, basic_function_dict, basic_bool_func_dict)
        elif isinstance(statement, ReturnNode):
            execute_return(statement, environment, var_dict, basic_function_dict, basic_bool_func_dict)
            return
        else:
            raise ValueError(f"Unknown AST node type: {type(statement)}")

def execute_function_call_assign(node: FunctionCallAssignNode, environment: SimpleCompEnv, var_dict: dict[str, int], basic_function_dict: dict[str, Function], basic_bool_func_dict: dict[str, BoolFunction]):
    """ Execute expressions like x3, x4 = basic_func(x1, x2); """
    arg_env_ids = [var_dict[arg] for arg in node.arg_names]
    out_obj_ids = environment.apply_function(node.func_name, arg_env_ids)
    for var_name, var_id in zip(node.var_names, out_obj_ids):
        var_dict[var_name] = var_id

def execute_bool_expr(node: BoolExprNode, environment: SimpleCompEnv, var_dict: dict[str, int], basic_function_dict: dict[str, Function], basic_bool_func_dict: dict[str, BoolFunction]):
    arg_ids = [var_dict[name] for name in node.arg_names]
    condition_obj_id = environment.apply_function(node.bool_func, arg_ids)[0]
    return environment.objects[condition_obj_id].value

# def execute_if(node: IfNode, environment: SimpleCompEnv, var_dict: dict[str, int], basic_function_dict: dict[str, Function], basic_bool_func_dict: dict[str, BoolFunction]):
#    """ Execute an if statement """
#    condition_value = execute_bool_expr(node.bool_expr, environment, var_dict, basic_function_dict, basic_bool_func_dict)
#    if condition_value:
#        execute_code_block(node.body, environment, var_dict, basic_function_dict, basic_bool_func_dict)

def execute_while(node: WhileNode, environment: SimpleCompEnv, var_dict: dict[str, int], basic_function_dict: dict[str, Function], basic_bool_func_dict: dict[str, BoolFunction]):
    """ Execute a while loop """
    while execute_bool_expr(node.bool_expr, environment, var_dict, basic_function_dict, basic_bool_func_dict):
        execute_code_block(node.block, environment, var_dict, basic_function_dict, basic_bool_func_dict)

def execute_return(node: ReturnNode, environment: SimpleCompEnv, var_dict: dict[str, int], basic_function_dict: dict[str, Function], basic_bool_func_dict: dict[str, BoolFunction]):
    environment.assign_solution_object(var_dict[node.return_var_name])

# def execute_else(node: ElseNode, environment: SimpleCompEnv, var_dict: dict[str, int], basic_function_dict: dict[str, Function], basic_bool_func_dict: dict[str, BoolFunction]):
#    """ Execute an if statement """
#    execute_code_block(node.body, environment, var_dict, basic_function_dict, basic_bool_func_dict)

def execute_ifelse(node: IfElseNode, environment: SimpleCompEnv, var_dict: dict[str, int], basic_function_dict: dict[str, Function], basic_bool_func_dict: dict[str, BoolFunction]):
    condition_value = execute_bool_expr(node.bool_expr, environment, var_dict, basic_function_dict, basic_bool_func_dict)
    if condition_value:
        execute_code_block(node.if_block, environment, var_dict, basic_function_dict, basic_bool_func_dict)
    else:
        execute_code_block(node.else_block, environment, var_dict, basic_function_dict, basic_bool_func_dict)