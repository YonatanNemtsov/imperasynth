from .syntax_tree import ASTNode, BoolExprNode, WhileNode, IfNode, BlockNode, FunctionCallAssignNode, ReturnNode, ElseNode, IfElseNode, DirectAssignNode, RepeatNode

from lark import Lark, Tree, Token

# Define the grammar
grammar = """
    start: block

    block: "{" statement* "}"

    statement: ifelse_statement
             | while_statement
             | repeat_statement
             | function_call_assign
             | direct_assign
             | block
             | return_statement
    
    direct_assign: var_list "<-" var_list ";"
    ifelse_statement: if_statement else_statement

    if_statement: "if" "(" bool_expr ")" block
    else_statement: "else" block
    
    while_statement: "while" "(" bool_expr ")" block
    repeat_statement: "repeat" "(" INT ")" block

    return_statement: "return" [var_list] ";"
    function_call_assign: var_list "=" func_name "(" arg_list ")" ";" | var_list "=" func_name "()"
    var_list: CNAME ("," CNAME)*
    arg_list: CNAME ("," CNAME)*

    bool_expr: func_name "(" arg_list ")" | func_name "()"

    func_name: CNAME
    CNAME: /[a-zA-Z_][a-zA-Z0-9_]*/
    INT: /[0-9]+/
    %import common.WS
    %ignore WS
"""

# Function to translate Lark's tree into the custom AST
def translate_to_custom_ast(tree):
    if tree.data == "start":
        return translate_to_custom_ast(tree.children[0])  # Unwrap the start node
    
    elif tree.data == "statement":
        tree = tree.children[0]
    
    if tree.data == "block":
        statements = [translate_to_custom_ast(child) for child in tree.children]
        return BlockNode(statements)
    
    elif tree.data == "direct_assign":
        target_vars = translate_to_custom_ast(tree.children[0])  # First var_list
        source_vars = translate_to_custom_ast(tree.children[1])  # Second var_list
        return DirectAssignNode(target_vars, source_vars)
    
    elif tree.data == "ifelse_statement":
        # First child is always the if_statement
        if_tree = tree.children[0]
        bool_expr = translate_to_custom_ast(if_tree.children[0])
        if_block = translate_to_custom_ast(if_tree.children[1])
        
        # Second child exists only if there's an else clause
        else_block = None
        else_tree = tree.children[1]  # else_statement
        else_block = translate_to_custom_ast(else_tree.children[0])  # The block inside else
            
        return IfElseNode(bool_expr, if_block, else_block)
    
    # Keep these for backward compatibility (can remove later)
    # elif tree.data == "if_statement":
    #    bool_expr = translate_to_custom_ast(tree.children[0])
    #    block = translate_to_custom_ast(tree.children[1])
    #    return IfElseNode(bool_expr, block, None)  # No else block
    
    elif tree.data == "else_statement":
        # Should only be reached when part of ifelse_statement
        return translate_to_custom_ast(tree.children[0])
    
    elif tree.data == "while_statement":
        bool_expr = translate_to_custom_ast(tree.children[0])
        block = translate_to_custom_ast(tree.children[1])
        return WhileNode(bool_expr, block)
    
    elif tree.data == "repeat_statement":
        count_node = tree.children[0]
        block_node = translate_to_custom_ast(tree.children[1])
        
        # Handle both variable name and integer literal cases
        if hasattr(count_node, 'type'):
            count_value = int(count_node.value)
        else:
            count_value = count_node
        
        return RepeatNode(count_value, block_node)

    elif tree.data == "function_call_assign":
        var_names = translate_to_custom_ast(tree.children[0])
        func_name = translate_to_custom_ast(tree.children[1])
        arg_names = translate_to_custom_ast(tree.children[2])
        return FunctionCallAssignNode(var_names, func_name, arg_names)

    elif tree.data == "bool_expr":
        func_name = translate_to_custom_ast(tree.children[0])
        if len(tree.children) == 2:
            arg_names = translate_to_custom_ast(tree.children[1])
        else:
            arg_names = ()
        return BoolExprNode(func_name, arg_names)
    
    elif tree.data == "return_statement":
        arg_name = translate_to_custom_ast(tree.children[0])[0]
        return ReturnNode(arg_name)
    
    elif tree.data == "var_list":
        return tuple(child.value for child in tree.children if isinstance(child, Token))

    elif tree.data == "arg_list":
        return tuple(child.value for child in tree.children if isinstance(child, Token))

    elif tree.data == "func_name":
        return tree.children[0].value  # Extract the function name

    elif isinstance(tree, Token):
        return tree.value  # Handle tokens (e.g., variable names)

    else:
        raise ValueError(f"Unknown tree node: {tree.data}")

def parse_code_str(code_string):
    parser = Lark(grammar, parser='lalr')
    lark_tree = parser.parse(code_string)
    custom_ast = translate_to_custom_ast(lark_tree)
    return custom_ast

def ast_to_code_str(ast: ASTNode, indent_level: int = 0) -> str:
    INDENT = '    '  # 4 spaces per indent level
    
    if isinstance(ast, BlockNode):
        if not ast.statements:
            return '{}'
            
        statements = []
        for stmt in ast.statements:
            stmt_code = ast_to_code_str(stmt, indent_level + 1)
            statements.append(INDENT * (indent_level + 1) + stmt_code)
        
        return '{\n' + '\n'.join(statements) + '\n' + INDENT * indent_level + '}'
    
    elif isinstance(ast, FunctionCallAssignNode):
        vars_str = ', '.join(ast.var_names)
        args_str = ', '.join(ast.arg_names)
        return f"{vars_str} = {ast.func_name}({args_str});"
    
    elif isinstance(ast, BoolExprNode):
        args_str = ', '.join(ast.arg_names)
        return f"{ast.bool_func}({args_str})"
    
    elif isinstance(ast, IfElseNode):
        condition = ast_to_code_str(ast.bool_expr, indent_level)
        if_block = ast_to_code_str(ast.if_block, indent_level)
        
        code = f"if ({condition}) {if_block}"
        
        if ast.else_block:
            else_block = ast_to_code_str(ast.else_block, indent_level)
            code += f" else {else_block}"
            
        return code
    
    elif isinstance(ast, WhileNode):
        condition = ast_to_code_str(ast.bool_expr, indent_level)
        block = ast_to_code_str(ast.block, indent_level)
        return f"while ({condition}) {block}"
    
    elif isinstance(ast, DirectAssignNode):
        targets = ", ".join(ast.target_vars)
        sources = ", ".join(ast.source_vars)
        return f"{targets} <- {sources};"
    
    elif isinstance(ast, RepeatNode):
        block = ast_to_code_str(ast.block, indent_level)
        return f"repeat ({ast.count_var}) {block}"
    
    elif isinstance(ast, ReturnNode):
        return f"return {ast.return_var_name};"
    
    else:
        raise ValueError(f"Unsupported AST node type: {type(ast).__name__}")
    
################ Example #######################

def example():
    # Parse input code into ASTNode tree.
    input_code = """
    {
        if (is_valid(x1, x2)) {
            x3, x4 = basic_func(x1, x2);
        }
        else {
        }

        x3, x2 <- x4, x1;
        while (is_ready(x3)) {
            x5 = another_func(x3);
        }
        
        return x5;
    }
    """

    custom_ast = parse_code_str(input_code)
    print(custom_ast.statements)

if __name__ == '__main__':
    example()