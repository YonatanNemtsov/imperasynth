from core_lang_env.syntax_tree import *
from core_lang_env.parser import *
from core_lang_env.smart_exec_code import *
from core_lang_env.comp_env import *
import itertools

####### tool functions ########

def get_var_definition_positions(block: BlockNode, position: list[int], var_definitions):
    top_position = position[-1]
    statements = block.statements
    if top_position >= len(statements):
        return var_definitions
    statement = statements[top_position]
    if isinstance(statement, FunctionCallAssignNode):
        for i, x in enumerate(statement.var_names):
            if x not in var_definitions:
                var_definitions[x] = []
            var_definitions[x].append(position)

    if isinstance(statement, IfElseNode):
        get_var_definition_positions(statement.if_block, position + [1,0], var_definitions)
        get_var_definition_positions(statement.else_block, position + [2,0], var_definitions)
    
    if isinstance(statement, WhileNode):
        get_var_definition_positions(statement.block, position + [1,0], var_definitions)
    
    new_position = position[:-1] + [top_position + 1]
    get_var_definition_positions(block, new_position, var_definitions)
    return var_definitions

def is_available(pos, target_pos):
    if len(pos) == len(target_pos):
        return target_pos[:-1] == pos[:-1] and target_pos[-1] > pos[-1]
    if len(pos) < len(target_pos):
        return pos[:len(pos)-1] == target_pos[:len(pos)-1] and pos[len(pos)-1] <= target_pos[len(pos)-1]
    if len(pos) > len(target_pos):
        return False

def get_available_vars_at_position(block: BlockNode, position:list[int], input_vars: list[str]):
    var_positions = get_var_definition_positions(block, [0], {})
    available = set(input_vars)
    for x, positions in var_positions.items():
        for pos in positions:
            if is_available(pos, position):
                available.add(x)
    return available

def get_var_type(block: BlockNode, var_name: str, func_dict: dict[str, Function]) -> type:
    var_defs = get_var_definition_positions(block, [0], {})
    if not var_name in var_defs:
        return None
    position = var_defs[var_name][0]
    node: FunctionCallAssignNode = block.get_node_at_position(position)
    var_index = node.var_names.index(var_name)
    return func_dict[node.func_name].output_types[var_index]


#### AST Augmentation functions ####

def get_possible_insertion_positions(ast_tree: ASTNode) -> list[list[int]]:
    """
    Returns a list of positions where new nodes can be inserted (at the end of each BlockNode).
    Each position is represented as a list of indices representing the path to the insertion point.
    
    For example:
    - [0, 1, 2] means you can insert at position 2 in the block at path [0, 1]
    - [] means you can insert at the root level (if the root is a BlockNode)
    """
    positions = []
    
    def traverse(node: ASTNode, current_path: list[int]):
        if isinstance(node, BlockNode):
            if not ReturnNode in [type(statement) for statement in node.statements]:
                positions.append(current_path + [len(node.statements)])
            
        if isinstance(node, BlockNode):
            for i, child in enumerate(node.statements):
                traverse(child, current_path + [i])
        
        elif isinstance(node, WhileNode):
            traverse(node.block, current_path + [1])
        
        elif isinstance(node, RepeatNode):
            traverse(node.block, current_path + [0])
            
        elif isinstance(node, IfElseNode):
            traverse(node.if_block, current_path + [1])
            if isinstance(node, IfElseNode):
                traverse(node.else_block, current_path + [2])
    
    traverse(ast_tree, [])
    return positions



def get_possible_args(func: Function, available_var_types: dict[str, type]) -> list[list[str]]:
    matching_vars_per_param = []
    for param_type in func.input_types:
        matching_vars = [var for var, var_type in available_var_types.items() 
                        if var_type == param_type]
        
        if not matching_vars:
            return []
            
        matching_vars_per_param.append(matching_vars)
        
    return [list(combination) for combination in itertools.product(*matching_vars_per_param)]

def get_possible_output_names(func: Function, var_types: dict[str, type]) -> list[list[str]]:
    existing_nums = []
    for var in var_types:
        if var.startswith('x'):
            try:
                existing_nums.append(int(var[1:]))
            except ValueError:
                pass
    
    max_num = max(existing_nums, default=0)
    
    vars_by_type = {}
    for var, typ in var_types.items():
        vars_by_type.setdefault(typ, []).append(var)
    
    valid_combinations = []
    for new_vars_num in range(0, len(func.output_types) + 1):
        for new_var_positions in itertools.combinations(range(len(func.output_types)), new_vars_num):
            new_vars = [f'x{max_num + i + 1}' for i in range(new_vars_num)]
            possible_vars = []

            new_var_count = 0
            valid = True
            for idx, type in enumerate(func.output_types):
                if idx in new_var_positions:
                    possible_vars.append([new_vars[new_var_count]])
                    new_var_count += 1
                else:
                    if type in vars_by_type:
                        possible_vars.append(vars_by_type[type])
                    else:
                        valid = False
                        break
            if valid:
                valid_combinations += list(itertools.product(*possible_vars))

    return valid_combinations


def get_possible_func_assign(var_types: dict[str, type], available_vars: set[str], func_dict: dict[str, Function]) -> list[FunctionCallAssignNode]:
    possible_assignments = []
    available_var_types = {var: var_types[var] for var in available_vars if var in var_types}
    
    for func_name, func in func_dict.items():
        possible_args = get_possible_args(func, available_var_types)
        possible_outputs = get_possible_output_names(func, var_types)

        for args in possible_args:
            for outputs in possible_outputs:
                
                # TODO: make it tuples by default in FunctionCallAssignNode
                assign_node = FunctionCallAssignNode(
                    var_names=list(outputs),
                    func_name=func_name,
                    arg_names=list(args)
                )
                possible_assignments.append(assign_node)
    
    return possible_assignments


def get_possible_ifelse(var_types: dict[str, type], available_vars: set[str], bool_dict: dict[str, BoolFunction]) -> list[IfElseNode]:
    possible_ifelses = []
    available_var_types = {var: var_types[var] for var in available_vars if var in var_types}
    
    for bool_name, bool_func in bool_dict.items():
        possible_args = get_possible_args(bool_func, available_var_types)
        for args in possible_args:
            bool_expr = BoolExprNode(bool_name, args)
            if_block = BlockNode([])
            else_block = BlockNode([])
            ifelse_node = IfElseNode(bool_expr, if_block, else_block)
            possible_ifelses.append(ifelse_node)
    
    return possible_ifelses

def get_possible_while(var_types: dict[str, type], available_vars: set[str], bool_dict: dict[str, BoolFunction]) -> list[WhileNode]:
    possible_whiles = []
    available_var_types = {var: var_types[var] for var in available_vars if var in var_types}
    
    for bool_name, bool_func in bool_dict.items():
        possible_args = get_possible_args(bool_func, available_var_types)
        
        for args in possible_args:
            bool_expr = BoolExprNode(bool_name, args)
            while_block = BlockNode([])
            while_node = WhileNode(bool_expr, while_block)
            possible_whiles.append(while_node)
    
    return possible_whiles

def get_possible_repeat(var_types: dict[str, type], 
                       available_vars: set[str], 
                       repeat_counts: range = range(1, 11)) -> list[RepeatNode]:
    """
    Generate all possible repeat nodes with counts in specified range.
    The repeated block is initialized as an empty BlockNode that can be filled later.
    """
    possible_repeats = []
    
    for count in repeat_counts:
        # Create repeat node with empty block (to be filled during augmentation)
        repeat_node = RepeatNode(str(count), BlockNode([]))
        possible_repeats.append(repeat_node)
    
    return possible_repeats

def get_possible_direct_assign(var_types: dict[str, type], 
                              available_vars: set[str]) -> list[DirectAssignNode]:
    """
    Generate all possible direct assignment nodes between compatible variables.
    """
    possible_assignments = []
    
    # Get all variables with defined types
    typed_vars = [(var, typ) for var, typ in var_types.items() if var in available_vars]
    
    # Generate all valid target <- source pairs
    for target_var, target_type in typed_vars:
        for source_var, source_type in typed_vars:
            if source_var != target_var and source_type == target_type:
                possible_assignments.append(
                    DirectAssignNode(target_var, source_var)
                )
    
    return possible_assignments

def get_possible_returns(var_types: dict[str, type], available_vars: set[str], output_type: type) -> list[ReturnNode]:
    return [ReturnNode(var) for var in available_vars if var_types.get(var) == output_type]

##### central augmentation funciton #####

def get_possible_augmentations(ast_tree: ASTNode, input_var_types: dict[str, type], known_functions: dict[str, Function], known_bools: dict[str, BoolFunction], return_type: type) -> list[ASTNode]:
    possible_augmentations = []
    
    insertion_positions = get_possible_insertion_positions(ast_tree)
    for position in insertion_positions:
        parent = ast_tree.get_node_at_position(position[:-1])
        if not isinstance(parent, BlockNode):
            continue
        
        available_vars = get_available_vars_at_position(ast_tree, position, list(input_var_types.keys()))

        all_var_types = input_var_types.copy()
        var_defs = get_var_definition_positions(ast_tree, [0], {})
        # TODO: refactor
        for var, positions in var_defs.items():
            if positions and is_available(positions[0], position):
                def_pos = positions[0]
                def_node = ast_tree.get_node_at_position(def_pos)
                if isinstance(def_node, FunctionCallAssignNode):
                    var_idx = def_node.var_names.index(var)
                    func = known_functions[def_node.func_name]
                    all_var_types[var] = func.output_types[var_idx]
        
        func_assigns = get_possible_func_assign(all_var_types, available_vars, known_functions)
        for node in func_assigns:
            possible_augmentations.append((position, node, 'func'))
        
        ifelses = get_possible_ifelse(all_var_types, available_vars, known_bools)
        for node in ifelses:
            possible_augmentations.append((position, node, 'ifelse'))
        
        whiles = get_possible_while(all_var_types, available_vars, known_bools)
        for node in whiles:
            possible_augmentations.append((position, node, 'while'))

        direct_assigns = get_possible_direct_assign(all_var_types, available_vars)
        for node in direct_assigns:
            possible_augmentations.append((position, node, 'direct_assign'))
        
        repeats = get_possible_repeat(all_var_types, available_vars)
        for node in repeats:
            possible_augmentations.append((position, node, 'repeat'))

        returns = get_possible_returns(all_var_types, available_vars, return_type)
        for node in returns:
            possible_augmentations.append((position, node, 'return'))
    
    return possible_augmentations

############################ The Searcher ################################

class Problem:
    def __init__(self, input_types, output_types, examples: list[list]): 
        self.input_types = input_types
        self.output_types = output_types
        self.examples = examples

def get_complexity(ast: ASTNode) -> int:
    """
    Calculate complexity based on:
    - Total number of AST nodes
    - Number of unique variables used
    Returns: node_count + variable_count
    """
    if ast is None:
        return 0
    
    node_count = 0
    variables = set()
    
    def count_nodes(node: ASTNode):
        nonlocal node_count
        if node is None:
            return
        
        node_count += 1
        
        if isinstance(node, BlockNode):
            for stmt in node.statements:
                count_nodes(stmt)
                
        elif isinstance(node, FunctionCallAssignNode):
            variables.update(node.var_names)
            variables.update(node.arg_names)
            
        elif isinstance(node, BoolExprNode):
            variables.update(node.arg_names)
            
        elif isinstance(node, IfElseNode):
            node_count -= 1
            count_nodes(node.bool_expr)
            count_nodes(node.if_block)
            count_nodes(node.else_block)
            
        elif isinstance(node, WhileNode):
            count_nodes(node.bool_expr)
            count_nodes(node.block)
            
        elif isinstance(node, ReturnNode):
            if hasattr(node, 'return_var_name'):  # Handle both single and multi-return
                if isinstance(node.return_var_name, list):
                    variables.update(node.return_var_name)
                else:
                    variables.add(node.return_var_name)
    
    count_nodes(ast)
    return node_count + len(variables)

import heapq
import hashlib
from typing import Optional

def ast_hash(ast: ASTNode) -> str:
    """Create a unique hash for an AST to detect duplicates"""
    return hashlib.md5(str(ast).encode()).hexdigest()

def search_solution(problem: Problem, 
                   known_functions: dict, 
                   known_bools: dict, 
                   max_iterations: int = 5000,
                   max_complexity = 10,
                   max_eval_steps = 30) -> Optional[ASTNode]:
    """
    Priority-queue based search that:
    1. Always expands lowest-complexity programs first
    2. Maintains comprehensive visited tracking
    3. Provides detailed progress reporting
    4. Properly tracks complexity statistics
    """
    # Priority queue: (complexity, ast_hash, ast)
    heap = []
    heapq.heappush(heap, (0, ast_hash(BlockNode([])), BlockNode([])))
    
    visited = set()  # Tracks visited ast_hashes
    best = (None, -1)  # (best_ast, best_score)
    
    # Statistics tracking
    stats = {
        'generated': 0,
        'evaluated': 0,
        'duplicates': 0,
        'max_complexity_seen': 0,  # Highest complexity encountered
        'current_complexity': 0,   # Complexity being explored now
        'last_reported_complexity': -1  # For progress reporting
    }

    for iteration in range(max_iterations):
        if not heap:
            print("Search queue exhausted")
            break
            
        # Get lowest-complexity program
        current_complexity, _, current_ast = heapq.heappop(heap)
        # print(current_ast, current_complexity)
        current_hash = ast_hash(current_ast)
        stats['current_complexity'] = current_complexity
        
        # Update max complexity seen
        if current_complexity > stats['max_complexity_seen']:
            stats['max_complexity_seen'] = current_complexity
            if current_complexity > stats['last_reported_complexity'] + 4:
                print(f"Reached new complexity level: {current_complexity}")
                stats['last_reported_complexity'] = current_complexity
        
        # Skip duplicates
        if current_hash in visited:
            stats['duplicates'] += 1
            continue
            
        visited.add(current_hash)
        stats['evaluated'] += 1
        
        # Evaluate program
        score, solved = evaluate_program(
            current_ast,
            problem,
            known_functions,
            known_bools,
            max_steps=max_eval_steps
        )
        
        # Progress reporting
        if iteration % 100 == 0:
            print(f"\nIteration {iteration}:")
            print(f"  Current complexity: {current_complexity}")
            print(f"  Max complexity seen: {stats['max_complexity_seen']}")
            print(f"  Best score so far: {best[1]:.2f}")
            print(f"  Queue size: {len(heap)}")
            print(f"  Stats: {stats}")
            # print(ast_to_code_str(current_ast))
        
        # Check for solution
        if solved:
            print(f"\nSolution found after {iteration} iterations!")
            print(f"Final complexity: {current_complexity}")
            print(f"Program: {current_ast}")
            return current_ast
        
        # Update best solution
        if score > best[1]:
            best = (current_ast, score)
            print(f"\nNew best score: {score:.2f} at complexity {current_complexity}")
            print(f"Program: {current_ast}")
        
        # Generate successors
        augmentations = get_possible_augmentations(
            current_ast,
            problem.input_types,
            known_functions,
            known_bools,
            problem.output_types[0]
        )
        
        for pos, new_node, _ in augmentations:
            try:
                new_ast = current_ast.insert_node(pos, new_node)
                new_complexity = get_complexity(new_ast)
                if new_complexity > max_complexity:
                    continue
                new_hash = ast_hash(new_ast)
                
                if new_hash not in visited:
                    heapq.heappush(heap, (new_complexity, new_hash, new_ast))
                    stats['generated'] += 1
                    
            except Exception as e:
                continue
    
    print("\nSearch completed without finding perfect solution")
    print(f"Best solution found (score={best[1]:.2f}) at complexity {get_complexity(best[0])}:")
    print(best[0])
    return best[0]

def evaluate_program(ast: ASTNode,
                    problem: Problem,
                    known_functions: dict,
                    known_bools: dict,
                    max_steps: int = 100) -> tuple[float, bool]:
    """Robust evaluation with detailed error handling"""
    correct = 0
    
    for inputs, expected in problem.examples:
        try:
            env = SimpleCompEnv(known_functions)
            var_dict = {}
            
            # Initialize inputs
            for name, value in zip(problem.input_types.keys(), inputs):
                var_dict[name] = env.add_input_object(CompObject(problem.input_types[name], value))
            
            # Execute with step limit
            controller = ProgramController(ast, env, var_dict, known_functions, known_bools)
            for _ in range(max_steps):
                done, _, _ = controller.step()
                if done:
                    break
            
            # Verify result
            if env.solution_object_id is None:
                return (0.0, False)
                
            result = env.objects[env.solution_object_id].value
            if result != expected:
                return (correct / len(problem.examples), False)
                
            correct += 1
            
        except Exception:
            print('fail')
            print(env, controller)
            return (0.0, False)
    
    score = correct / len(problem.examples)
    return (score, score == 1.0)