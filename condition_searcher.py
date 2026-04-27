from core_lang_env.comp_env import *
from core_lang_env.exec_code_v2 import *
from core_lang_env.parser import *
from core_lang_env.syntax_tree import *
from searchers_utils import *

""" Search boolean traces """

############################## Extract problem from annotated ast ###################################

def get_conditional_problem_instances_from_annotated_ast_and_ifelse_node_position(annotated_ast: AnnotatedAST, node_position: ASTNodePosition, trace: SimpleCompEnv):
    """
    Build training instances (input values + True/False labels)
    for a conditional node (IfElseNode or WhileNode) from an annotated AST.

    Each instance is a tuple:
        ((val1, val2, ..., valN), condition_result)
    """

    node = annotated_ast.ast.get_node_at_position(node_position)
    assert isinstance(node, IfElseNode)
    exec_positions = get_if_else_node_exec_positions(annotated_ast, node_position)

    # --- Extract variable states ---
    true_var_states = [annotated_ast.get_variable_state_at_position(state) for state in exec_positions["entries"]]
    false_var_states = [annotated_ast.get_variable_state_at_position(state) for state in exec_positions["skips"]]

    # --- Collect all variable names to ensure consistent ordering ---
    all_vars = set.intersection(*[set(s.keys()) for s in true_var_states + false_var_states]) if (true_var_states or false_var_states) else set()
    ordered_var_names = sorted(all_vars)

    # --- Build true and false instances ---
    true_instances = []
    input_types = None
    for var_state in true_var_states:
        if input_types == None:
            input_types = tuple(trace.objects[var_state[v]].value_type for v in ordered_var_names)
        instance_vals = tuple( trace.objects[var_state[v]].value for v in ordered_var_names)
        true_instances.append((instance_vals, True))

    false_instances = []
    for var_state in false_var_states:
        instance_vals = tuple(trace.objects[var_state[v]].value for v in ordered_var_names)
        false_instances.append((instance_vals, False))

    return {"input_types": input_types, "entries": true_instances, "skips": false_instances}, ordered_var_names

def extract_ifelse_conditional_problem(annotated_ast: AnnotatedAST, ifelse_node_position: ASTNodePosition, trace: SimpleCompEnv) -> Problem | None:
    """
    Create a Problem describing the conditional in an if-else node.
    Inputs: variables visible at the conditional point.
    Output: boolean (True if if-block executed, False if else-block executed).
    """
    node = annotated_ast.ast.get_node_at_position(ifelse_node_position)
    if not isinstance(node, IfElseNode):
        raise TypeError("Node at position is not an IfElseNode")

    exec_positions = get_if_else_node_exec_positions(annotated_ast, ifelse_node_position)

    # --- Collect variable states ---
    true_var_states = [annotated_ast.get_variable_state_at_position(state) for state in exec_positions["entries"]]
    false_var_states = [annotated_ast.get_variable_state_at_position(state) for state in exec_positions["skips"]]

    if not true_var_states and not false_var_states:
        return None  # no data to learn from

    # --- Compute intersection of variable names ---
    var_names = sorted(list(set.intersection(*[set(s.keys()) for s in true_var_states + false_var_states])))

    # --- Gather input-output instances ---
    instances = {}
    index = 0
    for state, label in [(true_var_states, True), (false_var_states, False)]:
        for var_state in state:
            inputs = tuple(trace.objects[var_state[v]].value for v in var_names)
            output = (label,)
            instances[index] = (inputs, output)
            index += 1

    # --- Determine input and output types ---
    if instances:
        sample_inputs, sample_output = next(iter(instances.values()))
        input_types = tuple(type(x) for x in sample_inputs)
        output_type = type(sample_output[0])
    else:
        input_types = ()
        output_type = bool

    return Problem(input_types, output_type, instances), var_names

def extract_ifelse_conditional_problem_for_group(annotated_asts: list[AnnotatedAST], ifelse_node_position: ASTNodePosition, traces: list[SimpleCompEnv]) -> tuple[Problem | None, list[str]]:
    """
    Build one boolean-learning Problem from multiple annotated ASTs
    and their corresponding traces.

    Returns:
        (Problem or None, var_names)
    """

    assert len(annotated_asts) == len(traces)

    merged_problem = None
    var_names = None

    for ann, trace in zip(annotated_asts, traces):

        result = extract_ifelse_conditional_problem(ann, ifelse_node_position, trace)

        if result is None:
            continue

        sub_problem, sub_var_names = result

        if merged_problem is None:
            merged_problem = sub_problem
            var_names = tuple(sub_var_names)
            continue

        if tuple(sub_var_names) != var_names:
            continue

        if sub_problem.input_types != merged_problem.input_types:
            continue

        if sub_problem.output_types != merged_problem.output_types:
            continue

        offset = len(merged_problem.instances)
        for k, inst in sub_problem.instances.items():
            merged_problem.instances[offset + k] = inst

    if merged_problem is None:
        return None, []

    return merged_problem, list(var_names)


def extract_while_conditional_problem(annotated_ast: AnnotatedAST, while_node_position: ASTNodePosition, trace: SimpleCompEnv) -> Problem | None:
    node = annotated_ast.ast.get_node_at_position(while_node_position)
    if not isinstance(node, WhileNode):
        raise TypeError("Node at position is not an WhileNode")

    exec_position_groups = get_while_node_exec_position_groups(annotated_ast, while_node_position)

        # --- Collect variable states ---
    true_var_states = [[annotated_ast.get_variable_state_at_position(state) for state in exec_positions["entries"]] for exec_positions in exec_position_groups]
    false_var_states = [[annotated_ast.get_variable_state_at_position(state) for state in exec_positions["skips"]] for exec_positions in exec_position_groups]

    if not true_var_states and not false_var_states:
        return None  # no data to learn from

    # --- Compute intersection of variable names ---
        # --- Compute intersection of variable names ---
    # Flatten all true/false states across all loop executions
    all_var_states = [vs for group in (true_var_states + false_var_states) for vs in group]
    if not all_var_states:
        return None

    var_names = sorted(list(set.intersection(*[set(s.keys()) for s in all_var_states])))

    # --- Build problems for each execution group ---
    problem_group = []
    for exec_index, (true_states, false_states) in enumerate(zip(true_var_states, false_var_states)):
        instances = {}
        index = 0
        for state_group, label in [(true_states, True), (false_states, False)]:
            for var_state in state_group:
                inputs = tuple(trace.objects[var_state[v]].value for v in var_names)
                output = (label,)
                instances[index] = (inputs, output)
                index += 1

        if instances:
            sample_inputs, sample_output = next(iter(instances.values()))
            input_types = tuple(type(x) for x in sample_inputs)
            output_type = type(sample_output[0])
        else:
            input_types = ()
            output_type = bool

        problem_group.append(InterDependentProblem(input_types, output_type, instances))

    if not problem_group:
        return None

    return InterDependentProblemGroup(problem_group), var_names








######################## search_boolean_traces ######################

import itertools

def create_maps_for_conditional_problem(problem: Problem, known_funcs: dict[FunctionName, Function], depth: int) -> list[ComputationalMap]:
    """
    For each instance in the Problem, create a ComputationalMap that:
      - starts from the instance's input objects
      - expands outward using known functions up to `depth` steps
    Returns: list of ComputationalMaps, one per instance.
    """
    maps = []

    for idx, (inputs, outputs) in problem.instances.items():
        # --- Step 1: initialize an empty map for this instance ---
        comp_map = ComputationalMap.init_empty_map()
        comp_map.functions = known_funcs.copy()

        # Initialize with input objects
        for t, obj in zip(problem.input_types, inputs):
            comp_map.objects[(t,obj)] = set()

        # --- Step 2: expand outward up to the given depth ---
        current_objects = set(zip(problem.input_types, inputs))
        for _ in range(depth):
            new_objects = set()
            for func_name, func in comp_map.functions.items():
                # Generate all possible argument tuples matching input types
                input_candidates = tuple(
                    tuple(obj for obj_type, obj in current_objects if obj_type == t)
                    for t in func.input_types
                )

                if not all(input_candidates):
                    continue  # skip if we don't have compatible objects

                for args in itertools.product(*input_candidates):
                    # Avoid repeating known actions
                    action_key = (func_name,) + tuple(args)
                    if action_key in comp_map.actions:
                        continue
                    try:
                        outputs = comp_map.act(func_name, args)
                        for t, out in zip(comp_map.functions[func_name].output_types, outputs):
                            new_objects.add((t, out))
                            #('object_added')
                    except Exception:
                        print("something wrong")
                        continue

            # If no new objects were generated, stop expanding
            #if not new_objects:
            #    break
            current_objects |= new_objects

        maps.append(comp_map)
    clean_maps = []
    for idx, comp_map in enumerate(maps):
        clean_map = extract_minimal_subgraph(comp_map, problem.instances[idx][1][0])
        clean_maps.append(clean_map)

    return clean_maps

"""
def get_all_available_actions(trace_env: SimpleCompEnv, comp_map: ComputationalMap) -> list[ShortAction]:
    value_to_ids = {}

    for obj_id, obj in trace_env.objects.items():
        value_to_ids.setdefault((obj.value_type, obj.value), []).append(obj_id)

    available_actions = []
    for action_key in comp_map.actions:
        func_name, *input_vals = action_key
        if all((type(val), val) in value_to_ids for val in input_vals):
            for input_ids in product(*(value_to_ids[(type(val), val)] for val in input_vals)):
                candidate_action = (func_name, input_ids)

                if candidate_action in trace_env.action_history_short:
                    continue

                available_actions.append(candidate_action)
    return available_actions
"""

def get_all_available_actions(trace_env: SimpleCompEnv, comp_map: ComputationalMap) -> list[ShortAction]:
    value_to_ids = {}
    for obj_id, obj in trace_env.objects.items():
        value_to_ids.setdefault((obj.value_type, obj.value), []).append(obj_id)

    available_actions = []
    for action_key in comp_map.actions:
        func_name, *input_vals = action_key
        func = comp_map.functions[func_name]
        typed_inputs = list(zip(func.input_types, input_vals))

        # Only allow actions if all input (type, value) exist in current trace
        if all(tv in value_to_ids for tv in typed_inputs):
            for input_ids in product(*(value_to_ids[tv] for tv in typed_inputs)):
                candidate = (func_name, input_ids)
                if candidate not in trace_env.action_history_short:
                    available_actions.append(candidate)

    return available_actions



import itertools

def search_boolean_traces(bool_problem: Problem, known_funcs: dict[FunctionName, Function], max_depth: int):
    traces = []
    instance_items = list(bool_problem.instances.items())
    for instance_id, (inputs, expected) in instance_items:
        env = SimpleCompEnv(known_funcs.copy())
        for val in inputs:
            env.add_input_object(CompObject(type(val), val))
        traces.append(env)
        # print(env.objects)
    
    maps = create_maps_for_conditional_problem(bool_problem, known_funcs, max_depth)
    print(maps)
    obj_dependencies = {obj_id: set() for obj_id in traces[0].input_objects}
    for depth in range(max_depth):
        available_actions = set.intersection(*[set(get_all_available_actions(trace, comp_map)) for trace, comp_map in zip(traces, maps)])
        #print(len(available_actions))

        for action in available_actions:
            func_name, input_ids = action

            # --- 3c. Compute depth of this action ---
            action_deps_union = set().union(*(obj_dependencies[obj_id] for obj_id in input_ids))
            action_depth = len(action_deps_union)
            if action_depth > max_depth:
                continue  # skip action exceeding max_depth

            # apply action otherwise
            for trace_idx, trace in enumerate(traces):
                output_ids = trace.apply_function(func_name, input_ids)
            
            # update dependencies for new objects
            for out_id in output_ids:
                obj_dependencies[out_id] = action_deps_union.copy()
                obj_dependencies[out_id].add(action)

            all_solved = True
            for trace_idx, (_, (_, (expected_bool,))) in enumerate(instance_items):
                trace = traces[trace_idx]
                last_action = trace.action_history[-1]
                output_values = [trace.objects[obj_id].value for obj_id in last_action[2]]
                # print(expected_bool[0], output_values[0])
                #TODO: rewrite this so it takes output index into account.
                if not any(type(v) == bool and v == expected_bool for v in output_values):
                    all_solved = False
                    break
                else:
                    trace.assign_solution_object(last_action[2][0])

            if all_solved:
                print('solved')
                clean_traces = [get_clean_comp_env(trace) for trace in traces]
                return clean_traces # ,obj_dependencies  # stop immediately, solution found
    
    return traces  # return traces after search ends

############################### search_boolean_traces_with_interdependence ######################

def search_boolean_traces_with_interdependence(bool_problem: Problem, dependent_variables_input, known_funcs: dict[FunctionName, Function], max_depth: int):
    """ 
    Sketch of the algorithm:
    first, search a transformation scheme for the dependent variables. then, create maps with inputs as in the problem + scheme.
    then, conduct a search with those maps in the same manner as search_boolean_traces, except now we can use the constraint that 
    the dependent variables must be used in the traces, since otherwise there is no point to have them, and there is a simple 
    solution using search_boolean_traces. Now, if a solution isn't found with the maps we got, we go on to another dependent variable
    scheme, (increasing in complexity level of the transformation if all transformations of the lesser level are tried already). 
    the scheme is assumed to be quite simple, so the number of iterations shouldn't be large.

    TODO: Implement. ~100 - 400 lines
    """


############################### bundle into a program expression ######################

def create_bool_program_expressions(trace: SimpleCompEnv,  input_var_names: list[str]) -> list[FunctionCallAssignNode | BoolExprNode]:
    var_dict = {inp_id: inp_name for inp_id, inp_name in zip(trace.input_objects, input_var_names)} # initialize var_dict
    aux_var_num = 0
    nodes = []
    for func_name, arg_ids, output_ids in trace.action_history[:-1]:
        arg_names = [var_dict[arg_id] for arg_id in arg_ids]
        out_var_names = [f'b{i}' for i in range(aux_var_num, aux_var_num + len(output_ids))]
        aux_var_num += len(output_ids)
        var_dict.update({out_id: out_var for out_id, out_var in zip(output_ids, out_var_names)})
        node = FunctionCallAssignNode(out_var_names, func_name, arg_names)
        nodes.append(node)
    
    last_func_name, arg_ids = trace.action_history_short[-1]
    arg_names = [var_dict[arg_id] for arg_id in arg_ids]

    bool_expr_node = BoolExprNode(last_func_name, arg_names)
    nodes.append(bool_expr_node)
    return nodes



####################### TODO: make this into a nice class structure, with good interface etc #############################

class ConditionSearcher:
    def __init__(self, problem: Problem, general_funcs: dict[FunctionName, Function], bool_functions: dict[FunctionName, BoolFunction]):
        pass

    