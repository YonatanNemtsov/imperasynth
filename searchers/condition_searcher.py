from core_lang_env.comp_env import *
from core_lang_env.exec_code_v2 import *
from core_lang_env.parser import *
from core_lang_env.syntax_tree import *
from .searchers_utils import *

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


TBD_CONDITIONAL_STR = "TBD_CONDITIONAL"


def _eval_bool_expr(bool_expr: BoolExprNode, sim_values: dict, known_bools: dict) -> bool | None:
    """Evaluate a BoolExprNode against sim_values. Returns None if unevaluable
    (TBD condition, missing var, or unknown bool function)."""
    if bool_expr.bool_func == TBD_CONDITIONAL_STR:
        return None
    bf = (known_bools or {}).get(bool_expr.bool_func)
    if bf is None:
        return None
    try:
        args = tuple(sim_values[a] for a in bool_expr.arg_names)
    except KeyError:
        return None
    try:
        out = bf.func(*args)
    except Exception:
        return None
    if isinstance(out, (tuple, list)):
        out = out[0]
    return bool(out)


def _simulate_block_until(block: BlockNode, target_idx: int, sim_values: dict,
                          known_funcs: dict, known_bools: dict | None = None) -> bool:
    """Simulate `block.statements[0..target_idx-1]` in place on sim_values.
    Returns True on success, False if any statement fails. Handles
    FunctionCallAssign, DirectAssign, IfElseNode (evaluates the condition and
    recurses into the chosen branch), and WhileNode (iterates until condition
    is false). Returns False on any unevaluable condition or runtime error.
    """
    for stmt in block.statements[:target_idx]:
        if isinstance(stmt, (FunctionCallAssignNode, DirectAssignNode)):
            if not simulate_simple_stmt(stmt, sim_values, known_funcs):
                return False
        elif isinstance(stmt, IfElseNode):
            cond = _eval_bool_expr(stmt.bool_expr, sim_values, known_bools)
            if cond is None:
                return False
            chosen = stmt.if_block if cond else stmt.else_block
            if not _simulate_block_until(chosen, len(chosen.statements), sim_values, known_funcs, known_bools):
                return False
        elif isinstance(stmt, WhileNode):
            for _ in range(1000):
                cond = _eval_bool_expr(stmt.bool_expr, sim_values, known_bools)
                if cond is None:
                    return False
                if not cond:
                    break
                if not _simulate_block_until(stmt.block, len(stmt.block.statements), sim_values, known_funcs, known_bools):
                    return False
            else:
                return False  # didn't terminate
        else:
            return False
    return True


def _simulate_while_iterations(body: BlockNode, sim_values: dict, known_funcs: dict,
                               known_bools: dict | None = None, max_iters: int = 1000):
    """Run the while body iteration by iteration on sim_values, recording each
    iteration's pre-iteration state as an "entry". Stop when the body fails
    to run (would crash on next iteration); record that failure-triggering
    state as the "skip". Returns (entries, skip) — the skip is the var_state
    when the loop should exit, or None if the simulation didn't terminate.
    """
    entries = []
    for _ in range(max_iters):
        before = dict(sim_values)
        # Run a copy of body on sim_values; if it fails, this iteration is the skip.
        trial = dict(sim_values)
        ok = _simulate_block_until(body, len(body.statements), trial, known_funcs, known_bools)
        if not ok:
            return entries, before  # body would fail → exit before this iteration
        entries.append(before)
        sim_values.clear()
        sim_values.update(trial)
    return entries, None  # didn't terminate within max_iters


def extract_while_conditional_problem_for_group(
    ast: BlockNode,
    while_node_position: ASTNodePosition,
    traces: list[SimpleCompEnv],
    known_funcs: dict,
    input_var_states: list[dict[str, "ObjId"]],
    known_bools: dict | None = None,
) -> tuple[Problem | None, list[str]]:
    """Build a flat boolean-learning Problem for a while loop's condition by
    simulating the program forward for each trace from its initial inputs.

    Walks the top-level block up to the while loop, then iterates the body
    until it would fail. Each iteration's pre-iteration variable state is a
    True instance; the failure-triggering state is the False instance.

    Differs from `extract_while_conditional_problem`: doesn't read the
    AnnotatedAST (which is frozen at build phase) — the simulation gives us
    accurate execute-phase iteration data.
    """
    while_node = ast.get_node_at_position(while_node_position)
    if not isinstance(while_node, WhileNode):
        raise TypeError("Node at position is not a WhileNode")

    # Only top-level while loops handled here — extend later for nested.
    if len(while_node_position) != 1:
        return None, []

    while_idx = while_node_position[0]
    body = while_node.block

    all_entries: list[dict] = []
    all_skips: list[dict] = []

    for trace, init_state in zip(traces, input_var_states):
        sim_values = {k: trace.objects[v].value for k, v in init_state.items()}
        if not _simulate_block_until(ast, while_idx, sim_values, known_funcs, known_bools):
            continue
        entries, skip = _simulate_while_iterations(body, sim_values, known_funcs, known_bools)
        if skip is None:
            # Body would iterate forever — no condition can make this loop
            # terminate, so the program is invalid. Reject.
            return None, []
        all_entries.extend(entries)
        all_skips.append(skip)

    if not all_entries and not all_skips:
        return None, []

    # Variable names: intersection of keys present in every entry/skip state.
    all_states = all_entries + all_skips
    var_names = sorted(set.intersection(*(set(s.keys()) for s in all_states)))
    if not var_names:
        return None, []

    instances = {}
    idx = 0
    for state, label in [(s, True) for s in all_entries] + [(s, False) for s in all_skips]:
        inputs = tuple(state[v] for v in var_names)
        instances[idx] = (inputs, (label,))
        idx += 1

    sample_inputs, _ = next(iter(instances.values()))
    input_types = tuple(type(x) for x in sample_inputs)
    return Problem(input_types, bool, instances), var_names






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



######################## BoolSearchEnvironment ######################


class BoolSearchEnvironment:
    """Stateful, value-canonicalized bool expression search.

    Each "object" in this env is identified by its per-trace value tuple,
    not by a SimpleCompEnv-style canonical id. Two expressions producing
    the same value tuple (e.g. `add(x, x)` and `mult(x, 2)` when 2 is
    available) collapse to one node — the dominant cost saver for
    realizability queries on rich function spaces.

    `realizable_partitions(value_tuples)` returns {True-set: stmts} for
    a given list of input value tuples (one per trace, each a tuple of
    typed values in canonical input order). Cached by the value tuples
    themselves — same in-scope values across different search states /
    variable names hit the same cache entry."""

    def __init__(self, known_funcs: dict, known_bools: dict, max_depth: int = 2):
        self.known_funcs = known_funcs
        self.known_bools = known_bools
        self.max_depth = max_depth
        # key: tuple(value_tuple_per_input) → {true_set: list[stmts]}
        self._cache: dict = {}

    def realizable_partitions(
        self,
        value_tuples_per_input: tuple,
        input_var_names: list[str],
    ) -> dict:
        """value_tuples_per_input: tuple of length num_inputs. Each entry
        is a tuple of length n_traces giving the input's value in each trace.
        input_var_names: names assigned to each input in the realizing
        expressions. Returns {frozenset[trace_idx]: list[stmts]}."""
        cache_key = value_tuples_per_input
        if cache_key in self._cache:
            return self._cache[cache_key]
        result = self._enumerate(value_tuples_per_input, input_var_names)
        self._cache[cache_key] = result
        return result

    def _enumerate(self, value_tuples_per_input, input_var_names):
        all_funcs = {**self.known_funcs, **self.known_bools}
        n_inputs = len(value_tuples_per_input)
        if n_inputs == 0 or any(len(vt) == 0 for vt in value_tuples_per_input):
            return {}
        n_traces = len(value_tuples_per_input[0])

        # Each canonical "node" is identified by its per-trace value tuple.
        # nodes: list of (value_tuple, type, depth, parent) — parent is
        # (func_name, input_node_ids, output_idx) or None for inputs.
        nodes: list = []
        value_to_node: dict = {}
        nodes_by_type: dict = {}

        def add_node(value_tuple, t, depth, parent):
            # Key by (type, value_tuple): without the type, Python's
            # bool↔int equivalence (False == 0, True == 1) collapses bool
            # outputs onto int input nodes when values coincidentally match.
            key = (t, value_tuple)
            if key in value_to_node:
                return value_to_node[key]
            nid = len(nodes)
            nodes.append((value_tuple, t, depth, parent))
            value_to_node[key] = nid
            nodes_by_type.setdefault(t, []).append(nid)
            return nid

        # Inputs
        for i, vt in enumerate(value_tuples_per_input):
            t = type(vt[0]) if vt else type(None)
            add_node(tuple(vt), t, 0, None)

        realizable: dict = {}
        applied_action_sigs: set = set()

        for depth in range(1, self.max_depth + 1):
            progressed = False
            for func_name, func in all_funcs.items():
                slots = [nodes_by_type.get(t, []) for t in func.input_types]
                if func.input_types and not all(slots):
                    continue
                arg_id_combos = product(*slots) if func.input_types else [()]
                for input_ids in arg_id_combos:
                    in_depths = [nodes[i][2] for i in input_ids]
                    action_depth = (max(in_depths) if in_depths else 0) + 1
                    if action_depth != depth:
                        continue
                    sig = (func_name, input_ids)
                    if sig in applied_action_sigs:
                        continue
                    applied_action_sigs.add(sig)

                    # Apply per-trace; build output value tuples.
                    input_value_tuples = [nodes[i][0] for i in input_ids]
                    per_trace_outputs = []
                    failed = False
                    for tr in range(n_traces):
                        args = tuple(ivt[tr] for ivt in input_value_tuples)
                        try:
                            outs = func.func(*args)
                        except Exception:
                            failed = True
                            break
                        if not isinstance(outs, (tuple, list)):
                            outs = (outs,)
                        per_trace_outputs.append(outs)
                    if failed:
                        continue

                    num_outs = min(len(o) for o in per_trace_outputs) if per_trace_outputs else 0
                    if num_outs == 0:
                        continue
                    progressed = True

                    for out_idx in range(num_outs):
                        out_vt = tuple(per_trace_outputs[tr][out_idx] for tr in range(n_traces))
                        out_type = func.output_types[out_idx] if out_idx < len(func.output_types) else type(out_vt[0])
                        new_node_id = add_node(out_vt, out_type, action_depth, (func_name, input_ids, out_idx))
                        if out_type is bool:
                            true_set = frozenset(i for i, v in enumerate(out_vt) if v)
                            if true_set not in realizable:
                                realizable[true_set] = _reconstruct_canonical(new_node_id, nodes, input_var_names)
            if not progressed:
                break

        return realizable


def _reconstruct_canonical(target_id: int, nodes: list, input_var_names: list[str]) -> list:
    """Walk the canonical-DAG backward from target_id. Inputs are nodes
    0..len(input_var_names)-1 with parent=None. Helper outputs get fresh
    b{i} names; the final node becomes a BoolExprNode."""
    n_inputs = len(input_var_names)
    var_dict: dict = {i: input_var_names[i] for i in range(n_inputs)}

    actions_by_sig: dict = {}  # (func_name, input_ids) → list of (out_idx, node_id)
    for nid, (_, _, _, parent) in enumerate(nodes):
        if parent is None:
            continue
        func_name, input_ids, out_idx = parent
        actions_by_sig.setdefault((func_name, input_ids), []).append((out_idx, nid))
    for v in actions_by_sig.values():
        v.sort()

    visit_order: list = []
    seen_sigs: set = set()

    def visit(nid: int):
        if nid < n_inputs:
            return
        parent = nodes[nid][3]
        sig = (parent[0], parent[1])
        if sig in seen_sigs:
            return
        seen_sigs.add(sig)
        for inp_id in parent[1]:
            visit(inp_id)
        visit_order.append(sig)

    visit(target_id)

    target_parent = nodes[target_id][3]
    target_sig = (target_parent[0], target_parent[1])

    stmts = []
    aux = 0
    for sig in visit_order:
        if sig == target_sig:
            continue
        func_name, input_ids = sig
        outs = actions_by_sig[sig]  # list of (out_idx, node_id)
        out_names = []
        for _, nid in outs:
            if nid not in var_dict:
                var_dict[nid] = f"b{aux}"
                aux += 1
            out_names.append(var_dict[nid])
        stmts.append(FunctionCallAssignNode(
            out_names, func_name, [var_dict[a] for a in input_ids],
        ))
    stmts.append(BoolExprNode(
        target_parent[0], [var_dict[a] for a in target_parent[1]],
    ))
    return stmts


def enumerate_realizable_partitions(
    in_scope_values_per_trace: list[dict],
    known_funcs: dict,
    known_bools: dict,
    max_depth: int,
) -> dict:
    """Convenience wrapper: build a one-shot BoolSearchEnvironment and query.
    For repeated queries with the same library, instantiate BoolSearchEnvironment
    once and call realizable_partitions on it (cached by value tuples)."""
    if not in_scope_values_per_trace:
        return {}
    common_vars = set.intersection(*[set(d.keys()) for d in in_scope_values_per_trace])
    if not common_vars:
        return {}
    var_names = sorted(common_vars)
    value_tuples_per_input = tuple(
        tuple(d[name] for d in in_scope_values_per_trace)
        for name in var_names
    )
    env = BoolSearchEnvironment(known_funcs, known_bools, max_depth)
    return env.realizable_partitions(value_tuples_per_input, var_names)


####################### TODO: make this into a nice class structure, with good interface etc #############################

class ConditionSearcher:
    def __init__(self, problem: Problem, general_funcs: dict[FunctionName, Function], bool_functions: dict[FunctionName, BoolFunction]):
        pass

    