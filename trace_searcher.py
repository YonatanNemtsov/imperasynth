from core_lang_env.comp_env import *
from core_lang_env.syntax_tree import *
from searchers_utils import *

from queue import PriorityQueue
from collections import deque
from itertools import product
from typing import Hashable, List, Tuple, Dict, Any
from itertools import product

ActionGroup = list[ShortAction]

NO_ACTION = "NO_ACTION"
RETURN_FUNC_NAME = "RETURN_FUNC_NAME"

######################## creating initial objects from a problem #######################

def create_traces_from_problem(problem: Problem, known_funcs: dict[FunctionName, Function]) -> list[SimpleCompEnv]:
    traces = []
    for i, (inputs, outputs) in problem.instances.items():
        trace = SimpleCompEnv(known_funcs.copy())
        for val_type, val in zip(problem.input_types, inputs):
            trace.add_input_object(CompObject(val_type, val))
        traces.append(trace)
    return traces

def create_maps_from_problem(problem: Problem, known_funcs: dict[FunctionName, Function], heuristic_distance: callable, heuristic_cutoff: float, map_search_size=1000) -> list[ComputationalMap]:
    maps = []
    for i, (inputs, outputs) in problem.instances.items():
        mapper = SimpleMapper(problem.input_types, inputs, problem.output_types[0], outputs[0], known_funcs, heuristic_distance)
        mapper.search(map_search_size, heuristic_cutoff)
        cmap = extract_minimal_subgraph(mapper.comp_map, outputs[0])
        maps.append(cmap)
    return maps

############################ Trace Group #####################################
class TraceGroup:
    def __init__(self, traces: list[SimpleCompEnv], known_funcs: dict[str, Function]):
        self.traces = traces
        self.known_funcs = known_funcs

    @staticmethod
    def create_new_group_from_inputs_and_known_funcs(input_group: list[tuple[tuple[type,object]]], known_funcs: dict[str, Function]):
        traces = []
        for inputs in input_group:
            trace = SimpleCompEnv(known_funcs.copy())
            for value_type, value in inputs:
                trace.add_input_object(CompObject(value_type, value))
            traces.append(trace)
        
        return TraceGroup(traces, known_funcs)
    
    @staticmethod
    def init_new_group_from_problem_and_funcs(problem: Problem, known_funcs: dict[FunctionName, Function]):
        traces = create_traces_from_problem(problem, known_funcs)
        return TraceGroup(traces, known_funcs)

    def get_group_signature(self):
        return tuple(trace.signature.to_tuple() for trace in self.traces)
    
    def apply_action_group(self, action_group: ActionGroup):
        out_ids_list = []
        for trace, short_action in zip(self.traces, action_group):
            if short_action == NO_ACTION:
                out_ids_list.append(None)
                continue
            if short_action[0] == RETURN_FUNC_NAME:
                out_ids_list.append(None)
                trace.assign_solution_object(short_action[1][0])
                continue
            out_ids = trace.apply_action(short_action)
            out_ids_list.append(out_ids)
        return out_ids_list
    
    def try_run_action_group(self, action_group: ActionGroup):
        outputs = []
        for trace, short_action in zip(self.traces, action_group):
            if short_action == NO_ACTION:
                outputs.append(NO_ACTION)
            elif short_action[0] == RETURN_FUNC_NAME:
                return_values = tuple(trace.objects[obj_id].value for obj_id in short_action[1])
                outputs.append((RETURN_FUNC_NAME, return_values))
            else:
                outputs.append(trace.try_run_function(*short_action))
        return outputs
    
    def add_function(self, func_name: FunctionName, func: Function):
        for trace in self.traces:
            trace.add_function(func_name, func)

    def update_functions(self, funcs: dict[FunctionName, Function]):
        for trace in self.traces:
            trace.known_functions.update(funcs)

    def copy(self):
        return TraceGroup([trace.copy() for trace in self.traces], self.known_funcs.copy())
    
    def __repr__(self):
        return f"TraceGroup(traces={self.traces})"

################################ find_possible_actions -> finding all available actions #####################

################################ FunctionCall actions ##########################################
def get_all_available_function_call_actions(trace_env: SimpleCompEnv, comp_map: ComputationalMap) -> list[ShortAction]:
    value_to_ids = {}

    for obj_id, obj in trace_env.objects.items():
        value_to_ids.setdefault(obj.value, []).append(obj_id)

    available_actions = []
    for action_key in comp_map.actions:
        func_name, *input_vals = action_key
        if all(val in value_to_ids for val in input_vals):
            for input_ids in product(*(value_to_ids[val] for val in input_vals)):
                candidate_action = (func_name, input_ids)

                if candidate_action in trace_env.action_history_short:
                    continue

                available_actions.append(candidate_action)

    return available_actions

def get_all_available_function_call_actions_with_variables(trace: SimpleCompEnv, comp_map: ComputationalMap, variables: dict[str, 'ObjId']) -> dict[ShortAction, list[ActionByVarName]]:
    """
    Return all possible actions (both object-id-level and variable-level) 
    that can be applied given the trace and computational map.

    example output:

    {
        ('get_head', (0,)): [('get_head', ('x',))],
        ('get_tail', (0,)): [('get_tail', ('x',))]
    }
    """
    # Build a mapping from values to variable names
    if trace.solution_object_id != None:
        return {}

    value_to_vars = {}
    for var, obj_id in variables.items():
        value = trace.objects[obj_id].value
        value_to_vars.setdefault(value, []).append(var)
    
    available_actions = {}

    for action_key in comp_map.actions:
        func_name, *input_vals = action_key

        if all(val in value_to_vars for val in input_vals):
            var_combos = product(*(value_to_vars[val] for val in input_vals))
            for var_tuple in var_combos:
                input_ids = tuple(variables[v] for v in var_tuple)
                candidate_action = (func_name, input_ids)

                if candidate_action in trace.action_history_short:
                    # print(candidate_action)
                    continue

                available_actions.setdefault(candidate_action, []).append((func_name, var_tuple))

    return available_actions

def find_possible_actions(traces: list[SimpleCompEnv], maps: list['ComputationalMap'], variable_list: list[dict[str, 'ObjId']]):
    """
    Given multiple traces, computational maps, and their variable contexts,
    return aligned possible actions across traces.

    Each element in the returned list corresponds to one "action schema"
    (i.e., same function name + variable names pattern),
    and contains the candidate actions (or 'NO_ACTION') for each trace.
    """
    assert len(traces) == len(maps) == len(variable_list)

    # Step 1 — Collect all available actions (with variable forms) for each trace
    all_action_dicts = tuple(get_all_available_function_call_actions_with_variables(trace, comp_map, variables) for trace, comp_map, variables in zip(traces, maps, variable_list))

    # Step 2 — Gather all unique "schemas" (func_name, var_tuple)
    all_schemas = set()
    for action_dict in all_action_dicts:
        for var_action_list in action_dict.values():
            for func_name, var_tuple in var_action_list:
                all_schemas.add((func_name, var_tuple))

    # Step 3 — Align all traces by schema
    aligned_actions = []
    for schema in all_schemas:
        func_name, var_tuple = schema
        row = []
        # For each trace, find matching action
        for actions_dict in all_action_dicts:
            found_action = None
            for short_action, var_action_list in actions_dict.items():
                for (vfunc, vtuple) in var_action_list:
                    if vfunc == func_name and vtuple == var_tuple:
                        found_action = short_action
                        break
                if found_action:
                    break
            if found_action:
                row.append(found_action)
            else:
                row.append(NO_ACTION)
        output_len = len(traces[0].known_functions[func_name].output_types)
        aligned_actions.append((schema, tuple(row), output_len))
    return aligned_actions

def filter_actions_by_active_trace_indices(aligned_actions: list[tuple[ShortActionByVarName, tuple[ShortAction, ...], int]], group_indices: set[int]) -> list[tuple[ShortActionByVarName, tuple[ShortAction, ...], int]]:
    filtered = []

    for var_action, short_actions, output_len in aligned_actions:
        if any(short_actions[i] == NO_ACTION for i in group_indices):
            continue

        new_short_actions = tuple(short_actions[i] if i in group_indices else NO_ACTION for i in range(len(short_actions)))

        filtered.append((var_action, new_short_actions, output_len))

    return filtered



################################ Return actions ##########################################

def get_all_available_return_actions_with_variables(trace: SimpleCompEnv, target: object, variables: dict[str, ObjId]) -> dict[ShortAction, list[ActionByVarName]]:
    available_returns = {}

    for var, obj_id in variables.items():
        obj = trace.objects[obj_id]
        if obj.value_type == type(target) and obj.value == target:
            available_returns.setdefault((RETURN_FUNC_NAME, (obj_id,)),[]).append((RETURN_FUNC_NAME, (var,)))
    return available_returns


def find_possible_return_actions(traces, targets, variable_list) -> list[tuple[ShortActionByVarName, list[ShortAction]]]:
    all_return_action_dicts = tuple(get_all_available_return_actions_with_variables(trace, target, variables) for trace, target, variables in zip(traces, targets, variable_list))
    
    all_schemas = set()
    for action_dict in all_return_action_dicts:
        for var_action_list in action_dict.values():
            for return_action, var_tuple in var_action_list:
                all_schemas.add((return_action, var_tuple))

    aligned_actions = []
    for schema in all_schemas:
        func_name, var_tuple = schema
        row = []
        # For each trace, find matching action
        for actions_dict in all_return_action_dicts:
            found_action = None
            for short_action, var_action_list in actions_dict.items():
                for (vfunc, vtuple) in var_action_list:
                    if vfunc == func_name and vtuple == var_tuple:
                        found_action = short_action
                        break
                if found_action:
                    break
            if found_action:
                row.append(found_action)
            else:
                row.append(NO_ACTION)
        aligned_actions.append((schema, tuple(row), 0))
    return aligned_actions

################################ Start If Actions ##########################################

def find_possible_start_if_actions(group_indices: set[int]) -> list[set[int]]:
    """
    Return all non-empty proper subsets of group_indices.
    Each subset represents possible 'if' branch indices.
    """
    from itertools import chain, combinations

    s = list(group_indices)
    all_subsets = chain.from_iterable(combinations(s, r) for r in range(1, len(s)))
    return [set(subset) for subset in all_subsets]



################################ End If Actions ##########################################

def find_possible_end_if_actions(group_indices: set[int]) -> list[set[int]]:
    return [group_indices]

################################ End Else Actions ##########################################

from itertools import permutations

def find_possible_end_else_actions(
    traces: list['SimpleCompEnv'],
    if_vars: list[dict[str, 'ObjId']],
    else_vars: list[dict[str, 'ObjId']],
    input_if_vars: list[dict[str, 'ObjId']],
    if_indices: set[int],
    else_indices: set[int],
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """
    Determine all valid (target <- source) variable merges at EndElse.

    Parameters:
        traces        : List of SimpleCompEnv for all examples
        if_vars       : Per-trace variable dicts from the IF branch
        else_vars     : Per-trace variable dicts from the ELSE branch
        input_if_vars : Per-trace variable dicts before entering IF (available as fallback)
        if_indices    : Indices of traces in IF branch
        else_indices  : Indices of traces in ELSE branch

    Returns:
        List of (target_vars, source_vars), each a valid one-to-one merge.
    """
    if not if_indices or not else_indices:
        return []

    n_traces = len(traces)

    # possible sources = union of else-vars and original input vars
    else_or_input = [
        {**input_if_vars[i], **else_vars[i]}
        for i in range(n_traces)
    ]

    valid_pairs: list[tuple[str, str]] = []

    # Choose *representative* indices for iteration instead of hard-coded 0.
    # These are just used to get candidate names; the loops below check all indices.
    rep_if = min(if_indices)
    rep_else = min(else_indices)

    # Candidate target variables: those that exist in all IF traces
    candidate_if_vars = [
        v for v in if_vars[rep_if].keys()
        if all(v in if_vars[i] for i in if_indices)
    ]

    # Candidate source variables: those that exist in all ELSE traces' else_or_input
    candidate_src_vars = [
        v for v in else_or_input[rep_else].keys()
        if all(v in else_or_input[j] for j in else_indices)
    ]

    for v_if in candidate_if_vars:
        for v_src in candidate_src_vars:
            match_all = True

            # Determine target type from one IF trace
            trace_if0 = traces[rep_if]
            obj_if0 = trace_if0.objects[if_vars[rep_if][v_if]]
            target_type = obj_if0.value_type

            # Check all IF indices
            for i in if_indices:
                if v_if not in if_vars[i]:
                    match_all = False
                    break
                trace_if = traces[i]
                obj_if = trace_if.objects[if_vars[i][v_if]]
                if obj_if.value_type != target_type:
                    match_all = False
                    break

            if not match_all:
                continue

            # Check all ELSE indices
            for j in else_indices:
                if v_src not in else_or_input[j]:
                    match_all = False
                    break
                trace_else = traces[j]
                obj_src = trace_else.objects[else_or_input[j][v_src]]
                if obj_src.value_type != target_type:
                    match_all = False
                    break

            if match_all:
                valid_pairs.append((v_if, v_src))

    # Build permutations of distinct (target, source) pairs
    results: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for n in range(1, len(valid_pairs) + 1):
        for combo in permutations(valid_pairs, n):
            targets, sources = zip(*combo)
            if len(set(targets)) == len(set(sources)) == n:
                target_to_source = dict(zip(targets, sources))
                
                # Sort targets and get corresponding sources in the same order
                ordered_targets = tuple(sorted(targets))
                ordered_sources = tuple(target_to_source[t] for t in ordered_targets)
                results.add((ordered_targets, ordered_sources))

    return list(results)

################# --------- rating actions and action groups ------------ #####################

##############################################################

def rate_action(trace: SimpleCompEnv, short_action: ShortAction, target_value, available_variables: dict[str, ObjId], object_importances: dict[object, float]) -> float:
    """
    Rate an action based on the importance of its created objects and whether
    those objects already exist in the environment.

    Args:
        trace: current SimpleCompEnv
        short_action: (func_name, input_obj_ids)
        target_value: the desired target output value (used indirectly for importance)
        available_variables: variables already present in the current context
        object_importances: mapping from object (or (type, value)) to importance in [0, 1]

    Returns:
        float: importance-based score for this action
    """
    if short_action == NO_ACTION:
        return 0.
    
    if short_action[0] == RETURN_FUNC_NAME:
        if trace.objects[short_action[1][0]].value == target_value:
            return 2.
        return -100
    
    func_name, input_ids = short_action
    func = trace.known_functions[func_name]

    # --- Compute outputs (simulate function) ---
    input_vals = [trace.objects[obj_id].value for obj_id in input_ids]
    try:
        output_vals = func.func(*input_vals)
    except Exception:
        return 0.0  # invalid or unsafe action

    if not isinstance(output_vals, (tuple, list)):
        output_vals = (output_vals,)

    total_score = 0.0

    for out_type, out_val in zip(func.output_types, output_vals):
        key = (out_type, out_val)
        importance = object_importances.get(key, 0.0)

        # --- Check if the object already exists ---
        already_exists = (
            any(obj.value == out_val and obj.value_type == out_type for obj in trace.objects.values())
            or any(trace.objects[obj_id].value == out_val for obj_id in available_variables.values())
        )

        if already_exists:
            contribution = 0.0
        else:
            contribution = importance

        total_score += contribution

    # Normalize by number of outputs for consistency
    total_score /= max(1, len(func.output_types))
    return total_score

def rate_action_group(trace_group: TraceGroup, action_group: ActionGroup, target_values: list, available_var_list: list[dict[str, ObjId]], obj_importance_list: list[dict[object, float]]):
    total_score = 0
    for trace, action, target, variables, importanes, in zip(trace_group.traces, action_group, target_values, available_var_list, obj_importance_list):
        total_score += rate_action(trace, action, target, variables, importanes)
    return total_score

def get_ranked_action_groups(trace_group: TraceGroup, var_annotated_action_group_list: list[tuple[ShortActionByVarName, ActionGroup, int]], target_values: list, available_var_list: list[dict[str, ObjId]], obj_importance_list: list[dict[object, float]]) -> list[tuple[float, ActionGroup]]:
    ranked_actions = []
    for var_action, group, output_len in var_annotated_action_group_list:
        rating = rate_action_group(trace_group, group, target_values, available_var_list, obj_importance_list)
        ranked_actions.append((rating, var_action, group, output_len))
    return sorted(ranked_actions)

###########################


