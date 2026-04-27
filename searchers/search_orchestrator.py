from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

from core_lang_env.comp_env import *
from core_lang_env.syntax_tree import *
from .searchers_utils import *

from . import trace_searcher as tsr
from . import condition_searcher as csr
from . import ast_searcher_v3 as asr


@dataclass(slots=True)
class AugmentationRequest:
    exec_position: ExecutionPositionTuple
    parent_indices: set[int]
    group_indices: set[int]

@dataclass(slots=True)
class AugmentationRequestFuncCall(AugmentationRequest):
    """Generic request: 'insert some function call here for these traces'."""
    pass


@dataclass(slots=True)
class AugmentationRequestReturn(AugmentationRequest):
    pass

@dataclass(slots=True)
class AugmentationRequestStartIf(AugmentationRequest):
    pass


@dataclass(slots=True)
class AugmentationRequestEndIf(AugmentationRequest):
    pass


@dataclass(slots=True)
class AugmentationRequestEndElse(AugmentationRequest):
    pass

@dataclass(slots=True)
class AugmentationRequestStartWhile(AugmentationRequest):
    pass

@dataclass(slots=True)
class AugmentationRequestEndWhile(AugmentationRequest):
    pass

@dataclass(slots=True)
class AugmentationRequestEnterWhile(AugmentationRequest):
    pass

@dataclass(slots=True)
class AugmentationRequestSkipWhile(AugmentationRequest):
    pass

@dataclass(slots=True)
class AugmentationRequestExecuteBlock(AugmentationRequest):
    pass

@dataclass(slots=True)
class AugmentationRequestExecuteFuncCall(AugmentationRequest):
    pass

@dataclass(slots=True)
class AugmentationRequestExecuteIfElse(AugmentationRequest):
    pass

@dataclass(slots=True)
class AugmentationRequestExecuteDirectAssign(AugmentationRequest):
    pass

@dataclass(slots=True)
class FuncCallCandidate:
    exec_position: ExecutionPositionTuple
    group_indices: tuple[int, ...]
    var_action: ShortActionByVarName
    short_actions: tuple[ShortAction, ...]
    output_length: int

@dataclass(slots=True)
class ReturnCandidate:
    exec_position: ExecutionPositionTuple
    group_indices: Tuple[int, ...]
    return_var: str                             # e.g. "x2"
    short_actions: Tuple[ShortAction, ...]    # full length (NO_ACTION outside group)

@dataclass(slots=True)
class StartIfCandidate:
    exec_position: ExecutionPositionTuple
    if_indices: Tuple[int, ...]                 # may be empty (per your choice A)
    else_indices: Tuple[int, ...]               # may be empty

@dataclass(slots=True)
class EndIfCandidate:
    # exec_position: ExecutionPositionTuple
    group_indices: Tuple[int, ...]

@dataclass(slots=True)
class EndElseCandidate:
    exec_position: ExecutionPositionTuple
    group_indices: Tuple[int, ...]
    target_vars: Tuple[str, ...]
    source_vars: Tuple[str, ...]

@dataclass(slots=True)
class StartWhileCandidate:
    exec_position: ExecutionPositionTuple
    while_indices: Tuple[int, ...]   # traces that enter the loop body at least once

@dataclass(slots=True)
class EndWhileCandidate:
    exec_position: ExecutionPositionTuple
    group_indices: Tuple[int, ...]
    target_vars: Tuple[str, ...] = ()    # variables alive after each iteration (typically loop input names)
    source_vars: Tuple[str, ...] = ()    # body-output variables that rebind into target_vars

@dataclass(slots=True)
class EnterWhileCandidate:
    exec_position: ExecutionPositionTuple
    while_indices: Tuple[int, ...]   # traces that take this iteration

@dataclass(slots=True)
class SkipWhileCandidate:
    exec_position: ExecutionPositionTuple
    group_indices: Tuple[int, ...]   # traces that don't take any more iterations

@dataclass(slots=True)
class ExecuteBlockCandidate:
    exec_position: ExecutionPositionTuple
    group_indices: Tuple[int, ...]
    subnodes: Tuple[type, ...]   # types of children of the block being executed

@dataclass(slots=True)
class ExecuteFuncCallCandidate:
    exec_position: ExecutionPositionTuple
    group_indices: Tuple[int, ...]

@dataclass(slots=True)
class ExecuteIfElseCandidate:
    exec_position: ExecutionPositionTuple
    if_indices: Tuple[int, ...]      # traces that took the if branch
    group_indices: Tuple[int, ...]   # all traces active at this if/else

@dataclass(slots=True)
class ExecuteDirectAssignCandidate:
    exec_position: ExecutionPositionTuple
    group_indices: Tuple[int, ...]




################################### Search State #################################

class SearchState:
    def __init__(self, trace_group: tsr.TraceGroup, ast_group: asr.AnnotatedAstGroup, aug_stack: asr.AugmentationStack, initial_vars: list[dict[str, ObjId]], variable_states: list[dict[str, ObjId]], search_concluded: bool):
        self.trace_group = trace_group
        self.ast_group = ast_group
        self.aug_stack = aug_stack
        self.initial_vars = initial_vars
        self.variable_states = variable_states
        self.search_concluded = search_concluded
        

    @staticmethod
    def init_new_search_state_from_problem_and_funcs(problem: Problem, known_funcs: dict[FunctionName, Function]):
        trace_group = tsr.TraceGroup.init_new_group_from_problem_and_funcs(problem, known_funcs)
        initial_var_states = [{f'x{i}':i for i in range(len(problem.input_types))} for _ in problem.instances]
        ast_group = asr.AnnotatedAstGroup.initialize_new_group(initial_var_states)
        aug_stack = asr.AugmentationStack.init_new_stack(set(i for i in range(len(problem.instances))))
        search_concluded = False
        return SearchState(trace_group, ast_group, aug_stack, initial_var_states.copy(), initial_var_states.copy(), search_concluded)
    
    def copy(self):
        # shallow copies of structures that are already cloned internally
        return SearchState(self.trace_group.copy(), self.ast_group.copy(), self.aug_stack.copy(), deepcopy(self.initial_vars), deepcopy(self.variable_states), self.search_concluded)

    def __repr__(self):
        return f"SearchState(n_traces={len(self.trace_group.traces)}, ast={self.ast_group.ast})"
    
    def apply_func_call_candidate(self, candidate: FuncCallCandidate) -> "SearchState":
        new_state = self.copy()

        detailed_action_group: asr.DetailedActionGroup = (candidate.var_action, candidate.short_actions, candidate.output_length)
        new_state.ast_group = asr.apply_augmentation_function_call_group(new_state.ast_group, candidate.exec_position, detailed_action_group)
        out_vars = new_state.ast_group.ast.get_node_at_position(candidate.exec_position[0]).var_names
        out_ids_list = new_state.trace_group.apply_action_group(candidate.short_actions)
        for i, out_ids in enumerate(out_ids_list):
            if out_ids == None:
                continue
            for var_name, out_id in zip(out_vars, out_ids):
                new_state.variable_states[i][var_name] = out_id
                
        choice_indices = set(candidate.group_indices)
        new_state.aug_stack.apply_transition(asr.AUG_FUNC_CALL, choice_indices, None)
        if not new_state.aug_stack.stack:
            new_state.search_concluded = True

        return new_state

    def apply_return_candidate(self, cand: ReturnCandidate) -> "SearchState":
        ns = self.copy()

        return_var_action = (asr.RETURN_FUNC_NAME, (cand.return_var,))
        detailed_action_group = (return_var_action, cand.short_actions, 0)

        ns.ast_group = asr.apply_augmentation_return_group(ns.ast_group, cand.exec_position, detailed_action_group)

        ns.trace_group.apply_action_group(list(cand.short_actions))
        ns.aug_stack.apply_transition(asr.AUG_RETURN, set(cand.group_indices), None)

        if not ns.aug_stack.stack:
            ns.search_concluded = True

        return ns

    def apply_start_if_candidate(self, cand: StartIfCandidate) -> "SearchState":
        ns = self.copy()

        ns.ast_group = asr.apply_augmentation_start_if_group(ns.ast_group, cand.exec_position)
        ns.aug_stack.apply_transition(asr.AUG_START_IF, set(cand.if_indices), None)

        if not ns.aug_stack.stack:
            ns.search_concluded = True

        return ns

    def apply_end_if_candidate(self, cand: EndIfCandidate) -> "SearchState":
        ns = self.copy()

        ns.ast_group = asr.apply_augmentation_end_if_group(ns.ast_group) #, cand.exec_position)
        ns.aug_stack.apply_transition(asr.AUG_END_IF, set(cand.group_indices), None)

        if not ns.aug_stack.stack:
            ns.search_concluded = True

        return ns
    
    def apply_end_else_candidate(self, cand: EndElseCandidate) -> "SearchState":
        ns = self.copy()
        ns.ast_group = asr.apply_augmentation_end_else_group(ns.ast_group, cand.exec_position, cand.target_vars, cand.source_vars, cand.group_indices)


        for idx in cand.group_indices:
            for target, source in zip(cand.target_vars, cand.source_vars):
                ns.variable_states[idx][target] = ns.variable_states[idx][source]
        

        variables_defined_in_if = get_variables_defined_in_node(ns.ast_group.ast, cand.exec_position[0][:-2]+ (1,))
        variables_defined_in_else = get_variables_defined_in_node(ns.ast_group.ast, cand.exec_position[0][:-2] + (2,))

        for var in variables_defined_in_if.union(variables_defined_in_else):
            if not var in cand.target_vars:
                for var_state in ns.variable_states:
                    if var in var_state:
                        del var_state[var]

        ns.aug_stack.apply_transition(asr.AUG_END_ELSE, set(cand.group_indices), None)

        if not ns.aug_stack.stack:
            ns.search_concluded = True


        return ns
    
    def apply_start_while_candidate(self, cand: "StartWhileCandidate") -> "SearchState":
        ns = self.copy()
        ns.ast_group = asr.apply_augmentation_start_while_group(ns.ast_group, cand.exec_position)
        ns.aug_stack.apply_transition(asr.AUG_START_WHILE, set(cand.while_indices), None)
        if not ns.aug_stack.stack:
            ns.search_concluded = True
        return ns

    def apply_end_while_candidate(self, cand: "EndWhileCandidate") -> "SearchState":
        ns = self.copy()

        ns.ast_group = asr.apply_augmentation_end_while_group(
            ns.ast_group, cand.exec_position, cand.target_vars, cand.source_vars, set(cand.group_indices),
        )

        if cand.target_vars:
            # Apply the rebinding to active traces' variable_states.
            # Pop sources first to handle swap-style assignments (a, b <- b, a).
            for idx in cand.group_indices:
                obj_ids = [ns.variable_states[idx].pop(src) for src in cand.source_vars]
                for tgt, oid in zip(cand.target_vars, obj_ids):
                    ns.variable_states[idx][tgt] = oid

            # Clean up body-defined variables that aren't kept as targets.
            body_position = cand.exec_position[0][:-1]
            variables_defined_in_body = get_variables_defined_in_node(ns.ast_group.ast, body_position)
            for var in variables_defined_in_body:
                if var not in cand.target_vars:
                    for var_state in ns.variable_states:
                        if var in var_state:
                            del var_state[var]

        ns.aug_stack.apply_transition(asr.AUG_END_WHILE, set(cand.group_indices), None)
        if not ns.aug_stack.stack:
            ns.search_concluded = True
        return ns

    def apply_enter_while_candidate(self, cand: "EnterWhileCandidate") -> "SearchState":
        """Take another iteration through the loop body. Trace state advances
        when the body's execute_* candidates fire next."""
        ns = self.copy()
        ns.aug_stack.apply_transition(asr.AUG_ENTER_WHILE, set(cand.while_indices), None)
        if not ns.aug_stack.stack:
            ns.search_concluded = True
        return ns

    def apply_skip_while_candidate(self, cand: "SkipWhileCandidate") -> "SearchState":
        """Stop iterating; traces leave the loop. No trace mutation here —
        their state is whatever the last iteration left them."""
        ns = self.copy()
        ns.aug_stack.apply_transition(asr.AUG_SKIP_WHILE, set(cand.group_indices), None)
        if not ns.aug_stack.stack:
            ns.search_concluded = True
        return ns

    def apply_execute_block_candidate(self, cand: "ExecuteBlockCandidate") -> "SearchState":
        """Unroll the body — push a frontier per child node. Trace state is
        unchanged here; child execute_* operations will advance it."""
        ns = self.copy()
        ns.aug_stack.apply_transition(asr.AUG_EXECUTE_BLOCK, set(cand.group_indices), list(cand.subnodes))
        if not ns.aug_stack.stack:
            ns.search_concluded = True
        return ns

    def apply_execute_func_call_candidate(self, cand: "ExecuteFuncCallCandidate") -> "SearchState":
        """Replay the FunctionCallAssignNode at exec_position for all active traces."""
        ns = self.copy()
        func_call_node = ns.ast_group.ast.get_node_at_position(cand.exec_position[0])
        short_actions = []
        for i in range(len(ns.trace_group.traces)):
            if i in cand.group_indices:
                input_ids = tuple(ns.variable_states[i][arg] for arg in func_call_node.arg_names)
                short_actions.append((func_call_node.func_name, input_ids))
            else:
                short_actions.append(asr.NO_ACTION)
        out_ids_list = ns.trace_group.apply_action_group(short_actions)
        for i, out_ids in enumerate(out_ids_list):
            if out_ids is None:
                continue
            for var_name, out_id in zip(func_call_node.var_names, out_ids):
                ns.variable_states[i][var_name] = out_id
        ns.aug_stack.apply_transition(asr.AUG_EXECUTE_FUNC_CALL, set(cand.group_indices), None)
        if not ns.aug_stack.stack:
            ns.search_concluded = True
        return ns

    def apply_execute_if_else_candidate(self, cand: "ExecuteIfElseCandidate") -> "SearchState":
        """Split traces between if/else branches. Trace state is unchanged here —
        the chosen branch's execute_block will advance the trace state on the
        traces that take it."""
        ns = self.copy()
        ns.aug_stack.apply_transition(asr.AUG_EXECUTE_IF_ELSE, set(cand.if_indices), None)
        if not ns.aug_stack.stack:
            ns.search_concluded = True
        return ns

    def apply_execute_direct_assign_candidate(self, cand: "ExecuteDirectAssignCandidate") -> "SearchState":
        """Replay a DirectAssignNode: pop sources, assign to targets (handles swap pattern)."""
        ns = self.copy()
        da_node = ns.ast_group.ast.get_node_at_position(cand.exec_position[0])
        for i in cand.group_indices:
            obj_ids = [ns.variable_states[i].pop(src) for src in da_node.source_vars]
            for tgt, oid in zip(da_node.target_vars, obj_ids):
                ns.variable_states[i][tgt] = oid
        ns.aug_stack.apply_transition(asr.AUG_EXECUTE_DIRECT_ASSIGN, set(cand.group_indices), None)
        if not ns.aug_stack.stack:
            ns.search_concluded = True
        return ns

    def get_signature(self):
        # Include each trace's action_history_short — the AnnotatedAST.signature
        # used by ast_group.get_signature is frozen at build-phase end and
        # doesn't capture execute-phase iteration counts. Two states differing
        # only in how many while-iterations a trace took would otherwise dedupe
        # to the same signature.
        trace_sigs = tuple(tuple(t.action_history_short) for t in self.trace_group.traces)
        return (hash(self.ast_group.get_signature()), hash(self.aug_stack.get_signature()), hash(trace_sigs))




BASE_REQUEST_TYPES = {
    asr.AUG_FUNC_CALL: AugmentationRequestFuncCall,
    asr.AUG_RETURN:    AugmentationRequestReturn,
    asr.AUG_START_IF:  AugmentationRequestStartIf,
    asr.AUG_END_IF:    AugmentationRequestEndIf,
    asr.AUG_END_ELSE:  AugmentationRequestEndElse,
}

WHILE_REQUEST_TYPES = {
    asr.AUG_START_WHILE:           AugmentationRequestStartWhile,
    asr.AUG_END_WHILE:             AugmentationRequestEndWhile,
    asr.AUG_ENTER_WHILE:           AugmentationRequestEnterWhile,
    asr.AUG_SKIP_WHILE:            AugmentationRequestSkipWhile,
    asr.AUG_EXECUTE_BLOCK:         AugmentationRequestExecuteBlock,
    asr.AUG_EXECUTE_FUNC_CALL:     AugmentationRequestExecuteFuncCall,
    asr.AUG_EXECUTE_IF_ELSE:       AugmentationRequestExecuteIfElse,
    asr.AUG_EXECUTE_DIRECT_ASSIGN: AugmentationRequestExecuteDirectAssign,
}


def get_cls_map(enable_while_loops: bool = False):
    """Return the AUG_* → AugmentationRequest* dict the orchestrator should use.

    `enable_while_loops=False` (the default) silently filters while-related
    options out of request generation, so the search proceeds as if while
    loops were never an option — even though the transition functions still
    syntactically include AUG_START_WHILE in their frontier option sets.
    """
    if enable_while_loops:
        return {**BASE_REQUEST_TYPES, **WHILE_REQUEST_TYPES}
    return dict(BASE_REQUEST_TYPES)


def generate_augmentation_requests_from_state(
    state: 'SearchState',
    cls_map: dict | None = None,
) -> list[AugmentationRequest]:
    """
    Return AugmentationRequest objects for all allowed options
    at the top of the augmentation stack.

    `cls_map` defaults to the base (no-while) map, matching pre-while behavior.
    Callers that want while-loop augmentations should pass `get_cls_map(True)`.
    """
    if cls_map is None:
        cls_map = BASE_REQUEST_TYPES

    frontier = state.aug_stack.peek()
    if not frontier:
        return []

    exec_pos, aug_options, parent_indices, group_indices = frontier

    return [
        cls_map[opt](exec_pos, set(parent_indices), set(group_indices))
        for opt in aug_options
        if opt in cls_map
    ]

################################### Candidate generation ########################

def generate_func_call_candidates(state: SearchState, request: AugmentationRequestFuncCall, cmaps: list[ComputationalMap]) -> list[FuncCallCandidate]:
    """
    Generate all possible function call candidates for the current search state,
    based on the current execution position and active trace indices.
    """
    traces = state.trace_group.traces

    # 1. Get per-trace variable states right before this position
    # variable_list = [annot_ast.get_variable_state_at_position(request.exec_position) for annot_ast in state.ast_group.annot_asts]
    
    # 3. Get all available actions and filter by active indices
    aligned_actions = tsr.find_possible_actions(traces, cmaps, state.variable_states)
    filtered = tsr.filter_actions_by_active_trace_indices(aligned_actions, request.group_indices)

    # 4. Construct candidates
    candidates = [
        FuncCallCandidate(
            exec_position=request.exec_position,
            group_indices=tuple(sorted(request.group_indices)),
            var_action=var_action,
            short_actions=short_actions,
            output_length=output_len
        )
        for var_action, short_actions, output_len in filtered
    ]
    return candidates

def generate_start_if_candidates(state: SearchState, request: AugmentationRequestStartIf, cmaps: list[ComputationalMap]) -> list[StartIfCandidate]:
    candidates = [StartIfCandidate(request.exec_position, if_indices, request.group_indices.difference(if_indices)) for if_indices in tsr.find_possible_start_if_actions(request.group_indices)]
    return candidates

def generate_end_if_candidates(state: SearchState, request: AugmentationRequestEndIf, cmaps: list[ComputationalMap]) -> list[EndIfCandidate]:
    return [EndIfCandidate(request.group_indices)]


def generate_start_while_candidates(state: SearchState, request: 'AugmentationRequestStartWhile', cmaps: list[ComputationalMap]) -> list['StartWhileCandidate']:
    return [
        StartWhileCandidate(request.exec_position, tuple(sorted(while_indices)))
        for while_indices in tsr.find_possible_start_while_actions(request.group_indices)
    ]


def generate_end_while_candidates(state: SearchState, request: 'AugmentationRequestEndWhile', cmaps: list[ComputationalMap]) -> list['EndWhileCandidate']:
    body_position = request.exec_position[0][:-1]
    body_var_names = get_variables_defined_in_node(state.ast_group.ast, body_position)

    rebinds = tsr.find_possible_end_while_actions(
        state.trace_group.traces,
        state.variable_states,
        body_var_names,
        request.group_indices,
    )

    candidates = [
        EndWhileCandidate(
            exec_position=request.exec_position,
            group_indices=tuple(sorted(request.group_indices)),
            target_vars=target_vars,
            source_vars=source_vars,
        )
        for target_vars, source_vars in rebinds
    ]
    # Always include the no-rebinding option for completeness; the search's
    # heuristic will deprioritize loops with empty bodies/no rebinding.
    candidates.append(EndWhileCandidate(
        exec_position=request.exec_position,
        group_indices=tuple(sorted(request.group_indices)),
    ))
    return candidates


def _can_trace_execute_while_body(state: SearchState, trace_idx: int, body: BlockNode) -> bool:
    """Simulate the while body's statements on a trace's current value state.

    Returns True iff every FunctionCallAssignNode produces the expected number
    of outputs (so var_names actually get assigned) and every DirectAssignNode's
    sources are in scope. We work directly with VALUES (no object/id machinery)
    since this is a check, not a state mutation.

    Nested if/else/while inside the body are treated optimistically (assumed
    valid) — full simulation of those would require condition resolution, which
    we don't have until Phase 3. For straight-line bodies this is exact.
    """
    trace = state.trace_group.traces[trace_idx]
    sim_values = {k: trace.objects[v].value for k, v in state.variable_states[trace_idx].items()}

    for child in body.children:
        if isinstance(child, FunctionCallAssignNode):
            try:
                input_vals = tuple(sim_values[arg] for arg in child.arg_names)
            except KeyError:
                return False
            func = trace.known_functions.get(child.func_name)
            if func is None:
                return False
            try:
                outputs = func.func(*input_vals)
            except Exception:
                return False
            if not isinstance(outputs, (tuple, list)):
                outputs = (outputs,)
            if len(outputs) != len(child.var_names):
                return False
            for var, val in zip(child.var_names, outputs):
                sim_values[var] = val
        elif isinstance(child, DirectAssignNode):
            try:
                src_vals = [sim_values[s] for s in child.source_vars]
            except KeyError:
                return False
            for s in child.source_vars:
                sim_values.pop(s, None)
            for tgt, val in zip(child.target_vars, src_vals):
                sim_values[tgt] = val
        # else: optimistic for nested if/else/while
    return True


def generate_enter_while_candidates(state: SearchState, request: 'AugmentationRequestEnterWhile', cmaps: list[ComputationalMap]) -> list['EnterWhileCandidate']:
    """Enumerate which traces take this iteration, but only over the subset of
    traces whose loop body would actually execute successfully on their current
    variable state. Traces that can't (e.g. empty-list trace + get_head body)
    are excluded — the search would otherwise crash on KeyError downstream."""
    while_node = state.ast_group.ast.get_node_at_position(request.exec_position[0])
    body = while_node.block
    valid = {i for i in request.group_indices
             if _can_trace_execute_while_body(state, i, body)}
    return [
        EnterWhileCandidate(request.exec_position, tuple(sorted(while_indices)))
        for while_indices in tsr.find_possible_start_while_actions(valid)
    ]


def generate_skip_while_candidates(state: SearchState, request: 'AugmentationRequestSkipWhile', cmaps: list[ComputationalMap]) -> list['SkipWhileCandidate']:
    return [SkipWhileCandidate(request.exec_position, tuple(sorted(request.group_indices)))]


def generate_execute_block_candidates(state: SearchState, request: 'AugmentationRequestExecuteBlock', cmaps: list[ComputationalMap]) -> list['ExecuteBlockCandidate']:
    block_pos = request.exec_position[0]
    block = state.ast_group.ast.get_node_at_position(block_pos)
    subnodes = tuple(type(child) for child in block.children)
    return [ExecuteBlockCandidate(request.exec_position, tuple(sorted(request.group_indices)), subnodes)]


def generate_execute_func_call_candidates(state: SearchState, request: 'AugmentationRequestExecuteFuncCall', cmaps: list[ComputationalMap]) -> list['ExecuteFuncCallCandidate']:
    return [ExecuteFuncCallCandidate(request.exec_position, tuple(sorted(request.group_indices)))]


def generate_execute_if_else_candidates(state: SearchState, request: 'AugmentationRequestExecuteIfElse', cmaps: list[ComputationalMap]) -> list['ExecuteIfElseCandidate']:
    # Use the execute-phase enumerator (allows ∅ and full set), NOT the
    # start_if one (which excludes both — only correct for build phase).
    return [
        ExecuteIfElseCandidate(request.exec_position, tuple(sorted(if_indices)), tuple(sorted(request.group_indices)))
        for if_indices in tsr.find_possible_execute_if_else_actions(request.group_indices)
    ]


def generate_execute_direct_assign_candidates(state: SearchState, request: 'AugmentationRequestExecuteDirectAssign', cmaps: list[ComputationalMap]) -> list['ExecuteDirectAssignCandidate']:
    return [ExecuteDirectAssignCandidate(request.exec_position, tuple(sorted(request.group_indices)))]



def generate_end_else_candidates(state: SearchState, request: AugmentationRequestEndElse, cmaps: list[ComputationalMap]) -> list[EndElseCandidate]:
    else_indices = request.group_indices
    if_indices = request.parent_indices.difference(else_indices)

    ast_group = state.ast_group

    if_position = request.exec_position[0][:-2] + (1,)
    else_position = request.exec_position[0][:-1]
    vars_defined_in_if = get_variables_defined_in_node(state.ast_group.ast, if_position)
    vars_defined_in_else = get_variables_defined_in_node(state.ast_group.ast, else_position)
    # print(vars_defined_in_else)
    vars_used_as_inputs_in_if = get_variables_used_as_inputs_in_node(state.ast_group.ast, if_position)
    
    if_vars_list = []
    else_vars_list = []
    if_inp_vars_list = []
    for i, annot_ast in enumerate(ast_group.annot_asts):
        # all_vars = annot_ast.get_variable_state_at_position(request.exec_position)
        all_vars = state.variable_states[i]
        if_vars = {k:v for k,v in all_vars.items() if k in vars_defined_in_if}
        if_vars_list.append(if_vars)
        else_vars = {k:v for k,v in all_vars.items() if k in vars_defined_in_else}
        else_vars_list.append(else_vars)
        if_input_vars = {k:v for k,v in all_vars.items() if k in vars_used_as_inputs_in_if}
        if_inp_vars_list.append(if_input_vars)

    possible_assignment_pairs = tsr.find_possible_end_else_actions(state.trace_group.traces, if_vars_list, else_vars_list, if_inp_vars_list, if_indices, else_indices)
    # print(possible_assignment_pairs)

    # TODO: Use these: source and target must include all variables in unused_if_vars, and unused_else_vars. 
    unused_if_vars = set(get_unused_variables_in_ast(ast_group.ast.get_node_at_position(if_position)))
    unused_else_vars = set(get_unused_variables_in_ast(ast_group.ast.get_node_at_position(else_position)))

    filtered_pairs = []
    for target, source in possible_assignment_pairs:
        if unused_else_vars.issubset(set(target)) and unused_if_vars.issubset(set(source)):
            filtered_pairs.append((target, source))

    candidates = [
        EndElseCandidate(request.exec_position, else_indices, target, source)
        for target, source in filtered_pairs
    ]

    return candidates

def generate_return_candidates(state: SearchState, request: AugmentationRequestEndIf, cmaps: list[ComputationalMap], targets: list[object]) -> list[ReturnCandidate]:
    unused_vars = get_unused_variables_in_ast(state.ast_group.ast)
    if len(unused_vars) > 1:
        return []
    # Use the live variable_states (mutated by the execute phase),
    # NOT annot_ast.get_variable_state_at_position which is frozen at build time.
    possible_returns = tsr.find_possible_return_actions(state.trace_group.traces, targets, state.variable_states)
    filtered_possible_returns = tsr.filter_actions_by_active_trace_indices(possible_returns, request.group_indices)
    return [ReturnCandidate(request.exec_position, request.group_indices, schema[0][1][0], schema[1]) for schema in filtered_possible_returns]


################################### AST / Trace Scoring ###################################

from collections import defaultdict
from queue import PriorityQueue

# -------------------------------------------------------------
# AST SIZE METRIC
# -------------------------------------------------------------

def compute_ast_size(node: ASTNode, depth: int = 0) -> int:
    if isinstance(node, FunctionCallAssignNode):
        base = 1
    elif isinstance(node, DirectAssignNode):
        base = 0
    elif isinstance(node, ReturnNode):
        base = 0
    elif isinstance(node, BoolExprNode):
        base = 0
    elif isinstance(node, IfElseNode):
        base = 1
    elif isinstance(node, WhileNode):
        base = 1
    elif isinstance(node, BlockNode):
        base = 1
    else:
        base = 1

    # multiplicative depth penalty
    multiplier = 1.0 # + 0.2 * depth   # cost grows 20% per depth level

    total = base * multiplier

    for child in node.children:
        total += compute_ast_size(child, depth + 1)

    return total


def estimate_final_trace_length(state: SearchState, cmaps: list[ComputationalMap], targets):
    estimates = []
    # shared_var_names = set.intersection((set(var_state.keys()) for var_state in state.variable_states))
    for trace, variable_state, cmap, target in zip(state.trace_group.traces, state.variable_states, cmaps, targets):
        # obj_ids = [variable_state[var] for var in shared_var_names]
        available_object_values = tuple(trace.objects[obj_id].value for obj_id in variable_state.values())
        # print(available_object_values, target)
        # print(find_minimal_steps_in_comp_map(cmap, available_object_values, target))
        estimates.append(len(trace.action_history) + find_minimal_steps_in_comp_map(cmap, available_object_values, target))
    # return sum(estimates)/len(estimates)
    return max(estimates)


def score_state(state: SearchState, problem: Problem, cmaps: list["ComputationalMap"], tie_breaker: int):
    """Priority key (lower → explored first).

    Primary: ast_size (the actual program complexity — what the search is
    trying to minimize). Stack depth is intentionally NOT included here:
    execute-phase operations swell the stack but don't change the AST, so
    penalizing stack depth would bury the only valid path through a while
    loop's execute phase below SKIP-and-build alternatives.
    Secondary: importance_dist (estimated remaining work to reach targets).
    Tertiary: tie_breaker (FIFO among equal-priority states).
    """
    targets = [problem.instances[i][1][0] for i in range(len(problem.instances))]
    ast_size = compute_ast_size(state.ast_group.ast)
    importance_dist = estimate_final_trace_length(state, cmaps, targets)
    return (ast_size, importance_dist, tie_breaker)


##################### Orchestrator ###################################

class SearchOrchestrator:
    def __init__(self, problem: Problem, known_funcs, known_bools, cmaps, enable_while_loops: bool = False):
        self.problem = problem
        self.known_funcs = known_funcs
        self.known_bools = known_bools
        self.cmaps = cmaps
        self.enable_while_loops = enable_while_loops
        self.cls_map = get_cls_map(enable_while_loops)

        self.search_queue: PriorityQueue[tuple[tuple, SearchState]] = PriorityQueue()
        self.visited_states = set()
        self.tie_counter = 0
        self.last_processed_state = None


        # store completed programs (states with concluded search)
        self.program_skeleton_candidates: list[SearchState] = []
        self.completed_programs: list[SearchState] = []

    # -----------------------------------------------------------------
    @staticmethod
    def create_new_orchestrator_from_problem(problem: Problem, known_funcs: dict[FunctionName, Function], known_bools: dict[FunctionName, BoolFunction], heuristic_distance: callable, heuristic_cutoff: float,  map_size: int = 1000, enable_while_loops: bool = False) -> "SearchOrchestrator":

        # 1. build computational maps per instance
        cmaps = tsr.create_maps_from_problem(problem, known_funcs, heuristic_distance, heuristic_cutoff, map_size)

        # 2. initial empty AST + traces
        initial_state = SearchState.init_new_search_state_from_problem_and_funcs(problem, known_funcs)

        # 3. orchestrator object
        orch = SearchOrchestrator(problem, known_funcs, known_bools, cmaps, enable_while_loops=enable_while_loops)

        # 4. enqueue initial state
        pr = score_state(initial_state, problem, cmaps, orch.tie_counter)
        orch.search_queue.put((pr, initial_state))
        orch.tie_counter += 1

        return orch

    def enqueue(self, state: "SearchState"):
        pr = score_state(state, self.problem, self.cmaps, self.tie_counter)
        self.search_queue.put((pr, state))
        self.tie_counter += 1

    def step(self, trace_length_limit=None, max_ast_len=None) -> "SearchState | None":
        if self.search_queue.empty():
            return

        score, state = self.search_queue.get()
        #sig = state.ast_group.get_signature()
        #if sig in self.visited_states_signatures:
        #    return state
        
        # If this state is already solved, store it and return it.
        if state.search_concluded:
            self.program_skeleton_candidates.append(state)
            return state
        
        if score[1] >= trace_length_limit:
            return state

        if score[0] > max_ast_len:
            return state
        
        requests = generate_augmentation_requests_from_state(state, self.cls_map)

        for req in requests:
            if isinstance(req, AugmentationRequestFuncCall):
                candidates = generate_func_call_candidates(state, req, self.cmaps)
            elif isinstance(req, AugmentationRequestStartIf):
                candidates = generate_start_if_candidates(state, req, self.cmaps)
            elif isinstance(req, AugmentationRequestEndIf):
                candidates = generate_end_if_candidates(state, req, self.cmaps)
            elif isinstance(req, AugmentationRequestEndElse):
                candidates = generate_end_else_candidates(state, req, self.cmaps)
            elif isinstance(req, AugmentationRequestStartWhile):
                candidates = generate_start_while_candidates(state, req, self.cmaps)
            elif isinstance(req, AugmentationRequestEndWhile):
                candidates = generate_end_while_candidates(state, req, self.cmaps)
            elif isinstance(req, AugmentationRequestEnterWhile):
                candidates = generate_enter_while_candidates(state, req, self.cmaps)
            elif isinstance(req, AugmentationRequestSkipWhile):
                candidates = generate_skip_while_candidates(state, req, self.cmaps)
            elif isinstance(req, AugmentationRequestExecuteBlock):
                candidates = generate_execute_block_candidates(state, req, self.cmaps)
            elif isinstance(req, AugmentationRequestExecuteFuncCall):
                candidates = generate_execute_func_call_candidates(state, req, self.cmaps)
            elif isinstance(req, AugmentationRequestExecuteIfElse):
                candidates = generate_execute_if_else_candidates(state, req, self.cmaps)
            elif isinstance(req, AugmentationRequestExecuteDirectAssign):
                candidates = generate_execute_direct_assign_candidates(state, req, self.cmaps)
            elif isinstance(req, AugmentationRequestReturn):
                targets = [self.problem.instances[i][1][0] for i in range(len(self.problem.instances))]
                candidates = generate_return_candidates(state, req, self.cmaps, targets)
            else:
                continue

            for cand in candidates:
                next_state = self.apply_candidate(state, cand)
                self.last_processed_state = next_state
                next_state_sig = next_state.get_signature()
                if next_state_sig in self.visited_states:
                    continue

                self.visited_states.add(next_state_sig)
                
                if next_state.search_concluded:
                    self.program_skeleton_candidates.append(next_state)
                    print(f"Candidate Program Found: {next_state.ast_group.ast}")
                    full_ast = self.search_boolean_expressions(next_state)
                    if full_ast:
                        self.completed_programs.append(full_ast)
                else:
                    self.enqueue(next_state)

        return state  # no solution this step

    def run(self, max_steps: int = 10000) -> "SearchState | None":
        for _ in range(max_steps):
            result = self.step()
            if result is not None:  # a program finished
                return result

        return None  # no solution found in time

    def apply_candidate(self, state: "SearchState", cand) -> "SearchState":
        if isinstance(cand, FuncCallCandidate):
            return state.apply_func_call_candidate(cand)
        if isinstance(cand, StartIfCandidate):
            return state.apply_start_if_candidate(cand)
        if isinstance(cand, EndIfCandidate):
            return state.apply_end_if_candidate(cand)
        if isinstance(cand, EndElseCandidate):
            return state.apply_end_else_candidate(cand)
        if isinstance(cand, StartWhileCandidate):
            return state.apply_start_while_candidate(cand)
        if isinstance(cand, EndWhileCandidate):
            return state.apply_end_while_candidate(cand)
        if isinstance(cand, EnterWhileCandidate):
            return state.apply_enter_while_candidate(cand)
        if isinstance(cand, SkipWhileCandidate):
            return state.apply_skip_while_candidate(cand)
        if isinstance(cand, ExecuteBlockCandidate):
            return state.apply_execute_block_candidate(cand)
        if isinstance(cand, ExecuteFuncCallCandidate):
            return state.apply_execute_func_call_candidate(cand)
        if isinstance(cand, ExecuteIfElseCandidate):
            return state.apply_execute_if_else_candidate(cand)
        if isinstance(cand, ExecuteDirectAssignCandidate):
            return state.apply_execute_direct_assign_candidate(cand)
        if isinstance(cand, ReturnCandidate):
            return state.apply_return_candidate(cand)

        raise TypeError(f"Unrecognized candidate type: {type(cand)}")
    
    def search_boolean_expression_ifelse_node(self, search_state: SearchState, node_position: ASTNodePosition, max_depth=20):
        bool_problem, input_vars = csr.extract_ifelse_conditional_problem_for_group(search_state.ast_group.annot_asts, node_position, search_state.trace_group.traces)
        #funcs = self.known_bools.copy()
        #funcs.update(self.known_funcs)
        #print(funcs)
        bool_trace_solution = csr.search_boolean_traces(bool_problem, self.known_bools, max_depth)
        if any(trace.solution_object_id == None for trace in bool_trace_solution):
            print('solution to bool problem not found')
            return None
        boolean_expression_statements = csr.create_bool_program_expressions(bool_trace_solution[0], input_vars)
        return boolean_expression_statements

    def integrate_boolean_expression_ifelse_node(self, ast: BlockNode, if_else_node_position: ASTNodePosition, boolean_expression_statements: list[ASTNode]):
        full_ast = ast.copy()
        
        full_ast.replace_node(if_else_node_position + (0,), boolean_expression_statements[-1])

        for statement in boolean_expression_statements[-2::-1]:
            full_ast.insert_node(if_else_node_position, statement)
        
        return full_ast
    
    def get_ifelse_positions(self, search_state: SearchState) -> list[ASTNodePosition]:
        positions = []
        def traverse(node, pos):
            if isinstance(node, IfElseNode):
                positions.append(pos)
            for i, child in enumerate(node.children):
                traverse(child, pos + (i,))

        traverse(search_state.ast_group.ast, ())
        return positions


    def search_boolean_expressions(self, search_state: SearchState):
        ifelse_positions = self.get_ifelse_positions(search_state)
        full_ast = search_state.ast_group.ast.copy()
        for pos in reversed(ifelse_positions):
            boolean_expression_statements = self.search_boolean_expression_ifelse_node(search_state, pos)
            if not boolean_expression_statements:
                print('bool expr not found')
                return None
            full_ast = self.integrate_boolean_expression_ifelse_node(full_ast, pos, boolean_expression_statements)
        return full_ast


#################### Improved Search Orchestrator #######################


class SearchOrchestratorV2:
    pass