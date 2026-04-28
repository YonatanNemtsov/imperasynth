from queue import PriorityQueue
from collections import deque, defaultdict
from itertools import product
from typing import Hashable

from core_lang_env.comp_env import *
from core_lang_env.syntax_tree import *
from core_lang_env.exec_code_v2 import *

######################## Action definitions by different representations, and translation functions ##############################

# Defined in comp_env:
# Action = tuple[str, tuple[ObjId], tuple[ObjId]]
# ShortAction = tuple[str, tuple[ObjId]] # (func_name, (inp_id1, ...,inp_idn))

ActionByValue = tuple[str, tuple[object], tuple[object]]
ShortActionByValue = tuple[str, tuple[object]] # (func_name, (inp_val1, ...,inp_valn))

ActionByVarName = tuple[str, tuple[str], tuple[str]]
ShortActionByVarName = tuple[str, tuple[str]] # (func_name, (var1, ...,varn))


######################## Computational Map ########################

class ComputationalMap:
    def __init__(self, objects: dict[object, set], functions: dict[str, 'Function'], actions: dict[tuple, tuple]):
        self.objects = objects
        self.functions = functions
        self.actions = actions
        
    def act(self, func_name: str, inputs: tuple[Hashable]):
        if func_name not in self.functions:
            raise ValueError(f"Function '{func_name}' is not registered.")

        func_obj = self.functions[func_name]

        for obj, t in zip(inputs, func_obj.input_types):
            if not isinstance(obj, t):
                raise TypeError(f"Function '{func_name}' got wrong type {type(obj)} argument")

        # Call the underlying function
        outputs = func_obj.func(*inputs)

        # Record the action
        action_key = (func_name,) + tuple(inputs)
        self.actions[action_key] = tuple((t,obj) for t, obj in  zip(func_obj.output_types, outputs))

        # Track outputs
        for t, out in zip(func_obj.output_types, outputs):
            if (t, out) not in self.objects:
                self.objects[(t, out)] = set()
            self.objects[(t, out)].add(action_key)

        return outputs

    def add_function(self, name: str, func: 'Function'):
        self.functions[name] = func

    def trace(self, t, obj):
        return self.objects.get((t,obj), set())
    
    def get_actions_for_function(self, func_name: str) -> list[tuple]:
        """Return all actions that used the given function name."""
        return [action for action in self.actions.keys() if action[0] == func_name]
    
    @staticmethod
    def init_empty_map() -> 'ComputationalMap':
        return ComputationalMap({}, {}, {})
    
    def __repr__(self):
        return (f"ComputationalMap(objects={len(self.objects)}, "
                f"functions={list(self.functions.keys())}, "
                f"actions={len(self.actions)})")

class SimpleMapper:
    def __init__(self, inp_types, inputs, target_type, target, functions, heuristic_distance: callable):
        self.comp_map = ComputationalMap.init_empty_map()
        self.target = target
        self.target_type = target_type
        self.heuristic_distance = heuristic_distance
        self.objects = set((inp_type, inp) for inp_type, inp in zip(inp_types, inputs))
        self.queues = {name: deque() for name in functions}
        self.visited_actions = set()
        self.target_found = False

        # Register functions
        for name, func in functions.items():
            self.comp_map.add_function(name, func)

        # Initialize queues
        for name, func in functions.items():
            valid_objects_by_type = [
                [obj for obj_type, obj in self.objects if obj_type == t]
                for t in func.input_types
            ]
            for inputs_tuple in product(*valid_objects_by_type):
                self.queues[name].append(inputs_tuple)
    
    def step(self, heuristic_cutoff=float('inf')):
        for func_name, queue in self.queues.items():
            if not queue:
                continue
            inputs = queue.popleft()
            if (func_name,) + inputs in self.visited_actions:
                continue
            self.visited_actions.add((func_name,) + inputs)

            # Apply function
            outputs = self.comp_map.act(func_name, inputs)
            # Check target
            for out_type, out in zip(self.comp_map.functions[func_name].output_types, outputs):
                if out_type == type(self.target) and self.heuristic_distance(out, self.target) > heuristic_cutoff:
                    continue
                self.objects.add((out_type, out))
                if out == self.target and out_type == type(self.target) and not self.target_found:
                    self.target_found = True
                    print(f"*** Target found via {func_name}{inputs} = {out}")

            # Queue new function applications
            for f_name, f_obj in self.comp_map.functions.items():
                valid_objects_by_type = [
                    [obj for obj_type, obj in self.objects if obj_type == t]
                    for t in f_obj.input_types
                ]
                for new_inputs in product(*valid_objects_by_type):
                    if (f_name,) + new_inputs not in self.visited_actions:
                        self.queues[f_name].append(new_inputs)

    def search(self, max_steps=1000, heuristic_cutoff=float('inf')):
        for _ in range(max_steps):
            if all(len(q) == 0 for q in self.queues.values()):
                break
            self.step(heuristic_cutoff)
        if self.target_found:
            print("Target was discovered!")
        else:
            print("Target was NOT discovered.")

def extract_minimal_subgraph(comp_map, target_obj):
    minimal_map = ComputationalMap.init_empty_map()
    visited_objects = set()
    visited_actions = set()

    def trace_back(obj_type, obj):
        if (obj_type, obj) in visited_objects:
            return
        visited_objects.add((obj_type, obj))

        if (obj_type, obj) not in minimal_map.objects:
            minimal_map.objects[(obj_type, obj)] = set()

        # If no parents, it's an input
        if (obj_type, obj) not in comp_map.objects or len(comp_map.objects[(obj_type, obj)]) == 0:
            return

        for action_key in comp_map.objects[(obj_type, obj)]:
            if action_key in visited_actions:
                continue
            visited_actions.add(action_key)

            func_name = action_key[0]
            inputs = action_key[1:]
            func = comp_map.functions[func_name]
            input_types = func.input_types

            minimal_map.actions[action_key] = comp_map.actions[action_key]
            if func_name not in minimal_map.functions:
                minimal_map.functions[func_name] = func

            # connect outputs
            for t, out in comp_map.actions[action_key]:
                if (t, out) not in minimal_map.objects:
                    minimal_map.objects[(t, out)] = set()
                minimal_map.objects[(t, out)].add(action_key)

            # recurse on inputs with DECLARED types
            for (decl_t, inp_val) in zip(input_types, inputs):
                trace_back(decl_t, inp_val)

    trace_back(type(target_obj), target_obj)
    return minimal_map

def find_minimal_steps_in_comp_map(comp_map, start_values, target_value):
    """
    Clean forward BFS approach.
    """
    target_type = type(target_value)
    target_node = (target_type, target_value)
    
    # Track available objects and their discovery distance
    available = {}  # object_node -> distance when discovered
    for val in start_values:
        node = (type(val), val)
        if node == target_node:
            return 0
        available[node] = 0
    
    # Track which actions we've applied
    applied_actions = set()
    
    # We'll keep applying actions until no new objects are discovered
    # or we find the target
    changed = True
    while changed:
        changed = False
        
        # Try all actions
        for action_key in comp_map.actions:
            if action_key in applied_actions:
                continue
            
            # Check if we have all inputs for this action
            func_name = action_key[0]
            inputs = action_key[1:]
            func = comp_map.functions[func_name]
            input_types = func.input_types
            
            # Check if all inputs are available
            all_inputs_available = True
            max_input_dist = 0
            for inp_type, inp_val in zip(input_types, inputs):
                inp_node = (inp_type, inp_val)
                if inp_node not in available:
                    all_inputs_available = False
                    break
                max_input_dist = max(max_input_dist, available[inp_node])
            
            if not all_inputs_available:
                continue
            
            # We can apply this action!
            applied_actions.add(action_key)
            action_dist = max_input_dist + 1
            
            # Add all outputs
            for out_type, out_val in comp_map.actions[action_key]:
                out_node = (out_type, out_val)
                
                if out_node not in available or action_dist < available[out_node]:
                    available[out_node] = action_dist
                    changed = True
                    
                    if out_node == target_node:
                        return action_dist
    
    return float('inf')
####################### Computational Map Visualizer #####################

from graphviz import Digraph

def visualize_computational_map(cm: "ComputationalMap", solution_objects: set = None, filename="computational_map"):
    """
    Visualizes the ComputationalMap using Graphviz.

    - Function nodes are circles
    - Object nodes are boxes (type, value)
    - Inputs → Function → Outputs

    Args:
        cm: ComputationalMap instance
        solution_objects: optional set of (type, value) pairs to highlight in red
        filename: base name for output file (without extension)
    """
    dot = Digraph(comment="Computational Map")
    dot.attr(rankdir='TB')  # top-to-bottom

    if solution_objects is None:
        solution_objects = set()

    added_nodes = set()

    # Build reverse map: which (type,value) objects are used as inputs
    input_usage = set()
    for action_key, outputs in cm.actions.items():
        _, *inputs = action_key
        input_usage.update(inputs)

    for idx, (action_key, outputs) in enumerate(cm.actions.items()):
        func_name, *inputs = action_key

        # --- Function node ---
        func_node_id = f"F_{idx}_{func_name}"
        if func_node_id not in added_nodes:
            dot.node(func_node_id, func_name,
                     shape="circle",
                     style="filled",
                     fillcolor="lightblue",
                     fontname="helvetica")
            added_nodes.add(func_node_id)

        # --- Input objects ---
        for obj in inputs:
            # obj is a raw value (e.g. 20), but cm.objects uses (type, value)
            obj_type = type(obj)
            obj_pair = (obj_type, obj)
            obj_label = f"{obj_type.__name__}:{obj}"

            obj_node_id = f"O_{hash(obj_pair)}"
                # produced by other actions → orange
            color = "orange"
            if cm.objects[obj_pair] == set():
                # initial inputs → lightgreen
                color = "lightgreen"

            if obj_node_id not in added_nodes:
                dot.node(obj_node_id, obj_label,
                         shape="box",
                         style="filled",
                         fillcolor=color,
                         fontname="helvetica")
                added_nodes.add(obj_node_id)
            dot.edge(obj_node_id, func_node_id, arrowsize="0.7", penwidth="1.5")

        # --- Output objects ---
        for obj_type, obj in outputs:
            obj_pair = (obj_type, obj)
            obj_label = f"{obj_type.__name__}:{obj}"

            obj_node_id = f"O_{hash(obj_pair)}"
            color = "red" if obj_pair in solution_objects else "orange"

            if obj_node_id not in added_nodes:
                dot.node(obj_node_id, obj_label,
                         shape="box",
                         style="filled",
                         fillcolor=color,
                         fontname="helvetica")
                added_nodes.add(obj_node_id)
            dot.edge(func_node_id, obj_node_id, arrowsize="0.7", penwidth="1.5")

    # --- Legend ---
    with dot.subgraph(name='cluster_legend') as legend:
        legend.attr(label='Legend', style='rounded', color='gray')
        legend.node('L_func', 'Function', shape="circle", style="filled", fillcolor="lightblue")
        legend.node('L_input', 'Input', shape="box", style="filled", fillcolor="lightgreen")
        legend.node('L_output', 'Intermediate Object', shape="box", style="filled", fillcolor="orange")
        legend.node('L_solution', 'Solution Object', shape="box", style="filled", fillcolor="red")
        legend.attr(rank='same')

    # --- Render ---
    dot.render(filename, format="png", cleanup=True)
    print(f"Computational map saved as '{filename}.png'")
    return dot



######################## tools for variables defined in ASTs ########################


def simulate_simple_stmt(stmt, sim_values: dict, known_funcs: dict) -> bool:
    """Apply a single FunctionCallAssign or DirectAssign in place on sim_values.
    Returns True on success, False on missing input var, wrong output count,
    runtime error, or unrecognized stmt type. Used by both the optimistic
    body-feasibility check and the exact Phase-3 simulator."""
    if isinstance(stmt, FunctionCallAssignNode):
        try:
            input_vals = tuple(sim_values[arg] for arg in stmt.arg_names)
        except KeyError:
            return False
        func = known_funcs.get(stmt.func_name)
        if func is None:
            return False
        try:
            outs = func.func(*input_vals)
        except Exception:
            return False
        if not isinstance(outs, (tuple, list)):
            outs = (outs,)
        if len(outs) != len(stmt.var_names):
            return False
        for v, o in zip(stmt.var_names, outs):
            sim_values[v] = o
        return True
    if isinstance(stmt, DirectAssignNode):
        try:
            src_vals = [sim_values[s] for s in stmt.source_vars]
        except KeyError:
            return False
        for s in stmt.source_vars:
            sim_values.pop(s, None)
        for t, v in zip(stmt.target_vars, src_vals):
            sim_values[t] = v
        return True
    return False


def get_variables_defined_in_node(ast: BlockNode, node_pos: ASTNodePosition) -> set[str]:
    block = ast.get_node_at_position(node_pos)
    if not isinstance(block, BlockNode):
        raise TypeError(f"Node at position {node_pos} is not a BlockNode")

    defined: set[str] = set()

    for child in block.children:
        if isinstance(child, FunctionCallAssignNode):
            for v in child.var_names:
                if v and v[0] == 'x':
                    defined.add(v)

        elif isinstance(child, DirectAssignNode):
            for v in child.target_vars:
                if v and v[0] == 'x':
                    defined.add(v)

        elif isinstance(child, IfElseNode):
            else_block = child.else_block
            if else_block.children:
                final_assignment = else_block.children[-1]
                if isinstance(final_assignment, DirectAssignNode):
                    for v in final_assignment.target_vars:
                        if v and v[0] == 'x':
                            defined.add(v)

        elif isinstance(child, WhileNode):
            while_block = child.block
            if while_block.children:
                final_assignment = while_block.children[-1]
                if isinstance(final_assignment, DirectAssignNode):
                    for v in final_assignment.target_vars:
                        if v and v[0] == 'x':
                            defined.add(v)

    return defined

def get_variables_used_as_inputs_in_node(ast: BlockNode, node_pos: ASTNodePosition):
    # TODO: needs to be fixed.
    block = ast.get_node_at_position(node_pos)
    if not isinstance(block, BlockNode):
        raise TypeError(f"Node at position {node_pos} is not a BlockNode")

    used: set[str] = set()
    defined = get_variables_defined_in_node(ast, node_pos)
    for child in block.children:
        if isinstance(child, FunctionCallAssignNode):
            for v in child.arg_names:
                if v and v[0] == 'x' and v not in defined:
                    used.add(v)

        elif isinstance(child, DirectAssignNode):
            for v in child.source_vars:
                if v and v[0] == 'x' and v not in defined:
                    used.add(v)
    return used


def get_unused_variables_in_ast(ast: BlockNode) -> list[str]:
    defined: set[str] = set()
    used: set[str] = set()

    def visit_block(block: BlockNode):
        for child in block.children:

            # -------- definitions --------
            if isinstance(child, FunctionCallAssignNode):
                for v in child.var_names:
                    if v and v[0] == "x":
                        defined.add(v)

                # -------- uses --------
                for v in child.arg_names:
                    if v and v[0] == "x":
                        used.add(v)

            elif isinstance(child, DirectAssignNode):
                for v in child.target_vars:
                    if v and v[0] == "x":
                        defined.add(v)

                for v in child.source_vars:
                    if v and v[0] == "x":
                        used.add(v)

            elif isinstance(child, IfElseNode):
                # condition uses (if you have them)
                if hasattr(child, "cond_vars"):
                    for v in child.cond_vars:
                        if v and v[0] == "x":
                            used.add(v)

                visit_block(child.if_block)
                visit_block(child.else_block)

            elif isinstance(child, WhileNode):
                # condition uses
                if hasattr(child, "cond_vars"):
                    for v in child.cond_vars:
                        if v and v[0] == "x":
                            used.add(v)

                visit_block(child.block)
            elif isinstance(child, ReturnNode):
                used.add(child.return_var_name)
            else:
                pass

    # traverse whole AST
    if not isinstance(ast, BlockNode):
        raise TypeError("AST root must be BlockNode")

    visit_block(ast)

    # -------- return uses (final assignment convention) --------
    if len(ast.children) > 0:
        last = ast.children[-1]

        if isinstance(last, DirectAssignNode):
            for v in last.source_vars:
                if v and v[0] == "x":
                    used.add(v)

        elif isinstance(last, FunctionCallAssignNode):
            for v in last.var_names:
                if v and v[0] == "x":
                    used.add(v)

        elif isinstance(last, IfElseNode):
            if last.else_block.children:
                final = last.else_block.children[-1]
                if isinstance(final, DirectAssignNode):
                    for v in final.source_vars:
                        if v and v[0] == "x":
                            used.add(v)

        elif isinstance(last, WhileNode):
            if last.block.children:
                final = last.block.children[-1]
                if isinstance(final, DirectAssignNode):
                    for v in final.source_vars:
                        if v and v[0] == "x":
                            used.add(v)

    unused = defined - used
    return sorted(unused)



######################## Annotated AST tools ########################

def get_subnode_action_exec_positions(annotated_ast: AnnotatedAST, node_position: ASTNodePosition) -> list[tuple]:
    """
    Return all ExecutionPosition tuples that occur inside the subtree rooted at node_position.
    """
    node_prefix = node_position
    positions = []
    for action, exec_position in annotated_ast.mapping.items():
        pos_tuple = exec_position[0]
        if pos_tuple[:len(node_prefix)] == node_prefix:  # in this subtree
            positions.append(exec_position)
    positions.sort(key=lexico_position)
    return positions

def get_node_exec_positions(annotated_ast: AnnotatedAST, node_position: ASTNodePosition):
    """
    Return entries and skips of a node using set arithmetic.
    Each element is (node_position, temporal_stack_prefix).
    """
    parent_node_position = node_position[:-1]
    parent_entry_times = {exec_position[1][:len(parent_node_position)+1] for exec_position in get_subnode_action_exec_positions(annotated_ast, parent_node_position)}
    child_entry_times = {exec_position[1][:len(node_position)+1] for exec_position in get_subnode_action_exec_positions(annotated_ast, node_position)}
    # Entries are just the child_entry_times
    entries = [(node_position, et) for et in child_entry_times]
    # Skips: parent times that do NOT have any child entry extending them
    skips = [(node_position, pt + (1,)) for pt in parent_entry_times if not any(ct[:len(pt)] == pt for ct in child_entry_times)]
    return {"entries": entries, "skips": skips}

def get_while_node_exec_positions(annotated_ast: AnnotatedAST, while_node_position: ASTNodePosition):
    """
    Return 'entries' and 'skips' for a while loop node, correctly handling nesting.
    
    entries: times when the while loop body was entered.
    skips: times when the while loop was evaluated but not entered (loop condition false).
    
    This works even for nested loops — it computes one skip per parent iteration.
    """

    # --- Step 1: Get execution positions for the while node itself ---
    # (we do not use a specific "body" child; while node covers the entire loop)
    while_exec = get_node_exec_positions(annotated_ast, while_node_position)
    entries = while_exec["entries"]  # loop body executions
    skips = while_exec["skips"]      # basic skips (usually from body skip events)
    # --- Step 2: Get parent entries (the context in which this loop may run) ---
    parent_node_position = while_node_position[:-1]
    parent_exec = get_node_exec_positions(annotated_ast, parent_node_position)
    parent_entries = parent_exec["entries"]

    # --- Step 3: Group all while entries by their parent entry prefix ---
    # e.g. for while_node (0,1,1), parent prefix length = len(parent_node_position)+1 = 3
    prefix_len = len(parent_node_position) + 1
    grouped_by_parent = {}
    for entry in entries:
        prefix = entry[1][:prefix_len]
        grouped_by_parent.setdefault(prefix, []).append(entry)

    # --- Step 4: For each parent iteration, find the last while iteration ---
    final_skips = []
    for parent_entry in parent_entries:
        parent_prefix = parent_entry[1][:prefix_len]
        inner_entries = grouped_by_parent.get(parent_prefix, [])

        if inner_entries:
            # loop ran → skip happens after its last iteration
            last_time = max(e[1] for e in inner_entries)
            skip_time = tuple(list(last_time[:-1]) + [last_time[-1] + 1])
        else:
            # loop skipped entirely under this parent → skip time = parent time + 1
            skip_time = tuple(list(parent_entry[1]) + [1])

        final_skips.append((while_node_position, skip_time))

    # --- Step 5: Merge and sort ---
    all_skips = list(set(skips).union(set(final_skips)))
    entries.sort(key=lambda x: x[1])
    all_skips.sort(key=lambda x: x[1])

    return {"entries": entries, "skips": all_skips}

def get_if_else_node_exec_positions(annotated_ast: AnnotatedAST, if_else_node_position: ASTNodePosition):
    """
    Return entries and skips for an if-else node (IF block non-empty assumption).

    - entries: times when IF block was entered
    - skips: parent iterations where IF was not entered
    """

    # IF block child position
    if_node_position = if_else_node_position + (1,)

    # --- Step 1: Get IF block executions ---
    if_exec = get_node_exec_positions(annotated_ast, if_node_position)
    entries = if_exec["entries"]

    # --- Step 2: Get parent entries ---
    parent_node_position = if_else_node_position[:-1]
    parent_exec = get_node_exec_positions(annotated_ast, parent_node_position)
    parent_entries = parent_exec["entries"]

    prefix_len = len(parent_node_position) + 1

    # --- Step 3: Compute skips ---
    grouped_if_entries = {}
    for entry in entries:
        prefix = entry[1][:prefix_len]
        grouped_if_entries.setdefault(prefix, []).append(entry)

    skips = []
    for parent_entry in parent_entries:
        parent_prefix = parent_entry[1][:prefix_len]
        inner_if_entries = grouped_if_entries.get(parent_prefix, [])

        if not inner_if_entries:
            # IF block never ran for this parent iteration → skip at parent time + 1
            skip_time = tuple(list(parent_entry[1]) + [1])
            skips.append((if_node_position, skip_time))

    # --- Step 4: Sort ---
    entries.sort(key=lambda x: x[1])
    skips.sort(key=lambda x: x[1])

    return {"entries": entries, "skips": skips}

def get_while_node_exec_position_groups(annotated_ast: AnnotatedAST, node_position: ASTNodePosition):
    """
    Group while loop executions by their parent entry.
    Returns a list of groups, where each group is a dict:
        {"entries": [...], "skips": [...]}
    Each group represents one full run of the while loop (one parent-level activation).
    """

    exec_positions = get_while_node_exec_positions(annotated_ast, node_position)
    entries = exec_positions["entries"]
    skips = exec_positions["skips"]

    groups: dict[tuple, dict[str, list]] = {}

    for tag, positions in (("entries", entries), ("skips", skips)):
        for node_pos, exec_time in positions:
            # Group by parent-level prefix
            parent_prefix = exec_time[:len(node_position)]
            if parent_prefix not in groups:
                groups[parent_prefix] = {"entries": [], "skips": []}
            groups[parent_prefix][tag].append((node_pos, exec_time))

    # Sort within each group
    for group in groups.values():
        group["entries"].sort(key=lambda p: p[1])
        group["skips"].sort(key=lambda p: p[1])

    # Return list ordered by parent prefix (temporal order)
    return [groups[k] for k in sorted(groups.keys())]


    
    

####################### Tools/ important general functions ############################

from itertools import product


def get_objects_by_type(trace: SimpleCompEnv) -> dict[type, list[ObjId]]:
        by_type = {}
        for obj_id, obj in trace.objects.items():
            by_type.setdefault(obj.value_type, [])
            by_type[obj.value_type].append(obj_id)
        return by_type

def propagate_importance_multi(comp_map: ComputationalMap, target_obj, max_passes=10, tol=1e-6, available=None) -> dict[object, float]:
    """
    Iterative propagation of importance scores for objects in comp_map
    relative to target_obj, allowing multiple passes for smoother distribution.

    Args:
        comp_map (ComputationalMap): Computational graph.
        target_obj (object): The target object to propagate importance from.
        max_passes (int): Max number of iterations.
        tol (float): Convergence tolerance.
        available (set): Objects already in the trace; treated as cost-free.

    Returns:
        dict: {obj: importance_score}
    """
    if available is None:
        available = set()

    importance = defaultdict(float)
    importance[type(target_obj), target_obj] = 1.0

    for _ in range(max_passes):
        new_importance = defaultdict(float)
        new_importance[type(target_obj), target_obj] = 1.0  # target always fixed at 1

        # Already-available objects are "satisfied", so clamp them at 0
        for obj, score in importance.items():
            if obj in available or score == 0:
                continue

            producing_actions = list(comp_map.objects.get(obj, []))
            if not producing_actions:
                continue

            action_value = score / len(producing_actions)

            for action in producing_actions:
                inputs = set(zip(comp_map.functions[action[0]].input_types, action[1:]))
                if not inputs:
                    continue
                per_input_value = action_value / len(inputs)
                for inp in set(inputs):
                    new_importance[inp] += per_input_value

        # check convergence
        diff = sum(abs(new_importance[k] - importance.get(k, 0.0)) for k in set(new_importance) | set(importance))
        importance = new_importance
        if diff < tol:
            break

    return dict(importance)

####################### Trace searcher ###############################
class Problem:
    def __init__(self, input_types: tuple[type], output_types: tuple[type], instances: dict[int, tuple[tuple]]):
        self.input_types = input_types
        self.output_types = output_types
        self.instances = instances  # {0: (inputs0, outputs0), ...}
        self.inputs = [instances[i][0] for i in range(len(instances))]
        self.targets = [instances[i][1][0] for i in range(len(instances))]

    def add_instance(self, instance: tuple[tuple]):
        idx = len(self.instances)
        self.instances[idx] = instance

    def __repr__(self):
        return f"Problem(instances={self.instances})"
    
    def __str__(self):
        res = 'Problem: \n'
        for instance in self.instances.values():
            res += f"{instance[0]} -> {instance[1]} \n"
        return res
    
    def get_inputs(self):
        return tuple(tuple(zip(self.input_types, instance[0])) for instance in self.instances.values())

    def extend(self, other: "Problem") -> "Problem":
        if self.input_types != other.input_types:
            raise TypeError(
                f"Cannot merge problems: input types differ:\n"
                f"  self:  {self.input_types}\n"
                f"  other: {other.input_types}"
            )

        if self.output_types != other.output_types:
            raise TypeError(
                f"Cannot merge problems: output types differ:\n"
                f"  self:  {self.output_types}\n"
                f"  other: {other.output_types}"
            )

        next_idx = len(self.instances)
        for _, inst in other.instances.items():
            self.instances[next_idx] = inst
            next_idx += 1

        return self


class InterDependentProblem(Problem):
    def __repr__(self):
        return (f"InterDependentProblem(inputs={tuple(t.__name__ for t in self.input_types)}, output={self.output_types.__name__}, n_instances={len(self.instances)})")
    
    def __str__(self):
        return super().__str__()


class InterDependentProblemGroup:
    def __init__(self, problems: list[InterDependentProblem]):
        self.problems = problems

    def __repr__(self):
        inner = ", ".join([f" {i}:{repr(p)}" for i, p in enumerate(self.problems)])
        return f"InterDependentProblemGroup([{inner}])"

    __str__ = __repr__


class TraceSearcher:
    def __init__(self, problem: Problem, known_funcs: dict[str:Function]):
        self.problem = problem
        self.possible_trace_sets = PriorityQueue()
        self.known_funcs = known_funcs
        self.maps = dict()
        initial_traces = dict()
        
        #TODO: add type verification
        for i, (inputs, outputs) in problem.instances.items():
            trace = SimpleCompEnv(known_funcs)
            for inp in inputs:
                inp_obj = CompObject(type(inp), inp)
                trace.add_input_object(inp_obj)
            initial_traces[i] = [trace, outputs]
        
        self.possible_trace_sets.put((0, initial_traces))
    
    def create_computational_maps(self):
        for i, (inputs, outputs) in self.problem.instances.items():
            mapper = SimpleMapper(self.problem.input_types, inputs, outputs[0], self.problem.output_types, self.known_funcs, heuristic_distance=None)
            mapper.search(max_steps=10000)
            clean_map = extract_minimal_subgraph(mapper.comp_map, outputs[0])
            self.maps[i] = clean_map

########################### Syntax tree tools ###############################

def compute_func_call_output_hash(output_index: int, func_name: FunctionName, inputs: tuple[str], variable_hashes: dict[str, int]):
    return hash((func_name, *(variable_hashes[inp] for inp in inputs), output_index))

def compute_direct_assign_target_hash(target_index: int, source_var: str, variable_hashes: dict[str, int]):
    return hash((target_index, variable_hashes[source_var]))

def compute_func_call_hash(func_call: FunctionCallAssignNode, variable_hashes: dict[str, int]):
    return hash((func_call.func_name, *(variable_hashes[inp] for inp in func_call.arg_names)))

def compute_direct_assign_hash(direct_assign: DirectAssignNode, variable_hashes: dict[str, int]):
    return hash(tuple(variable_hashes[s] for s in direct_assign.source_vars))

def compute_bool_expr_hash(bool_expr: BoolExprNode, variable_hashes: dict[str, int]):
    return hash((bool_expr.bool_func, *(variable_hashes[inp] for inp in bool_expr.arg_names)))

def compute_return_hash(return_node: ReturnNode, variable_hashes):
    return hash(('return', variable_hashes[return_node.return_var_name]))

def compute_block_hash(block: BlockNode, variable_hashes: dict[str, int]) -> int:
    var_hashes = variable_hashes.copy()
    child_hashes = []
    for node in block.children:
        if isinstance(node, FunctionCallAssignNode):
            child_hashes.append(compute_func_call_hash(node, var_hashes))
            for i, out in enumerate(node.var_names):
                var_hashes[out] = compute_func_call_output_hash(i, node.func_name, node.arg_names, var_hashes)

        elif isinstance(node, DirectAssignNode):
            child_hashes.append(compute_direct_assign_hash(node, var_hashes))
            for i, target in enumerate(node.target_vars):
                var_hashes[target] = compute_direct_assign_target_hash(i, node.source_vars[i], var_hashes)

        elif isinstance(node, IfElseNode):
            if_hash, new_var_hashes = compute_if_else_hash(node, var_hashes)
            var_hashes = new_var_hashes
            child_hashes.append(if_hash)

        elif isinstance(node, WhileNode):
            while_hash, new_var_hashes = compute_while_hash(node, var_hashes)
            var_hashes = new_var_hashes
            child_hashes.append(while_hash)

    return hash(tuple(sorted(child_hashes))), var_hashes


def compute_while_hash(while_node: 'WhileNode', variable_hashes: dict[str, int]):
    """Hash a WhileNode and propagate the body's terminating-rebind target_vars
    out to the surrounding scope (those are the only body vars visible after
    the loop). Body-internal vars (intermediate computations) stay scoped."""
    var_hashes = variable_hashes.copy()
    bool_expr_hash = compute_bool_expr_hash(while_node.bool_expr, var_hashes)
    body_hash, body_var_hashes = compute_block_hash(while_node.block, var_hashes)

    # Only the body's last DirectAssign target_vars survive past the loop —
    # those are the rebind that end_while inserts.
    if while_node.block.children:
        final = while_node.block.children[-1]
        if isinstance(final, DirectAssignNode):
            for tgt in final.target_vars:
                if tgt in body_var_hashes:
                    var_hashes[tgt] = body_var_hashes[tgt]

    return hash((bool_expr_hash, body_hash)), var_hashes

def compute_if_else_hash(if_else: IfElseNode, variable_hashes: dict[str, int]):
    var_hashes = variable_hashes.copy()
    
    bool_expr = if_else.bool_expr
    if_block = if_else.if_block
    else_block = if_else.else_block

    bool_expr_hash = compute_bool_expr_hash(bool_expr, var_hashes)
    if_hash, if_var_hashes = compute_block_hash(if_block, var_hashes)
    else_hash, else_var_hashes = compute_block_hash(else_block, var_hashes)
    
    for var in if_var_hashes:
        if var in var_hashes:
            continue
        elif var in else_var_hashes:
            new_var_hash = hash((if_var_hashes[var], else_var_hashes[var]))
            var_hashes[var] = new_var_hash

    return hash((bool_expr_hash, if_hash, else_hash)), var_hashes




#################### Examples / Main ##############################

def main():
    def example1():
        # Functions
        def add(x, y): return (x + y,)
        def get_head(lst): return (lst[0],) if lst else ()
        def get_tail(lst): return (lst[1:],) if lst else ()

        add_func = Function(add, [int, int], [int])
        head_func = Function(get_head, [tuple], [int])
        tail_func = Function(get_tail, [tuple], [tuple])

        known_funcs = {
            'add': add_func,
            'get_head': head_func,
            'get_tail': tail_func
        }

        problem = Problem((list,),(int,), {
            0: (((1,4,6),),(11,)),
            1: (((4,10,4,2),), (20,)),
        })

        trace_searcher = TraceSearcher(problem, known_funcs)
        trace_searcher.create_computational_maps()
        return trace_searcher
    
    trace_searcher: TraceSearcher = example1()
    map: ComputationalMap = trace_searcher.maps[1]
    traces = trace_searcher.possible_trace_sets.get()
    print(traces[1][1][0])
    print(map.actions)
    # print(get_all_available_actions(traces[1][1][0], map))
    print()
    print(propagate_importance_multi(map, 20))

if __name__ == '__main__':
    main()