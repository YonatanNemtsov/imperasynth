from collections import deque
from itertools import product
from typing import Hashable
from core_lang_env.comp_env import Function

class ComputationalMap:
    def __init__(self, objects: dict = None, functions: dict[str, 'Function'] = None, actions: dict = None):
        self.objects: dict[object, set] = objects if objects else {}
        self.functions: dict[str, 'Function'] = functions if functions else {}
        self.actions: dict[tuple, tuple] = actions if actions else {}

    def act(self, func_name: str, inputs: tuple[Hashable]):
        if func_name not in self.functions:
            raise ValueError(f"Function '{func_name}' is not registered.")

        func_obj = self.functions[func_name]


        for obj, t in zip(inputs, func_obj.input_types):
            if type(obj) != t:
                raise TypeError(f"Function '{func_name}' got wrong type {type(obj)} argument")

        # Call the underlying function
        outputs = func_obj.func(*inputs)

        # Record the action
        action_key = (func_name,) + tuple(inputs)
        self.actions[action_key] = outputs

        # Track outputs
        for out in outputs:
            if out not in self.objects:
                self.objects[out] = set()
            self.objects[out].add(action_key)

        return outputs

    def add_function(self, name: str, func: 'Function'):
        self.functions[name] = func

    def trace(self, obj):
        return self.objects.get(obj, set())

    def __repr__(self):
        return (f"ComputationalMap(objects={len(self.objects)}, "
                f"functions={list(self.functions.keys())}, "
                f"actions={len(self.actions)})")

class SimpleMapper:
    def __init__(self, inputs, target, functions, heuristic_distance):
        self.comp_map = ComputationalMap()
        self.target = target
        self.heuristic_distance = heuristic_distance
        self.objects = set(inputs)
        self.queues = {name: deque() for name in functions}
        self.visited_actions = set()
        self.target_found = False

        # Register functions
        for name, func in functions.items():
            self.comp_map.add_function(name, func)

        # Initialize queues
        for name, func in functions.items():
            valid_objects_by_type = [
                [obj for obj in self.objects if isinstance(obj, t)]
                for t in func.input_types
            ]
            for inputs_tuple in product(*valid_objects_by_type):
                self.queues[name].append(inputs_tuple)

    def step(self, heuristic_cutoff='inf'):
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
            for out in outputs:
                self.objects.add(out)
                if out == self.target and not self.target_found:
                    self.target_found = True
                    print(f"*** Target found via {func_name}{inputs} = {out}")

            # Queue new function applications
            for f_name, f_obj in self.comp_map.functions.items():
                valid_objects_by_type = [
                    [obj for obj in self.objects if isinstance(obj, t)]
                    for t in f_obj.input_types
                ]
                for new_inputs in product(*valid_objects_by_type):
                    if (f_name,) + new_inputs not in self.visited_actions:
                        self.queues[f_name].append(new_inputs)

    def search(self, max_steps=1000, heuristic_cutoff='inf'):
        for _ in range(max_steps):
            if all(len(q) == 0 for q in self.queues.values()):
                break
            self.step(heuristic_cutoff)
        if self.target_found:
            print("Target was discovered!")
        else:
            print("Target was NOT discovered.")



def extract_minimal_subgraph(comp_map, target_obj):
    """
    Extracts a minimal ComputationalMap that contains only
    the objects and actions needed to compute `target_obj`.
    
    Args:
        comp_map (ComputationalMap): The full computational graph.
        target_obj (object): The object to trace back.
    
    Returns:
        ComputationalMap: Minimal subgraph.
    """
    minimal_map = ComputationalMap()

    visited_objects = set()
    visited_actions = set()

    def trace_back(obj):
        if obj in visited_objects:
            return
        visited_objects.add(obj)

        # Add the object to the minimal map
        if obj not in minimal_map.objects:
            minimal_map.objects[obj] = set()

        # If this object has no parent actions, it's an input
        if obj not in comp_map.objects or len(comp_map.objects[obj]) == 0:
            return

        # Trace each parent action that produces this object
        for action_key in comp_map.objects[obj]:
            if action_key in visited_actions:
                continue
            visited_actions.add(action_key)

            # Add action to minimal map
            func_name = action_key[0]
            inputs = action_key[1:]

            minimal_map.actions[action_key] = comp_map.actions[action_key]

            # Ensure the function is registered
            if func_name not in minimal_map.functions:
                minimal_map.functions[func_name] = comp_map.functions[func_name]

            # Connect outputs
            for out in comp_map.actions[action_key]:
                if out not in minimal_map.objects:
                    minimal_map.objects[out] = set()
                minimal_map.objects[out].add(action_key)

            # Recursively trace inputs
            for inp in inputs:
                trace_back(inp)

    # Start tracing from the target
    trace_back(target_obj)
    return minimal_map







############## Visualizer #################

from graphviz import Digraph

def visualize_computational_map(cm: "ComputationalMap", solution_objects: set = None, filename="computational_map"):
    """
    Visualizes the ComputationalMap using Graphviz.
    
    - Functions are circles
    - Objects are squares
    - Inputs → Function → Outputs
    
    Args:
        cm: ComputationalMap instance
        solution_objects: optional set of objects to highlight in red
        filename: base name for output file (without extension)
    """
    dot = Digraph(comment="Computational Map")
    dot.attr(rankdir='TB')  # Top-to-bottom layout
    
    if solution_objects is None:
        solution_objects = set()

    added_nodes = set()

    # Build reverse map to track which objects are used as inputs
    input_usage = set()
    for action_key, outputs in cm.actions.items():
        _, *inputs = action_key
        input_usage.update(inputs)

    # Iterate through actions
    for idx, (action_key, outputs) in enumerate(cm.actions.items()):
        func_name, *inputs = action_key

        # Add function node
        func_node_id = f"F_{idx}_{func_name}"
        if func_node_id not in added_nodes:
            dot.node(func_node_id, func_name,
                     shape="circle",
                     style="filled",
                     fillcolor="lightblue",
                     fontname="helvetica")
            added_nodes.add(func_node_id)

        # Add input object nodes (green if they are only inputs, orange if produced by other actions)
        for obj in inputs:
            obj_node_id = f"O_{id(obj)}"
            color = "lightgreen" if obj not in cm.objects else "orange"
            if obj_node_id not in added_nodes:
                dot.node(obj_node_id, str(obj),
                         shape="box",
                         style="filled",
                         fillcolor=color,
                         fontname="helvetica")
                added_nodes.add(obj_node_id)
            dot.edge(obj_node_id, func_node_id, arrowsize="0.7", penwidth="1.5")

        # Add output object nodes
        for obj in outputs:
            obj_node_id = f"O_{id(obj)}"
            if obj in solution_objects:
                color = "red"
            else:
                color = "orange"
            if obj_node_id not in added_nodes:
                dot.node(obj_node_id, str(obj),
                         shape="box",
                         style="filled",
                         fillcolor=color,
                         fontname="helvetica")
                added_nodes.add(obj_node_id)
            dot.edge(func_node_id, obj_node_id, arrowsize="0.7", penwidth="1.5")

    # Legend
    with dot.subgraph(name='cluster_legend') as legend:
        legend.attr(label='Legend', style='rounded', color='gray')
        legend.node('L_func', 'Function', shape="circle", style="filled", fillcolor="lightblue")
        legend.node('L_input', 'Input', shape="box", style="filled", fillcolor="lightgreen")
        legend.node('L_output', 'Intermediate Object', shape="box", style="filled", fillcolor="orange")
        legend.node('L_solution', 'Solution Object', shape="box", style="filled", fillcolor="red")
        legend.attr(rank='same')

    # Save and render
    dot.render(filename, format="png", cleanup=True)
    print(f"Computational map saved as '{filename}.png'")
    return dot
