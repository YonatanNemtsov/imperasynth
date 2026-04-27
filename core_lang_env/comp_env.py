from dataclasses import dataclass
import hashlib
import bisect

############################################# SimpleCompEnv #############################################

class CompObject:
    def __init__(self, value_type, value):
        self.value_type = value_type
        self.value = value
        self.id = None

    def update(self, value):
        self.value = value
    
    def copy(self):
        """Returns a new CompObject with the same attributes but no ID"""
        return CompObject(self.value_type, self.value)

    def __str__(self):
        return f"CompObject({self.value})"
    
    def __repr__(self):
        return f"CompObject({self.value})"

class Function:
    def __init__(self, func, input_types: list[type], output_types: list[type]):
        self.func = func
        self.input_types = input_types
        self.output_types = output_types
    
    def run(self, input_objects: list[CompObject]):
        outputs = self.func(*[obj.value for obj in input_objects])
        return tuple(CompObject(self.output_types[out_index], output) for out_index, output in enumerate(outputs))
    
    def try_run(self, input_objects: list[CompObject]):
        return self.func(*(obj.value for obj in input_objects))
    
    def copy(self):
        """Returns a new Function with the same attributes"""
        return Function(self.func, self.input_types.copy(), self.output_types.copy())

    def __str__(self):
        f"Function({self.func.__name__})"

class BoolCompObject(CompObject):
    def __init__(self, value: bool):
        super().__init__(bool, value)
        if not isinstance(value, bool):
            raise TypeError("BoolCompObject requires boolean value")
    
    def copy(self):
        """Returns a new BoolCompObject with the same value"""
        return BoolCompObject(self.value)

    def __str__(self):
        return f"BoolCompObject({self.value})"
    
    def __repr__(self):
        return f"BoolCompObject({self.value})"
    

class BoolFunction(Function):
    def run(self, input_objects: list[CompObject]):
        outputs = self.func(*[obj.value for obj in input_objects])
        return tuple(BoolCompObject(output) for output in outputs)
    
    def copy(self):
        """Returns a new BoolFunction with the same attributes"""
        return BoolFunction(self.func, self.input_types.copy(), self.output_types.copy())
    
    def __str__(self):
        f"BoolFunction({self.func.__name__})"

ObjId = int
Action = tuple[str, tuple[ObjId], tuple[ObjId]]
ShortAction = tuple[str, tuple[ObjId]] # (func_name, (inp_id1, ...,inp_idn))

class CompSignature:
    def __init__(self, history_short: list[ShortAction]):
        self.sig = sorted(history_short)
    
    def append_action(self, action: Action):
        bisect.insort(self.sig, action)

    def __eq__(self, other: 'CompSignature'):
        return self.sig == other.sig
    
    def __repr__(self):
        return f"CompSignature({self.sig})"
    
    def to_tuple(self):
        return tuple(self.sig)
    
    def copy(self):
        return CompSignature(self.sig.copy())


FunctionName = str 

def create_canonical_id(func_name, input_ids, output_index):
        id_str = f"{func_name}-{'-'.join(map(str, input_ids))}-{output_index}"
        # Compute a 128-bit hash
        hash_bytes = hashlib.md5(id_str.encode('utf-8')).digest()
        # Convert to integer
        return int.from_bytes(hash_bytes, 'big')
    

class SimpleCompEnv:
    def __init__(self, known_funcs={}):
        self.objects: dict[ObjId, CompObject] = {}  # Maps object IDs to CompObject instances
        self.input_objects: list[ObjId] = []
        self.known_functions: dict[FunctionName, Function] = known_funcs  # Maps function names to Function instances
        # self.function_ids = {i:name for i, name in enumerate(known_funcs.keys())}
        self.action_history: list[Action] = []  # Now a list of (func_name, input_ids, output_ids) tuples
        self.action_history_short: list[ShortAction] = []
        self.signature = CompSignature([])
        self.parent_graph = {}
        self.solution_object_id = None
    
    def add_object(self, obj: CompObject, func_name: str, input_ids: tuple, output_index):
        obj_id = create_canonical_id(func_name, input_ids, output_index)  # Assign a new ID
        self.objects[obj_id] = obj
        obj.id = obj_id
        self.parent_graph[obj_id] = None
        return obj_id
    
    def add_input_object(self, obj: CompObject):
        obj_id = len(self.input_objects)  # Assign a new ID
        self.objects[obj_id] = obj
        obj.id = obj_id
        self.parent_graph[obj_id] = None
        self.input_objects.append(obj_id)
        return obj_id

    def add_function(self, name, func: Function):
        self.known_functions[name] = func
    
    def add_bool_function(self, name, func: BoolFunction):
        self.known_bool_functions[name] = func

    def apply_function(self, func_name: str, input_ids: tuple):
        func = self.known_functions[func_name]
        input_objects = tuple(self.objects[obj_id] for obj_id in input_ids)
        
        # Use the Function's run method to process the inputs
        output_objects = func.run(input_objects)
        
        output_ids = []
        for output_index, output_obj in enumerate(output_objects):
            output_id = self.add_object(output_obj, func_name, input_ids, output_index)
            output_ids.append(output_id)
            self.parent_graph[output_id] = (func_name,) + tuple(input_ids)

        # Record the action in the history and the ADG as a tuple
        action = (func_name, tuple(input_ids), tuple(output_ids))
        short_action = action[:-1]
        self.action_history.append(action)
        self.action_history_short.append(short_action)
        self.signature.append_action(short_action)
        return tuple(output_ids)
    
    def apply_action(self, action_key: ShortAction):
        return self.apply_function(action_key[0], action_key[1])
    
    def try_run_function(self, func_name: str, input_ids: tuple):
        func = self.known_functions[func_name]
        input_objects = tuple(self.objects[obj_id] for obj_id in input_ids)
        output_values = func.try_run(input_objects)
        return output_values


    
    def assign_solution_object(self, obj_id):
        self.solution_object_id = obj_id

    
    def copy(self):
        """Returns a deep copy of the environment"""
        new_env = SimpleCompEnv(self.known_functions.copy())
        
        # Copy all objects
        #for obj_id, obj in self.objects.items():
        #    new_obj = obj.copy()
        #    new_obj.id = obj_id
        #    new_env.objects[obj_id] = new_obj
        
        new_env.objects = self.objects.copy()
        
        # Copy other attributes
        new_env.input_objects = self.input_objects.copy()
        new_env.action_history = self.action_history.copy()
        new_env.parent_graph = self.parent_graph.copy()
        new_env.solution_object_id = self.solution_object_id
        
        return new_env
    
    def __repr__(self):
        return f"SimpleCompEnv(inputs={[self.objects[inp] for inp in self.input_objects]} , action_history={str(self.action_history[0:3])[:-1] + ((', ...' + self.action_history[-1] + ']') if len(self.action_history) > 3 else ']')})"
    
    @staticmethod
    def merge(env1:'SimpleCompEnv', env2: 'SimpleCompEnv', id_mapping: dict):
        """
        TODO: THIS NEEDS TO BE REWRITTEN ACCORDING TO NEW ID GENERATION. 

        Merge two environments, ensuring specified objects match.
        
        Args:
            env1: First environment (will be modified to contain the merge result)
            env2: Second environment (will remain unchanged)
            id_mapping: Dictionary {env1_obj_id: env2_obj_id} of objects that must match
        
        Returns:
            The merged environment (a clean copy) and a mapping from env2 object IDs
            to their new IDs in the merged environment
        """
        for env1_id, env2_id in id_mapping.items():
            obj1 = env1.objects[env1_id]
            obj2 = env2.objects[env2_id]
            if obj1.value != obj2.value:
                raise ValueError(f"Objects {env1_id} and {env2_id} have different values: "
                                f"{obj1.value} != {obj2.value}")
        
        id_translation = {}
        merged_env = env1.copy()
        shift = max(merged_env.objects.keys(), default=-1) - len(id_mapping) + 1

        # Add all env2 objects with shifted IDs
        for env2_id, obj in env2.objects.items():
            if env2_id not in id_mapping.values():
                new_id = env2_id + shift
                new_obj = obj.copy()
                new_obj.id = new_id
                merged_env.objects[new_id] = new_obj
                id_translation[env2_id] = new_id
            
            for env1_id, env2_id in id_mapping.items():
                id_translation[env2_id] = env1_id
        
        for env2_id, obj in env2.objects.items():
            if env2_id in id_translation:
                new_id = id_translation[env2_id]
                if env2.parent_graph[env2_id] is not None:
                    # Translate the parent graph entry
                    func_name, *input_ids = env2.parent_graph[env2_id]
                    translated_inputs = [id_translation[i] for i in input_ids]
                    merged_env.parent_graph[new_id] = (func_name, *translated_inputs)
        
        # Merge action histories (with translated IDs)
        for action in env2.action_history:
            func_name, input_ids, output_ids = action
            translated_inputs = [id_translation[i] for i in input_ids]
            translated_outputs = [id_translation[i] for i in output_ids]
            if (func_name, translated_inputs, translated_outputs) not in merged_env.action_history:
                merged_env.action_history.append((func_name, translated_inputs, translated_outputs))
        
        # Merge known functions (without duplicates)
        merged_env.known_functions.update({
            name: func for name, func in env2.known_functions.items()
            if name not in merged_env.known_functions
        })
        
        # Handle solution object if present in env2
        if env2.solution_object_id is not None:
            merged_env.solution_object_id = id_translation[env2.solution_object_id]
        
        return merged_env, id_translation

def get_clean_comp_env(comp_env: SimpleCompEnv):
    """
    Returns a new SimpleCompEnv containing only the objects and actions
    that are ancestors of the solution object, preserving deterministic IDs.
    """
    if comp_env.solution_object_id is None:
        raise ValueError("No solution found.")

    clean_env = SimpleCompEnv(comp_env.known_functions.copy())

    # Keep track of objects already added
    added_objects = set()
    added_actions = set()
    for input_id in comp_env.input_objects:
        inp_obj = comp_env.objects[input_id]
        obj = inp_obj.copy()
        clean_env.add_input_object(obj)
        added_objects.add(obj.id)

    def reconstruct_path(obj_id):
        if obj_id in added_objects:
            # input object, already added. 
            return

        parent_info = comp_env.parent_graph[obj_id]
        if parent_info is None:
            # input object, already added. 
            return

        func_name, *input_ids = parent_info

        # First reconstruct all input objects
        for input_id in input_ids:
            reconstruct_path(input_id)

        # Identify the correct output index for this obj_id
        # Search original action_history for the action that produced obj_id
        action = None
        for act in comp_env.action_history:
            if act[0] == func_name and tuple(act[1]) == tuple(input_ids) and obj_id in act[2]:
                action = act
                break

        if action is None:
            raise RuntimeError(f"Cannot find action producing object {obj_id}")

        func_name, orig_input_ids, orig_output_ids = action
        output_index = orig_output_ids.index(obj_id)

        # Recreate the specific output object
        orig_obj = comp_env.objects[obj_id]
        clean_env.add_object(orig_obj.copy(), func_name, tuple(input_ids), output_index)

        # Record this action if not already added
        short_action = (func_name, tuple(input_ids))
        if short_action not in added_actions:
            clean_env.action_history.append(action)
            clean_env.action_history_short.append(short_action)
            clean_env.signature.append_action(short_action)
            added_actions.add(short_action)

        added_objects.add(obj_id)

    # Reconstruct only the path to the solution
    reconstruct_path(comp_env.solution_object_id)

    # Mark the solution object
    clean_env.solution_object_id = comp_env.solution_object_id

    return clean_env


############################################# Action Dependency Graph #############################################

class ActionDependencyGraph:
    def __init__(self, action_history: list[tuple]):
        """
        Builds an action dependency graph (ADG) from the action history.
        Each node points to the actions it depends on (its prerequisites).

        Args:
            action_history: List of actions, each of the form
                (function_name, (input_ids), (output_ids))
        """
        self.graph = {}        # key: action_node_id, value: tuple of dependencies (action_node_ids)
        self.node_map = {}     # maps output_id -> producing action_node_id
        self.node_ordering = []  # list of action_node_ids in lexicographical order for canonization
        
        for action in action_history:
            self.add_action(action)

    def add_action(self, action: tuple[str, tuple, tuple]):
        func_name, input_ids, output_ids = action

        # Canonical node ID for this action. TODO: outputs_ids can be neglected.
        node_id = (func_name, input_ids, output_ids)

        # Skip if already added
        if node_id in self.graph:
            return

        # Determine dependencies (sorted tuple for deterministic order)
        deps = []
        for inp in input_ids:
            if inp in self.node_map:
                deps.append(self.node_map[inp])
        deps = tuple(sorted(deps))  # deterministic ordering

        self.graph[node_id] = deps

        # Map outputs to this action node
        for out in output_ids:
            self.node_map[out] = node_id

        # Insert in the node ordering for canonization
        self.node_ordering.append(node_id)
        self.node_ordering.sort()

    def hash(self):
        """
        Computes a deterministic hash of the ADG.
        Uses the node ordering and the sorted dependencies for consistency.
        """
        # Build a canonical string representation
        parts = []
        for node_id in self.node_ordering:
            deps = self.graph[node_id]
            # Represent node and dependencies as string
            node_str = f"{node_id}|{deps}"
            parts.append(node_str)
        
        # Concatenate all nodes into a single string
        full_str = ";".join(parts)

        # Compute 128-bit hash (MD5) for compactness
        return hashlib.md5(full_str.encode("utf-8")).hexdigest()
    
    def __hash__(self):
        return int(self.hash(), 16)
    
    def __eq__(self, other):
        return self.graph == other.graph
    
    def __le__(self, other):
        # TODO: test.
        return set(self.node_ordering) < set(other.node_ordering)

    def __str__(self):
        return f"{str(self.graph)}"
    
    def __repr__(self):
        return f"ActionDependencyGraph(" + f"{self.graph.keys()})"[11:]

    



############################################# Basic Func Seacher #############################################


from queue import PriorityQueue

class BasicFuncSearcher:
    def __init__(self, problem: list[list[CompObject], CompObject], known_functions: dict[str, Function], ranker):
        self.input = problem[0]
        self.target_output = problem[1]
        self.comp_env: SimpleCompEnv = SimpleCompEnv()
        for n, f in known_functions.items():
            self.comp_env.add_function(n, f)
        for inp in self.input:
            inp_id = self.comp_env.add_object(inp)
        self.rank = ranker
        self.search_queue = PriorityQueue()  # Priority queue for BFS with ranking
        self.visited = set()  # Track visited state_keys
        self.search_completed = False
        self.solution_obj_id = None
        self.object_depths = {}  # Track the depth of each object
        self.values_considered = set()

        for inp in self.input:
            self.object_depths[inp.id] = 0

    def get_objects_by_type(self):
        dic = {}
        for obj_id, obj in self.comp_env.objects.items():
            if obj.value_type in dic:
                dic[obj.value_type].append(obj_id)
            else:
                dic[obj.value_type] = [obj_id]
        return dic
    
    def get_possible_args(self, func: Function, max_depth: int):
        """
        Generate valid input combinations with strict type enforcement.
        Allows the same object to be used multiple times in a combination.
        For Function(append, [NumList, Num], [NumList]) it will return:
        [(numlist_id, num_id), (numlist_id, numlist_id), ...] where types match
        """
        # First get all candidate objects that meet depth requirement
        candidates = []
        for obj_id, obj in self.comp_env.objects.items():
            if self.object_depths.get(obj_id, 0) <= max_depth:
                candidates.append((obj_id, obj.value_type))
        
        # Then generate valid combinations
        possible_args = []
        
        def backtrack(position, current_combination):
            if position == len(func.input_types):
                possible_args.append(tuple(current_combination))
                return
            
            required_type = func.input_types[position]
            for obj_id, obj_type in candidates:
                if obj_type == required_type:  # Strict type equality check
                    backtrack(
                        position + 1,
                        current_combination + [obj_id]  # No object removal from candidates
                    )
        
        backtrack(0, [])
        return possible_args


    def search_step(self):
        if self.search_completed:
            return

        # Initialize queue with all possible actions at current depth
        if self.search_queue.empty():
            current_max_depth = max(self.object_depths.values(), default=0)
            self._populate_queue(max_depth=current_max_depth + 1)

        if self.search_queue.empty():
            self.search_completed = True
            print("Search completed, no solution found.")
            return
  
        # Process next item from queue
        depth, r, state_key = self.search_queue.get()
        func_name, *input_ids = state_key

        # Apply function and get new objects
        new_obj_ids = self.comp_env.apply_function(func_name, input_ids)
        
        # Update depths for new objects
        new_depth = depth + 1
        for new_obj_id in new_obj_ids:
            self.object_depths[new_obj_id] = new_depth

        # Check for solution
        if self._check_solution(new_obj_ids):
            return

        # Expand search with new objects
        self._expand_search(new_depth)

    def _populate_queue(self, max_depth):
        """Populate queue with all valid actions up to max_depth"""
        for fname, f in self.comp_env.known_functions.items():
            # Handle zero-argument functions
            if not f.input_types:
                self._queue_zero_arg_function(fname, f)
                continue
                
            # Handle functions with arguments
            possible_args = self.get_possible_args(f, max_depth)
            for args in possible_args:
                self._queue_function_call(fname, f, args)

    def _queue_zero_arg_function(self, fname, f):
        """Queue a zero-argument function call"""
        try:
            result = f.try_run([])
            r = self.rank(result, self.target_output.value)
            state_key = (fname,)
            if state_key not in self.visited:
                self.search_queue.put((1, r, state_key))
                self.visited.add(state_key)
        except Exception:
            pass

    def _queue_function_call(self, fname, f, args):
        """Queue a function call with arguments"""
        input_objs = [self.comp_env.objects[obj_id] for obj_id in args]
        try:
            result = f.try_run(input_objs)
            r = self.rank(result, self.target_output.value)
            state_key = (fname,) + tuple(args)
            if state_key not in self.visited:
                depth = max(self.object_depths.get(input_id, 0) for input_id in args)
                self.search_queue.put((depth, r, state_key))
                self.visited.add(state_key)
        except Exception:
            pass

    def _check_solution(self, new_obj_ids):
        """Check if any new objects match the target"""
        for new_obj_id in new_obj_ids:
            if self.comp_env.objects[new_obj_id].value == self.target_output.value:
                print(f"Solution found at object {new_obj_id}")
                self.solution_obj_id = new_obj_id
                self.search_completed = True
                return True
        return False

    def _expand_search(self, new_depth):
        """Expand search with new depth level"""
        self._populate_queue(max_depth=new_depth)

    def get_clean_comp_env(self):
        """
        Returns a new SimpleCompEnv containing only the steps that lead to the solution.
        Now works with the new action_history format.
        """
        if not self.solution_obj_id:
            raise ValueError("No solution found.")

        clean_env = SimpleCompEnv()
        clean_env.known_functions = self.comp_env.known_functions
        # Map original object IDs to new object IDs
        id_map = {}  # {original_id: new_id}

        # Add input objects to the clean environment and populate the ID map
        for inp in self.input:
            new_id = clean_env.add_input_object(inp)
            id_map[inp.id] = new_id
        
        # Reconstruct the solution path using the parent graph and action_history
        def reconstruct_path(obj_id):
            parent_info = self.comp_env.parent_graph[obj_id]
            if not parent_info:
                return  # Reached an input object

            # Find the corresponding action in action_history
            action = None
            for act in self.comp_env.action_history:
                if act[0] == parent_info[0] and tuple(act[1]) == parent_info[1:]:
                    action = act
                    break

            if not action:
                raise ValueError("Could not find action in history")

            func_name, input_ids, original_output_ids = action

            # Recursively reconstruct the path for input objects
            for input_id in input_ids:
                if input_id not in id_map:
                    reconstruct_path(input_id)

            # Map the input IDs to their new IDs
            new_input_ids = [id_map[input_id] for input_id in input_ids]

            # Apply the function in the clean environment
            new_output_ids = clean_env.apply_function(func_name, new_input_ids)

            # Update the ID map with the new output IDs
            for original_output_id, new_output_id in zip(original_output_ids, new_output_ids):
                id_map[original_output_id] = new_output_id

        # Reconstruct the path to the solution object
        reconstruct_path(self.solution_obj_id)

        return clean_env

    def get_solution_actions(self):
        """
        Returns the sequence of actions that led to the solution in human-readable format.
        Each action is (func_name, input_values, output_values).
        """
        if not self.solution_obj_id:
            raise ValueError("No solution found.")

        solution_actions = []
        obj_id = self.solution_obj_id

        def trace_back(obj_id):
            parent_info = self.comp_env.parent_graph[obj_id]
            if not parent_info:
                return

            # Find the action in action_history
            for action in reversed(self.comp_env.action_history):
                if action[0] == parent_info[0] and tuple(action[1]) == parent_info[1:]:
                    # Recursively trace back inputs first
                    for input_id in action[1]:
                        trace_back(input_id)
                    # Add this action to solution
                    input_values = [self.comp_env.objects[i].value for i in action[1]]
                    output_values = [self.comp_env.objects[o].value for o in action[2]]
                    solution_actions.append((action[0], input_values, output_values))
                    break

        trace_back(self.solution_obj_id)
        return solution_actions

############################################# Visualization #############################################    

from graphviz import Digraph

def visualize_history_graphviz(env: SimpleCompEnv, filename='comp_history'):
    """
    Visualizes the computation history of the SimpleCompEnv using Graphviz.
    Functions are represented as circles, and objects as squares.
    Now works with the new action_history format (list of (func_name, input_ids, output_ids) tuples).
    """
    # Create a directed graph
    dot = Digraph(comment="Computation History",filename=f'images/{filename}')
    dot.attr(rankdir='TB')  # Left-to-right graph layout
    
    # Track which objects and functions have been added to the graph
    added_nodes = set()
    
    # Iterate through the action history
    for time, action in enumerate(env.action_history):
        func_name, input_ids, output_ids = action
        
        # Add the function node (circle)
        func_node_id = f"F_{time}_{func_name}"
        if func_node_id not in added_nodes:
            dot.node(func_node_id, func_name, 
                    shape="circle", 
                    style="filled", 
                    fillcolor="lightblue",
                    fontname="helvetica")
            added_nodes.add(func_node_id)
        
        # Add input object nodes (squares)
        for obj_id in input_ids:
            obj_node_id = f"O_{obj_id}"
            if obj_node_id not in added_nodes:
                obj_value = str(env.objects[obj_id].value)
                dot.node(obj_node_id, obj_value, 
                        shape="box", 
                        style="filled", 
                        fillcolor="lightgreen",
                        fontname="helvetica")
                added_nodes.add(obj_node_id)
            # Connect input object to function
            dot.edge(obj_node_id, func_node_id,
                    arrowsize="0.7",
                    penwidth="1.5")
        
        # Add output object nodes (squares)
        for obj_id in output_ids:
            obj_node_id = f"O_{obj_id}"
            if obj_node_id not in added_nodes:
                obj_value = str(env.objects[obj_id].value)
                dot.node(obj_node_id, obj_value, 
                        shape="box", 
                        style="filled", 
                        fillcolor="orange",
                        fontname="helvetica")
                added_nodes.add(obj_node_id)
            # Connect function to output object
            dot.edge(func_node_id, obj_node_id,
                    arrowsize="0.7",
                    penwidth="1.5")
        
    # paint solution object red
    s_obj_id = env.solution_object_id
    if s_obj_id != None:
        obj_value = str(env.objects[s_obj_id].value)
        dot.node(f"O_{s_obj_id}", obj_value, 
            shape="box", 
            style="filled", 
            fillcolor="red",
            fontname="helvetica")

    
    # Add legend
    with dot.subgraph(name='cluster_legend') as legend:
        legend.attr(label='Legend', style='rounded', color='gray')
        legend.node('L_func', 'Function', shape="circle", style="filled", fillcolor="lightblue")
        legend.node('L_input', 'Input', shape="box", style="filled", fillcolor="lightgreen")
        legend.node('L_output', 'Intermidiate Object', shape="box", style="filled", fillcolor="orange")
        legend.node('L_solution', 'Return Object', shape="box", style="filled", fillcolor="red")
        legend.attr(rank='same')
    
    # Render and display the graph
    dot.render(filename, format="png", cleanup=True)
    print("Flowchart saved as 'computation_history.png'")
    return dot

def visualize_adg(adg: ActionDependencyGraph, target_nodes=None):
    """
    Visualizes the Action Dependency Graph (ADG).
    Nodes are actions (function_name + input_ids), edges represent dependencies.

    Args:
        adg: ActionDependencyGraph instance.
        target_nodes: Optional set of action nodes to highlight (e.g., producing the final output).
    """
    dot = Digraph(comment="Action Dependency Graph", filename='../images/adg')
    dot.attr(rankdir='TB', splines='true')

    target_nodes = target_nodes or set()
    added_nodes = set()

    # Add all nodes
    for node in adg.graph:
        func_name, input_ids, output_ids = node
        label = f"{func_name}\n{tuple(n % 10000 for n in input_ids)}\n{tuple(n % 10000 for n in output_ids)}"
        color = "red" if node in target_nodes else "lightblue"
        shape = "box"
        dot.node(str(node), label, shape=shape, style="filled", fillcolor=color, fontname="helvetica")
        added_nodes.add(node)

    # Add edges (dependencies)
    for node, deps in adg.graph.items():
        for dep in deps:
            dot.edge(str(dep), str(node), arrowsize="0.7", penwidth="1.2")

    # Optional: legend
    with dot.subgraph(name='cluster_legend') as legend:
        legend.attr(label='Legend', style='rounded', color='gray')
        legend.node('L_action', 'Action Node', shape='box', style='filled', fillcolor='lightblue')
        legend.node('L_target', 'Target Node', shape='box', style='filled', fillcolor='red')
        legend.attr(rank='same')

    dot.render("adg_graph", format="png", cleanup=True)
    print("ADG graph saved as 'adg_graph.png'")
    return dot
