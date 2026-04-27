from dataclasses import dataclass

from core_lang_env.comp_env import *   # SimpleCompEnv, ObjId, ...
from core_lang_env.syntax_tree import *
from core_lang_env.exec_code_v2 import *  # ExecutionPosition, ExecutionPositionTuple, ASTNodePosition

from searchers_utils import *
from typing import Literal, Callable


#################### Augmentation Stack Rules ############################

AugType = str

AUG_RETURN:   AugType = "AUG_RETURN"
AUG_FUNC_CALL: AugType = "AUG_FUNC_CALL"
AUG_START_IF: AugType = "AUG_START_IF"
AUG_END_IF:   AugType = "AUG_END_IF"
AUG_END_ELSE: AugType = "AUG_END_ELSE"


# --- helpers to convert between tuple and object ---

def _tuple_to_exec_position(ep_tuple: ExecutionPositionTuple) -> ExecutionPosition:
    ast_pos, temporal_stack = ep_tuple
    return ExecutionPosition(list(ast_pos), list(temporal_stack))


def _exec_position_to_tuple(ep: ExecutionPosition) -> ExecutionPositionTuple:
    return ep.to_tuple()


######################### Construction Frontier ###########################

ConstructionFrontier = tuple[ExecutionPositionTuple, set[AugType], set[int]]

def make_frontier(exec_pos: ExecutionPositionTuple, aug_options: set[AugType], parent_indices: set[int], group_indices: set[int]) -> ConstructionFrontier:
    return (exec_pos, set(aug_options), set(parent_indices), set(group_indices))


######################### Transition Functions ############################
# NOTE:
#   * frontiers store a *static* ExecutionPositionTuple, but transitions
#     convert to ExecutionPosition, mutate with next_line / enter_context,
#     and convert back.
#   * This keeps a consistent temporal_stack evolution instead of
#     resetting everything to zeros.


def transition_return_option(construction_frontier: ConstructionFrontier, choice: AugType, choice_indices: set[int]):
    assert choice == AUG_RETURN
    _, aug_options, _ , frontier_indices = construction_frontier
    assert choice in aug_options
    assert set(choice_indices) == set(frontier_indices)

    # Conceptually: terminal; no more indices to handle.
    # We return a dummy frontier with empty indices so it will not be pushed.
    ep = ExecutionPosition.start_position()
    dummy_exec_pos = ep.to_tuple()
    new_frontier = make_frontier(dummy_exec_pos, set(), set(), set())
    return (new_frontier,)


def transition_func_call_option(construction_frontier: ConstructionFrontier, choice: AugType, choice_indices: set[int]):
    assert choice == AUG_FUNC_CALL
    exec_pos_tuple, aug_options, frontier_indices = construction_frontier
    assert choice in aug_options
    assert set(choice_indices) == set(frontier_indices)

    # Move structurally to "next line" in the same block.
    ep = _tuple_to_exec_position(exec_pos_tuple)
    ep.next_line()
    new_exec_pos = _exec_position_to_tuple(ep)

    new_frontier = make_frontier(new_exec_pos, aug_options.copy(), frontier_indices)
    return (new_frontier,)


def transition_start_if_option(construction_frontier: ConstructionFrontier, choice: AugType, choice_indices_if: set[int]):
    assert choice == AUG_START_IF
    exec_pos_tuple, aug_options, frontier_indices = construction_frontier
    assert choice in aug_options
    assert set(choice_indices_if) <= set(frontier_indices)

    # Current position is where the IfElse node is inserted / executed.
    ep_current = _tuple_to_exec_position(exec_pos_tuple)

    # --- if-body frontier ---
    ep_if = ep_current.copy()
    # structurally: enter the if/else context, then go to the "if" block
    ep_if.enter_context()
    ep_if.next_line()
    if_exec_pos = _exec_position_to_tuple(ep_if)
    if_aug_options = {AUG_FUNC_CALL, AUG_START_IF, AUG_END_IF}
    if_frontier = make_frontier(if_exec_pos, if_aug_options, set(choice_indices_if))

    # --- else-body frontier ---
    choice_indices_else = frontier_indices.difference(choice_indices_if)
    ep_else = ep_current.copy()
    ep_else.enter_context()
    # canonical: first next_line() would be "if" block, second next_line() -> "else" block
    ep_else.next_line()
    ep_else.next_line()
    else_exec_pos = _exec_position_to_tuple(ep_else)
    else_aug_options = {AUG_FUNC_CALL, AUG_START_IF, AUG_END_ELSE}
    else_frontier = make_frontier(else_exec_pos, else_aug_options, set(choice_indices_else))

    # --- parent frontier (line after the IfElse) ---
    ep_parent = ep_current.copy()
    ep_parent.next_line()
    parent_exec_pos = _exec_position_to_tuple(ep_parent)
    parent_frontier = make_frontier(parent_exec_pos, aug_options.copy(), set(frontier_indices))

    return (parent_frontier, else_frontier, if_frontier)


def transition_end_if_option(construction_frontier: ConstructionFrontier, choice: AugType, choice_indices: set[int]):
    assert choice == AUG_END_IF
    _, aug_options, frontier_indices = construction_frontier
    assert choice in aug_options
    assert set(choice_indices) == set(frontier_indices)

    # End-if itself does not introduce a new frontier; parent frontier
    # (created at START_IF) is still on the stack.
    return ()


def transition_end_else_option(construction_frontier: ConstructionFrontier, choice: AugType, choice_indices: set[int]):
    assert choice == AUG_END_ELSE
    _, aug_options, frontier_indices = construction_frontier
    assert choice in aug_options
    assert set(choice_indices) == set(frontier_indices)

    # After ending else, we also do not add a new frontier; we rely on the
    # already-existing parent frontier that START_IF pushed.
    return ()


TRANSITION_FUNCTIONS = {
    AUG_RETURN:   transition_return_option,
    AUG_FUNC_CALL: transition_func_call_option,
    AUG_START_IF: transition_start_if_option,
    AUG_END_IF:   transition_end_if_option,
    AUG_END_ELSE: transition_end_else_option,
}


############################ AugmentationStack ############################

class AugmentationStack:
    def __init__(self, stack: list[ConstructionFrontier], all_indices: set[int]):
        # list of (ExecutionPositionTuple, aug_options, indices)
        self.stack = stack
        self.all_indices = set(all_indices)

        # Track if–else relationships per structural parent position
        # key: (parent_ast_pos, frozenset(parent_indices))
        # val: (if_indices, else_indices)
        self.if_else_links: dict[tuple[ASTNodePosition, frozenset[int]], tuple[set[int], set[int]]] = {}

    @staticmethod
    def init_new_stack(all_indices: set[int]) -> "AugmentationStack":
        """
        Initialize a new stack with a single root frontier.

        Canonical choice:
        - start from ExecutionPosition.start_position()
        - enter the top-level BlockNode context (root block)
        which gives a position for the *first* top-level statement.
        """
        ep = ExecutionPosition.start_position()
        ep.enter_context()                      # go into root BlockNode
        root_exec_pos = ep.to_tuple()           # e.g. ast=[0], temporal=[1,0]

        root_frontier = make_frontier(
            root_exec_pos,
            {AUG_FUNC_CALL, AUG_START_IF, AUG_RETURN},
            set(all_indices),
        )
        return AugmentationStack([root_frontier], all_indices)

    def peek(self) -> ConstructionFrontier | None:
        """Return the top frontier without removing it."""
        return self.stack[-1] if self.stack else None

    def pop(self) -> ConstructionFrontier | None:
        """Pop the top frontier (if exists)."""
        return self.stack.pop() if self.stack else None

    def push(self, *new_frontiers: ConstructionFrontier):
        """Push one or more new frontiers on top of the stack."""
        for frontier in new_frontiers:
            if frontier and frontier[2]:  # skip empty or no-index frontiers
                self.stack.append(frontier)

    def apply_transition(self, choice: AugType, choice_indices: set[int]):
        current = self.pop()
        if current is None:
            raise RuntimeError("Stack underflow: no frontier to apply transition on")

        new_frontiers = TRANSITION_FUNCTIONS[choice](current, choice, choice_indices)

        # record mapping for START_IF transitions
        if choice == AUG_START_IF:
            parent_exec_pos, _, parent_indices = current
            _, else_frontier, if_frontier = new_frontiers

            parent_ast_pos = parent_exec_pos[0]  # ast_position part
            key = (tuple(parent_ast_pos), frozenset(parent_indices))
            self.if_else_links[key] = (if_frontier[2], else_frontier[2])

        self.push(*new_frontiers)
        return new_frontiers

    def get_if_else_indices(self, parent_ast_pos: ASTNodePosition, parent_indices: set[int]) -> tuple[set[int], set[int]]:
        """
        Retrieve (if_indices, else_indices) given the structural parent location
        (the AST position of the IfElse node) and its index set.
        """
        key = (tuple(parent_ast_pos), frozenset(parent_indices))
        if key not in self.if_else_links:
            raise KeyError(f"No recorded if–else mapping for {key}")
        return self.if_else_links[key]

    def copy(self) -> "AugmentationStack":
        new_stack = AugmentationStack(self.stack.copy(), self.all_indices.copy())
        new_stack.if_else_links = self.if_else_links.copy()
        return new_stack

    def __repr__(self):
        return f"AugmentationStack({str(self.stack[-1:])[:-1]}, ...], indices={self.all_indices})"


###################### Annotated AST group #######################

NO_ACTION = "NO_ACTION"
RETURN_FUNC_NAME = "RETURN_FUNC_NAME"

TBD_CONDITIONAL = "TBD_CONDITIONAL"


class AnnotatedAstGroup:
    def __init__(self, annotated_asts: list[AnnotatedAST]):
        self.ast = annotated_asts[0].ast
        self.initial_vars = annotated_asts[0].initial_vars
        self.annot_asts = annotated_asts

    @staticmethod
    def initialize_new_group(initial_vars_states: list[dict[str, ObjId]]):
        ast = BlockNode([])
        return AnnotatedAstGroup(
            [AnnotatedAST.create_new_annotated_ast(ast, initial_vars)
             for initial_vars in initial_vars_states]
        )

    def __eq__(self, other: "AnnotatedAstGroup"):
        return all(
            annot == other_annot
            for annot, other_annot in zip(self.annot_asts, other.annot_asts)
        )

    def copy(self):
        return AnnotatedAstGroup([a.copy() for a in self.annot_asts])

    def __repr__(self):
        return f"AnnotatedAstGroup(ast={self.ast}, annotated_asts={self.annot_asts})"


###################### Tool functions ##############################

def get_all_variables_in_ast(ast: ASTNode, initial_vars: set[str]):
    variables = set(initial_vars)
    ast = ast

    def traverse(node: ASTNode):
        for child in node.children:
            if isinstance(child, DirectAssignNode):
                variables.update(
                    var for var in child.target_vars if var[0] == 'x'
                )
            if isinstance(child, FunctionCallAssignNode):
                variables.update(
                    var for var in child.var_names if var[0] == 'x'
                )
            if type(child) in [BlockNode, WhileNode, IfElseNode]:
                traverse(child)

    traverse(ast)
    return variables


def get_largest_label_of_variables(ast: BlockNode, initial_vars: set[str]):
    """If ast contains x0, x1, x2, <str>, returns 2 (int)."""
    variables = get_all_variables_in_ast(ast, initial_vars)
    largest_label = max(int(x[1:]) for x in variables)
    return largest_label


def get_semantically_available_variables(ast: BlockNode,
                                         initial_vars: set[str],
                                         position: ASTNodePosition):
    """TODO: Implement"""
    # placeholder
    return initial_vars.copy()


####################### ---- Augmentation Functions ---- #######################

# NOTE: the form of an action group for augmentation is
# (action_by_var, (actions_by_ids...), output_len)
# e.g. (('get_tail', ('x0',)),
#       (('get_tail', (0,)), ('get_tail', (0,)), 'NO_ACTION'),
#       1)

DetailedActionGroup = tuple[ShortActionByVarName, tuple[ShortAction, ...], int]


###################### Return Augmentation ########################

def augment_ast_return(ast: BlockNode, insertion_pos: ASTNodePosition, return_vars: tuple[str]) -> BlockNode:
    """Add a final ReturnNode returning the first variable in return_vars."""
    new_ast = ast.copy()
    new_node = ReturnNode(return_vars[0])
    new_ast.insert_node(insertion_pos, new_node)
    return new_ast


def apply_augmentation_no_action(annotated_ast: AnnotatedAST, new_ast: BlockNode, exec_position: ExecutionPositionTuple, no_action: ShortAction) -> AnnotatedAST:
    assert no_action == NO_ACTION
    return AnnotatedAST(
        new_ast,
        annotated_ast.signature.copy(),
        annotated_ast.mapping.copy(),
        annotated_ast.initial_vars.copy()
    )


def apply_augmentation_return(annotated_ast: AnnotatedAST, new_ast: BlockNode, exec_position: ExecutionPositionTuple, return_action: ShortAction) -> AnnotatedAST:
    assert return_action[0] == RETURN_FUNC_NAME
    new_annot_ast = AnnotatedAST(new_ast, annotated_ast.signature.copy(), annotated_ast.mapping.copy(), annotated_ast.initial_vars.copy())
    new_annot_ast.add_action(return_action, exec_position)
    return new_annot_ast


def apply_augmentation_return_group(annotated_ast_group: AnnotatedAstGroup, exec_position: ExecutionPositionTuple,detailed_action_group: DetailedActionGroup) -> AnnotatedAstGroup:
    return_var_action, short_actions, output_len = detailed_action_group
    new_ast = augment_ast_return(annotated_ast_group.ast, exec_position[0], return_var_action[1])
    augmented_asts = []
    for annot_ast, short_action in zip(annotated_ast_group.annot_asts, short_actions):
        if short_action == NO_ACTION:
            new_annot = apply_augmentation_no_action(annot_ast, new_ast, exec_position, short_action)
        else:
            new_annot = apply_augmentation_return(annot_ast, new_ast, exec_position, short_action)
        augmented_asts.append(new_annot)

    return AnnotatedAstGroup(augmented_asts)


###################### augment AST function call  ###########################

def augment_ast_function_call(ast: BlockNode, initial_vars: set[str], insertion_pos: ASTNodePosition, variable_action: ShortActionByVarName, output_length: int) -> BlockNode:
    func_name, input_vars = variable_action

    largest_label = get_largest_label_of_variables(ast, initial_vars)
    output_vars = tuple(f'x{i}' for i in range(largest_label + 1, largest_label + 1 + output_length))

    new_node = FunctionCallAssignNode(output_vars, func_name, input_vars)
    new_ast: BlockNode = ast.copy()
    new_ast.insert_node(insertion_pos, new_node)

    return new_ast


def apply_augmentation_function_call(annotated_ast: AnnotatedAST,
                                     new_ast: ASTNode,
                                     exec_position: ExecutionPositionTuple,
                                     short_action: ShortAction) -> AnnotatedAST:
    new_annot_ast = AnnotatedAST(
        new_ast,
        annotated_ast.signature.copy(),
        annotated_ast.mapping.copy(),
        annotated_ast.initial_vars.copy()
    )
    new_annot_ast.add_action(short_action, exec_position)
    return new_annot_ast


def apply_augmentation_function_call_group(annotated_ast_group: AnnotatedAstGroup,
                                           exec_position: ExecutionPositionTuple,
                                           detailed_action_group: DetailedActionGroup
                                           ) -> AnnotatedAstGroup:
    var_action, short_actions, output_len = detailed_action_group
    new_ast = augment_ast_function_call(
        annotated_ast_group.ast,
        set(annotated_ast_group.initial_vars),
        exec_position[0],
        var_action,
        output_len
    )
    augmented_asts = []
    for annot_ast, short_action in zip(annotated_ast_group.annot_asts, short_actions):
        if short_action == NO_ACTION:
            new_annot = apply_augmentation_no_action(
                annot_ast, new_ast, exec_position, short_action
            )
        else:
            new_annot = apply_augmentation_function_call(
                annot_ast, new_ast, exec_position, short_action
            )
        augmented_asts.append(new_annot)

    return AnnotatedAstGroup(augmented_asts)


###################### augment AST Start If, End If, End Else  ###########################

def augment_ast_start_if(ast: BlockNode, insertion_pos: ASTNodePosition) -> BlockNode:
    new_if_else_node = IfElseNode(BoolExprNode(TBD_CONDITIONAL, []), BlockNode([]), BlockNode([]))
    new_ast = ast.copy()
    new_ast.insert_node(insertion_pos, new_if_else_node)
    return new_ast


def augment_ast_end_if(ast: BlockNode, insertion_pos: ASTNodePosition, target_vars: tuple[str], source_vars: tuple[str]) -> BlockNode:
    assert insertion_pos[-2] == 1  # quick check that it's the if block

    new_final_assign_node = DirectAssignNode(target_vars, source_vars)
    new_ast = ast.copy()
    new_ast.insert_node(insertion_pos, new_final_assign_node)
    return new_ast


def augment_ast_end_if_simple(ast: BlockNode) -> BlockNode:
    # For now, end-if does not add assignments; just returns a copy.
    return ast.copy()


def augment_ast_end_else(ast: BlockNode, insertion_pos: ASTNodePosition, target_vars: tuple[str], source_vars: tuple[str]) -> BlockNode:
    assert insertion_pos[-2] == 2  # quick check that it's the else block

    new_final_assign_node = DirectAssignNode(target_vars, source_vars)
    new_ast = ast.copy()
    new_ast.insert_node(insertion_pos, new_final_assign_node)
    return new_ast


def apply_augmentation_if_else_start_end(annotated_ast: AnnotatedAST, new_ast: BlockNode) -> AnnotatedAST:
    return AnnotatedAST(new_ast, annotated_ast.signature.copy(), annotated_ast.mapping.copy(), annotated_ast.initial_vars.copy())


def apply_augmentation_start_if_group(annotated_ast_group: AnnotatedAstGroup, exec_position: ExecutionPositionTuple) -> AnnotatedAstGroup:
    new_ast = augment_ast_start_if(annotated_ast_group.ast, exec_position[0])
    augmented_asts = [apply_augmentation_if_else_start_end(annot_ast, new_ast) for annot_ast in annotated_ast_group.annot_asts]
    return AnnotatedAstGroup(augmented_asts)


def apply_augmentation_end_if_group(annotated_ast_group: AnnotatedAstGroup) -> AnnotatedAstGroup:
    new_ast = augment_ast_end_if_simple(annotated_ast_group.ast)
    augmented_asts = [apply_augmentation_if_else_start_end(annot_ast, new_ast) for annot_ast in annotated_ast_group.annot_asts]
    return AnnotatedAstGroup(augmented_asts)


def apply_augmentation_end_else_group(annotated_ast_group: AnnotatedAstGroup, exec_position: ExecutionPositionTuple, target_vars: tuple[str], source_vars: tuple[str]) -> AnnotatedAstGroup:
    new_ast = augment_ast_end_else(annotated_ast_group.ast, exec_position[0], target_vars, source_vars)
    augmented_asts = [apply_augmentation_if_else_start_end(annot_ast, new_ast) for annot_ast in annotated_ast_group.annot_asts]
    return AnnotatedAstGroup(augmented_asts)
