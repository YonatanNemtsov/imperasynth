from dataclasses import dataclass

from core_lang_env.comp_env import *
from core_lang_env.syntax_tree import *
from core_lang_env.exec_code_v2 import *

from searchers_utils import *
from typing import Literal, Callable

#################### Augmentation Stack Rules ############################

AugType = str

AUG_RETURN: AugType = "AUG_RETURN"
AUG_FUNC_CALL: AugType = "AUG_FUNC_CALL"
AUG_START_IF: AugType = "AUG_START_IF"
AUG_END_IF: AugType = "AUG_END_IF"
AUG_END_ELSE: AugType = "AUG_END_ELSE"



ConstructionFrontier = tuple[ASTNodePosition, set[AugType], set[int]]
def make_frontier(insert_pos, aug_options, indices) -> ConstructionFrontier:
    return (tuple(insert_pos), set(aug_options), set(indices))

# options = (insetion_pos, aug_options, indecies)
def transition_return_option(construction_frontier: ConstructionFrontier, choice: AugType, choice_indices: set[int]):
    assert choice == AUG_RETURN
    assert choice in construction_frontier[1]
    assert set(choice_indices) == set(construction_frontier[2])

    new_frontier = make_frontier((), set(), set())
    return (new_frontier,)

def transition_func_call_option(construction_frontier: ConstructionFrontier, choice: AugType, choice_indices: set[int]):
    assert choice == AUG_FUNC_CALL
    assert choice in construction_frontier[1]
    assert set(choice_indices) == set(construction_frontier[2])

    insert_pos, aug_options, indices = construction_frontier
    new_insert_pos = insert_pos[:-1] + (insert_pos[-1] + 1,)
    new_frontier = make_frontier(new_insert_pos, aug_options.copy(), indices)
    return (new_frontier,)

def transition_start_if_option(construction_frontier: ConstructionFrontier, choice: AugType, choice_indices_if: set[int]):
    assert choice == AUG_START_IF
    assert choice in construction_frontier[1]
    assert set(choice_indices_if) <= set(construction_frontier[2])

    insert_pos, aug_options, indices = construction_frontier

    # inside the 'if' body
    if_insert_pos = insert_pos + (1, 0)
    if_aug_options = {AUG_FUNC_CALL, AUG_START_IF, AUG_END_IF}
    if_frontier = make_frontier(if_insert_pos, if_aug_options, set(choice_indices_if))

    # inside the 'else' body
    choice_indices_else = indices.difference(choice_indices_if)
    else_insert_pos = insert_pos + (2, 0)
    else_aug_options = {AUG_FUNC_CALL, AUG_START_IF, AUG_END_ELSE}
    else_frontier = make_frontier(else_insert_pos, else_aug_options, set(choice_indices_else))

    # the parent (where the if was inserted)
    parent_insert_pos = insert_pos[:-1] + (insert_pos[-1] + 1,)
    parent_frontier = make_frontier(parent_insert_pos, aug_options.copy(), set(indices))

    return (parent_frontier, else_frontier, if_frontier)

def transition_end_if_option(construction_frontier: ConstructionFrontier, choice: AugType, choice_indices: set[int]):
    assert choice == AUG_END_IF
    assert choice in construction_frontier[1]
    assert set(choice_indices) == set(construction_frontier[2])

    return ()

def transition_end_else_option(construction_frontier: ConstructionFrontier, choice: AugType, choice_indices: set[int]):
    assert choice == AUG_END_ELSE
    assert choice in construction_frontier[1]
    assert set(choice_indices) == set(construction_frontier[2])

    return ()


TRANSITION_FUNCTIONS = {
    AUG_RETURN: transition_return_option,
    AUG_FUNC_CALL: transition_func_call_option,
    AUG_START_IF: transition_start_if_option,
    AUG_END_IF: transition_end_if_option,
    AUG_END_ELSE: transition_end_else_option
}

class AugmentationStack:
    def __init__(self, stack: list[tuple], all_indices: set[int]):
        self.stack = stack                # list of (insert_pos, aug_options, indices)
        self.all_indices = set(all_indices)

        # Track if–else relationships per structural position
        # key: (insert_pos of parent, frozenset(parent_indices))
        # val: (if_indices, else_indices)
        self.if_else_links: dict[tuple[ASTNodePosition, frozenset[int]], tuple[set[int], set[int]]] = {}

    @staticmethod
    def init_new_stack(all_indices):
        """Initialize a new stack with a single root frontier."""
        root_frontier = ((0,), {AUG_FUNC_CALL, AUG_START_IF, AUG_RETURN}, set(all_indices))
        return AugmentationStack([root_frontier], all_indices)
    
    def peek(self):
        """Return the top frontier without removing it."""
        return self.stack[-1] if self.stack else None

    def pop(self):
        """Pop the top frontier (if exists)."""
        return self.stack.pop() if self.stack else None

    def push(self, *new_frontiers):
        """Push one or more new frontiers on top of the stack."""
        for frontier in new_frontiers:
            if frontier and frontier[2]:  # skip empty or no-index frontiers
                self.stack.append(frontier)
    
    def apply_transition(self, choice, choice_indices):
        current = self.pop()
        if current is None:
            raise RuntimeError("Stack underflow: no frontier to apply transition on")

        new_frontiers = TRANSITION_FUNCTIONS[choice](current, choice, choice_indices)

        # --- record mapping for START_IF transitions ---
        if choice == AUG_START_IF:
            parent_pos, _, parent_indices = current
            _, else_frontier, if_frontier = new_frontiers
            key = (tuple(parent_pos), frozenset(parent_indices))
            self.if_else_links[key] = (if_frontier[2], else_frontier[2])

        self.push(*new_frontiers)
        return new_frontiers

    def get_if_else_indices(self, insert_pos: tuple[int, ...], indices: set[int]) -> tuple[set[int], set[int]]:
        """Retrieve (if_indices, else_indices) given a structural location and index set."""
        key = (tuple(insert_pos), frozenset(indices))
        if key not in self.if_else_links:
            raise KeyError(f"No recorded if–else mapping for {key}")
        return self.if_else_links[key]

    def copy(self):
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
        return AnnotatedAstGroup([AnnotatedAST.create_new_annotated_ast(ast, initial_vars) for initial_vars in initial_vars_states])
    
    def __eq__(self, other: "AnnotatedAstGroup"):
        return all(annot == other_annot for annot, other_annot in zip(self.annot_asts, other.annot_asts))
    
    def copy(self):
        return AnnotatedAstGroup([a.copy() for a in self.annot_asts])
    
    def __repr__(self):
        return f"AnnotatedAstGroup(ast={self.ast}, annotated_asts={self.annot_asts})"

###################### Tool functions ##############################

def get_all_variables_in_ast(ast: ASTNode, initial_vars: set):
    variables = set(initial_vars)
    ast = ast
    
    def traverse(node: ASTNode):
        for child in node.children:
            if isinstance(child, DirectAssignNode):
                variables.update(set(var for var in child.target_vars if var[0] == 'x'))
            if isinstance(child, FunctionCallAssignNode):
                variables.update(set(var for var in child.var_names if var[0] == 'x'))
            if type(child) in [BlockNode, WhileNode, IfElseNode]:
                traverse(child)
    traverse(ast)
    return variables

def get_largest_label_of_variables(ast: BlockNode, initial_vars: set):
    """If ast contains the following variables: x0, x1, x2, <str>, returns 2 <int>. """
    variables = get_all_variables_in_ast(ast, initial_vars)
    largest_label = max(int(x[1:]) for x in variables)
    return largest_label

def get_semantically_available_variables(ast: BlockNode, initial_vars: set, position: ASTNodePosition):
    """TODO: Implement"""

####################### ---- Augmentation Functions ---- #######################

# NOTE: the form of an action group for augmentation is (action_by_var, (actions_by_ids...)) 
# for example:  (('get_tail', ('x0',)), (('get_tail', (0,)), ('get_tail', (0,)), ('get_tail', (0,)), 'NO_ACTION')),

DetailedActionGroup = tuple[ShortActionByVarName, tuple[ShortAction], int]

###################### Return Augmentation ########################

def augment_ast_return(ast: BlockNode, insertion_pos: ASTNodePosition, return_vars: tuple[str]) -> BlockNode:
    """Add a final ReturnNode returning the first variable in return_vars."""
    new_ast = ast.copy()
    new_node = ReturnNode(return_vars[0])
    new_ast.insert_node(insertion_pos, new_node)
    return new_ast

def apply_augmentation_no_action(annotated_ast: AnnotatedAST, new_ast: BlockNode, exec_position: ExecutionPositionTuple, no_action: ShortAction) -> AnnotatedAST:
    assert no_action == NO_ACTION

    new_annot_ast = AnnotatedAST(new_ast, annotated_ast.signature.copy(), annotated_ast.mapping.copy(), annotated_ast.initial_vars.copy())
    return new_annot_ast

def apply_augmentation_return(annotated_ast: AnnotatedAST, new_ast: BlockNode, exec_position: ExecutionPositionTuple, return_action: ShortAction) -> AnnotatedAST:
    assert return_action[0] == RETURN_FUNC_NAME

    new_annot_ast = AnnotatedAST(new_ast, annotated_ast.signature.copy(), annotated_ast.mapping.copy(), annotated_ast.initial_vars.copy())
    new_annot_ast.add_action(return_action, exec_position)
    return new_annot_ast

def apply_augmentation_return_group(annotated_ast_group: AnnotatedAstGroup, exec_position: ExecutionPositionTuple, detailed_action_group: DetailedActionGroup) -> AnnotatedAstGroup:
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

def augment_ast_function_call(ast: BlockNode, initial_vars: set, insertion_pos: ASTNodePosition, variable_action: ShortActionByVarName, output_length: int):
    func_name, input_vars = variable_action

    largest_label = get_largest_label_of_variables(ast, initial_vars)
    output_vars = tuple(f'x{i}' for i in range(largest_label + 1, largest_label + 1 + output_length))
    
    new_node = FunctionCallAssignNode(output_vars, func_name, input_vars)
    new_ast: BlockNode = ast.copy()
    new_ast.insert_node(insertion_pos, new_node)

    return new_ast

def apply_augmentation_function_call(annotated_ast: AnnotatedAST, new_ast: ASTNode, exec_position: ExecutionPositionTuple, short_action: ShortAction) -> AnnotatedAST:
    new_annot_ast = AnnotatedAST(new_ast, annotated_ast.signature.copy(), annotated_ast.mapping.copy(), annotated_ast.initial_vars.copy())
    new_annot_ast.add_action(short_action, exec_position)
    return new_annot_ast

def apply_augmentation_function_call_group(annotated_ast_group: AnnotatedAstGroup, exec_position: ExecutionPositionTuple, detailed_action_group: DetailedActionGroup) -> AnnotatedAstGroup:
    var_action, short_actions, output_len = detailed_action_group
    new_ast = augment_ast_function_call(annotated_ast_group.ast, set(annotated_ast_group.initial_vars), exec_position[0], var_action, output_len)
    augmented_asts = []
    for annot_ast, short_action in zip(annotated_ast_group.annot_asts, short_actions):
        if short_action == NO_ACTION:
            new_annot = apply_augmentation_no_action(annot_ast, new_ast, exec_position, short_action)
        else:
            new_annot = apply_augmentation_function_call(annot_ast, new_ast, exec_position, short_action)
        augmented_asts.append(new_annot)

    return AnnotatedAstGroup(augmented_asts)



###################### augment AST Start If, End If, End Else  ###########################

def augment_ast_start_if(ast: BlockNode, insertion_pos: ASTNodePosition):
    new_if_else_node = IfElseNode(BoolExprNode(TBD_CONDITIONAL, []), BlockNode([]), BlockNode([]))
    new_ast = ast.copy()
    new_ast.insert_node(insertion_pos, new_if_else_node)
    return new_ast

def augment_ast_end_if(ast: BlockNode, insertion_pos: ASTNodePosition, target_vars: tuple[str], source_vars: tuple[str]):
    assert insertion_pos[-2] == 1 # quick check that its the if block

    new_final_assign_node = DirectAssignNode(target_vars, source_vars)
    new_ast = ast.copy()
    new_ast.insert_node(insertion_pos, new_final_assign_node)
    return new_ast

def augment_ast_end_if_simple(ast: BlockNode):
    return ast.copy()


def augment_ast_end_else(ast: BlockNode, insertion_pos: ASTNodePosition, target_vars: tuple[str], source_vars: tuple[str]):
    assert insertion_pos[-2] == 2 # quick check that its the else block
    
    new_final_assign_node = DirectAssignNode(target_vars, source_vars)
    new_ast = ast.copy()
    new_ast.insert_node(insertion_pos, new_final_assign_node)
    return new_ast


def apply_augmentation_if_else_start_end(annotated_ast: AnnotatedAST, new_ast: BlockNode):
    return AnnotatedAST(new_ast, annotated_ast.signature.copy(), annotated_ast.mapping.copy(), annotated_ast.initial_vars.copy())


def apply_augmentation_start_if_group(annotated_ast_group: AnnotatedAstGroup, exec_position: ExecutionPositionTuple) -> AnnotatedAstGroup:
    new_ast = augment_ast_start_if(annotated_ast_group.ast, exec_position[0])
    augmented_asts = [apply_augmentation_if_else_start_end(annot_ast, new_ast) for annot_ast in annotated_ast_group.annot_asts]
    return AnnotatedAstGroup(augmented_asts)

def apply_augmentation_end_if_group(annotated_ast_group: AnnotatedAstGroup) -> AnnotatedAstGroup:
    # new_ast = augment_ast_end_if(annotated_ast_group.ast, exec_position[0], target_vars, source_vars)
    new_ast = augment_ast_end_if_simple(annotated_ast_group.ast)
    augmented_asts = [apply_augmentation_if_else_start_end(annot_ast, new_ast) for annot_ast in annotated_ast_group.annot_asts]
    return AnnotatedAstGroup(augmented_asts)

def apply_augmentation_end_else_group(annotated_ast_group: AnnotatedAstGroup, exec_position: ExecutionPositionTuple, target_vars: tuple[str], source_vars: tuple[str]) -> AnnotatedAstGroup:
    new_ast = augment_ast_end_else(annotated_ast_group.ast, exec_position[0], target_vars, source_vars)
    augmented_asts = [apply_augmentation_if_else_start_end(annot_ast, new_ast) for annot_ast in annotated_ast_group.annot_asts]
    return AnnotatedAstGroup(augmented_asts)


################--------------- Augmentation Searchers ------------------##################

class ASTAugmentationController:
    def __init__(self, annot_ast_group: AnnotatedAstGroup, aug_stack: AugmentationStack):
        self.annot_ast_group = annot_ast_group
        self.aug_stack = aug_stack

    @staticmethod
    def init_new_group_controller_from_problem(problem: Problem):
        initial_vars = {f'x{i}':i for i in range(len(problem.input_types))}
        AnnotatedAstGroup.initialize_new_group()
    
    def create_augmentation_request(self):
        """Implement"""

    def apply_augmentation(self, augmentation_request):
        """Implement"""
    
    def copy(self):
        pass

