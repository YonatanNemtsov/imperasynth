from dataclasses import dataclass
from core_lang_env.comp_env import *
from core_lang_env.syntax_tree import *
from core_lang_env.exec_code_v2 import *
from .searchers_utils import *
from typing import Literal, Callable

AugType = str

AUG_RETURN: AugType = "AUG_RETURN"
AUG_FUNC_CALL: AugType = "AUG_FUNC_CALL"
AUG_START_IF: AugType = "AUG_START_IF"
AUG_END_IF: AugType = "AUG_END_IF"
AUG_END_ELSE: AugType = "AUG_END_ELSE"

AUG_START_WHILE: AugType = "AUG_START_WHILE"
AUG_END_WHILE: AugType = "AUG_END_WHILE"
AUG_ENTER_WHILE: AugType = "AUG_ENTER_WHILE"
AUG_SKIP_WHILE: AugType = "AUG_SKIP_WHILE"

AUG_EXECUTE_FUNC_CALL: AugType = "AUG_EXECUTE_FUNC_CALL"
AUG_EXECUTE_IF_ELSE: AugType = "AUG_EXECUTE_IF_ELSE"
AUG_EXECUTE_BLOCK: AugType = "AUG_EXECUTE_BLOCK"
AUG_EXECUTE_DIRECT_ASSIGN: AugType = "AUG_EXECUTE_DIRECT_ASSIGN"

ConstructionFrontier = tuple[ExecutionPositionV2, set[AugType], set[int], set[int]]

def make_frontier(exec_pos: ExecutionPositionV2, aug_options: set[AugType], parent_indices: set[int], group_indices: set[int]) -> ConstructionFrontier:
    return (exec_pos, set(aug_options), set(parent_indices), set(group_indices))

def get_frontier_signature(cf: ConstructionFrontier):
    return (cf[0], tuple(sorted(cf[1])), tuple(sorted(cf[2])), tuple(sorted(cf[3])))

def transition_return_option(construction_frontier: ConstructionFrontier, choice: AugType, choice_indices: set[int], subnodes: list[type[ASTNode]]):
    assert choice == AUG_RETURN
    exec_pos, aug_options, parent_indices, group_indices = construction_frontier
    assert choice in aug_options
    assert subnodes == None

    assert set(choice_indices) == set(group_indices)
    return ()

def transition_func_call_option(construction_frontier: ConstructionFrontier, choice: AugType, choice_indices: set[int], subnodes: list[type[ASTNode]]):
    assert choice == AUG_FUNC_CALL
    exec_pos, aug_options, parent_indices, group_indices = construction_frontier
    assert choice in aug_options

    assert set(choice_indices) == set(group_indices)
    assert subnodes == None

    new_exec_pos = exec_pos.next_line()
    new_frontier = make_frontier(new_exec_pos, aug_options.copy(), parent_indices, group_indices)
    return (new_frontier,)

def transition_start_if_option(construction_frontier: ConstructionFrontier, choice: AugType, choice_indices_if: set[int], subnodes: list[type[ASTNode]]):
    assert choice == AUG_START_IF
    exec_pos, aug_options, parent_indices, group_indices = construction_frontier
    assert choice in aug_options

    assert set(choice_indices_if) <= set(group_indices)
    assert subnodes == None

    parent_for_if_level = set(group_indices)
    if_indices = set(choice_indices_if)
    else_indices = parent_for_if_level - if_indices

    # IF branch
    ep_if = exec_pos.enter_context().next_line().enter_context()
    if_exec_pos = ep_if
    if_aug_options = {AUG_FUNC_CALL, AUG_START_IF, AUG_END_IF, AUG_START_WHILE}
    if_frontier = make_frontier(if_exec_pos, if_aug_options, parent_for_if_level, if_indices)

    # ELSE branch
    ep_else = exec_pos.enter_context().next_line().next_line().enter_context()
    else_exec_pos = ep_else
    else_aug_options = {AUG_FUNC_CALL, AUG_START_IF, AUG_END_ELSE, AUG_START_WHILE}
    else_frontier = make_frontier(else_exec_pos, else_aug_options, parent_for_if_level, else_indices)

    # Parent after IF-ELSE
    ep_parent = exec_pos.next_line()
    parent_exec_pos = ep_parent
    parent_frontier = make_frontier(parent_exec_pos, aug_options.copy(), parent_indices, group_indices)

    return (parent_frontier, else_frontier, if_frontier)

def transition_end_if_option(construction_frontier: ConstructionFrontier, choice: AugType, choice_indices: set[int], subnodes: list[type[ASTNode]]=None):
    assert choice == AUG_END_IF
    exec_pos, aug_options, parent_indices, group_indices = construction_frontier
    assert choice in aug_options
    assert subnodes == None

    assert set(choice_indices) == set(group_indices)
    return ()

def transition_end_else_option(construction_frontier: ConstructionFrontier, choice: AugType, choice_indices: set[int], subnodes: list[type[ASTNode]]=None):
    assert choice == AUG_END_ELSE
    exec_pos, aug_options, parent_indices, group_indices = construction_frontier
    assert choice in aug_options
    assert subnodes == None

    assert set(choice_indices) == set(group_indices)
    return ()

# TODO: Implement the remaining transition functions

def transition_start_while_option(construction_frontier: ConstructionFrontier, choice: AugType, choice_indices_while: set[int], subnodes: list[type[ASTNode]]=None):
    assert choice == AUG_START_WHILE
    exec_pos, aug_options, parent_indices, group_indices = construction_frontier
    assert choice in aug_options
    assert set(choice_indices_while) <= set(group_indices)
    assert subnodes == None

    while_block_exec_pos = exec_pos.enter_context().next_line().enter_context()
    building_while_options = {AUG_FUNC_CALL, AUG_START_IF, AUG_START_WHILE, AUG_END_WHILE}
    building_while_frontier = make_frontier(while_block_exec_pos, building_while_options, group_indices, choice_indices_while)


    # Executing frontier sits AT the WhileNode (same exec_pos as start_while):
    # transition_enter_while_option does enter_context().next_line() to reach the body,
    # which only works if exec_pos is at the WhileNode itself, not one level above.
    executing_while_frontier = make_frontier(exec_pos, {AUG_ENTER_WHILE, AUG_SKIP_WHILE}, choice_indices_while, choice_indices_while)

    ep_parent = exec_pos.next_line()
    parent_exec_pos = ep_parent
    parent_frontier = make_frontier(parent_exec_pos, aug_options.copy(), parent_indices, group_indices)
    return (parent_frontier, executing_while_frontier, building_while_frontier)

def transition_end_while_option(construction_frontier: ConstructionFrontier, choice: AugType, choice_indices_while: set[int], subnodes: list[type[ASTNode]]=None):
    assert choice == AUG_END_WHILE
    exec_pos, aug_options, parent_indices, group_indices = construction_frontier
    assert choice in aug_options
    assert subnodes == None

    return ()

def transition_enter_while_option(construction_frontier: ConstructionFrontier, choice: AugType, choice_indices_while: set[int], subnodes: list[type[ASTNode]]):
    assert choice == AUG_ENTER_WHILE
    exec_pos, aug_options, parent_indices, group_indices = construction_frontier
    assert choice in aug_options
    assert set(choice_indices_while) <= set(group_indices)
    assert subnodes == None

    while_block_exec_pos = exec_pos.enter_context().next_line()
    block_frontier = make_frontier(while_block_exec_pos, {AUG_EXECUTE_BLOCK}, group_indices, choice_indices_while)

    new_exec_pos = while_block_exec_pos.exit_context()
    next_while_frontier = make_frontier(new_exec_pos, {AUG_ENTER_WHILE, AUG_SKIP_WHILE}, choice_indices_while, choice_indices_while)
    return (next_while_frontier, block_frontier)

def transition_skip_while_option(construction_frontier: ConstructionFrontier, choice: AugType, choise_indices: set[int], subnodes: list[type[ASTNode]]):
    assert choice == AUG_SKIP_WHILE
    exec_pos, aug_options, parent_indices, group_indices = construction_frontier
    assert choice in aug_options

    return ()

def transition_execute_block_option(construction_frontier: ConstructionFrontier, choice: AugType, choice_indices: set[int], subnodes: list[type[ASTNode]]):
    assert choice == AUG_EXECUTE_BLOCK
    exec_pos, aug_options, parent_indices, group_indices = construction_frontier
    assert choice in aug_options
    assert set(choice_indices) == set(group_indices)
    assert subnodes is not None

    frontiers = []
    curr_pos = exec_pos.enter_context()  # enter block context

    for node_type in subnodes:
        if node_type == WhileNode:
            node_options = {AUG_ENTER_WHILE, AUG_SKIP_WHILE}
        elif node_type == FunctionCallAssignNode:
            node_options = {AUG_EXECUTE_FUNC_CALL}
        elif node_type == DirectAssignNode:
            node_options = {AUG_EXECUTE_DIRECT_ASSIGN}
        elif node_type == IfElseNode:
            node_options = {AUG_EXECUTE_IF_ELSE}
        elif node_type == BlockNode:
            node_options = {AUG_EXECUTE_BLOCK}

        else:
            raise ValueError(f"Unknown block node type: {node_type}")

        node_frontier = make_frontier(curr_pos, node_options, group_indices, choice_indices)
        frontiers.append(node_frontier)

        # advance to next line after node
        curr_pos = curr_pos.next_line()

    return tuple(reversed(frontiers))




def transition_execute_func_call_option(construction_frontier: ConstructionFrontier, choice: AugType, choice_indices: set[int], subnodes: list[type[ASTNode]]):
    assert choice == AUG_EXECUTE_FUNC_CALL
    exec_pos, aug_options, parent_indices, group_indices = construction_frontier
    assert choice in aug_options
    assert subnodes is None

    return ()

def transition_execute_if_else_option(construction_frontier: ConstructionFrontier, choice: AugType, choice_indices: set[int], subnodes: list[type[ASTNode]]):
    assert choice == AUG_EXECUTE_IF_ELSE
    exec_pos, aug_options, parent_indices, group_indices = construction_frontier
    assert choice in aug_options
    assert subnodes is None
    assert set(choice_indices) <= set(group_indices)

    parent_for_if_level = set(group_indices)
    if_indices = set(choice_indices)
    else_indices = parent_for_if_level - if_indices


    # Land AT the body block (not inside): execute_block's transition does its
    # own enter_context to reach the first child. Cf. transition_enter_while_option,
    # which uses the same `enter_context().next_line()` pattern (no trailing
    # enter_context) for the same reason.
    if_exec_pos = exec_pos.enter_context().next_line()
    if_frontier = make_frontier(if_exec_pos, {AUG_EXECUTE_BLOCK}, parent_for_if_level, if_indices)

    else_exec_pos = exec_pos.enter_context().next_line().next_line()
    else_frontier = make_frontier(else_exec_pos, {AUG_EXECUTE_BLOCK}, parent_for_if_level, else_indices)

    return (else_frontier, if_frontier)


def transition_execute_direct_assign_option(construction_frontier: ConstructionFrontier, choice: AugType, choice_indices: set[int], subnodes: list[type[ASTNode]]):
    assert choice == AUG_EXECUTE_DIRECT_ASSIGN
    exec_pos, aug_options, parent_indices, group_indices = construction_frontier
    assert choice in aug_options
    assert subnodes is None

    return ()


TRANSITION_FUNCTIONS = {
    AUG_RETURN: transition_return_option,
    AUG_FUNC_CALL: transition_func_call_option, 
    AUG_START_IF: transition_start_if_option, 
    AUG_END_IF: transition_end_if_option,
    AUG_END_ELSE: transition_end_else_option,
    
    AUG_START_WHILE: transition_start_while_option,
    AUG_END_WHILE: transition_end_while_option,
    AUG_ENTER_WHILE: transition_enter_while_option,
    AUG_SKIP_WHILE: transition_skip_while_option,
    
    AUG_EXECUTE_FUNC_CALL: transition_execute_func_call_option,
    AUG_EXECUTE_IF_ELSE: transition_execute_if_else_option,
    AUG_EXECUTE_BLOCK: transition_execute_block_option,
    AUG_EXECUTE_DIRECT_ASSIGN: transition_execute_direct_assign_option,
}

class AugmentationStack:
    def __init__(self, stack: list[ConstructionFrontier], all_indices: set[int]):
        self.stack = stack
        self.all_indices = set(all_indices)

    @staticmethod
    def init_new_stack(all_indices: set[int]) -> "AugmentationStack":
        ep = ExecutionPositionV2.start_position()
        root_exec_pos = ep.enter_context()
        root_frontier = make_frontier(root_exec_pos, {AUG_FUNC_CALL, AUG_START_IF, AUG_START_WHILE, AUG_RETURN}, set(all_indices), set(all_indices))
        return AugmentationStack([root_frontier], all_indices)

    def peek(self) -> ConstructionFrontier | None:
        return self.stack[-1] if self.stack else None

    def pop(self) -> ConstructionFrontier | None:
        return self.stack.pop() if self.stack else None

    def push(self, *new_frontiers: ConstructionFrontier):
        for frontier in new_frontiers:
            if frontier and frontier[3]:
                self.stack.append(frontier)

    def apply_transition(self, choice: AugType, choice_indices: set[int], subnodes: list[type[ASTNode]]):
        current = self.pop()
        if current is None:
            raise RuntimeError("Stack underflow: no frontier to apply transition on")
        new_frontiers = TRANSITION_FUNCTIONS[choice](current, choice, choice_indices, subnodes)
        self.push(*new_frontiers)
        return new_frontiers

    def copy(self) -> "AugmentationStack":
        return AugmentationStack(self.stack.copy(), self.all_indices.copy())

    def __repr__(self):
        return f"AugmentationStack({str(self.stack[-1:])[:-1]}, ...], indices={self.all_indices})"
    
    def get_signature(self):
        return (tuple(get_frontier_signature(cf) for cf in self.stack))

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
    
    def get_signature(self):
        sig =  (compute_block_hash(self.ast, self.initial_vars.copy())[0], *((annot.signature.to_tuple()) for annot in self.annot_asts))
        return sig


def get_all_variables_in_ast(ast: ASTNode, initial_vars: set[str]):
    variables = set(initial_vars)
    def traverse(node: ASTNode):
        for child in node.children:
            if isinstance(child, DirectAssignNode):
                variables.update(var for var in child.target_vars if var[0] == 'x')
            if isinstance(child, FunctionCallAssignNode):
                variables.update(var for var in child.var_names if var[0] == 'x')
            if type(child) in [BlockNode, WhileNode, IfElseNode]:
                traverse(child)
    traverse(ast)
    return variables

def get_largest_label_of_variables(ast: BlockNode, initial_vars: set[str]):
    variables = get_all_variables_in_ast(ast, initial_vars)
    largest_label = max(int(x[1:]) for x in variables)
    return largest_label

def get_semantically_available_variables(ast: BlockNode, initial_vars: set[str], position: ASTNodePosition):
    return initial_vars.copy()

DetailedActionGroup = tuple[ShortActionByVarName, tuple[ShortAction, ...], int]

def augment_ast_return(ast: BlockNode, insertion_pos: ASTNodePosition, return_vars: tuple[str]) -> BlockNode:
    new_ast = ast.copy()
    new_node = ReturnNode(return_vars[0])
    new_ast.insert_node(insertion_pos, new_node)
    return new_ast

def apply_augmentation_no_action(annotated_ast: AnnotatedAST, new_ast: BlockNode, exec_position: ExecutionPositionTuple, no_action: ShortAction) -> AnnotatedAST:
    assert no_action == NO_ACTION
    return AnnotatedAST(new_ast, annotated_ast.signature.copy(), annotated_ast.mapping.copy(), annotated_ast.initial_vars.copy())

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

def augment_ast_function_call(ast: BlockNode, initial_vars: set[str], insertion_pos: ASTNodePosition, variable_action: ShortActionByVarName, output_length: int) -> BlockNode:
    func_name, input_vars = variable_action
    largest_label = get_largest_label_of_variables(ast, initial_vars)
    output_vars = tuple(f'x{i}' for i in range(largest_label + 1, largest_label + 1 + output_length))
    new_node = FunctionCallAssignNode(output_vars, func_name, input_vars)
    new_ast = ast.copy()
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

def augment_ast_start_if(ast: BlockNode, insertion_pos: ASTNodePosition) -> BlockNode:
    new_if_else_node = IfElseNode(BoolExprNode(TBD_CONDITIONAL, []), BlockNode([]), BlockNode([]))
    new_ast = ast.copy()
    new_ast.insert_node(insertion_pos, new_if_else_node)
    return new_ast

def augment_ast_end_if(ast: BlockNode, insertion_pos: ASTNodePosition, target_vars: tuple[str], source_vars: tuple[str]) -> BlockNode:
    assert insertion_pos[-2] == 1
    new_final_assign_node = DirectAssignNode(target_vars, source_vars)
    new_ast = ast.copy()
    new_ast.insert_node(insertion_pos, new_final_assign_node)
    return new_ast

def augment_ast_end_if_simple(ast: BlockNode) -> BlockNode:
    return ast.copy()

def augment_ast_end_else(ast: BlockNode, insertion_pos: ASTNodePosition, target_vars: tuple[str], source_vars: tuple[str]) -> BlockNode:
    assert insertion_pos[-2] == 2
    new_final_assign_node = DirectAssignNode(target_vars, source_vars)
    new_ast = ast.copy()
    new_ast.insert_node(insertion_pos, new_final_assign_node)
    return new_ast

def apply_augmentation_if_start_end(annotated_ast: AnnotatedAST, new_ast: BlockNode) -> AnnotatedAST:
    return AnnotatedAST(new_ast, annotated_ast.signature.copy(), annotated_ast.mapping.copy(), annotated_ast.initial_vars.copy())

def apply_augmentation_else_start(annotated_ast: AnnotatedAST, new_ast: BlockNode) -> AnnotatedAST:
    return AnnotatedAST(new_ast, annotated_ast.signature.copy(), annotated_ast.mapping.copy(), annotated_ast.initial_vars.copy())

def apply_augmentation_else_end(annotated_ast: AnnotatedAST, new_ast: BlockNode, exec_position: ExecutionPositionTuple, target_vars: tuple[str], source_vars: tuple[str]):
    new_annot = AnnotatedAST(new_ast, annotated_ast.signature.copy(), annotated_ast.mapping.copy(), annotated_ast.initial_vars.copy())
    new_annot.add_action(('ASSIGN', (target_vars, source_vars, exec_position)), exec_position)
    return new_annot

def apply_augmentation_start_if_group(annotated_ast_group: AnnotatedAstGroup, exec_position: ExecutionPositionTuple) -> AnnotatedAstGroup:
    new_ast = augment_ast_start_if(annotated_ast_group.ast, exec_position[0])
    augmented_asts = [apply_augmentation_if_start_end(annot_ast, new_ast) for annot_ast in annotated_ast_group.annot_asts]
    return AnnotatedAstGroup(augmented_asts)

def apply_augmentation_end_if_group(annotated_ast_group: AnnotatedAstGroup) -> AnnotatedAstGroup:
    new_ast = augment_ast_end_if_simple(annotated_ast_group.ast)
    augmented_asts = [apply_augmentation_if_start_end(annot_ast, new_ast) for annot_ast in annotated_ast_group.annot_asts]
    return AnnotatedAstGroup(augmented_asts)

def augment_ast_start_while(ast: BlockNode, insertion_pos: ASTNodePosition) -> BlockNode:
    """Insert a fresh WhileNode (with TBD condition and empty body) at insertion_pos."""
    new_while_node = WhileNode(BoolExprNode(TBD_CONDITIONAL, []), BlockNode([]))
    new_ast = ast.copy()
    new_ast.insert_node(insertion_pos, new_while_node)
    return new_ast


def apply_augmentation_start_while_group(annotated_ast_group: AnnotatedAstGroup, exec_position: ExecutionPositionTuple) -> AnnotatedAstGroup:
    new_ast = augment_ast_start_while(annotated_ast_group.ast, exec_position[0])
    augmented_asts = [apply_augmentation_if_start_end(annot_ast, new_ast) for annot_ast in annotated_ast_group.annot_asts]
    return AnnotatedAstGroup(augmented_asts)


def augment_ast_end_while(ast: BlockNode, insertion_pos: ASTNodePosition, target_vars: tuple[str], source_vars: tuple[str]) -> BlockNode:
    """Append a DirectAssignNode at the end of a while body — the rebinding step
    that lets the next iteration see updated values bound to the loop variable names.
    """
    new_final_assign_node = DirectAssignNode(target_vars, source_vars)
    new_ast = ast.copy()
    new_ast.insert_node(insertion_pos, new_final_assign_node)
    return new_ast


def apply_augmentation_end_while_group(
    annotated_ast_group: AnnotatedAstGroup,
    exec_position: ExecutionPositionTuple,
    target_vars: tuple[str],
    source_vars: tuple[str],
    group_indices: set[int],
) -> AnnotatedAstGroup:
    """End of while body: optionally insert a DirectAssign rebinding (similar
    to end_else). If target_vars is empty, this is a no-op AST refresh.
    """
    if not target_vars:
        new_ast = annotated_ast_group.ast.copy()
        augmented_asts = [apply_augmentation_if_start_end(annot_ast, new_ast) for annot_ast in annotated_ast_group.annot_asts]
        return AnnotatedAstGroup(augmented_asts)

    new_ast = augment_ast_end_while(annotated_ast_group.ast, exec_position[0], target_vars, source_vars)
    augmented_asts = [
        apply_augmentation_else_end(annot_ast, new_ast, exec_position, target_vars, source_vars)
        if i in group_indices
        else apply_augmentation_no_action(annot_ast, new_ast, exec_position, NO_ACTION)
        for i, annot_ast in enumerate(annotated_ast_group.annot_asts)
    ]
    return AnnotatedAstGroup(augmented_asts)


def apply_augmentation_end_else_group(annotated_ast_group: AnnotatedAstGroup, exec_position: ExecutionPositionTuple, target_vars: tuple[str], source_vars: tuple[str], group_indices: set[int]) -> AnnotatedAstGroup:
    new_ast = augment_ast_end_else(annotated_ast_group.ast, exec_position[0], target_vars, source_vars)
    augmented_asts = [
        apply_augmentation_else_end(annot_ast, new_ast, exec_position, target_vars, source_vars) 
        if i in group_indices 
        else apply_augmentation_no_action(annot_ast, new_ast, exec_position, NO_ACTION)
        for i, annot_ast in enumerate(annotated_ast_group.annot_asts)
    ]
    return AnnotatedAstGroup(augmented_asts)