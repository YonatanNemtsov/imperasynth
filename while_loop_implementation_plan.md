# While Loop Implementation Plan

## Summary

`ast_searcher_v3.py` has been updated with a two-phase while loop design:
- **Build phase** (`AUG_START_WHILE` / `AUG_END_WHILE`): constructs the loop body incrementally
- **Execute phase** (`AUG_ENTER_WHILE` / `AUG_SKIP_WHILE`): replays the pre-built body across traces

All transition functions now accept a `subnodes` parameter. New execution-phase transitions (`AUG_EXECUTE_BLOCK`, `AUG_EXECUTE_FUNC_CALL`, `AUG_EXECUTE_IF_ELSE`, `AUG_EXECUTE_DIRECT_ASSIGN`) handle replaying pre-built AST blocks.

---

## Changes Needed

### 1. `search_orchestrator.py` — Main file to update

#### 1a. Fix `apply_transition` calls
All existing calls pass `(choice, indices)` — add `subnodes=None` as the third argument.
- `apply_func_call_candidate` (line ~122)
- `apply_return_candidate` (line ~137)
- `apply_start_if_candidate` (line ~148)
- `apply_end_if_candidate` (line ~159)
- `apply_end_else_candidate` (line ~185)

#### 1b. New `AugmentationRequest` subclasses
- [ ] `AugmentationRequestStartWhile`
- [ ] `AugmentationRequestEndWhile`
- [ ] `AugmentationRequestEnterWhile`
- [ ] `AugmentationRequestSkipWhile`
- [ ] `AugmentationRequestExecuteBlock`
- [ ] `AugmentationRequestExecuteFuncCall`
- [ ] `AugmentationRequestExecuteIfElse`
- [ ] `AugmentationRequestExecuteDirectAssign`

#### 1c. Update `cls_map` dict
Add entries mapping each new `AUG_*` type to its request class so `generate_augmentation_requests_from_state` produces them.

#### 1d. New candidate classes
- [ ] `StartWhileCandidate`
- [ ] `EndWhileCandidate`
- [ ] `EnterWhileCandidate`
- [ ] `SkipWhileCandidate`
- [ ] Execute-phase candidates (if needed beyond the request classes)

#### 1e. New `generate_*_candidates` functions
- [ ] `generate_start_while_candidates` — use `searchers_utils.get_while_node_exec_positions`
- [ ] `generate_end_while_candidates`
- [ ] `generate_enter_while_candidates` — use `searchers_utils.get_while_node_exec_position_groups`
- [ ] `generate_skip_while_candidates`
- [ ] Execute-phase candidate generators

#### 1f. New `apply_*_candidate` methods on `SearchState`
- [ ] `apply_start_while_candidate`
- [ ] `apply_end_while_candidate`
- [ ] `apply_enter_while_candidate`
- [ ] `apply_skip_while_candidate`
- [ ] Execute-phase apply methods

#### 1g. Update `step()` dispatch (line ~441)
Add `elif` branches for each new request type.

#### 1h. Update `apply_candidate()` dispatch (line ~484)
Add cases for each new candidate type.

#### 1i. Boolean expression search for while conditions
`search_boolean_expressions` currently only handles `IfElseNode`. Needs:
- [ ] `get_while_positions` (like `get_ifelse_positions`)
- [ ] `search_boolean_expression_while_node`
- [ ] `integrate_boolean_expression_while_node`

---

### 2. `ast_searcher_v3.py` — Remaining TODOs

- [ ] `augment_ast_start_while` — insert a `WhileNode` into the AST (analogous to `augment_ast_start_if`)
- [ ] `apply_augmentation_start_while_group` — group-level AST mutation
- [ ] Mechanism for the orchestrator to determine what `subnodes` to pass to `transition_execute_block_option`

---

### 3. `trace_searcher.py`

- [ ] `find_possible_start_while_actions` — determine which traces enter the loop vs skip (similar to `find_possible_start_if_actions`)
- [ ] Helpers for aligning per-iteration actions across traces during the execute phase

---

### 4. `condition_searcher.py`

- [ ] `extract_while_conditional_problem_for_group` — learn while-loop boolean conditions from traces (analogous to `extract_ifelse_conditional_problem_for_group`)

---

## Modules that need NO changes

- **`core_lang_env/`** — `WhileNode`, execution (`execute_while_node`), and parsing already fully implemented
- **`data_flow_graph.py`** — already handles `WhileNode`
- **`searchers_utils.py`** — already has `get_while_node_exec_positions` and `get_while_node_exec_position_groups`
