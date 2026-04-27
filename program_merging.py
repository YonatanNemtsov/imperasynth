from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

from core_lang_env.comp_env import *
from core_lang_env.syntax_tree import *
from searchers_utils import *

import trace_searcher as tsr
import ast_searcher_v3 as asr

def merge_annot_ast_skeleton_groups(group1: asr.AnnotatedAstGroup, group2: asr.AnnotatedAstGroup, var_merge_dict: dict[str, str]):
    pass

def merge_two_linear_ast_skeleton_groups(group1: asr.AnnotatedAstGroup, group2: asr.AnnotatedAstGroup, var_merge: dict[str, str]):
    ast1 = group1.ast
    ast2 = group2.ast

