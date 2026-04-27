"""Reusable problem definitions for search benchmarks.

Each entry returns (problem, known_funcs, known_bools, expected_difficulty_hint).
Add new ones here so all benchmarks share a canonical set.
"""
from core_lang_env.comp_env import BoolFunction, Function
from searchers_utils import Problem


def _int_bools():
    return {
        "gt": BoolFunction(lambda x, y: (x > y,), [int, int], [bool]),
        "equal": BoolFunction(lambda x, y: (x == y,), [int, int], [bool]),
        "is_even": BoolFunction(lambda x: (x % 2 == 0,), [int], [bool]),
        "not": BoolFunction(lambda x: (not x,), [bool], [bool]),
    }


def max_of_two():
    """Return the larger of two ints. Should fall to a single if/else."""
    funcs = {
        "add": Function(lambda x, y: (x + y,), [int, int], [int]),
        "sub": Function(lambda x, y: (x - y,), [int, int], [int]),
        "identity": Function(lambda x: (x,), [int], [int]),
    }
    problem = Problem(
        (int, int),
        (int,),
        instances={
            0: ((3, 7), (7,)),
            1: ((10, 4), (10,)),
            2: ((11, 1), (11,)),
            3: ((18, 23), (23,)),
        },
    )
    return problem, funcs, _int_bools()


def double_or_diff():
    """If x0 > x1: 2*(x0+x1), else x1-x0. Needs if/else with non-trivial branches."""
    funcs = {
        "add": Function(lambda x, y: (x + y,), [int, int], [int]),
        "sub": Function(lambda x, y: (x - y,), [int, int], [int]),
        "mult": Function(lambda x, y: (x * y,), [int, int], [int]),
        "identity": Function(lambda x: (x,), [int], [int]),
    }
    problem = Problem(
        (int, int),
        (int,),
        instances={
            # if-branch (x0 > x1): 2*(x0 + x1)
            0: ((5, 3), (16,)),
            1: ((8, 2), (20,)),
            # else-branch: x1 - x0
            2: ((6, 7), (1,)),
            3: ((5, 6), (1,)),
        },
    )
    return problem, funcs, _int_bools()


ALL = {
    "max_of_two": max_of_two,
    "double_or_diff": double_or_diff,
}
