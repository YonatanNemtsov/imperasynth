"""Reusable problem definitions for search benchmarks.

Each entry returns (problem, known_funcs, known_bools, expected_difficulty_hint).
Add new ones here so all benchmarks share a canonical set.
"""
from core_lang_env.comp_env import BoolFunction, Function
from searchers.searchers_utils import Problem


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


def sum_of_list():
    """Sum the elements of a list. Genuinely needs a while loop — variable
    iteration count per trace can't be expressed without one (or repeated
    if/else of arbitrary depth)."""
    funcs = {
        "add": Function(lambda x, y: (x + y,), [int, int], [int]),
        "get_head": Function(lambda lst: ((lst[0],) if lst else ()), [tuple], [int]),
        "get_tail": Function(lambda lst: ((lst[1:],) if lst else ((),)), [tuple], [tuple]),
        "zero": Function(lambda: (0,), [], [int]),
    }
    bools = {
        "is_empty": BoolFunction(lambda lst: ((len(lst) == 0,)), [tuple], [bool]),
        "not": BoolFunction(lambda b: ((not b,)), [bool], [bool]),
    }
    problem = Problem(
        (tuple,), (int,),
        instances={
            0: (((1,),), (1,)),       # 1 iter
            1: (((1, 2),), (3,)),     # 2 iters
        },
    )
    return problem, funcs, bools


def sum_of_evens():
    """Sum only the even elements of a list. Needs a while loop with an
    if/else inside its body — the condition gates whether each element is
    accumulated."""
    funcs = {
        "add": Function(lambda x, y: (x + y,), [int, int], [int]),
        "get_head": Function(lambda lst: ((lst[0],) if lst else ()), [tuple], [int]),
        "get_tail": Function(lambda lst: ((lst[1:],) if lst else ((),)), [tuple], [tuple]),
        "zero": Function(lambda: (0,), [], [int]),
        "identity": Function(lambda x: (x,), [int], [int]),
    }
    bools = {
        "is_empty": BoolFunction(lambda lst: ((len(lst) == 0,)), [tuple], [bool]),
        "is_even": BoolFunction(lambda x: (x % 2 == 0,), [int], [bool]),
        "not": BoolFunction(lambda b: ((not b,)), [bool], [bool]),
    }
    problem = Problem(
        (tuple,), (int,),
        instances={
            0: (((2,),), (2,)),          # one even
            1: (((1,),), (0,)),          # one odd
            2: (((1, 2),), (2,)),        # mix, len 2
            3: (((2, 3, 4),), (6,)),     # mix, len 3
        },
    )
    return problem, funcs, bools


ALL = {
    "max_of_two": max_of_two,
    "double_or_diff": double_or_diff,
    "sum_of_list": sum_of_list,
    "sum_of_evens": sum_of_evens,
}
