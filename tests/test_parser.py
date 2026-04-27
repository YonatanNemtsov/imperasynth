"""Parser round-trip — `parse_code_str` and `ast_to_code_str` should be inverses (modulo whitespace).

Strong assertion: parsing the formatted output of an AST yields an AST equal
to the original. This catches regressions in either direction without needing
to compare exact strings.
"""
import pytest

from core_lang_env.parser import ast_to_code_str, parse_code_str


def _round_trip_equal(code: str):
    ast1 = parse_code_str(code)
    formatted = ast_to_code_str(ast1)
    ast2 = parse_code_str(formatted)
    assert ast1 == ast2, f"Round-trip mismatch.\nOriginal AST: {ast1}\nReparsed AST: {ast2}\nFormatted code:\n{formatted}"


def test_round_trip_function_call():
    _round_trip_equal("""
    {
        x2 = add(x0, x1);
        return x2;
    }
    """)


def test_round_trip_if_else():
    _round_trip_equal("""
    {
        if (gt(x0, x1)) {
            x2 = add(x0, x1);
        } else {
            x2 = sub(x0, x1);
        }
        return x2;
    }
    """)


def test_round_trip_while():
    _round_trip_equal("""
    {
        while (gt(x1, x0)) {
            x0 = increment(x0);
        }
        return x0;
    }
    """)


def test_round_trip_direct_assign_swap():
    _round_trip_equal("""
    {
        x0, x1 <- x1, x0;
        return x0;
    }
    """)


def test_round_trip_nested_while_if_with_assign():
    """The non-trivial example used across several notebooks."""
    _round_trip_equal("""
    {
        while (gt(limit, x2)) {
            x3 = add(x2, x1);
            while (gt(lim2, limit)) {
                limit = increment(limit);
            }
            if (is_even(x3)) {
                x3 = add(x3, x3);
            }
            else {}
            x1, x2 <- x2, x3;
        }
        return x1;
    }
    """)


@pytest.mark.xfail(
    reason=(
        "Parser/formatter inconsistency: grammar accepts `x1 = zero()` (no semicolon) "
        "for zero-arg calls, but ast_to_code_str always emits a trailing semicolon, "
        "so the formatted output can't be reparsed."
    ),
    strict=True,
)
def test_round_trip_zero_arg_function():
    _round_trip_equal("""
    {
        x1 = zero();
        return x1;
    }
    """)
