"""Make the parent directory importable so tests can `from searchers_utils import *` etc.

The simplified_version modules use flat imports (`import searchers_utils`,
`import ast_searcher_v3 as asr`) rather than a package layout, so we need
to put the parent directory on sys.path before tests run.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
