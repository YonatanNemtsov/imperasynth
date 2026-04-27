"""Make the parent directory importable, mirroring tests/conftest.py."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
