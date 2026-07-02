"""
Pytest configuration: makes the `src/` package importable as top-level modules
(e.g. `import analytics`) so tests can mirror how the notebook/scripts import
things, without needing to install the project as a package.
"""
import os
import sys

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))
