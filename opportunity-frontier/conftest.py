"""Make ``core`` and ``providers`` importable when running pytest from anywhere
inside the project."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
