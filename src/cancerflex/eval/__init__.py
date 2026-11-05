"""Evaluation drivers and the shared eval contract.

The modules in this package import each other by flat module name (the layout
of the original research scripts); this directory is therefore placed on
``sys.path`` on import.
"""
import sys as _sys
from pathlib import Path as _Path

_HERE = str(_Path(__file__).resolve().parent)
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
