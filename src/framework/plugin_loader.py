from __future__ import annotations

import os
import sys
import importlib.util
import inspect
from typing import List, Type

from .base import ModelInterface
from .registry import register_model


def discover_and_register_models(plugins_dir: str) -> List[Type[ModelInterface]]:
    """Load all .py files in plugins_dir, find ModelInterface subclasses, and register them.

    Returns list of registered classes.
    """
    registered: List[Type[ModelInterface]] = []
    if not os.path.isdir(plugins_dir):
        return registered

    # Ensure plugins dir in sys.path to allow intra-plugin imports
    if plugins_dir not in sys.path:
        sys.path.insert(0, plugins_dir)

    for fname in os.listdir(plugins_dir):
        if not fname.endswith('.py') or fname.startswith('_'):
            continue
        fpath = os.path.join(plugins_dir, fname)
        mod_name = f"plugin_{os.path.splitext(fname)[0]}"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, fpath)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                for _, obj in inspect.getmembers(mod, inspect.isclass):
                    if issubclass(obj, ModelInterface) and obj is not ModelInterface:
                        register_model(obj)
                        registered.append(obj)
        except Exception as e:
            # Keep discovery resilient; optionally log/print
            print(f"[WARNING] Failed to load plugin {fname}: {e}")
            continue

    return registered
