"""Base utilities for lazy imports and module management."""

from typing import Dict, List, Optional, Any
import sys


def create_lazy_getattr(
    module_name: str,
    decorator_map: Optional[Dict[str, str]] = None,
    submodules: Optional[List[str]] = None,
) -> Any:
    """
    Create a reusable __getattr__ function for lazy imports.
    
    Args:
        module_name: The name of the module (for error messages)
        decorator_map: Dictionary mapping attribute names to their import paths
                      Format: {"attr_name": ".module.path"} for relative imports
        submodules: List of submodule names to try importing from
    
    Returns:
        A __getattr__ function that can be assigned to a module's __getattr__
    
    Example:
        # In your module's __init__.py:
        from core.base import create_lazy_getattr
        
        __getattr__ = create_lazy_getattr(
            __name__,
            decorator_map={"singleton": ".decorators"},
            submodules=["context", "startup"]
        )
    """
    decorator_map = decorator_map or {}
    submodules = submodules or []
    
    def __getattr__(name: str) -> Any:
        """Lazy import for decorators, context, and startup modules."""
        # Get the calling module's globals to cache values
        frame = sys._getframe(1)
        module_globals = frame.f_globals
        
        # Import from decorator_map (lightweight, no heavy dependencies)
        if name in decorator_map:
            module_path = decorator_map[name]
            # Handle relative imports (e.g., ".decorators")
            if module_path.startswith("."):
                # Use __import__ with level parameter for relative imports
                module_name_part = module_path.lstrip(".")
                module = __import__(module_name_part, fromlist=[name], level=1)
            else:
                module = __import__(module_path, fromlist=[name])
            
            if hasattr(module, name):
                value = getattr(module, name)
                module_globals[name] = value  # Cache for future access
                return value
        
        # Import from submodules (has database dependencies - only load when needed)
        for submodule_name in submodules:
            try:
                submodule = __import__(submodule_name, fromlist=[name], level=1)
                if hasattr(submodule, name):
                    value = getattr(submodule, name)
                    module_globals[name] = value  # Cache for future access
                    return value
            except ImportError:
                continue
        
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
    
    return __getattr__
