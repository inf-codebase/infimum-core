from abc import ABC, abstractmethod
import functools
from typing import Any, Callable, List, NoReturn, Dict, Optional

from functools import wraps
from inspect import iscoroutinefunction, signature
from loguru import logger

def func_decorator(func):
    def param_decorator(param):
        return func(param)

    if func.__code__.co_argcount:
        return param_decorator
    else:
        return func

class ParameterizedInjection(ABC):
    @abstractmethod
    def on_call_function_action_and_return_params(self)-> List[Any]:
        pass  
    
    @abstractmethod
    def on_call_params_action(self)-> NoReturn:
        pass  
    

def inject(injection_by:ParameterizedInjection):
    def decorator(func: Callable) -> Callable:
        additional_params = injection_by.on_call_function_action_and_return_params()
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            injection_by.on_call_params_action()
            return func(*args,*additional_params, **kwargs)
                
        return wrapper
    
    return decorator

def singleton(cls):
    """Decorate for singleton class
    example:
    @singleton
    class SingletonClass:
        def __init__(self):
            self.value = None
        
        def some_method(self):
            pass
    
    Supports parameterized singletons - different instances for different arguments.
    """
    instances = {}
    def get_instance(*args, **kwargs):
        # Create cache key from class and arguments
        # Use args and sorted kwargs for consistent hashing
        key = (cls, args, tuple(sorted(kwargs.items())))
        if key not in instances:
            instances[key] = cls(*args, **kwargs)
        return instances[key]
    return get_instance

def _safe_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    # Pydantic v1/v2
    for m in ("dict", "model_dump"):
        if hasattr(obj, m) and callable(getattr(obj, m)):
            try:
                return getattr(obj, m)()
            except Exception:
                pass
    # Best-effort
    out = {}
    for k in dir(obj):
        if k.startswith("_"):
            continue
        try:
            v = getattr(obj, k)
            if not callable(v):
                out[k] = v
        except Exception:
            continue
    return out

def _pick(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def vlm_persist_after(task, queue: str = "vlm", inject_task_id: bool = True):
    """
    DEPRECATED: This decorator has been moved to src.core.ai.vlm.decorators.
    
    This function is kept for backward compatibility but will be removed
    in a future release. Please update your imports:
    
    OLD: from core.engine.decorators import vlm_persist_after
    NEW: from src.core.ai.vlm.decorators import vlm_persist_after
    
    Args:
        task: Celery task instance for async processing
        queue: Queue name for task (default: "vlm")
        inject_task_id: Whether to inject task ID into result (default: True)
    
    Returns:
        Decorator function
    """
    import warnings
    from core.ai.vlm.decorators import vlm_persist_after as _new_vlm_persist_after
    
    warnings.warn(
        "vlm_persist_after has been moved to src.core.ai.vlm.decorators. "
        "Please update your imports. This compatibility shim will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return _new_vlm_persist_after(task, queue=queue, inject_task_id=inject_task_id)