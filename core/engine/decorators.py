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
    """
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
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
    Decorator chuyên cho API predict ảnh → lưu DB (VLMAnalysis).
    Kỳ vọng response của route chứa các field phổ biến:
      - transcript (bắt buộc)
      - prompt hoặc prompt_used
      - video_metadata (dict)
      - video_id / frame_index / frame_timestamp_seconds (tuỳ chọn)
    Nếu route có tham số 'payload' và có 'id' → dùng làm video_segment_id.
    Nếu không, cố gắng đọc từ result['video_segment_id'] (nếu có).
    """
    def _decorator(fn):
        if iscoroutinefunction(fn):
            @wraps(fn)
            async def _async_wrapper(*args, **kwargs):
                result = await fn(*args, **kwargs)
                try:
                    result_d = _safe_to_dict(result)

                    # Lấy payload nếu có (để đọc .id)
                    bound = signature(fn).bind_partial(*args, **kwargs)
                    payload = bound.arguments.get("payload", None)
                    payload_d = _safe_to_dict(payload)

                    # Map fields theo quy ước
                    transcript = _pick(result_d, "transcript")
                    prompt     = _pick(result_d, "prompt", "prompt_used", default=None)
                    video_md   = _pick(result_d, "video_metadata", default={}) or {}
                    video_id   = _pick(result_d, "video_id", default=None)
                    frame_idx  = _pick(result_d, "frame_index", default=0)
                    frame_ts   = _pick(result_d, "frame_timestamp_seconds", default=None)

                    video_segment_id = (
                        _pick(payload_d, "id", default=None) or
                        _pick(result_d, "video_segment_id", default=None)
                    )

                    if not transcript:
                        logger.warning("[vlm_persist_after] Missing transcript in route result; skip enqueue.")
                        return result
                    if not video_segment_id:
                        logger.warning("[vlm_persist_after] Missing video_segment_id (payload.id/result.video_segment_id); skip enqueue.")
                        return result

                    inference_payload = {
                        "result": {
                            "transcript": transcript,
                            "video_metadata": video_md,
                            "details": None,
                        },
                        "meta": {
                            "image_path": None,
                            "prompt": prompt,
                            "video_segment_id": video_segment_id,
                            "external_video_id": video_id,
                            "frame_index": frame_idx or 0,
                            "frame_timestamp_seconds": frame_ts,
                        }
                    }

                    async_res = task.apply_async(kwargs={"inference_payload": inference_payload}, queue=queue)

                    # Nhét task_id vào result.video_metadata nếu có
                    if inject_task_id:
                        try:
                            # result có thể là model; cố gắng chèn vào dict video_metadata
                            if isinstance(video_md, dict):
                                video_md["async_persist_task_id"] = async_res.id
                                # nếu result là dict, set lại cho client thấy
                                if isinstance(result, dict):
                                    result.setdefault("video_metadata", video_md)
                                else:
                                    # nếu là Pydantic model có field, thử gán
                                    if hasattr(result, "video_metadata"):
                                        setattr(result, "video_metadata", video_md)
                        except Exception as e:
                            logger.warning(f"[vlm_persist_after] Cannot inject task id: {e}")

                except Exception as e:
                    # Không để việc enqueue fail làm hỏng response
                    logger.error(f"[vlm_persist_after] Enqueue error: {e}")

                return result
            return _async_wrapper

        else:
            @wraps(fn)
            def _sync_wrapper(*args, **kwargs):
                result = fn(*args, **kwargs)
                try:
                    result_d = _safe_to_dict(result)
                    bound = signature(fn).bind_partial(*args, **kwargs)
                    payload = bound.arguments.get("payload", None)
                    payload_d = _safe_to_dict(payload)

                    transcript = _pick(result_d, "transcript")
                    prompt     = _pick(result_d, "prompt", "prompt_used", default=None)
                    video_md   = _pick(result_d, "video_metadata", default={}) or {}
                    video_id   = _pick(result_d, "video_id", default=None)
                    frame_idx  = _pick(result_d, "frame_index", default=0)
                    frame_ts   = _pick(result_d, "frame_timestamp_seconds", default=None)

                    video_segment_id = (
                        _pick(payload_d, "id", default=None) or
                        _pick(result_d, "video_segment_id", default=None)
                    )

                    if not transcript or not video_segment_id:
                        return result

                    inference_payload = {
                        "result": {
                            "transcript": transcript,
                            "video_metadata": video_md,
                            "details": None,
                        },
                        "meta": {
                            "image_path": None,
                            "prompt": prompt,
                            "video_segment_id": video_segment_id,
                            "external_video_id": video_id,
                            "frame_index": frame_idx or 0,
                            "frame_timestamp_seconds": frame_ts,
                        }
                    }
                    async_res = task.apply_async(kwargs={"inference_payload": inference_payload}, queue=queue)

                    if inject_task_id and isinstance(video_md, dict):
                        video_md["async_persist_task_id"] = async_res.id
                        if isinstance(result, dict):
                            result.setdefault("video_metadata", video_md)
                        else:
                            if hasattr(result, "video_metadata"):
                                setattr(result, "video_metadata", video_md)
                except Exception as e:
                    logger.error(f"[vlm_persist_after] Enqueue error: {e}")

                return result
            return _sync_wrapper
    return _decorator