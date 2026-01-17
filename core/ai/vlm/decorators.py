"""
VLM-specific decorators for video/image processing workflows.

This module contains decorators specific to Vision Language Model (VLM) operations,
such as persisting analysis results to the database.
"""

from functools import wraps
from inspect import iscoroutinefunction, signature
from typing import Any, Dict
from loguru import logger


def _safe_to_dict(obj: Any) -> Dict[str, Any]:
    """Safely convert object to dictionary."""
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
    """Pick first available key from dictionary."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def vlm_persist_after(task, queue: str = "vlm", inject_task_id: bool = True):
    """
    Decorator for VLM API predict → save to DB (VLMAnalysis).
    
    This decorator is specific to VLM workflows and expects route responses
    to contain common fields:
      - transcript (required)
      - prompt or prompt_used
      - video_metadata (dict)
      - video_id / frame_index / frame_timestamp_seconds (optional)
    
    If the route has a 'payload' parameter with an 'id', it will be used
    as video_segment_id. Otherwise, it tries to read from result['video_segment_id'].
    
    Args:
        task: Celery task instance for async processing
        queue: Queue name for task (default: "vlm")
        inject_task_id: Whether to inject task ID into result (default: True)
    
    Example:
        ```python
        @vlm_persist_after(vlm_analysis_task, queue="vlm")
        async def predict_image(payload: ImagePayload):
            # Process image and return result with transcript
            return {"transcript": "...", "video_metadata": {...}}
        ```
    """
    def _decorator(fn):
        if iscoroutinefunction(fn):
            @wraps(fn)
            async def _async_wrapper(*args, **kwargs):
                result = await fn(*args, **kwargs)
                try:
                    result_d = _safe_to_dict(result)

                    # Get payload if available (to read .id)
                    bound = signature(fn).bind_partial(*args, **kwargs)
                    payload = bound.arguments.get("payload", None)
                    payload_d = _safe_to_dict(payload)

                    # Map fields according to convention
                    transcript = _pick(result_d, "transcript")
                    prompt = _pick(result_d, "prompt", "prompt_used", default=None)
                    video_md = _pick(result_d, "video_metadata", default={}) or {}
                    video_id = _pick(result_d, "video_id", default=None)
                    frame_idx = _pick(result_d, "frame_index", default=0)
                    frame_ts = _pick(result_d, "frame_timestamp_seconds", default=None)

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

                    # Inject task_id into result.video_metadata if requested
                    if inject_task_id:
                        try:
                            # result might be a model; try to inject into dict video_metadata
                            if isinstance(video_md, dict):
                                video_md["async_persist_task_id"] = async_res.id
                                # if result is dict, set it for client to see
                                if isinstance(result, dict):
                                    result.setdefault("video_metadata", video_md)
                                else:
                                    # if it's a Pydantic model with field, try to set it
                                    if hasattr(result, "video_metadata"):
                                        setattr(result, "video_metadata", video_md)
                        except Exception as e:
                            logger.warning(f"[vlm_persist_after] Cannot inject task id: {e}")

                except Exception as e:
                    # Don't let enqueue failure break the response
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
                    prompt = _pick(result_d, "prompt", "prompt_used", default=None)
                    video_md = _pick(result_d, "video_metadata", default={}) or {}
                    video_id = _pick(result_d, "video_id", default=None)
                    frame_idx = _pick(result_d, "frame_index", default=0)
                    frame_ts = _pick(result_d, "frame_timestamp_seconds", default=None)

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
