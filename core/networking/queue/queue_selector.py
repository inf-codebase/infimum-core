"""
Queue Selector Utility
Automatically detects GPU availability and selects appropriate Celery queue

Uses centralized GPU detection from auto_config:
- auto_config.CUDA_AVAILABLE (bool): True if CUDA GPU is available
- auto_config.GPU_COUNT (int): Number of detected GPUs
- auto_config.CUDA_VERSION (str): CUDA version or None

Queue naming convention (adapted to celery_app configuration):
- fastvlm.gpu - For FastVLM GPU tasks
- general.cpu - For general CPU tasks (fallback or CPU-only mode)
"""
import hashlib
from typing import Literal
from core.utils import auto_config

QueueType = Literal["fastvlm.gpu", "general.cpu"]


def detect_available_queues() -> dict:
    """
    Detect which queues are available based on hardware.

    Uses centralized GPU detection from auto_config.CUDA_AVAILABLE.

    Returns:
        dict: {
            "has_gpu": bool,
            "gpu_queues": list of GPU queue names,
            "cpu_queue": str,
            "all_queues": list of all available queues
        }
    """
    has_gpu = auto_config.CUDA_AVAILABLE

    # Use single GPU queue as configured in celery_app
    # The celery workers handle load balancing internally
    gpu_queues = ["fastvlm.gpu"] if has_gpu else []
    cpu_queue = "general.cpu"

    return {
        "has_gpu": has_gpu,
        "gpu_queues": gpu_queues,
        "cpu_queue": cpu_queue,
        "all_queues": gpu_queues + [cpu_queue] if has_gpu else [cpu_queue]
    }


def select_queue(group_id: str = None, prefer_gpu: bool = True, idx = None) -> QueueType:
    """
    Automatically select the best queue based on GPU availability.

    Args:
        group_id: Optional group ID (unused - kept for backward compatibility)
        prefer_gpu: If True and GPU available, use GPU queue. If False, use CPU queue.
        idx: Optional index (unused - kept for backward compatibility)

    Returns:
        Queue name to use ("fastvlm.gpu" or "general.cpu")

    Examples:
        >>> # Auto-detect and use GPU if available
        >>> queue = select_queue()

        >>> # Force CPU even if GPU available
        >>> queue = select_queue(prefer_gpu=False)
    """
    queues = detect_available_queues()

    # If no GPU or user prefers CPU, return CPU queue
    if not queues["has_gpu"] or not prefer_gpu:
        return queues["cpu_queue"]

    # If GPU available and preferred, return GPU queue
    # Note: Load balancing is handled by multiple celery workers listening to the same queue
    gpu_queues = queues["gpu_queues"]
    return gpu_queues[0] if gpu_queues else queues["cpu_queue"]


def get_queue_info() -> dict:
    """
    Get information about current queue configuration.

    Uses centralized GPU detection from auto_config.

    Returns:
        dict: Queue configuration details including GPU and MPS availability
    """
    import os

    queues = detect_available_queues()

    return {
        "gpu_available": auto_config.CUDA_AVAILABLE,
        "gpu_count": auto_config.GPU_COUNT,
        "cuda_version": auto_config.CUDA_VERSION,
        "mps_available": auto_config.MPS_AVAILABLE,
        "has_acceleration": auto_config.HAS_ACCELERATION,
        "active_gpu_index": int(os.getenv("GPU_INDEX", "0")),
        "recommended_queue": select_queue(),
        "all_available_queues": queues["all_queues"],
        "gpu_queues": queues["gpu_queues"],
        "cpu_queue": queues["cpu_queue"],
    }
