"""
Distributed Task Processing Utilities

Provides reusable patterns for distributing batch work across multiple Celery workers.
This allows full utilization of all available GPU/CPU workers for parallel processing.

Usage Example:
    from src.core.networking.queue.distributed_tasks import distribute_batch_task

    # Automatically chunks and distributes items across workers
    result = distribute_batch_task(
        task=my_processing_task,
        items=image_list,
        queue="fastvlm.gpu",
        chunk_size=8
    )
"""

from celery import group
from celery.result import AsyncResult, GroupResult
from typing import List, Dict, Any, Optional, Callable
from time import time as now
from loguru import logger

from src.core.utils import auto_config
from src.core.networking.queue.celery_app import celery_app
from src.core.networking.queue.queue_selector import select_queue


def distribute_batch_task(
    task: Callable,
    items: List[Any],
    queue: Optional[str] = None,
    chunk_size: Optional[int] = None,
    prefer_gpu: bool = True,
    group_id: Optional[str] = None,
    **task_kwargs
) -> GroupResult:
    """
    Distribute a batch of items across multiple workers for parallel processing.

    This function automatically:
    1. Chunks items into optimal batch sizes
    2. Creates parallel Celery tasks for each chunk
    3. Distributes tasks across all available workers
    4. Returns a GroupResult for tracking progress

    Args:
        task: Celery task to execute (must be a @shared_task)
        items: List of items to process
        queue: Target queue name (auto-detected if None)
        chunk_size: Items per chunk (uses VLM_BATCH_SIZE if None)
        prefer_gpu: Prefer GPU queue if available (default: True)
        group_id: Optional group identifier for logging
        **task_kwargs: Additional kwargs passed to each task

    Returns:
        GroupResult: Celery group result for tracking all tasks

    Example:
        >>> from src.core.networking.queue.analyze_image_tasks import batch_analyze_task
        >>>
        >>> items = [{"temp_path": "img1.jpg", "prompt": "Describe"}] * 100
        >>> result = distribute_batch_task(
        ...     task=batch_analyze_task,
        ...     items=items,
        ...     prefer_gpu=True,
        ...     enqueue_ts=time.time()
        ... )
        >>>
        >>> # Check progress
        >>> print(f"Completed: {result.completed_count()}/{len(result.results)}")
        >>>
        >>> # Get results when ready
        >>> if result.ready():
        ...     all_results = result.get()
    """
    if not items:
        raise ValueError("items cannot be empty")

    # Auto-detect queue if not specified
    if queue is None:
        queue = select_queue(group_id=group_id, prefer_gpu=prefer_gpu)

    # Use configured batch size if not specified
    if chunk_size is None:
        chunk_size = auto_config.VLM_BATCH_SIZE

    # Chunk items for distribution
    chunks = [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]

    logger.info(
        f"[distribute_batch_task] Distributing {len(items)} items "
        f"across {len(chunks)} tasks (chunk_size={chunk_size}, queue={queue})"
    )

    # Create group of tasks - each worker can pick up chunks in parallel
    job = group(
        task.s(chunk, **task_kwargs).set(
            queue=queue,
            routing_key=queue
        )
        for chunk in chunks
    )

    # Execute and return group result
    group_result = job.apply_async()

    # IMPORTANT: Save the group result to backend so it can be restored later
    # This is required for GroupResult.restore() to work
    group_result.save()

    return group_result


def get_distributed_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get status of a distributed task (group or single).

    Handles both:
    - GroupResult: Multiple parallel tasks
    - AsyncResult: Single task

    Args:
        task_id: Task or group ID from distribute_batch_task

    Returns:
        dict: Status information with standardized format

    Response for Group (still running):
        {
            "task_id": "abc-123",
            "type": "group",
            "status": "RUNNING",
            "total_tasks": 13,
            "completed": 8,
            "pending": 5,
            "ready": False,
            "message": "Processing: 8/13 tasks completed"
        }

    Response for Group (completed):
        {
            "task_id": "abc-123",
            "type": "group",
            "status": "SUCCESS",
            "total_tasks": 13,
            "completed": 13,
            "ready": True,
            "result": {
                "success": True,
                "total_items": 100,
                "total_chunks": 13,
                "data": [...]
            }
        }

    Response for Single Task:
        {
            "task_id": "def-456",
            "type": "single",
            "status": "SUCCESS",
            "ready": True,
            "result": {...}
        }
    """
    try:
        # Try to restore as GroupResult first
        result = GroupResult.restore(task_id, app=celery_app)

        if result is not None:
            # This is a group of distributed tasks
            completed = result.completed_count()
            total = len(result.results)

            response = {
                "task_id": task_id,
                "type": "group",
                "total_tasks": total,
                "completed": completed,
                "pending": total - completed,
                "ready": result.ready(),
                "successful": result.successful() if result.ready() else None,
            }

            if result.ready():
                # All tasks completed
                if result.successful():
                    # Aggregate results from all chunks
                    all_results = result.get(timeout=1.0)
                    aggregated_data = []
                    total_items = 0

                    for chunk_result in all_results:
                        if chunk_result and chunk_result.get("success"):
                            chunk_data = chunk_result.get("data", [])
                            aggregated_data.extend(chunk_data)
                            total_items += len(chunk_data)

                    response["status"] = "SUCCESS"
                    response["message"] = f"All {total} tasks completed successfully"
                    response["result"] = {
                        "success": True,
                        "total_items": total_items,
                        "total_chunks": total,
                        "data": aggregated_data
                    }
                else:
                    # Some tasks failed
                    response["status"] = "FAILURE"
                    response["message"] = "Some tasks failed"
                    response["errors"] = [
                        str(r.info) for r in result.results
                        if r.failed()
                    ]
            else:
                # Still running
                response["status"] = "RUNNING"
                response["message"] = f"Processing: {completed}/{total} tasks completed"

            return response

    except Exception as e:
        logger.debug(f"Not a group result: {e}")

    # Fallback to single task
    result = AsyncResult(task_id, app=celery_app)

    response = {
        "task_id": task_id,
        "type": "single",
        "status": result.status,
        "ready": result.ready(),
    }

    if result.status == "PENDING":
        response["message"] = "Task is waiting in queue or does not exist"

    elif result.status == "STARTED":
        response["message"] = "Task is currently running"

    elif result.status == "SUCCESS":
        response["message"] = "Task completed successfully"
        response["result"] = result.result

    elif result.status == "FAILURE":
        response["message"] = "Task failed"
        response["error"] = str(result.info)
        if hasattr(result, 'traceback'):
            response["traceback"] = str(result.traceback)

    else:
        response["message"] = f"Task status: {result.status}"
        if result.ready():
            try:
                response["result"] = result.get(timeout=0.1)
            except Exception as e:
                response["error"] = str(e)

    return response


def create_distributed_task_wrapper(
    task_name: str,
    default_queue: str,
    default_chunk_size: Optional[int] = None
):
    """
    Factory function to create a reusable distributed task submitter.

    This creates a specialized wrapper function for your specific task type,
    so you don't need to specify task/queue/chunk_size every time.

    Args:
        task_name: Name of the Celery task (e.g., "fastvlm.batch_analyze")
        default_queue: Default queue for this task type
        default_chunk_size: Default chunk size (None = use VLM_BATCH_SIZE)

    Returns:
        Callable: Function that submits distributed batches

    Example:
        >>> # Create specialized wrapper
        >>> submit_fastvlm_batch = create_distributed_task_wrapper(
        ...     task_name="fastvlm.batch_analyze",
        ...     default_queue="fastvlm.gpu",
        ...     default_chunk_size=8
        ... )
        >>>
        >>> # Use it anywhere
        >>> result = submit_fastvlm_batch(items=image_list, enqueue_ts=time.time())
    """
    from src.core.networking.queue.celery_app import celery_app

    task = celery_app.tasks[task_name]

    def wrapper(items: List[Any], **kwargs) -> Dict[str, Any]:
        """
        Submit a distributed batch for processing.

        Args:
            items: List of items to process
            **kwargs: Additional task arguments

        Returns:
            dict: Submission info with task_id, chunks, etc.
        """
        chunk_size = kwargs.pop('chunk_size', default_chunk_size or auto_config.VLM_BATCH_SIZE)
        queue = kwargs.pop('queue', default_queue)

        result = distribute_batch_task(
            task=task,
            items=items,
            queue=queue,
            chunk_size=chunk_size,
            **kwargs
        )

        chunks = [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]

        return {
            "success": True,
            "task_id": result.id,
            "queue": queue,
            "count": len(items),
            "chunks": len(chunks),
            "chunk_size": chunk_size,
            "message": f"Distributed {len(items)} items across {len(chunks)} parallel tasks"
        }

    return wrapper


# Pre-configured wrappers for common tasks
def submit_distributed_fastvlm_batch(items: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
    """
    Submit FastVLM batch analysis with automatic worker distribution.

    Args:
        items: List of dicts with 'temp_path' and 'prompt'
        **kwargs: Additional args (enqueue_ts, etc.)

    Returns:
        dict: Submission info

    Example:
        >>> items = [{"temp_path": "img1.jpg", "prompt": "Describe"}] * 100
        >>> result = submit_distributed_fastvlm_batch(items, enqueue_ts=time.time())
        >>> print(result["task_id"])
    """
    from src.core.networking.queue.analyze_image_tasks import batch_analyze_task

    queue = select_queue(prefer_gpu=True)
    chunk_size = kwargs.pop('chunk_size', auto_config.VLM_BATCH_SIZE)

    result = distribute_batch_task(
        task=batch_analyze_task,
        items=items,
        queue=queue,
        chunk_size=chunk_size,
        **kwargs
    )

    chunks = [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]

    return {
        "success": True,
        "task_id": result.id,
        "queue": queue,
        "count": len(items),
        "chunks": len(chunks),
        "chunk_size": chunk_size,
        "message": f"Distributed {len(items)} items across {len(chunks)} parallel tasks"
    }
