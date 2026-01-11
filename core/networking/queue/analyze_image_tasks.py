# src/queue/tasks.py
import time
import logging
from celery import shared_task
from celery.utils.log import get_task_logger

from src.features.ai_models.services.fastvlm_service import get_fastvlm_service
import os
from types import SimpleNamespace
from src.core.ai.visual_language_model.fastvlm_inference import predict_batch,predict_batch_true_batched
from loguru import logger
from src.core.utils import auto_config

@shared_task(bind=True, name="fastvlm.analyze_image")
def analyze_image_task(self, temp_path: str, prompt: str, enqueue_ts: float = None):
    """
    Đo thời gian từng task theo worker:
      - queue_wait_ms: thời gian chờ trong hàng đợi
      - exec_ms: thời gian thực thi trên worker
      - total_ms: tổng từ enqueue -> finish
    """
    start_ts = time.time()
    # meta worker/queue
    worker = getattr(self.request, "hostname", None)
    delivery = getattr(self.request, "delivery_info", None) or {}
    queue = delivery.get("routing_key")
    fastvlm_service = get_fastvlm_service()
    # (tuỳ chọn) phát tín hiệu STARTED + queue_wait
    if enqueue_ts:
        try:
            self.update_state(
                state="STARTED",
                meta={"stage": "started", "queue_wait_ms": round((start_ts - enqueue_ts) * 1000, 2)}
            )
        except Exception:
            pass  # không critical

    try:
        out = fastvlm_service.analyze_image(temp_path, prompt)
        finish_ts = time.time()

        metrics = {
            "enqueue_ts": enqueue_ts,
            "start_ts": start_ts,
            "finish_ts": finish_ts,
            "queue_wait_ms": round((start_ts - enqueue_ts) * 1000, 2) if enqueue_ts else None,
            "exec_ms":       round((finish_ts - start_ts) * 1000, 2),
            "total_ms":      round((finish_ts - enqueue_ts) * 1000, 2) if enqueue_ts else None,
            "worker": worker,
            "queue": queue,
            "task_id": self.request.id,
        }

        logger.info("[METRICS] %s", metrics)

        # DON'T update_state to SUCCESS - it overwrites the return value!
        # Celery automatically sets SUCCESS state when the task returns
        # try:
        #     self.update_state(state="SUCCESS", meta={"stage": "finished", **{k: v for k, v in metrics.items() if k.endswith('_ms')}})
        # except Exception:
        #     pass

        return {"success": True, "data": out, "metrics": metrics}

    except Exception as e:
        finish_ts = time.time()
        metrics = {
            "enqueue_ts": enqueue_ts,
            "start_ts": start_ts,
            "finish_ts": finish_ts,
            "queue_wait_ms": round((start_ts - enqueue_ts) * 1000, 2) if enqueue_ts else None,
            "exec_ms":       round((finish_ts - start_ts) * 1000, 2),
            "total_ms":      round((finish_ts - enqueue_ts) * 1000, 2) if enqueue_ts else None,
            "worker": worker,
            "queue": queue,
            "task_id": self.request.id,
            "error": str(e),
        }
        logger.exception("[METRICS][ERROR] %s", metrics)
        raise


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

@shared_task(bind=True, name="fastvlm.batch_analyze")
def batch_analyze_task(self, items, enqueue_ts: float = None):
    start_ts = time.time()
    worker = getattr(self.request, "hostname", None)
    delivery = getattr(self.request, "delivery_info", None) or {}
    queue = delivery.get("routing_key")

    svc = get_fastvlm_service()
    try:
        tokenizer, model, image_processor, _ = svc.get_handles()

        args_list = [
            SimpleNamespace(
                model_path=svc.model_path,
                prompt=it.get("prompt") or "Describe the image.",
                image_file=it["temp_path"],
                conv_mode="qwen_2",
                device=svc.device,
                temperature=0.0,
                top_p=None,
                num_beams=1,
            )
            for it in items
        ]

        outs = []
        for idx, sub in enumerate(_chunks(args_list, auto_config.VLM_BATCH_SIZE)):
            # Nếu có hàm predict_batch_true_batched(...) của bạn, thay ở đây:
            # sub_outs = predict_batch_true_batched(sub, tokenizer=..., model=..., image_processor=..., ...)
            logger.info(f"Processing sub-batch: {idx} of size {len(sub)}")
            sub_outs = predict_batch_true_batched(sub, tokenizer, model, image_processor, start_idx=idx * auto_config.VLM_BATCH_SIZE)
            outs.extend(sub_outs)

        finish_ts = time.time()
        metrics = {
            "enqueue_ts": enqueue_ts,
            "start_ts": start_ts,
            "finish_ts": finish_ts,
            "queue_wait_ms": round((start_ts - enqueue_ts) * 1000, 2) if enqueue_ts else None,
            "exec_ms": round((finish_ts - start_ts) * 1000, 2),
            "total_ms": round((finish_ts - enqueue_ts) * 1000, 2) if enqueue_ts else None,
            "worker": worker,
            "queue": queue,
            "task_id": self.request.id,
        }
        logger.info("[METRICS][BATCH] %s", metrics)
        # DON'T update_state to SUCCESS - it overwrites the return value!
        # Celery automatically sets SUCCESS state when the task returns
        # try:
        #     self.update_state(
        #         state="SUCCESS",
        #         meta={"stage": "finished", **{k: v for k, v in metrics.items() if k.endswith('_ms')}}
        #     )
        # except Exception:
        #     pass

        # Dọn file tạm từng item (an toàn)
        for it in items:
            try:
                path = os.path.abspath(it["temp_path"])
                # chỉ xóa file trong thư mục tạm của bạn nếu muốn giới hạn phạm vi
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

        # Chuẩn hóa output cho client:
        norm_outs = [
            {
                "prompt": o.get("prompt"),
                "rag_answer": o.get("rag_answer"),
                "text": o.get("text"),
                "transcript": o.get("transcript"),
                "raw": o,
            }
            for o in outs
        ]

        return {"success": True, "data": norm_outs, "metrics": metrics}

    except Exception:
        finish_ts = time.time()
        metrics = {
            "enqueue_ts": enqueue_ts,
            "start_ts": start_ts,
            "finish_ts": finish_ts,
            "queue_wait_ms": round((start_ts - enqueue_ts) * 1000, 2) if enqueue_ts else None,
            "exec_ms": round((finish_ts - start_ts) * 1000, 2),
            "total_ms": round((finish_ts - enqueue_ts) * 1000, 2) if enqueue_ts else None,
            "worker": worker,
            "queue": queue,
            "task_id": self.request.id,
        }
        logger.exception("[METRICS][BATCH][ERROR] %s", metrics)
        # dọn file tạm khi lỗi
        for it in items:
            try:
                path = os.path.abspath(it["temp_path"])
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass
        raise
