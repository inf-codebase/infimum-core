# src/queue/persist_analysis_tasks.py
from typing import Dict, Any, Optional
from celery import states
from celery.utils.log import get_task_logger

from core.networking.queue.celery_app import celery_app
from features.ai_models.entities.vlm_analysis import VLMAnalysis
from features.ai_models.repositories.vlm_analysis_repository import VLMAnalysisRepository

logger = get_task_logger(__name__)

@celery_app.task(
    name="fastvlm.persist_vlm_analysis",
    bind=True, acks_late=True, track_started=True,
    autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 3},
)
def persist_vlm_analysis(self, inference_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nhận output từ fastvlm.analyze_image rồi:
      - Lưu VLMAnalysis (consolidated with metadata)
      - Sinh embedding & đẩy Milvus
    """
    self.update_state(state=states.STARTED, meta={"step": "persist"})

    result = inference_payload.get("result", {})
    meta   = inference_payload.get("meta", {})

    transcript = result.get("transcript")
    video_metadata = result.get("video_metadata")
    details_raw = result.get("details")

    # Consolidate all metadata into a single JSON field
    consolidated_metadata = {
        "video_metadata": video_metadata,
        "details": details_raw,
    }

    repo = VLMAnalysisRepository()

    try:
        model = VLMAnalysis(
            video_segment_id=meta["video_segment_id"],
            external_video_id=meta.get("external_video_id"),
            frame_index=meta.get("frame_index") or 0,
            frame_timestamp_seconds=meta.get("frame_timestamp_seconds"),
            prompt=meta.get("prompt"),
            transcript=transcript,
            analysis_metadata=consolidated_metadata,
            success=True,
        )

        created = repo.create(model)  # ✅ insert + embedding + Milvus
        return {
            "analysis_id": created.id,
            "video_segment_id": created.video_segment_id,
            "success": True,
        }
    except Exception as e:
        # ghi log failure (không sinh embedding)
        failed_metadata = {
            "video_metadata": video_metadata,
            "error": str(e),
        }
        failed = repo.log_failure(
            error_details=str(e),
            video_segment_id=meta.get("video_segment_id"),
            external_video_id=meta.get("external_video_id"),
            frame_index=meta.get("frame_index"),
            frame_timestamp_seconds=meta.get("frame_timestamp_seconds"),
            prompt=meta.get("prompt"),
            transcript=None,
            analysis_metadata=failed_metadata,
        )
        return {"analysis_id": failed.id, "success": False, "error": str(e)}
