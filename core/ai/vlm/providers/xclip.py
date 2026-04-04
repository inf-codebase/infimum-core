"""
X-CLIP Provider for Video Event Detection and Fine-tuning.

Implements zero-shot video classification and contrastive fine-tuning
using Microsoft's X-CLIP (microsoft/xclip-base-patch32).

X-CLIP extends CLIP with a video-specific multi-frame token (MIT) and
temporal attention, making it natively suited for understanding short
video clips rather than single images.

Supports:
  - Zero-shot inference against arbitrary text labels (Provided by Client)
  - Batched chunk processing for long videos (configurable BATCH_SIZE)
  - Image input (replicates single frame to NUM_FRAMES)
  - Per-label temporal deduplication and event merging
  - Optional fine-tuned model path or HuggingFace hub model name
  - Contrastive fine-tuning via XCLIPTrainer
"""

import json
import logging
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import XCLIPModel, XCLIPProcessor
from core.utils import auto_config

from ...base.providers import BaseProvider, ModelConfig, ModelHandle
from ...base.providers import ProviderRegistry, ProviderMetadata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLIP normalisation constants (same as OpenAI CLIP / X-CLIP pre-training)
# ---------------------------------------------------------------------------
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class XCLIPProvider(BaseProvider):
    """
    Provider for Microsoft X-CLIP video event detection.

    Configuration via ModelConfig.extra_params:
        labels (list[str]):    [REQUIRED] Custom label list from client.
        action_group (list[str]): [OPTIONAL] Labels that get confidence boost.
        num_frames (int):      Frames per chunk, default 8
        sample_interval (int): Frame stride, default 3
        batch_size (int):      Chunks per inference call, default 32
        confidence_threshold (float): Minimum score, default 0.25
        max_frame_gap (int):   Max gap (frames) to merge events, default 5
        min_event_frames (int): Min chunks to keep an event, default 2
        action_confidence_boost (float): Multiplier for action labels, default 1.0
    """

    DEFAULT_NUM_FRAMES = int(auto_config.DEFAULT_NUM_FRAMES) if auto_config.DEFAULT_NUM_FRAMES else 8
    DEFAULT_SAMPLE_INTERVAL = int(auto_config.DEFAULT_SAMPLE_INTERVAL) if auto_config.DEFAULT_SAMPLE_INTERVAL else 3
    DEFAULT_BATCH_SIZE = int(auto_config.DEFAULT_BATCH_SIZE) if auto_config.DEFAULT_BATCH_SIZE else 32
    DEFAULT_CONFIDENCE_THRESHOLD = float(auto_config.DEFAULT_CONFIDENCE_THRESHOLD) if auto_config.DEFAULT_CONFIDENCE_THRESHOLD else 0.25
    DEFAULT_MAX_FRAME_GAP = int(auto_config.DEFAULT_MAX_FRAME_GAP) if auto_config.DEFAULT_MAX_FRAME_GAP else 5
    DEFAULT_MIN_EVENT_FRAMES = int(auto_config.DEFAULT_MIN_EVENT_FRAMES) if auto_config.DEFAULT_MIN_EVENT_FRAMES else 2
    DEFAULT_ACTION_CONFIDENCE_BOOST = float(auto_config.DEFAULT_ACTION_CONFIDENCE_BOOST) if auto_config.DEFAULT_ACTION_CONFIDENCE_BOOST else 1.0

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__(config)
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._model: Optional[XCLIPModel] = None
        self._processor: Optional[XCLIPProcessor] = None
        self._text_features: Optional[torch.Tensor] = None
        self._labels: List[str] = []
        self._action_group: List[str] = []

        # Parallel CPU preprocessing
        n_workers = max(2, (os.cpu_count() or 4) - 1)
        self._preprocess_executor = ThreadPoolExecutor(max_workers=n_workers)
        logger.debug(f"[XCLIPProvider] Preprocess workers: {n_workers}")

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state.pop("_preprocess_executor", None)
        return state

    def __setstate__(self, state: Dict) -> None:
        self.__dict__.update(state)
        n_workers = max(2, (os.cpu_count() or 4) - 1)
        self._preprocess_executor = ThreadPoolExecutor(max_workers=n_workers)

    # ------------------------------------------------------------------
    # BaseProvider contract
    # ------------------------------------------------------------------

    def load_model(self, config: ModelConfig) -> ModelHandle:
        model_id = config.model_path or config.model_name
        logger.info(f"[XCLIPProvider] Loading model from: {model_id}")

        self._processor = XCLIPProcessor.from_pretrained(model_id)
        self._model = XCLIPModel.from_pretrained(model_id).to(self.device)
        self._model.eval()

        # [UPDATE] Yêu cầu client phải truyền danh sách labels
        self._labels = config.extra_params.get("labels")
        if not self._labels or not isinstance(self._labels, list):
            raise ValueError(
                "[XCLIPProvider] 'labels' must be provided as a non-empty list "
                "in config.extra_params by the client."
            )

        self._action_group = config.extra_params.get("action_group", [])

        # Pre-compute text features
        self._precompute_text_features()

        logger.info(
            f"[XCLIPProvider] Ready — {len(self._labels)} labels on {self.device}"
        )

        return ModelHandle(
            model={
                "model": self._model,
                "processor": self._processor,
                "text_features": self._text_features,
                "labels": self._labels,
            },
            config=config,
            metadata={
                "name": "X-CLIP",
                "model_id": model_id,
                "device": self.device,
                "labels_count": len(self._labels),
            },
        )

    def unload_model(self, handle: ModelHandle) -> None:
        if self._model is not None:
            del self._model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._model = None
        self._processor = None
        self._text_features = None
        if hasattr(self, "_preprocess_executor"):
            self._preprocess_executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Các hàm inference giữ nguyên logic
    # ------------------------------------------------------------------

    def detect_events(
        self,
        input_path: str,
        output_csv_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        file_type = self._detect_file_type(input_path)
        raw_detections, fps = self._run_inference(input_path, file_type)
        final_events = self._deduplicate_events(raw_detections, file_type)

        if output_csv_path and final_events:
            self._save_to_csv(final_events, output_csv_path)

        return final_events

    def _precompute_text_features(self) -> None:
        logger.info(
            f"[XCLIPProvider] Pre-computing text features for {len(self._labels)} labels …"
        )
        with torch.no_grad():
            text_inputs = self._processor(
                text=self._labels,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            text_features = self._model.get_text_features(**text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self._text_features = text_features

    def _read_config(self, key: str, default: Any) -> Any:
        if self.config and self.config.extra_params:
            return self.config.extra_params.get(key, default)
        return default

    @staticmethod
    def _detect_file_type(path: str) -> str:
        import mimetypes
        mime, _ = mimetypes.guess_type(path)
        if mime:
            if mime.startswith("image"):
                return "image"
            if mime.startswith("video"):
                return "video"
        ext = os.path.splitext(path)[1].lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return "image"
        if ext in {".mp4", ".avi", ".mov", ".mkv"}:
            return "video"
        return "unknown"

    def _preprocess_frame_chunk(self, frames: List[np.ndarray]) -> torch.Tensor:
        arr = np.stack(frames, axis=0)
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
        t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
        t = (t - _CLIP_MEAN) / _CLIP_STD
        return t

    def _preprocess_chunks_parallel(self, chunks: List[List[np.ndarray]]) -> List[torch.Tensor]:
        return list(self._preprocess_executor.map(self._preprocess_frame_chunk, chunks))

    def _infer_batch(self, stacked: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            video_features = self._model.get_video_features(pixel_values=stacked)
            video_features = video_features / video_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * video_features @ self._text_features.T).softmax(dim=-1)
        return similarity.cpu().float().numpy()

    def _process_batch_predictions(
        self,
        batch_probs: np.ndarray,
        batch_meta: List[Dict[str, Any]],
        raw_detections: List[Dict[str, Any]],
        confidence_threshold: float,
        action_confidence_boost: float,
    ) -> None:
        for probs, meta in zip(batch_probs, batch_meta):
            for idx, label in enumerate(self._labels):
                score = float(probs[idx])
                if label in self._action_group:
                    score = max(min(score * action_confidence_boost, 0.99), 0.0)
                if score >= confidence_threshold:
                    raw_detections.append({
                        "frame_id": meta["frame_id"],
                        "timestamp": meta["timestamp"],
                        "event_label": label,
                        "confidence": score,
                    })

    def _run_inference(self, input_path: str, file_type: str) -> Tuple[List[Dict[str, Any]], float]:
        if file_type == "image":
            return self._run_image_inference(input_path)
        return self._run_video_inference(input_path)

    def _run_image_inference(self, image_path: str) -> Tuple[List[Dict[str, Any]], float]:
        num_frames = self._read_config("num_frames", self.DEFAULT_NUM_FRAMES)
        confidence_threshold = self._read_config("confidence_threshold", self.DEFAULT_CONFIDENCE_THRESHOLD)
        action_boost = self._read_config("action_confidence_boost", self.DEFAULT_ACTION_CONFIDENCE_BOOST)

        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        frames = [img_rgb] * num_frames
        chunk_tensor = self._preprocess_frame_chunk(frames)
        batch = chunk_tensor.unsqueeze(0).to(self.device)
        batch_probs = self._infer_batch(batch)

        raw_detections: List[Dict] = []
        self._process_batch_predictions(
            batch_probs,
            [{"frame_id": 0, "timestamp": 0.0}],
            raw_detections,
            confidence_threshold,
            action_boost,
        )
        return raw_detections, 1.0

    def _run_video_inference(self, video_path: str) -> Tuple[List[Dict[str, Any]], float]:
        num_frames = self._read_config("num_frames", self.DEFAULT_NUM_FRAMES)
        sample_interval = self._read_config("sample_interval", self.DEFAULT_SAMPLE_INTERVAL)
        batch_size = self._read_config("batch_size", self.DEFAULT_BATCH_SIZE)
        confidence_threshold = self._read_config("confidence_threshold", self.DEFAULT_CONFIDENCE_THRESHOLD)
        action_boost = self._read_config("action_confidence_boost", self.DEFAULT_ACTION_CONFIDENCE_BOOST)

        cap = cv2.VideoCapture(video_path)
        fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"[XCLIPProvider] Processing {total_frames} frames "
            f"(fps={fps:.1f}, interval={sample_interval}, batch={batch_size})"
        )

        raw_detections: List[Dict] = []
        frame_buffer: deque = deque(maxlen=num_frames)
        pending_chunks: List[List[np.ndarray]] = []
        batch_meta: List[Dict] = []
        frame_count = 0

        pbar = tqdm(total=total_frames, desc="X-CLIP inference")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(frame_rgb)

            if len(frame_buffer) == num_frames and (frame_count - num_frames + 1) % sample_interval == 0:
                pending_chunks.append(list(frame_buffer))
                batch_meta.append({"frame_id": frame_count, "timestamp": frame_count / fps})

                if len(pending_chunks) == batch_size:
                    tensors = self._preprocess_chunks_parallel(pending_chunks)
                    stacked = torch.stack(tensors).to(self.device)
                    probs = self._infer_batch(stacked)
                    self._process_batch_predictions(
                        probs, batch_meta, raw_detections,
                        confidence_threshold, action_boost,
                    )
                    pending_chunks.clear()
                    batch_meta.clear()

            frame_count += 1
            pbar.update(1)

        if pending_chunks:
            tensors = self._preprocess_chunks_parallel(pending_chunks)
            stacked = torch.stack(tensors).to(self.device)
            probs = self._infer_batch(stacked)
            self._process_batch_predictions(
                probs, batch_meta, raw_detections,
                confidence_threshold, action_boost,
            )

        cap.release()
        pbar.close()
        logger.info(f"[XCLIPProvider] Raw detections: {len(raw_detections)}")
        return raw_detections, fps

    def _deduplicate_events(self, raw_data: List[Dict], file_type: str) -> List[Dict]:
        if not raw_data:
            return []

        if file_type == "image":
            for item in raw_data:
                item["end_timestamp"] = item["timestamp"]
            return sorted(raw_data, key=lambda x: x["confidence"], reverse=True)

        max_gap = self._read_config("max_frame_gap", self.DEFAULT_MAX_FRAME_GAP)
        min_frames = self._read_config("min_event_frames", self.DEFAULT_MIN_EVENT_FRAMES)

        df = pd.DataFrame(raw_data)
        final_events: List[Dict] = []

        for label, group in df.groupby("event_label"):
            group = group.sort_values("frame_id")
            current: Optional[Dict] = None

            for _, row in group.iterrows():
                fid = int(row["frame_id"])
                ts = float(row["timestamp"])
                conf = float(row["confidence"])

                if current is None:
                    current = {
                        "event_label": label, "start_frame": fid, "end_frame": fid,
                        "timestamp": ts, "end_timestamp": ts, "sum_conf": conf, "count": 1,
                    }
                elif fid - current["end_frame"] <= max_gap:
                    current["end_frame"] = fid
                    current["end_timestamp"] = ts
                    current["sum_conf"] += conf
                    current["count"] += 1
                else:
                    if current["count"] >= min_frames:
                        current["confidence"] = current["sum_conf"] / current["count"]
                        final_events.append(current)
                    current = {
                        "event_label": label, "start_frame": fid, "end_frame": fid,
                        "timestamp": ts, "end_timestamp": ts, "sum_conf": conf, "count": 1,
                    }

            if current and current["count"] >= min_frames:
                current["confidence"] = current["sum_conf"] / current["count"]
                final_events.append(current)

        for ev in final_events:
            for k in ("sum_conf", "count", "start_frame", "end_frame"):
                ev.pop(k, None)

        return sorted(final_events, key=lambda x: x["timestamp"])

    def _save_to_csv(self, events: List[Dict], output_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df = pd.DataFrame(events)
        df["timestamp"] = df["timestamp"].map(lambda x: f"{float(x):.3f}")
        df["end_timestamp"] = df["end_timestamp"].map(lambda x: f"{float(x):.3f}")
        df["confidence"] = df["confidence"].map(lambda x: f"{float(x):.3f}")
        df = df[["timestamp", "end_timestamp", "event_label", "confidence"]]
        df.to_csv(output_path, index=False)
        logger.info(f"[XCLIPProvider] Saved {len(events)} events → {output_path}")

# ---------------------------------------------------------------------------
# Dataset & Trainer (đã xóa hardcode thư mục Pickleball)
# ---------------------------------------------------------------------------

class XCLIPVideoDataset(Dataset):
    _MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    _STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    def __init__(self, json_file: str, video_dir: str, num_frames: int = 8) -> None:
        self.video_dir = video_dir
        self.num_frames = num_frames

        if not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON annotation file not found: {json_file}")

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = data.get("sentences", [])
        self.records = [
            item for item in records
            if os.path.exists(os.path.join(video_dir, f"{item['video_id']}.mp4"))
        ]
        logger.info(f"[XCLIPVideoDataset] {len(self.records)} valid videos found.")

    def __len__(self) -> int:
        return len(self.records)

    def _load_video(self, video_path: str) -> torch.Tensor:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise ValueError(f"Video has no frames: {video_path}")

        indices = torch.linspace(0, total - 1, self.num_frames).long().tolist()
        frames: List[torch.Tensor] = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                frame_t = frames[-1] if frames else torch.zeros(3, 1, 1)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frames.append(frame_t)

        cap.release()
        return torch.stack(frames, dim=0)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        item = self.records[idx]
        video_path = os.path.join(self.video_dir, f"{item['video_id']}.mp4")
        try:
            frames = self._load_video(video_path)
            frames = F.interpolate(frames, size=(224, 224), mode="bilinear", align_corners=False)
            frames = (frames - self._MEAN) / self._STD
        except Exception as exc:
            logger.warning(f"[XCLIPVideoDataset] Skip {video_path}: {exc}")
            return None
        return {"video": frames, "text": item["caption"], "video_id": item["video_id"]}

def _xclip_collate_fn(batch: List[Optional[Dict]]) -> Optional[Tuple[torch.Tensor, List[str], List[str]]]:
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    videos = torch.stack([b["video"] for b in batch], dim=0)
    texts = [b["text"] for b in batch]
    video_ids = [b["video_id"] for b in batch]
    return videos, texts, video_ids

@dataclass
class XCLIPTrainConfig:
    # Bắt buộc client cấu hình các params này
    json_file: str = ""
    video_dir: str = ""
    save_dir: str = "xclip_finetuned_model"
    num_frames: int = 8
    batch_size: int = 16
    num_epochs: int = 10
    learning_rate: float = 1e-5
    grad_accum_steps: int = 4
    num_workers: int = 4
    trainable_keywords: List[str] = field(default_factory=lambda: [
        "visual_projection", "text_projection", "logit_scale", "prompts_generator", "mit",
    ])

class XCLIPTrainer:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._train_cfg = self._build_train_config(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[XCLIPTrainer] Device: {self.device}")

    def train(self) -> None:
        cfg = self._train_cfg
        if not cfg.json_file or not cfg.video_dir:
            raise ValueError("[XCLIPTrainer] 'json_file' and 'video_dir' must be provided in extra_params.")

        model_id = self.config.model_path or self.config.model_name
        logger.info(f"[XCLIPTrainer] Loading model: {model_id}")
        processor = XCLIPProcessor.from_pretrained(model_id)
        model = XCLIPModel.from_pretrained(model_id).to(self.device)

        self._freeze_backbone(model, cfg.trainable_keywords)

        dataset = XCLIPVideoDataset(
            json_file=cfg.json_file,
            video_dir=cfg.video_dir,
            num_frames=cfg.num_frames,
        )
        dataloader = DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=True,
            collate_fn=_xclip_collate_fn, drop_last=True,
            num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
        )

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=cfg.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        logger.info("[XCLIPTrainer] 🚀 Training started ...")

        for epoch in range(cfg.num_epochs):
            total_loss = 0.0
            valid_steps = 0
            optimizer.zero_grad()

            for step, batch in enumerate(dataloader):
                if batch is None:
                    continue

                videos, texts, video_ids = batch
                if len(set(video_ids)) < len(video_ids):
                    continue

                text_inputs = processor(
                    text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77,
                )
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                text_inputs["pixel_values"] = videos.to(self.device, non_blocking=True)

                outputs = model(**text_inputs)
                logits_video = outputs.logits_per_video
                logits_text  = outputs.logits_per_text

                bs = logits_video.size(0)
                labels = torch.arange(bs, device=self.device)

                loss = (loss_fn(logits_video, labels) + loss_fn(logits_text, labels)) / 2
                (loss / cfg.grad_accum_steps).backward()

                if (valid_steps + 1) % cfg.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item()
                valid_steps += 1

            if valid_steps > 0 and valid_steps % cfg.grad_accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            avg = total_loss / max(1, valid_steps)
            logger.info(f"[XCLIPTrainer] 🔥 Epoch {epoch+1} done — avg_loss={avg:.4f}")

        os.makedirs(cfg.save_dir, exist_ok=True)
        model.save_pretrained(cfg.save_dir)
        processor.save_pretrained(cfg.save_dir)
        logger.info(f"[XCLIPTrainer] ✅ Model saved → {cfg.save_dir}")

    @staticmethod
    def _freeze_backbone(model: XCLIPModel, trainable_keywords: List[str]) -> None:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if any(kw in name for kw in trainable_keywords):
                param.requires_grad = True

    @staticmethod
    def _build_train_config(config: ModelConfig) -> XCLIPTrainConfig:
        ep = config.extra_params or {}
        return XCLIPTrainConfig(
            json_file=ep.get("json_file", XCLIPTrainConfig.json_file),
            video_dir=ep.get("video_dir", XCLIPTrainConfig.video_dir),
            save_dir=ep.get("save_dir", XCLIPTrainConfig.save_dir),
            num_frames=ep.get("num_frames", XCLIPTrainConfig.num_frames),
            batch_size=ep.get("batch_size", XCLIPTrainConfig.batch_size),
            num_epochs=ep.get("num_epochs", XCLIPTrainConfig.num_epochs),
            learning_rate=ep.get("learning_rate", XCLIPTrainConfig.learning_rate),
            grad_accum_steps=ep.get("grad_accum_steps", XCLIPTrainConfig.grad_accum_steps),
            num_workers=ep.get("num_workers", XCLIPTrainConfig.num_workers),
            trainable_keywords=ep.get(
                "trainable_keywords", XCLIPTrainConfig.__dataclass_fields__["trainable_keywords"].default_factory()
            ),
        )

ProviderRegistry.register(
    model_type="vlm",
    provider_name="xclip",
    provider_class=XCLIPProvider,
    metadata=ProviderMetadata(
        model_type="vlm", provider_name="xclip", version="0.1.1",
        capabilities={"video_classification", "event_detection", "zero_shot", "fine_tuning"},
        description="Microsoft X-CLIP (Dynamic Labels).",
        requirements=["torch", "transformers>=4.30.0", "opencv-python", "pandas", "tqdm"],
    ),
)
