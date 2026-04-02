"""
X-CLIP Provider for Video Event Detection and Fine-tuning.

Implements zero-shot video classification and contrastive fine-tuning
using Microsoft's X-CLIP (microsoft/xclip-base-patch32).

X-CLIP extends CLIP with a video-specific multi-frame token (MIT) and
temporal attention, making it natively suited for understanding short
video clips rather than single images.

Supports:
  - Zero-shot inference against arbitrary text labels
  - Batched chunk processing for long videos (configurable BATCH_SIZE)
  - Image input (replicates single frame to NUM_FRAMES)
  - Per-label temporal deduplication and event merging
  - Optional fine-tuned model path or HuggingFace hub model name
  - Contrastive fine-tuning via XCLIPTrainer (mirrors train_fixed.py)
"""

import json
import logging
import os
from collections import deque
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

from ...base.providers import BaseProvider, ModelConfig, ModelHandle
from ...base.providers import ProviderRegistry, ProviderMetadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default event taxonomy (mirrors run.py from SportVibe X-CLIP)
# ---------------------------------------------------------------------------

_RALLY_GROUP: List[str] = [
    "long rally",
    "short intense rally",
    "baseline rally",
    "fast net exchange",
    "Pickleball serve",
]

_SERVE_TEMPLATES: List[str] = [
    "Pickleball serve",
    "a photo of a player serving in pickleball",
    "underhand serve action",
    "pickleball player hitting a serve",
]

_ACTION_GROUP: List[str] = [
    "Pickleball serve",
    "underhand serve",
    "start of play serve",
    "player sprinting",
    "defensive scramble",
    "powerful serve",
    "volley winner",
    "overhead smash",
    "drop shot",
    "lob shot",
    "passing shot",
    "backhand winner",
    "forehand winner",
    "player celebration",
    "crowd cheering",
    "break point rally",
    "match point conversion",
    "deuce point",
    "momentum shift",
]

# De-duplicate while preserving insertion order
_ALL_LABELS_DEFAULT: List[str] = list(
    dict.fromkeys(_RALLY_GROUP + _ACTION_GROUP + _SERVE_TEMPLATES)
)


# ---------------------------------------------------------------------------
# CLIP normalisation constants (same as OpenAI CLIP / X-CLIP pre-training)
# ---------------------------------------------------------------------------
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class XCLIPProvider(BaseProvider):
    """
    Provider for Microsoft X-CLIP video event detection.

    Features:
    - Zero-shot classification using X-CLIP (HuggingFace transformers)
    - Batched chunk processing — configurable batch size
    - Temporal frame sampling (sliding window of NUM_FRAMES)
    - Per-label event de-duplication with configurable gap tolerance
    - Supports both video and image inputs
    - Fine-tuned or hub model paths supported via ModelConfig.model_path /
      ModelConfig.model_name

    Configuration via ModelConfig.extra_params:
        num_frames (int):      Frames per chunk, default 8
        sample_interval (int): Frame stride, default 3
        batch_size (int):      Chunks per inference call, default 32
        confidence_threshold (float): Minimum score, default 0.25
        max_frame_gap (int):   Max gap (frames) to merge events, default 5
        min_event_frames (int): Min chunks to keep an event, default 2
        action_confidence_boost (float): Multiplier for action labels, default 1.0
        labels (list[str]):    Custom label list, default _ALL_LABELS_DEFAULT
    """

    DEFAULT_NUM_FRAMES = 8
    DEFAULT_SAMPLE_INTERVAL = 3
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_CONFIDENCE_THRESHOLD = 0.25
    DEFAULT_MAX_FRAME_GAP = 5
    DEFAULT_MIN_EVENT_FRAMES = 2
    DEFAULT_ACTION_CONFIDENCE_BOOST = 1.0

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__(config)
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self._model: Optional[XCLIPModel] = None
        self._processor: Optional[XCLIPProcessor] = None
        self._text_features: Optional[torch.Tensor] = None
        self._labels: List[str] = []
        self._action_group: List[str] = []

    # ------------------------------------------------------------------
    # BaseProvider contract
    # ------------------------------------------------------------------

    def load_model(self, config: ModelConfig) -> ModelHandle:
        """
        Load X-CLIP model and pre-compute text embeddings.

        Args:
            config: Must set model_path (local dir) or model_name (HF hub).
                    extra_params.labels may override the default label list.
        """
        model_id = config.model_path or config.model_name
        logger.info(f"[XCLIPProvider] Loading model from: {model_id}")

        self._processor = XCLIPProcessor.from_pretrained(model_id)
        self._model = XCLIPModel.from_pretrained(model_id).to(self.device)
        self._model.eval()

        # Resolve label list
        self._labels = config.extra_params.get("labels", _ALL_LABELS_DEFAULT)
        self._action_group = config.extra_params.get("action_group", _ACTION_GROUP)

        # Pre-compute text features (done once, reused for every chunk)
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
        """Release model memory."""
        if self._model is not None:
            del self._model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._model = None
        self._processor = None
        self._text_features = None

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    def detect_events(
        self,
        input_path: str,
        output_csv_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect events in a video or image file.

        Args:
            input_path:       Path to an .mp4 video or an image file.
            output_csv_path:  If set, write results to this CSV path.

        Returns:
            List of event dicts with keys:
              timestamp, end_timestamp, event_label, confidence
        """
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

    # ------------------------------------------------------------------
    # Private helpers — pre-compute
    # ------------------------------------------------------------------

    def _precompute_text_features(self) -> None:
        """Encode the label list into unit-norm text embeddings (once)."""
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
        self._text_features = text_features  # (num_labels, D)

    # ------------------------------------------------------------------
    # Private helpers — inference
    # ------------------------------------------------------------------

    def _read_config(self, key: str, default: Any) -> Any:
        """Read a parameter from self.config.extra_params with a fallback."""
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

    def _preprocess_frame_chunk(
        self, frames: List[np.ndarray]
    ) -> torch.Tensor:
        """
        Convert a list of BGR/RGB uint8 frames into a normalised tensor.

        Returns:
            Tensor of shape (num_frames, 3, 224, 224) on CPU.
        """
        arr = np.stack(frames, axis=0)  # (F, H, W, 3)
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0  # (F,3,H,W)
        t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
        mean = _CLIP_MEAN
        std = _CLIP_STD
        t = (t - mean) / std
        return t  # (F, 3, 224, 224) — still on CPU

    def _infer_batch(
        self, stacked: torch.Tensor
    ) -> np.ndarray:
        """
        Run X-CLIP on a batch of video chunks.

        Args:
            stacked: Tensor shape (B, F, 3, 224, 224) on GPU.

        Returns:
            numpy array of shape (B, num_labels) — softmax scores.
        """
        with torch.no_grad():
            video_features = self._model.get_video_features(
                pixel_values=stacked
            )  # (B, D)
            video_features = video_features / video_features.norm(
                dim=-1, keepdim=True
            )
            similarity = (
                100.0 * video_features @ self._text_features.T
            ).softmax(dim=-1)  # (B, num_labels)
        return similarity.cpu().float().numpy()

    def _process_batch_predictions(
        self,
        batch_probs: np.ndarray,
        batch_meta: List[Dict[str, Any]],
        raw_detections: List[Dict[str, Any]],
        confidence_threshold: float,
        action_confidence_boost: float,
    ) -> None:
        """Append per-chunk detections above threshold to raw_detections."""
        for probs, meta in zip(batch_probs, batch_meta):
            for idx, label in enumerate(self._labels):
                score = float(probs[idx])
                if label in self._action_group:
                    score = min(score * action_confidence_boost, 0.99)
                if score >= confidence_threshold:
                    raw_detections.append(
                        {
                            "frame_id": meta["frame_id"],
                            "timestamp": meta["timestamp"],
                            "event_label": label,
                            "confidence": score,
                        }
                    )

    def _run_inference(
        self, input_path: str, file_type: str
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Dispatch inference to image or video handler."""
        if file_type == "image":
            return self._run_image_inference(input_path)
        return self._run_video_inference(input_path)

    def _run_image_inference(
        self, image_path: str
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Replicate a single image to NUM_FRAMES and run inference."""
        num_frames = self._read_config("num_frames", self.DEFAULT_NUM_FRAMES)
        confidence_threshold = self._read_config(
            "confidence_threshold", self.DEFAULT_CONFIDENCE_THRESHOLD
        )
        action_boost = self._read_config(
            "action_confidence_boost", self.DEFAULT_ACTION_CONFIDENCE_BOOST
        )

        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        frames = [img_rgb] * num_frames
        chunk_tensor = self._preprocess_frame_chunk(frames)          # (F,3,224,224)
        batch = chunk_tensor.unsqueeze(0).to(self.device)            # (1,F,3,224,224)
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

    def _run_video_inference(
        self, video_path: str
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Sliding-window frame sampling + batched X-CLIP inference."""
        num_frames = self._read_config("num_frames", self.DEFAULT_NUM_FRAMES)
        sample_interval = self._read_config(
            "sample_interval", self.DEFAULT_SAMPLE_INTERVAL
        )
        batch_size = self._read_config("batch_size", self.DEFAULT_BATCH_SIZE)
        confidence_threshold = self._read_config(
            "confidence_threshold", self.DEFAULT_CONFIDENCE_THRESHOLD
        )
        action_boost = self._read_config(
            "action_confidence_boost", self.DEFAULT_ACTION_CONFIDENCE_BOOST
        )

        cap = cv2.VideoCapture(video_path)
        fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            f"[XCLIPProvider] Processing {total_frames} frames "
            f"(fps={fps:.1f}, interval={sample_interval}, batch={batch_size})"
        )

        raw_detections: List[Dict] = []
        frame_buffer: deque = deque(maxlen=num_frames)
        batch_tensors: List[torch.Tensor] = []
        batch_meta: List[Dict] = []
        frame_count = 0

        pbar = tqdm(total=total_frames, desc="X-CLIP inference")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_buffer.append(frame_rgb)

            # Emit a chunk every `sample_interval` frames once the buffer is full
            if len(frame_buffer) == num_frames and frame_count % sample_interval == 0:
                chunk_tensor = self._preprocess_frame_chunk(list(frame_buffer))
                batch_tensors.append(chunk_tensor)
                batch_meta.append(
                    {"frame_id": frame_count, "timestamp": frame_count / fps}
                )

                # Flush when batch is full
                if len(batch_tensors) == batch_size:
                    stacked = torch.stack(batch_tensors).to(self.device)  # (B,F,3,224,224)
                    probs = self._infer_batch(stacked)
                    self._process_batch_predictions(
                        probs, batch_meta, raw_detections,
                        confidence_threshold, action_boost,
                    )
                    batch_tensors.clear()
                    batch_meta.clear()

            frame_count += 1
            pbar.update(1)

        # Flush remaining chunks
        if batch_tensors:
            stacked = torch.stack(batch_tensors).to(self.device)
            probs = self._infer_batch(stacked)
            self._process_batch_predictions(
                probs, batch_meta, raw_detections,
                confidence_threshold, action_boost,
            )

        cap.release()
        pbar.close()
        logger.info(f"[XCLIPProvider] Raw detections: {len(raw_detections)}")
        return raw_detections, fps

    # ------------------------------------------------------------------
    # Private helpers — post-processing
    # ------------------------------------------------------------------

    def _deduplicate_events(
        self, raw_data: List[Dict], file_type: str
    ) -> List[Dict]:
        """
        Merge nearby detections of the same label into a single event.

        For images: no merging, just sort by confidence.
        For videos:  group by label, merge frames within MAX_FRAME_GAP,
                     filter out short events (< MIN_EVENT_FRAMES).
        """
        if not raw_data:
            return []

        if file_type == "image":
            for item in raw_data:
                item["end_timestamp"] = item["timestamp"]
            return sorted(raw_data, key=lambda x: x["confidence"], reverse=True)

        max_gap = self._read_config("max_frame_gap", self.DEFAULT_MAX_FRAME_GAP)
        min_frames = self._read_config(
            "min_event_frames", self.DEFAULT_MIN_EVENT_FRAMES
        )

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
                        "event_label": label,
                        "start_frame": fid,
                        "end_frame": fid,
                        "timestamp": ts,
                        "end_timestamp": ts,
                        "sum_conf": conf,
                        "count": 1,
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
                        "event_label": label,
                        "start_frame": fid,
                        "end_frame": fid,
                        "timestamp": ts,
                        "end_timestamp": ts,
                        "sum_conf": conf,
                        "count": 1,
                    }

            if current and current["count"] >= min_frames:
                current["confidence"] = current["sum_conf"] / current["count"]
                final_events.append(current)

        # Strip internal tracking fields
        for ev in final_events:
            for k in ("sum_conf", "count", "start_frame", "end_frame"):
                ev.pop(k, None)

        return sorted(final_events, key=lambda x: x["timestamp"])

    def _save_to_csv(self, events: List[Dict], output_path: str) -> None:
        """Write events to a CSV file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df = pd.DataFrame(events)
        df["timestamp"] = df["timestamp"].map(lambda x: f"{float(x):.3f}")
        df["end_timestamp"] = df["end_timestamp"].map(lambda x: f"{float(x):.3f}")
        df["confidence"] = df["confidence"].map(lambda x: f"{float(x):.3f}")
        df = df[["timestamp", "end_timestamp", "event_label", "confidence"]]
        df.to_csv(output_path, index=False)
        logger.info(f"[XCLIPProvider] Saved {len(events)} events → {output_path}")


# ---------------------------------------------------------------------------
# Dataset for fine-tuning (mirrors train_fixed.py: PickleballXCLIPDataset)
# ---------------------------------------------------------------------------

class XCLIPVideoDataset(Dataset):
    """
    PyTorch Dataset for X-CLIP contrastive fine-tuning.

    Expects a JSON file with structure::

        {"sentences": [{"video_id": "v001", "caption": "overhead smash"}, ...]}

    Args:
        json_file:   Path to caption JSON.
        video_dir:   Directory containing ``<video_id>.mp4`` files.
        num_frames:  Frames sampled per clip (default 8).
    """

    _MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    _STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    def __init__(self, json_file: str, video_dir: str, num_frames: int = 8) -> None:
        self.video_dir = video_dir
        self.num_frames = num_frames

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
        """Sample ``num_frames`` uniformly and return tensor (F, C, H, W)."""
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
        return torch.stack(frames, dim=0)  # (F, C, H, W)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        item = self.records[idx]
        video_path = os.path.join(self.video_dir, f"{item['video_id']}.mp4")
        try:
            frames = self._load_video(video_path)          # (F, C, H, W)
            frames = F.interpolate(
                frames, size=(224, 224), mode="bilinear", align_corners=False
            )
            frames = (frames - self._MEAN) / self._STD
        except Exception as exc:
            logger.warning(f"[XCLIPVideoDataset] Skip {video_path}: {exc}")
            return None
        return {"video": frames, "text": item["caption"], "video_id": item["video_id"]}


def _xclip_collate_fn(
    batch: List[Optional[Dict]],
) -> Optional[Tuple[torch.Tensor, List[str], List[str]]]:
    """Collate, drop None items. Returns None when the whole batch is empty."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    videos = torch.stack([b["video"] for b in batch], dim=0)  # (B, F, C, H, W)
    texts = [b["text"] for b in batch]
    video_ids = [b["video_id"] for b in batch]
    return videos, texts, video_ids


# ---------------------------------------------------------------------------
# Trainer (mirrors train_fixed.py)
# ---------------------------------------------------------------------------

@dataclass
class XCLIPTrainConfig:
    """
    Training hyper-parameters for :class:`XCLIPTrainer`.

    All fields can be set via ``ModelConfig.extra_params`` when creating the
    trainer through the framework (see :class:`XCLIPTrainer` docstring).
    """
    json_file: str = "dataset_pickleball/pickleball_caption.json"
    video_dir: str = "dataset_pickleball/videos"
    save_dir: str = "pickleball_xclip_model"
    num_frames: int = 8
    batch_size: int = 16
    num_epochs: int = 10
    learning_rate: float = 1e-5
    grad_accum_steps: int = 4
    num_workers: int = 4
    # Layers to unfreeze during training
    trainable_keywords: List[str] = field(default_factory=lambda: [
        "visual_projection",
        "text_projection",
        "logit_scale",
        "prompts_generator",
        "mit",
    ])


class XCLIPTrainer:
    """
    Fine-tunes an X-CLIP model with contrastive (CLIP-style) loss.

    Wraps the logic from ``train_fixed.py`` into the infimum-core framework.
    Instantiate directly or through ``ModelConfig``.

    Example usage::

        from core.ai.vlm import XCLIPTrainer
        from core.ai.base.providers import ModelConfig

        config = ModelConfig(
            model_type="vlm",
            provider="xclip",
            model_name="microsoft/xclip-base-patch32",
            extra_params={
                "json_file": "dataset_pickleball/pickleball_caption.json",
                "video_dir": "dataset_pickleball/videos",
                "save_dir":  "pickleball_xclip_model",
                "num_epochs": 10,
                "batch_size": 16,
                "learning_rate": 1e-5,
            },
        )
        trainer = XCLIPTrainer(config)
        trainer.train()

    Training strategy:
        - Freeze the entire backbone.
        - Unfreeze only projection heads, logit_scale, prompts_generator, MIT.
        - Symmetric contrastive loss: (loss_video→text + loss_text→video) / 2.
        - Gradient accumulation with gradient norm clipping.
        - Skip batches containing duplicate ``video_id`` to avoid contradictions.
    """

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._train_cfg = self._build_train_config(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[XCLIPTrainer] Device: {self.device}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop and save the fine-tuned model."""
        cfg = self._train_cfg
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
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=_xclip_collate_fn,
            drop_last=True,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        trainable = [p for p in model.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in trainable)
        logger.info(f"[XCLIPTrainer] Trainable parameters: {n_params:,}")

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
                    logger.debug(f"Epoch {epoch+1} | step {step+1}: empty batch, skip.")
                    continue

                videos, texts, video_ids = batch

                # Skip batch with duplicate video_ids → contradiction in contrastive loss
                if len(set(video_ids)) < len(video_ids):
                    logger.debug(f"Epoch {epoch+1} | step {step+1}: duplicate video_id, skip.")
                    continue

                text_inputs = processor(
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77,
                )
                text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
                text_inputs["pixel_values"] = videos.to(self.device, non_blocking=True)

                outputs = model(**text_inputs)
                logits_video = outputs.logits_per_video   # (B, B)
                logits_text  = outputs.logits_per_text    # (B, B)

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
                logger.info(
                    f"Epoch [{epoch+1}/{cfg.num_epochs}] "
                    f"step [{step+1}/{len(dataloader)}] "
                    f"loss={loss.item():.4f}"
                )

            # Flush remaining gradients at end of epoch
            if valid_steps > 0 and valid_steps % cfg.grad_accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            avg = total_loss / max(1, valid_steps)
            logger.info(
                f"[XCLIPTrainer] 🔥 Epoch {epoch+1} done — "
                f"avg_loss={avg:.4f}, valid_steps={valid_steps}"
            )

        # Save fine-tuned model
        os.makedirs(cfg.save_dir, exist_ok=True)
        model.save_pretrained(cfg.save_dir)
        processor.save_pretrained(cfg.save_dir)
        logger.info(f"[XCLIPTrainer] ✅ Model saved → {cfg.save_dir}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _freeze_backbone(
        model: XCLIPModel, trainable_keywords: List[str]
    ) -> None:
        """Freeze all params then re-enable the projection/MIT layers."""
        for param in model.parameters():
            param.requires_grad = False

        for name, param in model.named_parameters():
            if any(kw in name for kw in trainable_keywords):
                param.requires_grad = True

    @staticmethod
    def _build_train_config(config: ModelConfig) -> XCLIPTrainConfig:
        """Build XCLIPTrainConfig from ModelConfig.extra_params."""
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


# ---------------------------------------------------------------------------
# Register provider (auto-registered at import time)
# ---------------------------------------------------------------------------
ProviderRegistry.register(
    model_type="vlm",
    provider_name="xclip",
    provider_class=XCLIPProvider,
    metadata=ProviderMetadata(
        model_type="vlm",
        provider_name="xclip",
        capabilities={
            "video_classification",
            "event_detection",
            "zero_shot",
            "temporal_reasoning",
            "fine_tuning",
        },
        description=(
            "Microsoft X-CLIP (xclip-base-patch32) for video event detection. "
            "Supports zero-shot and fine-tuned inference on video clips. "
            "Use XCLIPTrainer for contrastive fine-tuning."
        ),
        version="0.1.1",
        requirements=["torch", "transformers>=4.30.0", "opencv-python", "pandas", "tqdm"],
    ),
)
