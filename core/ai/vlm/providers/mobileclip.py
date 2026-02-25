"""
MobileCLIP Provider for Event Detection.

Implements zero-shot video classification using Apple's MobileCLIP2.
"""

import os
import sys
import logging
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import cv2

from PIL import Image
from tqdm import tqdm

from ...base.providers import BaseProvider, ModelConfig, ModelHandle
from ...base.providers import ProviderRegistry, ProviderMetadata

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None
    logger.warning("torch is not installed. Please install it via 'pip install torch'")
    

# Third-party imports
try:
    import open_clip
except ImportError:
    open_clip = None


class MobileClipProvider(BaseProvider):
    """
    Provider for MobileCLIP2 Event Detection.
    
    Features:
    - Zero-shot classification with 21 pickleball events
    - Smart selection logic (Action Priority vs Rally Summation)
    - Temporal deduplication
    - Confidence boosting for short actions
    """

    # --- CONSTANTS ---
    MODEL_NAME = "MobileCLIP2-S0"
    
    # Event Categories
    RALLY_GROUP = [
        "long rally", "short intense rally", "baseline rally", "fast net exchange"
    ]
    
    ACTION_GROUP = [
        "player sprinting", "defensive scramble", "powerful serve", "aggressive return",
        "volley winner", "overhead smash", "drop shot", "lob shot", "passing shot",
        "backhand winner", "forehand winner", "player celebration", "crowd cheering",
        "break point rally", "match point conversion", "deuce point", "momentum shift"
    ]
    
    ALL_LABELS = RALLY_GROUP + ACTION_GROUP

    # Logic Thresholds
    ACTION_THRESHOLD = 0.35 
    RALLY_THRESHOLD = 0.60
    ACTION_CONFIDENCE_BOOST = 1.5
    SAMPLE_INTERVAL = 1  # Process every frame for maximum accuracy

    def __init__(self, config: Optional[ModelConfig] = None):
        """Initialize provider."""
        super().__init__(config)
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.preprocess = None
        self.text_features = None
        self._model = None

    def load_model(self, config: ModelConfig) -> ModelHandle:
        """
        Load MobileCLIP model and precompute text features.
        """
        if open_clip is None:
            raise ImportError("open_clip is required. Install via 'pip install open_clip_torch'")

        logger.info(f"Loading {self.MODEL_NAME} from {config.model_path}...")
        
        try:
            # Load Model
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.MODEL_NAME, 
                pretrained=config.model_path
            )
            self.tokenizer = open_clip.get_tokenizer(self.MODEL_NAME)
            
            model.to(self.device)
            model.eval()
            self._model = model
            self.preprocess = preprocess

            # Precompute Text Embeddings (Optimization)
            self._precompute_text_features()

            return ModelHandle(
                model={
                    "model": model,
                    "tokenizer": self.tokenizer,
                    "preprocess": preprocess,
                    "text_features": self.text_features
                },
                config=config,
                metadata={
                    "name": self.MODEL_NAME,
                    "device": self.device,
                    "labels_count": len(self.ALL_LABELS)
                }
            )

        except Exception as e:
            logger.error(f"Failed to load MobileCLIP: {e}")
            raise

    def unload_model(self, handle: ModelHandle) -> None:
        """Clear model from memory."""
        if self._model:
            del self._model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._model = None
        self.text_features = None

    def _precompute_text_features(self):
        """Encode 21 event labels into embeddings once."""
        if not self._model or not self.tokenizer:
            return

        logger.info("Precomputing text features for 21 events...")
        with torch.no_grad():
            text_tokens = self.tokenizer(self.ALL_LABELS).to(self.device)
            text_features = self._model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            self.text_features = text_features

    def detect_events(self, video_path: str, output_csv_path: Optional[str] = None) -> List[Dict]:
        """
        Main entry point for video processing.
        
        Args:
            video_path: Path to input MP4.
            output_csv_path: Optional path to save CSV result.
            
        Returns:
            List of detected events (dicts).
        """
        if not self._model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # 1. Inference Loop
        raw_detections, fps = self._run_inference(video_path)
        
        # 2. Deduplication & Smoothing
        final_events = self._deduplicate_events(raw_detections)
        
        # 3. Output Handling
        if output_csv_path:
            self._save_to_csv(final_events, output_csv_path)
            
        return final_events

    def _run_inference(self, video_path: str) -> Tuple[List[Dict], float]:
        """Process video frames and apply Smart Selection logic."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing {total_frames} frames at {fps} FPS...")
        
        raw_detections = []
        
        # Prepare indices for fast lookup
        label_to_idx = {label: i for i, label in enumerate(self.ALL_LABELS)}
        rally_indices = [label_to_idx[l] for l in self.RALLY_GROUP]
        action_indices = [label_to_idx[l] for l in self.ACTION_GROUP]

        frame_count = 0
        pbar = tqdm(total=total_frames, desc="Inference")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % self.SAMPLE_INTERVAL == 0:
                # Preprocess Image
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
                
                # Inference
                with torch.no_grad(), torch.cuda.amp.autocast():
                    image_features = self._model.encode_image(image_tensor)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    probs = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                
                probs_np = probs.cpu().float().numpy()[0]
                
                # --- SMART SELECTION LOGIC ---
                
                # 1. Find Best Action Candidate
                best_action_idx = max(action_indices, key=lambda i: probs_np[i])
                action_score = probs_np[best_action_idx]
                best_action_label = self.ALL_LABELS[best_action_idx]
                
                # 2. Find Best Rally Candidate & Sum
                best_rally_idx = max(rally_indices, key=lambda i: probs_np[i])
                rally_score_total = sum(probs_np[i] for i in rally_indices)
                best_rally_label = self.ALL_LABELS[best_rally_idx]
                
                selected_label = None
                selected_conf = 0.0
                
                # Priority 1: Action (if above threshold) -> Apply Boost
                if action_score > self.ACTION_THRESHOLD:
                    selected_label = best_action_label
                    selected_conf = min(action_score * self.ACTION_CONFIDENCE_BOOST, 0.99)
                # Priority 2: Rally (if total sum is high)
                elif rally_score_total > self.RALLY_THRESHOLD:
                    selected_label = best_rally_label
                    selected_conf = rally_score_total
                
                if selected_label:
                    raw_detections.append({
                        "frame_id": frame_count,
                        "timestamp": frame_count / fps,
                        "event_label": selected_label,
                        "confidence": selected_conf
                    })
            
            frame_count += 1
            pbar.update(1)
            
        cap.release()
        pbar.close()
        return raw_detections, fps

    def _deduplicate_events(self, raw_data: List[Dict]) -> List[Dict]:
        """Merge consecutive identical frames into single events."""
        if not raw_data:
            return []
            
        logger.info("Deduplicating events...")
        merged_events = []
        current_event = None
        
        for item in raw_data:
            label = item['event_label']
            ts = item['timestamp']
            conf = item['confidence']
            fid = item['frame_id']
            
            if current_event is None:
                current_event = {
                    "frame_id": fid, "timestamp": ts, "end_timestamp": ts,
                    "event_label": label, "sum_confidence": conf, "count": 1
                }
            else:
                # Merge logic: Same Label OR Both are Rally types
                is_same = (label == current_event['event_label'])
                if label in self.RALLY_GROUP and current_event['event_label'] in self.RALLY_GROUP:
                    is_same = True
                
                if is_same:
                    current_event['end_timestamp'] = ts
                    current_event['sum_confidence'] += conf
                    current_event['count'] += 1
                else:
                    # Finalize previous event
                    avg_conf = current_event['sum_confidence'] / current_event['count']
                    current_event['confidence'] = avg_conf
                    del current_event['sum_confidence']
                    del current_event['count']
                    merged_events.append(current_event)
                    
                    # Start new event
                    current_event = {
                        "frame_id": fid, "timestamp": ts, "end_timestamp": ts,
                        "event_label": label, "sum_confidence": conf, "count": 1
                    }
        
        # Append last event
        if current_event:
            avg_conf = current_event['sum_confidence'] / current_event['count']
            current_event['confidence'] = avg_conf
            del current_event['sum_confidence']
            del current_event['count']
            merged_events.append(current_event)
            
        return merged_events

    def _save_to_csv(self, events: List[Dict], output_path: str):
        """Save formatted results to CSV."""
        if not events:
            logger.warning("No events to save.")
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = pd.DataFrame(events)
        
        # Format for output
        df['timestamp'] = df['timestamp'].map(lambda x: f"{x:.3f}")
        df['end_timestamp'] = df['end_timestamp'].map(lambda x: f"{x:.3f}")
        df['confidence'] = df['confidence'].map(lambda x: f"{x:.3f}")
        
        # Column ordering
        cols = ["frame_id", "timestamp", "end_timestamp", "event_label", "confidence"]
        df = df[cols]
        
        df.to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")

# Register provider
ProviderRegistry.register(
    "mobileclip",
    ProviderMetadata(
        model_type="vlm",
        provider_name="mobileclip",
        capabilities={"video_classification", "event_detection", "zero_shot"},
        description="Apple MobileCLIP2-S0 for fast pickleball event detection",
        version="0.1.0"
    )
)