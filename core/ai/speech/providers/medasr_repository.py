"""
Repository for MedASR model operations.
"""

import io
import os
import re
import time
import torch
import numpy as np
from typing import Optional
from transformers import AutoModelForCTC, AutoProcessor
import librosa
import decouple

from src.entity.transcription_entity import TranscriptionEntity, WordConfidence


def clean_medasr_transcript(transcript: str) -> str:
    """
    Clean MedASR transcript by removing formatting markers and fixing character repetition.
    
    MedASR outputs special formatting tokens:
    - Square brackets [ ]: Section headers (EXAM TYPE, INDICATION, etc.)
    - Curly brackets { }: Punctuation markers (period, colon, comma, etc.)
    - Character repetition: CTC decoding artifacts (e.g., "EEEXAAMM" -> "EXAM")
    
    Args:
        transcript: Raw transcript from MedASR
    
    Returns:
        Cleaned transcript with formatting markers removed and text normalized
    """
    if not transcript:
        return transcript
    
    # Remove square bracket section headers
    # Pattern: [ [WORD]] or [WORD] - extract the content
    transcript = re.sub(r'\[\s*\[?([A-Z\s]+)\]+\]', r'\1', transcript)
    
    # Remove curly bracket punctuation markers and replace with actual punctuation
    transcript = re.sub(r'\{\s*\{periodperiod\}\}', '.', transcript)
    transcript = re.sub(r'\{\s*\{coloncolon\}\}', ':', transcript)
    transcript = re.sub(r'\{\s*\{commacomma\}\}', ',', transcript)
    transcript = re.sub(r'\{\s*\{newnew\s+paragraphparagraph\}\}', '\n\n', transcript)
    
    # Fix character repetition (CTC decoding artifacts)
    # Remove triple+ character repeats (e.g., "EEEXAAMM" -> "EXAM")
    transcript = re.sub(r'([A-Z])\1{2,}', r'\1', transcript)
    # Fix spaced character repeats (e.g., "T TYYPE" -> "TYPE")
    transcript = re.sub(r'([A-Z])\s+\1', r'\1', transcript)
    # Fix word repetition (e.g., "word word" -> "word")
    transcript = re.sub(r'\b(\w+)\s+\1\b', r'\1', transcript)
    
    # Clean up extra spaces and normalize whitespace
    transcript = re.sub(r'\s+', ' ', transcript)
    # Remove spaces before punctuation
    transcript = re.sub(r'\s+([.,:;])', r'\1', transcript)
    # Add space after punctuation if missing
    transcript = re.sub(r'([.,:;])([A-Za-z])', r'\1 \2', transcript)
    
    # Capitalize section headers (common medical report sections)
    section_headers = ['EXAM TYPE', 'INDICATION', 'TECHNIQUE', 'FINDINGS', 'IMPRESSION']
    for header in section_headers:
        # Case-insensitive replacement
        pattern = re.compile(re.escape(header), re.IGNORECASE)
        transcript = pattern.sub(header, transcript)
    
    return transcript.strip()


class MedASRRepository:
    """Repository for Google MedASR model operations."""
    
    def __init__(self, model_name: str = "google/medasr", device: Optional[str] = None, token: Optional[str] = None):
        """
        Initialize MedASR repository.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to use ('cuda', 'cpu', or None for auto)
            token: Hugging Face token (if None, reads from HF_TOKEN env var)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        self._initialized = False
        
        # Get Hugging Face token from parameter, environment variable, or config
        self.token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        if not self.token:
            try:
                self.token = decouple.config("HF_TOKEN", default=None)
            except Exception:
                pass
    
    def _initialize_model(self):
        """Lazy initialization of model and processor."""
        if self._initialized:
            return
        
        try:
            # Use token for authentication if available
            # The token is needed for gated models like google/medasr
            model_kwargs = {}
            if self.token:
                model_kwargs["token"] = self.token
            
            # Load processor and model using AutoProcessor/AutoModelForCTC
            # This matches the official MedASR notebook approach
            # AutoProcessor will automatically handle the underlying LasrProcessor class
            try:
                # Try with trust_remote_code=True first (needed for custom processor classes)
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        **model_kwargs
                    )
                except (ValueError, OSError, KeyError) as proc_error:
                    error_str = str(proc_error)
                    # If "Unrecognized processing class", try loading LasrProcessor directly
                    if "Unrecognized processing class" in error_str or "Can't instantiate" in error_str:
                        try:
                            # Try loading LasrProcessor directly as fallback
                            from transformers import LasrProcessor
                            self.processor = LasrProcessor.from_pretrained(
                                self.model_name,
                                trust_remote_code=True,
                                **model_kwargs
                            )
                        except ImportError:
                            # LasrProcessor not available - this means transformers version is wrong
                            raise RuntimeError(
                                f"Failed to initialize MedASR processor: {error_str}\n\n"
                                f"AutoProcessor cannot find the processor class, and LasrProcessor is not available.\n"
                                f"This means transformers is not installed from the correct source.\n\n"
                                f"Please ensure:\n"
                                f"1. Uninstall old transformers: pip uninstall transformers -y\n"
                                f"2. Install from specific commit: pip install git+https://github.com/huggingface/transformers.git@65dc261512cbdb1ee72b88ae5b222f2605aad8e5\n"
                                f"3. Or use uv sync (pyproject.toml is already configured)\n"
                                f"4. Restart your Python environment after installation\n"
                                f"5. Clear Hugging Face cache: rm -rf ~/.cache/huggingface/hub/models--google--medasr"
                            )
                        except Exception as lasr_error:
                            # LasrProcessor failed for other reasons
                            raise RuntimeError(
                                f"Failed to initialize MedASR processor.\n"
                                f"AutoProcessor error: {error_str}\n"
                                f"LasrProcessor error: {str(lasr_error)}\n\n"
                                f"Please ensure transformers is installed from source with the correct commit."
                            )
                    else:
                        raise
                
                # Load the model
                try:
                    self.model = AutoModelForCTC.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        **model_kwargs
                    ).to(self.device)
                except (ValueError, OSError) as model_error:
                    error_str = str(model_error)
                    # If architecture error, try without trust_remote_code
                    if "lasr_ctc" not in error_str.lower() and "does not recognize this architecture" not in error_str.lower():
                        self.model = AutoModelForCTC.from_pretrained(
                            self.model_name,
                            **model_kwargs
                        ).to(self.device)
                    else:
                        raise
                        
            except ValueError as e:
                # Handle architecture errors with helpful message
                error_str = str(e)
                if "lasr_ctc" in error_str.lower() or "does not recognize this architecture" in error_str.lower():
                    raise RuntimeError(
                        f"Failed to initialize MedASR model: {error_str}\n\n"
                        f"The MedASR model requires transformers >= 5.0.0 installed from source.\n"
                        f"Please install it with:\n"
                        f"  pip install git+https://github.com/huggingface/transformers.git@65dc261512cbdb1ee72b88ae5b222f2605aad8e5\n\n"
                        f"Or update requirements.txt to use the specific commit.\n"
                        f"After installing, you may need to restart your Python environment."
                    )
                if "Unrecognized processing class" in error_str or "Can't instantiate" in error_str:
                    raise RuntimeError(
                        f"Failed to initialize MedASR processor: {error_str}\n\n"
                        f"This usually means transformers is not up to date or not installed from source.\n"
                        f"Please ensure:\n"
                        f"1. Transformers >= 5.0.0 is installed from source:\n"
                        f"   pip install git+https://github.com/huggingface/transformers.git@65dc261512cbdb1ee72b88ae5b222f2605aad8e5\n"
                        f"2. Restart your Python environment after installation\n"
                        f"3. Clear Hugging Face cache if needed: rm -rf ~/.cache/huggingface/hub/models--google--medasr"
                    )
                raise
            
            self.model.eval()
            self._initialized = True
        except RuntimeError:
            # Re-raise our custom errors
            raise
        except Exception as e:
            error_msg = str(e)
            if "gated" in error_msg.lower() or "401" in error_msg or "restricted" in error_msg.lower():
                raise RuntimeError(
                    f"Failed to initialize MedASR model: {error_msg}\n"
                    f"Please ensure HF_TOKEN environment variable is set with a valid Hugging Face token.\n"
                    f"Get your token at: https://huggingface.co/settings/tokens"
                )
            raise RuntimeError(f"Failed to initialize MedASR model: {str(e)}")
    
    def transcribe(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000,
        return_confidences: bool = True,
        return_timestamps: bool = False
    ) -> TranscriptionEntity:
        """
        Transcribe audio using MedASR model.
        
        Args:
            audio_array: Audio array (mono, 16kHz)
            sample_rate: Sample rate of audio
            return_confidences: Whether to return confidence scores
            return_timestamps: Whether to return timestamps (not fully supported yet)
        
        Returns:
            TranscriptionEntity with transcript and confidence scores
        """
        if not self._initialized:
            self._initialize_model()
        
        start_time = time.time()
        
        try:
            # Prepare inputs
            inputs = self.processor(
                audio_array,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )
            # Move all inputs to device 
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate transcription using model.generate()
            with torch.no_grad():
                outputs = self.model.generate(**inputs)
            
            # Decode text
            decoded_text = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True
            )[0]
            
            # Clean the transcript (remove formatting markers and fix artifacts)
            decoded_text = clean_medasr_transcript(decoded_text)
            
            # Get confidence scores if requested
            word_confidences = []
            overall_confidence = 1.0
            
            if return_confidences:
                # Get logits for confidence calculation
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                
                # Calculate confidence from logits
                # Apply softmax to get probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get max probability for each token (confidence)
                # Shape: (batch, seq_len) - get max prob for each position
                max_probs = torch.max(probs, dim=-1)[0]
                
                # Average confidence across sequence (excluding padding)
                # Flatten and filter very low probabilities
                valid_probs = max_probs[max_probs > 0.01]
                if len(valid_probs) > 0:
                    overall_confidence = float(torch.mean(valid_probs).item())
                else:
                    overall_confidence = 0.5  # Default if calculation fails
                
                # Word-level confidences (simplified - split by spaces)
                # In a real implementation, you'd get per-token confidences
                if decoded_text:
                    words = decoded_text.split()
                    if words:
                        word_confidence = overall_confidence
                        word_confidences = [
                            WordConfidence(
                                word=word,
                                confidence=word_confidence,
                                start_time=None,
                                end_time=None
                            )
                            for word in words
                        ]
            
            audio_duration = len(audio_array) / sample_rate
            
            return TranscriptionEntity(
                transcript=decoded_text,
                overall_confidence=overall_confidence,
                word_confidences=word_confidences,
                audio_duration=audio_duration,
                sample_rate=sample_rate,
                model_name=self.model_name,
                retry_count=0,
                errors=[]
            )
            
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {str(e)}")
    
    def transcribe_from_bytes(
        self,
        audio_bytes: bytes,
        return_confidences: bool = True,
        return_timestamps: bool = False
    ) -> TranscriptionEntity:
        """
        Transcribe audio from bytes.
        
        Args:
            audio_bytes: Audio file bytes
            return_confidences: Whether to return confidence scores
            return_timestamps: Whether to return timestamps
        
        Returns:
            TranscriptionEntity with transcript and confidence scores
        """
        audio_buffer = io.BytesIO(audio_bytes)
        audio_array, sample_rate = librosa.load(audio_buffer, sr=16000, mono=True)
        return self.transcribe(
            audio_array,
            sample_rate,
            return_confidences,
            return_timestamps
        )
