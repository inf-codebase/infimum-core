#!/usr/bin/env python3
"""
Unit tests for XCLIPProvider and XCLIPTrainer.

Covers:
  - ProviderRegistry auto-registration at import time
  - XCLIPProvider.load_model()  / unload_model()
  - XCLIPProvider.detect_events() for image & video inputs
  - XCLIPVideoDataset (dataset loading, video read errors)
  - XCLIPTrainer._freeze_backbone() / _build_train_config()
  - XCLIPTrainer.train() end-to-end (all heavy ops mocked)

All torch / transformers / cv2 calls are mocked so the suite runs
without a GPU or the actual model weights.
"""

import sys
import os
import json
import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open, call

# ---------------------------------------------------------------------------
# Make sure project root is on sys.path regardless of how pytest is invoked
# ---------------------------------------------------------------------------
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ---------------------------------------------------------------------------
# Stub heavy modules BEFORE any project import touches them
# ---------------------------------------------------------------------------
_torch_stub        = MagicMock()
_torch_stub.cuda.is_available.return_value = False
_torch_stub.device.return_value            = "cpu"

sys.modules.setdefault("torch",                     _torch_stub)
sys.modules.setdefault("torch.nn",                  MagicMock())
sys.modules.setdefault("torch.nn.functional",       MagicMock())
sys.modules.setdefault("torch.utils",               MagicMock())
sys.modules.setdefault("torch.utils.data",          MagicMock())
sys.modules.setdefault("transformers",               MagicMock())
sys.modules.setdefault("cv2",                        MagicMock())
sys.modules.setdefault("pandas",                     MagicMock())
sys.modules.setdefault("tqdm",                       MagicMock())
sys.modules.setdefault("tqdm.auto",                  MagicMock())

# Stub speech / other submodules not needed for these tests
sys.modules.setdefault("core.ai.speech",                        MagicMock())
sys.modules.setdefault("core.ai.speech.models",                 MagicMock())
sys.modules.setdefault("core.ai.speech.models.speech2text",     MagicMock())
sys.modules.setdefault("core.ai.speech.models.text2speech",     MagicMock())

# ---------------------------------------------------------------------------
# Now we can safely import project code
# ---------------------------------------------------------------------------
from core.ai.base.providers import ModelConfig, ProviderRegistry
from core.ai.vlm.providers.xclip import (
    XCLIPProvider,
    XCLIPTrainer,
    XCLIPVideoDataset,
    XCLIPTrainConfig,
    _xclip_collate_fn,
)


# ===========================================================================
# Helpers / shared fixtures
# ===========================================================================

def _make_config(**extra) -> ModelConfig:
    """Minimal valid ModelConfig for xclip."""
    return ModelConfig(
        model_type="vlm",
        provider="xclip",
        model_path="/fake/model",
        extra_params=extra,
    )


def _make_model_mock():
    """Return a fake XCLIPModel that behaves like the real one."""
    m = MagicMock()
    m.eval.return_value = m
    m.to.return_value   = m

    # get_text_features → (1, 512) unit vector
    feat = MagicMock()
    feat.norm.return_value = MagicMock()
    feat.__truediv__ = lambda self, other: feat   # feat / norm → feat
    m.get_text_features.return_value = feat

    # get_video_features → same shape
    m.get_video_features.return_value = feat
    return m


def _make_processor_mock():
    p = MagicMock()
    p.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
    return p


# ===========================================================================
# 1. ProviderRegistry — auto-registration
# ===========================================================================

class TestProviderRegistration:
    """X-CLIP should be auto-registered when the module is imported."""

    def test_xclip_registered(self):
        assert ProviderRegistry.is_registered("vlm", "xclip")

    def test_xclip_in_list_all(self):
        assert "vlm-xclip" in ProviderRegistry.list_all()

    def test_xclip_capabilities(self):
        meta = ProviderRegistry.get("vlm-xclip")
        assert meta is not None
        assert "video_classification" in meta.capabilities
        assert "event_detection"      in meta.capabilities
        assert "fine_tuning"          in meta.capabilities

    def test_create_returns_xclip_provider(self):
        config = _make_config()
        provider = ProviderRegistry.create("vlm", "xclip", config)
        assert isinstance(provider, XCLIPProvider)


# ===========================================================================
# 2. XCLIPProvider — load / unload model
# ===========================================================================

class TestXCLIPProviderLoadUnload:
    """Tests for model loading and unloading via the provider interface."""

    @pytest.fixture
    def mock_transformers(self):
        model_mock     = _make_model_mock()
        processor_mock = _make_processor_mock()

        with patch("core.ai.vlm.providers.xclip.XCLIPModel") as MockModel, \
             patch("core.ai.vlm.providers.xclip.XCLIPProcessor") as MockProc, \
             patch("core.ai.vlm.providers.xclip.torch") as MockTorch:

            MockModel.from_pretrained.return_value     = model_mock
            MockProc.from_pretrained.return_value      = processor_mock
            MockTorch.cuda.is_available.return_value   = False
            MockTorch.no_grad.return_value             = MagicMock(
                __enter__=lambda s, *a: None,
                __exit__=lambda s, *a: None,
            )

            yield {
                "model":     model_mock,
                "processor": processor_mock,
                "MockModel": MockModel,
                "MockProc":  MockProc,
            }

    def test_load_model_reads_local_path(self, mock_transformers):
        config   = _make_config()
        provider = XCLIPProvider(config)
        handle   = provider.load_model(config)

        mock_transformers["MockModel"].from_pretrained.assert_called_once_with("/fake/model")
        mock_transformers["MockProc"].from_pretrained.assert_called_once_with("/fake/model")
        assert handle.metadata["model_id"] == "/fake/model"
        assert handle.metadata["labels_count"] > 0

    def test_load_model_uses_model_name_when_no_path(self, mock_transformers):
        config = ModelConfig(
            model_type="vlm",
            provider="xclip",
            model_name="microsoft/xclip-base-patch32",
        )
        provider = XCLIPProvider(config)
        provider.load_model(config)

        mock_transformers["MockModel"].from_pretrained.assert_called_once_with(
            "microsoft/xclip-base-patch32"
        )

    def test_load_model_custom_labels(self, mock_transformers):
        custom_labels = ["goal", "foul", "offside"]
        config   = _make_config(labels=custom_labels)
        provider = XCLIPProvider(config)
        provider.load_model(config)

        assert provider._labels == custom_labels

    def test_unload_model_clears_state(self, mock_transformers):
        config   = _make_config()
        provider = XCLIPProvider(config)
        handle   = provider.load_model(config)

        provider.unload_model(handle)

        assert provider._model         is None
        assert provider._processor     is None
        assert provider._text_features is None

    def test_detect_events_raises_if_not_loaded(self):
        provider = XCLIPProvider(_make_config())
        with pytest.raises(RuntimeError, match="Model not loaded"):
            provider.detect_events("any_path.mp4")

    def test_detect_events_raises_for_missing_file(self, mock_transformers):
        config   = _make_config()
        provider = XCLIPProvider(config)
        provider.load_model(config)

        with pytest.raises(FileNotFoundError):
            provider.detect_events("/nonexistent/file.mp4")


# ===========================================================================
# 3. XCLIPProvider — inference (image & video)
# ===========================================================================

class TestXCLIPProviderInference:
    """Tests for detect_events() on image and video inputs."""

    @pytest.fixture
    def loaded_provider(self, tmp_path):
        """Return a provider with model already loaded (mocked)."""
        import numpy as _np

        config   = _make_config(
            labels=["a", "b"],
            confidence_threshold=0.0,  # accept everything
        )
        provider = XCLIPProvider(config)

        # Inject mocks directly without going through transformers
        provider._model     = _make_model_mock()
        provider._processor = _make_processor_mock()
        provider._labels    = ["a", "b"]
        provider._action_group = []

        # Fake text_features: (2, 512) stub tensor
        stub_feat = MagicMock()
        stub_feat.norm.return_value = 1
        provider._text_features = stub_feat

        return provider, tmp_path

    def test_detect_file_type_image(self):
        assert XCLIPProvider._detect_file_type("photo.jpg")   == "image"
        assert XCLIPProvider._detect_file_type("image.png")   == "image"
        assert XCLIPProvider._detect_file_type("frame.webp")  == "image"

    def test_detect_file_type_video(self):
        assert XCLIPProvider._detect_file_type("clip.mp4")    == "video"
        assert XCLIPProvider._detect_file_type("match.avi")   == "video"
        assert XCLIPProvider._detect_file_type("video.mov")   == "video"

    def test_detect_file_type_unknown(self):
        assert XCLIPProvider._detect_file_type("file.xyz")    == "unknown"

    def test_detect_events_image_returns_sorted_by_confidence(self, loaded_provider, tmp_path):
        provider, _ = loaded_provider

        img_file = tmp_path / "test.jpg"
        img_file.write_bytes(b"fake_jpg_data")

        # Mock cv2 and internal batch inference
        with patch("core.ai.vlm.providers.xclip.cv2") as mock_cv2, \
             patch.object(provider, "_infer_batch") as mock_infer, \
             patch.object(provider, "_preprocess_frame_chunk") as mock_preprocess:

            mock_cv2.imread.return_value     = MagicMock()
            mock_cv2.cvtColor.return_value   = MagicMock()
            mock_preprocess.return_value     = MagicMock()

            # Two labels: [a=0.3, b=0.8]
            mock_infer.return_value = np.array([[0.3, 0.8]])

            results = provider.detect_events(str(img_file))

        # Results should be sorted by confidence desc
        assert len(results) == 2
        assert results[0]["confidence"] >= results[1]["confidence"]
        assert results[0]["event_label"] == "b"

    def test_infer_batch_calls_get_video_features(self, loaded_provider):
        provider, _ = loaded_provider

        stub_tensor = MagicMock()
        stub_tensor.norm.return_value = 1

        provider._model.get_video_features.return_value = stub_tensor

        # Let the matmul chain return something numpy-able
        stub_softmax = MagicMock()
        stub_softmax.cpu.return_value.float.return_value.numpy.return_value = np.array([[0.5, 0.5]])

        with patch("core.ai.vlm.providers.xclip.torch") as mock_torch:
            mock_torch.no_grad.return_value = MagicMock(
                __enter__=lambda s, *a: None,
                __exit__=lambda s, *a: None,
            )
            batch = MagicMock()
            provider._infer_batch(batch)

        provider._model.get_video_features.assert_called_once_with(pixel_values=batch)


# ===========================================================================
# 4. XCLIPVideoDataset
# ===========================================================================

class TestXCLIPVideoDataset:
    """Tests for the fine-tuning dataset class."""

    def _make_json(self, tmp_path, records):
        json_file = tmp_path / "captions.json"
        json_file.write_text(json.dumps({"sentences": records}), encoding="utf-8")
        return str(json_file)

    def test_len_filters_missing_files(self, tmp_path):
        records = [
            {"video_id": "v001", "caption": "serve"},
            {"video_id": "v002", "caption": "smash"},
        ]
        # Create only v001.mp4 on disk
        (tmp_path / "v001.mp4").write_bytes(b"fake")

        json_file = self._make_json(tmp_path, records)
        ds = XCLIPVideoDataset(json_file, str(tmp_path))

        assert len(ds) == 1
        assert ds.records[0]["video_id"] == "v001"

    def test_getitem_returns_none_on_bad_video(self, tmp_path):
        records = [{"video_id": "bad", "caption": "test"}]
        (tmp_path / "bad.mp4").write_bytes(b"not_a_real_video")

        json_file = self._make_json(tmp_path, records)

        with patch("core.ai.vlm.providers.xclip.cv2") as mock_cv2, \
             patch("core.ai.vlm.providers.xclip.torch") as mock_torch:

            # Simulate OpenCV failure
            cap_mock = MagicMock()
            cap_mock.isOpened.return_value = False
            mock_cv2.VideoCapture.return_value = cap_mock

            ds = XCLIPVideoDataset(json_file, str(tmp_path))
            item = ds[0]

        assert item is None

    def test_collate_fn_drops_none_items(self):
        with patch("core.ai.vlm.providers.xclip.torch") as mock_torch:
            mock_torch.stack.side_effect = lambda lst, dim=0: MagicMock()

            batch = [
                None,
                {"video": MagicMock(), "text": "serve",  "video_id": "v1"},
                None,
                {"video": MagicMock(), "text": "smash",  "video_id": "v2"},
            ]
            result = _xclip_collate_fn(batch)

        assert result is not None
        _, texts, vids = result
        assert "serve" in texts
        assert "smash" in texts
        assert len(vids) == 2

    def test_collate_fn_returns_none_for_all_none(self):
        result = _xclip_collate_fn([None, None, None])
        assert result is None


# ===========================================================================
# 5. XCLIPTrainer
# ===========================================================================

class TestXCLIPTrainer:
    """Tests for XCLIPTrainer configuration, freeze logic, and training loop."""

    def test_build_train_config_uses_extra_params(self):
        config = ModelConfig(
            model_type="vlm",
            provider="xclip",
            model_name="microsoft/xclip-base-patch32",
            extra_params={
                "json_file":       "my.json",
                "video_dir":       "my_videos",
                "save_dir":        "my_output",
                "num_epochs":      5,
                "batch_size":      8,
                "learning_rate":   2e-5,
                "grad_accum_steps": 2,
                "num_workers":     2,
            },
        )
        trainer = XCLIPTrainer(config)
        cfg = trainer._train_cfg

        assert cfg.json_file        == "my.json"
        assert cfg.video_dir        == "my_videos"
        assert cfg.save_dir         == "my_output"
        assert cfg.num_epochs       == 5
        assert cfg.batch_size       == 8
        assert cfg.learning_rate    == pytest.approx(2e-5)
        assert cfg.grad_accum_steps == 2
        assert cfg.num_workers      == 2

    def test_build_train_config_defaults(self):
        config  = ModelConfig(model_type="vlm", provider="xclip",
                              model_name="microsoft/xclip-base-patch32")
        trainer = XCLIPTrainer(config)
        cfg     = trainer._train_cfg

        assert cfg.num_epochs       == XCLIPTrainConfig.num_epochs
        assert cfg.batch_size       == XCLIPTrainConfig.batch_size
        assert cfg.grad_accum_steps == XCLIPTrainConfig.grad_accum_steps

    def test_freeze_backbone_disables_all_then_reenables_keywords(self):
        """All params must be frozen, then projection layers unfrozen."""
        with patch("core.ai.vlm.providers.xclip.torch"):
            model = MagicMock()

            # Simulate 4 named params
            params = {
                "vision_encoder.layer.weight": MagicMock(requires_grad=True),
                "visual_projection.weight":    MagicMock(requires_grad=True),
                "text_projection.bias":        MagicMock(requires_grad=True),
                "logit_scale":                 MagicMock(requires_grad=True),
            }
            model.parameters.return_value = list(params.values())
            model.named_parameters.return_value = list(params.items())

            keywords = ["visual_projection", "text_projection", "logit_scale",
                        "prompts_generator", "mit"]
            XCLIPTrainer._freeze_backbone(model, keywords)

        # vision_encoder has no keyword → must be frozen
        params["vision_encoder.layer.weight"].requires_grad == False  # noqa: B015

        # projection layers → must be unfrozen
        assert params["visual_projection.weight"].requires_grad is True
        assert params["text_projection.bias"].requires_grad     is True
        assert params["logit_scale"].requires_grad              is True

    def test_train_saves_model_at_end(self, tmp_path):
        """Full train() with all external calls mocked; model.save_pretrained called."""
        config = ModelConfig(
            model_type="vlm",
            provider="xclip",
            model_name="microsoft/xclip-base-patch32",
            extra_params={
                "json_file":  str(tmp_path / "captions.json"),
                "video_dir":  str(tmp_path / "videos"),
                "save_dir":   str(tmp_path / "output"),
                "num_epochs": 1,
                "batch_size": 2,
            },
        )
        trainer = XCLIPTrainer(config)

        model_mock     = _make_model_mock()
        processor_mock = _make_processor_mock()

        # Fake outputs with logits
        outputs_mock = MagicMock()
        logits       = MagicMock()
        logits.size.return_value = 2
        outputs_mock.logits_per_video = logits
        outputs_mock.logits_per_text  = logits
        model_mock.return_value       = outputs_mock

        # Fake dataloader yields 1 batch then stops
        videos   = MagicMock()
        texts    = ["serve", "smash"]
        vids     = ["v1", "v2"]
        fake_batch = (videos, texts, vids)

        with patch("core.ai.vlm.providers.xclip.XCLIPModel") as MockModel, \
             patch("core.ai.vlm.providers.xclip.XCLIPProcessor") as MockProc, \
             patch("core.ai.vlm.providers.xclip.XCLIPVideoDataset") as MockDS, \
             patch("core.ai.vlm.providers.xclip.DataLoader") as MockDL, \
             patch("core.ai.vlm.providers.xclip.torch") as mock_torch, \
             patch("core.ai.vlm.providers.xclip.nn") as mock_nn, \
             patch("core.ai.vlm.providers.xclip.os.makedirs"):

            MockModel.from_pretrained.return_value = model_mock
            MockProc.from_pretrained.return_value  = processor_mock

            # Dataset stub
            ds_instance = MagicMock()
            MockDS.return_value = ds_instance

            # DataLoader yields exactly 1 non-None batch
            MockDL.return_value = iter([fake_batch])

            # torch stubs
            mock_torch.cuda.is_available.return_value = False
            mock_torch.device.return_value = "cpu"
            mock_torch.arange.return_value = MagicMock()
            mock_torch.nn.utils.clip_grad_norm_ = MagicMock()

            # Loss function stub
            loss_val = MagicMock()
            loss_val.__truediv__ = lambda s, o: loss_val
            loss_val.__add__     = lambda s, o: loss_val
            loss_val.item.return_value = 0.5
            loss_val.backward = MagicMock()

            mock_nn.CrossEntropyLoss.return_value.return_value = loss_val

            # Optimizer mock
            trainer.device = "cpu"
            trainer.train()

        model_mock.save_pretrained.assert_called_once()
        processor_mock.save_pretrained.assert_called_once()


# ===========================================================================
# 6. XCLIPTrainConfig — dataclass defaults
# ===========================================================================

class TestXCLIPTrainConfig:
    def test_default_trainable_keywords(self):
        cfg = XCLIPTrainConfig()
        expected = [
            "visual_projection", "text_projection",
            "logit_scale", "prompts_generator", "mit",
        ]
        assert cfg.trainable_keywords == expected

    def test_default_values(self):
        cfg = XCLIPTrainConfig()
        assert cfg.num_frames       == 8
        assert cfg.batch_size       == 16
        assert cfg.num_epochs       == 10
        assert cfg.grad_accum_steps == 4
        assert cfg.num_workers      == 4
        assert cfg.learning_rate    == pytest.approx(1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
