"""Unit tests for core.utils.constants module."""
import pytest
from core.utils.constants import (
    CONTROLLER_HEART_BEAT_EXPIRATION,
    WORKER_HEART_BEAT_INTERVAL,
    LOGDIR,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
    API_V1,
    CrawlingType,
    Prompt
)


class TestConstants:
    """Test cases for constants module."""

    def test_controller_heart_beat_expiration(self):
        """Test CONTROLLER_HEART_BEAT_EXPIRATION constant."""
        assert isinstance(CONTROLLER_HEART_BEAT_EXPIRATION, int)
        assert CONTROLLER_HEART_BEAT_EXPIRATION == 30

    def test_worker_heart_beat_interval(self):
        """Test WORKER_HEART_BEAT_INTERVAL constant."""
        assert isinstance(WORKER_HEART_BEAT_INTERVAL, int)
        assert WORKER_HEART_BEAT_INTERVAL == 15

    def test_logdir(self):
        """Test LOGDIR constant."""
        assert isinstance(LOGDIR, str)
        assert LOGDIR == "."

    def test_ignore_index(self):
        """Test IGNORE_INDEX constant."""
        assert isinstance(IGNORE_INDEX, int)
        assert IGNORE_INDEX == -100

    def test_image_token_index(self):
        """Test IMAGE_TOKEN_INDEX constant."""
        assert isinstance(IMAGE_TOKEN_INDEX, int)
        assert IMAGE_TOKEN_INDEX == -200

    def test_default_image_token(self):
        """Test DEFAULT_IMAGE_TOKEN constant."""
        assert isinstance(DEFAULT_IMAGE_TOKEN, str)
        assert DEFAULT_IMAGE_TOKEN == "<image>"

    def test_default_image_patch_token(self):
        """Test DEFAULT_IMAGE_PATCH_TOKEN constant."""
        assert isinstance(DEFAULT_IMAGE_PATCH_TOKEN, str)
        assert DEFAULT_IMAGE_PATCH_TOKEN == "<im_patch>"

    def test_default_im_start_token(self):
        """Test DEFAULT_IM_START_TOKEN constant."""
        assert isinstance(DEFAULT_IM_START_TOKEN, str)
        assert DEFAULT_IM_START_TOKEN == "<im_start>"

    def test_default_im_end_token(self):
        """Test DEFAULT_IM_END_TOKEN constant."""
        assert isinstance(DEFAULT_IM_END_TOKEN, str)
        assert DEFAULT_IM_END_TOKEN == "<im_end>"

    def test_image_placeholder(self):
        """Test IMAGE_PLACEHOLDER constant."""
        assert isinstance(IMAGE_PLACEHOLDER, str)
        assert IMAGE_PLACEHOLDER == "<image-placeholder>"

    def test_api_v1(self):
        """Test API_V1 constant."""
        assert isinstance(API_V1, str)
        assert API_V1 == "v1"


class TestCrawlingType:
    """Test cases for CrawlingType enum."""

    def test_crawling_type_pdf(self):
        """Test CrawlingType.PDF."""
        assert CrawlingType.PDF == 1

    def test_crawling_type_web(self):
        """Test CrawlingType.WEB."""
        assert CrawlingType.WEB == 2

    def test_crawling_type_text(self):
        """Test CrawlingType.TEXT."""
        assert CrawlingType.TEXT == 3


class TestPrompt:
    """Test cases for Prompt class."""

    def test_prompt_next_crypto_event(self):
        """Test Prompt.NEXT_CRYPTO_EVENT."""
        assert isinstance(Prompt.NEXT_CRYPTO_EVENT, str)
        assert "cryptocurrency" in Prompt.NEXT_CRYPTO_EVENT.lower()
