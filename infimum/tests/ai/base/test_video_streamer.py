"""
Unit tests for VideoStreamer and helpers (get_video_content_type, parse_range_header).
"""

import sys
import tempfile
import types
import unittest
from pathlib import Path

# Add project root to path (same as test_data.py)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Avoid loading optional core.ai subpackages (llm, speech) so loader imports
# succeed without langchain/speech deps. Alias core.ai.core -> core.ai.base
# so loaders' "...core.data.base" resolves.
def _mock_package(name):
    m = types.ModuleType(name)
    m.__path__ = []
    return m

if "core.ai.llm" not in sys.modules:
    llm = _mock_package("core.ai.llm")
    for _name in (
        "Agent", "ToolManager", "Tool", "WebSearchTool", "CalculatorTool",
        "WeatherTool", "TimeTool", "Planner", "Memory",
        "AGENT_PROMPTS", "PLANNER_PROMPTS", "MEMORY_PROMPTS",
    ):
        setattr(llm, _name, None)
    sys.modules["core.ai.llm"] = llm
if "core.ai.speech" not in sys.modules:
    speech = _mock_package("core.ai.speech")
    speech.models = _mock_package("core.ai.speech.models")
    speech.models.speech2text = types.ModuleType("core.ai.speech.models.speech2text")
    speech.models.speech2text.Speech2Text = None
    speech.models.text2speech = types.ModuleType("core.ai.speech.models.text2speech")
    speech.models.text2speech.Text2Speech = None
    sys.modules["core.ai.speech"] = speech
    sys.modules["core.ai.speech.models"] = speech.models
    sys.modules["core.ai.speech.models.speech2text"] = speech.models.speech2text
    sys.modules["core.ai.speech.models.text2speech"] = speech.models.text2speech

# Loaders use "...core.data.base" (core.ai.core.data); alias to core.ai.base.data
sys.modules["core.ai.core"] = sys.modules["core.ai.base"]

from infimum.ai.data.loaders.video import (
    VideoStreamer,
    get_video_content_type,
    parse_range_header,
)


class TestGetVideoContentType(unittest.TestCase):
    """Tests for get_video_content_type."""

    def test_mp4_returns_video_mp4(self):
        self.assertEqual(get_video_content_type("/path/video.mp4"), "video/mp4")
        self.assertEqual(get_video_content_type(Path("/path/video.MP4")), "video/mp4")

    def test_webm_returns_video_webm(self):
        self.assertEqual(get_video_content_type("file.webm"), "video/webm")

    def test_ogg_returns_video_ogg(self):
        self.assertEqual(get_video_content_type("file.ogg"), "video/ogg")
        self.assertEqual(get_video_content_type("file.ogv"), "video/ogg")

    def test_mov_returns_quicktime(self):
        self.assertEqual(get_video_content_type("file.mov"), "video/quicktime")

    def test_avi_returns_x_msvideo(self):
        self.assertEqual(get_video_content_type("file.avi"), "video/x-msvideo")

    def test_mkv_returns_x_matroska(self):
        self.assertEqual(get_video_content_type("file.mkv"), "video/x-matroska")

    def test_unknown_extension_returns_octet_stream(self):
        self.assertEqual(
            get_video_content_type("file.unknown"),
            "application/octet-stream",
        )
        self.assertEqual(
            get_video_content_type("file"),
            "application/octet-stream",
        )


class TestParseRangeHeader(unittest.TestCase):
    """Tests for parse_range_header."""

    def test_bytes_0_499(self):
        start, end = parse_range_header("bytes=0-499", 1000)
        self.assertEqual(start, 0)
        self.assertEqual(end, 499)

    def test_bytes_100_to_end(self):
        start, end = parse_range_header("bytes=100-", 1000)
        self.assertEqual(start, 100)
        self.assertEqual(end, 999)

    def test_bytes_with_spaces(self):
        start, end = parse_range_header("  bytes = 10 - 20  ", 100)
        self.assertEqual(start, 10)
        self.assertEqual(end, 20)

    def test_case_insensitive(self):
        start, end = parse_range_header("Bytes=0-99", 500)
        self.assertEqual(start, 0)
        self.assertEqual(end, 99)

    def test_none_returns_none_none(self):
        start, end = parse_range_header(None, 1000)
        self.assertIsNone(start)
        self.assertIsNone(end)

    def test_empty_string_returns_none_none(self):
        start, end = parse_range_header("", 1000)
        self.assertIsNone(start)
        self.assertIsNone(end)
        start, end = parse_range_header("   ", 1000)
        self.assertIsNone(start)
        self.assertIsNone(end)

    def test_invalid_format_returns_none_none(self):
        self.assertEqual(parse_range_header("invalid", 1000), (None, None))
        self.assertEqual(parse_range_header("bytes=abc-100", 1000), (None, None))
        self.assertEqual(parse_range_header("bytes=0", 1000), (None, None))

    def test_clamped_to_file_size(self):
        start, end = parse_range_header("bytes=0-2000", 1000)
        self.assertEqual(start, 0)
        self.assertEqual(end, 999)

    def test_start_beyond_file_returns_none_none(self):
        start, end = parse_range_header("bytes=1000-2000", 1000)
        self.assertIsNone(start)
        self.assertIsNone(end)

    def test_start_after_end_returns_none_none(self):
        start, end = parse_range_header("bytes=100-50", 200)
        self.assertIsNone(start)
        self.assertIsNone(end)


class TestVideoStreamer(unittest.TestCase):
    """Tests for VideoStreamer."""

    def setUp(self):
        """Create a temporary file for streaming tests."""
        self.tmp_file = tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".mp4",
            delete=False,
        )
        # Write 5 KB of data
        self.tmp_file.write(b"x" * (5 * 1024))
        self.tmp_file.close()
        self.tmp_path = Path(self.tmp_file.name)

    def tearDown(self):
        """Remove temporary file."""
        if self.tmp_path.exists():
            self.tmp_path.unlink()

    def test_init_accepts_str_and_path(self):
        streamer = VideoStreamer(str(self.tmp_path))
        self.assertEqual(streamer.path, self.tmp_path.resolve())
        streamer2 = VideoStreamer(self.tmp_path)
        self.assertEqual(streamer2.path, self.tmp_path.resolve())

    def test_init_nonexistent_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            VideoStreamer("/nonexistent/video.mp4")
        self.assertIn("not found", str(ctx.exception))

    def test_init_directory_raises_value_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError) as ctx:
                VideoStreamer(d)
            self.assertIn("not a file", str(ctx.exception))

    def test_content_length(self):
        streamer = VideoStreamer(self.tmp_path)
        self.assertEqual(streamer.content_length(), 5 * 1024)

    def test_content_type(self):
        streamer = VideoStreamer(self.tmp_path)
        self.assertEqual(streamer.content_type(), "video/mp4")

    def test_stream_full_file_yields_all_bytes(self):
        streamer = VideoStreamer(self.tmp_path, chunk_size=1024)
        chunks = list(streamer.stream())
        total = sum(len(c) for c in chunks)
        self.assertEqual(total, 5 * 1024)
        self.assertEqual(len(chunks), 5)  # 5 chunks of 1024

    def test_stream_with_small_chunk_size(self):
        streamer = VideoStreamer(self.tmp_path, chunk_size=100)
        chunks = list(streamer.stream())
        total = sum(len(c) for c in chunks)
        self.assertEqual(total, 5 * 1024)
        self.assertGreater(len(chunks), 1)

    def test_stream_range_partial(self):
        streamer = VideoStreamer(self.tmp_path, chunk_size=1024)
        chunks = list(streamer.stream(0, 100))
        total = sum(len(c) for c in chunks)
        self.assertEqual(total, 101)
        self.assertEqual(b"".join(chunks), b"x" * 101)

    def test_stream_range_middle(self):
        streamer = VideoStreamer(self.tmp_path, chunk_size=50)
        # bytes 100-199 inclusive = 100 bytes
        chunks = list(streamer.stream(100, 199))
        total = sum(len(c) for c in chunks)
        self.assertEqual(total, 100)
        self.assertEqual(b"".join(chunks), b"x" * 100)

    def test_stream_range_empty_when_start_gt_end(self):
        streamer = VideoStreamer(self.tmp_path)
        chunks = list(streamer.stream(100, 50))
        self.assertEqual(chunks, [])

    def test_stream_range_clamped_to_file_end(self):
        streamer = VideoStreamer(self.tmp_path)
        # Request beyond file end
        chunks = list(streamer.stream(0, 100_000))
        total = sum(len(c) for c in chunks)
        self.assertEqual(total, 5 * 1024)

    def test_stream_range_header_full_file(self):
        streamer = VideoStreamer(self.tmp_path, chunk_size=1024)
        chunks = list(streamer.stream_range(None))
        self.assertEqual(sum(len(c) for c in chunks), 5 * 1024)

    def test_stream_range_header_bytes_spec(self):
        streamer = VideoStreamer(self.tmp_path, chunk_size=100)
        chunks = list(streamer.stream_range("bytes=0-99"))
        total = sum(len(c) for c in chunks)
        self.assertEqual(total, 100)
        self.assertEqual(b"".join(chunks), b"x" * 100)

    def test_stream_range_header_invalid_returns_full_file(self):
        streamer = VideoStreamer(self.tmp_path)
        chunks = list(streamer.stream_range("invalid"))
        self.assertEqual(sum(len(c) for c in chunks), 5 * 1024)


if __name__ == "__main__":
    unittest.main()
