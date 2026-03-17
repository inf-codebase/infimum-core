"""
Video loader implementation.

Strategy pattern: Implements BaseLoader for video data.

VideoStreamer streams raw file bytes in chunks for server-to-client delivery
(e.g. with HTTP Range support). Use VideoLoader for loading decoded frames
for AI/processing.
"""

import re
from typing import Callable, Union, List, Optional, Iterator, Tuple

from pathlib import Path
from core.ai.base import BaseLoader
from core.ai.base.data import DataItem
from tqdm import tqdm


VIDEO_EXTENSION_TO_MIMETYPE = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".ogg": "video/ogg",
    ".ogv": "video/ogg",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
}


def get_video_content_type(path: Union[str, Path]) -> str:
    """
    Infer Content-Type for a video file from its extension.

    Args:
        path: Video file path.

    Returns:
        MIME type string, or "application/octet-stream" if unknown.
    """
    ext = Path(path).suffix.lower()
    return VIDEO_EXTENSION_TO_MIMETYPE.get(ext, "application/octet-stream")


def parse_range_header(range_header: Optional[str], total_size: int) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse HTTP Range header (bytes=start-end) and return clamped (start, end).

    Args:
        range_header: Value of the Range header (e.g. "bytes=0-1023").
        total_size: Total size of the resource in bytes.

    Returns:
        (start, end) for the requested range (end is inclusive), or
        (None, None) if header is missing or invalid (meaning full file).
    """
    if not range_header or not range_header.strip():
        return (None, None)
    match = re.match(r"^\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*$", range_header.strip(), re.IGNORECASE)
    if not match:
        return (None, None)
    start_s, end_s = match.group(1), match.group(2)
    start = int(start_s) if start_s else 0
    end = int(end_s) if end_s else total_size - 1
    if start < 0 or end < start or start >= total_size:
        return (None, None)
    start = max(0, start)
    end = min(end, total_size - 1)
    return (start, end)


class VideoStreamer:
    """
    Streams a video file in byte chunks for server-to-client delivery.

    Memory-efficient: only one chunk is in memory at a time. Supports optional
    byte range (start, end) for HTTP Range requests (seeking). Use with
    FastAPI StreamingResponse and Range/Content-Range headers for full
    playback and seek support in browsers.

    Example with FastAPI (range requests for seeking)::

        from fastapi import Request
        from fastapi.responses import StreamingResponse

        @app.get("/video/{path:path}")
        def stream_video(path: str, request: Request):
            streamer = VideoStreamer(your_base_dir / path)
            range_header = request.headers.get("range")
            start, end = parse_range_header(range_header, streamer.content_length())
            if start is not None and end is not None:
                content_length = end - start + 1
                headers = {
                    "Accept-Ranges": "bytes",
                    "Content-Range": f"bytes {start}-{end}/{streamer.content_length()}",
                    "Content-Length": str(content_length),
                }
                return StreamingResponse(
                    streamer.stream(start, end),
                    status_code=206,
                    media_type=streamer.content_type(),
                    headers=headers,
                )
            return StreamingResponse(
                streamer.stream(),
                media_type=streamer.content_type(),
                headers={"Accept-Ranges": "bytes", "Content-Length": str(streamer.content_length())},
            )
    """

    def __init__(self, file_path: Union[str, Path], chunk_size: int = 1_048_576) -> None:
        """
        Args:
            file_path: Path to the video file.
            chunk_size: Size of each yielded chunk in bytes (default 1 MB).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the path is not a file.
        """
        self._path = Path(file_path).resolve()
        if not self._path.exists():
            raise FileNotFoundError(f"Video file not found: {self._path}")
        if not self._path.is_file():
            raise ValueError(f"Path is not a file: {self._path}")
        self._chunk_size = chunk_size

    @property
    def path(self) -> Path:
        """Resolved path to the video file."""
        return self._path

    def content_length(self) -> int:
        """Return the size of the video file in bytes."""
        return self._path.stat().st_size

    def content_type(self) -> str:
        """Return the MIME type for the video file (e.g. video/mp4)."""
        return get_video_content_type(self._path)

    def stream(
        self,
        range_start: Optional[int] = None,
        range_end: Optional[int] = None,
    ) -> Iterator[bytes]:
        """
        Yield the video file in chunks. Optionally restrict to a byte range.

        Args:
            range_start: Start byte (inclusive). If None, stream from start.
            range_end: End byte (inclusive). If None, stream to end of file.
                      Ignored if range_start is None.

        Yields:
            Chunks of up to chunk_size bytes.
        """
        total = self.content_length()
        if range_start is None:
            start, end = 0, total - 1
        else:
            start = max(0, range_start)
            end = min(range_end if range_end is not None else total - 1, total - 1)
            if start > end:
                return
        try:
            with open(self._path, "rb") as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    to_read = min(self._chunk_size, remaining)
                    chunk = f.read(to_read)
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk
        finally:
            pass

    def stream_range(self, range_header: Optional[str]) -> Iterator[bytes]:
        """
        Yield chunks for the range specified by an HTTP Range header.

        Args:
            range_header: Value of the Range header (e.g. "bytes=0-1023").
                         If None or invalid, streams the full file.

        Yields:
            Chunks of up to chunk_size bytes.
        """
        total = self.content_length()
        start, end = parse_range_header(range_header, total)
        yield from self.stream(start, end)


class VideoLoader(BaseLoader):
    """
    Video data loader.

    Loads video frames from files.
    """

    def _load(
        self, source: Union[str, Path], data_collator: Optional[Union[Callable, dict]] = None, frame_indices: List[int] = None
    ) -> DataItem:
        """
        Load video data with support for batch processing.

        Args:
            source: Video source (file path)
            data_collator: Optional collator - can be:
                - Callable: Process frames one-by-one (backward compatible)
                - Dict: {"batch_fn": callable, "batch_size": int} for batch processing
                - None: Return PIL images without processing
            frame_indices: Optional list of frame indices to load

        Returns:
            DataItem: Loaded video data (list of frames)
        """
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "Video loading requires opencv-python. "
                "Install with: pip install opencv-python"
            )

        if isinstance(source, Path):
            source = str(source)

        if isinstance(source, str):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Video file not found: {source}")

            # Open video
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {source}")

            frames = []
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Determine processing mode
            is_batch_mode = isinstance(data_collator, dict) and "batch_fn" in data_collator
            batch_fn = data_collator.get("batch_fn") if is_batch_mode else None
            batch_size = data_collator.get("batch_size", 16) if is_batch_mode else None
            single_fn = data_collator if callable(data_collator) else None

            if frame_indices is None:
                # Load all frames
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if single_fn:
                        frame = single_fn(frame)
                    frames.append(frame)
            else:
                # Load specific frames
                if is_batch_mode:
                    # Batch mode: accumulate PIL images using sequential grab() instead of
                    # random cap.set() seeks. A single seek to the first frame is made, then
                    # cap.grab() skips non-target frames (no pixel decode/alloc) and cap.read()
                    # is called only for target frames. ~3-5x faster on compressed video.
                    raw_batch = []
                    sorted_indices = sorted(frame_indices)
                    if sorted_indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, sorted_indices[0])
                        current_pos = sorted_indices[0]
                        for target_idx in tqdm(sorted_indices, desc="Loading frames", unit="frame"):
                            frames_to_skip = target_idx - current_pos
                            for _ in range(frames_to_skip):
                                cap.grab()
                            ret, frame = cap.read()
                            current_pos = target_idx + 1
                            if not ret:
                                continue
                            raw_batch.append(frame)
                            # When we reach batch_size, process and clear
                            if len(raw_batch) == batch_size:
                                processed_batch = batch_fn(raw_batch)
                                frames.extend(processed_batch)
                                raw_batch.clear()
                        # Process any remaining frames smaller than batch_size
                        if raw_batch:
                            processed_batch = batch_fn(raw_batch)
                            frames.extend(processed_batch)
                            raw_batch.clear()
                else:
                    # Single-frame mode (backward compatible) — same grab() optimisation
                    sorted_indices_single = sorted(frame_indices)
                    if sorted_indices_single:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, sorted_indices_single[0])
                        current_pos = sorted_indices_single[0]
                        for idx in sorted_indices_single:
                            frames_to_skip = idx - current_pos
                            for _ in range(frames_to_skip):
                                cap.grab()
                            ret, frame = cap.read()
                            current_pos = idx + 1
                            if ret:
                                if single_fn:
                                    frame = single_fn(frame)
                                frames.append(frame)
            cap.release()
            return DataItem(data=frames, data_type="video", source=str(path), metadata={"frame_count": frame_count, "loaded_frames": len(frames), "fps": fps})
        else:
            raise ValueError(f"Unsupported video source type: {type(source)}")
