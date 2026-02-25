"""
Video loader implementation.

Strategy pattern: Implements BaseLoader for video data.
"""

from typing import Union, List
from pathlib import Path
from PIL import Image
from ...base.data.base import BaseLoader
from ...base.data.item import DataItem


class VideoLoader(BaseLoader):
    """
    Video data loader.

    Loads video frames from files.
    """

    def _load(
        self, source: Union[str, Path], frame_indices: List[int] = None
    ) -> DataItem:
        """
        Load video data.

        Args:
            source: Video source (file path)
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

            if frame_indices is None:
                # Load all frames
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
            else:
                # Load specific frames
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(frame_rgb))

            cap.release()

            return DataItem(
                data=frames,
                data_type="video",
                source=str(path),
                metadata={
                    "frame_count": frame_count,
                    "loaded_frames": len(frames),
                    "fps": fps,
                },
            )
        else:
            raise ValueError(f"Unsupported video source type: {type(source)}")
