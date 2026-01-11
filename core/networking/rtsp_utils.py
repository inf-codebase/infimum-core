# -*- coding: utf-8 -*-
"""
rtsp_utils.py
--------------
Các tiện ích làm việc với RTSP / video bằng OpenCV:
- Cấu hình FFMPEG options cho RTSP (ổn định reconnect/timeout/buffer)
- Mở nguồn video với retry và fallback (FFMPEG -> default backend)
- Đọc 1 frame có timeout
- Validate RTSP kết nối và chụp preview
- Tính histogram HS + so sánh độ tương tự (dùng để skip frame giống nhau)

Gợi ý sử dụng:
    from src.core.utils.rtsp_utils import (
        ensure_rtsp_env, set_rtsp_env,
        open_capture, read_one_frame,
        validate_rtsp_connect, capture_preview,
        compute_hist_hs, hist_similarity,
    )
"""

from __future__ import annotations
import os
import time
import uuid
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------
# Cấu hình RTSP cho OpenCV-FFMPEG
# ---------------------------------------------------------------------

DEFAULT_RTSP_OPTS = (
    "rtsp_transport;tcp"       # RTSP over TCP để ổn định hơn
    "|stimeout;10000000"       # 10s socket timeout (microseconds)
    "|rw_timeout;10000000"     # 10s read/write timeout (microseconds)
    "|reconnect;1"             # tự reconnect
    "|reconnect_streamed;1"    # reconnect cả stream
    "|reconnect_delay_max;2"   # max delay 2s
    "|max_delay;500000"        # giảm latency 0.5s
    "|reorder_queue_size;0"
    "|buffer_size;102400"      # 100KB buffer
)

def ensure_rtsp_env() -> None:
    """
    Chỉ set biến môi trường OPENCV_FFMPEG_CAPTURE_OPTIONS nếu chưa có.
    """
    if "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = DEFAULT_RTSP_OPTS

def set_rtsp_env(options: Optional[str] = None, force: bool = False) -> None:
    """
    Set/cập nhật FFMPEG options cho OpenCV.
    - options: chuỗi option FFMPEG theo format của OpenCV
    - force: True để ghi đè nếu đã tồn tại
    """
    opts = options or DEFAULT_RTSP_OPTS
    if force or "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts

# ---------------------------------------------------------------------
# Mở nguồn video + đọc frame với timeout
# ---------------------------------------------------------------------

def open_capture(
    src: str,
    timeout: float = 8.0,
    try_ffmpeg_first: bool = True,
    fallback_default: bool = True,
) -> cv2.VideoCapture:
    """
    Mở nguồn video (RTSP / file / webcam) với thời hạn retry trong `timeout` giây.
    - Ưu tiên backend FFMPEG (ổn định RTSP), fallback sang backend mặc định (V4L/DSHOW) nếu cần.

    Raises:
        RuntimeError nếu không mở được trong thời gian `timeout`
    """
    ensure_rtsp_env()

    t0 = time.time()
    cap: Optional[cv2.VideoCapture] = None

    while time.time() - t0 < timeout:
        # Thử FFMPEG trước
        if try_ffmpeg_first:
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            if cap.isOpened():
                return cap
            cap.release()

        # Thử backend mặc định
        if fallback_default:
            cap = cv2.VideoCapture(src)
            if cap.isOpened():
                return cap
            cap.release()

        time.sleep(0.25)

    raise RuntimeError(f"Cannot open source: {src}")

def read_one_frame(cap: cv2.VideoCapture, timeout: float = 3.0):
    """
    Đọc 1 frame từ `cap` với timeout.
    Raises:
        RuntimeError nếu hết thời gian mà chưa đọc được frame hợp lệ.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        time.sleep(0.03)
    raise RuntimeError("Cannot read a frame (timeout)")

# ---------------------------------------------------------------------
# Validate RTSP + chụp preview
# ---------------------------------------------------------------------

def validate_rtsp_connect(rtsp_url: str, timeout_sec: float = 5.0) -> bool:
    """
    Kiểm tra mở RTSP và đọc được ít nhất 1 frame trong `timeout_sec`.
    Trả về True/False.
    """
    try:
        cap = open_capture(rtsp_url, timeout=timeout_sec, try_ffmpeg_first=True, fallback_default=False)
    except Exception:
        return False

    ok = False
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.05)
    cap.release()
    return bool(ok)

def capture_preview(rtsp_url: str, save_dir: str) -> str:
    """
    Chụp 1 frame từ RTSP và lưu thành JPEG vào `save_dir`.
    Trả về đường dẫn file đã lưu. Raise RuntimeError nếu thất bại.
    """
    os.makedirs(save_dir, exist_ok=True)

    cap = open_capture(rtsp_url, timeout=6.0, try_ffmpeg_first=True, fallback_default=False)
    ok = False
    frame = None
    for _ in range(15):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.05)
    cap.release()

    if not ok:
        raise RuntimeError("Không đọc được frame preview từ RTSP")

    out_path = os.path.join(save_dir, f"{uuid.uuid4()}.jpg")
    cv2.imwrite(out_path, frame)
    return out_path

# ---------------------------------------------------------------------
# Similarity helpers (HS histogram)
# ---------------------------------------------------------------------

def compute_hist_hs(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Tính histogram 2 kênh H và S (HSV) đã normalize về [0,1].
    Downscale giúp nhanh & ổn định.
    """
    small = cv2.resize(frame_bgr, (320, 180), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # 50 bins H, 60 bins S
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.astype("float32")

def hist_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    So sánh histogram bằng hệ số tương quan (CORRELATION) trong [-1, 1].
    1.0 ~ giống hệt; 0 ~ không tương quan; -1 ~ tương quan âm.
    """
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))

# ---------------------------------------------------------------------
# public exports
# ---------------------------------------------------------------------

__all__ = [
    "DEFAULT_RTSP_OPTS",
    "ensure_rtsp_env",
    "set_rtsp_env",
    "open_capture",
    "read_one_frame",
    "validate_rtsp_connect",
    "capture_preview",
    "compute_hist_hs",
    "hist_similarity",
]
