from miloco import phone_stream


def test_phone_stream_config_defaults_are_sane():
    cfg = phone_stream.PhoneStreamConfig()
    assert cfg.signaling_url.startswith("ws://")
    assert cfg.token
    assert cfg.quest_device_id
    assert cfg.android_device_id
    assert cfg.session_id
    assert 10 <= cfg.jpeg_quality <= 95
    assert cfg.buffer_size >= 3


def test_normalize_config_clamps_quality_and_buffer():
    cfg = phone_stream.PhoneStreamConfig(jpeg_quality=200, buffer_size=1)
    normalized = phone_stream._normalize_config(cfg)
    assert normalized.jpeg_quality == 95
    assert normalized.buffer_size == 3

    cfg2 = phone_stream.PhoneStreamConfig(jpeg_quality=1, buffer_size=1000)
    normalized2 = phone_stream._normalize_config(cfg2)
    assert normalized2.jpeg_quality == 10
    assert normalized2.buffer_size == 120


def test_normalize_config_strips_whitespace_from_url_and_ids():
    cfg = phone_stream.PhoneStreamConfig(
        signaling_url="  ws://example.com:8787  ",
        quest_device_id="  viewer-1  ",
        android_device_id="  phone-1  ",
        session_id="  sess-1  ",
        token="  ",
    )
    normalized = phone_stream._normalize_config(cfg)
    assert normalized.signaling_url == "ws://example.com:8787"
    assert normalized.quest_device_id == "viewer-1"
    assert normalized.android_device_id == "phone-1"
    assert normalized.session_id == "sess-1"
    # token 空白被 strip 后为空, 应该 fallback 到默认值
    assert normalized.token == phone_stream.PhoneStreamConfig.token


def test_get_latest_frames_handles_empty_buffer_and_nonpositive_count():
    capture = phone_stream.PhoneCapture()
    assert capture.get_latest_frames(0) == []
    assert capture.get_latest_frames(-1) == []
    assert capture.get_latest_frames(3) == []


def test_get_latest_frames_returns_newest_n():
    capture = phone_stream.PhoneCapture()
    with capture._lock:
        capture._frame_buffer = [f"frame-{i}".encode() for i in range(5)]
    assert capture.get_latest_frames(3) == [b"frame-2", b"frame-3", b"frame-4"]


def test_get_recent_frames_samples_evenly():
    capture = phone_stream.PhoneCapture()
    with capture._lock:
        capture._frame_buffer = [f"frame-{i}".encode() for i in range(12)]
    # count=4, step=12//4=3, 取 [0,3,6,9]
    assert capture.get_recent_frames(4) == [
        b"frame-0",
        b"frame-3",
        b"frame-6",
        b"frame-9",
    ]


def test_status_reports_running_false_before_start():
    capture = phone_stream.PhoneCapture()
    status = capture.status()
    assert status.running is False
    assert status.signaling_state == "idle"
    assert status.frames == 0
    assert status.latest_frame_age_ms is None
    assert status.signaling_url  # 非空


def test_status_reports_frame_count_and_dimensions_after_receiving():
    capture = phone_stream.PhoneCapture()
    with capture._lock:
        capture._frame_count = 42
        capture._width = 1080
        capture._height = 1920
        capture._fps = 30
        capture._latest_ts = 1234.0
    status = capture.status()
    assert status.frames == 42
    assert status.width == 1080
    assert status.height == 1920
    assert status.fps == 30
    assert status.latest_frame_age_ms is not None  # 有 ts 就有 age


def test_mjpeg_stream_emits_frame_when_available():
    class FakeCapture:
        fps = 30

        def __init__(self):
            self._frames = [b"jpeg-data-1", b"jpeg-data-2"]
            self._idx = 0

        def get_frame(self):
            if self._idx < len(self._frames):
                f = self._frames[self._idx]
                self._idx += 1
                return f
            return None

    stream = phone_stream._mjpeg_stream(FakeCapture())
    chunk = next(stream)
    assert b"jpeg-data-1" in chunk
    assert b"Content-Type: image/jpeg" in chunk
    assert b"--frame" in chunk


def test_load_config_from_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("MILOCO_PHONE_STREAM_SIGNALING_URL", "ws://custom:9999")
    monkeypatch.setenv("MILOCO_PHONE_STREAM_TOKEN", "my-token")
    monkeypatch.setenv("MILOCO_PHONE_STREAM_JPEG_QUALITY", "80")
    cfg = phone_stream._load_config_from_env()
    assert cfg.signaling_url == "ws://custom:9999"
    assert cfg.token == "my-token"
    assert cfg.jpeg_quality == 80


def test_phone_page_route_registered():
    """路由表应包含 phone 端点。"""
    paths = [r.path for r in phone_stream.router.routes]
    assert "/phone" in paths or "/phone/" in paths
    assert "/phone/stream" in paths
    assert "/phone/status" in paths
    assert "/phone/restart" in paths
    assert "/phone/analyze" in paths
