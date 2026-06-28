import pytest

from miloco import screen_monitor


def test_recent_frames_are_sampled_from_existing_buffer():
    capture = screen_monitor.ScreenCapture(fps=3, width=1280)
    with capture._lock:
        capture._frame_buffer = [f"frame-{i}".encode() for i in range(12)]

    frames = capture.get_recent_frames(4)

    assert frames == [b"frame-0", b"frame-3", b"frame-6", b"frame-9"]


def test_mjpeg_stream_uses_latest_frame_without_starting_capture():
    class FakeCapture:
        def __init__(self):
            self.calls = 0

        def get_frame(self):
            self.calls += 1
            return b"\xff\xd8fake-jpeg\xff\xd9"

    stream = screen_monitor._mjpeg_stream(FakeCapture())

    chunk = next(stream)

    assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n")
    assert b"fake-jpeg" in chunk


@pytest.mark.asyncio
async def test_analyze_screen_reuses_capture_buffer(monkeypatch):
    class FakeCapture:
        fps = 3

        def start(self):
            pass

        def get_recent_frames(self, count=9):
            return [b"\xff\xd8a\xff\xd9", b"\xff\xd8b\xff\xd9"]

    async def fake_encode(frames, fps):
        assert len(frames) == 2
        assert fps == 3
        return b"mp4"

    async def fake_call_omni(payload, config):
        assert payload["video_base64"] == "bXA0"
        assert config.model == "minicpm-v46-mlx"
        return {
            "choices": [{"message": {"content": "正在编辑代码。"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }

    monkeypatch.setattr(screen_monitor, "_capture", FakeCapture())
    monkeypatch.setattr(screen_monitor, "_analysis_lock", None)
    monkeypatch.setattr(screen_monitor, "_encode_frames_mp4", fake_encode)
    monkeypatch.setattr(screen_monitor, "call_omni", fake_call_omni)

    result = await screen_monitor.analyze_screen("看一下屏幕")

    assert result["content"] == "正在编辑代码。"
    assert result["tokens"] == {"prompt_tokens": 1, "completion_tokens": 2}
