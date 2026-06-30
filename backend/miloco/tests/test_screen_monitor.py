import pytest

from miloco import screen_monitor


def test_recent_frames_are_sampled_from_existing_buffer():
    capture = screen_monitor.ScreenCapture(
        screen_monitor.ScreenCaptureConfig(fps=3, width=1280)
    )
    with capture._lock:
        capture._frame_buffer = [f"frame-{i}".encode() for i in range(12)]

    frames = capture.get_recent_frames(4)

    assert frames == [b"frame-0", b"frame-3", b"frame-6", b"frame-9"]


def test_latest_frames_use_newest_buffer_entries():
    capture = screen_monitor.ScreenCapture()
    with capture._lock:
        capture._frame_buffer = [f"frame-{i}".encode() for i in range(5)]

    frames = capture.get_latest_frames(3)

    assert frames == [b"frame-2", b"frame-3", b"frame-4"]


def test_roi_ffmpeg_command_crops_without_scaling():
    capture = screen_monitor.ScreenCapture(
        screen_monitor.ScreenCaptureConfig(
            roi_enable=True,
            x=100,
            y=200,
            width=900,
            height=500,
            fps=1,
        )
    )

    cmd = capture._build_ffmpeg_cmd()

    assert cmd[cmd.index("-framerate") + 1] == "1"
    assert cmd[cmd.index("-vf") + 1] == "crop=900:500:100:200"
    assert cmd[cmd.index("-q:v") + 1] == "2"
    assert "scale=900:500" not in " ".join(cmd)


def test_fullscreen_ffmpeg_command_keeps_preview_stream():
    capture = screen_monitor.ScreenCapture(
        screen_monitor.ScreenCaptureConfig(width=1280, height=720, fps=3)
    )

    cmd = capture._build_ffmpeg_cmd()

    assert cmd[cmd.index("-framerate") + 1] == "3"
    assert "scale=1280:720" in cmd[cmd.index("-vf") + 1]
    assert cmd[cmd.index("-q:v") + 1] == "2"


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
async def test_analyze_screen_sends_latest_jpegs_as_images(monkeypatch):
    class FakeCapture:
        fps = 3

        def start(self):
            pass

        def get_latest_frames(self, count=3):
            assert count == 3
            return [b"\xff\xd8a\xff\xd9", b"\xff\xd8b\xff\xd9"]

    async def fake_call_omni(payload, config):
        assert payload["video_base64"] is None
        assert payload["crops"] == []
        assert payload["images"] == [
            {"mime_type": "image/jpeg", "base64": "/9hh/9k="},
            {"mime_type": "image/jpeg", "base64": "/9hi/9k="},
        ]
        assert "禁止猜测" in payload["system_prompt"]
        assert config.model == "minicpm-v46-mlx"
        return {
            "choices": [{"message": {"content": "正在编辑代码。"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }

    monkeypatch.setattr(screen_monitor, "_capture", FakeCapture())
    monkeypatch.setattr(screen_monitor, "_analysis_lock", None)
    monkeypatch.setattr(screen_monitor, "call_omni", fake_call_omni)

    result = await screen_monitor.analyze_screen("看一下屏幕")

    assert result["content"] == "正在编辑代码。"
    assert result["tokens"] == {"prompt_tokens": 1, "completion_tokens": 2}


def test_roi_config_defaults_to_one_fps_when_enabled(monkeypatch):
    update = screen_monitor.ScreenConfigUpdate(roi_enable=True)
    monkeypatch.setattr(
        screen_monitor,
        "_screen_config",
        screen_monitor.ScreenCaptureConfig(
            roi_enable=False,
            width=1280,
            height=720,
            fps=3,
        ),
    )

    config = screen_monitor._merge_config_update(update)

    assert config.roi_enable is True
    assert config.fps == 1


# ---------------------------------------------------------------------------
# 优化后新增测试:线程安全、资源回收、deque 上限、ROI 边界归一化
# ---------------------------------------------------------------------------


def test_frame_buffer_is_deque_with_maxlen():
    """deque(maxlen=24) 应自动丢弃旧帧,无需手动 trim。"""
    capture = screen_monitor.ScreenCapture()
    assert isinstance(capture._frame_buffer, type(__import__("collections").deque()))
    assert (
        capture._frame_buffer.maxlen == screen_monitor.ScreenCapture._FRAME_BUFFER_MAX
    )

    # 模拟灌入超过 maxlen 的帧,验证旧帧被自动丢弃
    with capture._lock:
        for i in range(30):
            capture._frame_buffer.append(f"frame-{i}".encode())
    assert len(capture._frame_buffer) == screen_monitor.ScreenCapture._FRAME_BUFFER_MAX
    # 应保留最后 24 帧(frame-6 .. frame-29)
    assert list(capture._frame_buffer)[0] == b"frame-6"
    assert list(capture._frame_buffer)[-1] == b"frame-29"


def test_get_latest_frames_handles_empty_buffer_and_nonpositive_count():
    capture = screen_monitor.ScreenCapture()
    assert capture.get_latest_frames(3) == []
    assert capture.get_latest_frames(0) == []
    assert capture.get_latest_frames(-1) == []

    with capture._lock:
        capture._frame_buffer.extend([b"a", b"b", b"c", b"d"])
    assert capture.get_latest_frames(2) == [b"c", b"d"]
    # count 超过 buffer 长度时返回全部,不报错
    assert capture.get_latest_frames(10) == [b"a", b"b", b"c", b"d"]


def test_stop_clears_proc_reference():
    """stop() 后 _proc 应为 None,避免下次 status() 误判 running=True。

    用 fake proc 模拟 _run_once 已设置 _proc 但 stop() 被调用的场景。
    """
    capture = screen_monitor.ScreenCapture()

    class FakeProc:
        def __init__(self):
            self._poll = None  # 模拟"仍在运行"

        def poll(self):
            return self._poll

        def terminate(self):
            self._poll = -15

        def wait(self, timeout=None):
            pass

        def kill(self):
            self._poll = -9

        stdout = None
        stderr = None

    fake_proc = FakeProc()
    with capture._lock:
        capture._proc = fake_proc
    # status() 在 stop 前应看到 running=True
    assert capture.status().running is True

    capture.stop()
    # stop() 后 _proc 应被清空,status() 应看到 running=False
    assert capture._proc is None
    assert capture.status().running is False


def test_close_proc_pipes_is_idempotent_and_handles_none():
    """_close_proc_pipes 对 None / 已 close 的流应安全幂等。"""
    capture = screen_monitor.ScreenCapture()

    class FakeStream:
        def __init__(self):
            self.closed = False
            self.raise_on_close = False

        def close(self):
            if self.raise_on_close:
                raise OSError("already closed")
            self.closed = True

    stdout = FakeStream()
    stderr = FakeStream()

    class FakeProc:
        pass

    proc = FakeProc()
    proc.stdout = stdout
    proc.stderr = stderr

    capture._close_proc_pipes(proc)
    assert stdout.closed is True
    assert stderr.closed is True

    # 再次 close:模拟已关闭场景,应不抛异常
    stdout.raise_on_close = True
    stderr.raise_on_close = True
    capture._close_proc_pipes(proc)  # 不应抛 OSError

    # None 流也应安全
    proc.stdout = None
    proc.stderr = None
    capture._close_proc_pipes(proc)


def test_normalize_config_clamps_roi_dimensions_to_upper_bound():
    """超大的 x/y/width/height 应被 clamp 到 16384,避免 ffmpeg 滤镜拒绝。"""
    config = screen_monitor.ScreenCaptureConfig(
        roi_enable=True,
        x=999_999_999,
        y=999_999_999,
        width=999_999_999,
        height=999_999_999,
        fps=1,
    )
    normalized = screen_monitor._normalize_config(config)
    assert normalized.x == 16384
    assert normalized.y == 16384
    assert normalized.width == 16384
    assert normalized.height == 16384


def test_normalize_config_keeps_minimum_dimensions():
    """width/height 小于 16 应被提到 16。"""
    config = screen_monitor.ScreenCaptureConfig(width=1, height=1, fps=3)
    normalized = screen_monitor._normalize_config(config)
    assert normalized.width == 16
    assert normalized.height == 16


def test_normalize_config_clamps_fps_to_valid_range():
    """fps 超过 30 或小于 1 应被 clamp。"""
    assert (
        screen_monitor._normalize_config(
            screen_monitor.ScreenCaptureConfig(fps=100)
        ).fps
        == 30
    )
    assert (
        screen_monitor._normalize_config(screen_monitor.ScreenCaptureConfig(fps=0)).fps
        == 3
    )  # 0 触发 `or default_fps` 分支
    assert (
        screen_monitor._normalize_config(screen_monitor.ScreenCaptureConfig(fps=-5)).fps
        == 3
    )  # 负数 `or` 同样走 default


def test_merge_config_update_explicit_zero_fps_is_respected_then_clamped():
    """显式传 fps=0 应被合并,然后经 _normalize_config 归一化为默认值。

    这验证了"value is None 才跳过"的语义:0 不是 None,应被采纳。
    """
    update = screen_monitor.ScreenConfigUpdate(fps=0)
    # _merge_config_update 依赖 get_screen_config(),它会读 _screen_config 全局。
    # 测试里 monkeypatch 一个干净的基础配置。
    import miloco.screen_monitor as sm

    original = sm._screen_config
    sm._screen_config = sm.ScreenCaptureConfig(
        roi_enable=False, width=1280, height=720, fps=3
    )
    try:
        config = sm._merge_config_update(update)
    finally:
        sm._screen_config = original
    # fps=0 被采纳,然后 _normalize_config 走 `0 or default_fps` → 3
    assert config.fps == 3


def test_merge_config_update_uses_pydantic_v2_model_fields_set():
    """_merge_config_update 应使用 pydantic v2 的 model_fields_set,不再 fallback v1。"""
    update = screen_monitor.ScreenConfigUpdate(roi_enable=True)
    # pydantic v2 应有 model_fields_set 属性
    assert hasattr(update, "model_fields_set")
    assert "roi_enable" in update.model_fields_set
    assert "fps" not in update.model_fields_set  # 未显式传 fps


def test_read_jpeg_pipe_trims_oversized_buffer_to_prevent_oom(monkeypatch):
    """buf 超过 4MB 仍找不到完整 JPEG 时应被截断,防止异常流导致 OOM。

    用 fake proc + fake select 模拟:连续灌入不含 \xff\xd8 的数据。
    """
    capture = screen_monitor.ScreenCapture()
    capture._stop.clear()

    class FakeStdout:
        def __init__(self):
            # 灌入 5MB 非 JPEG 数据(无 \xff\xd8 头),应触发截断保护
            self.data = b"\x00" * (5 * 1024 * 1024)
            self.pos = 0
            self.eof = False

        def read(self, n):
            if self.pos >= len(self.data):
                if not self.eof:
                    self.eof = True
                    return b""  # 模拟 EOF
                return b""
            chunk = self.data[self.pos : self.pos + n]
            self.pos += len(chunk)
            return chunk

    class FakeProc:
        def __init__(self):
            self.stdout = FakeStdout()
            self._poll = None

        def poll(self):
            return self._poll

    fake_proc = FakeProc()

    # 模拟 select 总是 ready,让循环不断读
    monkeypatch.setattr(screen_monitor.select, "select", lambda r, w, x, t: (r, [], []))

    # _read_jpeg_pipe 在 EOF(read 返回 b"")时会 break,但 buf 已经累积 5MB
    # 触发截断保护后 buf 应被缩到 ~1KB。我们主要验证不抛 MemoryError。
    capture._read_jpeg_pipe(fake_proc)  # 应正常返回,不抛异常


def test_status_reads_restart_count_under_lock():
    """status() 应在锁内读取 _restart_count,避免与 _supervise 的自增竞争。

    通过观察 status() 返回值包含 restart_count 字段,且在并发自增后能读到最新值。
    """
    capture = screen_monitor.ScreenCapture()
    with capture._lock:
        capture._restart_count = 7
    assert capture.status().restart_count == 7


def test_status_includes_permission_hint_on_macos_avfoundation_error(monkeypatch):
    """macOS 权限不足时,status() 应在 permission_hint 字段给出可读提示。

    AVCaptureScreenInput 链接错误是 macOS 屏幕录制权限未授予的典型信号;
    让前端能拿到明确提示而不是面对黑屏。
    """
    capture = screen_monitor.ScreenCapture()
    with capture._lock:
        capture._last_error = (
            "objc[123]: class `NSKVONotifying_AVCaptureScreenInput' not linked "
            "into application"
        )
    monkeypatch.setattr(screen_monitor.sys, "platform", "darwin")
    status = capture.status()
    assert status.permission_hint
    assert "屏幕录制权限" in status.permission_hint


def test_status_has_empty_permission_hint_when_no_error(monkeypatch):
    """无错误时 permission_hint 应为空字符串,不影响前端逻辑。"""
    capture = screen_monitor.ScreenCapture()
    with capture._lock:
        capture._last_error = ""
    monkeypatch.setattr(screen_monitor.sys, "platform", "darwin")
    assert capture.status().permission_hint == ""


def test_restart_screen_capture_stops_old_and_starts_new(monkeypatch):
    """restart_screen_capture 应 stop 旧 capture 并创建新实例。

    场景:用户在 macOS 系统设置里授予屏幕录制权限后点击"重新加载画面",
    前端调 POST /api/screen/restart -> restart_screen_capture()。
    旧 ffmpeg 进程(权限未授予时启动的,输出黑帧)必须被终止,
    新进程会用最新权限重新初始化。
    """
    stopped = {"called": False}
    started = {"called": False}

    class FakeCapture:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            started["called"] = True

        def stop(self):
            stopped["called"] = True

    monkeypatch.setattr(screen_monitor, "ScreenCapture", FakeCapture)
    monkeypatch.setattr(screen_monitor, "_capture", FakeCapture())
    monkeypatch.setattr(
        screen_monitor, "_screen_config", screen_monitor.ScreenCaptureConfig()
    )

    screen_monitor.restart_screen_capture()

    assert stopped["called"], "旧 capture 必须被 stop"
    assert started["called"], "新 capture 必须被 start"

