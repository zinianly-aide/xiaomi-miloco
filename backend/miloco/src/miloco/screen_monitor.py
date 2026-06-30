"""Local screen capture, MJPEG preview, and VLM analysis endpoints."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import select
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from miloco.middleware import verify_token, verify_token_query_fallback
from miloco.perception.engine.config import OmniConfig
from miloco.perception.engine.omni.omni_client import call_omni
from miloco.utils.common import escape_for_js_string

logger = logging.getLogger(__name__)


@dataclass
class ScreenCaptureConfig:
    roi_enable: bool = False
    x: int = 0
    y: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 3
    monitor: int = 0


@dataclass
class ScreenCaptureStatus:
    running: bool
    fps: int
    width: int
    height: int
    monitor: int
    frames: int
    latest_frame_age_ms: int | None
    restart_count: int
    last_error: str
    roi_enable: bool
    x: int
    y: int
    permission_hint: str = ""


class ScreenCapture:
    """Single ffmpeg screen capture session shared by MJPEG and VLM analysis."""

    # 帧缓冲环形容量：MJPEG 预览只看最新一帧，VLM 分析最多取 3 帧，
    # 24 帧足够覆盖分析窗口 + 容忍短时停滞，再多就是无谓内存。
    _FRAME_BUFFER_MAX = 24

    def __init__(self, config: ScreenCaptureConfig | None = None):
        self.config = _normalize_config(config or ScreenCaptureConfig())
        self.fps = self.config.fps
        self.width = self.config.width
        self.height = self.config.height
        self.monitor = self.config.monitor
        self._lock = threading.Lock()
        self._latest_frame: bytes | None = None
        self._latest_ts: float | None = None
        # deque(maxlen=...) 自动丢弃旧帧,无需手动 del 切片,且 append/pop 均为 O(1)。
        self._frame_buffer: deque[bytes] = deque(maxlen=self._FRAME_BUFFER_MAX)
        self._frame_count = 0
        self._restart_count = 0
        self._last_error = ""
        # _proc / _stderr_thread 在工作线程(_supervise→_run_once)里写,
        # 在主线程(start/stop/status)里读 → 所有访问必须经 _lock 保护。
        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr_thread: threading.Thread | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._supervise,
                name="screen-capture",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            proc = self._proc
            stderr_thread = self._stderr_thread
            thread = self._thread
        if proc:
            self._terminate_proc(proc)
        # 等 stderr drainer 退出,避免它在 stop 后还 readline() 阻塞残留。
        # ffmpeg 被 terminate 后 stderr 会 EOF,readline 返回 b"" 退出循环。
        if stderr_thread and stderr_thread.is_alive():
            stderr_thread.join(timeout=2)
        if thread and thread.is_alive():
            thread.join(timeout=3)
        with self._lock:
            # 清空引用,避免下次 start() 时 status() 误判 running=True。
            self._proc = None
            self._stderr_thread = None

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self._latest_frame

    def get_recent_frames(self, count: int = 9) -> list[bytes]:
        with self._lock:
            frames = list(self._frame_buffer)
        if len(frames) <= count:
            return frames
        step = max(1, len(frames) // count)
        return frames[::step][:count]

    def get_latest_frames(self, count: int = 3) -> list[bytes]:
        with self._lock:
            # deque 切片返回 list,但 -0 会被当作 0;显式处理 count<=0 与空 buffer。
            if count <= 0 or not self._frame_buffer:
                return []
            return list(self._frame_buffer)[-count:]

    def status(self) -> ScreenCaptureStatus:
        with self._lock:
            latest_ts = self._latest_ts
            frames = self._frame_count
            last_error = self._last_error
            restart_count = self._restart_count
            proc = self._proc
            config = self.config
        # poll() 是 syscall,放锁外避免持锁阻塞过久。
        running = bool(proc and proc.poll() is None)
        # macOS 屏幕录制权限不足时 ffmpeg 能启动但画面全黑或只有提示信息。
        # 在 last_error 里捕获到 AVCaptureScreenInput 相关链接警告时,
        # 给前端一个明确的权限提示,而不是让用户面对黑屏无从下手。
        permission_hint = ""
        if sys.platform == "darwin" and "AVCaptureScreenInput" in last_error:
            permission_hint = (
                "macOS 屏幕录制权限未授予当前进程。"
                "请前往 系统设置 > 隐私与安全 > 屏幕录制,"
                "允许运行本服务的终端/应用,然后刷新页面。"
            )
        elif "not linked into application" in last_error:
            permission_hint = "macOS 屏幕录制链接异常。请检查系统权限并重启服务。"
        return ScreenCaptureStatus(
            running=running,
            fps=self.fps,
            width=self.width,
            height=self.height,
            monitor=self.monitor,
            frames=frames,
            latest_frame_age_ms=(
                int((time.monotonic() - latest_ts) * 1000)
                if latest_ts is not None
                else None
            ),
            restart_count=restart_count,
            last_error=last_error,
            roi_enable=config.roi_enable,
            x=config.x,
            y=config.y,
            permission_hint=permission_hint,
        )

    def _supervise(self) -> None:
        while not self._stop.is_set():
            try:
                self._run_once()
            except Exception as e:  # noqa: BLE001
                with self._lock:
                    self._last_error = f"{type(e).__name__}: {e}"
                logger.warning("screen capture failed: %s", e)
            if not self._stop.is_set():
                with self._lock:
                    self._restart_count += 1
                time.sleep(2)

    def _run_once(self) -> None:
        cmd = self._build_ffmpeg_cmd()
        logger.info(
            "starting screen capture: monitor=%s %sx%s @ %sfps roi=%s",
            self.monitor,
            self.width,
            self.height,
            self.fps,
            self.config.roi_enable,
        )
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(proc,),
            name="screen-capture-stderr",
            daemon=True,
        )
        with self._lock:
            self._proc = proc
            self._stderr_thread = stderr_thread
        stderr_thread.start()
        try:
            self._read_jpeg_pipe(proc)
        finally:
            if proc.poll() is None:
                self._terminate_proc(proc)
            self._close_proc_pipes(proc)
            rc = proc.poll()
            with self._lock:
                self._last_error = "" if self._stop.is_set() else f"ffmpeg exited: {rc}"
                if self._proc is proc:
                    self._proc = None
                if self._stderr_thread is stderr_thread:
                    self._stderr_thread = None

    def _build_ffmpeg_cmd(self) -> list[str]:
        if self.config.roi_enable:
            vf = f"crop={self.width}:{self.height}:{self.config.x}:{self.config.y}"
        else:
            vf = (
                f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,"
                f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2"
            )

        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "avfoundation",
            "-framerate",
            str(self.fps),
            "-pixel_format",
            "bgr0",
            "-i",
            f"{self.monitor}:none",
            "-vf",
            vf,
            "-c:v",
            "mjpeg",
            "-q:v",
            "2",
            "-an",
            "-nostdin",
            "-f",
            "image2pipe",
            "pipe:1",
        ]

    def _terminate_proc(self, proc: subprocess.Popen[bytes]) -> None:
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
            return
        except subprocess.TimeoutExpired:
            pass
        except OSError:
            return
        try:
            proc.kill()
            proc.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            logger.warning("failed to stop screen capture ffmpeg process")

    def _close_proc_pipes(self, proc: subprocess.Popen[bytes]) -> None:
        """Close stdout/stderr pipe file descriptors to avoid fd leakage.

        Popen 不会自动 close PIPE 创建的文件对象;每次 _run_once 创建一个新
        ffmpeg 进程,若不 close,长时间运行后进程 fd 表会膨胀,最终触发
        ``Too many open files``。此函数幂等,可重复调用。
        """
        for stream in (proc.stdout, proc.stderr):
            if stream is None:
                continue
            try:
                stream.close()
            except OSError:
                # 已经 close 或被 GC 关闭过,忽略。
                pass

    def _drain_stderr(self, proc: subprocess.Popen[bytes]) -> None:
        if proc.stderr is None:
            return
        while not self._stop.is_set() and proc.poll() is None:
            line = proc.stderr.readline()
            if not line:
                break
            text = line.decode(errors="replace").strip()
            if text:
                with self._lock:
                    self._last_error = text[-500:]

    def _read_jpeg_pipe(self, proc: subprocess.Popen[bytes]) -> None:
        if proc.stdout is None:
            raise RuntimeError("ffmpeg stdout pipe was not created")
        buf = b""
        first_frame_deadline = time.monotonic() + 8.0
        got_first_frame = False
        while not self._stop.is_set() and proc.poll() is None:
            ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            if not ready:
                if not got_first_frame and time.monotonic() > first_frame_deadline:
                    with self._lock:
                        self._last_error = (
                            "ffmpeg started but produced no screen frames"
                        )
                    proc.terminate()
                    break
                continue
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            # 防 buf 在异常流(非 JPEG 数据)下无限增长:超过 4MB 仍找不到完整帧
            # 就丢弃前半部分,保留尾部(可能含下个帧头)。
            if len(buf) > 4 * 1024 * 1024:
                buf = buf[-1024:]
            while True:
                start = buf.find(b"\xff\xd8")
                end = buf.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                if start < 0 or end <= start:
                    break
                frame = buf[start : end + 2]
                buf = buf[end + 2 :]
                got_first_frame = True
                with self._lock:
                    self._latest_frame = frame
                    self._latest_ts = time.monotonic()
                    self._frame_count += 1
                    # deque(maxlen=...) 自动丢弃最旧帧,无需手动 trim。
                    self._frame_buffer.append(frame)


_capture: ScreenCapture | None = None
_capture_lock = threading.RLock()
_screen_config: ScreenCaptureConfig | None = None
_analysis_lock: asyncio.Lock | None = None
_proxy_env_lock = threading.Lock()
_last_analysis: dict = {
    "content": "等待分析",
    "time": "",
    "elapsed": 0,
    "tokens": {},
}

_PROXY_ENV_KEYS = (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
)
_NO_PROXY_ENV_KEYS = ("NO_PROXY", "no_proxy")


def _is_loopback_url(url: str) -> bool:
    try:
        host = urlparse(url).hostname
    except Exception:
        return False
    return host in {"127.0.0.1", "::1", "localhost"}


@contextmanager
def _without_proxy_env_for_loopback(url: str):
    if not _is_loopback_url(url):
        yield
        return
    with _proxy_env_lock:
        env_keys = (*_PROXY_ENV_KEYS, *_NO_PROXY_ENV_KEYS)
        old = {key: os.environ.get(key) for key in env_keys}
        try:
            for key in _PROXY_ENV_KEYS:
                os.environ.pop(key, None)
            for key in _NO_PROXY_ENV_KEYS:
                os.environ[key] = "127.0.0.1,localhost,::1"
            yield
        finally:
            for key in env_keys:
                value = old[key]
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _screen_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _screen_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_config(config: ScreenCaptureConfig) -> ScreenCaptureConfig:
    roi_enable = bool(config.roi_enable)
    default_fps = 1 if roi_enable else 3
    # 上限取 16384(8K 宽度),超过几乎肯定是配置错误或恶意输入;
    # 下限 16 是 ffmpeg crop 滤镜的合理最小尺寸。
    max_dim = 16384
    # fps<=0(包括显式 0、负数、None 经 int() 后)一律走 default_fps,
    # 再 clamp 到 [1, 30]。原来用 `fps or default_fps` 只在 fps==0 时触发默认,
    # 负数会被当 truthy 保留,然后被 clamp 到 1——与"负数=用默认"的直觉不符。
    raw_fps = int(config.fps) if config.fps is not None else 0
    fps = default_fps if raw_fps <= 0 else max(1, min(30, raw_fps))
    return ScreenCaptureConfig(
        roi_enable=roi_enable,
        x=max(0, min(int(config.x), max_dim)),
        y=max(0, min(int(config.y), max_dim)),
        width=max(16, min(int(config.width), max_dim)),
        height=max(16, min(int(config.height), max_dim)),
        fps=fps,
        monitor=max(0, int(config.monitor)),
    )


def _load_screen_config_from_env() -> ScreenCaptureConfig:
    roi_enable = _screen_bool_env("MILOCO_SCREEN_ROI_ENABLE", False)
    return _normalize_config(
        ScreenCaptureConfig(
            roi_enable=roi_enable,
            x=_screen_int_env("MILOCO_SCREEN_X", 0),
            y=_screen_int_env("MILOCO_SCREEN_Y", 0),
            width=_screen_int_env("MILOCO_SCREEN_WIDTH", 1280),
            height=_screen_int_env("MILOCO_SCREEN_HEIGHT", 720),
            fps=_screen_int_env("MILOCO_SCREEN_FPS", 1 if roi_enable else 3),
            monitor=_screen_int_env("MILOCO_SCREEN_MONITOR", 0),
        )
    )


def get_screen_config() -> ScreenCaptureConfig:
    global _screen_config
    with _capture_lock:
        if _screen_config is None:
            _screen_config = _load_screen_config_from_env()
        return _screen_config


def update_screen_config(config: ScreenCaptureConfig) -> ScreenCaptureConfig:
    global _capture, _screen_config
    normalized = _normalize_config(config)
    with _capture_lock:
        old_capture = _capture
        _capture = None
        _screen_config = normalized
    if old_capture is not None:
        old_capture.stop()
        get_screen_capture()
    return normalized


def get_screen_capture() -> ScreenCapture:
    global _capture
    with _capture_lock:
        if _capture is None:
            _capture = ScreenCapture(get_screen_config())
        _capture.start()
        return _capture


def restart_screen_capture() -> None:
    """Stop and re-create the capture so ffmpeg re-requests macOS screen permission.

    macOS 屏幕录制权限是 per-process 的:如果 ffmpeg 启动时权限还没授予,
    它会一直生成黑帧;即使用户在系统设置里授权后,已运行的 ffmpeg 进程
    也不会自动重试。必须终止旧进程、重新拉起一个,新进程才会触发系统
    重新检查权限并开始输出真实画面。

    场景:用户刚在 系统设置 > 隐私与安全 > 屏幕录制 授权了运行本服务的
    终端/应用,然后点击页面上的"重新加载画面"按钮——前端会先调
    ``POST /api/screen/restart`` 触发此函数,再重新拉 MJPEG 流。
    """
    global _capture
    with _capture_lock:
        old = _capture
        _capture = None
    if old is not None:
        old.stop()
    # 重建并立即 start,避免下一帧 status() 看到 running=False。
    # 新 ffmpeg 进程会用最新配置启动,并重新走 macOS 权限检查。
    get_screen_capture()


def shutdown_screen_capture() -> None:
    global _capture
    with _capture_lock:
        capture = _capture
        _capture = None
    if capture is not None:
        capture.stop()


def _get_analysis_lock() -> asyncio.Lock:
    global _analysis_lock
    if _analysis_lock is None:
        _analysis_lock = asyncio.Lock()
    return _analysis_lock


async def analyze_screen(query: str) -> dict:
    """Analyze recent frames from the shared screen capture buffer."""

    global _last_analysis
    capture = get_screen_capture()
    async with _get_analysis_lock():
        frames = capture.get_latest_frames(3)
        if not frames:
            _last_analysis = {
                "content": f"帧缓冲区不足，当前只有 {len(frames)} 帧，请稍后重试。",
                "time": time.strftime("%H:%M:%S"),
                "elapsed": 0,
                "tokens": {},
            }
            return dict(_last_analysis)

        try:
            images = [
                {
                    "mime_type": "image/jpeg",
                    "base64": base64.b64encode(frame).decode("ascii"),
                }
                for frame in frames
            ]
            payload = {
                "system_prompt": (
                    "你是屏幕识别助手。\n"
                    "你只能根据图像中清晰可见的内容回答。\n"
                    "如果文字、按钮、窗口名称看不清，必须写“无法识别”。\n"
                    "禁止猜测。\n"
                    "禁止根据上下文脑补。\n"
                    "禁止说图中不存在的按钮。\n"
                    "请按以下格式回答：\n\n"
                    "1. 当前窗口：\n"
                    "2. 可见文字：\n"
                    "3. 当前操作：\n"
                    "4. 无法确认内容："
                ),
                "user_content": query
                or "当前屏幕打开了哪些应用和窗口？请列出名称和它们大致在做什么。",
                "video_base64": None,
                "crops": [],
                "images": images,
            }
            config = OmniConfig(
                model=os.environ.get("MILOCO_SCREEN_VLM_MODEL", "minicpm-v46-mlx"),
                base_url=os.environ.get(
                    "MILOCO_SCREEN_VLM_BASE_URL",
                    "http://127.0.0.1:8001/v1",
                ),
                api_key=os.environ.get("MILOCO_SCREEN_VLM_API_KEY", "local-minicpm"),
                timeout=float(os.environ.get("MILOCO_SCREEN_VLM_TIMEOUT", "90")),
                max_completion_tokens=int(
                    os.environ.get("MILOCO_SCREEN_VLM_MAX_TOKENS", "256")
                ),
            )
            t0 = time.monotonic()
            with _without_proxy_env_for_loopback(config.base_url):
                resp = await call_omni(payload, config)
            elapsed = time.monotonic() - t0
            _last_analysis = {
                "content": resp["choices"][0]["message"]["content"],
                "time": time.strftime("%H:%M:%S"),
                "elapsed": round(elapsed, 1),
                "tokens": resp.get("usage", {}),
            }
        except Exception as e:  # noqa: BLE001
            logger.warning("screen VLM analysis failed: %s", e)
            _last_analysis = {
                "content": f"分析失败: {type(e).__name__}: {e}",
                "time": time.strftime("%H:%M:%S"),
                "elapsed": 0,
                "tokens": {},
            }
        return dict(_last_analysis)


def _mjpeg_stream(capture: ScreenCapture) -> Iterator[bytes]:
    last_frame = b""
    # 80ms 轮询,但 ffmpeg 只有 fps=3 时大约 333ms 才产一帧;
    # 太快会空转,太慢会让浏览器卡顿。按 fps 动态调整,最高 20fps(50ms)。
    interval = max(0.05, 1.0 / max(1, getattr(capture, "fps", 5) * 1.2))
    empty_streak = 0
    while True:
        frame = capture.get_frame()
        if frame and frame != last_frame:
            last_frame = frame
            empty_streak = 0
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + str(len(frame)).encode("ascii")
                + b"\r\n\r\n"
                + frame
                + b"\r\n"
            )
        else:
            empty_streak += 1
        # 持续无新帧说明采集停滞或画面静止,不要空轮询太久;
        # 90 次(约 7.2s 当 fps=3) 后适当睡长一点,避免 CPU 空转。
        if empty_streak > 90:
            time.sleep(0.3)
        else:
            time.sleep(interval)


class ScreenConfigUpdate(BaseModel):
    roi_enable: bool | None = None
    x: int | None = None
    y: int | None = None
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    monitor: int | None = None


def _merge_config_update(update: ScreenConfigUpdate) -> ScreenCaptureConfig:
    current = asdict(get_screen_config())
    # pydantic v2 用 model_fields_set 记录客户端显式传入的字段;
    # 项目要求 pydantic>=2.13,不再保留 v1 __fields_set__ fallback。
    provided: set[str] = set(update.model_fields_set)
    for key in current:
        value = getattr(update, key, None)
        # 注意:value is None 时跳过——这同时覆盖"字段未传"和"显式传 null"两种情况,
        # 对当前所有字段(None 都不是合法值)语义一致。若未来加可空字段需重新审视。
        if value is not None:
            current[key] = value
    # roi_enable 切换时,若用户没显式传 fps,自动套用 ROI=1fps / 全屏=3fps 默认值,
    # 避免开 ROI 后还跑 3fps 拖性能。
    if "roi_enable" in provided and "fps" not in provided:
        current["fps"] = 1 if current["roi_enable"] else 3
    return _normalize_config(ScreenCaptureConfig(**current))


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Miloco Screen Monitor</title>
<style>
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:#0b0c0f;color:#e8eaed;font:14px/1.5 system-ui,-apple-system,BlinkMacSystemFont,sans-serif}
body{display:flex;min-height:100vh;overflow:hidden}
.preview{flex:1;min-width:0;background:#000;display:flex;flex-direction:column}
.bar{height:40px;display:flex;align-items:center;justify-content:space-between;padding:0 14px;background:#111318;border-bottom:1px solid #242833;color:#9aa0aa}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:8px;background:#17c964}.dot.dead{background:#ef4444}
.preview-wrap{flex:1;position:relative;min-height:0;background:#000;display:flex;align-items:center;justify-content:center;overflow:hidden}
.preview img{max-width:100%;max-height:100%;object-fit:contain;background:#000}
.preview-overlay{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;background:rgba(0,0,0,.85);color:#e8eaed;padding:20px;text-align:center;pointer-events:none;opacity:0;transition:opacity .15s}
.preview-overlay.show{opacity:1;pointer-events:auto}
.preview-overlay .big{font-size:16px;margin-bottom:8px}
.preview-overlay .small{font-size:12px;color:#9aa0aa;max-width:420px;line-height:1.6}
.preview-overlay button{margin-top:14px;padding:8px 14px;border-radius:6px;border:0;background:#2563eb;color:#fff;font:inherit;font-weight:600;cursor:pointer;pointer-events:auto}
.panel{width:340px;max-width:36vw;background:#151821;border-left:1px solid #2b3040;display:flex;flex-direction:column}
.head{padding:12px 14px;border-bottom:1px solid #2b3040;font-weight:700;font-size:15px}
.controls{padding:10px 14px;border-bottom:1px solid #2b3040;flex:0 0 auto}
.controls select,.controls input{width:100%;margin-bottom:8px;padding:7px 9px;background:#0f1117;border:1px solid #343a4a;border-radius:6px;color:#e8eaed;font:inherit}
.controls button{width:100%;padding:8px 10px;border:0;border-radius:6px;background:#2563eb;color:#fff;font-weight:600;cursor:pointer;font-size:13px}
.controls button:disabled{background:#3b4252;color:#9aa0aa;cursor:wait}
.controls button.secondary{background:#374151!important;margin-top:6px}
.roi-header{margin-top:8px;display:flex;align-items:center;justify-content:space-between;gap:8px}
.roi-header button{flex:0 0 auto;width:auto;padding:6px 10px;background:transparent;border:1px solid #343a4a;color:#9aa0aa;font-size:12px;font-weight:500}
.roi{background:#0f1117;border:1px solid #23283a;border-radius:6px;padding:10px;margin-top:6px;display:none;grid-template-columns:1fr 1fr;gap:8px}
.roi.open{display:grid}
.roi label{font-size:11px;color:#9aa0aa}.roi input{margin:4px 0 0;padding:6px 8px}.roi input[type=checkbox]{width:auto;margin-right:6px}.roi .wide{grid-column:1/-1}
.hint{font-size:12px;color:#9aa0aa;margin-top:8px;line-height:1.5}
.hint.error{color:#f87171}
.hint.warn{color:#facc15}
.result{flex:1;min-height:160px;overflow:auto;padding:12px 14px;background:#0f1117}.meta{color:#9aa0aa;font-size:12px;margin-bottom:8px}.content{white-space:pre-wrap;word-break:break-word;font-size:14px;line-height:1.7}.status{padding:9px 14px;border-top:1px solid #2b3040;color:#9aa0aa;font-size:12px;min-height:36px}
@media(max-width:840px){body{flex-direction:column}.panel{width:100%;max-width:none;height:48vh;border-left:0;border-top:1px solid #2b3040}.preview{height:52vh}.bar{height:auto;min-height:38px}}
</style>
</head>
<body>
<main class="preview">
  <div class="bar">
    <span><span id="dot" class="dot"></span>屏幕实时采集</span>
    <span id="frame">等待画面</span>
  </div>
  <div class="preview-wrap">
    <img id="screen" alt="屏幕画面">
    <div id="overlay" class="preview-overlay">
      <div class="big" id="overlayTitle">画面加载中</div>
      <div class="small" id="overlayText">正在连接 MJPEG 推流...</div>
      <button id="reloadStream">重新加载画面</button>
    </div>
  </div>
</main>
<aside class="panel">
  <div class="head">VLM 屏幕分析</div>
  <div class="controls">
    <select id="preset">
      <option value="当前屏幕打开了哪些应用和窗口？请列出名称和它们大致在做什么。">列出应用窗口</option>
      <option value="屏幕上的主要内容是什么？有什么值得注意的信息？请简洁描述。">内容摘要</option>
      <option value="当前屏幕显示的是什么类型的页面或应用？是工作内容、娱乐、通讯还是其他？">识别场景类型</option>
      <option value="屏幕中有没有代码、终端、文档编辑等开发相关的内容？">开发检测</option>
      <option value="custom">自定义提问</option>
    </select>
    <input id="query" placeholder="自定义提问">
    <button id="analyze">分析当前屏幕</button>
    <div id="hint" class="hint"></div>
    <div class="roi-header">
      <span style="font-size:12px;color:#9aa0aa">ROI 区域设置</span>
      <button id="roiToggle" type="button">展开</button>
    </div>
    <div id="roiBox" class="roi">
      <label class="wide"><input id="roiEnable" type="checkbox"> 启用 ROI 模式</label>
      <label>X<input id="roiX" type="number" min="0" step="1"></label>
      <label>Y<input id="roiY" type="number" min="0" step="1"></label>
      <label>宽<input id="roiW" type="number" min="16" step="1"></label>
      <label>高<input id="roiH" type="number" min="16" step="1"></label>
      <label class="wide">FPS<input id="roiFps" type="number" min="1" max="30" step="1"></label>
      <button id="saveConfig" class="secondary">应用 ROI 设置</button>
      <button id="analyzeRoi" class="secondary">分析 ROI 区域</button>
    </div>
  </div>
  <div class="result">
    <div class="meta" id="meta"></div>
    <div class="content" id="content">选择上方提问方式后开始分析。</div>
  </div>
  <div class="status" id="status">就绪</div>
</aside>
<script>
const TOKEN = "__MILOCO_TOKEN__";
const img = document.getElementById("screen");
const dot = document.getElementById("dot");
const frame = document.getElementById("frame");
const overlay = document.getElementById("overlay");
const overlayTitle = document.getElementById("overlayTitle");
const overlayText = document.getElementById("overlayText");
const reloadBtn = document.getElementById("reloadStream");
const statusEl = document.getElementById("status");
const content = document.getElementById("content");
const meta = document.getElementById("meta");
const hint = document.getElementById("hint");
const preset = document.getElementById("preset");
const query = document.getElementById("query");
const btn = document.getElementById("analyze");
const saveConfigBtn = document.getElementById("saveConfig");
const analyzeRoiBtn = document.getElementById("analyzeRoi");
const roiToggle = document.getElementById("roiToggle");
const roiBox = document.getElementById("roiBox");
const roiEnable = document.getElementById("roiEnable");
const roiX = document.getElementById("roiX");
const roiY = document.getElementById("roiY");
const roiW = document.getElementById("roiW");
const roiH = document.getElementById("roiH");
const roiFps = document.getElementById("roiFps");
let failures = 0;
let statusFailures = 0;
let overlayManuallyHidden = false;

function setStatus(text, error=false){ statusEl.textContent = text; statusEl.style.color = error ? "#f87171" : ""; }
function streamUrl(){ return "/api/screen/stream?token=" + encodeURIComponent(TOKEN) + "&t=" + Date.now(); }
function showOverlay(title, text, isError=false){
  overlayTitle.textContent = title;
  overlayText.textContent = text;
  overlay.classList.add("show");
  dot.classList.add("dead");
  if(isError) overlayText.classList.add("error"); else overlayText.classList.remove("error");
}
function hideOverlay(){
  overlay.classList.remove("show");
  dot.classList.remove("dead");
  overlayManuallyHidden = false;
}
function connectStream(){
  img.removeAttribute("src");
  img.src = streamUrl();
  setStatus("正在拉流");
  if(!overlayManuallyHidden) showOverlay("画面加载中", "正在连接 MJPEG 推流,请稍候...");
}
function reconnect(){
  failures += 1;
  dot.classList.add("dead");
  setStatus("推流重连中 " + failures, true);
  showOverlay("推流中断", "第 " + failures + " 次尝试重连... 如持续黑屏,请检查服务状态或点击下方按钮手动刷新。", true);
  setTimeout(connectStream, Math.min(5000, 600 + failures * 300));
}
img.onload = () => {
  failures = 0;
  dot.classList.remove("dead");
  hideOverlay();
  if (img.naturalWidth) frame.textContent = img.naturalWidth + "x" + img.naturalHeight + " · 已连接";
  setStatus("推流已连接");
};
img.onerror = () => { reconnect(); };
reloadBtn.addEventListener("click", async () => {
  failures = 0;
  overlayManuallyHidden = true;
  reloadBtn.disabled = true;
  reloadBtn.textContent = "重启采集中...";
  showOverlay("重启采集", "正在终止旧的 ffmpeg 进程并重新申请屏幕权限,请稍候...");
  try {
    const resp = await fetch("/api/screen/restart", {
      method: "POST",
      headers: {"Authorization": "Bearer " + TOKEN},
    });
    if (!resp.ok) {
      const data = await resp.json().catch(() => ({}));
      throw new Error(data.detail || ("HTTP " + resp.status));
    }
    const data = await resp.json();
    if (data.permission_hint) {
      showOverlay("仍需屏幕录制权限", data.permission_hint, true);
      setStatus(data.permission_hint, true);
    } else {
      setStatus("采集已重启,正在拉流...");
    }
  } catch (e) {
    setStatus("重启失败: " + e, true);
    showOverlay("重启失败", String(e) + "\n可尝试直接刷新页面。", true);
  } finally {
    reloadBtn.disabled = false;
    reloadBtn.textContent = "重新加载画面";
  }
  // 无论重启成功与否都重新拉流,浏览器会展示新画面或新的错误状态。
  connectStream();
});
async function pollCaptureStatus(){
  try {
    const resp = await fetch("/api/screen/status?token=" + encodeURIComponent(TOKEN), {cache: "no-store"});
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();
    statusFailures = 0;
    if (data.width && data.height) frame.textContent = data.width + "x" + data.height + " · " + (data.frames || 0) + " 帧";
    const age = data.latest_frame_age_ms;
    if (data.permission_hint) {
      dot.classList.add("dead");
      setStatus(data.permission_hint, true);
      showOverlay("需要屏幕录制权限", data.permission_hint, true);
      return;
    }
    if (!data.running) {
      dot.classList.add("dead");
      setStatus(data.last_error || "屏幕采集未运行", true);
      showOverlay("采集未运行", data.last_error || "屏幕采集未运行,请刷新页面或检查服务日志。", true);
    } else if (age == null) {
      dot.classList.add("dead");
      setStatus("等待屏幕首帧", true);
      if(!overlayManuallyHidden) showOverlay("等待首帧", "ffmpeg 已启动,正在等待首帧。若超过 10 秒仍无画面,请检查 macOS 屏幕录制权限。", true);
    } else if (age > 5000) {
      dot.classList.add("dead");
      setStatus("屏幕采集停滞 " + Math.round(age / 1000) + "s", true);
    } else {
      dot.classList.remove("dead");
      setStatus("画面正常");
    }
  } catch (e) {
    statusFailures += 1;
    if (statusFailures >= 3) {
      dot.classList.add("dead");
      setStatus("无法读取采集状态: " + e, true);
    }
  }
}
setInterval(pollCaptureStatus, 3000);
pollCaptureStatus();
connectStream();

preset.addEventListener("change", () => { if (preset.value !== "custom") query.value = preset.value; });
roiToggle.addEventListener("click", () => {
  const open = roiBox.classList.toggle("open");
  roiToggle.textContent = open ? "收起" : "展开";
});

function intValue(el, fallback){ const n = parseInt(el.value, 10); return Number.isFinite(n) ? n : fallback; }
function fillConfig(data){
  roiEnable.checked = !!data.roi_enable;
  roiX.value = data.x ?? 0;
  roiY.value = data.y ?? 0;
  roiW.value = data.width ?? 1280;
  roiH.value = data.height ?? 720;
  roiFps.value = data.fps ?? (data.roi_enable ? 1 : 3);
}
async function loadConfig(){
  try {
    const resp = await fetch("/api/screen/config?token=" + encodeURIComponent(TOKEN), {cache: "no-store"});
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    fillConfig(await resp.json());
  } catch (e) {
    setStatus("读取配置失败: " + e, true);
  }
}
async function saveConfig(forceRoi){
  const payload = {
    roi_enable: forceRoi === true ? true : roiEnable.checked,
    x: intValue(roiX, 0),
    y: intValue(roiY, 0),
    width: intValue(roiW, 1280),
    height: intValue(roiH, 720),
    fps: intValue(roiFps, roiEnable.checked ? 1 : 3),
  };
  saveConfigBtn.disabled = true;
  setStatus("正在应用屏幕配置");
  try {
    const resp = await fetch("/api/screen/config", {
      method: "POST",
      headers: {"Content-Type": "application/json", "Authorization": "Bearer " + TOKEN},
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || ("HTTP " + resp.status));
    fillConfig(data);
    setStatus(data.roi_enable ? "ROI 推流已应用" : "全屏推流已应用");
    hint.textContent = "配置已保存。";
    hint.className = "hint";
    connectStream();
  } catch (e) {
    setStatus("配置失败: " + e, true);
    hint.textContent = "配置失败: " + e;
    hint.className = "hint error";
  } finally {
    saveConfigBtn.disabled = false;
  }
}
saveConfigBtn.addEventListener("click", () => saveConfig().catch(e => { setStatus("配置失败: " + e, true); hint.textContent = "配置失败: " + e; hint.className = "hint error"; }));
analyzeRoiBtn.addEventListener("click", async () => {
  try {
    roiEnable.checked = true;
    if (!roiFps.value) roiFps.value = "1";
    await saveConfig(true);
    setTimeout(() => btn.click(), 1200);
  } catch (e) {
    setStatus("ROI 分析失败: " + e, true);
  }
});
btn.addEventListener("click", async () => {
  let q = query.value.trim();
  if (!q && preset.value !== "custom") q = preset.value;
  if (!q) { setStatus("请输入分析问题", true); hint.textContent = "请选择预设或输入自定义问题。"; hint.className = "hint warn"; return; }
  btn.disabled = true;
  btn.textContent = "分析中";
  content.textContent = "正在调用本地视觉模型...";
  setStatus("VLM 推理中");
  try {
    const resp = await fetch("/api/screen/analyze?q=" + encodeURIComponent(q), { headers: {"Authorization": "Bearer " + TOKEN} });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || data.message || ("HTTP " + resp.status));
    meta.textContent = (data.time || "") + " | " + (data.elapsed || 0) + "s | tokens " + JSON.stringify(data.tokens || {});
    content.textContent = data.content || "";
    setStatus("分析完成");
  } catch (e) {
    content.textContent = "请求失败: " + e;
    setStatus("分析失败", true);
  } finally {
    btn.disabled = false;
    btn.textContent = "分析当前屏幕";
  }
});
loadConfig();
</script>
</body>
</html>
"""
router = APIRouter(prefix="/screen", tags=["Screen"])


@router.get(
    "", include_in_schema=False, dependencies=[Depends(verify_token_query_fallback)]
)
@router.get(
    "/", include_in_schema=False, dependencies=[Depends(verify_token_query_fallback)]
)
async def screen_page() -> HTMLResponse:
    get_screen_capture()
    from miloco.config import get_settings

    token = get_settings().server.token or ""
    html = HTML_PAGE.replace("__MILOCO_TOKEN__", escape_for_js_string(token))
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@router.get("/stream", dependencies=[Depends(verify_token_query_fallback)])
async def screen_stream() -> StreamingResponse:
    capture = get_screen_capture()
    return StreamingResponse(
        _mjpeg_stream(capture),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/status", dependencies=[Depends(verify_token_query_fallback)])
async def screen_status() -> dict:
    return asdict(get_screen_capture().status())


@router.get("/config", dependencies=[Depends(verify_token_query_fallback)])
async def screen_config_get() -> dict:
    return asdict(get_screen_config())


@router.post("/config", dependencies=[Depends(verify_token)])
async def screen_config_post(update: ScreenConfigUpdate) -> dict:
    try:
        config = _merge_config_update(update)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return asdict(update_screen_config(config))


@router.post("/restart", dependencies=[Depends(verify_token)])
async def screen_restart() -> dict:
    """重启屏幕采集 ffmpeg 进程。

    用途:macOS 屏幕录制权限授予后,旧 ffmpeg 进程不会自动重新申请权限,
    仍会输出黑帧。前端"重新加载画面"按钮调用此端点,杀掉旧进程、
    起新进程,让系统重新检查权限。返回新 status 供前端即时显示。
    """
    restart_screen_capture()
    # 给新进程一点启动时间,让 status() 能反映真实状态。
    # 不在这里阻塞等待首帧(可能 8s),否则前端会超时。
    await asyncio.sleep(0.3)
    return asdict(get_screen_capture().status())


@router.get("/analyze", dependencies=[Depends(verify_token)])
async def screen_analyze(
    q: str = Query(
        default="当前屏幕打开了哪些应用和窗口？请列出名称和它们大致在做什么。",
        max_length=500,
    ),
) -> dict:
    return await analyze_screen(q)
