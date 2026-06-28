"""Local screen capture, MJPEG preview, and VLM analysis endpoints."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import select
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse, StreamingResponse

from miloco.middleware import verify_token, verify_token_query_fallback
from miloco.perception.engine.config import OmniConfig
from miloco.perception.engine.omni.omni_client import call_omni
from miloco.utils.common import escape_for_js_string

logger = logging.getLogger(__name__)


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


class ScreenCapture:
    """Single ffmpeg screen capture session shared by MJPEG and VLM analysis."""

    def __init__(self, fps: int = 3, width: int = 1280, monitor: int = 0):
        self.fps = max(1, fps)
        self.width = width - (width % 2)
        self.monitor = monitor
        self.height = int(self.width * 9 / 16)
        self.height -= self.height % 2
        self._lock = threading.Lock()
        self._latest_frame: bytes | None = None
        self._latest_ts: float | None = None
        self._frame_buffer: list[bytes] = []
        self._frame_count = 0
        self._restart_count = 0
        self._last_error = ""
        self._proc: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
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
        proc = self._proc
        if proc:
            self._terminate_proc(proc)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)

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

    def status(self) -> ScreenCaptureStatus:
        with self._lock:
            latest_ts = self._latest_ts
            frames = self._frame_count
            last_error = self._last_error
        return ScreenCaptureStatus(
            running=bool(self._proc and self._proc.poll() is None),
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
            restart_count=self._restart_count,
            last_error=last_error,
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
                self._restart_count += 1
                time.sleep(2)

    def _run_once(self) -> None:
        cmd = [
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
            (
                f"scale={self.width}:{self.height}:force_original_aspect_ratio=decrease,"
                f"pad={self.width}:{self.height}:(ow-iw)/2:(oh-ih)/2"
            ),
            "-c:v",
            "mjpeg",
            "-q:v",
            "8",
            "-an",
            "-nostdin",
            "-f",
            "image2pipe",
            "pipe:1",
        ]
        logger.info(
            "starting screen capture: monitor=%s %sx%s @ %sfps",
            self.monitor,
            self.width,
            self.height,
            self.fps,
        )
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._proc = proc
        threading.Thread(
            target=self._drain_stderr,
            args=(proc,),
            name="screen-capture-stderr",
            daemon=True,
        ).start()
        try:
            self._read_jpeg_pipe(proc)
        finally:
            if proc.poll() is None:
                self._terminate_proc(proc)
            rc = proc.poll()
            with self._lock:
                self._last_error = "" if self._stop.is_set() else f"ffmpeg exited: {rc}"

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
                        self._last_error = "ffmpeg started but produced no screen frames"
                    proc.terminate()
                    break
                continue
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
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
                    self._frame_buffer.append(frame)
                    if len(self._frame_buffer) > 24:
                        del self._frame_buffer[: len(self._frame_buffer) - 24]


_capture: ScreenCapture | None = None
_capture_lock = threading.Lock()
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


def get_screen_capture() -> ScreenCapture:
    global _capture
    with _capture_lock:
        if _capture is None:
            _capture = ScreenCapture(
                fps=_screen_int_env("MILOCO_SCREEN_FPS", 3),
                width=_screen_int_env("MILOCO_SCREEN_WIDTH", 1280),
                monitor=_screen_int_env("MILOCO_SCREEN_MONITOR", 0),
            )
        _capture.start()
        return _capture


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


async def _encode_frames_mp4(frames: list[bytes], fps: int) -> bytes:
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, jpg in enumerate(frames):
            with open(os.path.join(tmpdir, f"f{i:04d}.jpg"), "wb") as f:
                f.write(jpg)
        mp4_path = os.path.join(tmpdir, "screen.mp4")
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(tmpdir, "f%04d.jpg"),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "30",
            "-pix_fmt",
            "yuv420p",
            "-an",
            mp4_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
        if proc.returncode != 0:
            raise RuntimeError(stderr.decode(errors="replace")[-500:])
        with open(mp4_path, "rb") as f:
            return f.read()


async def analyze_screen(query: str) -> dict:
    """Analyze recent frames from the shared screen capture buffer."""

    global _last_analysis
    capture = get_screen_capture()
    async with _get_analysis_lock():
        frames = capture.get_recent_frames(9)
        if len(frames) < 2:
            _last_analysis = {
                "content": f"帧缓冲区不足，当前只有 {len(frames)} 帧，请稍后重试。",
                "time": time.strftime("%H:%M:%S"),
                "elapsed": 0,
                "tokens": {},
            }
            return dict(_last_analysis)

        try:
            mp4 = await _encode_frames_mp4(frames, capture.fps)
            video_b64 = base64.b64encode(mp4).decode("ascii")
            payload = {
                "system_prompt": (
                    "你是一个桌面活动监控助手。请观察屏幕录屏，用中文说明用户正在做什么。"
                    "重点关注打开的应用窗口、正在编辑或浏览的内容、是否属于工作、娱乐或通讯。"
                    "不要描述正在对话、用户提问、聊天界面等元信息；直接描述屏幕上的实质内容。"
                    "回答保持简洁，3 到 5 句。"
                ),
                "user_content": query or "当前屏幕打开了哪些应用和窗口？请列出名称和它们大致在做什么。",
                "video_base64": video_b64,
                "crops": [],
                "images": [],
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
    while True:
        frame = capture.get_frame()
        if frame and frame != last_frame:
            last_frame = frame
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + str(len(frame)).encode("ascii")
                + b"\r\n\r\n"
                + frame
                + b"\r\n"
            )
        time.sleep(0.08)


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Miloco Screen Monitor</title>
<style>
*{box-sizing:border-box}html,body{margin:0;height:100%;background:#0b0c0f;color:#e8eaed;font:14px/1.5 system-ui,-apple-system,BlinkMacSystemFont,sans-serif}
body{display:flex;min-height:100vh;overflow:hidden}.preview{flex:1;min-width:0;background:#000;display:flex;flex-direction:column}.bar{height:40px;display:flex;align-items:center;justify-content:space-between;padding:0 14px;background:#111318;border-bottom:1px solid #242833;color:#9aa0aa}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:8px;background:#17c964}.dot.dead{background:#ef4444}.preview img{flex:1;width:100%;min-height:0;object-fit:contain;background:#000}.panel{width:420px;max-width:42vw;background:#151821;border-left:1px solid #2b3040;display:flex;flex-direction:column}
.head{padding:14px 16px;border-bottom:1px solid #2b3040;font-weight:700}.controls{padding:12px 16px;border-bottom:1px solid #2b3040}.controls select,.controls input{width:100%;margin-bottom:8px;padding:8px 10px;background:#0f1117;border:1px solid #343a4a;border-radius:6px;color:#e8eaed;font:inherit}.controls button{width:100%;padding:9px 12px;border:0;border-radius:6px;background:#2563eb;color:#fff;font-weight:700;cursor:pointer}.controls button:disabled{background:#3b4252;color:#9aa0aa;cursor:wait}
.result{flex:1;min-height:0;overflow:auto;padding:16px}.meta{color:#9aa0aa;font-size:12px;margin-bottom:8px}.content{white-space:pre-wrap;word-break:break-word;font-size:15px;line-height:1.7}.status{padding:10px 16px;border-top:1px solid #2b3040;color:#9aa0aa;font-size:12px}
@media(max-width:840px){body{flex-direction:column}.panel{width:100%;max-width:none;height:44vh;border-left:0;border-top:1px solid #2b3040}.bar{height:auto;min-height:38px}}
</style>
</head>
<body>
<main class="preview">
  <div class="bar"><span><span id="dot" class="dot"></span>屏幕实时采集</span><span id="frame">等待画面</span></div>
  <img id="screen" alt="屏幕画面">
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
  </div>
  <div class="result">
    <div class="meta" id="meta"></div>
    <div class="content" id="content">选择提问方式后开始分析。</div>
  </div>
  <div class="status" id="status">就绪</div>
</aside>
<script>
const TOKEN = "__MILOCO_TOKEN__";
const img = document.getElementById("screen");
const dot = document.getElementById("dot");
const frame = document.getElementById("frame");
const statusEl = document.getElementById("status");
const content = document.getElementById("content");
const meta = document.getElementById("meta");
const preset = document.getElementById("preset");
const query = document.getElementById("query");
const btn = document.getElementById("analyze");
let failures = 0;
let statusFailures = 0;

function setStatus(text, error=false){ statusEl.textContent = text; statusEl.style.color = error ? "#f87171" : ""; }
function streamUrl(){ return "/api/screen/stream?token=" + encodeURIComponent(TOKEN) + "&t=" + Date.now(); }
function connectStream(){
  img.src = streamUrl();
  setStatus("正在拉流");
}
function reconnect(){
  failures += 1;
  dot.classList.add("dead");
  setStatus("推流重连中 " + failures, true);
  setTimeout(connectStream, Math.min(5000, 600 + failures * 300));
}
img.onload = () => {
  failures = 0;
  dot.classList.remove("dead");
  if (img.naturalWidth) frame.textContent = img.naturalWidth + "x" + img.naturalHeight;
  setStatus("推流已连接");
};
img.onerror = reconnect;
async function pollCaptureStatus(){
  try {
    const resp = await fetch("/api/screen/status?token=" + encodeURIComponent(TOKEN), {cache: "no-store"});
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();
    statusFailures = 0;
    if (data.width && data.height) frame.textContent = data.width + "x" + data.height + " · " + (data.frames || 0) + " 帧";
    const age = data.latest_frame_age_ms;
    if (!data.running) {
      dot.classList.add("dead");
      setStatus(data.last_error || "屏幕采集未运行", true);
    } else if (age == null) {
      dot.classList.add("dead");
      setStatus("等待屏幕首帧", true);
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
preset.addEventListener("change", () => {
  if (preset.value !== "custom") query.value = preset.value;
});
btn.addEventListener("click", async () => {
  let q = query.value.trim();
  if (!q && preset.value !== "custom") q = preset.value;
  if (!q) { setStatus("请输入分析问题", true); return; }
  btn.disabled = true;
  btn.textContent = "分析中";
  content.textContent = "正在调用本地视觉模型...";
  setStatus("VLM 推理中");
  try {
    const resp = await fetch("/api/screen/analyze?q=" + encodeURIComponent(q), {
      headers: {"Authorization": "Bearer " + TOKEN},
    });
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
query.value = preset.value;
connectStream();
</script>
</body>
</html>"""


router = APIRouter(prefix="/screen", tags=["Screen"])


@router.get("", include_in_schema=False, dependencies=[Depends(verify_token_query_fallback)])
@router.get("/", include_in_schema=False, dependencies=[Depends(verify_token_query_fallback)])
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


@router.get("/analyze", dependencies=[Depends(verify_token)])
async def screen_analyze(
    q: str = Query(
        default="当前屏幕打开了哪些应用和窗口？请列出名称和它们大致在做什么。",
        max_length=500,
    ),
) -> dict:
    return await analyze_screen(q)
