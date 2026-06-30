"""Phone screen stream via WebRTC, MJPEG preview, and VLM analysis endpoints.

接收 Android 手机通过 QuestPhoneStream signaling server 推送的 WebRTC 屏幕流,
转 JPEG 后复用 screen_monitor 同款的 MJPEG 预览 + VLM 分析能力。

架构:
- ``PhoneCapture`` 在 FastAPI event loop 里跑后台 asyncio task:
  1. WebSocket 连接 signaling server (ws://host:8787)
  2. 注册为 "quest" 角色, create_session
  3. 等待 Android 端的 SDP offer, 回 answer
  4. ICE 交换
  5. RTCPeerConnection 收到 video track, 每帧 av.VideoFrame → JPEG → deque
- MJPEG stream (sync iterator, 在 threadpool 跑) 从 deque 读帧推流
- 共享状态用 threading.Lock 保护, 因为 async task 和 sync iterator 在不同线程

与 screen_monitor.ScreenCapture 的接口契约对齐:
- get_frame() / get_latest_frames() / get_recent_frames()
- status() 返回 dataclass
- start() / stop()
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import websockets
from aiortc import RTCIceCandidate, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaStreamTrack
from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image

from miloco.middleware import verify_token, verify_token_query_fallback
from miloco.perception.engine.config import OmniConfig
from miloco.perception.engine.omni.omni_client import call_omni
from miloco.utils.common import escape_for_js_string

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 配置与状态
# ---------------------------------------------------------------------------


@dataclass
class PhoneStreamConfig:
    """Phone stream 连接配置。

    所有的值都可以通过环境变量覆盖,方便部署时不改代码调整。
    """

    # signaling server 跑在本机 (QuestPhoneStream apps/signaling-server),
    # 用 127.0.0.1 避免走网络栈。手机 app 在局域网里用本机 LAN IP 连同一 server。
    signaling_url: str = "ws://127.0.0.1:8787"
    token: str = "dev-token"
    # deviceId 必须和手机 app 里配置的 Quest Device ID 一致,
    # 否则手机发的 SDP offer 目标对不上,miloco 收不到 offer。
    # QuestPhoneStream app 默认值是 quest-3s-001。
    quest_device_id: str = "quest-3s-001"
    android_device_id: str = "android-phone-001"
    session_id: str = "miloco-session-001"
    # JPEG 编码质量 (1-95)。手机推流带宽有限,75 是质量/带宽的平衡点。
    jpeg_quality: int = 75
    # 帧缓冲容量:VLM 分析最多取 3 帧,24 帧覆盖 ~8s @3fps 的分析窗口。
    buffer_size: int = 24


@dataclass
class PhoneCaptureStatus:
    running: bool
    signaling_state: str
    ice_state: str
    pc_state: str
    frames: int
    latest_frame_age_ms: int | None
    width: int
    height: int
    fps: int
    last_error: str
    signaling_url: str
    session_id: str
    android_device_id: str
    quest_device_id: str


# ---------------------------------------------------------------------------
# PhoneCapture
# ---------------------------------------------------------------------------


class PhoneCapture:
    """WebRTC 接收端,从 Android 手机接收屏幕推流并转 JPEG。

    生命周期:
    - ``start()`` 在 FastAPI event loop 里创建后台 asyncio task
    - task 跑 signaling 握手 + WebRTC 接收循环
    - ``stop()`` 取消 task, 关闭 PC 和 WS
    - ``get_frame()`` / ``status()`` 是线程安全的 sync 接口,
      供 MJPEG stream (threadpool) 和 status endpoint (async) 调用
    """

    def __init__(self, config: PhoneStreamConfig | None = None):
        self.config = _normalize_config(config or _load_config_from_env())
        # 共享状态: async task 写, sync reader 读。用 threading.Lock 保护。
        # threading.Lock 而非 asyncio.Lock 是因为 sync iterator 也要获取。
        self._lock = threading.Lock()
        self._latest_frame: bytes | None = None
        self._latest_ts: float | None = None
        self._frame_buffer: deque[bytes] = deque(maxlen=self.config.buffer_size)
        self._frame_count = 0
        self._last_error = ""
        self._signaling_state = "idle"
        self._ice_state = "new"
        self._pc_state = "new"
        self._width = 0
        self._height = 0
        self._fps = 0
        # asyncio 部分: 只在 event loop 线程访问, 不需要锁
        self._pc: RTCPeerConnection | None = None
        self._ws = None
        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None
        self._frame_times: deque[float] = deque(maxlen=30)  # 用于估算 fps

    # -- sync 接口 (供 MJPEG stream / status endpoint 调用) -----------------

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
            if count <= 0 or not self._frame_buffer:
                return []
            return list(self._frame_buffer)[-count:]

    def status(self) -> PhoneCaptureStatus:
        with self._lock:
            latest_ts = self._latest_ts
            frames = self._frame_count
            last_error = self._last_error
            sig_state = self._signaling_state
            ice_state = self._ice_state
            pc_state = self._pc_state
            width = self._width
            height = self._height
            fps = self._fps
        return PhoneCaptureStatus(
            running=self._task is not None and not self._task.done(),
            signaling_state=sig_state,
            ice_state=ice_state,
            pc_state=pc_state,
            frames=frames,
            latest_frame_age_ms=(
                int((time.monotonic() - latest_ts) * 1000)
                if latest_ts is not None
                else None
            ),
            width=width,
            height=height,
            fps=fps,
            last_error=last_error,
            signaling_url=self.config.signaling_url,
            session_id=self.config.session_id,
            android_device_id=self.config.android_device_id,
            quest_device_id=self.config.quest_device_id,
        )

    # -- 生命周期 -----------------------------------------------------------

    def start(self) -> None:
        """在当前 event loop 里启动后台接收 task。

        必须在 asyncio event loop 运行时调用 (FastAPI endpoint / startup hook)。
        重复调用是 no-op。
        """
        if self._task is not None and not self._task.done():
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(
            self._run_forever(), name="phone-stream-receiver"
        )

    async def stop(self) -> None:
        """停止接收,关闭 PC 和 WS。

        幂等。会等待后台 task 结束 (最多 3s)。
        """
        if self._stop_event:
            self._stop_event.set()
        # 取消 task 触发 CancelledError, _run_forever 的 finally 会清理 PC/WS
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass
        self._task = None
        with self._lock:
            self._signaling_state = "disconnected"

    async def restart(self) -> None:
        """重启接收 (用户点了重连按钮,或 signaling 断了想恢复)。"""
        await self.stop()
        # 短暂等待,避免端口/资源未释放
        await asyncio.sleep(0.3)
        self.start()

    # -- 后台 task 主体 -----------------------------------------------------

    async def _run_forever(self) -> None:
        """持续运行: 断线自动重连,直到 stop() 被调用。

        signaling 连接断开 (网络抖动 / server 重启) 时,等待 2s 后重试,
        不让瞬时故障导致整个接收端永久停止。
        """
        while self._stop_event is not None and not self._stop_event.is_set():
            try:
                await self._run_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                with self._lock:
                    self._last_error = f"{type(e).__name__}: {e}"
                logger.warning("phone stream failed: %s", e)
            if self._stop_event is not None and not self._stop_event.is_set():
                with self._lock:
                    self._signaling_state = "reconnecting"
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass  # 2s 到了, 重试

    async def _run_once(self) -> None:
        """一次完整的 signaling + WebRTC 建连 + 接收循环。

        抛异常会触发 _run_forever 的重连逻辑。
        """
        cfg = self.config
        with self._lock:
            self._signaling_state = "connecting"
            self._last_error = ""

        logger.info(
            "phone stream connecting: signaling=%s session=%s android=%s quest=%s",
            cfg.signaling_url,
            cfg.session_id,
            cfg.android_device_id,
            cfg.quest_device_id,
        )

        # 1. 连 signaling
        try:
            self._ws = await websockets.connect(
                cfg.signaling_url,
                max_size=2**20,  # SDP+ICE 通常 < 64KB, 1MB 上限足够
                ping_interval=15,
                ping_timeout=20,
            )
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self._signaling_state = "failed"
                self._last_error = f"signaling connect failed: {e}"
            raise

        # 2. 创建 PeerConnection (放在 ws 连上之后, 避免无谓资源)
        pc = RTCPeerConnection()
        self._pc = pc
        pc.on("iceconnectionstatechange", self._on_ice_state_change)
        pc.on("connectionstatechange", self._on_pc_state_change)
        pc.on("track", self._on_track)

        try:
            # 3. register
            await self._send(
                {
                    "type": "register",
                    "token": cfg.token,
                    "role": "quest",
                    "deviceId": cfg.quest_device_id,
                }
            )
            # 4. create_session
            await self._send(
                {
                    "type": "create_session",
                    "token": cfg.token,
                    "sessionId": cfg.session_id,
                    "androidDeviceId": cfg.android_device_id,
                    "questDeviceId": cfg.quest_device_id,
                }
            )
            with self._lock:
                self._signaling_state = "connected"

            # 5. 主循环: 收 signaling 消息处理
            async for raw in self._ws:
                if self._stop_event is not None and self._stop_event.is_set():
                    break
                msg = json.loads(raw)
                await self._handle_signaling_message(msg)

        finally:
            # 清理: 关 ws 和 pc。放在 finally 确保 _run_once 异常时也清理。
            if self._ws is not None:
                try:
                    await self._ws.close()
                except Exception:  # noqa: BLE001
                    pass
                self._ws = None
            try:
                await pc.close()
            except Exception:  # noqa: BLE001
                pass
            self._pc = None
            with self._lock:
                self._signaling_state = "disconnected"
                self._ice_state = "closed"
                self._pc_state = "closed"

    async def _send(self, msg: dict) -> None:
        if self._ws is None:
            return
        await self._ws.send(json.dumps(msg))

    async def _handle_signaling_message(self, msg: dict) -> None:
        mtype = msg.get("type")
        cfg = self.config
        if mtype == "registered":
            logger.info("phone stream registered as %s", msg.get("deviceId"))
        elif mtype == "session_created":
            logger.info("phone stream session created: %s", msg.get("sessionId"))
        elif mtype == "offer":
            # Android 发来 SDP offer, setRemoteDescription + createAnswer
            sdp = msg.get("sdp", "")
            await self._pc.setRemoteDescription(
                RTCSessionDescription(sdp=sdp, type="offer")
            )
            answer = await self._pc.createAnswer()
            await self._pc.setLocalDescription(answer)
            await self._send(
                {
                    "type": "answer",
                    "token": cfg.token,
                    "sessionId": cfg.session_id,
                    "from": cfg.quest_device_id,
                    "to": cfg.android_device_id,
                    "sdp": self._pc.localDescription.sdp,
                }
            )
            logger.info("phone stream sent SDP answer")
        elif mtype == "ice":
            candidate = msg.get("candidate", {})
            try:
                await self._pc.addIceCandidate(
                    RTCIceCandidate(
                        component=candidate.get("component", 1),
                        foundation=candidate.get("foundation", ""),
                        ip=candidate.get("ip", ""),
                        port=candidate.get("port", 0),
                        priority=candidate.get("priority", 0),
                        protocol=candidate.get("protocol", "udp"),
                        type=candidate.get("type", "host"),
                        sdpMid=candidate.get("sdpMid"),
                        sdpMLineIndex=candidate.get("sdpMLineIndex"),
                    )
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("phone stream addIceCandidate failed: %s", e)
        elif mtype == "error":
            with self._lock:
                self._last_error = f"signaling error: {msg.get('message', msg)}"
            logger.warning("phone stream signaling error: %s", msg)
        elif mtype == "peer_unavailable":
            with self._lock:
                self._last_error = (
                    f"Android peer {msg.get('deviceId')} unavailable, "
                    "请确认手机 app 已启动并注册到同一 signaling server"
                )
            logger.warning("phone stream peer unavailable: %s", msg)

    # -- WebRTC 回调 --------------------------------------------------------

    def _on_ice_state_change(self) -> None:
        if self._pc is None:
            return
        state = self._pc.iceConnectionState
        with self._lock:
            self._ice_state = state
        logger.info("phone stream ICE: %s", state)

    def _on_pc_state_change(self) -> None:
        if self._pc is None:
            return
        state = self._pc.connectionState
        with self._lock:
            self._pc_state = state
        logger.info("phone stream PC: %s", state)

    def _on_track(self, track: MediaStreamTrack) -> None:
        """收到 Android video track, 启动帧接收 task。"""
        if track.kind != "video":
            return
        logger.info("phone stream received video track: %s", track.id)
        # 在 event loop 里启动帧接收协程, 不能阻塞 on_track 回调
        asyncio.create_task(self._receive_frames(track))

    async def _receive_frames(self, track: MediaStreamTrack) -> None:
        """持续接收 video frame, 转 JPEG 存 buffer。

        av.VideoFrame → numpy (rgb24) → PIL.Image → JPEG bytes。
        用 PIL 而非 cv2 是因为 cv2 和 av 的 dylib 有冲突 (见启动警告)。
        """
        cfg = self.config
        try:
            while self._stop_event is not None and not self._stop_event.is_set():
                frame = await track.recv()
                # 转 JPEG
                img_array = frame.to_ndarray(format="rgb24")
                img = Image.fromarray(img_array, "RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=cfg.jpeg_quality)
                jpeg = buf.getvalue()

                now = time.monotonic()
                with self._lock:
                    self._latest_frame = jpeg
                    self._latest_ts = now
                    self._frame_count += 1
                    self._frame_buffer.append(jpeg)
                    self._width = frame.width
                    self._height = frame.height
                    # 估算 fps: 最近 30 帧的平均间隔
                    self._frame_times.append(now)
                    if len(self._frame_times) >= 2:
                        elapsed = self._frame_times[-1] - self._frame_times[0]
                        if elapsed > 0:
                            self._fps = int((len(self._frame_times) - 1) / elapsed)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self._last_error = f"frame receive stopped: {e}"
            logger.warning("phone stream frame receive ended: %s", e)


# ---------------------------------------------------------------------------
# 配置加载与校验
# ---------------------------------------------------------------------------


_PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
_NO_PROXY_ENV_KEYS = ("NO_PROXY", "no_proxy")
_proxy_env_lock = threading.Lock()


def _is_loopback_url(url: str) -> bool:
    return "://127.0.0.1" in url or "://localhost" in url


@contextmanager
def _without_proxy_env_for_loopback(url: str):
    """VLM 走 loopback 时临时清掉 proxy 环境变量,与 screen_monitor 对齐。"""
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


def _load_config_from_env() -> PhoneStreamConfig:
    """从环境变量加载配置, 覆盖默认值。

    环境变量命名: MILOCO_PHONE_STREAM_<FIELD>。
    """
    return PhoneStreamConfig(
        signaling_url=os.environ.get(
            "MILOCO_PHONE_STREAM_SIGNALING_URL",
            PhoneStreamConfig.signaling_url,
        ),
        token=os.environ.get("MILOCO_PHONE_STREAM_TOKEN", PhoneStreamConfig.token),
        quest_device_id=os.environ.get(
            "MILOCO_PHONE_STREAM_QUEST_DEVICE_ID",
            PhoneStreamConfig.quest_device_id,
        ),
        android_device_id=os.environ.get(
            "MILOCO_PHONE_STREAM_ANDROID_DEVICE_ID",
            PhoneStreamConfig.android_device_id,
        ),
        session_id=os.environ.get(
            "MILOCO_PHONE_STREAM_SESSION_ID",
            PhoneStreamConfig.session_id,
        ),
        jpeg_quality=int(
            os.environ.get(
                "MILOCO_PHONE_STREAM_JPEG_QUALITY",
                str(PhoneStreamConfig.jpeg_quality),
            )
        ),
        buffer_size=int(
            os.environ.get(
                "MILOCO_PHONE_STREAM_BUFFER_SIZE",
                str(PhoneStreamConfig.buffer_size),
            )
        ),
    )


def _normalize_config(config: PhoneStreamConfig) -> PhoneStreamConfig:
    """校验配置, 防止非法值导致后续崩溃。"""
    return PhoneStreamConfig(
        signaling_url=config.signaling_url.strip() or PhoneStreamConfig.signaling_url,
        token=(config.token or "").strip() or PhoneStreamConfig.token,
        quest_device_id=config.quest_device_id.strip()
        or PhoneStreamConfig.quest_device_id,
        android_device_id=config.android_device_id.strip()
        or PhoneStreamConfig.android_device_id,
        session_id=config.session_id.strip() or PhoneStreamConfig.session_id,
        jpeg_quality=max(10, min(95, int(config.jpeg_quality))),
        buffer_size=max(3, min(120, int(config.buffer_size))),
    )


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

_phone_capture: PhoneCapture | None = None
_phone_capture_lock = threading.RLock()
_phone_config: PhoneStreamConfig | None = None
_analysis_lock: asyncio.Lock | None = None
_last_analysis: dict = {
    "content": "等待分析",
    "time": "",
    "elapsed": 0,
    "tokens": {},
}


def get_phone_config() -> PhoneStreamConfig:
    global _phone_config
    if _phone_config is None:
        _phone_config = _load_config_from_env()
    return _phone_config


def update_phone_config(config: PhoneStreamConfig) -> PhoneStreamConfig:
    global _phone_config, _phone_capture
    normalized = _normalize_config(config)
    with _phone_capture_lock:
        old = _phone_capture
        _phone_capture = None
        _phone_config = normalized
    if old is not None:
        # stop 是 async, 但这里可能在 sync 上下文。用 fire-and-forget。
        # 实际调用方 (endpoint) 是 async, 会 await。这里兜底。
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(old.stop())
        except RuntimeError:
            # 没有 running loop, 同步清理 PC/WS (best effort)
            pass
    return normalized


def get_phone_capture() -> PhoneCapture:
    """获取或创建 PhoneCapture 单例并启动。

    必须在 asyncio event loop 运行时调用 (FastAPI endpoint)。
    """
    global _phone_capture
    with _phone_capture_lock:
        if _phone_capture is None:
            _phone_capture = PhoneCapture(get_phone_config())
        _phone_capture.start()
        return _phone_capture


async def restart_phone_capture() -> None:
    """重启 PhoneCapture (用户点了重连按钮)。"""
    global _phone_capture
    with _phone_capture_lock:
        capture = _phone_capture
    if capture is not None:
        await capture.restart()
    else:
        get_phone_capture()


def shutdown_phone_capture() -> None:
    """应用关闭时清理。在 sync shutdown hook 里调用。

    best effort: 尝试在 event loop 里 schedule stop, 如果没有 loop 就跳过。
    """
    global _phone_capture
    with _phone_capture_lock:
        capture = _phone_capture
        _phone_capture = None
    if capture is None:
        return
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(capture.stop())
    except RuntimeError:
        # 没有 running loop (sync shutdown), 直接关闭 PC
        if capture._pc is not None:
            try:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(capture._pc.close())
                loop.close()
            except Exception:  # noqa: BLE001
                pass


def _get_analysis_lock() -> asyncio.Lock:
    global _analysis_lock
    if _analysis_lock is None:
        _analysis_lock = asyncio.Lock()
    return _analysis_lock


# ---------------------------------------------------------------------------
# VLM 分析 (复用 screen_monitor 的 omni 调用模式)
# ---------------------------------------------------------------------------


async def analyze_phone(query: str) -> dict:
    """对手机屏幕最近几帧做 VLM 分析。

    与 screen_monitor.analyze_screen 共用同一套 omni client 调用,
    只是 frames 来源不同。system_prompt 强调"手机屏幕"语境。
    """
    global _last_analysis
    capture = get_phone_capture()
    async with _get_analysis_lock():
        frames = capture.get_latest_frames(3)
        if not frames:
            _last_analysis = {
                "content": (
                    f"手机推流帧缓冲区不足，当前只有 {len(frames)} 帧。"
                    "请确认手机 app 已启动推流,且 WebRTC 连接已建立。"
                ),
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
                    "你是手机屏幕识别助手。\n"
                    "你只能根据图像中清晰可见的内容回答。\n"
                    '如果文字、按钮、App 名称看不清,必须写"无法识别"。\n'
                    "禁止猜测。\n"
                    "禁止根据上下文脑补。\n"
                    "请按以下格式回答:\n\n"
                    "1. 当前 App:\n"
                    "2. 可见文字:\n"
                    "3. 当前操作:\n"
                    "4. 无法确认内容:"
                ),
                "user_content": query
                or "手机屏幕当前打开了哪个 App?在做什么?请简洁描述。",
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
            logger.warning("phone VLM analysis failed: %s", e)
            _last_analysis = {
                "content": f"分析失败: {type(e).__name__}: {e}",
                "time": time.strftime("%H:%M:%S"),
                "elapsed": 0,
                "tokens": {},
            }
        return dict(_last_analysis)


# ---------------------------------------------------------------------------
# MJPEG stream (sync iterator, 与 screen_monitor 同款)
# ---------------------------------------------------------------------------


def _mjpeg_stream(capture: PhoneCapture):
    last_frame = b""
    # 手机推流一般 30fps,但经过 WebRTC + JPEG 转码后实际帧率波动大。
    # 50ms 轮询足够灵敏,不会让浏览器卡顿。
    interval = 0.05
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
        # 持续 90 次 (约 4.5s) 无新帧, 放慢轮询省 CPU
        if empty_streak > 90:
            time.sleep(0.3)
        else:
            time.sleep(interval)


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/phone", tags=["Phone Stream"])


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Miloco Phone Stream</title>
<style>
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:#0b0c0f;color:#e8eaed;font:14px/1.5 system-ui,-apple-system,BlinkMacSystemFont,sans-serif}
body{display:flex;min-height:100vh;overflow:hidden}
.preview{flex:1;min-width:0;background:#000;display:flex;flex-direction:column}
.bar{height:40px;display:flex;align-items:center;justify-content:space-between;padding:0 14px;background:#111318;border-bottom:1px solid #242833;color:#9aa0aa}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:8px;background:#17c964}.dot.dead{background:#ef4444}.dot.warn{background:#facc15}
.preview-wrap{flex:1;position:relative;min-height:0;background:#000;display:flex;align-items:center;justify-content:center;overflow:hidden}
.preview img{max-width:100%;max-height:100%;object-fit:contain;background:#000}
.preview-overlay{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;background:rgba(0,0,0,.85);color:#e8eaed;padding:20px;text-align:center;pointer-events:none;opacity:0;transition:opacity .15s}
.preview-overlay.show{opacity:1;pointer-events:auto}
.preview-overlay .big{font-size:16px;margin-bottom:8px}
.preview-overlay .small{font-size:12px;color:#9aa0aa;max-width:440px;line-height:1.6;white-space:pre-wrap}
.preview-overlay button{margin-top:14px;padding:8px 14px;border-radius:6px;border:0;background:#2563eb;color:#fff;font:inherit;font-weight:600;cursor:pointer;pointer-events:auto}
.panel{width:340px;max-width:36vw;background:#151821;border-left:1px solid #2b3040;display:flex;flex-direction:column}
.head{padding:12px 14px;border-bottom:1px solid #2b3040;font-weight:700;font-size:15px}
.controls{padding:10px 14px;border-bottom:1px solid #2b3040;flex:0 0 auto}
.controls select,.controls input{width:100%;margin-bottom:8px;padding:7px 9px;background:#0f1117;border:1px solid #343a4a;border-radius:6px;color:#e8eaed;font:inherit}
.controls button{width:100%;padding:8px 10px;border:0;border-radius:6px;background:#2563eb;color:#fff;font-weight:600;cursor:pointer;font-size:13px;margin-top:4px}
.controls button:disabled{background:#3b4252;color:#9aa0aa;cursor:wait}
.controls button.secondary{background:#374151!important}
.hint{font-size:12px;color:#9aa0aa;margin-top:8px;line-height:1.5}
.hint.error{color:#f87171}
.hint.warn{color:#facc15}
.result{flex:1;min-height:160px;overflow:auto;padding:12px 14px;background:#0f1117}.meta{color:#9aa0aa;font-size:12px;margin-bottom:8px}.content{white-space:pre-wrap;word-break:break-word;font-size:14px;line-height:1.7}
.status{padding:9px 14px;border-top:1px solid #2b3040;color:#9aa0aa;font-size:12px;min-height:36px}
@media(max-width:840px){body{flex-direction:column}.panel{width:100%;max-width:none;height:48vh;border-left:0;border-top:1px solid #2b3040}.preview{height:52vh}.bar{height:auto;min-height:38px}}
</style>
</head>
<body>
<main class="preview">
  <div class="bar">
    <span><span id="dot" class="dot dead"></span>手机屏幕推流</span>
    <span id="frame">等待连接</span>
  </div>
  <div class="preview-wrap">
    <img id="screen" alt="手机画面">
    <div id="overlay" class="preview-overlay show">
      <div class="big" id="overlayTitle">等待手机推流</div>
      <div class="small" id="overlayText">请确认手机 app 已启动并注册到同一 signaling server。</div>
      <button id="reloadStream">重新连接</button>
    </div>
  </div>
</main>
<aside class="panel">
  <div class="head">VLM 手机屏幕分析</div>
  <div class="controls">
    <select id="preset">
      <option value="手机屏幕当前打开了哪个 App?在做什么?请简洁描述。">识别当前 App</option>
      <option value="手机屏幕上有哪些可见的通知、消息或弹窗?请列出。">通知/弹窗</option>
      <option value="手机屏幕上正在播放什么视频或显示什么主要内容?">视频/内容</option>
      <option value="手机屏幕上有没有未读消息、来电或其他需要关注的信息?">关注信息</option>
      <option value="custom">自定义提问</option>
    </select>
    <input id="query" placeholder="自定义提问">
    <button id="analyze">分析手机屏幕</button>
    <div id="hint" class="hint"></div>
    <button id="reconnectBtn" class="secondary">重新建立 WebRTC 连接</button>
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
const reconnectBtn = document.getElementById("reconnectBtn");
const statusEl = document.getElementById("status");
const content = document.getElementById("content");
const meta = document.getElementById("meta");
const hint = document.getElementById("hint");
const preset = document.getElementById("preset");
const query = document.getElementById("query");
const btn = document.getElementById("analyze");
let failures = 0;
let statusFailures = 0;
let overlayManuallyHidden = false;

function setStatus(text, error=false){ statusEl.textContent = text; statusEl.style.color = error ? "#f87171" : ""; }
function streamUrl(){ return "/api/phone/stream?token=" + encodeURIComponent(TOKEN) + "&t=" + Date.now(); }
function showOverlay(title, text, isError=false){
  overlayTitle.textContent = title;
  overlayText.textContent = text;
  overlay.classList.add("show");
  if(isError) dot.className = "dot dead"; else dot.className = "dot warn";
}
function hideOverlay(){ overlay.classList.remove("show"); dot.className = "dot"; overlayManuallyHidden = false; }
function connectStream(){
  img.removeAttribute("src");
  img.src = streamUrl();
  setStatus("正在拉流");
  if(!overlayManuallyHidden) showOverlay("画面加载中", "正在连接 MJPEG 推流,请稍候...");
}
function reconnect(){
  failures += 1;
  dot.className = "dot dead";
  setStatus("推流重连中 " + failures, true);
  showOverlay("推流中断", "第 " + failures + " 次尝试重连... 如持续失败,请点击下方重新建立 WebRTC 连接按钮。", true);
  setTimeout(connectStream, Math.min(5000, 600 + failures * 300));
}
img.onload = () => { failures = 0; hideOverlay(); if(img.naturalWidth) frame.textContent = img.naturalWidth + "x" + img.naturalHeight + " · 已连接"; setStatus("推流已连接"); };
img.onerror = () => { reconnect(); };

async function doReconnect(){
  reloadBtn.disabled = true; reconnectBtn.disabled = true;
  reloadBtn.textContent = "重启中...";
  setStatus("正在终止旧连接并重新协商 WebRTC...");
  showOverlay("重新连接", "正在关闭旧 PeerConnection 并重新走 signaling 握手...");
  try {
    const resp = await fetch("/api/phone/restart", { method: "POST", headers: {"Authorization": "Bearer " + TOKEN} });
    const data = await resp.json().catch(() => ({}));
    if(!resp.ok) throw new Error(data.detail || ("HTTP " + resp.status));
    setStatus("已重启,正在拉流...");
    hint.textContent = "WebRTC 连接已重启。";
    hint.className = "hint";
  } catch(e) {
    setStatus("重启失败: " + e, true);
    hint.textContent = "重启失败: " + e;
    hint.className = "hint error";
  } finally {
    reloadBtn.disabled = false; reconnectBtn.disabled = false;
    reloadBtn.textContent = "重新连接";
  }
  connectStream();
}
reloadBtn.addEventListener("click", () => { failures = 0; overlayManuallyHidden = true; doReconnect(); });
reconnectBtn.addEventListener("click", () => { failures = 0; doReconnect(); });

async function pollStatus(){
  try {
    const resp = await fetch("/api/phone/status?token=" + encodeURIComponent(TOKEN), {cache: "no-store"});
    if(!resp.ok) throw new Error("HTTP " + resp.status);
    const data = await resp.json();
    statusFailures = 0;
    if(data.width && data.height) frame.textContent = data.width + "x" + data.height + " · " + (data.frames || 0) + " 帧 · " + (data.fps || 0) + "fps";
    const age = data.latest_frame_age_ms;
    if(!data.running){
      dot.className = "dot dead";
      setStatus("接收未运行: " + (data.last_error || ""), true);
      if(!overlayManuallyHidden) showOverlay("接收未运行", data.last_error || "phone stream 未运行,请点击下方重新连接按钮。", true);
    } else if(data.signaling_state !== "connected"){
      dot.className = "dot warn";
      setStatus("signaling: " + data.signaling_state + (data.last_error ? " | " + data.last_error : ""), true);
      if(!overlayManuallyHidden) showOverlay("正在连接 signaling", "状态: " + data.signaling_state + "\\nsignaling: " + data.signaling_url, false);
    } else if(age == null){
      dot.className = "dot warn";
      setStatus("已连接 signaling,等待手机推流...", true);
      if(!overlayManuallyHidden) showOverlay("等待手机推流", "signaling 已连接,等待 Android 端发起 SDP offer。\\n请确认手机 app 已启动并选择同一 session。", false);
    } else if(age > 5000){
      dot.className = "dot dead";
      setStatus("推流停滞 " + Math.round(age/1000) + "s", true);
    } else {
      dot.className = "dot";
      setStatus("画面正常 · ICE " + data.ice_state);
    }
  } catch(e) {
    statusFailures += 1;
    if(statusFailures >= 3){ dot.className = "dot dead"; setStatus("无法读取状态: " + e, true); }
  }
}
setInterval(pollStatus, 2000);
pollStatus();
preset.addEventListener("change", () => { if(preset.value !== "custom") query.value = preset.value; });
btn.addEventListener("click", async () => {
  let q = query.value.trim();
  if(!q && preset.value !== "custom") q = preset.value;
  if(!q){ setStatus("请输入分析问题", true); hint.textContent = "请选择预设或输入自定义问题。"; hint.className = "hint warn"; return; }
  btn.disabled = true; btn.textContent = "分析中";
  content.textContent = "正在调用视觉模型...";
  setStatus("VLM 推理中");
  try {
    const resp = await fetch("/api/phone/analyze?q=" + encodeURIComponent(q), { headers: {"Authorization": "Bearer " + TOKEN} });
    const data = await resp.json();
    if(!resp.ok) throw new Error(data.detail || data.message || ("HTTP " + resp.status));
    meta.textContent = (data.time || "") + " | " + (data.elapsed || 0) + "s | tokens " + JSON.stringify(data.tokens || {});
    content.textContent = data.content || "";
    setStatus("分析完成");
  } catch(e) {
    content.textContent = "请求失败: " + e;
    setStatus("分析失败", true);
  } finally {
    btn.disabled = false; btn.textContent = "分析手机屏幕";
  }
});
</script>
</body>
</html>
"""


@router.get(
    "", include_in_schema=False, dependencies=[Depends(verify_token_query_fallback)]
)
@router.get(
    "/", include_in_schema=False, dependencies=[Depends(verify_token_query_fallback)]
)
async def phone_page() -> HTMLResponse:
    """手机屏幕推流查看页 (与 /api/screen 同款 UI,去掉 ROI 设置)。"""
    get_phone_capture()
    from miloco.config import get_settings

    token = get_settings().server.token or ""
    html = HTML_PAGE.replace("__MILOCO_TOKEN__", escape_for_js_string(token))
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@router.get("/stream", dependencies=[Depends(verify_token_query_fallback)])
async def phone_stream() -> StreamingResponse:
    """MJPEG 推流,浏览器 <img src> 直接消费。"""
    capture = get_phone_capture()
    return StreamingResponse(
        _mjpeg_stream(capture),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/status", dependencies=[Depends(verify_token_query_fallback)])
async def phone_status() -> dict:
    return asdict(get_phone_capture().status())


@router.post("/restart", dependencies=[Depends(verify_token)])
async def phone_restart() -> dict:
    """重启 WebRTC 接收端 (重新走 signaling 握手)。"""
    await restart_phone_capture()
    await asyncio.sleep(0.3)
    return asdict(get_phone_capture().status())


@router.get("/analyze", dependencies=[Depends(verify_token)])
async def phone_analyze(
    q: str = Query(
        default="手机屏幕当前打开了哪个 App?在做什么?请简洁描述。",
        max_length=500,
    ),
) -> dict:
    return await analyze_phone(q)
