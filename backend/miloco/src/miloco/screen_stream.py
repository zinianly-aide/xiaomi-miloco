"""屏幕采集 → H.264 WebSocket 推流服务。

独立进程运行，提供与 miloco 相同协议的 WS 视频流。
miloco 后端检测到虚拟摄像头时，把 WS 连接转发到这里。

用法：
    cd backend
    uv run python -m miloco.screen_stream --port 1811
"""
from __future__ import annotations

import asyncio
import logging
import struct
import time
from fractions import Fraction

import av
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger("screen_stream")


# ─── ffmpeg 子进程屏幕采集 ──────────────────────────────────────────────

async def _read_frames(proc: asyncio.subprocess.Process, width: int, height: int):
    """从 ffmpeg stdout 读取 rawvideo BGR 帧。"""
    frame_size = width * height * 3
    buf = b""
    while True:
        try:
            chunk = await proc.stdout.read(8192)
            if not chunk:
                break
            buf += chunk
            while len(buf) >= frame_size:
                raw = buf[:frame_size]
                buf = buf[frame_size:]
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                yield frame
        except Exception:
            break


async def screen_capture_loop(
    fps: int = 3, width: int = 1280, height: int = 720, monitor: int = 0
):
    """启动 ffmpeg 屏幕采集子进程，持续 yield BGR ndarray。"""
    # macOS: AVFoundation 屏幕采集
    # 先获取实际屏幕分辨率来正确缩放
    try:
        info = await asyncio.create_subprocess_exec(
            "system_profiler", "SPDisplaysDataType",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await info.communicate()
        sw = 2560
        for line in stdout.decode().splitlines():
            line = line.strip()
            if line.startswith("Resolution:"):
                parts = line.split(":")[1].strip().split("x")
                sw = int(parts[0].strip())
                break
    except Exception:
        sw = 2560

    target_h = int(height * sw / width) if width > 0 else height
    # 偶数
    w = width - (width % 2)
    h = target_h - (target_h % 2)

    cmd = [
        "ffmpeg",
        "-f", "avfoundation", "-framerate", str(fps),
        "-i", f"{monitor}:none",
        "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
               f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,format=bgr24",
        "-c:v", "rawvideo", "-pix_fmt", "bgr24",
        "-an", "-nostdin",
        "-f", "rawvideo", "pipe:1",
    ]

    logger.info(f"启动屏幕采集: {' '.join(cmd[:8])}...")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # 启动后短暂等待确认没有立即退出
    await asyncio.sleep(0.5)
    if proc.returncode is not None:
        stderr = (await proc.stderr.read()).decode()[-500:]
        raise RuntimeError(f"ffmpeg 启动失败: {stderr}")

    # 后台吞 stderr（避免缓冲区满阻塞）
    async def drain_stderr():
        while True:
            data = await proc.stderr.read(4096)
            if not data:
                break

    asyncio.create_task(drain_stderr())

    async for frame in _read_frames(proc, w, h):
        yield frame

    await proc.wait()


# ─── H.264 编码器 ───────────────────────────────────────────────────────

class ScreenH264Encoder:
    """轻量 H.264 encoder，输出的 NAL 单元供 WS 发送。"""

    def __init__(self, gop: int = 30):
        self._gop = gop
        self._codec: av.codec.CodecContext | None = None
        self._width = 0
        self._height = 0
        self._pts_counter = 0

    def _ensure_codec(self, width: int, height: int):
        if self._codec and self._width == width and self._height == height:
            return
        if self._codec:
            # resolution change — close old
            self._codec = None
        codec = av.codec.CodecContext.create("libx264", "w")
        codec.width = width
        codec.height = height
        codec.pix_fmt = "yuv420p"
        codec.time_base = Fraction(1, 1000)
        codec.framerate = Fraction(3, 1)  # 3 fps
        codec.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "crf": "30",
            "profile": "baseline",
            "level": "3.1",
        }
        codec.gop_size = self._gop
        self._codec = codec
        self._width = width
        self._height = height

    def encode(self, frame: NDArray[np.uint8]) -> list[tuple[bytes, bool]]:
        """Encode BGR frame → list of (nal_bytes, is_keyframe)."""
        h, w = frame.shape[:2]
        self._ensure_codec(w, h)

        av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        av_frame.pts = self._pts_counter
        self._pts_counter += 1

        packets = self._codec.encode(av_frame)
        result = []
        for pkt in packets:
            if pkt.is_keyframe:
                # 预插 SPS/PPS（extradata）
                ext = self._codec.extradata
                if ext:
                    result.insert(0, (bytes(ext), True))
            result.append((bytes(pkt), pkt.is_keyframe))
        return result


# ─── WebSocket 服务器 ────────────────────────────────────────────────────

# 与 miloco ws.py 相同的 wire protocol
HEADER_SIZE = 16  # frame_type:u8 + padding:7 + timestamp:u64 big-endian


class ScreenStreamServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 1811):
        self.host = host
        self.port = port
        self._clients: set[asyncio.Queue] = set()
        self._encoder = ScreenH264Encoder()
        self._running = False

    async def _broadcast(self, data: bytes):
        dead = set()
        for q in self._clients:
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                dead.add(q)
        self._clients -= dead

    async def _capture_and_stream(self):
        """主循环：采集屏幕 → 编码 → 广播。"""
        logger.info("屏幕推流循环启动")
        async for frame in screen_capture_loop():
            if not self._clients:
                await asyncio.sleep(0.5)
                continue

            try:
                nal_packets = self._encoder.encode(frame)
            except Exception as e:
                logger.error(f"编码失败: {e}")
                continue

            ts_ms = int(time.time() * 1000) & 0xFFFFFFFFFFFFFFFF
            for nal_bytes, is_key in nal_packets:
                header = struct.pack(
                    ">B7xQ",
                    1 if is_key else 0,  # frame_type
                    ts_ms,                 # timestamp
                )
                await self._broadcast(header + nal_bytes)

        logger.info("屏幕推流循环结束")

    async def _handle_client(self, websocket):
        """处理单个 WS 客户端。"""
        q: asyncio.Queue = asyncio.Queue(maxsize=30)

        # 发送 init 消息
        await websocket.send_json({
            "type": "init",
            "codec": "h264",
            "container": "annexb",
        })

        self._clients.add(q)
        cid = id(q)
        logger.info(f"WS 客户端连接: {cid}, 当前 {len(self._clients)} 个")

        # WS → 客户端发送队列
        async def sender():
            try:
                while True:
                    data = await q.get()
                    try:
                        await websocket.send_bytes(data)
                    except Exception:
                        break
            except Exception:
                pass

        send_task = asyncio.create_task(sender())

        try:
            # 只接收 keepalive
            while True:
                try:
                    await asyncio.wait_for(websocket.receive(), timeout=30)
                except asyncio.TimeoutError:
                    pass  # 30s 心跳无数据，继续
        except Exception:
            pass
        finally:
            self._clients.discard(q)
            send_task.cancel()
            try:
                await websocket.close()
            except Exception:
                pass
            logger.info(f"WS 客户端断开: {cid}")

    async def start(self):
        """启动 WebSocket 服务器。"""
        # 尝试使用 websockets 库
        try:
            import websockets
        except ImportError:
            logger.error("需要 websockets 库: uv pip install websockets")
            raise

        self._running = True
        asyncio.create_task(self._capture_and_stream())

        async def handler(ws):
            await self._handle_client(ws)

        logger.info(f"屏幕推流 WS 服务启动: ws://{self.host}:{self.port}")
        async with websockets.serve(handler, self.host, self.port):
            await asyncio.Future()  # run forever


def main():
    import argparse
    parser = argparse.ArgumentParser(description="屏幕 → H.264 WS 推流")
    parser.add_argument("--port", type=int, default=1811)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    server = ScreenStreamServer(host=args.host, port=args.port)
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("停止")


if __name__ == "__main__":
    main()
