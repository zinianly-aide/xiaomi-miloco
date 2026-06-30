"""屏幕推流 + VLM 分析一体化服务。

启动后提供：
  1. http://127.0.0.1:1812/        — 实时屏幕预览 (MJPEG) + VLM 分析面板
  2. http://127.0.0.1:1812/stream   — 原始 MJPEG 流
  3. http://127.0.0.1:1812/api/analyze — 触发 VLM 分析

用法:
    cd backend/miloco
    uv run python tests/perception/engine/screen_service.py
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """多线程 HTTP 服务器，避免 MJPEG 长连接阻塞其他请求。"""
    daemon_threads = True

# 绕过 SOCKS 代理
for k in list(os.environ):
    if k.lower() in ("all_proxy", "no_proxy"):
        del os.environ[k]
os.environ["no_proxy"] = "127.0.0.1,localhost"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [screen] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("screen_service")

# ─── 屏幕采集（ffmpeg 子进程 → BGR rawvideo） ───────────────────────────

class ScreenCapture:
    def __init__(self, fps: int = 3, width: int = 1280):
        self.fps = fps
        self.target_w = width - (width % 2)
        self._proc = None
        self._buf = b""
        self._lock = threading.Lock()
        self._latest_frame: bytes | None = None  # JPEG bytes
        self._frame_buffer: list[bytes] = []  # 保存最近 N 帧用于 VLM 分析
        self._restart_count = 0

    @property
    def target_h(self) -> int:
        return int(self.target_w * 9 / 16)  # 16:9

    def start(self):
        w = self.target_w
        h = self.target_h - (self.target_h % 2)

        cmd = [
            "ffmpeg",
            "-f", "avfoundation", "-framerate", str(self.fps),
            "-i", "0:none",
            "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                   f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "mjpeg", "-q:v", "8",
            "-an", "-nostdin",
            "-f", "image2pipe", "-",
        ]

        logger.info(f"启动屏幕采集: {w}x{h} @ {self.fps}fps")
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        threading.Thread(target=self._read_loop, daemon=True).start()
        # 看门狗：ffmpeg 崩溃后自动重启
        threading.Thread(target=self._watchdog, daemon=True).start()

    def _watchdog(self):
        """监控 ffmpeg 进程，崩溃后自动重启。"""
        while True:
            if self._proc:
                rc = self._proc.wait()
                self._restart_count += 1
                logger.warning(f"ffmpeg 退出 (code={rc})，第 {self._restart_count} 次重启...")
                time.sleep(2)
                try:
                    self.start()
                    return  # start() 会启动新的 watchdog
                except Exception as e:
                    logger.error(f"ffmpeg 重启失败: {e}")
                    time.sleep(5)
            else:
                time.sleep(5)

    def _read_loop(self):
        buf = b""
        while self._proc and self._proc.poll() is None:
            try:
                chunk = self._proc.stdout.read(4096)
                if not chunk:
                    break
                buf += chunk
                # 提取完整的 JPEG 帧
                while True:
                    start = buf.find(b"\xff\xd8")
                    end = buf.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                    if start >= 0 and end > start:
                        with self._lock:
                            self._latest_frame = buf[start:end + 2]
                            self._frame_buffer.append(buf[start:end + 2])
                            if len(self._frame_buffer) > 15:  # 保留最近 15 帧
                                self._frame_buffer.pop(0)
                        buf = buf[end + 2:]
                    else:
                        break
            except Exception:
                break

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self._latest_frame

    def get_recent_frames(self, count: int = 9) -> list[bytes]:
        """获取最近 N 帧 JPEG，用于 VLM 分析（不启动新的抓屏进程）。"""
        with self._lock:
            buf = list(self._frame_buffer)
        if len(buf) <= count:
            return buf
        step = len(buf) // count
        return [buf[i] for i in range(0, len(buf), step)][:count]

    def stop(self):
        if self._proc:
            self._proc.terminate()
            self._proc = None


# ─── VLM 调用 ────────────────────────────────────────────────────────────

_last_analysis: dict = {"content": "等待分析...", "time": "", "elapsed": 0}

async def run_vlm_analysis(query: str, capture: ScreenCapture | None = None):
    """从已采集的帧缓冲区取帧 → 编码 MP4 → VLM 分析"""
    global _last_analysis

    from miloco.perception.engine.config import OmniConfig
    from miloco.perception.engine.omni.omni_client import call_omni

    # 从推流缓冲区获取最近帧（不启动新 ffmpeg，避免设备冲突）
    if capture is None:
        capture = Handler.capture
    if capture is None:
        _last_analysis = {
            "content": "屏幕采集未启动",
            "time": time.strftime("%H:%M:%S"), "elapsed": 0, "tokens": {},
        }
        return

    jpeg_frames = capture.get_recent_frames(9)
    if len(jpeg_frames) < 2:
        _last_analysis = {
            "content": f"帧缓冲区不足（当前 {len(jpeg_frames)} 帧），请稍后重试",
            "time": time.strftime("%H:%M:%S"), "elapsed": 0, "tokens": {},
        }
        return

    logger.info(f"VLM 分析：使用 {len(jpeg_frames)} 帧")

    # JPEG 帧 → ffmpeg 合成 MP4
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, jpg in enumerate(jpeg_frames):
            with open(os.path.join(tmpdir, f"f{i:04d}.jpg"), "wb") as f:
                f.write(jpg)
        mp4_path = os.path.join(tmpdir, "out.mp4")
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y",
            "-framerate", "3", "-i", f"{tmpdir}/f%04d.jpg",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
            "-pix_fmt", "yuv420p", "-an", mp4_path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.wait_for(proc.communicate(), timeout=10)

        with open(mp4_path, "rb") as fh:
            mp4_data = fh.read()

    if len(mp4_data) < 500:
        _last_analysis = {
            "content": "视频编码失败",
            "time": time.strftime("%H:%M:%S"), "elapsed": 0, "tokens": {},
        }
        return

    video_b64 = base64.b64encode(mp4_data).decode("ascii")
    logger.info(f"编码完成: {len(mp4_data)//1024}KB → base64 {len(video_b64)} 字符")

    payload = {
        "system_prompt": (
            "你是一个桌面活动监控助手。观察屏幕截图/录屏，用中文描述用户正在做什么。"
            "重点关注：1) 打开了哪些应用窗口 2) 正在编辑什么内容 3) 是工作/娱乐/通讯场景。"
            "不要描述「有一个用户提问」或「聊天界面」这种元信息——直接描述屏幕上的实质内容。"
            "例如看到代码就说在写代码，看到文档就说在编辑文档。简洁，3-5 句即可。"
        ),
        "user_content": query or "请描述当前屏幕上的内容",
        "video_base64": video_b64,
        "crops": [],
        "images": [],
    }

    config = OmniConfig(
        model="minicpm-v46-mlx",
        base_url="http://127.0.0.1:8001/v1",
        api_key="local-minicpm",
        timeout=60.0,
        max_completion_tokens=256,
    )

    t0 = time.monotonic()
    try:
        resp = await call_omni(payload, config)
        elapsed = time.monotonic() - t0
        content = resp["choices"][0]["message"]["content"]
        _last_analysis = {
            "content": content,
            "time": time.strftime("%H:%M:%S"),
            "elapsed": round(elapsed, 1),
            "tokens": resp.get("usage", {}),
        }
    except Exception as e:
        _last_analysis = {
            "content": f"分析失败: {e}",
            "time": time.strftime("%H:%M:%S"),
            "elapsed": 0,
            "tokens": {},
        }


# ─── HTTP 服务器 ─────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>屏幕感知</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#1a1a2e;color:#e0e0e0;font-family:-apple-system,system-ui,sans-serif;
     display:flex;height:100vh;overflow:hidden}
.left{flex:1;display:flex;flex-direction:column;background:#000;min-width:0}
.left .bar{background:#16213e;padding:8px 16px;font-size:13px;color:#7f8c8d;flex-shrink:0;
  display:flex;justify-content:space-between;align-items:center}
.left .bar .status-dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:6px}
.left .bar .alive{background:#00e676}
.left .bar .dead{background:#e94560}
.left img{flex:1;width:100%;object-fit:contain;background:#000}
.right{width:420px;background:#16213e;display:flex;flex-direction:column;flex-shrink:0}
.right .head{background:#0f3460;padding:16px;font-size:18px;font-weight:700}
.right .controls{padding:12px 16px;border-bottom:1px solid #1a1a3e}
.right .controls select{width:100%;padding:8px 12px;background:#1a1a3e;border:1px solid #333;
  border-radius:6px;color:#e0e0e0;font-size:14px;margin-bottom:8px}
.right .controls input{width:100%;padding:8px 12px;background:#1a1a3e;border:1px solid #333;
  border-radius:6px;color:#e0e0e0;font-size:14px;margin-bottom:8px}
.right .controls button{width:100%;padding:10px;background:#e94560;color:#fff;border:none;
  border-radius:6px;font-size:15px;cursor:pointer;font-weight:600}
.right .controls button:hover{background:#ff6b81}
.right .controls button:disabled{background:#555;cursor:not-allowed}
.right .result{flex:1;overflow-y:auto;padding:16px}
.right .result .time{font-size:12px;color:#7f8c8d;margin-bottom:8px}
.right .result .elapsed{font-size:11px;color:#e94560;margin-bottom:12px}
.right .result .content{font-size:15px;line-height:1.7;white-space:pre-wrap;word-break:break-word}
.right .status{padding:12px 16px;border-top:1px solid #1a1a3e;font-size:12px;color:#7f8c8d}
</style>
</head>
<body>
<div class="left">
  <div class="bar">
    <span><span class="status-dot alive" id="dot"></span>📺 屏幕实时 · 3fps</span>
    <span style="font-size:11px" id="frame-info"></span>
  </div>
  <img src="/stream" id="screen" alt="屏幕画面"
       onerror="reconnect()"
       onload="onFrameLoad()" />
</div>
<div class="right">
  <div class="head">🧠 VLM 屏幕分析</div>
  <div class="controls">
    <select id="preset" onchange="onPreset()">
      <option value="custom">🖊️ 自定义提问</option>
      <option value="当前屏幕打开了哪些应用和窗口？请列出名称和它们大致在做什么。">📋 列出应用窗口</option>
      <option value="屏幕上的主要内容是什么？有什么值得注意的信息？请简洁描述。">📝 内容摘要</option>
      <option value="当前屏幕显示的是什么类型的页面或应用？是工作内容、娱乐、通讯还是其他？">🏷️ 识别场景类型</option>
      <option value="屏幕中有没有代码、终端、文档编辑等开发相关的内容？">💻 开发检测</option>
    </select>
    <input id="query" placeholder="自定义提问（选上面的预设或自己写）" />
    <button id="analyze-btn" onclick="analyze()">🔍 分析当前屏幕</button>
  </div>
  <div class="result" id="result">
    <div class="time" id="time"></div>
    <div class="elapsed" id="elapsed"></div>
    <div class="content" id="content">选择一个提问方式，点击「分析」按钮开始...</div>
  </div>
  <div class="status" id="status">🟢 就绪</div>
</div>
<script>
var streamFailures = 0;

function onFrameLoad(){
  document.getElementById('dot').className = 'status-dot alive';
  streamFailures = 0;
  // 更新帧大小信息
  var img = document.getElementById('screen');
  if(img.naturalWidth) {
    document.getElementById('frame-info').textContent = img.naturalWidth + '×' + img.naturalHeight;
  }
}

function reconnect(){
  streamFailures++;
  var img = document.getElementById('screen');
  document.getElementById('dot').className = 'status-dot dead';
  if(streamFailures > 10){
    document.getElementById('status').textContent = '🔴 推流中断，请刷新页面';
    return;
  }
  var src = img.src;
  img.src = '';
  setTimeout(function(){ img.src = src + '?t=' + Date.now(); }, 1000);
  document.getElementById('status').textContent = '🟡 重连中 (' + streamFailures + ')...';
}

function onPreset(){
  var v = document.getElementById('preset').value;
  if(v !== 'custom') document.getElementById('query').value = v;
}

async function analyze(){
  var btn=document.getElementById('analyze-btn');
  var q=document.getElementById('query').value.trim();
  if(!q) q = document.getElementById('preset').value;
  if(!q || q === 'custom') { alert('请选择或输入提问'); return; }

  btn.disabled=true; btn.textContent='⏳ 分析中...';
  document.getElementById('content').textContent='正在分析屏幕内容...';
  document.getElementById('status').textContent='🔵 VLM 推理中...';
  try{
    var r=await fetch('/api/analyze?'+new URLSearchParams({q:q}));
    var d=await r.json();
    document.getElementById('time').textContent='🕐 '+d.time;
    document.getElementById('elapsed').textContent='⚡ '+d.elapsed+'s | tokens: '+JSON.stringify(d.tokens);
    document.getElementById('content').textContent=d.content;
    document.getElementById('status').textContent='✅ 完成 ('+d.elapsed+'s)';
  }catch(e){
    document.getElementById('content').textContent='❌ 请求失败: '+e;
    document.getElementById('status').textContent='🔴 分析失败';
  }
  btn.disabled=false; btn.textContent='🔍 分析当前屏幕';
}

// 初始选中第一个预设
document.getElementById('preset').selectedIndex = 1;
document.getElementById('query').value = document.getElementById('preset').value;
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    capture: ScreenCapture | None = None

    def log_message(self, format, *args):
        pass  # 静默

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self._serve_html()
        elif parsed.path == "/stream":
            self._serve_mjpeg()
        elif parsed.path == "/api/analyze":
            self._serve_analyze(parsed)
        elif parsed.path == "/api/status":
            self._serve_json({"status": "ok", "fps": 3})
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode())

    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()

        cap = self.__class__.capture
        if not cap:
            self.wfile.write(b"--frame\r\n\r\nno capture\r\n")
            return

        last = b""
        deadline = time.time() + 300  # 最多 5 分钟
        while time.time() < deadline:
            frame = cap.get_frame()
            if frame and frame != last:
                last = frame
                try:
                    self.wfile.write(
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                        + frame + b"\r\n"
                    )
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    break
            else:
                time.sleep(0.1)

    def _serve_analyze(self, parsed):
        params = parse_qs(parsed.query)
        query = params.get("q", [""])[0]

        # 在后台线程执行 VLM 分析
        cap = self.__class__.capture
        def _run():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(run_vlm_analysis(query, cap))
                loop.close()
            except Exception as e:
                import traceback
                global _last_analysis
                _last_analysis = {
                    "content": f"分析异常: {type(e).__name__}: {e}\n{traceback.format_exc()[-500:]}",
                    "time": time.strftime("%H:%M:%S"),
                    "elapsed": 0,
                    "tokens": {},
                }
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=60)  # 等待最多 60 秒

        self._serve_json(_last_analysis)

    def _serve_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode())


def main():
    import argparse
    parser = argparse.ArgumentParser(description="屏幕推流 + VLM 分析服务")
    parser.add_argument("--port", type=int, default=1812)
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()

    logger.info(f"启动屏幕推流服务: http://127.0.0.1:{args.port}")
    logger.info(f"  MJPEG 流: http://127.0.0.1:{args.port}/stream")
    logger.info(f"  VLM 分析: http://127.0.0.1:{args.port}/api/analyze?q=...")

    cap = ScreenCapture(fps=args.fps, width=args.width)
    cap.start()
    Handler.capture = cap

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
