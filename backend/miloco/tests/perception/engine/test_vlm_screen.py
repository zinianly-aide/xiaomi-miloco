"""屏幕采集 → VLM 分析：持续截取 Mac 屏幕，发给视觉大模型理解。

用法：
    cd backend/miloco
    uv run python tests/perception/engine/test_vlm_screen.py           # 单次分析
    uv run python tests/perception/engine/test_vlm_screen.py --loop    # 每 10s 分析一次
    uv run python tests/perception/engine/test_vlm_screen.py --duration 5 --query "屏幕有什么？"
"""
import argparse
import asyncio
import base64
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2

# 绕过 SOCKS 代理
for k in list(os.environ):
    if k.lower() in ("all_proxy", "no_proxy"):
        del os.environ[k]
os.environ["no_proxy"] = "127.0.0.1,localhost"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from miloco.perception.engine.config import OmniConfig
from miloco.perception.engine.omni.omni_client import call_omni


def capture_screen(duration: int = 5, fps: int = 3, scale: int = 1280) -> bytes:
    """用 ffmpeg 抓取屏幕，返回 mp4 字节。"""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        out_path = f.name

    # 获取屏幕分辨率 → 按比例缩放到 scale 宽度
    try:
        info = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=10
        )
        for line in info.stdout.splitlines():
            line = line.strip()
            if line.startswith("Resolution:"):
                # e.g. "Resolution: 2560 x 1440"
                parts = line.split(":")[1].strip().split("x")
                sw, sh = int(parts[0].strip()), int(parts[1].strip().split()[0])
                break
        else:
            sw, sh = 2560, 1440  # fallback
    except Exception:
        sw, sh = 2560, 1440

    target_h = int(sh * scale / sw)
    # 确保宽高为偶数（h264 要求）
    target_w = scale - (scale % 2)
    target_h = target_h - (target_h % 2)

    print(f"📺 屏幕分辨率: {sw}x{sh} → 采集: {target_w}x{target_h} @ {fps}fps × {duration}s")
    t0 = time.monotonic()

    cmd = [
        "ffmpeg", "-y",
        "-f", "avfoundation", "-framerate", str(fps),
        "-i", "0:none",             # video device 0 (screen), no audio
        "-t", str(duration),
        "-vf", f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
        "-pix_fmt", "yuv420p", "-an",
        out_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=duration + 30)
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        stderr = result.stderr[-500:] if result.stderr else ""
        raise RuntimeError(f"ffmpeg 屏幕采集失败 (code={result.returncode}):\n{stderr}")

    with open(out_path, "rb") as fh:
        data = fh.read()
    os.unlink(out_path)

    print(f"   采集完成: {len(data)//1024}KB, 耗时 {elapsed:.1f}s")
    return data


def frames_to_base64_video(mp4_bytes: bytes) -> str:
    """直接返回 mp4 的 base64 编码。"""
    return base64.b64encode(mp4_bytes).decode("ascii")


async def analyze_screen(duration: int, fps: int, scale: int, query: str):
    """采集屏幕 → VLM 分析 → 打印结果。"""
    mp4_data = capture_screen(duration=duration, fps=fps, scale=scale)
    video_b64 = frames_to_base64_video(mp4_data)

    payload = {
        "system_prompt": (
            "你是一个桌面/屏幕内容分析助手。请仔细观察屏幕截图/录屏，"
            "用中文描述你看到的内容：有哪些窗口、应用、文字、图像等。"
            "简洁清晰，重点突出。"
        ),
        "user_content": query,
        "video_base64": video_b64,
        "crops": [],
        "images": [],
    }

    config = OmniConfig(
        model="minicpm-v46-mlx",
        base_url="http://127.0.0.1:8001/v1",
        api_key="local-minicpm",
        timeout=120.0,
        max_completion_tokens=256,
    )

    print(f"\n🚀 调用 VLM ({config.model})...")
    t0 = time.monotonic()

    try:
        resp = await call_omni(payload, config)
        elapsed = time.monotonic() - t0
        content = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})

        print(f"✅ 推理完成 ({elapsed:.1f}s) | "
              f"tokens: prompt={usage.get('prompt_tokens','?')} "
              f"completion={usage.get('completion_tokens','?')}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(content)
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    except Exception as e:
        print(f"❌ VLM 调用失败: {e}")
        raise


async def main():
    parser = argparse.ArgumentParser(description="屏幕采集 → VLM 分析")
    parser.add_argument("--duration", type=int, default=5, help="采集时长(秒)")
    parser.add_argument("--fps", type=int, default=3, help="采集帧率")
    parser.add_argument("--scale", type=int, default=1280, help="缩放宽度")
    parser.add_argument("--query", type=str, default="请描述当前屏幕上的内容。有哪些打开的窗口和应用？",
                        help="分析提问")
    parser.add_argument("--loop", action="store_true", help="循环模式：每隔 duration+5 秒分析一次")
    parser.add_argument("--interval", type=int, default=10, help="循环间隔(秒)")
    args = parser.parse_args()

    print(f"\n🖥️  屏幕 → VLM 分析")
    print(f"   采集: {args.duration}s @ {args.fps}fps, 缩放宽度={args.scale}")
    print(f"   模型: minicpm-v46-mlx @ http://127.0.0.1:8001/v1")
    print(f"   提问: {args.query[:60]}...")
    print()

    if args.loop:
        count = 0
        while True:
            count += 1
            print(f"\n{'='*60}")
            print(f"  第 {count} 轮 ({time.strftime('%H:%M:%S')})")
            print(f"{'='*60}")
            try:
                await analyze_screen(args.duration, args.fps, args.scale, args.query)
            except Exception as e:
                print(f"  ⚠️ 本轮失败: {e}")
            print(f"\n  ⏳ 等待 {args.interval}s ...")
            await asyncio.sleep(args.interval)
    else:
        await analyze_screen(args.duration, args.fps, args.scale, args.query)


if __name__ == "__main__":
    asyncio.run(main())
