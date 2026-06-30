"""快速测试视觉大模型：从视频抽帧 → 发给 omni VLM"""
import asyncio
import base64
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2

# 绕过 SOCKS 代理（localhost 不需要走代理）
for k in list(os.environ):
    if k.lower() in ("all_proxy", "no_proxy"):
        del os.environ[k]
os.environ["no_proxy"] = "127.0.0.1,localhost"

# 添加项目路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from miloco.perception.engine.config import OmniConfig
from miloco.perception.engine.omni.omni_client import call_omni
from miloco.perception.utils import snapshot_from_video
from miloco.perception.types import PerceptionDevice


async def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_video.mp4"
    query = sys.argv[2] if len(sys.argv) > 2 else "请用中文描述视频中的场景，包括有什么物体、人物、他们在做什么。"

    device = PerceptionDevice(
        did="test-cam", name="测试相机", device_type="camera", room_name="客厅"
    )

    print(f"📹 读取视频: {video_path}")
    t0 = time.monotonic()
    snapshot = snapshot_from_video(video_path, device=device, target_fps=3)
    frames = [f.data for f in snapshot.video.frames]  # NDArray list
    print(f"   解析完成: {len(frames)} 帧, 耗时 {time.monotonic()-t0:.1f}s")

    # 抽取关键帧（取首/中/尾 → ffmpeg h264 → base64）
    n = len(frames)
    indices = [0, n // 2, n - 1] if n >= 3 else list(range(n))
    key_frames = [frames[i] for i in indices]

    # 写临时帧 → ffmpeg 编码 mp4 → base64
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, f in enumerate(key_frames):
            cv2.imwrite(os.path.join(tmpdir, f"f{idx:04d}.png"), f)
        mp4_path = os.path.join(tmpdir, "out.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", "1", "-i", f"{tmpdir}/f%04d.png",
             "-c:v", "libx264", "-preset", "ultrafast", "-crf", "30",
             "-pix_fmt", "yuv420p", "-an", mp4_path],
            capture_output=True,
        )
        with open(mp4_path, "rb") as fh:
            video_b64 = base64.b64encode(fh.read()).decode("ascii")
    print(f"   视频编码: {len(video_b64)} 字符 base64")

    # 构建 payload
    payload = {
        "system_prompt": "你是一个家庭智能助手。请用中文回答。回复简洁清晰。",
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
        max_completion_tokens=512,
    )

    print(f"\n🚀 调用 VLM: {config.model} @ {config.base_url}")
    print(f"   Query: {query[:80]}...")
    t0 = time.monotonic()

    try:
        resp = await call_omni(payload, config)
        elapsed = time.monotonic() - t0
        content = resp["choices"][0]["message"]["content"]
        usage = resp.get("usage", {})

        print(f"\n✅ 推理完成 ({elapsed:.1f}s)")
        print(f"   Tokens: prompt={usage.get('prompt_tokens','?')}, completion={usage.get('completion_tokens','?')}")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(content)
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
