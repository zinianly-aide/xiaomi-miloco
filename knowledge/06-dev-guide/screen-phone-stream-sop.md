# 屏幕采集与手机推流 SOP

本 SOP 覆盖 miloco 屏幕监控体系的三路信号源：本机屏幕采集（virtual-screen-0）、手机屏幕推流（virtual-phone-0）、以及共享的 VLM 分析后端。

## 1. 信号源概览

| 信号源 | did | 采集方式 | 入口端点 | 适用场景 |
|---|---|---|---|---|
| 本机屏幕 | `virtual-screen-0` | ffmpeg + avfoundation 抓屏 → MJPEG | `/api/screen` | 桌面感知、屏幕内容理解 |
| 手机推流 | `virtual-phone-0` | WebRTC（aiortc）接收 Android MediaProjection → JPEG | `/api/phone` | 手机 App 画面感知、远程查看 |

两路信号源共享同一套 VLM 分析后端（OmniClient + minicpm-v46-mlx），仅 system_prompt 语境不同。

## 2. 本机屏幕采集（virtual-screen-0）

### 2.1 启动条件

- **macOS**：需要授予运行 miloco-backend 的进程「屏幕录制」权限。
  - 路径：系统设置 > 隐私与安全 > 屏幕录制
  - 授权对象：实际启动 miloco-backend 的应用（Terminal / iTerm / WorkBuddy 等），不是 Python 解释器本身
  - **权限是 per-binary 的**：换终端/换启动方式都要重新授权
- **Linux**：需要 X11 或 PipeWire 屏幕捕获权限
- **Windows**：需要启用「图形捕获」

### 2.2 黑屏排查

| 现象 | 原因 | 解决 |
|---|---|---|
| 画面全黑，status.running=true，frames 在增加 | macOS 屏幕录制权限未授予 | 系统设置授权后，点页面「重新加载画面」按钮（调 `POST /api/screen/restart` 重启 ffmpeg 进程） |
| status.last_error 含 `AVCaptureScreenInput not linked` | 同上，avfoundation 链接失败 | 同上 |
| status.running=false | ffmpeg 启动失败 | 看 `last_error` 字段；常见是显示器索引错（`monitor` 配置） |
| 画面卡住，latest_frame_age_ms 持续增大 | ffmpeg 进程 hang | 点「重新加载画面」重启采集 |

### 2.3 关键端点

```
GET  /api/screen              # 查看页（含 MJPEG 预览 + VLM 分析面板）
GET  /api/screen/stream       # MJPEG 推流（multipart/x-mixed-replace）
GET  /api/screen/status       # 采集状态（含 permission_hint 权限提示）
GET  /api/screen/config       # 当前配置（ROI、fps、分辨率）
POST /api/screen/config       # 更新配置
POST /api/screen/restart      # 重启 ffmpeg 进程（macOS 授权后必须调）
GET  /api/screen/analyze?q=   # VLM 分析当前屏幕
```

### 2.4 ROI 模式

ROI（Region of Interest）模式只采集屏幕指定区域，降低带宽和 CPU：

- 开启 ROI 后 fps 默认降到 1（全屏默认 3）
- 坐标原点是屏幕左上角
- 配置项：`x, y, width, height, fps`
- 用例：只看某个窗口、只看屏幕一角的通知区

## 3. 手机屏幕推流（virtual-phone-0）

### 3.1 架构

```
Android App (QuestPhoneStream)
  MediaProjection → WebRTC H.264 video track
  ↓ SDP offer + ICE
Signaling Server (ws://host:8787)
  ↓ relay
miloco PhoneCapture (aiortc)
  RTCPeerConnection → av.VideoFrame → PIL → JPEG → deque
  ↓ MJPEG stream
浏览器 /api/phone
```

### 3.2 启动流程

1. **启动 signaling server**
   ```bash
   cd /path/to/QuestPhoneStream/apps/signaling-server
   pnpm dev
   # 监听 ws://0.0.0.0:8787
   ```

2. **启动 miloco-backend**（自动连接 signaling）
   ```bash
   cd backend
   .venv/bin/miloco-backend
   ```

3. **手机 app 配置**（关键！deviceId 必须和 miloco 一致）
   - Signaling URL: `ws://<电脑局域网IP>:8787`
   - Token: `dev-token`
   - Session ID: `miloco-session-001`
   - Android Device ID: `android-phone-001`
   - Quest Device ID: `quest-3s-001`（必须和 miloco 的 `quest_device_id` 一致）

4. **手机 app 启动推流** → miloco 自动收到 offer、建立 WebRTC 连接

### 3.3 配置覆盖

miloco 端通过环境变量调整（无需改代码）：

```bash
MILOCO_PHONE_STREAM_SIGNALING_URL=ws://127.0.0.1:8787
MILOCO_PHONE_STREAM_TOKEN=dev-token
MILOCO_PHONE_STREAM_QUEST_DEVICE_ID=quest-3s-001
MILOCO_PHONE_STREAM_ANDROID_DEVICE_ID=android-phone-001
MILOCO_PHONE_STREAM_SESSION_ID=miloco-session-001
MILOCO_PHONE_STREAM_JPEG_QUALITY=75       # 10-95
MILOCO_PHONE_STREAM_BUFFER_SIZE=24         # 帧缓冲容量
```

### 3.4 排查

| 现象 | 排查 |
|---|---|
| `peer_unavailable` | 手机 app 没注册到 signaling，或 deviceId 不匹配。查 signaling 日志确认 `register` 消息 |
| `ice_state: new` 一直不变 | SDP 协商未完成。查 signaling 日志是否有 `offer` 和 `answer` |
| `ice_state: failed` | NAT/防火墙阻断 UDP。检查 STUN 配置，或改用 TURN |
| 画面卡顿/帧率低 | 见下方「稳定性与画质优化」 |
| `ConnectError` on analyze | VLM 模型服务未启动（见第 4 节） |

### 3.5 signaling server 调试日志

`apps/signaling-server/src/index.ts` 的 `handleMessage` 里有调试日志（register/create_session/offer/answer/ice），排查协商问题时打开 `tail -f` 看：

```bash
tail -f /tmp/qps-signaling.log
```

## 4. VLM 分析后端

两路信号源共享同一个 VLM 调用路径：

```
帧缓冲 deque → base64 编码 → OmniClient → minicpm-v46-mlx (http://127.0.0.1:8001/v1)
```

### 4.1 启动 VLM 服务

```bash
# mlx-community 模型（Apple Silicon）
mlx_lm.server --model mlx-community/minicpm-v46-mlx --port 8001
```

或用环境变量指向其他兼容 OpenAI API 的多模态服务：

```bash
MILOCO_SCREEN_VLM_BASE_URL=http://<vlm-host>/v1
MILOCO_SCREEN_VLM_MODEL=<model-name>
MILOCO_SCREEN_VLM_API_KEY=<key>
MILOCO_SCREEN_VLM_TIMEOUT=90
MILOCO_SCREEN_VLM_MAX_TOKENS=256
```

### 4.2 代理环境变量注意

VLM 走 loopback（127.0.0.1）时，miloco 会临时清除 `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` 环境变量，避免代理拦截本地请求。但如果系统设置了 `ALL_PROXY=socks5://...`，httpx 仍会尝试走 SOCKS 代理，需要安装 `socksio`：

```bash
uv pip install socksio   # 或 pip install "httpx[socks]"
```

## 5. 日常运维

### 5.1 检查信号源状态

```bash
TOKEN=<your-token>
# 屏幕采集
curl -s "http://127.0.0.1:1810/api/screen/status?token=$TOKEN" | python3 -m json.tool
# 手机推流
curl -s "http://127.0.0.1:1810/api/phone/status?token=$TOKEN" | python3 -m json.tool
```

### 5.2 重启信号源

```bash
# 重启屏幕采集（macOS 授权后必须调）
curl -X POST "http://127.0.0.1:1810/api/screen/restart" -H "Authorization: Bearer $TOKEN"

# 重启手机推流 WebRTC 连接
curl -X POST "http://127.0.0.1:1810/api/phone/restart" -H "Authorization: Bearer $TOKEN"
```

### 5.3 摄像头列表

两路虚拟设备都会出现在 `/api/miot/scope/cameras`，和真实米家摄像头并列。前端 `/api/miot/watch?camera_id=virtual-screen-0` 或 `virtual-phone-0` 会嵌入对应查看页。

## 6. 已知限制

- **macOS 屏幕权限**：per-binary，换启动方式需重新授权
- **WebRTC ICE**：当前只用 STUN（stun.l.google.com），复杂 NAT 环境下可能需要 TURN
- **VLM 延迟**：minicpm-v46-mlx 在 M 系列芯片上单次分析约 5-15s，不适合实时分析
- **手机推流分辨率**：由 Android app 决定，miloco 端只转码不缩放
- **并发**：PhoneCapture 是单例，同一时间只能接收一路手机推流
