# 手机推流稳定性与画面质量优化方向

本文档分析当前手机屏幕推流（virtual-phone-0）的实现，列出可提升稳定性和画面质量的优化方向。作为后续迭代的参考，非当前 sprint 必做项。

## 现状基线

| 维度 | 当前值 | 说明 |
|---|---|---|
| 编码 | H.264（Android WebRTC 默认） | 手机端编码，miloco 端解码 |
| 转码 | av.VideoFrame → PIL → JPEG (quality=75) | 每帧都做 RGB 转换 + JPEG 重压缩 |
| 分辨率 | 320×180 ~ 640×360（动态） | 由 Android app MediaProjection 决定 |
| 帧率 | 30-61 fps（观测值） | WebRTC 自适应，带宽好时偏高 |
| 缓冲 | deque(maxlen=24) | 约 0.4-0.8s 的帧窗口 |
| MJPEG 推流 | 50ms 轮询 | 固定间隔，非事件驱动 |
| ICE | STUN only (stun.l.google.com) | 无 TURN，NAT 穿透能力有限 |
| 重连 | 2s 固定间隔，无限重试 | 无退避策略 |

---

## 一、稳定性优化

### 1.1 ICE / NAT 穿透

**问题**：当前只用 `stun:stun.l.google.com`，在对称型 NAT（symmetric NAT）、企业防火墙、CGNAT 环境下 ICE 会 failed。

**方案**：
- 增加 TURN server 配置（自建 coturn 或用商业 TURN 服务）
- ICE 配置改为环境变量可配：
  ```python
  ice_servers = [
      {"urls": "stun:stun.l.google.com:19302"},
      {"urls": "turn:turn.example.com:3478", "username": "...", "credential": "..."},
  ]
  ```
- 支持 `MILOCO_PHONE_STREAM_ICE_SERVERS` 环境变量（JSON 数组）

**优先级**：高（直接影响跨网络可用性）

### 1.2 重连退避策略

**问题**：`_run_forever` 断线后固定 2s 重试，无限循环。signaling server 持续不可用时会 CPU 空转 + 日志刷屏。

**方案**：
- 指数退避：2s → 4s → 8s → 16s → 30s（上限）
- 连续失败 N 次后切换到「低频心跳」模式（每 60s 探测一次）
- 成功连接后重置退避计数

```python
backoff = min(30, 2 ** min(failure_count, 4))
await asyncio.wait_for(self._stop_event.wait(), timeout=backoff)
```

**优先级**：中

### 1.3 心跳与连接保活

**问题**：当前依赖 websockets 库的 ping/pong（15s 间隔）。WebRTC 层没有应用层心跳，PC 进入 `disconnected` 状态时不会主动重启。

**方案**：
- 监听 `pc.on("connectionstatechange")`，状态变 `disconnected` 或 `failed` 时主动 `restart()`
- 加 ICE restart：`pc.restartIce()` 而非完全重建 PC（更快恢复）
- 设置 `latest_frame_age_ms > 10000` 触发自动重连

**优先级**：高

### 1.4 signaling 连接竞态

**问题**：`_run_once` 里 ws 和 pc 的创建/清理在 `finally` 块，但 `_handle_signaling_message` 是在 `async for raw in self._ws` 循环里调用的。如果 `_send`（发 answer）时 ws 已经被另一个重连任务关闭，会静默失败。

**方案**：
- 给所有 ws 操作加 `if self._ws is None or self._ws.closed: return` 守卫
- 用 `_ws_lock` 保护 ws 的 send/close，避免并发

**优先级**：中

---

## 二、画面质量优化

### 2.1 减少转码损失

**问题**：当前链路 `H.264 → av.VideoFrame (rgb24) → PIL Image → JPEG`，每帧都做一次颜色空间转换 + JPEG 重压缩。H.264 已经是压缩格式，再转 JPEG 会引入二次量化损失。

**方案 A（推荐）：直接用 H.264 关键帧**
- 从 WebRTC track 提取 I-frame（关键帧），直接作为 JPEG（H.264 I-frame 和 JPEG 的 DCT 系数结构相似）
- 或用 `frame.to_ndarray(format="yuv420p")` 保留 YUV，避免 YUV→RGB→YUV 的来回转换

**方案 B：提高 JPEG quality**
- 当前 quality=75 偏低，文字会有伪影
- 提到 85-90，带宽增加约 30% 但文字清晰度显著提升
- 做成动态：VLM 分析时用 quality=90，MJPEG 预览用 quality=70

**优先级**：中（方案 B 立即可做，方案 A 需调研）

### 2.2 分辨率提升

**问题**：Android app 默认推 320×180 或 640×360，文字基本看不清，VLM 分析准确率受限。

**方案**：
- Android app 端配置 `MediaProjection` 采集原始分辨率（如 1080×1920）
- miloco 端按需缩放：MJPEG 预览降到 720p 省带宽，VLM 分析用原分辨率
- 环境变量 `MILOCO_PHONE_STREAM_TARGET_WIDTH` / `TARGET_HEIGHT` 控制

**优先级**：高（直接影响 VLM 分析质量）

### 2.3 帧率控制

**问题**：观测到 61fps，远超人眼需要。高帧率浪费带宽和 CPU（每帧都要 JPEG 编码）。

**方案**：
- miloco 端做帧率限制：只取最新帧，按目标 fps（如 10-15fps）推 MJPEG
- `_receive_frames` 里加 `await asyncio.sleep(1/target_fps)` 节流
- VLM 分析只取 3 帧，不需要高帧率缓冲

```python
target_fps = 15
min_interval = 1.0 / target_fps
last_recv = 0
while ...:
    now = time.monotonic()
    if now - last_recv < min_interval:
        await track.recv()  # 丢弃
        continue
    frame = await track.recv()
    last_recv = now
```

**优先级**：高（立竿见影降 CPU + 带宽）

### 2.4 MJPEG 推流优化

**问题**：`_mjpeg_stream` 用固定 50ms 轮询 `get_frame()`，空轮询浪费 CPU。

**方案**：用 `asyncio.Event` 或 `threading.Condition` 做事件驱动：
- `_receive_frames` 每存一帧就 `notify`
- `_mjpeg_stream` `wait(timeout=1.0)`，有新帧立即推

**优先级**：低（当前 CPU 占用不是瓶颈）

---

## 三、健壮性优化

### 3.1 异常帧检测

**问题**：如果 WebRTC 解码出错，`frame.to_ndarray()` 可能返回全黑或花屏帧，这些帧会被存入 buffer 并送给 VLM，导致分析结果无意义。

**方案**：
- 检测全黑帧（像素标准差 < 阈值），跳过不存
- 检测花屏帧（相邻帧 PSNR 异常低），跳过
- 记录 `dropped_frames` 计数到 status

**优先级**：中

### 3.2 内存控制

**问题**：`deque(maxlen=24)` 存的是 JPEG bytes，每帧 10-50KB，24 帧 = 240KB-1.2MB。长时间运行不会泄漏，但如果分辨率提升到 1080p，单帧可能 200KB+，24 帧 = 4.8MB。

**方案**：
- maxlen 按字节而非帧数限制（如 `maxlen_bytes=2MB`）
- 或保持帧数但降低到 12（VLM 只取 3 帧，12 帧足够）

**优先级**：低

### 3.3 多路推流支持

**问题**：当前 `PhoneCapture` 是全局单例，只能接一路手机推流。

**方案**：
- 改为 `dict[device_id, PhoneCapture]`，按 `android_device_id` 索引
- 端点改为 `/api/phone/{device_id}/stream`
- 摄像头列表注入多个 `virtual-phone-0`, `virtual-phone-1` ...

**优先级**：低（当前需求只有一路）

---

## 四、监控与可观测性

### 4.1 状态指标补充

当前 `status()` 返回的字段建议补充：

| 字段 | 说明 |
|---|---|
| `dropped_frames` | 解码/转码失败丢弃的帧数 |
| `total_bytes` | 累计接收字节数（带宽估算） |
| `rtt_ms` | ICE 连接的 RTT（`pc.getStats()` 可取） |
| `jitter_ms` | 抖动 |
| `nacks_received` | NACK 重传次数 |
| `connected_since` | 当前连接建立时间（算连接时长） |

### 4.2 日志结构化

当前用 `logger.info` 打字符串，建议改 structured logging：

```python
logger.info("phone_frame_received", extra={
    "device_id": cfg.android_device_id,
    "width": frame.width,
    "height": frame.height,
    "jpeg_bytes": len(jpeg),
    "fps": self._fps,
})
```

便于后续接入 ELK / Loki 做聚合分析。

---

## 五、优先级排序

| 优先级 | 优化项 | 预期收益 |
|---|---|---|
| P0 | ICE restart + connectionstatechange 监听 (1.3) | 连接断开后自动恢复，无需手动点重连 |
| P0 | 帧率限制到 15fps (2.3) | CPU 降 60%，带宽降 75% |
| P1 | TURN server 支持 (1.1) | 跨网络可用性 |
| P1 | 分辨率提升到 720p+ (2.2) | VLM 分析准确率显著提升 |
| P1 | JPEG quality 提到 85 (2.1B) | 文字清晰度提升，带宽增加可控 |
| P2 | 重连指数退避 (1.2) | signaling 不可用时减少无效重试 |
| P2 | 异常帧检测 (3.1) | 避免 VLM 分析黑屏/花屏帧 |
| P2 | 状态指标补充 (4.1) | 可观测性 |
| P3 | 事件驱动 MJPEG (2.4) | 微优化 CPU |
| P3 | 多路推流 (3.3) | 扩展性，当前无需求 |
| P3 | 直接用 H.264 关键帧 (2.1A) | 消除转码损失，实现复杂 |

---

## 六、实施建议

1. **短期（本周）**：P0 两项 + P1 的 JPEG quality，改动小、收益大
2. **中期（下个 sprint）**：P1 的 TURN + 分辨率，需要 Android app 配合
3. **长期**：P2/P3 按需推进

每个优化项实施后，补充对应的单元测试，重点关注：
- 帧率限制的正确性（用 mock track 验证 sleep 间隔）
- 异常帧检测的阈值（用真实黑屏/花屏样本调参）
- 重连退避的时序（用 monkeypatch 加速时间）
