# Fourth 会议转录 HTTP 接口

每次发送一包音频，服务处理完就返回 **HTTP 200** 和一个字符串字段 `src`。调用方按顺序把每包新提交的纯正文追加到会议记录，空字符串可以直接略过。

Fourth 在模型历史和逐轮日志的 `text` 中保留 `<|SPK_CHANGE|>`。需要换人位置的 HTTP 客户端可在当前请求完成后、发送下一包之前读取 `GET /v1/meetings/{meeting_id}/result`，其中 `turns[].transcript_parts` 给出每段的 `text` 和布尔值 `speaker_change`。WebSocket 的 `result` 事件也包含 `transcript_parts`，网页据此在换人处另起一段。

## 上传音频

`POST /v1/meetings/{meeting_id}/chunks`，请求头为 `Content-Type: application/json`。

~~~json
{"data":"<音频的 Base64 字符串>"}
~~~

`data` 是 16 kHz、单声道、小端 float32 原始波形的标准 Base64 编码，幅值归一化到 `[-1, 1]`。1 秒是 16000 个样本。调用方直接发送已有的 Base64 字符串即可。

假设一包有 3 秒。第 1 秒提交“请联系”，第 2 秒等待，第 3 秒提交“张伟。”。调用方只会收到一次响应：

~~~json
{"src":"请联系张伟。"}
~~~

整包没有新增正文时返回 `{"src":""}`。同一会议使用同一 ID，等到这一包返回后再发送下一包。一次请求包含多秒音频时，服务依序完成各个模型步长再返回。同步耗时随音频长度、GPU 负载和生成的字数变化。

### 按整包静音时长将末尾句号换成逗号

在 `Fourth/http_config.sh` 设置 `PACKET_PERIOD_TO_COMMA=true` 并重启 HTTP 服务。默认值为 `false`。可通过 `/health` 的 `packet_period_to_comma` 查看运行状态。

每个 JSON 音频包可传一次 `silence_ms`，表示**本次上传音频里所有静音加起来的毫秒数**。取有限的非负数。原对接字段名 `trailing_silence_ms` 也受支持，两个字段任选其一。只发音频时，服务保留模型原标点。

~~~json
{"data":"<音频的 Base64 字符串>","silence_ms":200}
~~~

例如一包 5 秒，模型按 1 秒解码，整包拼出的正文是“我听到了。我知道。”。开关开启且整包静音为 `200` 时，返回 `{"src":"我听到了。我知道，"}`。即使第 3～4 秒有停顿，服务也只在整包结束时检查一次；正文中间的句号照常保留。整包静音达到 `300` 毫秒时保留末尾句号。如果最后一个音频轮是 `WAIT`，检查本包最后一次提交的正文。整包都在等待时返回空字符串。下一个请求使用修改后的历史。

逗号根据结尾文字和标点选择，例如中文用“，”，可识别的日文用“、”，英文用“,”，阿拉伯文用“،”。问号、叹号和省略号保持原样。使用 1 秒整数倍的音频包可让包末与推理轮边界对齐。对于留有不足一步音频的包，服务暂存尾部；与下包拼接时使用新包读数，直接结束时使用暂存读数。

### Python 对接示例

~~~python
import requests

meeting = "http://服务器内网IP:18892/v1/meetings/meeting-001"

# 每来一包就调用一次，audio_base64 是上游已有的字符串。
def transcribe(audio_base64, silence_ms=None):
    packet = {"data": audio_base64}
    if silence_ms is not None:
        packet["silence_ms"] = silence_ms
    response = requests.post(meeting + "/chunks", json=packet, timeout=60)
    response.raise_for_status()
    return response.json()["src"]

# 会议结束时调用一次，取走最后的新增文字并释放会议。
def finish():
    response = requests.post(meeting + "/end", timeout=60)
    response.raise_for_status()
    return response.json()["src"]
~~~

### 在服务器上测试一包 WAV

用 16 kHz、单声道、PCM16 WAV 可以直接检查服务。接口也支持裸 PCM16 小端样本，请求头为 `Content-Type: audio/pcm`。

~~~bash
curl -fsS -H 'Content-Type: audio/wav' --data-binary @chunk.wav \
  http://127.0.0.1:18892/v1/meetings/demo-001/chunks
curl -fsS -X POST http://127.0.0.1:18892/v1/meetings/demo-001/end
~~~

## 部署

在 `Flex_web` 容器中编辑 [http_config.sh](http_config.sh)。`ASR_MODEL` 填训练后合并的 HF 模型目录，`CUDA_VISIBLE_DEVICES` 填服务使用的 GPU。`HTTP_ARGS` 中的 `--port` 填调用方访问的端口。仓库配置使用 GPU 6、checkpoint 2721、端口 18892 和 1000 ms 步长。保存后从仓库根目录启动，修改配置后重启生效。

~~~bash
cd /data/yb/Code/OmniCommit
bash Fourth/start_http.sh
~~~

~~~bash
curl -fsS http://127.0.0.1:18892/health
~~~

服务就绪后，健康检查中的 `action_protocol` 应为 `speaker`。显式运行 `bash Fourth/start_http.sh MODEL [参数...]` 时，使用调用方的 GPU 环境和命令行参数。网页的启动参数仍在 `Fourth/web_config.sh`。

HTTP 默认按 1 秒步长解码，单包最长 8 秒。如果对方每包最长 30 秒，将配置中的 `--max-packet-seconds` 改为 `30`。默认每轮输出上限 512 token，总上下文 4352 token，音频窗口 60 秒，每次裁剪 50 秒。默认最多保留 16 场会议、同时接收或处理 8 个请求、每批至多 4 路。前端或其他服务已在使用 GPU 时，应按 `nvidia-smi` 的实际空闲显存选卡。

音频格式错误或超长、同一会议重叠发包、容量已满、引擎失败，都会返回相应的非 200 状态码。网络超时时，已经接收的包可能仍在处理，重复 POST 会有重复转录的风险。部署方排查截断时可以查看服务日志或最近一包的诊断记录 `GET /v1/meetings/{meeting_id}/result`。正常对接只用两个 POST 接口。

### 定位某次 HTTP 409

新版服务会把拒绝原因和推理异常同时打印到启动终端的标准错误。即使用原来的 `> logs/service.log 2>&1` 方式启动，也能在这个文件中搜索 `request_rejected`。例如同一会议的上一包还在推理时，日志会给出 `reason=in_flight`、`blocking_phase=processing`、阻塞请求的 ID 和已耗时的毫秒数。

调用方遇到 409 时，记录会议 ID、响应头 `X-Request-ID` 和响应正文中的 `detail`。Python 调用方可以在 `raise_for_status()` 之前读取这些信息。

~~~python
if response.status_code == 409:
    print(response.headers.get("X-Request-ID"), response.text)
~~~

服务默认将请求日志按 JSONL 追加到仓库下的 `Fourth/logs/http-requests.jsonl`。可在启动命令中加 `--request-log /data/logs/fourth-http.jsonl` 指定位置。启动时会打印实际路径，目录无法写入时改写到标准错误。每条记录使用 UTC 时间，文件达到 20 MiB 后轮转，最多保留 5 个历史文件。日志中的转录统计使用字符数，会议音频和转录正文留在原有的数据通路中。

在仓库根目录执行以下命令，按会议或请求查询。较早的记录也可能在同目录的 `.1` 至 `.5` 文件中。

~~~bash
jq -c --arg m 'meeting-001' 'select(.meeting_id == $m)' Fourth/logs/http-requests.jsonl
jq -c --arg r '替换为请求ID' \
  'select(.request_id == $r or .blocking_request_id == $r or .last_request_id == $r)' \
  Fourth/logs/http-requests.jsonl
~~~

- `request_rejected` 且 `reason=in_flight` 表示本次请求被拒绝。用 `blocking_request_id` 查上一条请求。`blocking_operation` 标识正在执行的操作，`blocking_phase=upload` 表示还在收包，`processing` 表示正在处理。每个会议串行发包，最后一包返回后再调用 `/end`。
- `reason=ended`、`failed`、`expired` 分别表示会议已经结束、处理失败、空闲过期。用 `last_request_id` 回查该会议最后一次请求，新会议分配新的 ID。空闲超时默认 600 秒，实际值见 `/health` 中的 `idle_timeout_seconds`。
- 成功请求的 `request_finished` 给出耗时 `duration_ms` 和新增文本的字符数 `src_chars`。`src_chars=0` 是有效结果。如果前一包在客户端超时，服务器可能仍在处理，先查该会议的日志或 `GET /v1/meetings/{meeting_id}/result`。该接口返回最近一包的状态，自动重发同一包可能重复转录。

### 通过 events.jsonl 逐轮检查会议转录

默认 HTTP 配置已启用 `--debug-dir logs/http-debug`，每场会议写一份 `Fourth/logs/http-debug/<会议ID的SHA-256>/events.jsonl`。它保存逐轮转录正文、模型输入上下文、HTTP 响应和错误详情，音频输入以采样区间和请求体摘要记录。需要调整目录时，修改 `http_config.sh` 中的 `--debug-dir` 再重启。

每个会议使用单独目录，目录名是会议 ID 的完整 SHA-256。目录中的 `events.jsonl` 按事件记录 UTC 时间、会议 ID 和请求 ID。第一包的请求 ID 同时标识这一次会话，重复使用同一会议 ID 时，可按 `session_started` 找到新的起点。

| 要检查的问题 | 查看哪些记录 |
|---|---|
| 是否收到了同一包数据？ | `request_received` 有请求类型和声明长度。对于完整收到且在大小限制内的受支持请求体，`request_body` 记录 `content_type`、实际字节数 `body_bytes` 和 `body_sha256`。比较两次请求的摘要即可核对请求体字节是否一致。 |
| 这一轮到底听到了哪段？ | `packet_audio` 和 `turn_started` 都给出整场会议的 `start_sample`、`end_sample`，区间左闭右开。两个区间相交的部分就是这包贡献给这一轮的音频。除以 16000 可换算成秒。 |
| 模型实际说了什么？ | `model_call` 记录交给后端的消息历史、各音频输入的采样点数和解码前缀。`model_reply.raw_text` 保存后端交回的原始字符串，包括动作 token。初次解码、截断重试、补识别按 `kind`、`call_id` 区分。后端抛异常时查 `model_error`。 |
| 服务做了什么处理，最终回了什么？ | `turn_finished.turn_result` 有动作、显示正文、时间区间、耗时及该轮产生的恢复或截断信息。`response` 记录服务生成的响应码和完整响应体。客户端断线时，此记录仍可用于核对服务端结果。 |

在仓库根目录执行以下命令，可以查看某场会议每轮模型的原话，再按请求检查返回的 `src`。

~~~bash
meeting='meeting-001'
meeting_hash=$(printf '%s' "$meeting" | sha256sum | cut -d ' ' -f1)
trace="Fourth/logs/http-debug/$meeting_hash/events.jsonl"
jq -c 'select(.event=="model_reply") | {request_id,turn,kind,raw_text}' "$trace"
jq -c 'select(.event=="response") | {request_id,status_code,body}' "$trace"
~~~

若请求在读包前因 409 被拒绝，按同一请求 ID 查看 `request_rejected` 和 `response` 即可定位原因。升级后沿用原启动命令并重启服务即可切换到仅保存事件的格式。旧版已生成的 `.wav` 和 `.body` 文件由部署方按需清理。

## 用真实音频测延迟

~~~bash
python Fourth/tools/bench_http.py /data/test/meeting.16k.wav \
  --url http://127.0.0.1:18892 --wire-format json-f32 \
  --concurrency 1 2 4 8 --seconds 16 --slo-ms 1000
~~~

该示例需要至少 240 秒的 16 kHz 单声道 PCM16 WAV。每个模拟会议取不同的片段，按音频就绪时间发包，测量调用方实际拿到 `src` 的耗时。空字符串也算一次成功的 HTTP 回复，压测会单独统计空结果。
