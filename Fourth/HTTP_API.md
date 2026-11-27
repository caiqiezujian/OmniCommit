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

在 `Fourth/http_config.sh` 中设置开关和静音阈值，再重启 HTTP 服务。开关默认关闭。以下配置按 300 毫秒启用这条规则。

~~~bash
PACKET_PERIOD_TO_COMMA=true
PACKET_PERIOD_THRESHOLD_MS=300
~~~

仅当整包静音时长**严格小于**阈值时，才把符合条件的包末句号改为逗号。例如把阈值改为 `500`，上游传 `tail_silence=0.499` 秒会触发，传 `0.500` 秒保留句号。通过 `GET /health` 的 `packet_period_to_comma` 和 `packet_period_threshold_ms` 可核对生效配置。

每个 JSON 音频包可用 `tail_silence` 传入调用方统计的整包累计静音时长，**单位为秒**，支持非负浮点数。例如 `"tail_silence": 0.3` 表示 300 毫秒。已有调用方可继续用 `silence_ms` 或 `trailing_silence_ms` 传毫秒数。每包只传其中一个字段，只发音频时保留模型原标点。`GET /v1/meetings/{meeting_id}/result` 中的 `silence_ms` 以及调试日志统一使用毫秒。

~~~json
{"data":"<音频的 Base64 字符串>","tail_silence":0.2}
~~~

例如按上面的 300 毫秒配置，一包 5 秒，模型按 1 秒解码，整包拼出的正文是“我听到了。我知道。”。`tail_silence` 为 `0.2` 时，返回 `{"src":"我听到了。我知道，"}`。服务只在整包结束时检查一次，正文中间的句号照常保留。如果最后一个音频轮是 `WAIT`，检查本包最后一次提交的正文。整包都在等待时返回空字符串。下一个请求使用修改后的历史。

逗号根据结尾文字和标点选择，例如中文用“，”，可识别的日文用“、”，英文用“,”，阿拉伯文用“،”。问号、叹号和省略号保持原样。使用 1 秒整数倍的音频包可让包末与推理轮边界对齐。对于留有不足一步音频的包，服务暂存尾部；与下包拼接时使用新包读数，直接结束时使用暂存读数。

### Python 对接示例

~~~python
import requests

meeting = "http://服务器内网IP:18892/v1/meetings/meeting-001"

# 每来一包就调用一次，audio_base64 是上游已有的字符串。
def transcribe(audio_base64, tail_silence=None):
    packet = {"data": audio_base64}
    if tail_silence is not None:
        packet["tail_silence"] = tail_silence  # 秒
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

服务默认将全局请求日志追加到仓库下的 `Fourth/logs/http-requests.jsonl`。可在启动命令中加 `--request-log /data/logs/fourth-http.jsonl` 指定位置。启动时会打印实际路径，目录无法写入时改写到标准错误。每条记录使用 UTC 时间，文件达到 20 MiB 后轮转，最多保留 5 个历史文件。用这份日志查请求号、状态码、耗时和转录字符数。要看每轮正文，打开下文的会议日志。

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

### 按会议号查看逐轮转录

默认 HTTP 配置使用 `HTTP_EVENT_LOG_MODE=readable`。每次会话写一份 `events.jsonl`，例如

~~~text
Fourth/logs/http-debug/6424126945/2026-11-27_14-32-05_8e10cfcff830/events.jsonl
~~~

第一层是调用方的会议号，第二层是北京时间的开始时间和本次会话的短 ID。同一会议号再次开会时会生成新的场次目录。建会前就被拒绝的请求也按会议号记录。含路径分隔符等特殊字符或过长的会议号会以可读前缀加摘要保存，日志中有完整会议号。在仓库根目录用以下命令可以跟随该会议号最近写入的日志。

~~~bash
meeting=6424126945
file=$(ls -t "Fourth/logs/http-debug/$meeting"/*/events.jsonl | head -n 1)
tail -n 20 -f "$file"
~~~

| 事件 | 看什么 |
|---|---|
| 收包 | 完整收到并校验音频后、进入推理队列前写入。录音区间以秒计，上游的整包静音换算成毫秒，`null` 表示未传。 |
| 逐轮 | 每完成一轮立即写入轮号、新增音频区间、动作和耗时。`模型正文` 是该轮提交、尚未做包末标点修正的可展示文字，可以包含前几轮暂存的内容。`WAIT` 表示这一轮没有提交文字。 |
| 返回 | `HTTP` 为服务端响应码，成功时 `src` 是服务写入响应的字段值。若包末改了标点，同一行给出目标轮号、静音测量和阈值。遇到暂存、解码缺口或失败也在这一行注明。 |
| 拒绝 | 写明状态码和具体原因。例如同一会议的请求重叠会记 409、占用请求号以及对方仍在上传还是转录。 |
| 结束请求、收尾轮 | 对应 `/end`。有暂存音频时先补跑音频轮，再记录收尾轮和最终返回。 |

例如第 7 包的末轮是 `WAIT` 时，可以依次看到第 19 轮提交“我觉得”、第 20 轮提交“可以。”、第 21 轮等待。若 200 ms 触发 300 ms 的句号规则，返回行同时写出 `"src":"我觉得可以，"` 和第 20 轮的修改原因。请求号对应响应头 `X-Request-ID`，可用来对照全局 `http-requests.jsonl`。包号按本场收到的音频请求依次编号，包含被拒绝的请求。

需要查看原始模型回复、完整 prompt、缓存或重试时，在 `Fourth/http_config.sh` 设置 `HTTP_EVENT_LOG_MODE=detailed` 并重启。同一场次目录会额外写入 `trace.jsonl`，沿用原始事件名和 UTC 时间。此前的详细日志仍可在原 SHA-256 目录中读取。两种模式都只写文本。用 `GET /health` 的 `event_log_mode` 可查看当前模式。

## 用真实音频测延迟

~~~bash
python Fourth/tools/bench_http.py /data/test/meeting.16k.wav \
  --url http://127.0.0.1:18892 --wire-format json-f32 \
  --concurrency 1 2 4 8 --seconds 16 --slo-ms 1000
~~~

该示例需要至少 240 秒的 16 kHz 单声道 PCM16 WAV。每个模拟会议取不同的片段，按音频就绪时间发包，测量调用方实际拿到 `src` 的耗时。空字符串也算一次成功的 HTTP 回复，压测会单独统计空结果。
