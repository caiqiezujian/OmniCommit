# Third 会议转录 HTTP 接口

每次发送一包音频，服务处理完就返回 **HTTP 200** 和一个字符串字段 `src`。它只包含本包新提交的转录。调用方按顺序把每次的 `src` 追加到会议记录，空字符串可以直接略过。

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

### Python 对接示例

~~~python
import requests

meeting = "http://服务器内网IP:18892/v1/meetings/meeting-001"

# 每来一包就调用一次，audio_base64 是上游已有的字符串。
def transcribe(audio_base64):
    response = requests.post(meeting + "/chunks", json={"data": audio_base64}, timeout=60)
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

在现有 Third 的 vLLM 0.11.1 推理环境中启动。将模型路径换成实际合并的 Third HF 模型目录，选择一张显存足够的 GPU。

~~~bash
cd /data/yb/Code/OmniCommit
CUDA_VISIBLE_DEVICES=0 bash Third/start_http.sh \
  /data/yb/Code/OmniCommit/models/third-explicit-merged \
  --port 18892 --chunk-ms 1000
~~~

~~~bash
curl -fsS http://127.0.0.1:18892/health
~~~

HTTP 默认按 1 秒步长解码，单包最长 8 秒。如果对方每包最长 30 秒，启动时加 `--max-packet-seconds 30`。默认每轮输出上限 512 token，总上下文 4352 token，音频窗口 60 秒，每次裁剪 50 秒。默认最多保留 16 场会议、同时接收或处理 8 个请求、每批至多 4 路。前端或其他服务已在使用 GPU 时，应按 `nvidia-smi` 的实际空闲显存选卡。

音频格式错误或超长、同一会议重叠发包、容量已满、引擎失败，都会返回相应的非 200 状态码。网络超时时，已经接收的包可能仍在处理，重复 POST 会有重复转录的风险。部署方排查截断时可以查看服务日志或最近一包的诊断记录 `GET /v1/meetings/{meeting_id}/result`。正常对接只用两个 POST 接口。

## 用真实音频测延迟

~~~bash
python Third/tools/bench_http.py /data/test/meeting.16k.wav \
  --url http://127.0.0.1:18892 --wire-format json-f32 \
  --concurrency 1 2 4 8 --seconds 16 --slo-ms 1000
~~~

该示例需要至少 240 秒的 16 kHz 单声道 PCM16 WAV。每个模拟会议取不同的片段，按音频就绪时间发包，测量调用方实际拿到 `src` 的耗时。空字符串也算一次成功的 HTTP 回复，压测会单独统计空结果。
