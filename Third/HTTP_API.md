# Third 会议转录 HTTP API

每个会议使用独立的 `meeting_id`。调用方按录音顺序上传短音频，同一会议等上一包返回最终结果再发下一包。不同会议可以同时发送。会议 ID 放在 URL 中，请求体直接放音频字节。接口面向内网，无需令牌或客户端序号。

## 1. 启动

在现有 Third 推理环境使用 **vLLM 0.11.1** 和已合并的 Third HF 模型。下面按 1 秒训练步长举例。800 毫秒训练的模型改为 `--chunk-ms 800`。HTTP 服务与 18891 网页各自加载一份模型，单卡压测时请把这张 GPU 单独分配给 HTTP 进程。

~~~bash
cd /data/yb/Code/OmniCommit/Third
python -m pip install -r requirements-web.txt
CUDA_VISIBLE_DEVICES=0 bash start_http.sh \
  /data/yb/Code/OmniCommit/models/third-explicit-merged \
  --port 18892 --chunk-ms 1000 \
  --window-seconds 60 --discard-seconds 50 \
  --max-meetings 16 --max-in-flight 8 --batch-size 4 \
  --response-timeout-ms 900
~~~

~~~bash
curl -fsS http://127.0.0.1:18892/health
~~~

看到 `"ready":true` 和 `"service":"third_http"` 即可接入。同一端口的 `/openapi.json` 提供接口定义。一个进程持有会议上下文和一份 GPU 引擎。模型使用导出时的 `explicit` 或 `plain` 协议，窗口超过 60 秒时按 50 秒裁剪。HTTP 默认走常规解码。若需和网页比较连续有声 WAIT 的补识别，显式模型启动时可加 `--recovery`，并按该配置重新压测。

| 参数 | 作用 |
| --- | --- |
| `--max-meetings 16` | 同时保留 16 场会议的上下文，包含暂时没有新音频的会议。 |
| `--max-in-flight 8` | 同时接受最多 8 个上传或处理中的会议操作，容量满时返回 429。 |
| `--batch-size 4`、`--batch-wait-ms 2` | vLLM 一次最多为 4 个独立请求生成，每批从首个请求排队开始，最多等 2 毫秒凑批。只有一路时也可立即或在等待期结束后处理。 |
| `--chunk-ms` | 默认 800。服务会跨请求累积不足一轮的音频，再按这个长度送模型。 |
| `--max-packet-seconds 8` | 单个上传包的最长音频。要逐包获取一轮模型结果，建议每包恰好发送一个模型步长。 |
| `--upload-timeout-seconds 10` | 接收单包请求体的时限。超时释放该会议的上传名额，返回 408，可以重新发该包。 |
| `--response-timeout-ms 900` | 从服务端收到完整请求体开始计算的同步等待预算。超过预算就返回 202，GPU 继续完成该包。 |
| `--idle-timeout-seconds 600` | 没有在处理的会议连续闲置达到该时长后释放。会议状态存在内存，重启进程会清空。 |

## 2. 传一包、取结果、结束

假设 `chunk.wav` 包含会议最初的 1 秒，格式为单声道、16 kHz、PCM16 无压缩 WAV。在服务器上可以直接测试：

~~~bash
curl -i -X POST http://127.0.0.1:18892/v1/meetings/demo-001/chunks \
  -H 'Content-Type: audio/wav' --data-binary @chunk.wav
~~~

如果上游提供裸 PCM，将请求头改为 `Content-Type: audio/pcm`，请求体填写有符号 16 位小端、单声道、16 kHz 的样本。`application/octet-stream` 也按相同格式处理。1 秒裸 PCM 正好 32000 字节。每个 WAV 包须自带 WAV 头。

| 响应 | 调用方处理 |
| --- | --- |
| `200`，`status="ok"`，`action="DIRECT"` | 将 `text` 追加到该会议的转录中。只包含本包新提交的正文，保留原始空格。 |
| `200`，`status="ok"`，`action="WAIT"` | 本次模型确实返回了 WAIT，正文为空。可以发送下一包。 |
| `200`，`status="buffered"` | 已收音频尚不足一次模型步长，`action=null`。继续发下一包，剩余秒数见 `buffered_seconds`。 |
| `202`，`status="processing"` | 已接受这包，模型结果尚未就绪。用 GET 查询同一响应中的 `result_url`，拿到最终结果后再发下一包。 |
| `409` | 检查同一会议是否已有未取回的操作，或 ID 已结束。用 GET 读取其状态。 |
| `429` | 这次上传未被接收，调用方可稍后重新发送该包。 |
| `400 / 413 / 415` | 分别检查数据完整性、包时长/体积、音频格式。 |
| `500`，`status="error"` | 该会议本次处理失败。`turns` 和 `text` 保留同一包内已成功的轮次，`unprocessed_seconds` 给出尚未完成的音频量。该会议结束。 |

例如下面是一条超时响应，HTTP 状态为 202：

~~~json
{"meeting_id":"demo-001","status":"processing","action":null,"result_url":"/v1/meetings/demo-001/result"}
~~~

~~~bash
curl -i http://127.0.0.1:18892/v1/meetings/demo-001/result
~~~

GET 在处理期间返回 202。完成后可读到最近一个操作的最终结果。一个请求覆盖多个模型步长时，`turns` 保留每轮的 `action`、`display_text` 及音频时间，顶层 `text` 按顺序拼接新增正文，`heard_seconds` 是会议已经过模型处理的累计秒数。一次 POST 响应在网络中丢失时，先查结果；本协议按顺序发送，没有逐包重发去重能力。

最后一包发送并取回后结束会议，服务处理不足一步的尾音，再执行模型的结束轮。此接口同样可能返回 202。终态为 `status="ended"`，需要取走其中新增的 `text`。结束或失败的最近结果在内存中短暂保留，建议每场新会议使用新的 ID。

~~~bash
curl -i -X POST http://127.0.0.1:18892/v1/meetings/demo-001/end
~~~

## 3. 在 A800 上测一秒内能返回多少路实际结果

准备一段多说话、含正常停顿的 **16 kHz 单声道 PCM16 WAV**。下面在客户端按真实音频节奏发包，依次模拟 1、2、4、8 场会议，并为每场截取不同的录音区间。每场 16 秒，整段 WAV 至少需要 240 秒。省略 `--seconds` 会根据可用时长自动分配，每场最多 30 秒。默认从 `/health` 读取服务的模型步长。

~~~bash
cd /data/yb/Code/OmniCommit/Third
python tools/bench_http.py /data/test/meeting.16k.wav \
  --url http://127.0.0.1:18892 --concurrency 1 2 4 8 \
  --seconds 16 --slo-ms 1000 > a800-concurrency.jsonl
cat a800-concurrency.jsonl
~~~

- `timely_percent_of_planned` 是全部计划发送的包里，首次 POST 就在预算内拿到真实 `DIRECT` 或 `WAIT` 的比例。要每包 1 秒内拿到推理判断，应检查该比例是否达到 100%，并核对 `first_post_202`、`unsent`、`errors` 及 `cleanup_errors`。模型是否确实出字还可分别看 `direct`、`wait`。
- `final_model_result_ms.p95/p99` 从客户端将请求体写入套接字计到读完整个模型结果，包含 202 后的查询时间。`audio_ready_to_model_result_ms.p99` 还包含上一个请求过慢导致的积压。`model_results_late_from_audio_ready` 直接统计这类端到端超时。建议在实际调用方机器上测，也让网络路径与生产一致。
- `/health` 中的 `batch_stats.largest_batch` 是启动以来一次提交给模型的最大请求数，可检查服务是否真正合批。稳定档位还应跨窗口验证，例如 4 路、每路 120 秒需要至少 480 秒音频，使用 `--concurrency 4 --seconds 120` 复测。

上述 16 场常驻、8 个在途、4 个合批都是可调整的配置。A800 能满足一秒目标的并发数取决于权重、上下文长度、输出文本长度、预热及负载，需要读取实际压测结果。
