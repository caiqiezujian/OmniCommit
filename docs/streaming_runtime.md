# 流式推理与 Web 服务

本服务接收 16 kHz 单声道新音频。网页默认每轮接收 0.5 秒，开始会议前可以选择预设或输入自定义步长（如 200 ms）。服务使用已导出的本地
Qwen3-Omni 模型输出 DIRECT / WAIT。最后发送文本事件 `<end_of_audio>`，
不会补一个假的未来音频块。语言、说话人标签从训练后的 tokenizer 读取，
不会在推理时添加随机初始化的新 token。

## 音视频文件逐轮诊断

在原来的 GPU 推理环境中，从仓库根目录执行。把输入和模型路径替换为这场会议实际用的文件和合并模型。需要可执行的 FFmpeg，脚本直接加载模型，无需启动网页。

```bash
cd /data/yb/Code/OmniCommit
CUDA_VISIBLE_DEVICES=0 python tools/diagnose_stream.py /data/test/meeting.mp4 \
  --model /data/yb/Code/OmniCommit/models/omnicommit-merged \
  --chunk-ms 1000 \
  --output /data/yb/Code/OmniCommit/outputs/meeting.1000ms.jsonl
```

第二版使用同一条命令，将 `--model` 改成实际的 `second-merged` 目录。若仓库的 `models/omnicommit-merged` 和 `models/second-merged` 中只有一个可用模型，可省略 `--model`。`--chunk-ms` 的单位是毫秒，例如 `200`、`800` 或 `1000`。默认采用启动脚本的 vLLM、rolling 和 normalized 配置。若网页服务用了 HF 或自定义上下文预算，请传入相同的参数。做对照时在网页用文件输入，关闭自动参照，使用相同代码版本且不添加人工提交边界或说话人参照。命令行按文件顺序处理每块，运行时间由推理速度决定。

用 `.jsonl` 输出时，每轮完成即写入，可在另一个终端用 `tail -f` 查看。用 `.json` 则在结束时得到事件数组。Web 日志选「逐轮日志 → 下载完整日志 JSONL」，对齐两边 `type=result` 的 `turn`、`event`、`end_seconds`、`action` 和 `text`。播放器的 `file_wait` 只表示等待文件播放。

在 Bash 中也可以用 `jq` 和 `diff` 找到两边第一个不同的轮次。将下例两个文件名换成实际下载和生成的 JSONL。

```bash
fields='select(.type == "result") | [.turn, .event, .end_seconds, .action, .text]'
diff -u <(jq -c "$fields" web-events.jsonl) <(jq -c "$fields" meeting.1000ms.jsonl)
```

- **模型回复**：`raw_model_text` 保留后端返回的原文，包括 `<|WAIT|>`。解析失败的原文保存在错误事件里。`consecutive_audio_waits` 只统计连续返回 WAIT 的音频块。
- **上下文裁剪**：`history_start_seconds` 是当前模型保留的最早一轮音频起点。发生裁剪时，`trimmed_interval_seconds` 给出本轮移出的区间。默认 rolling 保留近期约 12 秒，并受提交边界约束，只能按完整轮次移出较早历史。
- **预算报错**：错误行中的 `heard_seconds` 是最后成功处理的位置，`attempted_end_seconds` 是尝试处理的位置。`commit_watermark_seconds` 是当前策略使用的边界，`pending_audio_seconds` 是截至最后成功轮次仍待确认的秒数。若下一块会超过 30 秒待确认预算，服务和脚本都在该轮调用后端前停止。此时 `backend_called` 为 `false`，`raw_model_text` 为 `null`，命令退出码为 1，先前轮次保留。上下文时长和模型 token 数另有各自的上限。

如果两边从同一位置持续 WAIT，可以从人工听到的新语种起点截出一段有清晰人声的音频，用同一条命令和新的输出文件单独跑。如果短片能出字，优先检查切换前的历史及训练中的切换示例。如果短片仍持续 WAIT，再与离线 ASR 对照。两条路径共用模型、提示词和会话逻辑，这个实验可以排查网页传输或显示差异，单凭一致的 WAIT 无法确定是训练数据造成的。

### 日德切换的 500 ms 缓存对照

把下面三条推理命令中的输入文件和模型目录换成同一场会议和同一份 checkpoint。每条命令单独启动一个进程，依次运行。示例将精度统一为 BF16。如果原服务使用 FP16，就把三条命令中的 `bfloat16` 都改为 `float16`。两次 vLLM 推理都给多模态预处理缓存分配 1 GiB。

```bash
cd /data/yb/Code/OmniCommit

CUDA_VISIBLE_DEVICES=0 python tools/diagnose_stream.py /data/test/ja-de.mp4 \
  --model /data/yb/Code/OmniCommit/models/omnicommit-merged \
  --backend hf --chunk-ms 500 --dtype bfloat16 \
  --output outputs/ja-de.500ms.hf.jsonl

CUDA_VISIBLE_DEVICES=0 python tools/diagnose_stream.py /data/test/ja-de.mp4 \
  --model /data/yb/Code/OmniCommit/models/omnicommit-merged \
  --backend vllm --chunk-ms 500 --dtype bfloat16 --mm-processor-cache-gb 1 \
  --prefix-caching --output outputs/ja-de.500ms.vllm-on.jsonl

CUDA_VISIBLE_DEVICES=0 python tools/diagnose_stream.py /data/test/ja-de.mp4 \
  --model /data/yb/Code/OmniCommit/models/omnicommit-merged \
  --backend vllm --chunk-ms 500 --dtype bfloat16 --mm-processor-cache-gb 1 \
  --no-prefix-caching --output outputs/ja-de.500ms.vllm-off.jsonl

python tools/compare_stream_traces.py \
  outputs/ja-de.500ms.hf.jsonl \
  outputs/ja-de.500ms.vllm-on.jsonl \
  outputs/ja-de.500ms.vllm-off.jsonl
```

每份日志的 `ready.enable_prefix_caching` 记录引擎初始化后的实际设置：HF 为 `null`，vLLM 默认开启。开关只控制跨请求的 KV 前缀复用，三个实验都保留流式会话的历史音频和先前回复。比较脚本逐对报告最早出现差异的轮次、音频区间和原始回复。优先看两份 vLLM 日志；首次分歧之后，先前输出也已不同，后面的差异可能来自各自累积的历史。HF 同时更换了推理后端，适合作为独立参照。

## 1. 环境与离线真实模型测试

在已用于 FLEX 的推理环境执行；不要用这些命令升级 torch/CUDA。
真实 HF 后端按 Transformers 4.57.1 接口实现，需要已有 torch、transformers、
accelerate。Web 额外依赖单独安装：

```bash
cd /path/to/OmniCommit
python -m pip install -r requirements-data.txt -r requirements-web.txt

python tools/stream_infer.py /data/meeting-short.wav \
  --model /outputs/omnicommit-merged \
  --output /outputs/meeting-short.events.jsonl \
  --chunk-seconds 0.5 --number-format normalized \
  --context-policy verified
```

使用正式导出目录，它必须包含 `omnicommit_tokens.json`。启动时检查全部新增
token 的训练 ID、特殊 token 属性、tokenizer 长度和模型词表行数，另检查
processor 的音频 chat wrapper 与继承的 FLEX/SWIFT 模板一致。不兼容就报错，
不调用 `add_tokens` 或 `resize_token_embeddings` 修补。

`--number-format` 必须与对应训练数据一致。上例的 normalized 仅在训练数据保留
标准数字格式时使用。输出每行是一轮事件，包含完整输出、语言/说话人段、
真实音频起止秒、生成耗时、RTF、上下文秒数、待核验秒数、使用的说话人参照。
录音不足一轮的尾部可以更短；只有编码时补到至少 0.5 秒，时间仍是原音频时间。

遇到生成截断、非法标签、CUDA 错误、上下文超限时写入 `event:error`，返回非零。
不会把异常伪装成 WAIT、重发已经提交的文字或静默丢掉一块音频。

## 2. 长会议的两种上下文策略

`DIRECT` 仅证明模型提交了一段文字，不证明本轮所有声音都已消费。例如模型可能
先发“请联系”，把同一块里的姓名前半部分留到后面。不能见到 DIRECT 就清空待提交音频。

- `--context-policy verified` 是默认正确性基线。只有可信外部对齐器或人工核验提交的
  音频 watermark 才允许裁剪较早历史。近期窗口默认 12 秒，未核验历史默认最多 30 秒。
  超限会明确停止，不能无限累积，也不偷删待提交语音。
- `--context-policy rolling` 用于无人值守的长会议实验。DIRECT 后推测可消费到当前时间
  减去 `--rolling-tail-seconds 2`；仍保留近期 12 秒，WAIT 链不推进这个推测位置。
  这是启发式，长时间暂存的实体有可能跨过保留窗口。每轮记录策略、推测 watermark，
  首次基于推测丢弃音频后 `heuristic_eviction:true`。不能据此宣称完整性有保证。

```bash
python tools/stream_infer.py /data/long-meeting.wav \
  --model /outputs/omnicommit-merged \
  --output /outputs/long-meeting.rolling.events.jsonl \
  --chunk-seconds 1 --number-format normalized \
  --context-policy rolling --recent-seconds 12 \
  --rolling-tail-seconds 2 --max-context-seconds 30 --max-pending-seconds 30
```

即使是 rolling，持续 WAIT 超出预算也不会删除待提交音频继续冒充成功。
音频秒数预算与 token 预算分开检查，默认编码上下文加生成预算不得超过 4096。
需要更多上下文时应先验证模型训练长度；不能只调大 Python 的秒数。

## 3. 说话人缓存与可信控制接口

`StreamSession.register_reference()` 从本会话**已经听到且仍在历史中的精确区间**
截取参考音频；支持人工或已验证的外部声纹/时间对齐组件提供确认，不自动使用最新一块。
需要 `verified:true, clean:true, overlap:false`；单参照默认最多 5 秒，全部参照最多
30 秒。不允许未来区间、未知说话人、已裁掉的区间或自动循环复用编号。

参照音频与带编号的转录放在输入侧，在首个保留的 live user 消息之前明确标成
“只用于身份参照”。历史滚动不删除参照。同一人换语言仍使用原编号。
单个模型 spk 标签不是独立的身份置信度。Web 还提供可选的 Qwen 强制对齐实验性自动参照：它利用模型输出和已听音频挑选候选，但**不验证身份或重叠**，不是人工或外部独立核验结果。默认关闭，详见[会议 Web 服务说明](web_meeting_service_zh.md)。

`StreamSession.verify_commit()` 的语义是：**该时间之前的全部语音都已经提交且没有
剩余未提交的实体或数字**。只找到一个晚出现的词的时间戳不满足这个条件。
开发者可以将真实外部验证器的结果传给这两个 API；当前没有内置自动声纹确认模型。实验性 Qwen 时间对齐不会自动成为可信的提交边界。

CLI 支持按时间调度可信控制 JSONL，用于“无参照/人工正确参照”的因果对照实验：

```jsonl
{"after_seconds":2.0,"op":"reference","speaker":"spk1","language":"zh","text":"大家上午好。","start":0.2,"end":1.8,"verified":true,"clean":true,"overlap":false}
{"after_seconds":2.0,"op":"commit","through_seconds":1.8,"verified":true}
```

上面只展示字段，不是可套用到任意录音的真实对齐。参照身份、转录与区间必须来自
当前录音实际已发生的语音；同一批数据中不同录音的 spk1 不能当成同一个人。

```bash
python tools/stream_infer.py /data/meeting.wav \
  --model /outputs/omnicommit-merged --output /outputs/meeting.verified.events.jsonl \
  --chunk-seconds 0.5 --number-format normalized --context-policy verified \
  --events /data/meeting.verified-controls.jsonl
```

事件按 `after_seconds` 非递减排序，在对应音频处理完成后才应用。超出音频结尾的事件
报错。该 CLI 可信控制接口提供对照验证与外部验证器集成；Web 可选实验性缓存不能作为高质量身份确认。

## 4. 启动 Web

FLEX 风格的启动方式、文件/系统音频、说话人修正和实验性自动参照见
[会议 Web 服务说明](web_meeting_service_zh.md)。服务不要求访问口令。

```bash
bash start_server.sh /outputs/omnicommit-merged
```

默认使用 vLLM、0.5 秒块、normalized、rolling、一个会话，绑定 `0.0.0.0:18889`，直接访问 `http://服务器内网IP:18889`，不使用令牌。文件上传、系统声音和实验性自动参照的具体配置见 [会议 Web 服务说明](web_meeting_service_zh.md)。

网页通过 AudioWorklet 采集麦克风或系统声音为 16 kHz PCM；浏览器如不能提供所需采样率会明确提示。文件上传后服务器先保留源采样率解码，再根据播放进度裁剪并逐块重采样；压缩格式自身解码前瞻及音画同步需实测。
点击“结束并处理尾部”会先发送残余短块，再发送文本 EOS。
服务端同一模型只有一个串行 worker，会话状态、编号、声音缓存分别管理；默认只接受
一个并发会话，多个模型调用不会争抢同一会话的状态。

客户端逐轮等待确认，最多允许约 10 秒积压。推理持续跟不上时明确停止采集并处理
已接收音频；停止后的现场语音不会被记录。这个保护防止无界增长，不能替代达标的硬件速度。
页面显示已处理音频、单轮模型及服务耗时和积压，逐轮事件记录含 RTF 与待核验秒数；支持 TXT、完整 JSON 下载与说话人修正。
“人工核验与说话人参照”面板直接调用上述可信控制接口。修正会清理受影响参照，但不会改写模型已接收的历史回复。

在内网绑定 `0.0.0.0` 后，知道地址的人可以直接使用，不需要输入访问口令。
远程浏览器的麦克风和系统声音需要受信任的 HTTPS 或 localhost 隧道；
可用 HTTPS 反代或 `--ssl-certfile /path/cert.pem --ssl-keyfile /path/key.pem`。

## 5. 性能与验证边界

HF 后端每轮重新编码有限历史与参照，关闭音频生成分支，WAIT token 立即停止本轮生成。
当前没有跨轮 KV cache 或音频特征缓存复用，不能保证 0.5 秒内完成一轮。先测试单会话
RTF 与 p95 耗时。测试 0.8 秒或 1 秒时，用对应步长重铸的数据进行训练和验证。
另提供从 FLEX 适配的可选 vLLM 0.11.1 后端，使用原生 prefix caching、有限的多模态
processor cache，以及按音频内容计算的稳定 UUID。音频计数采用独立预处理器测量；
只在已审查的版本组合中按音频长度缓存计数，并核对生成实际展开后的 prompt 长度。
不会把旧 KV 手工拼接到裁剪后的新对话。参照更新或上下文滚动会改变 prompt，缓存命中
并不保证；processor cache 也不等于永远免除音频编码器计算。

在已有 FLEX vLLM 0.11.1 推理环境中运行，不要向训练环境直接安装/升级 vLLM：

```bash
python tools/serve_web.py --backend vllm --model /outputs/omnicommit-merged \
  --chunk-seconds 0.5 --number-format normalized --context-policy rolling \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.85 \
  --mm-processor-cache-gb 1 --host 127.0.0.1 --port 7860
```

同样可在 `tools/stream_infer.py` 使用 `--backend vllm`。默认 TP=1；模型显存不够时
按已有机器条件明确设置 TP。为了继承 Qwen3-Omni 初始化兼容性，默认保留 1 个 image
槽位，但所有请求仍只有音频；确认本环境无需该兼容后可用 `--vllm-image-slots 0`。
这两种后端都尚未在本地加载真实权重运行；CPU 测试覆盖请求构建与状态，不是 GPU 吞吐验收。

CPU 验证：

```bash
python -m pytest tests/test_streaming.py -q
python tools/serve_web.py --backend replay --host 127.0.0.1 --port 7860
```

replay 默认始终 WAIT，只验收传输、EOS、缓存与资源管理，**没有 ASR 能力**。`--replies`
可读入一个脚本化输出 JSON 数组用于定向测试。真正的语种、长间隔身份恢复、重叠说话、
重复转录、实体完整性和 GPU 延迟必须在真实训练模型上验收。

后端接口依据：[Transformers 4.57.1 Qwen3-Omni](https://huggingface.co/docs/transformers/v4.57.1/model_doc/qwen3_omni_moe)、
[对应 processor 源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_omni_moe/processing_qwen3_omni_moe.py)。
