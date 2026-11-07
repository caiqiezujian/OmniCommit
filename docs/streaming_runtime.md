# 流式推理与 Web 服务

本服务接收 16 kHz 单声道新音频，每轮 0.5 秒或 1 秒，使用已导出的本地
Qwen3-Omni 模型输出 DIRECT / WAIT。最后发送文本事件 `<end_of_audio>`，
不会补一个假的未来音频块。语言、说话人标签从训练后的 tokenizer 读取，
不会在推理时添加随机初始化的新 token。

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
模型输出的一个 spk 标签不是独立的身份置信度，本实现不凭这一个结果自动污染参照。

`StreamSession.verify_commit()` 的语义是：**该时间之前的全部语音都已经提交且没有
剩余未提交的实体或数字**。只找到一个晚出现的词的时间戳不满足这个条件。
开发者可以将真实外部验证器的结果传给这两个 API；当前没有内置自动声纹确认模型。

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
报错。该接口提供对照验证与外部验证器集成，不代表已有自动高质量缓存选择器。

## 4. 启动 Web

```bash
python tools/serve_web.py --model /outputs/omnicommit-merged \
  --chunk-seconds 0.5 --number-format normalized \
  --context-policy rolling --host 127.0.0.1 --port 7860 --max-sessions 1
```

浏览器访问 `http://127.0.0.1:7860`。远程服务器最简单的方式是本地 SSH 端口转发：

```bash
ssh -L 7860:127.0.0.1:7860 USER@SERVER
```

网页通过 AudioWorklet 采集 16 kHz PCM；浏览器如不能提供所需采样率会明确提示。
点击“结束并提交尾部”会先发送残余短块，再发送文本 EOS。
服务端同一模型只有一个串行 worker，会话状态、编号、声音缓存分别管理；默认只接受
一个并发会话，多个模型调用不会争抢同一会话的状态。

客户端逐轮等待确认，最多允许约 10 秒积压。推理持续跟不上时明确停止采集并处理
已接收音频；停止后的现场语音不会被记录。这个保护防止无界增长，不能替代达标的硬件速度。
页面显示已处理音频位置、单轮耗时、RTF、积压和待核验秒数，支持下载事件 JSONL。
“人工核验与说话人参照”面板直接调用上述可信控制接口。

如必须绑定 `0.0.0.0`，设置强随机访问口令到 `OMNICOMMIT_WEB_TOKEN` 环境变量，
并用 HTTPS 反代或 `--ssl-certfile /path/cert.pem --ssl-keyfile /path/key.pem`。
网页口令只在首次 WebSocket 消息提交，不放在 URL。远程普通 HTTP 无法正常申请麦克风。

## 5. 性能与验证边界

HF 后端每轮重新编码有限历史与参照，关闭音频生成分支，WAIT token 立即停止本轮生成。
当前没有跨轮 KV cache 或音频特征缓存复用，不能保证 0.5 秒内完成一轮。先测试单会话
RTF 与 p95 耗时；若 0.5 秒不足，测 1 秒档并用对应 1 秒重构数据训练/验证。
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
