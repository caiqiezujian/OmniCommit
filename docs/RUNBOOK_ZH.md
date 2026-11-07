# OmniCommit：从文本标注到流式 Web 的执行手册

这份文档对应本仓库当前的 **文本教师一次调用路线**。原始输入是 `audios + messages`，assistant 中含完整 ASR、语种和说话人标签；最终使用真实音频的 0.5 秒或 1 秒增量轮次训练与推理。

下面是 Linux/云道服务器命令。本地代码在 `D:\ICLR\OmniCommit`；服务器上的仓库、数据、HF 基座、MCore 基座和输出路径需要替换成实际挂载路径。**HF 原始基座与 MCore 训练基座是两个用途不同的目录。**

```text
原始 JSONL + 音频
  → annotate_asr.py：一次文本 LLM 调用，细切分 + 初步 WAIT/DIRECT
  → force_align_asr.py：已有正文与真实音频强制对齐（或导入外部时间戳）
  → align_asr.py：将字/词时间对应到完整语义单元
  → recast_asr.py：按 0.5/1 秒音频网格重构动作 + 明确 EOS
  → prepare_sft.py：FLEX 风格 audios/messages + token 清单
  → prepare_windows.py：长样本安全拆窗 + 因果说话人参照（按需）
  → 结构/音频检查 + 真实模型模板长度检查
  → train_sft.sh：在训练前注册 token 并训练词表层
  → merge_lora.sh：合并、保留 tokenizer 和训练过的词表权重
  → stream_infer.py：离线音频流式验收
  → serve_web.py：麦克风 Web 流式服务
```

## 0. 先确认环境和文件

项目使用三个独立环境：轻量数据工具环境、强制对齐环境、原有 FLEX 的训练/推理环境。不要把新的对齐依赖安装到现有 `omni_ms` 训练容器里，也不要用这里的 CPU requirements 替换已有 CUDA/torch。

```bash
cd /workspace/OmniCommit

# 以下变量每次进入一个新 shell/容器都要重新设置。
export WORK=/data/OmniCommit/run-001
export ANNOTATED=/data/annotations/shard_stream_output.jsonl
export HF_BASE=/models/qwen3_omni_original_hf
export MCORE_BASE=/models/qwen3_omni_mcore_3.12.1
export TRAIN_OUT=/outputs/OmniCommit/sft-001
export MERGED=/models/omnicommit-sft-001
export DATA_ENV=/data/envs/omnicommit-data
mkdir -p "$WORK"

# 在轻量工具环境执行。
python3.11 -m venv "$DATA_ENV"
source "$DATA_ENV/bin/activate"
python -m pip install -r requirements-dev.txt -r requirements-eval.txt
python -m pytest -q

# 不依赖模型的贯通测试；输出是合成音调，禁止用于正式训练。
python tools/run_text_pipeline_demo.py \
  --output-dir outputs/text-demo-500ms --chunk-seconds 0.5
python tools/run_text_pipeline_demo.py \
  --output-dir outputs/text-demo-1000ms --chunk-seconds 1.0
```

这些测试只证明程序接口与状态管理。它们不证明真实识别、声纹一致性、GPU 训练、延迟或模型导出已经验收。

## 1. 云道一次调用：文本切分 + 初步标签

将 `Data-Scripts/annotate_asr.py` 作为单独脚本上传。十二套 prompt 已内嵌，不需要额外上传 prompt 文件；旁边的 txt 是便于审查的同内容副本。

```bash
# 在原云道任务挂载和模型服务已经就绪的环境执行。
python /path/to/annotate_asr.py
```

保留原有七个 CLI 参数以及默认挂载目录；模型请求仍是 `dsv4`、`temperature=0.1`、`top_p=0.8`、`max_tokens=2048`、`timeout=300`、并发 32。每条仅调用一次，不增加隐藏修复调用。输入目录递归读取 JSONL；输出保留相对目录及 `_stream_output` 文件名后缀。

语种按音频路径中精确的 `data/<语种>/` 目录路由；无匹配时回退首个正文语种标签。支持 `ar/ja/ko/it/fr/pt/es/ru/th/en/zh/de`。输入中相同归属的重复标签会去掉，首次和真实变化处保留完整 `<lang><spk>`；原始 `messages` 字段不会被改写。

成功结果仍写入原记录的 **字符串字段** `output`：

```json
{"segments":[{"source_text":"<zh><spk1>请联系","action":"DIRECT"},{"source_text":"张伟","action":"WAIT"},{"source_text":"<zh><spk1>确认方案。","action":"DIRECT"}],"flush":false,"error":null}
```

**输入去重，输出每个 DIRECT 带完整前缀。** 同一归属也重复写；重复前缀不触发 WAIT。正文及真实归属逐项校验，允许输出补充这些正确的重复前缀。模型漏写冗余前缀时，程序在验证通过后按继承状态补齐，不再调用大模型。这里 `source_text` 只保存本片源内容；最终 SFT 的 DIRECT 才合并此前 WAIT 暂存的内容。

错误、截断、正文不守恒或解析失败写 `output:""`，继续下一条。输出文件已有的成功行和失败行都会跳过，因此**换 prompt 后要使用新输出目录**；它不会自动重新处理旧版输出。没有备份步骤。

若旧结果的正文、切分和动作已经正确，只是后续 DIRECT 缺少重复前缀，不需要重新调用模型：

```bash
python Data-Scripts/repair_direct_prefixes.py /data/old.annotations.jsonl \
  --output /data/full-prefix.annotations.jsonl
```

这个离线工具校验原文和真实归属后补前缀，保留其余字段与切分动作；旧空结果仍为空，坏行写空并继续。

## 2. 得到真实字/词时间戳

如果已有时间戳，直接跳至第 3 步。必须是音频对齐得到的时间；不能按字数均分音频，也不能把 LLM 文本切分长度当成 0.5 秒。

### 2A. Qwen3 强制对齐：十种单语记录

```bash
# 独立 Python 3.11 环境；服务器需具备合适的 CUDA torch。
python3.11 -m venv /envs/omnicommit-align-qwen
source /envs/omnicommit-align-qwen/bin/activate
python -m pip install -r requirements-alignment-qwen3.txt

cd /workspace/OmniCommit
python Data-Scripts/force_align_asr.py "$ANNOTATED" \
  --backend qwen3 --qwen-model Qwen/Qwen3-ForcedAligner-0.6B \
  --device cuda:0 --dtype bfloat16 \
  --output "$WORK/timestamps.jsonl"
```

支持本项目的中、英、日、韩、俄、西、葡、法、德、意单语记录。默认单条音频最长 60 秒；有明确声学子段边界的长会议可先对子段对齐。遇到异常预测时间、缺字或词序不对应，拒绝该行，保留错误原因，不用插值填出一条看似有效的数据。

### 2B. 阿拉伯语 / 泰语 / 混合语种

```bash
python3.11 -m venv /envs/omnicommit-align-whisperx
source /envs/omnicommit-align-whisperx/bin/activate
python -m pip install -r requirements-alignment.txt

python Data-Scripts/force_align_asr.py /data/ar.annotated.jsonl \
  --backend whisperx --device cuda \
  --output /data/ar.timestamps.jsonl

# 泰语：仅在已经有经过核验的兼容 CTC 模型时使用此入口。
python Data-Scripts/force_align_asr.py /data/th.annotated.jsonl \
  --backend whisperx --device cuda --align-model /models/verified-th-ctc \
  --output /data/th.timestamps.jsonl
```

WhisperX 不提供本项目泰语的默认对齐模型。**混合语种目前需导入外部可靠字词时间戳**，或先获得真实的语种音频区间，再分段对齐并汇总。代码没有把“多语言文本协议可用”冒充成“所有混合语言都已自动对齐验证”。WhisperX 无法对齐的数字或字形也会被拒绝。

两种模型适配器与所有十二语种的通用时间戳格式见 [对齐与重构说明](alignment_and_recast.md)。Qwen 的实际支持列表来自其[官方模型说明](https://github.com/QwenLM/Qwen3-ASR)；WhisperX 的按语种模型选择见[官方对齐源码](https://github.com/m-bain/whisperX/blob/v3.7.4/whisperx/alignment.py)。

通用外部格式如下，时间相对这一条记录的原始音频起点：

```json
{"id":"meeting-001","audio":"/data/audio/meeting.wav","words":[{"text":"请联系","start_ms":20,"end_ms":300},{"text":"张伟","start_ms":320,"end_ms":820},{"text":"确认方案","start_ms":850,"end_ms":1400}]}
```

`id/key` 和音频路径用于对应源记录；不要将不同录音的 `spk1` 当成同一人，也不要将裁剪前后的绝对时间混用。

## 3. 将时间戳分配给语义单元

切回轻量数据环境，保持当前 shell 中的路径变量。

```bash
source "$DATA_ENV/bin/activate"
python Data-Scripts/align_asr.py "$ANNOTATED" \
  --timestamps "$WORK/timestamps.jsonl" \
  --output "$WORK/aligned.jsonl"
```

程序要求完整正文按顺序对应，不模糊替换实体和数字。一个语义单元覆盖多个字词时，取它们的完整起止区间；词时间不够细时，使用整个词区间，提交可能稍晚，不猜测子词时间。

## 4. 用固定音频块重构标签

先选定一档完成训练和推理，不要不经检查混用两档。

```bash
# 500ms 版本
python Data-Scripts/recast_asr.py "$WORK/aligned.jsonl" \
  --chunk-seconds 0.5 --number-format normalized \
  --output "$WORK/trajectory.500ms.jsonl"

# 如果要实验1秒，另建一份数据。
python Data-Scripts/recast_asr.py "$WORK/aligned.jsonl" \
  --chunk-seconds 1.0 --number-format normalized \
  --output "$WORK/trajectory.1000ms.jsonl"
```

原转录已经是标准数字时用 `normalized`；如果保留口语数词，则全链路改用 `verbatim`。此参数只选择指令，不在重构阶段改写文字。

重构采用以下明确规则：

1. 单元完整结束后才允许提交；实体和数值内部不拆开。
2. 本轮首次完成一个初步 WAIT 单元，整轮 WAIT；即使本块同时覆盖后面的 DIRECT，也不会抹掉这次等待。
3. 同一个等待事件只触发一次。下一轮没有新触发时，可释放此前全部已完成但未提交的内容。普通内容可以连续 DIRECT。
4. 所有真实音频处理完后，增加文本 user 事件 `<end_of_audio>`，释放尾部积压；没有积压就 WAIT。**它没有音频，也不是第三种动作。**
5. 第一次初始化归属、相同前缀重复，都不是语种或说话人切换；同一人换语言不换 `spk`。

## 5. 生成 SFT 数据与完整 token 清单

```bash
python SFT-Scripts/prepare_sft.py "$WORK/trajectory.500ms.jsonl" \
  --output "$WORK/full.500ms.jsonl"

python SFT-Scripts/check_sft.py "$WORK/full.500ms.jsonl" --rows 2 \
  --tokens-file "$WORK/full.500ms.jsonl.tokens.txt"
```

真正训练输入形态如下。`audios` 中每个虚拟引用对应一段新增真实音频；reader 在原始采样上截取，再重采样，不创建大量小 WAV。

```json
{
  "audios": ["omnicommitaudio:v1:...", "omnicommitaudio:v1:...", "omnicommitaudio:v1:..."],
  "messages": [
    {"role":"system","content":"共享的增量ASR协议提示"},
    {"role":"user","content":"<audio>"},
    {"role":"assistant","content":"<|DIRECT|><zh><spk1>请联系"},
    {"role":"user","content":"<audio>"},
    {"role":"assistant","content":"<|WAIT|>"},
    {"role":"user","content":"<audio>"},
    {"role":"assistant","content":"<|DIRECT|><zh><spk1>张伟确认方案。"},
    {"role":"user","content":"<end_of_audio>"},
    {"role":"assistant","content":"<|WAIT|>"}
  ]
}
```

每个 DIRECT 的开头带完整 `<lang><spk>`，同次提交内部归属变更也带完整前缀。WAIT 没有语种/说话人/原因字段。SFT 实际输出还保留 schema、区间和来源元数据，方便检查。

完整会议样本的 speaker ID 应按首次出现顺序使用 `spk1, spk2, ...`，与推理注册规则一致。若原数据是从更大会议截出的片段、首人仍叫 `spk7`，请先明确会议范围并统一对应的参考信息；准备器会拒绝此类不一致记录，不悄悄重排身份。拆窗产生的后续样本可以从既有编号开始，因为它们带有此前编号表和声音参照。

## 6. 长样本拆窗与 Speaker Prompt Cache 训练

如果数据原本是十几秒的短音频，可直接使用 `full.500ms.jsonl`。长会议不能直接依靠训练器截断到 4096 token；必须在没有待提交内容、没有跨界半个实体的位置拆窗。

需要先准备同一会议中已经听过的清晰参考片段。参考来源是人工或可靠的外部说话人/时间验证，`verified:true` 不是脚本自动估计的置信度。例如：

```json
{"id":"meeting-001","references":[{"audio":"/data/audio/meeting.wav","start_ms":200,"end_ms":1800,"speaker":"spk1","language":"zh","text":"大家上午好。","verified":true,"clean":true,"overlap":false}]}
```

```bash
python SFT-Scripts/prepare_windows.py "$WORK/full.500ms.jsonl" \
  --output "$WORK/train.500ms.jsonl" \
  --max-audio-seconds 20 --max-reference-seconds 5 --max-speakers 8 \
  --max-total-reference-seconds 30 \
  --speaker-references /data/verified-speaker-references.jsonl

export TRAIN_JSONL="$WORK/train.500ms.jsonl"

# 若是短样本直接训练，则使用这一行替代上面的 TRAIN_JSONL。
# export TRAIN_JSONL="$WORK/full.500ms.jsonl"
```

参照音频和编号转录只作为 user 输入，不成为 assistant 训练目标。选取的参照必须早于当前窗口，覆盖此前已出现的说话人；没有可靠参照的长会议记录会被拒绝并记日志，不使用会议后面的声音补齐。没有足够短的安全边界也会拒绝，不能从姓名中间硬截。

窗口输入还带有此前最后已提交的语种/说话人状态，用来区分窗口首处是否发生真实切换；参考片段自身的语种不覆盖这个状态。经有效父会议及对齐区间验证的纯静音窗口保留为 WAIT 训练样本，任意全 WAIT 失败记录不会变成静音监督。

窗口最大秒数只是第一道预算。说话人参照和音频展开仍会占 token，下一步还要用真实训练模板检查长度。

## 7. 训练前完成结构、音频与真实长度检查

```bash
python SFT-Scripts/audit_sft.py "$TRAIN_JSONL" \
  --decode --report "$WORK/structural-audit.json"

# 在原有 omni_ms / ms-swift3.12.1 训练环境执行；此命令只加载处理器。
python SFT-Scripts/audit_encoded_lengths.py "$TRAIN_JSONL" \
  --model "$HF_BASE" --max-length 4096 \
  --report "$WORK/encoded-audit.json"
```

长度检查包括音频特征展开、完整对话模板、参考输入和监督目标；不只是 tokenizer 编码正文。超长时重新缩短安全窗口或参照预算，再检查。该命令逐行检查并汇总；如果仍有无效样本，会返回非零，不把未经验证的数据称为训练就绪。

多分片/训练验证集必须共用一份固定 token 清单，不能分别添加后重新排序。清单应覆盖实际训练的语种和说话人数；推理时不能新增一个没训练过的 `<spk17>` 来承诺识别第十七个人。

## 8. SFT：token 在训练前加入

```bash
bash SFT-Scripts/train_sft.sh \
  "$MCORE_BASE" "$TRAIN_JSONL" "$TRAIN_OUT" \
  --gpu-ids 0,1,2,6
```

保留 FLEX 原配方：四卡、LoRA r8/alpha32、EP2、micro/global batch 4/16、学习率 1e-4、3 epochs、packing、最大长度 4096。云道文本教师的量化 DeepSeek 是数据标注模型，与这里被训练的 Omni 基座不同。

`--new_special_tokens` 在 SFT 前注册 WAIT、DIRECT、语种与说话人 token；`--modules_to_save word_embeddings output_layer` 保存它们训练后的权重。**不是训练结束后再往 tokenizer 里加随机新 token。** `protocol_tokens.txt` 只有两种动作，正式训练应使用数据生成的完整 `.tokens.txt`。

旧驱动兼容入口仍是 `--compat-old-driver`；它会采用已有 FLEX 兼容配方，详细变化见 [SFT 说明](../SFT-Scripts/README.md)。

## 9. 导出：HF 基座 + 实际 PEFT checkpoint

将下面 `checkpoint-3000` 替换成真实保存出的 checkpoint，不能直接把整个训练输出根目录当成 adapter。

```bash
export ADAPTER="$TRAIN_OUT/checkpoint-3000"

# 先检查 token、训练参数和张量形状；不加载完整权重合并。
bash SFT-Scripts/merge_lora.sh "$HF_BASE" "$ADAPTER" "$MERGED" \
  --special-tokens "$TRAIN_JSONL.tokens.txt" --gpu-ids 0 --check-only

# 正式合并与逐项核验。
bash SFT-Scripts/merge_lora.sh "$HF_BASE" "$ADAPTER" "$MERGED" \
  --special-tokens "$TRAIN_JSONL.tokens.txt" --gpu-ids 0
```

最终目录含完整模型、tokenizer/processor、`omnicommit_tokens.json`、`omnicommit_tokens.txt` 和合并校验报告。程序核对 token 顺序与 ID、完整词表层及新增 token 已训练权重，不沿用 FLEX 的硬编码三个 token ID。未完成导出保留在 `.incomplete-*` 工作目录，不能当成正式模型使用；它不是备份副本。

## 10. 先跑短音频，再做长会议验收

在已有 Qwen3-Omni 推理环境中执行：

```bash
python -m pip install -r requirements-web.txt

python tools/stream_infer.py /data/test/short-meeting.wav \
  --backend hf --model "$MERGED" \
  --chunk-seconds 0.5 --number-format normalized \
  --context-policy verified \
  --output "$WORK/short-meeting.events.jsonl"
```

默认 `verified` 保留没有可信提交时间边界的音频；普通 DIRECT 不能证明这一段声音已全部转录。默认 30 秒预算不是会议最长时长，而是未经确认、不可安全丢弃的缓存上限。长会议必须持续提供可信提交边界，或显式选择下一段的滚动实验策略。

长会议试验可使用有界滚动模式：

```bash
python tools/stream_infer.py /data/test/long-meeting.wav \
  --backend hf --model "$MERGED" \
  --chunk-seconds 0.5 --number-format normalized \
  --context-policy rolling --recent-seconds 12 --rolling-tail-seconds 2 \
  --max-context-seconds 30 --max-pending-seconds 30 \
  --events /data/test/long-meeting.verified-controls.jsonl \
  --output "$WORK/long-meeting.events.jsonl"
```

如果暂时没有参照控制文件，可去掉 `--events`，这就是无 SPC 的对照组。`rolling` 根据提交和尾部保留时长推测裁剪边界，**不保证长期挂起内容完全不丢**；它会明确记录发生了启发式裁剪。持续 WAIT 超过预算仍会报错，不偷偷丢音频。

可信控制文件示例：

```jsonl
{"after_seconds":2.0,"op":"reference","speaker":"spk1","language":"zh","text":"大家上午好。","start":0.2,"end":1.8,"verified":true,"clean":true,"overlap":false}
{"after_seconds":2.0,"op":"commit","through_seconds":1.8,"verified":true}
```

它只在相应声音已经输入后应用。`reference` 维护身份参照；`commit` 表示该位置以前所有语音已经提交且没有挂起内容。两者不能互相代替，也不能从最新一块 0.5 秒音频和模型当前一个 spk 标签直接猜出。

## 11. 评估长间隔身份恢复与速度

对同一段真实会议比较：无参照、人工正确且来自过去的参照、自动组件提供的参照。至少包含 A 发言 → B/C 长发言 → A 在两分钟后回来；确保 A 的初始音频已经离开近期窗口。还要覆盖 A 换语言、短应答、人名、数值、噪声及重叠。

```bash
python tools/evaluate_stream.py \
  --reference /data/test/reference.jsonl --id meeting-001 \
  --predictions "$WORK/long-meeting.events.jsonl" \
  --output "$WORK/long-meeting.report.json"
```

报告包括文字 CER、匹配文字覆盖率、匹配文字上的语种/说话人正确率、额外 speaker ID、WAIT 比例、EOS 完整性、运行错误、生成 RTF 与 p50/p95。它**不对每块单独重排 speaker 编号**，也不是声学 DER。文字错很多时，不能只看“匹配文字上的说话人准确率”。可在参考中添加 `speaker_return_probes` 的规范化字符区间，单独检查回归发言段。

0.5 秒输入步长不代表 0.5 秒推理耗时。逐轮 p95、总 RTF、队列积压都要在目标 GPU 上量测。HF 后端当前每轮重新编码有限上下文；有界缓存解决的是无限增长，不能单独保证实时吞吐。另已提供从 FLEX 适配的 vLLM 0.11.1 后端，使用原生 prefix/MM processor 缓存和稳定音频 UUID；缓存命中依赖实际请求与窗口变化，不能将它说成保证免除全部音频计算。缓存条件下的声纹匹配能力也必须在实际 Omni 模型上验证。

## 12. 启动 Web

```bash
python tools/serve_web.py --backend hf --model "$MERGED" \
  --chunk-seconds 0.5 --number-format normalized \
  --context-policy rolling --recent-seconds 12 \
  --host 127.0.0.1 --port 7860 --max-sessions 1

# 或：在原 FLEX vLLM 0.11.1 推理环境中使用可选缓存后端。
# 与上面的 HF 命令二选一，不同时占同一个端口。
python tools/serve_web.py --backend vllm --model "$MERGED" \
  --chunk-seconds 0.5 --number-format normalized --context-policy rolling \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.85 \
  --mm-processor-cache-gb 1 --host 127.0.0.1 --port 7860 --max-sessions 1

# 在本地电脑另开终端，把服务器端口转到本机。
ssh -L 7860:127.0.0.1:7860 USER@SERVER
```

本地浏览器打开 `http://127.0.0.1:7860`，开启麦克风；结束时点击“结束并提交尾部”。页面显示逐轮结果、耗时、队列和待核验时长，支持导出事件 JSONL，另有人工提交边界与说话人参照入口。

vLLM 后端同样可用于 `stream_infer.py --backend vllm` 的离线验收；不要向训练环境升级安装 vLLM。模型单卡放不下时，按实际机器显存设置 `--tensor-parallel-size`。首次部署先用短音频验证 HF 与 vLLM 的 token、音频输入和输出协议一致，再比较速度。

模型只加载一份，调用串行处理，各会议的音频、编号和说话人参照隔离。输入队列有界；持续跟不上会明确停止录音并处理已接收部分，停止后的现场音频没有录入，不能误称整场会议已完整采集。远程麦克风需要 HTTPS 或上述 localhost 隧道。直接绑定公网需设置访问口令和 TLS，详见 [流式服务说明](streaming_runtime.md)。

## 当前完成范围

已实现数据协议、真实对齐适配器、固定步长重构、安全拆窗、SFT、token 导出核验、离线流式推理、SPC 可信接口、Web 和离线评估。正式 GPU 对齐/训练/导出/推理尚需在目标服务器验收；当前没有自动声纹确认模型，混合语种自动对齐也不在已验证范围内。代码提供可运行的基线、明确失败路径与观测指标，不把未测过的长会议速度或身份恢复准确率当成保证。
