# OmniCommit

**多语言、多说话人流式 ASR：学习何时等待，何时提交转录。**

OmniCommit 使用 Qwen3-Omni 对增量音频进行原语言转录。每轮接收一个新音频块，模型只输出以下一种结果：

```text
<|WAIT|>
<|DIRECT|><zh><spk1>新增转录
```

WAIT 不输出文本；DIRECT 输出尚未提交的文本，并为每段标明语种和说话人。同一个说话人切换语种时保留编号。语种或说话人变化时，DIRECT 内的新归属使用完整 `<语言><说话人>` 前缀。真实模型能否正确判断等待、语种及身份，需要实际训练和评测。

本仓库包含：文本大模型一次调用的 ASR 标注、音频时间戳对齐、0.5/1 秒标签重铸、FLEX 风格音频/文本交错 SFT、模型导出、流式推理与 Web。默认示例路径均为占位路径，必须换成实际服务器路径。仓库不包含用户音频、训练数据、模型权重或线上服务。

## 先选择当前输入

| 当前已有数据 | 下一步 |
|---|---|
| 原始音频和完整的带语种、说话人标签的转录 | 用 `Data-Scripts/annotate_asr.py` 一次调用文本模型切分并构建初步标签，再取得声学时间戳。 |
| 每行已有 `audios`、`messages`、`output`、`timestamps` | 使用 `Data-Scripts/build_asr_sft.py`，直接重铸并生成精简 SFT JSONL。 |
| 已有精简的 SFT JSONL 和对应 `.tokens.txt` | 直接进行音频、结构和真实训练模板长度检查，之后运行 SFT。 |
| 需要将长录音拆成训练窗口并加入说话人参照 | 使用保留中间元数据的分阶段路线；精简 JSONL 不能直接提供安全拆窗信息。 |

详细字段见 [数据格式](docs/data-format.md)。与此独立的音频教师累计前缀路线仍然保留，作为另一种数据构造实验，见 [方法与边界](docs/method.md)；不需要将它接在当前文本标注流程之后。

## 当前主线

```text
原始音频 + JSONL：audios / messages（完整转录及语言、说话人标签）
  ↓
annotate_asr.py：文本大模型一次调用，细切分 + 初步 WAIT / DIRECT
  ↓
可选 filter_asr_output.py：过滤标注格式和确定性协议错误
  ↓
取得音频对齐时间：每个切分单元及必要的内部归属变化都有时间戳
  ↓
build_asr_sft.py：按真实音频时长和 0.5 秒网格重铸，生成精简 SFT
  ↓
结构/音频检查 + 真实模板编码长度检查
  ↓
train_sft.sh：MCore Qwen3-Omni + Megatron LoRA SFT
  ↓
merge_lora.sh：原始 HF 基座 + 实际 adapter + 训练时的 token 清单
  ↓
start_server.sh：启动网页和流式推理服务
```

如果时间戳保存在独立文件，或需要保留安全边界用于拆窗，请使用 [分阶段执行手册](docs/RUNBOOK_ZH.md) 和 [对齐与重构说明](docs/alignment_and_recast.md)。**外部对齐结果与一体化脚本使用的行内 `timestamps` 是不同格式**，不能直接混用。

## 1. 原始数据和初步标注

每条 JSONL 记录是一条音频及对应的完整转录，形状例如：

```json
{"audios":["/data/audio/example.wav"],"messages":[{"role":"user","content":"<audio>**Entity Reference:** []"},{"role":"assistant","content":"<zh><spk1>请联系张伟确认方案。"}]}
```

`annotate_asr.py` 针对 `ar/ja/ko/it/fr/pt/es/ru/th/en/zh/de` 路由对应 prompt；输入中重复的相同归属标签会去重。输出保留原字段，将分段结果写入 JSON **字符串** `output`：

```json
{"segments":[{"source_text":"<zh><spk1>请联系","action":"DIRECT"},{"source_text":"张伟","action":"WAIT"},{"source_text":"<zh><spk1>确认方案。","action":"DIRECT"}],"flush":false,"error":null}
```

上面展示的是 `output` 解码后的内容。`source_text` 是该片原文，即使动作是 WAIT 也不为空；WAIT 的内容会在后续可提交时释放。标注不是时间戳，每片不等于 0.5 秒。普通片段可以连续 DIRECT。切分、实体、数字及归属的语义质量仍需抽查。

云道入口为单文件脚本，沿用现有云道参数。失败行写 `output:""` 并继续；具体默认路径和恢复行为见 [执行手册](docs/RUNBOOK_ZH.md)。需要过滤标注时，修改 `Data-Scripts/filter_asr_output.py` 顶部的 `INPUT_JSONL`、`OUTPUT_JSONL` 后运行该脚本；它不会调用模型，也不会验证时间戳和声学正确性。

## 2. 带时间戳的一体化 SFT 构建

对于当前使用的行内时间戳格式，每条输入包含原始 `audios/messages`、`output` 和 `timestamps`，可选 `tags`。`timestamps` 按分段对应，时间使用相对于该音频起点的毫秒；若一个单元内部发生语种或说话人变化，还需要提供带真实时间的 `spans`。输入必须是 JSONL，即每行一个完整 JSON 对象。

```bash
cd /data/yb/Code/OmniCommit

python Data-Scripts/build_asr_sft.py \
  /data/OmniCommit/timestamped.jsonl \
  --output /data/OmniCommit/run-001/train.500ms.jsonl \
  --chunk-seconds 0.5 \
  --number-format normalized
```

脚本读取真实音频时长，核对分段、归属与时间，按音频网格重铸，并产生：

- `train.500ms.jsonl`：成功数据，**每行只有 `audios` 和 `messages`**。
- `train.500ms.jsonl.tokens.txt`：训练需要的动作、语种和说话人 token 清单。
- `train.500ms.jsonl.rejected.jsonl`：被拒绝输入行的位置和原因。
- `train.500ms.jsonl.review.jsonl`：至少 2 秒未标注区间，供听音复查；未标注不等于静音。

单行错误不会中断后续处理；全部失败时返回非零。输出及 sidecar 已存在时不会覆盖。当前一体化脚本**不支持自动断点续跑**；系统级中断后可能留下不完整输出，需确认任务完成并使用新输出路径重新构建。

数词参数必须与原转录一致：`normalized` / `verbatim` 只改变 system 指令，**不会改写原文**。可用 `--chunk-seconds 1.0` 另建 1 秒数据。完整规则、可选间隙过滤与错误处理见 [时间戳直接生成 SFT](docs/timestamped_to_sft_zh.md)。

### 最终训练数据

下面是示意，`audios` 中的引用占位文字不能直接训练：

```json
{"audios":["音频块引用1","音频块引用2","音频块引用3"],"messages":[{"role":"system","content":"训练和推理共用的短协议"},{"role":"user","content":"<audio>"},{"role":"assistant","content":"<|DIRECT|><zh><spk1>请联系"},{"role":"user","content":"<audio>"},{"role":"assistant","content":"<|WAIT|>"},{"role":"user","content":"<audio>"},{"role":"assistant","content":"<|DIRECT|><zh><spk1>张伟确认方案。"},{"role":"user","content":"<end_of_audio>"},{"role":"assistant","content":"<|WAIT|>"}]}
```

每个 `<audio>` 对应 `audios` 中一个新增音频区间。真实引用以 `omnicommitaudio:v1:` 开头，编码保存原音频**绝对路径**和起止时间；训练插件按区间读取原始文件，不创建大量小 WAV。训练容器必须能够从同一绝对路径访问原音频。

重铸只提交已完整覆盖的源单元；真实语种/说话人切换在对应归属起点所在块触发 WAIT，其他源 WAIT 在完成块触发。触发当轮不提交；之后等待后续源 DIRECT 完整再合并释放。最后的 `<end_of_audio>` 是无音频输入事件，用于释放尾部，并非第三种动作。边界使用对齐结果，不能由文本自动证明实体完整或说话人正确。

## 3. 检查和训练

两个全量检查**只读并生成报告，不会过滤训练文件**：结构检查的错误详情默认最多记录 100 条，错误总数仍会完整统计。`--decode` 会逐块读取音频，可能耗时较长；不加时不验证每个音频块的解码。

```bash
python SFT-Scripts/audit_sft.py \
  /data/OmniCommit/run-001/train.500ms.jsonl \
  --decode \
  --report /data/OmniCommit/run-001/structural-audit.json

python SFT-Scripts/audit_encoded_lengths.py \
  /data/OmniCommit/run-001/train.500ms.jsonl \
  --model /models/qwen3_omni_original_hf \
  --max-length 4096 \
  --report /data/OmniCommit/run-001/encoded-audit.json
```

第二条使用原始 **HF** 模型的 processor 和实际 SWIFT 模板计算长度，不加载模型权重；它需要在原有训练环境运行。两项检查均需 token 清单，默认查找 `训练文件路径.tokens.txt`，可通过 `--tokens-file` 指定同一份清单。多分片必须共享覆盖所有数据的固定清单；继续训练和导出不能改变 token 顺序。超长数据不会自动拆分或过滤。

检查通过后，在现有 `omni_ms` / ms-swift 3.12.1 环境使用 **MCore** 基座训练：

```bash
bash SFT-Scripts/train_sft.sh \
  /models/qwen3_omni_mcore_3.12.1 \
  /data/OmniCommit/run-001/train.500ms.jsonl \
  /outputs/OmniCommit/sft-001 \
  --gpu-ids 0,1,2,6
```

启动器沿用 FLEX **虚拟音频 SFT 版本**的主要配方：四卡、LoRA r8/alpha32、EP2、micro/global 4/16、学习率 `1e-4`、3 epochs、packing、最大长度 4096；默认每 **3000 步**保存，原始 FLEX 旧脚本是 1000 步。启动器只抽查两条，不代替全量检查。默认保存未合并 adapter 和训练过的词表层，且不保存 optimizer/RNG，因此不能保证精确断点续训。详见 [SFT 文档](SFT-Scripts/README.md)。

训练前通过 `--new_special_tokens` 注册清单；结束后必须保留同一清单和对应权重，不能仅在训练后手动增加随机 token。

## 4. 导出和 Web

使用原始未量化 HF 基座和实际保存的 `checkpoint-N` 导出；替换下方占位路径：

```bash
bash SFT-Scripts/merge_lora.sh \
  /models/qwen3_omni_original_hf \
  /outputs/OmniCommit/sft-001/checkpoint-N \
  /outputs/OmniCommit/merged \
  --special-tokens /data/OmniCommit/run-001/train.500ms.jsonl.tokens.txt \
  --gpu-ids 0
```

模型和网页放在同一 GPU 服务器时，使用导出的 **HF 模型目录**启动：

```bash
bash start_server.sh /outputs/OmniCommit/merged
```

网页默认监听 `0.0.0.0:18889`，无需访问令牌，支持麦克风、系统声音、音视频文件、实时转录、逐段修正、说话人合并/改名及 TXT/JSON 导出。远程浏览器采集麦克风或系统声音仍需要受信任的 HTTPS 或 localhost；文件模式受浏览器支持格式限制。

自动说话人参照可选启用，使用独立 Python 环境中的 Qwen 对齐器进行**实验性**候选构建；它不验证身份或重叠发言，当前阿拉伯语和泰语不会自动建参照。默认 `rolling` 是启发式裁剪，可能遗漏尚未提交的语音；连续 WAIT 超过预算可能停止。真实 30 分钟准确率、速度和跨长间隔身份一致性尚未验收。启动参数、依赖和完整说明见 [会议 Web 服务](docs/web_meeting_service_zh.md)。

## 文档与验证

- [数据格式](docs/data-format.md)：原始数据、标注、时间戳与两种 SFT 格式。
- [时间戳直接生成 SFT](docs/timestamped_to_sft_zh.md)：行内时间戳、重铸与错误处理。
- [分阶段执行手册](docs/RUNBOOK_ZH.md)：标注、对齐、拆窗、训练与导出。
- [SFT 文档](SFT-Scripts/README.md)：token、训练参数和导出。
- [会议 Web 服务](docs/web_meeting_service_zh.md) 与 [流式状态说明](docs/streaming_runtime.md)：运行和限制。
- [方法与验证边界](docs/method.md)、[FLEX 对应关系](docs/FLEX_ADAPTATION.md)：研究路线和实现差异。

在隔离的数据开发环境可以执行 `python -m pytest -q` 和 `node tests/test_web_ui_contract.mjs`。测试中的 replay、合成音频和模拟模型只验证代码接口与状态，**不代表真实听音质量、GPU 训练、模型导出、实时吞吐或 30 分钟说话人稳定性已验证**。
