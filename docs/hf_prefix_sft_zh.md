# 本地 Omni 批量前缀推理与 SFT 构建

这条路线直接使用完整音频、带语种/说话人标签的转录，以及**仅在真实变化处提供的时间戳**。不需要先调用文本大模型切分，不需要旧的 `output.segments` 或逐词时间戳。

实现入口已经完成。CPU 测试覆盖了音频裁切、批量接口、缓存恢复、提交规则及现有 SFT 检查器；真实 HF 权重、GPU 吞吐、转录质量和训练效果仍需在服务器验证。本机测试中的模拟教师不代表模型准确率。

## 1. 如何处理现有数据

| 数据情况 | 教师实际接收的声音 | 归属和提交处理 |
|---|---|---|
| 一个语种、一个说话人 | 从文件起点逐步累计 | 重复语义标签合并；原文字、空格、标点保留 |
| 一个语种、多个说话人 | 从当前语种段起点继续累计，不在换人处重置 | 用原文确定说话人；在换人时间处另取一次前缀锚点 |
| 语种变化 | 旧语种在边界结束；新语种从边界重新累计 | 不让新语种教师听旧语种；最终学生仍听完整混合音频 |
| 一个块跨过变化处 | 教师证据按各语种段计算 | 当轮允许提交旧归属的确定尾部，新归属暂存到后续轮 |

例如中文结束于全局 2.3 秒：中文教师额外识别 `0–2.3s`；英文教师识别 `2.3–2.5s`、`2.3–3.0s`……。英文前两次实际得到的是 0.2 秒、0.7 秒。学生音频块仍是 `2.0–2.5s`、`2.5–3.0s`，没有删掉跨语种块。

对于 `A → AA → AAAC`，当前实现用额外的 `AAA` 边界锚点获得旧段结果，不依赖模型在 `AAAC` 中自行判断该截到哪里。保留同语种上下文是指保留累计声音，不是把完整参考答案塞进教师 prompt。

旧的每段 `<ar><spk1>` 语义标签不会触发 WAIT，也不要求额外时间戳。若真实变化缺少时间戳，该行拒绝；不能猜测发生时刻。参考说话人编号应按整条音频首次出现依次为 `spk1/spk2/...`；不同来源文件各自的 `spk1` 不能未经身份核实就视为同一人。

## 2. 输入格式

JSONL 每行一个完整对象，单行只有一段音频和一条 assistant 转录。单语单人可以完全没有 `timestamps`：

```json
{"audios":["/data/audio/one.wav"],"messages":[{"role":"user","content":"<audio>**Entity Reference:** []"},{"role":"assistant","content":"<ar><spk1>مرحبا.<ar><spk1> كيف حالك؟"}]}
```

变化时间相对于 `audios[0]` 的实际文件起点，单位毫秒；不从文件名推测长录音偏移。只需覆盖每一次真实语种或说话人变化：

```json
{
  "audios": ["/data/audio/mixed.wav"],
  "messages": [
    {"role": "user", "content": "<audio>**Entity Reference:** []"},
    {"role": "assistant", "content": "<zh><spk1>请检查方案。<en><spk1>Please check.<en><spk2> I agree."}
  ],
  "timestamps": [
    {"time_ms": 2300, "changed": ["language"], "previous_language": "zh", "previous_speaker": "spk1", "language": "en", "speaker": "spk1"},
    {"time_ms": 4100, "changed": ["speaker"], "previous_language": "en", "previous_speaker": "spk1", "language": "en", "speaker": "spk2"}
  ]
}
```

示例为便于阅读展开，实际文件必须一行一个对象。支持用户已有的 `content_original`、内嵌 `[20440ms]`、`message_index`、`tag_positions` 和 `backend` 元数据；出现的内嵌时间必须与外部变化事件对应。`tag_positions` 指向**插入时间标记前的原始 content**，不是插入后移动的位置。`part_index` 不用作语义句编号。无内嵌时间标记时不要求 `content_original`。

相同归属的相邻段合并后，时间戳可选的 `text` 必须与该完整归属段的正文一致。时间严格递增，不能超出实际音频，也不接受重叠说话被伪装成单一切换。当前版本不会判断实际声音是否重叠，需在数据来源层排除或专项检查。

## 3. 第一步：真实 HF 批量推理

在能够加载离线 ASR 模型的 GPU 服务器执行。教师是**离线 ASR 的 HF 目录**，不是 MCore，也不要求已经学过 WAIT/DIRECT 或存在 `omnicommit_tokens.json`。

模型调用按 Transformers 4.57.1 的 Qwen3-Omni 接口实现，复用现有与 CUDA 匹配的 torch、transformers、accelerate。CPU 音频依赖见 `requirements-data.txt`。这里直接运行 HF，与 Web 的 vLLM 版本无关。

下面的 HF 路径和源文件路径是明确的占位路径，需要替换成服务器上的真实位置；所有产物放在 `/data` 下：

```bash
cd /data/yb/Code/OmniCommit

python Data-Scripts/collect_hf_prefixes.py \
  /data/yb/Code/OmniCommit/datasets/data_006/source.switches.jsonl \
  --model /data/models/qwen3_omni_offline_asr_hf \
  --output /data/yb/Code/OmniCommit/datasets/data_006/prefix.500ms.jsonl \
  --gpu-ids 0 \
  --batch-size 4 \
  --chunk-seconds 0.5 \
  --number-format normalized
```

模型只加载一次。每个 batch 是当前记录的独立前缀请求，按音频长度排序降低 padding；不是将 batch 中不同音频拼成一段。长作业逐条读取，原音频每条只读一次，**先按源采样点裁切，再重采样为 16kHz**，不把未来波形参与的重采样结果倒填给早期前缀。

教师只有简短转录指令、已知语种及数词格式，不接收原始 system、Entity Reference、参考转录、上一轮识别或任何未来音频。生成确定性文本，关闭音频输出，保留原始 token IDs 和文本作为证据。

主要参数：

- `--batch-size 4`：一个 batch 的请求数量。遇到普通批次失败或 GPU OOM，后端递归拆小；单请求仍失败就记录错误，不无限重试。
- `--chunk-seconds 0.5`：支持 0.5 / 0.8 / 1.0。现有 Web 入口仍只开放 0.5 / 1.0；需要直接接现有服务时优先 0.5，不把 0.8 数据支持等同于服务已经支持。
- `--max-new-tokens 512`：达到上限仍没有正常 EOS 的输出记为失败，不训练成 WAIT。
- `--max-context-tokens 32768`：教师上下文加生成预留的上限；不是学生的 4096 训练长度。
- `--max-audio-seconds 300`：超长记录拒绝，不静默裁短。
- `--number-format normalized`：与当前原始数据的数字格式指令一致；若原文保留数词，改为 `verbatim`。脚本不自动改写参考正文。
- `--audio-root`：只用于相对路径。绝对路径须在当前容器可访问。

输出：

- `prefix.500ms.jsonl`：每条完整的教师轨迹，保留时间、原始预测、原文归属、模型配置摘要及错误状态。
- `.queries.sqlite3`：成功查询的恢复缓存，不是备份；每批提交，未完成记录也能复用。
- `.run.json`：配置、模型文件签名和代码签名。
- `.summary.json`：本次完成/失败/恢复及查询统计。

中断后保持代码、源数据、模型和参数一致，在原命令末尾增加 `--resume`。已完成的成功或失败记录都会跳过；重试失败记录应筛选对应输入后使用新的输出路径。成功查询写入缓存后，即使当前整行尚未完成，也无需重复推理。半截损坏 JSONL 不自动修复或备份；脚本会明确报告。模型/代码/配置改变会拒绝混用旧缓存。

逐行格式、音频及单次生成错误不会中止后续记录。模型加载失败、无法写文件、主机内存耗尽或无法继续使用的 CUDA 错误属于作业失败，不能假装成功继续。`--resume` 无需重复加载完全已完成数据的教师。

## 4. 第二步：校验、重铸、过滤并直接输出 SFT

```bash
python Data-Scripts/build_prefix_sft.py \
  /data/yb/Code/OmniCommit/datasets/data_006/prefix.500ms.jsonl \
  --output /data/yb/Code/OmniCommit/datasets/data_006/train.500ms.jsonl
```

本步骤只用 CPU，不再调用 Omni。默认比较方式 `surface`：折叠空白，忽略普通标点，保留大小写、变音/附加符号、数字邻接标点、撇号及连字符；原文输出不被归一化。`1.5` 不等于 `15`，`-5` 不等于 `5`，`a part` 不等于 `apart`。可用 `--comparison exact` 比较原始字符，包括标点和空格；严格模式可能因标点差异增加等待或拒绝。

提交处理：

1. 每个语种段的完整教师转录必须与参考正文按指定规则完全一致，否则拒绝整条；不通过模糊分数强行覆盖错误。
2. 每次**整条**前缀假设匹配完整锚点的开头，才贡献稳定长度；保留截至当前最长合格结果。错误尾部不能通过截掉错误字符获得部分信用。
3. 稳定长度映射回原文，保留参考文字和归属；说话人编号从现有标注取得，不接受独立教师请求里的临时 speaker 编号。
4. 未来切换和当前块中的切换限制新归属提交；最多提交到最早受限边界。旧归属的已确认尾部仍然可以输出。
5. 下一块解除前一块的切换限制，但仍需教师证据。没有新词或只有静音也可以释放之前已确认的积压，不再等待另一个文本切分中的 DIRECT。
6. 末块保留真实余长。最后追加无音频 `<end_of_audio>`；只有所有完整锚点已经通过核对时，才释放最后仍受切换门控的文本。

音频区间用 `[start,end)`；切换恰好等于块末端时属于下一块。一个块内多次切换全部记录，按最早边界限制本轮输出。源语义段长短不参与动作决策。

协议继续使用：

```text
有新增已确认文字：<|DIRECT|><zh><spk1>新增文字
没有可提交的新文字：<|WAIT|>
```

每次 DIRECT 开头和其内部每次真实归属变化都使用完整语言/说话人标签；不在 DIRECT 末尾加 WAIT。有旧段尾部可发时不额外插入一轮 WAIT。连续多轮 DIRECT 是正常的。

这借鉴了 [R2T2 的完整假设筛选](https://arxiv.org/html/2611.06043v1#S4.SS3)，但采用分语种教师、参考归属、归一化字符比较、变化门控和增量交错输出。论文的稳定区间 WAIT 与这里“无新增则 WAIT”的交互动作不同；中间轨迹保存稳定观察和区间供检查，不宣称原样复现论文训练。

输出文件：

- `train.500ms.jsonl`：每行仅 `audios/messages`，可直接交现有 SFT 管线。
- `.tokens.txt`：完整的 WAIT、DIRECT、语种及说话人 token 清单。
- `.rejected.jsonl`：失败行和原因。
- `.review.jsonl`：WAIT 音频轮次至少 80%，或至少一半正文推迟至 EOS 的待检查记录。
- `.trajectory.jsonl`：成功行的音频范围、提交轨迹、门控证据和诊断，便于回放与研究。
- `.summary.json`：保留/拒绝、各语种行数和正文字符数。多语种行会计入多个语种，不等于实际语音时长统计。

参考不一致、教师失败、时间或归属不一致、结构错误会被过滤，后面的行继续。默认 WAIT/EOS 占比只报告，不自动按这些比率删数据；如已确定实验阈值，可增加 `--max-wait-ratio`、`--max-eos-text-ratio`。输出必须是新路径，不覆盖、不自动备份。本 CPU 导出阶段不支持续跑，异常中断后选择新输出路径重建即可，已完成的教师推理可以保留。

实际训练音频使用现有虚拟区间引用，编码原始绝对路径、起止秒和必要的尾块补零长度。不会复制出大量 WAV，原始音频地址没有丢失。学生每轮只新增一个块，前面的音频/文本保留在前面轮次。原文和教师诊断不放进学生 user/system。

## 5. 检查、训练和服务衔接

```bash
python SFT-Scripts/audit_sft.py \
  /data/yb/Code/OmniCommit/datasets/data_006/train.500ms.jsonl \
  --decode \
  --report /data/yb/Code/OmniCommit/datasets/data_006/structural-audit.json

python SFT-Scripts/audit_encoded_lengths.py \
  /data/yb/Code/OmniCommit/datasets/data_006/train.500ms.jsonl \
  --model /data/models/qwen3_omni_original_hf \
  --max-length 4096 \
  --report /data/yb/Code/OmniCommit/datasets/data_006/encoded-audit.json

bash SFT-Scripts/train_sft.sh \
  /data/data_process/lzy/qwen3_omni_mega/train/qwen3_omni_mcore_3.12.1 \
  /data/yb/Code/OmniCommit/datasets/data_006/train.500ms.jsonl \
  /data/yb/Code/OmniCommit/outputs/sft-prefix-001 \
  --gpu-ids 1,2,3,4 \
  --save_interval 1000
```

教师 HF 与学生原始 HF/MCore 基座不是同一个参数用途；编码审计应使用实际学生基座及其配套 processor。训练仍沿用已有 FLEX 风格脚本。token 在训练前注册，导出沿用同一 `.tokens.txt`；不需要为新数据增加第三种动作，也不改旧模型 token ID 或现有服务解析器。训练命令只应在真实长度审计和样本听音通过后执行。

没有逐词时间戳时，同一归属内部的提前补全仍可能恰好匹配全文；LSP 和切换时间不能证明每个字的声学完成时刻。尤其需要检查数字、固定朗诵、重复短句、自然混语及错误拼接边界。教师只听单语片段而学生听混合流，最终模型能否学会切换，需要独立混语测试；标签来源一致不等于已经验证语音身份。严格全文核对也可能降低数据保留率，应先看各语种拒绝比例和样例，再决定后续标注改进。
