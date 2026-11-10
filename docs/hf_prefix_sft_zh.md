# 本地 Omni 批量前缀推理与 SFT 构建

这条路线直接使用完整音频、带语种/说话人标签的转录，以及**仅在真实变化处提供的时间戳**。不需要先调用文本大模型切分，不需要旧的 `output.segments` 或逐词时间戳。

实现入口已经完成。CPU 测试覆盖了音频裁切、批量接口、缓存恢复、提交规则及现有 SFT 检查器；真实 HF 权重、GPU 吞吐、转录质量和训练效果仍需在服务器验证。本机测试中的模拟教师不代表模型准确率。

## 1. 如何处理现有数据

| 数据情况 | 教师实际接收的声音 | 归属和提交处理 |
|---|---|---|
| 一个语种、一个说话人 | 从文件起点逐步累计 | 重复语义标签合并；原文字、空格、标点保留 |
| 一个语种、多个说话人 | 主轨迹保留累计上下文，另采集后续归属段的局部前缀 | 用原文确定说话人；累计结果漏掉插话时，局部轨迹仍可识别后文 |
| 语种变化 | 旧语种在边界结束；新语种从边界重新累计 | 不让新语种教师听旧语种；最终学生仍听完整混合音频 |
| 一个块跨过变化处 | 教师证据按各语种段计算 | 当轮允许提交旧归属的确定尾部，新归属暂存到后续轮 |

例如中文结束于全局 2.3 秒：中文教师额外识别 `0–2.3s`；英文教师识别 `2.3–2.5s`、`2.3–3.0s`……。英文前两次实际得到的是 0.2 秒、0.7 秒。学生音频块仍是 `2.0–2.5s`、`2.5–3.0s`，没有删掉跨语种块。

对于 `A → AA → AAAC`，当前实现用额外的 `AAA` 边界锚点获得旧段结果，不依赖模型在 `AAAC` 中自行判断该截到哪里。保留同语种上下文是指保留累计声音，不是把完整参考答案塞进教师 prompt。

现在默认启用“完整片段参考补足”：有真实语种/说话人变化的记录中，某个归属段已经完整收到、其切换暂存已结束，但教师没有正确识别时，可以用操作方提供的可信参考正文构造剩余目标。此模式将输入参考视为可信监督，不代表脚本已经核验声学真值。单语单人、没有真实变化的记录仍走原严格教师核对路线。

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
- `--chunk-seconds 0.5`：支持 0.5 / 0.8 / 1.0。采集计划、标签门控、SFT 音频区间以及 CLI/Web 服务均支持这三种步长；默认仍为 0.5。
- `--max-new-tokens 512`：达到上限仍没有正常 EOS 的输出记为失败，不训练成 WAIT。
- `--max-context-tokens 32768`：教师上下文加生成预留的上限；不是学生的 4096 训练长度。
- `--max-audio-seconds 300`：超长记录拒绝，不静默裁短。
- `--number-format normalized`：与当前原始数据的数字格式指令一致；若原文保留数词，改为 `verbatim`。脚本不自动改写参考正文。
- `--audio-root`：只用于相对路径。绝对路径须在当前容器可访问。

输出：

- `prefix.500ms.jsonl`：每条完整的教师轨迹，`queries` 保存主轨迹，`local_queries` 保存同语种后续说话人段的局部轨迹；保留原始预测、原文归属、时间及错误状态。
- `.queries.sqlite3`：成功查询的恢复缓存，不是备份；每批提交，未完成记录也能复用。
- `.run.json`：配置、模型文件签名和代码签名。
- `.summary.json`：本次完成/失败/恢复及查询统计。

中断后保持代码、源数据、模型和参数一致，在原命令末尾增加 `--resume`。已完成的成功或失败记录都会跳过；重试失败记录应筛选对应输入后使用新的输出路径。成功查询写入缓存后，即使当前整行尚未完成，也无需重复推理。半截损坏 JSONL 不自动修复或备份；脚本会明确报告。模型/代码/配置改变会拒绝混用旧缓存。

逐行格式、音频及单次生成错误不会中止后续记录。模型加载失败、无法写文件、主机内存耗尽或无法继续使用的 CUDA 错误属于作业失败，不能假装成功继续。`--resume` 无需重复加载完全已完成数据的教师。

### 切换为 800ms 或 1 秒

步长由命令参数决定，不从文件名推断。例如 800ms：

```bash
python Data-Scripts/collect_hf_prefixes.py \
  /data/yb/Code/OmniCommit/datasets/data_006/source.switches.jsonl \
  --model /data/models/qwen3_omni_offline_asr_hf \
  --output /data/yb/Code/OmniCommit/datasets/data_006/prefix.800ms.jsonl \
  --gpu-ids 0 \
  --batch-size 4 \
  --chunk-seconds 0.8 \
  --number-format normalized
```

1 秒使用 `--chunk-seconds 1.0`，并建议将文件名改为 `prefix.1000ms.jsonl`。后续构建器直接读取 trace 内的 `chunk_seconds`，不再要求重复指定步长，避免采集和重铸配置不一致。改变步长应使用新输出路径，不用旧作业的 `--resume`。

R2T2 的前缀标注网格是 **80ms（0.08 秒）**，主要实验是 **160ms**；不是固定 800ms。这里 500/800/1000ms 是本项目的实验配置，不是论文数值的直接复刻。[论文 §5.2](https://arxiv.org/html/2611.06043v1#S5.SS2)、[§7](https://arxiv.org/html/2611.06043v1#S7)

## 4. 第二步：校验、重铸、过滤并直接输出 SFT

```bash
python Data-Scripts/build_prefix_sft.py \
  /data/yb/Code/OmniCommit/datasets/data_006/prefix.500ms.jsonl \
  --output /data/yb/Code/OmniCommit/datasets/data_006/train.500ms.jsonl \
  --reference-fallback completed-spans
```

本步骤只用 CPU，不再调用 Omni。默认比较方式 `surface`：折叠空白，忽略普通标点，保留大小写、变音/附加符号、数字邻接标点、撇号及连字符；原文输出不被归一化。`1.5` 不等于 `15`，`-5` 不等于 `5`，`a part` 不等于 `apart`。可用 `--comparison exact` 比较原始字符，包括标点和空格；严格模式可能因标点差异增加等待或拒绝。

提交处理：

1. `--reference-fallback completed-spans` 为默认值。在有真实切换的记录中，参考是目标正文；完整教师输出与参考的差异作为诊断保存，不再一律拒绝整条。`--reference-fallback off` 恢复旧版严格核对；没有真实切换的单语单人记录默认也保持严格核对。
2. 每次**整条**前缀假设匹配参考开头，才贡献较早提交的证据；保留截至当前最长合格结果。错误尾部不能通过截掉错误字符获得部分信用。主轨迹和局部轨迹都保留原样，不把参考内容伪装成教师输出。
3. 同语种后续说话人段的局部前缀以该段起点重新识别，独立对照该段参考。这样累计识别漏掉 B 后，即使主轨迹一直是 `A+C`，也不会使正确的局部 C 永远无法提交。正常主轨迹仍保留同语种上下文。
4. 对每个归属段，优先提交教师已经确认的前缀；仅在整个片段声音已收到后，才允许用参考补足剩余后缀。没有逐词时间戳时，不对尚未结束的长段提前补全文。
5. 未来切换和当前块中的切换限制新归属提交；旧归属已经满足条件的尾部仍可输出。下一块解除该段自己的切换限制，后面出现新切换不会重新冻结旧段。按原文顺序推进，不重复、不跳过缺口。
6. 末块保留真实余长。最后追加无音频 `<end_of_audio>`，可以释放已经完整收到且符合所选监督策略的剩余文字，不制造额外音频来等待。技术错误、缺失查询、格式和时间错误仍拒绝，不能冒充有效参考补足。

音频区间用 `[start,end)`；切换恰好等于块末端时属于下一块。一个块内多次切换全部记录，按最早边界限制本轮输出。源语义段长短不参与动作决策。

例如 B 在 `[1200,1400)`、C 在 `[1400,2000)`：两次切换都位于第三个 500ms 块。第三块只提交旧 A 的可用尾部；第四块可以补入教师未识别的可信 B，并紧接着提交 C 的可用部分，不额外让每个语种排一轮。若 C 改为 1500ms 开始，则第四块刚进入 C，第四块只能释放 B，C 留待后续。B 已输出过一部分时只补后缀。A→B→A 的第二个 A 是新的时间段。

参考补足只写入 assistant 目标。参考文本不进入学生或教师的 user/system，线上服务也没有参考补字功能。该方式依赖操作方对输入参考及非重叠边界的信任；不是“教师没有识别出来也证明声音足够清晰”。

正常结束但只生成已知语种/说话人标签、没有正文的预测，在启用补足的多归属记录中按空识别处理；严格模式仍拒绝。未知标签、动作标记、生成截断及请求错误不按空识别放行。

协议继续使用：

```text
有新增已确认文字：<|DIRECT|><zh><spk1>新增文字
没有可提交的新文字：<|WAIT|>
```

每次 DIRECT 开头和其内部每次真实归属变化都使用完整语言/说话人标签；不在 DIRECT 末尾加 WAIT。有旧段尾部可发时不额外插入一轮 WAIT。连续多轮 DIRECT 是正常的。

这借鉴了 [R2T2 的完整假设筛选](https://arxiv.org/html/2611.06043v1#S4.SS3)，但采用分语种教师、参考锚定、归一化字符比较、局部识别、完整片段参考监督、变化门控和增量交错输出。参考补足明确记录为 `reference_completed_span`，不叫作教师 LSP。论文的稳定区间 WAIT 与这里“无新增则 WAIT”的交互动作不同，不宣称原样复现论文训练。

输出文件：

- `train.500ms.jsonl`：每行仅 `audios/messages`，可直接交现有 SFT 管线。
- `.tokens.txt`：完整的 WAIT、DIRECT、语种及说话人 token 清单。
- `.rejected.jsonl`：失败行和原因。
- `.review.jsonl`：所有使用参考补足的记录、WAIT 音频轮次至少 80%，或至少一半正文推迟至 EOS 的待检查记录；含实际补入位置及数量。
- `.trajectory.jsonl`：成功行的音频范围、提交轨迹、门控证据和诊断，便于回放与研究。
- `.summary.json`：保留/拒绝、各语种行数和正文字符数、实际补入的行/段/字符及语种分布，以及 `chunk_seconds` 分布。多语种行会计入多个语种，不等于实际语音时长统计。

教师请求/生成技术失败、时间或归属不一致、结构错误仍会被过滤，后面的行继续；完整转录不一致是否过滤由上述参考模式决定。默认 WAIT/EOS/参考补入占比只报告，如已确定实验阈值，可增加 `--max-wait-ratio`、`--max-eos-text-ratio`、`--max-fallback-text-ratio`。输出必须是新路径，不覆盖、不自动备份。本 CPU 导出阶段不支持续跑，异常中断后选择新输出路径重建即可。

旧版已完成的 `local_prefix_trace` 可直接交给新构建器，不必仅为了开启参考补足重跑模型。旧 trace 没有 `local_queries` 时会记录缺少局部恢复轨迹；仍可在完整段结束后补足，但后文较早提交的证据可能较少。新版采集的 v2 代码/计划签名与 v1 不同，不会以 `--resume` 混入旧采集作业。

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

800ms 数据对应服务可这样启动；1 秒时把参数改为 `1.0`：

```bash
bash start_server.sh /data/yb/Code/OmniCommit/outputs/merged --chunk-seconds 0.8
```

网页接收服务端的步长后按对应采样数发送音频。参数支持不等于模型已经验证跨步长泛化；先使用与训练数据相同的步长测量准确率和延迟。

没有逐词时间戳时，同一归属内部的提前补全仍可能恰好匹配全文；前缀筛选和切换时间不能证明每个字的声学完成时刻。尤其需要检查数字、固定朗诵、重复短句、自然混语及错误拼接边界。教师只听单语片段而学生听混合流，最终模型能否学会切换，需要独立混语测试；标签来源一致不等于已经验证语音身份。应检查各语种拒绝及补入比例，避免数据主要变成“长时间等待、到边界整段输出”。
