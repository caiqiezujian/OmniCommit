# 数据构建脚本

当前新路线使用完整音频、带语种/说话人标签的转录，以及仅在真实变化处标注的 `time_ms`：

1. `collect_hf_prefixes.py`：本地 HF Omni 加载一次，独立累计前缀批量推理，成功查询逐批缓存，可 `--resume`。
2. `build_prefix_sft.py`：核对完整教师转录、筛选稳定前缀、应用变化门控，过滤错误行并直接生成精简 SFT 和 token 清单。

同语种换人保留累计音频上下文，换语种才重新开始；相同归属的旧语义标签合并。最终学生仍接收完整录音的连续新增块。具体输入、容错、检查和训练命令见 [本地 Omni 前缀到 SFT](../docs/hf_prefix_sft_zh.md)。这两个入口不使用 API 或旧的 `output.segments`，也不要求逐词时间戳。

## 旧版文本教师切分路线

以下脚本继续支持已有数据：调用一次文本教师做细切分与 `WAIT/DIRECT` 标注，获取单元时间戳，再生成精简流式 SFT。其字段不同于上面的变化时间戳。完整输入和训练行样例见 [数据格式](../docs/data-format.md)。

1. `annotate_asr.py`：原始 `audios/messages` → `output` JSON 字符串；十二语种 prompt 内置，每条请求一次。坏输入、接口或输出校验失败时 `output` 为空并继续；不直接对音频产生时间戳。
2. `filter_asr_output.py`：按脚本开头的 `INPUT_JSONL`、`OUTPUT_JSONL` 过滤标注错误行。脚本参数写死、输出必须是新文件；检查格式和一致性不能证明文本和标签的声学准确性。
3. 获取实际对齐时间戳：如使用项目的分阶段对齐器或已有外部结果，参照 [对齐与重构](../docs/alignment_and_recast.md)。带行内 `timestamps` 的数据可直接使用下一步，不要重复运行对齐。
4. `build_asr_sft.py`：对带 `output/timestamps` 的每行读取原音频时长、校验、按 0.5/1 秒重铸并输出只有 `audios/messages` 的训练行，配套 `.tokens.txt`、`.rejected.jsonl` 和 `.review.jsonl`。坏行继续，所有行无效时非零返回，输出不覆盖且不支持中断续跑。

```bash
python Data-Scripts/build_asr_sft.py /data/timestamped.jsonl \
  --output /data/sft/train.500ms.jsonl \
  --chunk-seconds 0.5 --number-format normalized
```

`--number-format` 决定系统提示、不改写原文；模型输入为虚拟音频区间，训练端仍需能读取原文件。默认仅报告长未标注区间，不能把这些区间当已确认静音；可用 `--max-unannotated-gap-seconds` 选择过滤阈值。训练前还需完整解码和实际模板长度检查。运行细节见 [时间戳直接生成 SFT](../docs/timestamped_to_sft_zh.md) 和 [完整执行手册](../docs/RUNBOOK_ZH.md)。

## 可选：音频教师累计前缀路线

以下是独立的音频教师前缀路线，不是上述文本标注路线的必要步骤，二者不能简单串接或视为相同协议。其教师与轨迹的 `schema_version:1` 和当前一体化输出不同。

本目录实现真实音频前缀识别与离线轨迹构建，和 `SFT-Scripts/` 的训练准备器衔接。请上传整个 OmniCommit 项目；脚本通过同级 `omnicommit/` 包工作，不依赖本机 FLEX 目录。

## 三种数据

1. **原始输入**：单音频 `audios` 加 `messages`。必须有一条非空完整 assistant 转录，通常含语言与说话人标签。`id` 或 `key` 优先作为 ID。多音频、缺失或空 assistant 文本仍按单条错误处理；原参考标签不合法默认只影响参考校审，不影响独立音频教师识别。
2. **教师 trace**：完整音频识别 `full.segments` 和每个累计音频前缀的 `prefixes[].segments`。只读原参考保存在 `reference`，合法标签解析存入 `reference_segments`；解析失败则保留原文，写 `reference_segments:null` 和 `reference_parse_error` 错误文本，不猜测修复。参考不传入任何教师请求。
3. **增量 trajectory**：每轮有真实 `start_ms/end_ms`、`DIRECT/WAIT` 与新增 `segments`。SFT 准备器将其转成 `<|DIRECT|><zh><spk1>文字` 或 `<|WAIT|>`。

每个 segment 是 `{"language":"zh","speaker":"spk1","text":"文字"}`。语言代码使用小写；说话人按首次出现编号，整个累计音频内同一人保持身份。相邻同语言同说话人的段自动合并，文字、空格、标点保留。语言/说话人属性参与稳定前缀比较；仅改变 JSON 分段方式不会造成假等待。

## 实际音频服务

服务必须接受 OpenAI 兼容 `/chat/completions` 的 `input_audio` 内容块，内含 WAV base64。只有文本能力的服务不能使用此入口；本地文件路径不能代替真实音频上传。不同服务若仅支持 `audio_url` 或私有多模态协议，需要先实现明确的 transport 适配，当前代码不会猜协议。

提供自己的配置文件：

```json
{
  "api_base": "http://127.0.0.1:8080/v1",
  "model": "YOUR_AUDIO_MODEL_ID",
  "generation": {
    "temperature": 0.1,
    "top_p": 0.8,
    "max_tokens": 2048
  },
  "timeout": 300
}
```

上面的采样参数仅展示如何显式提供；请按实际服务配置填写。代码将 `generation` 原样传给服务，不默认偷偷追加 JSON mode、推理预算或其它生成参数；若不提供 generation 则遵循服务默认值。密钥放在 `OMNICOMMIT_API_KEY` 环境变量中，或通过 `--api-key-env NAME` 指定环境变量名，不写入配置、日志或 trace。

```bash
python Data-Scripts/collect_prefixes.py /data/source.jsonl \
  --output /data/omnicommit/teacher_trace.jsonl \
  --config /path/to/teacher.json \
  --chunk-seconds 0.5 \
  --number-format verbatim

python Data-Scripts/build_trajectories.py /data/omnicommit/teacher_trace.jsonl \
  --output /data/omnicommit/trajectories.jsonl
```

`--api-base/--model` 可以覆盖配置中的同名项；最终设置与 prompt 摘要写入 trace，密钥除外。相对音频路径默认相对于原始 JSONL；`--audio-root` 可改变相对路径的根；`--path-map '/old/root=/new/root'` 用于服务迁移，允许重复提供多个映射。

每个记录先识别完整音频，再识别 0～0.5 秒、0～1.0 秒等累计前缀。最后一个端点复用完整音频结果；不足一个 chunk 的尾段也保留。前缀请求之间没有聊天历史，也不强迫教师继承此前结果。所有请求都只含当次裁剪的真实音频与统一教师 prompt，不含完整参考答案、源 system、未来音频或过去转录。

先截原始采样再下混/重采样，输出内存 PCM16 WAV，不创建小音频文件。默认单条音频上限 300 秒（可通过 `--max-audio-seconds` 调整），超限整条报错而非静默截断。逐条处理且只保留当前记录的音频/识别结果，当前版本不并发请求；大规模任务可按源文件分片，在不同输出路径并行运行。增加 chunk 频率会明显增加教师调用次数。

## 数字策略与等待

- `--number-format verbatim`（默认）：保留原始数词口语形式，例如“三万五千”或“one hundred million”。原始 JSONL 可能使用阿拉伯数字，因此词面审查分数较低不一定是 ASR 错误。
- `--number-format normalized`：prompt 要求完整数量级确认后再输出规范化数值。轨迹阶段另外保护最终输出中的连续数字串、日期和小数，避免把最终 `100000000` 的前缀 `100` 提前提交。该规则不是完整实体识别器，不保证跨词数量表达、名字或说话人身份绝对正确。
- `--holdback-chars N`：可在非最终轮额外暂缓 N 个归属字符，最终轮清空。它是字符缓冲，不是论文的 BPE token 预算，也不意味着保留一个完整实体。

轨迹的 `policy.number_format` 传给 SFT 准备器，用于选择匹配的训练提示；SFT 准备器不会再次修改数字文本。

LSP 使用论文的 **whole-earlier-output** 条件：只有某次完整教师输出是最终输出的前缀才贡献长度，后续猜错时保留先前最长的合格结果。比较单位扩展为 `(language, speaker, character)`，不是某次错误假设中恰好相同的部分最长公共前缀。相邻稳定前缀取差分：非空则 DIRECT，否则 WAIT。仅新增空格的轮次暂缓到后续有文字时一起提交，避免丢失英文词间空格。最后一轮没有新增文字可以是 WAIT；全段无语音目前不导出到 SFT。

这是**参考论文方法的字符级、多属性扩展**。教师自身的一致性并不能证明声学上已消歧，独立调用的说话人编号也不保证身份正确；不能把这些轨迹称为人工真值或严格声学稳定性证明。实体、重叠语音、语种/人员切换仍需专项抽查。SFT 的音频文本交错训练形态也不等于论文累计前缀的原始 SFT 配方。

## 校审、失败与恢复

构建器生成 `reference_audit`，包括全文词面相似度及语言/说话人集合。它不修改教师全文，也不按参考答案重写某轮输出。可用 `--min-reference-agreement 0.8` 设置保守筛选，但数词规范化、拼写规范差异会影响该分数。集合相同不代表说话人身份映射正确。

例如原始参考含 `<zh><spk2><spk3>呃` 时，无法知道哪个空标签正确：默认保留这些原始字符并记录解析错误，继续从音频独立得到教师结果。轨迹保留 `reference/reference_parse_error`，并写 `reference_audit.status:"unavailable"`、`text_agreement:null`、`reference_languages:null`、`reference_speakers:null`，教师语言与说话人集合仍保留。合法参考的审查状态为 `available`；明确的空数组与不可用的 `null` 不混同。若启用任何大于 0 的 `--min-reference-agreement`，参考不可用的记录会被拒绝，不能绕过阈值。collector 的 `--strict-reference` 可恢复严格拒绝原参考标签错误；该开关进入恢复配置指纹，不能中途切换后混写。

除默认降级的原参考标签解析问题外，单条解析、音频、HTTP、截断响应、空内容或结构错误：该条写 `status:error, output:""`，并写 `OUTPUT.rejected.jsonl`；继续下一条，不重试、不停止服务。输入 JSONL 非对象行或语法错误也按单条处理。成功轨迹写 `turns`，错误行不进入 SFT。扫描完成即返回 0，可从末尾计数和拒绝报告识别是否全部失败；文件/配置/恢复一致性等任务级错误返回 2。没有自动备份。

`--resume` 跳过已经写出的记录，包括失败记录。输入内容、顺序、音频文件大小/修改时间、参数、prompt 或 replay 发生变化会拒绝混写，不会自动修复、回滚、删除旧输出。损坏或未写完的输出末行也拒绝追加，请保留并选择新输出路径。当前恢复粒度是**整条音频**，中途失败的部分前缀不会自动续接。

## 不调用服务的 replay

人工准备测试响应，按输入顺序存一条一个录音的 JSONL：

```json
{"id":"demo","full":{"segments":[{"language":"en","speaker":"spk1","text":"Hello world"}]},"prefixes":[{"end_ms":500,"segments":[{"language":"en","speaker":"spk1","text":"Hello"}]}]}
```

对应音频长 1 秒时，1000 ms 端点自动复用 full，无需再写一遍。除最终端点外，每个配置的时间点必须恰好有一条响应；音频仍须可读，因而同时验证真实采样边界。replay 不是语音识别效果评测。

```bash
python Data-Scripts/collect_prefixes.py /data/demo_source.jsonl \
  --output /data/demo_trace.jsonl \
  --replay-responses /data/demo_replay.jsonl

python -m pytest Data-Scripts/tests -q
```

测试覆盖 whole-output 与部分 LCP 的区别、语言/说话人属性、教师修订、数字保护、空格延迟、末尾等待、HTTP 参数和音频载荷、服务截断/非法格式、失败继续和恢复、连续空说话人标签降级校审及阈值拒绝、先裁剪后重采样的未来音频隔离。测试不会调用真实模型或上传录音。
