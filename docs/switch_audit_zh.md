# 审查 SFT 在语种和说话人切换处学到了什么

`SFT-Scripts/audit_switches.py` 读取实际用于训练的 JSONL，自动识别第一版和 Second 的输出协议。在 Second 中，连续发言会继承前一轮标签。每轮时间取自虚拟音频引用，500、700、800、1000 ms 等步长均按文件实际值读取。环境使用项目数据依赖，全程在 CPU 执行。

## 先查看最终训练目标

```bash
cd /data/yb/Code/OmniCommit
python SFT-Scripts/audit_switches.py \
  /data/yb/Code/OmniCommit/datasets/data_005/train_v1.jsonl \
  --output-dir /data/yb/Code/OmniCommit/outputs/switch-audit-001
```

打开输出目录下的 `review.md`。默认从全量切换点中固定种子随机抽 30 处，展示切换提交前 2 秒、后 4 秒的目标。在只有训练文件时，可以看到连续 WAIT 和每次提交的文字。参考切换时刻和提交延迟标为未知。若 SFT 完全漏标了某次切换，需提供原标注才能检出。

## 带时间标注试听日语切德语

下面的 `source.switches.jsonl` 需换成**构建这份训练文件时使用的标注**，支持只给切换打 `time_ms` 的原始记录，也支持带逐段起止时间的构建输入。换一个输出目录即可保留两次结果。

```bash
python SFT-Scripts/audit_switches.py \
  /data/yb/Code/OmniCommit/datasets/data_005/train_v1.jsonl \
  --source-jsonl /data/yb/Code/OmniCommit/datasets/data_005/source.switches.jsonl \
  --from-language ja --to-language de \
  --before-seconds 3 --after-seconds 12 --write-clips \
  --output-dir /data/yb/Code/OmniCommit/outputs/switch-audit-ja-de
```

若音频没有挂载，省略 `--write-clips` 即可做文本与时间审查。试听片段保留原采样率与声道。日语切德语同时换人的情况也会保留。想看很短的中间语种，把 `--after-seconds` 设大可以看到后续过程；逐语种的模型计数会在下一个已知切换点截止。

若第一版前缀构建器留下了轨迹，可加 `--trajectory-jsonl /实际路径/trajectory.jsonl`；脚本也会自动读取与训练文件同名的 `.trajectory.jsonl` 伴随文件。逐轮时间、动作、正文和归属都相符时才展示轨迹。可在报告中查看 `blocked_new_text` 和 `reference_fallback_characters` 等构造字段，判断某轮是否拦住了已有候选正文，或者用参考补出了多少字符。

## 把三种推理与标准答案并排看

在上面的命令中追加以下三个参数。日志须属于同一条原始录音，且模型输入的分块起止与 SFT 相符。新选输出目录即可独立保存本次结果。

```bash
  --trace outputs/ja-de.500ms.hf.jsonl \
  --trace outputs/ja-de.500ms.vllm-on.jsonl \
  --trace outputs/ja-de.500ms.vllm-off.jsonl
```

- `--audio /data/audio/meeting.wav` 可以只看这一条训练录音。若日德测试音频属于独立验证集，用**验证音频自己的最终 SFT 格式标注文件**作为第一个参数。只有测试日志时，可用已有的 [推理区间对照工具](../tools/compare_switch_segment.py) 比较整段与冷启动。
- 网页日志经常只有文件名。用 `--trace outputs/web.jsonl /data/audio/meeting.wav` 显式指定原文件。网页记录中的原文件时间会直接使用。
- 把 58 秒后另存成一个 WAV 跑命令行时，用 `--trace outputs/fresh58.jsonl /data/audio/meeting.wav 58`。对从第 60 秒起跑的日志独立传 `60`。必须确认裁剪文件来自该原录音。即使说话人编号因新会话而重排，仍可比对正文和语种。

## 输出与判读

| 文件 | 可以核对的内容 |
|---|---|
| `review.md`、`clips/` | 抽样案例、试听链接、逐轮完整目标与模型原文。音频秒数相对原录音。 |
| `switches.jsonl` | 所有通过筛选的切换事件。包含类型、提交量、时间依据、可用的归属核对及每份日志的观测统计。 |
| `cases.jsonl` | 抽样案例的全部展示轮次，可供 Python 二次分析。 |
| `summary.json` | 输入行数、切换方向、引用匹配数、组内 WAIT 比例与平均每个 DIRECT 的字符数。 |
| `issues.jsonl` | 错行、原文不一致、时间依据无法关联、日志原录音不匹配或试听片段导出失败的原因。 |

先听原音，记下人第一次可以确认新语种的时间，再看下列三种现象。

1. **切换后出字慢。** 对照标注切换时刻、SFT 首次发出新段正文的轮次、该轮字数及前面的连续 WAIT。首个被完整对齐的语义单元可能很长；仅有切换起点时，也可以量出从标注起点算的提交延迟，但无法知道每个字何时说完。
2. **正文先出来，标签仍是旧语种。** 独立参考与 SFT 正文能逐字对应时，报告检查新正文首轮的实际归属，并另记首次采用正确标签的轮次。模型栏统计观测到的有效语种；像 `<ja><spk3>Ich bin …` 是日语标签下的正文，需试听判断文本。遇到 `Hello` 等可能属于多语的词，也按录音和标注政策核对。
3. **连续 WAIT 或后续持续错误。** 看模型日志覆盖、最长连续 WAIT、接下来的原始回复和生成耗时。缺失的轮次、不同时间网格和服务报错各有状态。可对同一原音附上整段、58 秒起跑、60 秒起跑的日志，观察此前上下文是否改变了后续表现；单个案例无法确立训练、历史或缓存的因果关系。

`gold_audio_turn_profiles` 用有完整原文核对和切换时间的整条训练录音，分别统计单一归属录音、切换后的音频轮，以及同一批切换录音中远离切换的位置。邻域宽度用 `--after-seconds`，切换前留同宽的间隔以免混入普通位置。同一轮只计一次，结束事件单列在逐轮详情中。这是按所选录音统计的背景分布，方向和类型筛选仅作用于切换案例。字符统计含正文空格、标点，各语种书写系统与语速会影响字符数。

同一条参考只有语义断句时，连续重复的语种和说话人标签不会计为切换。参考和 SFT 全文不一致或裁剪窗口无法唯一定位时，该行回退到仅查看 SFT 自身，并记录原因。模型首个目标语种的时间仅依据日志覆盖内带该有效标签的正文；试听可进一步区分首个正确词、标签正确、整句正确三个时刻。

只有 SFT 时可看 `sft_commit_profiles`。其中 `switch_commit` 是输出中发生标签变化的提交轮，`continuing_attribution` 是延续当前归属的提交轮。对照两组的 `characters_per_commit` 和 `mean_preceding_wait_turns`，可以量化模型训练目标在切换提交时是否更倾向一次出很多字。切换前累计的 WAIT 可能包含上一人的停顿；其声学原因需要原音核对。
