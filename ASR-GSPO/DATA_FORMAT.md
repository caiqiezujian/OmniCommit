# 一条录音在 ASR-GSPO 中的数据格式

操作方准备一条音频路径、对应的整段原语言转录，以及已有的时间标注。下面用同一条 2.75 秒的中英双人示例走完整个流程。示例的时间和文字用于说明格式；数据准备与奖励已用临时 WAV 验证，候选回复为手写示例。

## 1. 原始输入 `timestamped.jsonl`

每一行对应一条完整录音。文件采用 UTF-8，下面为了便于阅读展开了一行。如果 `audios` 中写相对路径，默认从输入 JSONL 所在目录寻找音频，亦可用 `--audio-root` 指定目录。

```json
{
  "id": "demo-001",
  "meeting_id": "meeting-01",
  "audios": ["demo.wav"],
  "messages": [
    {"role": "user", "content": "<audio>**Entity Reference:** []"},
    {"role": "assistant", "content": "<zh><spk1>好了。<en><spk2>OK. Next."}
  ],
  "timestamps": [
    {"time_ms": 680, "language": "en", "speaker": "spk2", "changed": ["language", "speaker"]}
  ],
  "alignment": [
    {"text": "好", "start_ms": 100, "end_ms": 400},
    {"text": "了", "start_ms": 410, "end_ms": 620},
    {"text": "OK", "start_ms": 680, "end_ms": 950},
    {"text": "Next", "start_ms": 2100, "end_ms": 2530}
  ]
}
```

原始文件已有的 system 消息和重复语义标签可以保留。常规输入需要 `audios` 中的一条可读录音，以及 `messages` 中一条含完整转录的 assistant 消息。原始行无须填写 `DIRECT` 或 `WAIT`。`id` 也可用 `key`，两者均缺省时准备器使用音频绝对路径。`meeting_id` 可选，同一会议的多个片段可以据此放入同一数据划分。

时间以当前 WAV 开头为零。`timestamps` 记录真实归属变化的起点，`alignment` 记录参考中的字或词何时开始、何时结束。缺少词级 CTC 时可以省略 `alignment`；只有切换起点的样本仍能训练正文和切换处的覆盖。如果只有音频和带标签正文，这两个时间字段都可以省略。已有同语言同说话人的重复标签不需要打时间戳。

## 2. 准备输出 `tasks.jsonl`

运行 `bash 01_prepare.sh 原始.jsonl --output tasks.jsonl --chunk-ms 1000` 后，**一条通过校验的录音生成一条任务**。外层有三个字段。下面为了展示层次，缩写了两串机器生成的字符串。

```json
{
  "messages": [
    {"role": "system", "content": "Transcribe new speech in its original language. Begin new text with <|DIRECT|>. Emit <|WAIT|> if nothing is ready. Use digits for numbers."},
    {"role": "user", "content": "<audio>"}
  ],
  "audios": ["omnicommitaudio:v1:…"],
  "asr_task": "{\"id\":\"demo-001\",…}"
}
```

`audios[0]` 是**第一块音频**的虚拟地址，解码后包含原 WAV 路径、`start=0.0`、`end=1.0`、`input_seconds=1.0` 和文件校验信息。程序按需读取这段波形，不需要为每秒另存 WAV。

`asr_task` 在文件里是 JSON 字符串。调用 `json.loads(row["asr_task"])` 后，下例展示解码内容。原路径按 Linux 示例目录书写，文件大小和修改时间由准备器从实际文件读取。

```json
{
  "id": "demo-001",
  "audio_path": "/data/asr/demo.wav",
  "audio_size_bytes": 88044,
  "audio_mtime_ns": 1794999785078068400,
  "source_ids": ["/data/asr/demo.wav", "meeting:meeting-01"],
  "bucket": "switch",
  "duration_seconds": 2.75,
  "chunks": [
    {"start_seconds": 0.0, "end_seconds": 1.0, "input_seconds": 1.0},
    {"start_seconds": 1.0, "end_seconds": 2.0, "input_seconds": 1.0},
    {"start_seconds": 2.0, "end_seconds": 2.75, "input_seconds": 0.75}
  ],
  "reference_text": "好了。OK. Next.",
  "spans": [
    {"text": "好了。", "lang": "zh"},
    {"text": "OK. Next.", "lang": "en"}
  ],
  "alignment": [
    {"text": "好", "start_ms": 100, "end_ms": 400, "char_start": 0, "char_end": 1},
    {"text": "了", "start_ms": 410, "end_ms": 620, "char_start": 1, "char_end": 2},
    {"text": "OK", "start_ms": 680, "end_ms": 950, "char_start": 3, "char_end": 5},
    {"text": "Next", "start_ms": 2100, "end_ms": 2530, "char_start": 7, "char_end": 11}
  ],
  "switches": [
    {"char_start": 3, "time_ms": 680.0, "lang": "en", "speaker": "spk2"}
  ]
}
```

`chunks` 描述新音频的到达顺序。`reference_text` 保留说出的原语言，`spans` 记录连续语言段，`switches` 在有切换时间标注时保留实际切点。字符偏移计入标点和空格，左闭右开。准备器会替换原来的离线提示词；模型的首轮输入只有上面的短 system 和第一块音频。训练时模型接触不到评分用的整段参考、语言段和时间戳。

## 3. 划分输出 `split/train.jsonl` 和 `split/dev.jsonl`

运行 `split_data.py` 后，训练集和评测集**逐行沿用上述任务格式**。`split.json` 保存划分统计。同一条任务进入其中一个集合；同一来源的关联样本放在一起。把 `split/train.jsonl` 交给 `02_train.sh`，把 `split/dev.jsonl` 交给 `evaluate.py model`。批量划分需要有足够多的独立来源，单条演示样本可直接用来检查准备结果。

## 4. 在线训练才出现每一轮的动作

调度器追加新块，保留该候选已经听到的音频和自己之前的回复。示例里第 1 块的 680 毫秒处同时换人、换语种，`OK` 在 950 毫秒已经说完。下面是一条可以得到满分的候选，第二块是停顿。

```text
本轮新输入                 assistant 的完整回复
0.00–1.00 秒              <|DIRECT|>好了。OK.
1.00–2.00 秒              <|WAIT|>
2.00–2.75 秒              <|DIRECT|> Next.
<end_of_audio>            <|WAIT|>
```

第一轮可同时提交两种语言。第二轮等待不会追加正文。最后是文本结束事件，不增加音频。把三轮音频里的新增正文串起来是 `好了。OK. Next.`。训练默认对同一录音在线采样 8 条候选，分别评分；磁盘里的训练任务仍然只有那一行。

## 5. 训练日志和评测文件

训练目录的 `rewards/rollouts-rank*-pid*.jsonl` 中，每一行是一条候选的整段轨迹。下面摘出该行最常用的字段，实际还包含插入、删除、替换、时延覆盖等指标。

```json
{
  "task_id": "demo-001",
  "mode": "quality_latency",
  "turns": [
    {"kind": "audio", "audio_end_seconds": 1.0, "response": "<|DIRECT|>好了。OK.", "finish_reason": "stop"},
    {"kind": "audio", "audio_end_seconds": 2.0, "response": "<|WAIT|>", "finish_reason": "stop"},
    {"kind": "audio", "audio_end_seconds": 2.75, "response": "<|DIRECT|> Next.", "finish_reason": "stop"},
    {"kind": "eof", "audio_end_seconds": 2.75, "response": "<|WAIT|>", "finish_reason": "stop"}
  ],
  "valid": true,
  "reward": 1.0
}
```

对上述手写候选运行项目评分器得到下表。三行共用同一份参考与时间戳。

| 候选的提交方式 | quality | quality_latency |
|---|---:|---:|
| 首轮提交“好了。OK.”，末块提交“ Next.” | 1.000 | 1.000 |
| 首轮只出“好了。”，把 OK 和 Next 都放到末块 | 1.000 | 0.989 |
| 首轮出“好了。”，末块出“ Next.”，漏掉 OK | 0.550 | 0.513 |

单独运行 `evaluate.py` 时，每行结果的外层字段为 `id`、`bucket`、`mode`、`turns`、`score`，奖励在 `score.reward`。同目录的 `*.summary.json` 汇总全体及各类样本的指标。

在服务器查看第一条真实任务，可以从 `ASR-GSPO/` 执行：

```bash
python3 - <<'PY'
import json
from pprint import pprint

with open("../datasets/gspo01/tasks.jsonl", encoding="utf-8") as f:
    row = json.loads(next(f))
pprint(row["messages"], sort_dicts=False)
pprint(json.loads(row["asr_task"]), sort_dicts=False)
PY
```
