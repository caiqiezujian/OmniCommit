# Fourth GSPO：一条录音、一条任务、在线八条轨迹

## 1. 原始输入

使用用户现有的 `audios + messages + output + timestamps + tags` 文件即可。完整识别参考只取 messages 中的 assistant.content，不重新拼接 output.segments，不受旧 WAIT/DIRECT 策略约束。旧段落重分配空格不能改写完整参考正文。

以用户提供的俄中样例为例，无标签正文是：

~~~text
Ты что лыбишься?  Господи! Как же хорошо. На матрасе.  Значит, всё дело в матрасе?  Юль, делай что хочешь, я на диван больше не вернусь.150克之间，诸如矮萌以其敏捷的动作和团队合作的生活方式而闻名。
~~~

原 `<ru>`、`<zh>` 用于评分侧语言区间；原 `<spk1>` 等用于评分侧真正的换人位置。重复的同人/同语种标签不生成新边界。Fourth 输出仍只有 DIRECT / WAIT / SPK_CHANGE，不要求输出语言标签和持久说话人编号。旧 WAIT 段中的“Господи!”、“Значит,”、“150克之间，”照样保留在完整参考里，不能丢弃。

下列缩短例子用来说明字段；假设 WAV 总长 3 秒，JSONL 实际每行一个对象：

~~~json
{
  "id":"example-1",
  "audios":["/data/asr/example.wav"],
  "messages":[
    {"role":"user","content":"<audio>"},
    {"role":"assistant","content":"<zh><spk1>开始。<en><spk2>OK."}
  ],
  "output":{"segments":[
    {"source_text":"<zh><spk1>开始。","action":"WAIT"},
    {"source_text":"<en><spk2>OK.","action":"DIRECT"}
  ]},
  "timestamps":[
    {"start_ms":100,"end_ms":800,"source_text":"<zh><spk1>开始。"},
    {"start_ms":1000,"end_ms":1600,"source_text":"<en><spk2>OK."}
  ]
}
~~~

output 可以为 JSON 字符串，像用户样例一样保留旧 action 字段。末尾过滤仅核对相关尾段及其最后 end，不要求整份旧分段再次严格对齐。所有时间相对当前 WAV。

本次默认 `--timing-mode tail-only` 不把中间 timestamps/CTC 转为逐词延迟目标。若后续明确使用可信全量对齐，可选择 `--timing-mode aligned`；该路径仍支持 alignment、chars、words、word_segments，并保留严格内容与偏移核验。

有末尾标点时需要可信的最终 end；仅有切换起点 time_ms 不足以筛选。真实末尾没有标点时，不因 end 缺失或间隔不足一秒而排除，但必须满足正文、音频及末尾一致性等基本校验。

## 2. 先过滤，再准备为任务

~~~bash
python3 filter_tail.py source.jsonl --output filtered.jsonl
bash 01_prepare.sh filtered.jsonl --output tasks.jsonl --chunk-ms 1000 --tail-policy keep --timing-mode tail-only --rejects rejected.jsonl
~~~

上例末尾是 `OK.`，最后 DIRECT 有句末标点，尾部差 1400 ms，因此整行保留。若 end 为 2300 ms，差 700 ms，则整行过滤。如果最后 DIRECT 为没有标点的 `OK`，则差 700 ms 也保留；完整参考若仍是 `OK.`，正文不改写，仅设 `terminal_punctuation_trusted:false`，不以这个不一致的末尾句号约束模型。句内标点继续评分。下面是保留记录的部分任务字段：

~~~json
{
  "protocol":"fourth_speaker_v1",
  "source_duration_seconds":3.0,
  "duration_seconds":3.0,
  "end_policy":"eof",
  "tail_policy":"keep",
  "timing_mode":"tail-only",
  "terminal_period_removed":false,
  "reference_text":"开始。OK.",
  "speaker_scorable":true,
  "speaker_changes":[3],
  "spans":[{"lang":"zh","text":"开始。"},{"lang":"en","text":"OK."}],
  "alignment":[],
  "chunks":[
    {"start_seconds":0.0,"end_seconds":1.0,"input_seconds":1.0},
    {"start_seconds":1.0,"end_seconds":2.0,"input_seconds":1.0},
    {"start_seconds":2.0,"end_seconds":3.0,"input_seconds":1.0}
  ]
}
~~~

实际还保留原音频路径、文件身份、完整归属区间等。tail-only 不向奖励器提供中间语音时间戳，也不提供 switches 时间条目。原 WAV 不修改、不另存切片；全部 3 秒继续作为模型输入，参考原文不改动。末尾间隔不是 VAD 验证的静音；有漏标音频时仍需改善源标注。

磁盘训练行外层为：

~~~json
{
  "messages":[
    {"role":"system","content":"Transcribe new speech. Use <|DIRECT|> for text, <|WAIT|> to wait, and insert <|SPK_CHANGE|> when the speaker changes. Use digits for numbers."},
    {"role":"user","content":"<audio>"}
  ],
  "audios":["omnicommitaudio:v1:机器生成的第一块引用"],
  "asr_task":"评分参考和音频调度计划的JSON字符串"
}
~~~

actor 不接触 asr_task 的参考、语种、说话人和未来时间戳；调度器每轮追加新音频。原始离线长 system 被替换为与 Fourth SFT 一致的短协议。

## 3. 在线轨迹

同一任务在线采样八条完整候选，各自根据已经输出的历史继续。以下为一条合规候选：

~~~text
新增音频 0.0～1.0 秒： <|DIRECT|>开始。
新增音频 1.0～2.0 秒： <|DIRECT|><|SPK_CHANGE|>OK.
新增音频 2.0～3.0 秒： <|WAIT|>
用户 <end_of_audio>：  <|WAIT|>
~~~

别的候选可能晚输出、漏英文、误标换人或多句号。奖励可核对完整内容、语言和边界；本次 tail-only 没有精确时间监督，不能仅因输出较晚就按未知的词结束时间扣分。没有离线写死八条答案。最后音频轮仍由模型自由生成，不会强制用参考替换。

keep 样本另有用户 <end_of_audio> 事件，对应 WAIT。既有 short-open 等实验产生的 annotation_end 样本仍在最后音频轮结束，不追加空轮；本次整行过滤后 keep 流程不使用这种尾部投影。

## 4. 日志

rewards/rollouts-*.jsonl 每行一条候选。示例：

~~~json
{
  "task_id":"example-1",
  "mode":"quality_latency",
  "turns":[
    {"kind":"audio","audio_end_seconds":1.0,"response":"<|DIRECT|>开始。","finish_reason":"stop"},
    {"kind":"audio","audio_end_seconds":2.0,"response":"<|DIRECT|><|SPK_CHANGE|>OK.","finish_reason":"stop"},
    {"kind":"audio","audio_end_seconds":3.0,"response":"<|WAIT|>","finish_reason":"stop"},
    {"kind":"eof","audio_end_seconds":3.0,"response":"<|WAIT|>","finish_reason":"stop"}
  ],
  "valid":true,
  "reward_version":"fourth_boundary_v2",
  "switch_grace_turns":1,
  "reward":1.0,
  "cer":0.0,
  "timed_units":0,
  "latency_penalty":0.0,
  "speaker_correct_changes":1,
  "extra_internal_periods":0
}
~~~

evaluate.py 输出逐条评分与 summary，覆盖 CER、切换漏段、换人错位/归错字符和额外断句。tail-only 中计时单元为 0、延迟罚分为 0、延迟百分位未知，不能解释为真实零延迟。以后使用 aligned 评测时，延迟百分位只统计拼写正确且未检测到归属错误的可计时单元，要同时看删除与覆盖。确认宽限不写入 actor 输入，不预设 WAIT 答案。

整份文件作为训练，不调用 split_data.py；保留该独立工具供明确需要按来源分组划分的其他实验使用。
