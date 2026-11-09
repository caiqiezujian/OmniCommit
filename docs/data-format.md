# ASR 数据格式

当前主线从已有语音和完整标注转录出发：一次文本大模型切分与初步动作标注，取得真实时间戳，再按 0.5 秒或 1 秒重铸，最终生成音频与文本交错的 SFT 数据。本文中的路径、语句和音频引用均是示例；JSONL 中每行必须是一个完整 JSON 对象，下面为便于阅读使用多行排版。

## 1. 输入：音频和完整转录

```json
{
  "audios": ["/data/audio/example.wav"],
  "messages": [
    {"role":"system","content":"Transcribe speech."},
    {"role":"user","content":"<audio>**Entity Reference:** []"},
    {"role":"assistant","content":"<zh><spk1>请联系张伟确认方案。"}
  ]
}
```

每条仅一个音频文件，assistant 包含完整原语言转录及语种、说话人标签。同一归属的重复前缀可以存在，模型输入时会被去重；真实语种或说话人变化仍保留。不能据空标签猜说话人，例如无正文的连续 `<spk2><spk3>` 需要核查。`id` 或 `key` 可选；不应认为同名 spk 编号在不同录音之间指同一人。

## 2. 单次文本模型标注

`annotate_asr.py` 保留输入字段，成功时在 `output` 写入 **JSON 字符串**；下面是解析后的内容：

```json
{
  "segments": [
    {"source_text":"<zh><spk1>请联系","action":"DIRECT"},
    {"source_text":"张伟","action":"WAIT"},
    {"source_text":"<zh><spk1>确认方案。","action":"DIRECT"}
  ],
  "flush":false,
  "error":null
}
```

`source_text` 是本片原文：即使 `action` 是 WAIT，也必须保留文字，供后面时间对齐和合并；它不是最终 assistant 输出。每个 DIRECT 带完整语种与说话人前缀，WAIT 可继承上一归属；切换归属需标 WAIT。去除冗余前缀后，各片正文及实际归属须与原转录一致。相同前缀重复不表示发生切换。普通片段可连续 DIRECT；模型可能漏标实体或错误切分，结构校验不能证明语义标注正确。

单条请求或校验失败通常写 `output:""` 并继续；过滤器按代码中写死的输入输出路径运行，文件级错误与单行拒绝区别说明见 [Data-Scripts 说明](../Data-Scripts/README.md)。

## 3. 已有真实时间戳的输入

一体化入口要求 `audios`、`messages`、`output` 和 `timestamps` 在同一 JSONL 行。以下时间仅示意：

```json
{
  "audios":["/data/audio/example.wav"],
  "messages":[{"role":"assistant","content":"<zh><spk1>请联系张伟确认方案。"}],
  "output":"{\"segments\":[{\"source_text\":\"<zh><spk1>请联系\",\"action\":\"DIRECT\"},{\"source_text\":\"张伟\",\"action\":\"WAIT\"},{\"source_text\":\"<zh><spk1>确认方案。\",\"action\":\"DIRECT\"}],\"flush\":false,\"error\":null}",
  "timestamps":[
    {"index":1,"source_text":"<zh><spk1>请联系","start_ms":100,"end_ms":300,"text_plain":"请联系","language":"zh","speaker":"spk1"},
    {"index":2,"source_text":"张伟","start_ms":400,"end_ms":850,"text_plain":"张伟","language":"zh","speaker":"spk1"},
    {"index":3,"source_text":"<zh><spk1>确认方案。","start_ms":1100,"end_ms":1800,"text_plain":"确认方案。","language":"zh","speaker":"spk1"}
  ],
  "tags":["<|DIRECT|>","<|WAIT|>","<|DIRECT|>"]
}
```

时间戳相对当前音频起点，单位为毫秒，须与输出片段逐项对应；有 `index` 时要唯一且完整覆盖 1 到 N。`source_text`、`text_plain`、`text`、`language` 和 `speaker` 等若提供，须与标注对应。`tags` 可省略；提供时须与每段 action 一致。片段内部有归属变化时，必须用 `spans` 给出各归属片段的正文、语种、说话人和对应 `start_ms/end_ms`，不能由字符长度猜时间。输入的 `output` 也可为已解析的对象。

时间必须来自音频对齐，不能按字数均分。真实音频路径需在构建和训练机器上可读；程序从文件本身获取时长，而不是用最后一个时间戳代替。详细格式和命令见 [已有时间戳直接构建 SFT](timestamped_to_sft_zh.md)。

## 4. 重铸后：精简 SFT 数据

`Data-Scripts/build_asr_sft.py` 的成功行只有 `audios` 和 `messages`；假定上面的音频共 2 秒，下面示意 0.5 秒的结构，不可复制占位音频引用直接训练：

```json
{
  "audios":["虚拟引用1","虚拟引用2","虚拟引用3","虚拟引用4"],
  "messages":[
    {"role":"system","content":"实际使用的增量识别协议"},
    {"role":"user","content":"<audio>"},
    {"role":"assistant","content":"<|DIRECT|><zh><spk1>请联系"},
    {"role":"user","content":"<audio>"},
    {"role":"assistant","content":"<|WAIT|>"},
    {"role":"user","content":"<audio>"},
    {"role":"assistant","content":"<|WAIT|>"},
    {"role":"user","content":"<audio>"},
    {"role":"assistant","content":"<|DIRECT|><zh><spk1>张伟确认方案。"},
    {"role":"user","content":"<end_of_audio>"},
    {"role":"assistant","content":"<|WAIT|>"}
  ]
}
```

每个 `<audio>` 按顺序对应 `audios` 里的一个 `omnicommitaudio:v1:` 虚拟引用。引用封装原文件路径、起止时间和输入时长；原音频地址被编码在引用里，不创建小 WAV。训练必须使用 OmniCommit 音频插件，且训练机能通过相同绝对路径读取原音频。模型只输出新增文本；WAIT 不输出文字，后续满足条件的 DIRECT 一起释放待提交内容，每次 DIRECT 以完整语言和说话人前缀开始，内部归属变化时再次添加。最后的 `<end_of_audio>` 是文本事件而非音频或第三个动作；它释放尾部剩余的已完整内容，没有剩余时为 WAIT。

真实切换在带时间的变化起点所在块触发 WAIT；其他标注 WAIT 在其片段完成所在块触发。WAIT 事件同块覆盖后续 DIRECT 时整块仍 WAIT；之后等后续源 DIRECT 单元完整且本块无新等待事件才释放。由于标注不含等待原因，同一 WAIT 同时涉及切换和实体时只按切换触发，不能保证额外的实体等待。切换时间是归属文本对齐起点的近似，不保证等同声学切换点。

构建程序输出训练文件及三个旁文件：`.tokens.txt` 是训练前注册的有序 token 清单；`.rejected.jsonl` 保存失败行号与原因；`.review.jsonl` 列出至少 2 秒的未标注区间。未标注不等于静音：默认保留，若包含未转录语音，则这些区间形成错误 WAIT 监督；可选择阈值拒绝。`--number-format` 只选择数字格式指令，不改写原文。坏行被逐条拒绝并继续，全部无有效行返回非零；输出存在则不覆盖，当前一体化入口不支持中断后自动续跑。

## 5. 检查与其他路线

`audit_sft.py` 检查而不修改或过滤训练数据，`--decode` 会逐块读音频；`audit_encoded_lengths.py` 使用真实训练模板和 HF 处理器检查 4096 长度。两者不能证明转录、说话人或对齐结果声学正确。精简训练行没有安全拆窗元数据，不能直接交给 `prepare_windows.py`；需要长会话拆窗时须从保留的原标注与时间戳重建适用的富元数据流程。实际 GPU 训练和实时效果须另行验收。

此前的**音频教师累计前缀路线**是独立的可选研究路线，不是上述文本标注和时间戳路线的必要步骤；其 `schema_version:1` 教师轨迹、富元数据 SFT 和新入口的精简输出不应混淆。见 [数据脚本说明](../Data-Scripts/README.md) 和 [方法说明](method.md)。
