# 数据格式 v1

## 1. 原始输入

一行一个会话，当前支持一个本地音频和一个完整 assistant 转录。`id`/`key`可用于追踪。没有显式 ID 时 collector 生成与输入位置绑定的标识，因此恢复时必须保持文件和行序一致。

```json
{
  "id": "example-001",
  "audios": ["/data/audio/example.wav"],
  "messages": [
    {"role": "system", "content": "Transcribe speech."},
    {"role": "user", "content": "<audio>**Entity Reference:** []"},
    {"role": "assistant", "content": "<zh><spk1>我同意。<en><spk1>Any questions?<zh><spk2>没有。"}
  ]
}
```

支持完整重复的标签头和只在变化时插入的标签，但一段文字开始前必须已知语言和说话人。`<spk2><spk3>呃`这样的连续空speaker段无法可靠解析；默认记录参考解析错误，保留原文，让教师继续独立识别音频，不猜哪个人说了“呃”。`--strict-reference` 可改为拒绝这类记录。无法解析参考时一致性审计不可用；若指定正的 `--min-reference-agreement`，轨迹阶段会拒绝该条，不能绕过质量筛选。源messages仅是读取原始标注的容器，不作为教师调用的messages直接转发。

## 2. 教师响应与回放

每次教师必须只返回如下JSON对象，不返回分析、动作或代码围栏。音频无可辨识语音时允许 `segments: []`。

```json
{"segments":[{"language":"zh","speaker":"spk1","text":"我同意。"}]}
```

language使用小写语言代码，例如zh/en/ja；speaker按当前累计音频的首次出现顺序标spk1、spk2。不同语种不代表不同说话人。同一语言、说话人的相邻段在比较时合并，避免分段差异导致假不稳定。数字模式由collector选择。

离线replay文件每行对应一个原始输入ID，次序与输入一致。时间单位是相对原音频起点的毫秒；省略末端prefix即可，collector复用full作为末端结果。

```json
{
  "id": "example-001",
  "full": {"segments": [{"language":"zh","speaker":"spk1","text":"我同意。"}]},
  "prefixes": [
    {"end_ms":500,"segments":[]},
    {"end_ms":1000,"segments":[{"language":"zh","speaker":"spk1","text":"我同意。"}]}
  ]
}
```

此例仅展示形状，实际端点必须和真实音频长度、chunk设置一致。replay不生成声学事实；它用于测试或导入已经采集的教师预测。

## 3. 稳定提交轨迹

```json
{
  "schema_version": 1,
  "id": "example-001",
  "audio": "/data/audio/example.wav",
  "duration_ms": 1500,
  "policy": {"number_format": "verbatim"},
  "turns": [
    {"start_ms":0,"end_ms":500,"action":"WAIT","segments":[]},
    {"start_ms":500,"end_ms":1000,"action":"DIRECT","segments":[{"language":"zh","speaker":"spk1","text":"我同意。"}]},
    {"start_ms":1000,"end_ms":1500,"action":"WAIT","segments":[]}
  ]
}
```

字段约束：

- `turns`从0开始连续、严格递增，最后覆盖整个音频。这里是新增音频区间；collector的识别输入则是累计区间。
- `DIRECT`必须有非空新增文本段；每个段包含language/speaker/text。
- `WAIT`的segments必须为空，不预先发出空的语言或说话人标签。
- `text`不含动作标签或尖括号；语言/说话人不在text字段重复。
- 单条会话允许最后WAIT，不能复用翻译流程“末项不得WAIT”的约束。
- `status:error`及`output:""`表示整条构建失败，与合法的WAIT回合不同。
- 记录还含输入/配置指纹、参考审查、教师来源等诊断信息；具体生成内容以脚本为准。

## 4. SFT 数据

`prepare_sft.py` 将轨迹转为以下结构，区间引用由脚本生成，不能手写不完整base64：

```text
audios: [虚拟音频区间引用1, 引用2, 引用3]
messages:
  system: 本会话数字策略及转录协议
  user: <audio>
  assistant: <|WAIT|>
  user: <audio>
  assistant: <|DIRECT|><zh><spk1>我同意。
  user: <audio>
  assistant: <|WAIT|>
```

一个user音频占位符对应一个audios条目。DIRECT中的多个段以换行分隔，每段完整写出语言和说话人。使用原生SWIFT多轮assistant监督；系统指令和音频输入本身不作为要模仿的assistant答案。

同时保存可读的`audio_segments`、`audio_chunk_durations`和`turn_provenance`便于审查。虚拟引用仅适用于本项目插件，不能直接交给普通音频读取器。token inventory由准备器写到`TRAIN.jsonl.tokens.txt`，需随数据、适配器保存。

## 运行与数据质量

准备脚本做结构和音频范围校验，不重新识别音频、不替用户修改转录，也不能证明说话人标签正确。训练前应检查拒绝日志、参考差异、极长WAIT、全句最后集中提交、早期数字提交和切换点准确率；这些是质量诊断，不是整个云任务遇到一条坏样本就退出的理由。

真实音频路径、转录、教师trace和训练文件默认被`.gitignore`排除，不会随代码推送。
