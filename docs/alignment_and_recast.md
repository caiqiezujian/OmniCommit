# 从文本标注到固定音频轮次

此路径接收 `Data-Scripts/annotate_asr.py` 的新版本输出：原 JSONL 字段保留，`output` 是包含 `segments/source_text/action`、`flush`、`error` 的 JSON 字符串。文本教师只调用一次。后续步骤不再调用文本教师，不改写正文。

## 1. 获取真实声学时间戳

需要词级或字级的音频起止时间。不能根据字数、词数或标点将总时长均分。文本单元内部没有足够细的时间戳时，程序使用覆盖它的完整声学词区间，宁可稍晚提交，不猜测子词时间。

已有可靠时间戳时，直接准备如下 JSONL；`id` 对应原记录的 `id/key`，没有 ID 时使用完全相同的 `audio` 路径关联。最好两者都填，程序会交叉检查。

```json
{"id":"meeting-001","audio":"/data/audio/meeting.wav","duration_ms":2000,"words":[{"text":"请联系","start_ms":20,"end_ms":200},{"text":"张伟","start_ms":250,"end_ms":900},{"text":"确认方案","start_ms":1100,"end_ms":1800}]}
```

也支持 `chars` 字符数组，以及 WhisperX 的 `word_segments:[{word,start,end}]` 秒单位格式。所有词或字必须按转录顺序完整覆盖正文，不能漏掉数字、外语、人名。这里只忽略普通标点、空白和大小写差别进行对应，保留数值内部的分隔符；不会模糊匹配错误转录或猜测缺失时间。普通时间戳存在误差，程序检查不能代替听音抽检。

如果目前只有音频和转录，可以在独立环境中运行真正的强制对齐模型。**不要将对齐依赖安装进已有 `omni_ms` 训练环境。**

```bash
# 独立 Python 3.11 环境；GPU torch 按服务器 CUDA 安装。
python3.11 -m venv /envs/omnicommit-align-qwen
source /envs/omnicommit-align-qwen/bin/activate
python -m pip install -r requirements-alignment-qwen3.txt

python Data-Scripts/force_align_asr.py /data/annotated.jsonl \
  --backend qwen3 --qwen-model Qwen/Qwen3-ForcedAligner-0.6B \
  --device cuda:0 --dtype bfloat16 --output /data/timestamps.jsonl
```

Qwen3 adapter 使用已知转录做强制对齐，不让 ASR 重新生成正文。当前配置覆盖中文、英文、日文、韩文、俄文、西班牙文、葡萄牙文、法文、德文、意大利文的单语记录。模型给出的词表会去掉部分标点；adapter 精确对应模型输入的已知文字并恢复原始数字/标点，绝不猜测替换。上游代码包含时间异常插值，本 adapter 关闭该修复，遇到非单调时间预测拒绝该行。

阿拉伯语可使用另一个独立环境里的 WhisperX：

```bash
python3.11 -m venv /envs/omnicommit-align-whisperx
source /envs/omnicommit-align-whisperx/bin/activate
python -m pip install -r requirements-alignment.txt

python Data-Scripts/force_align_asr.py /data/ar.annotated.jsonl \
  --backend whisperx --device cuda --output /data/ar.timestamps.jsonl
```

WhisperX 默认模型涵盖本项目所需的十一种语言，缺少泰语默认模型；泰语需提供经验证的兼容 Wav2Vec2 CTC 模型 `--align-model /models/th-ctc`，或使用外部声学时间戳。WhisperX adapter 只取字符真实对齐结果，不使用缺字的词级时间插值。它会拒绝声学词表未覆盖的数字/字母/变音符号，不把失败样本当成有效对齐。

**中英混合等跨语种记录目前必须输入可靠的外部字词时间戳。** 脚本不会选一个语言模型对全条混合记录冒充已验证的多语言对齐。已知跨语种切换文本并不能提供切换时刻。可采用支持相应语种的外部对齐器，或先获得真实音频语言区间再分别对齐并汇总为上述通用格式。

默认单条强制对齐上限 60 秒，避免未经切分的长会议使动态规划/模型上下文耗尽；`--max-audio-seconds` 可显式调整，但这不是已验证任意长度。现有长会议应使用具有真实声学边界的子段进行离线对齐。`--resume` 跳过已经写入的成功和失败行；换模型/参数需要新输出路径。

## 2. 将时间戳关联到语义单元

```bash
python Data-Scripts/align_asr.py /data/annotated.jsonl \
  --timestamps /data/timestamps.jsonl --output /data/aligned.jsonl
```

输出包含每个完整 `source_text` 的 `start_ms/end_ms` 和真实语言/说话人归属。实体、数字由文本阶段定义为整体；程序按其全部字词的起止区间取并集。重复词按出现顺序逐个对应，正文与参考音频标识不一致时拒绝，不能用错误的重音位置补齐。

原始标签只在首处或真实切换处保留；读取状态与提交状态分开。相同标签的重复标注不会产生切换。跨语言仍可保留同一 `spk`。对齐时间允许顺序一致的区间重叠，但不支持无法按转录顺序排序的复杂重叠发言；此类样本需要独立审查，不能宣称单流线性标注能完整表示。

## 3. 固定 0.5 秒或 1 秒重构动作

```bash
python Data-Scripts/recast_asr.py /data/aligned.jsonl \
  --chunk-seconds 0.5 --number-format normalized \
  --output /data/trajectories.500ms.jsonl

# 若选择1秒，单独构建另一份；训练和推理应使用同样的步长。
python Data-Scripts/recast_asr.py /data/aligned.jsonl \
  --chunk-seconds 1.0 --number-format normalized \
  --output /data/trajectories.1000ms.jsonl
```

`--number-format` 必须与原始转录相符，只影响训练指令，不会在此阶段归一化数值。原数据是口语数词时使用 `verbatim`。

重构规则：

1. 完整语义单元的结束时间不晚于当前音频末尾，才有资格提交；实体听到一半时不能输出前半实体。
2. 当前轮首次遇到已完成的 `WAIT` 单元，整轮输出 `WAIT`。即使该轮还覆盖后面的 `DIRECT`，硬等待仍保留。
3. 该硬等待事件只消费一次。后续没有新硬等待的轮次，提交此前全部未提交的完整内容；连续普通单元可以连续 `DIRECT`。
4. 每个 `DIRECT` 含完整语言和说话人前缀；同一次提交中的真实归属变化另起完整前缀。
5. 最后增加显式无音频事件 `<end_of_audio>`。它不是新音频或假静音：如果仍有挂起内容则 `DIRECT` 释放，否则 `WAIT`。因此文本标注 `flush:false` 仍可能需要真实 EOF 释放。

例如 0–0.5 秒内同时完成“请联系”(DIRECT)、“张伟”(WAIT)、“确认”(DIRECT)，这一物理轮为 `WAIT`；下一轮可提交“请联系张伟确认”。如果音频就在 0.5 秒结束，释放发生在 `<end_of_audio>`，不再伪造后续 0.5 秒声音。

输出 `schema_version:2`，普通轮 `event:audio`，最后一轮 `event:end_of_audio`。`safe_cut` 表示既没有完整但未提交内容，也没有已听到一部分的原子单元，供长 SFT 样本拆窗使用。仅凭 `DIRECT` 不能判断可安全裁剪：它可能只提交了前缀，后面的姓名尚未听完整。

长会议中的纯静音窗口可以保留为连续 WAIT 监督，但必须来自已经通过检查的、有转录的完整会议；窗口两端安全且不与任何已对齐语音区间重叠。准备器保留的 `source_speech_spans` 只用于离线验证，不传入模型。任意失败识别或缺少这些证据的全 WAIT 样本仍被拒绝。

独立训练窗口还会携带此前最后一次 DIRECT 的最后一段语言/说话人状态，使新窗口能区分延续发言与真实切换。这是“最后已提交状态”，不能冒充最新听到但尚未提交的声音归属；WAIT 和参考音频都不会更新它。训练和流式推理使用同一个状态提示函数。说话人参考音频默认合计不超过 30 秒，超出预算拒绝该会议拆窗，避免静默忘记已知身份。

每条失败记录写 `status:error, output:""`，另写 `.rejected.jsonl`，继续下一条；启动参数、文件权限、恢复配置不一致等文件级问题仍明确报错，防止污染已有输出。无需备份，不会修改源文件。

## 验证范围与来源

本地测试覆盖真实 WAV 长度、字词时间导入、数字/实体跨块、同块 WAIT/DIRECT 冲突、语言/说话人变化、连续 DIRECT、EOS、重复词、全部十二语言的外部格式、坏行继续和恢复。模型适配器测试使用 mock，**不代表已完成真实 GPU 对齐质量验收**。需要在目标 GPU 上听音抽查数字、人名、切换点与重叠声音。

适配器接口核对自 [Qwen3 强制对齐源码](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/inference/qwen3_forced_aligner.py) 和 [WhisperX 3.7.4 对齐源码](https://github.com/m-bain/whisperX/blob/v3.7.4/whisperx/alignment.py)。
