# 已有时间戳：重铸并直接生成 SFT

本入口处理每行同时包含 `audios`、原始 `messages`、文本模型 `output` 和 `timestamps` 的 JSONL。它不调用大模型，也不再运行强制对齐：读取真实音频长度，校验转录与时间戳，重铸 0.5 秒或 1 秒轮次，直接输出只含 `audios`、`messages` 的训练 JSONL。所有命令在 Linux/云道服务器的项目根目录运行，示例路径需替换为实际挂载路径。

## 1. 检查输入

- 必须是 **JSONL**：每行一个完整 JSON 对象，不是跨多行排版的单个 JSON。`output` 可以是 JSON 字符串或对象，内部为 `segments`、`flush`、`error`；每个单元的 `source_text` 和 `action` 必须与原始转录相符。
- `timestamps` 数量必须与 `segments` 相同，按单元逐项对应；含 `index` 时必须完整、唯一覆盖从 1 开始的编号。每项使用相对于这条原始音频起点的毫秒数 `start_ms/end_ms`。若有 `source_text`、`text_plain`、`text`、`language`、`speaker` 等字段，内容和归属须一致。可选的顶层 `tags` 必须与动作逐项一致。
- 单元内部有语言或说话人变化时，须用 `spans` 提供变化前后的实际起止时间、正文、`language` 和 `speaker`；没有内部变化时可不填 `spans`。这些时间必须来自音频对齐，不能由文字长度均分或猜测。标签前缀在相同归属下重复不会被视为切换。
- 每条音频必须在运行机器上真实存在且可读取。脚本从音频文件本身读取时长，不以最后一个文本时间戳当成录音结束。相对路径默认基于输入文件目录，或通过 `--audio-root` 指定基目录；输出虚拟引用保存绝对路径，训练时必须能从同一路径访问。该数据是一条录音一个样本，不会自动产生说话人参考缓存。
- 检查源文本确实覆盖音频中的发言、数字、实体、语种及说话人，尤其检查切换点和重叠发言。字段一致或对齐成功不证明声学结果正确。复杂重叠发言不应直接视为已经解决。

## 2. 一次构建

在已有 CPU 数据处理环境已满足依赖时，不需要安装。若缺少依赖，请在独立的数据处理环境执行 `python -m pip install -r requirements-data.txt`；不要随意更改已有 `omni_ms` 训练容器的依赖。

```bash
cd /workspace/OmniCommit
mkdir -p /data/OmniCommit/run-001

python Data-Scripts/build_asr_sft.py /data/OmniCommit/timestamped.jsonl \
  --output /data/OmniCommit/run-001/train.500ms.jsonl \
  --chunk-seconds 0.5 --number-format normalized
```

如原始转录保留口语数词，改用 `--number-format verbatim`。该参数只改变模型系统指令，**不会改写原文或替你把文字数词转换成数字**。若希望实验 1 秒，请选择新输出路径，将 `--chunk-seconds` 改成 `1.0`。最后不足完整一块时按真实剩余音频处理，并在读取时对最后短块补零到至少 0.5 秒；不会读取录音之后不存在的语音。

输出及伴随文件：

- `train.500ms.jsonl`：成功行，每行只有 `audios` 和 `messages`；`audios` 是指向原文件范围的虚拟音频引用，不复制小 WAV。
- `train.500ms.jsonl.tokens.txt`：从成功样本收集的动作、语种和说话人 token 清单；SFT 前使用这份清单注册 token。需要合并多个分片时，须先建立一致且覆盖训练、验证数据的清单，再进行全量检查；恢复训练时不能改变已有 token 顺序。
- `train.500ms.jsonl.rejected.jsonl`：逐行拒绝记录，含原输入行号、可用 ID 和错误原因。它不是原始样本副本，按行号回到输入排查。单行解析、时间、内容或音频错误会拒绝该行并继续；控制台统计读入、保留、拒绝数，零成功时返回非零。
- `train.500ms.jsonl.review.jsonl`：记录每段至少 2 秒的未标注区间，含输入行号、音频路径和起止毫秒。它用于听音审查，不等于已证实静音，也包含可选阈值拒绝的记录。默认不会因间隙过长自动拒绝；这类区间没有新增文本时会形成 WAIT 监督，若实际上含未转录语音就是错误标签。短于 2 秒的遗漏也可能存在，仍需抽检。

可选地使用 `--max-unannotated-gap-seconds 2`，对任意 **超过** 2 秒的未标注区间拒绝该条；这是保守过滤，不是静音识别，也可能排除真实长停顿。日志中的间隙统计只识别 ≥2 秒区间。输出和上述三个伴随文件若已存在，程序会拒绝覆盖；使用新路径重新运行。没有自动断点续跑；系统级中断可能留下不完整文件，不能当作成功结果。

用户提供的样例中，上一文本单元在 14.160 秒结束，下一单元在 19.520 秒开始，中间有约 5.36 秒未标注区间；需要听音确认其间是否有未转录语音。样例音频不在本机，因此目前无法判断这一段究竟是什么。

## 3. 训练样本和重铸规则

示意：假定“请联系”在第 1 块完整、初始 WAIT 的“张伟”在第 2 块完整、“确认方案。”在第 3 块完整，之后音频结束。下面只展示消息内容，实际 `audios` 还会含每块真实音频的虚拟引用，system 是完整的共享协议：

```json
{"audios":["虚拟音频引用1","虚拟音频引用2","虚拟音频引用3"],"messages":[
  {"role":"system","content":"增量识别协议（此处为示意）"},
  {"role":"user","content":"<audio>"},
  {"role":"assistant","content":"<|DIRECT|><zh><spk1>请联系"},
  {"role":"user","content":"<audio>"},
  {"role":"assistant","content":"<|WAIT|>"},
  {"role":"user","content":"<audio>"},
  {"role":"assistant","content":"<|DIRECT|><zh><spk1>张伟确认方案。"},
  {"role":"user","content":"<end_of_audio>"},
  {"role":"assistant","content":"<|WAIT|>"}
]}
```

以上虚拟引用是占位说明，不能作为实际音频路径使用。WAIT 不输出原文；等待单元的内容保留在构造器的待提交状态里，满足释放条件后与后面的内容共同输出。每次 DIRECT 都有完整 `<语言><说话人>` 前缀，同次 DIRECT 内遇到真实归属变化也加完整前缀。可以有连续 DIRECT，也可以连续 WAIT，取决于实际覆盖和事件。

时间映射采用以下规则：

1. 每轮只输入新到达的 0.5 秒或 1 秒；未完整听到的源单元不提交。如果一个块同时完成多个单元，按该块整体决定动作，而不是每个单元生成一轮。
2. 真实语言或说话人切换，在带时间的归属变化开始所在轮 WAIT；其他初始 WAIT 单元在其完成所在轮 WAIT。事件发生的同轮即使同时有完成的 DIRECT，也保持 WAIT。首次初始化及重复相同前缀不是切换。
3. 每个事件不重复触发；还要等其后的源 DIRECT 单元完整，且当前轮没有新等待事件，才一次释放已完整但未提交的内容。如果没有新的语音可提交则继续 WAIT。原始 WAIT 如同时包含切换和实体，因为没有原因字段，不能区分两种等待意图，代码按切换事件处理一次，不能保证实体完成时又单独 WAIT。
4. 真实音频结束后加无音频的 `<end_of_audio>` 事件，释放仍待提交的完整正文；无待提交内容时返回 WAIT。不添加假音频。`flush:false` 不会取消这个结束事件。

时间戳精确也只保证规则按所给边界执行；实体切分、标签、原文覆盖、说话人和语言归属仍可能出错。切换时间取归属片段的文本对齐起点，它未必等于真实语音中说话人或语言变化的准确起点。语音太短时实际推理还可能无法判断身份，这需真实样本和模型评估，不能从静态重铸保证。

## 4. 全量检查，再启动训练

```bash
# 全量结构检查并实际解码音频；报告路径必须是新文件。
python SFT-Scripts/audit_sft.py /data/OmniCommit/run-001/train.500ms.jsonl --decode \
  --report /data/OmniCommit/run-001/structural-audit.json

# 在原 omni_ms / ms-swift 3.12.1 训练环境执行；HF 是处理器目录。
python SFT-Scripts/audit_encoded_lengths.py /data/OmniCommit/run-001/train.500ms.jsonl \
  --model /models/qwen3_omni_original_hf --max-length 4096 \
  --report /data/OmniCommit/run-001/encoded-audit.json

# 通过检查后，在相同训练环境启动。GPU 编号按服务器调整为四张卡。
bash SFT-Scripts/train_sft.sh \
  /models/qwen3_omni_mcore_3.12.1 \
  /data/OmniCommit/run-001/train.500ms.jsonl \
  /outputs/OmniCommit/sft-001 \
  --gpu-ids 0,1,2,6
```

检查命令全量处理时若存在错误会返回非零；模板长度检查不加载模型权重，但需实际 SWIFT、原 HF processor 和真实音频。默认训练脚本只抽查两行并检查插件，不能代替全量音频解码和模板编码。训练脚本在训练前通过 `--new_special_tokens` 加入 token，并保存训练过的词表层；不是训练结束后添加随机 token。实际模型、GPU 训练、保存、加载和推理仍需在目标环境验收。

**长会议限制：** 精简行没有 `safe_cut`、原始语音区间或 speaker reference 元数据，不能直接用于 `prepare_windows.py`；长录音往往超过真实模板 4096 token，必须先检查。需要拆窗时保留原始标注和时间戳，使用支持该时间戳格式的分阶段路线重新构建带完整元数据的输入，再于安全边界拆窗、检查并提供来自过去的可靠说话人参照；不能靠截断保证完整训练。精简路线本身不保证长间隔说话人身份一致，也没有自动构建 Speaker Prompt Cache。

当前交付只做程序级与模拟测试；用户样例引用的远程真实音频、真实模板编码与 GPU 训练尚未在本地验收。建议先对真实音频小批量核对每轮标签、间隙、姓名数字、切换和 EOS，再扩大数据量。
