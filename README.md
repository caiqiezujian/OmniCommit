# OmniCommit

**Learning to Wait and Emit for Multilingual Speaker-Attributed Streaming ASR**

多语言、多说话人的流式 ASR 数据构建与 SFT 工程。模型逐轮接收音频，只输出新增文字，并学习何时等待；动作只有 `<|DIRECT|>`、`<|WAIT|>`。代码沿用 FLEX 的 Qwen3-Omni / ms-swift 3.12.1 / Megatron SFT 及虚拟音频区间读取方式。

本版本提供文本标注与真实音频教师两条数据路线，包含时间戳对齐、0.5/1 秒标签重构、安全拆窗、SFT、模型导出、流式推理和 Web 入口。没有附带真实语音数据、教师服务或模型权重。CPU 测试与模拟回放不能替代真实服务及 GPU 训练验收。

**已有行内时间戳的标注数据**：使用 [时间戳直接生成 SFT](docs/timestamped_to_sft_zh.md)，一条命令完成重铸、过滤和精简输出，再完成音频与真实模板长度检查。其他输入请按 [完整执行手册](docs/RUNBOOK_ZH.md) 操作。阶段细节见 [对齐与重构](docs/alignment_and_recast.md)、[SFT](SFT-Scripts/README.md)、[流式服务](docs/streaming_runtime.md)。下面保留原音频教师路线作为独立对照，不需要两条路线都执行。

## 目录与流程

```text
原始音频 + 带语言/说话人标签的 JSONL
  -> Data-Scripts/collect_prefixes.py    完整音频及累计音频前缀的独立教师识别
  -> Data-Scripts/build_trajectories.py  whole-output 稳定前缀 -> DIRECT/WAIT 增量
  -> SFT-Scripts/prepare_sft.py          虚拟音频 + 多轮 messages + token 清单
  -> SFT-Scripts/check_sft.py            数据/音频/协议检查
  -> SFT-Scripts/train_sft.sh            Megatron LoRA SFT
```

- `omnicommit/`：CPU 数据构建的协议、音频、I/O 与教师接口。
- `Data-Scripts/prompts/teacher_zh.txt`：音频教师的独立识别指令。
- `SFT-Scripts/`：自包含的训练准备和启动工具；不需要旁边存在 FLEX 仓库。
- `tests/`：跨阶段集成测试；各阶段还有自身测试。
- [数据格式](docs/data-format.md)、[方法与边界](docs/method.md)、[FLEX 适配说明](docs/FLEX_ADAPTATION.md)。

## 先运行离线示例

数据准备支持 Python 3.9+。训练沿用已有 Python 3.11 `omni_ms` 容器；不要用 CPU requirements 替换整套 GPU 框架。

```bash
python -m pip install -r requirements-dev.txt
python tools/run_offline_demo.py --output-dir outputs/offline-demo
python -m pytest -q
```

示例会创建一段合成音调和脚本化教师预测，跑完 replay → 轨迹 → SFT → 音频读取检查。**音调没有对应的人声，生成文件仅用于程序联调，禁止作为真实 SFT 样本。** 所有生成数据默认被 Git 忽略；重复运行请选择新的输出目录。

## 1. 收集真实音频教师识别

支持一条记录一个音频、一个完整 assistant 转录。源格式如下；`id` 建议使用数据集内唯一标识：

```json
{"id":"meeting-001","audios":["/data/audio/meeting.wav"],"messages":[{"role":"system","content":"Transcribe with speaker and language labels."},{"role":"user","content":"<audio>**Entity Reference:** []"},{"role":"assistant","content":"<zh><spk1>我同意。<en><spk1>Any questions?<zh><spk2>没有。"}]}
```

音频教师需支持 OpenAI 兼容 Chat Completions 的 `input_audio`，接收 WAV base64，而非仅接收服务器文件路径。`--model` 填服务实际模型别名，不能使用纯文本模型代替。接口路由/请求参数可在命令行和配置中指定，示例配置沿用此前 temperature=0.1、top_p=0.8、max_tokens=2048、timeout=300。如果服务采用其他音频请求协议，需要适配 `omnicommit/teacher.py`。

```bash
python Data-Scripts/collect_prefixes.py /data/raw.jsonl \
  --output outputs/teacher-traces.jsonl \
  --api-base http://127.0.0.1:8080/v1 \
  --model YOUR_AUDIO_CAPABLE_MODEL \
  --config configs/teacher.example.json \
  --chunk-seconds 0.5 \
  --number-format normalized
```

需要鉴权时把密钥放在环境变量，并使用 `--api-key-env` 指定变量名；不要将密钥写入配置或 Git。完整参考转录、原始 system/user 指令不会被转发给前缀识别调用，避免答案泄漏。教师使用完整音频和每个累计前缀分别识别，不强制复制历史结果。

`--number-format verbatim` 保留数字口语形式，`normalized` 要求数值明确后再归一化。两者应与最终训练目标一致。默认是 verbatim；上面的命令显式选择原始数据示例要求的 normalized。

使用相同输入、音频和配置追加 `--resume` 可续跑；已写失败记录也视为已处理，避免云端重启后无限重复失败调用。单条请求或内容错误记录为空 `output`，继续下一条，详细错误保留在输出和拒绝日志。目录权限、输入/输出冲突或损坏的恢复记录属于运行配置/文件错误，会明确退出，不能冒充成功。

## 2. 构建稳定提交轨迹

```bash
python Data-Scripts/build_trajectories.py outputs/teacher-traces.jsonl \
  --output outputs/trajectories.jsonl
```

基于完整教师输出筛选符合其前缀的 **整条历史识别**，同时比较文字、语言和说话人归属，再取相邻累计稳定前缀的差分。参考 JSONL 用于离线审查，不会悄悄替换早期音频应输出的内容。可选 `--holdback-chars` 是保守尾部缓冲，不是完整实体保护保证；`--min-reference-agreement` 是参考一致性筛选阈值，不是声学正确率。

语言或说话人标签可以在序列化时统一，但归属必须参与稳定性判断。原始参考中的空说话人段、重复冲突标签等不会被猜测修复：默认记录解析错误，仍让教师独立听音频；收集时可用 `--strict-reference` 拒绝此类记录。参考不可解析时，一致性分数不可用，正的 `--min-reference-agreement` 会拒绝该条。具体边界见 [方法说明](docs/method.md)。

## 3. 准备及检查 SFT

```bash
python SFT-Scripts/prepare_sft.py outputs/trajectories.jsonl \
  --output outputs/train.jsonl

python SFT-Scripts/check_sft.py outputs/train.jsonl --rows 2 \
  --tokens-file outputs/train.jsonl.tokens.txt
```

输出是 FLEX 风格的 `audios`、`messages`，但只使用两种动作。`audios` 指向原音频的虚拟区间；训练读取时先截原始采样再重采样，不写小 WAV。服务器上必须能访问记录中的原音频绝对路径。

准备器同时生成 `.tokens.txt`、`.rejected.jsonl` 和 `.summary.json`。数字策略传递到 system prompt；语言及说话人 tokens 按本数据生成。合并训练集后再统一准备 token 清单，验证集也须被这份清单覆盖；恢复已有 checkpoint 时不得更换清单顺序或重新分配 token ID。不要将不同词表训练出的适配器混用。

单条非法轨迹被拒绝并继续；尾部 WAIT 合法。当前只接受含转录的会话，全部 WAIT 的样本不作为真实静音监督。增加纯静音数据需要独立验证 no-speech 标签，不能把教师失败当成静音。

## 4. 在原 FLEX 环境训练

```bash
bash SFT-Scripts/train_sft.sh \
  /models/qwen3_omni_mcore_3.12.1 \
  /data/OmniCommit/train.jsonl \
  /outputs/OmniCommit/sft-v1 \
  --gpu-ids 0,1,2,6
```

第一个参数是原训练环境的 **MCore 模型目录**，不是教师 API 地址，也不是任意 HF checkpoint。启动器先检查数据、tokens 和框架插件。保持 FLEX 的四卡 LoRA r8/alpha32、EP2、micro4/global16、lr=1e-4、3 epochs、packing、max_length4096、每3000步保存的配方；只训练语言主干相关参数，冻结感知编码与对齐模块，保存未合并适配器及训练过的词嵌入/输出层。

旧驱动环境可沿用 `--compat-old-driver`，具体环境前提与检查见 [SFT 说明](SFT-Scripts/README.md)。默认不升级 torch、Transformers、SWIFT 或 CUDA。新增 token、真实模板编码、packing、模型保存和加载必须在目标机器用少量数据先验收；本地 CPU 测试不加载模型权重。

长会话可能超过4096训练token，需要检查实际模板编码长度；不能直接把截断后的片段称为完整会话训练。旧的分阶段文本路线提供安全拆窗与因果说话人参照；新的精简输出不保留拆窗所需元数据，不能直接用 `prepare_windows.py` 拆窗。不同录音之间不能凭相同 spk 编号合并身份，真实跨长间隔声纹一致性仍需模型验收。

## 研究定位

参考 [Confucius4-R2T2](https://arxiv.org/abs/2611.06043) 的监督构造思路，加入语言与说话人属性，并适配 FLEX 的音频文本交错训练格式。这里的工程基础不代表已经验证论文效果、低延迟或 speaker attribution 准确率。当前未实现 GRPO，也不把前缀一致性当成实体/数值语义正确性的证明。
