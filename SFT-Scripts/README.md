# OmniCommit SFT

本目录将已经审查的流式轨迹转成音频、文本交错的 SFT 数据，沿用 FLEX 的
**ms-swift 3.12.1 / Megatron / Qwen3-Omni / 虚拟音频区间**方案。
所有模块在本目录内，不依赖其他 FLEX 目录。准备和检查不调用教师模型。

## 环境

数据准备、音频检查：

```bash
python -m pip install -r SFT-Scripts/requirements.txt
```

训练继续使用已有 `omni_ms` 容器：Python 3.11、ms-swift 3.12.1、
Transformers 4.57.1，以及原有兼容的 Megatron/Transformer Engine/CUDA 环境。
这里没有自动升级或安装 CUDA 训练组件。`MODEL` 必须是与原训练配方一致的
**MCore 基座目录**，不是任意 Hugging Face checkpoint。

## 输入和输出

输入为每行一条已经构造完毕的轨迹，例如：

```json
{"schema_version":1,"id":"sample-001","audio":"/data/audio/session.wav","duration_ms":1500,"policy":{"number_format":"verbatim"},"turns":[{"start_ms":0,"end_ms":500,"action":"DIRECT","segments":[{"language":"zh","speaker":"spk1","text":"我同意。"}]},{"start_ms":500,"end_ms":1000,"action":"DIRECT","segments":[{"language":"en","speaker":"spk1","text":"Any questions?"}]},{"start_ms":1000,"end_ms":1500,"action":"WAIT","segments":[]}]}
```

时间是当前原始音频起点的毫秒数。每轮区间必须从 0 开始连续、无重叠，
覆盖整个音频；这份配方限制每轮不超过 8 秒，通常为 0.5 秒。
`DIRECT` 必须有非空文本段，`WAIT` 必须 `segments=[]`。
最后一轮可以是 `WAIT`，但当前版本不接收整条全部 `WAIT` 的静音训练样本。
语言、说话人不是动作。每个输出段都带完整 `<language><speaker>` 头；
同一说话人换语言不更换 ID。同轮多个段以换行分隔。

`policy.number_format` 支持 `verbatim`（默认，保留口语数词）或 `normalized`
（完整数值明确后转数字）。准备器据此选择 system 提示，**不会改写监督文本**。
未知策略、空说话人段、内嵌控制标记、错误区间均拒绝该行并继续处理其他行。

```bash
python SFT-Scripts/prepare_sft.py /data/trajectories.jsonl \
  --output /data/sft/train.jsonl
```

相对音频路径默认相对于输入文件目录；也可传 `--audio-root /data/audio`。
输出必须使用新路径，避免覆盖已有数据；没有备份行为。

生成：

- `train.jsonl`：`audios + messages`，另保留区间、源 ID、数词策略与轮次来源。
- `train.jsonl.tokens.txt`：本数据使用的特殊 token；固定先 WAIT、DIRECT，再语言和说话人。
- `train.jsonl.rejected.jsonl`：坏行位置及原因，不复制音频。
- `train.jsonl.summary.json`：有效/拒绝数与 token 文件位置。

每行即时写入并 flush。单条数据或音频错误不会中断其余行；没有任何有效样本时
返回非零退出码，避免启动空数据训练。文件系统写入错误仍然会终止准备程序。

输出主结构如下（虚拟引用这里缩写）：

```json
{"audios":["omnicommitaudio:v1:...","omnicommitaudio:v1:..."],"messages":[{"role":"system","content":"..."},{"role":"user","content":"<audio>"},{"role":"assistant","content":"<|DIRECT|><zh><spk1>我同意。"},{"role":"user","content":"<audio>"},{"role":"assistant","content":"<|WAIT|>"}]}
```

`audios` 每项对应一个**新增音频块**，不是从录音开头累积的完整音频。
之前的音频和转录由之前的对话轮次保留。原始全音频不会先编码后再切特征。
reader 按范围读取原始采样，随后才下混、重采样，生成内存 WAV，不创建小音频文件。
不足 500 ms 的最后尾块在内存补静音至 500 ms，避免只有一个采样点的尾块成为
几乎为空的编码器输入；来源时间仍保留真实结束点，补零不计为新的原始音频。

训练音频路径必须在服务器可访问。若迁移机器，应在目标机器重新准备轨迹到
训练文件，或正确重建虚拟引用；只改可读的 `audio_segments` 不能修改编码引用。

## 检查和训练

```bash
# 少量实际音频解码；不加载模型、无 GPU 作业
python SFT-Scripts/check_sft.py /data/sft/train.jsonl --rows 2

# 全量检查；--decode 进一步读取每个片段的采样
python SFT-Scripts/audit_sft.py /data/sft/train.jsonl \
  --decode --report /data/sft/audit.json

# 在现有训练容器中启动
bash SFT-Scripts/train_sft.sh \
  /models/qwen3_omni_mcore_3.12.1 \
  /data/sft/train.jsonl \
  /outputs/omnicommit_sft_001 \
  --gpu-ids 0,1,2,3
```

训练入口先抽查两个样本并导入框架插件。原配方保持四进程、EP=2、LoRA r8/alpha32、
micro/global batch=4/16、lr=1e-4、3 epochs、packing、max_length=4096、
每 3000 步保存未合并适配器；音频/视觉与 aligner 冻结，LLM 训练。
保留 `modules_to_save word_embeddings output_layer`，以保存新特殊 token 的完整词表层。
不启用语音输出分支。长会话应提前在真实 tokenizer/框架里检查长度，抽样音频检查
不等于 token 长度验收；不能随意截掉后半轮次后仍声称保留完整轨迹。

token 文件默认使用 `TRAIN_JSONL.tokens.txt`，也可传 `--tokens-file FILE`。
多数据分片训练必须使用包含全部分片标签的同一份固定 token 表。
继续训练、导出和推理时要保存并沿用训练时的 token 顺序、tokenizer 和基座版本；
不要在已有适配器上重新排序或随意换一份 token 表。

原环境中遇到已知旧驱动 FlashAttention 问题时，可传 `--compat-old-driver`：
沿用 FLEX 的分段 math SDPA 和 TE unfused 实现，并关闭 packing/padding_free，
micro 改为 1、global 保持 16。该模式拒绝冲突的 attention/packing 参数。
它不保证修复所有驱动问题。独立小型 GPU 检查入口：

```bash
python SFT-Scripts/check_sft_attention.py --gpu-id 0
```

这会执行小型注意力前后向，不加载大模型；它与默认 CPU preflight 不同。

## 验证边界

CPU 测试覆盖真实区间读取、重采样前隔离未来采样、标签与说话人格式、
尾部 WAIT、坏行继续、特殊 token 清单，以及 shell 参数构造。
兼容模式另有真实小型 HF 注意力的分段等价性/梯度测试；依赖不可用时明确跳过。
模板注册测试使用 SWIFT stub。**尚未把本项目作为真实四卡作业训练并保存 checkpoint。**
应先在目标服务器以少量样本完成实际模板编码、训练、保存和加载，再做全量训练。

```bash
python -m pytest SFT-Scripts -q
```

继承依据：FLEX/SFT-Scripts 的虚拟音频与 Megatron 配方；
[SWIFT 3.12.1 模板](https://github.com/modelscope/ms-swift/blob/v3.12.1/swift/llm/template/template/qwen.py)、
[Transformers 4.57.1 Omni 注意力](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py)。
