# OmniCommit SFT

本目录将已经审查的流式轨迹转成音频、文本交错的 SFT 数据，沿用 FLEX 的
**ms-swift 3.12.1 / Megatron / Qwen3-Omni / 虚拟音频区间**方案。
不依赖其他 FLEX 目录；请保留本仓库的 `omnicommit/`，SFT 和服务共用其中的
`runtime_protocol.py`，确保系统提示、说话人参照和结束标记一致。准备和检查不调用教师模型。

## 已有行内时间戳：直接生成 SFT

如果原始记录已有 `audios`、`messages`、LLM `output` 和 `timestamps`，优先使用一体化构建器，无需重新调用模型或先生成轨迹文件。输入必须是一行一条 JSON 对象，音频必须在构建服务器可读：

```bash
python Data-Scripts/build_asr_sft.py /data/timestamped.jsonl \
  --output /data/sft/train.jsonl --chunk-seconds 0.5 --number-format normalized
```

主文件每行仅含 `audios` 和 `messages`；另有 `.tokens.txt`、`.rejected.jsonl`、`.review.jsonl`。最后一个文件记录至少 2 秒的未标注区间，不能当作已验证静音；默认不因该区间拒绝数据，可用 `--max-unannotated-gap-seconds` 指定过滤阈值。异常行写入拒绝清单并继续；没有成功样本时整体返回非零。输出和伴随文件须使用新路径，一体化入口不支持中断续跑。

`audios` 是逐块虚拟引用，编码了原始音频绝对路径和区间，不是实际小 WAV；训练容器必须能从相同绝对路径访问原文件。`--number-format` 只选择系统指令，不改写原始数字。重铸规则、完整输入格式和云道命令参见 [带时间戳直接生成 SFT](../docs/timestamped_to_sft_zh.md)。

精简文件丢弃了 `safe_cut`、原始区间和说话人参照等离线元数据，**不能直接传给 `prepare_windows.py` 拆窗**；需要从原始标注及时间戳重新构建富元数据路线，再安全拆窗。两种路线生成的文件都必须在目标训练环境检查真实模板长度。

## 分阶段文本路线：v2、结束轮和长会议

新版 `recast_asr.py` 输出 `schema_version:2`：每个真实音频块为 `event:"audio"`，
最后另有 `event:"end_of_audio"`、`start_ms == end_ms == duration_ms` 的结束轮。
准备器为这个结束轮生成 user 文本 `<end_of_audio>`，不新增音频、虚构时间或补一段未来音频。
目标为剩余的 DIRECT 内容，或无剩余时的 WAIT。它是普通输入控制文本，不是第三种动作 token。

v2 每个 DIRECT 以完整语言/说话人标签开头；同轮实际归属变化也写完整标签。
正文里的真实换行保留；不会只为新增元数据而插入换行。原 v1 格式继续兼容。

```bash
python SFT-Scripts/prepare_sft.py /data/recast.jsonl --output /data/sft/full.jsonl

# 短录音可直接使用 full.jsonl。长录音在可安全提交边界切成窗口：
python SFT-Scripts/prepare_windows.py /data/sft/full.jsonl \
  --output /data/sft/train.jsonl --max-audio-seconds 20 \
  --speaker-references /data/verified_refs.jsonl \
  --max-reference-seconds 5 --max-total-reference-seconds 30 --max-speakers 8
```

窗口仅在重铸器给出的 `safe_cut:true` 切开：既没有已完成但尚未提交的文字，也没有
跨过该边界的未听完原子段。仅仅见到 DIRECT 不足以证明边界安全。
超预算而无安全边界的记录会拒绝并继续其余记录，不切坏实体。

参照文件每行形如：

```json
{"id":"meeting-001","references":[{"audio":"/data/audio/session.wav","start_ms":0,"end_ms":1500,"speaker":"spk1","language":"zh","text":"我们开始讨论。","verified":true,"clean":true,"overlap":false}]}
```

`id` 对应原轨迹 ID，音频必须是同一条完整会议录音；参照必须已由外部流程或人工核实。
JSON 中写 true 本身不是质量验证。每个窗口只选开始之前的参照，每位已出现说话人最多一段，
长度 0.5 秒至所设上限。中间窗口需要全部既有说话人的有效参照；缺失时拒绝该会议，
不会跨无关 JSONL 行拼接 `spk1`。参照音频及转录放在第一个 live **user** 输入中，
不会生成对应 assistant 目标，因此默认 assistant-only loss 不训练复读缓存。
只有实际会议最后一个窗口含结束轮，窗口裁剪不会伪装成会议结束。
全部说话人参照合计还必须满足 `--max-total-reference-seconds`，默认 30 秒，与服务默认一致；
超预算会拒绝该会议，不丢掉某位已有说话人或切短经过核实的参照。
窗口的 `--max-speakers` 是缓存预算，服务中的同名参数也是运行预算；它们都不会创造
未训练过的说话人 token。实际可输出的编号仍受本次训练、导出的 token 清单约束。

20 秒只是音频预算，不能保证模型 token 长度不超限。请在训练容器里跑真实模板编码：

```bash
python SFT-Scripts/audit_encoded_lengths.py /data/sft/train.jsonl \
  --model /models/qwen3_omni_original_hf \
  --max-length 8192 --report /data/sft/encoded-audit.json
```

该检查使用实际 processor、虚拟音频读取和 SWIFT 训练模板，包含音频展开和参考输入，
不加载模型权重。超长或无监督目标会报错，不静默截断。成功后再运行下文的训练命令。
本地 CPU 测试未加载真实 Qwen processor；实际环境编码和训练验收仍需要上述命令。

## 新增 token 与训练后导出

新增 WAIT、DIRECT、语种、说话人 token 在 **SFT 前**通过训练启动器的
`--new_special_tokens` 注册；词嵌入与输出层通过 `modules_to_save` 参与保存。
不要先训练，再只往 tokenizer 里加 token：那样缺少学习过的对应权重。

训练后沿用 FLEX 的 HF 基座加 PEFT adapter 导出方式：

```bash
# base 是原始未量化 HF 目录，和训练命令的 MCore 目录不同。
bash SFT-Scripts/merge_lora.sh \
  /models/qwen3_omni_original_hf /outputs/omnicommit/checkpoint-N \
  /models/omnicommit-merged \
  --special-tokens /data/sft/train.jsonl.tokens.txt --gpu-ids 0
```

追加 `--check-only` 只做 tokenizer、配置和张量 header 校验。
完整导出检查训练 args 中的 token 顺序、完整词表模块、导出的 tokenizer/processor、
以及所有新增动作/语种/说话人 token 的词嵌入和输出层权重确实来自已训练适配器。
导出目录包含 `omnicommit_tokens.json`、`omnicommit_tokens.txt` 和
`omnicommit_merge_report.json`；服务根据清单解析实际 ID，不沿用 FLEX 固定 ID。
未经验证的中间导出保留在 `.incomplete-*` 路径，不会发布为正式模型；这是输出暂存，
不会备份或覆盖原模型。CPU 测试使用微型合成张量，不等于已完成真实模型合并与推理。

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
同一说话人换语言不更换 ID。上例为 v1，同轮多个段以换行分隔；v2 保留原始正文，不因归属切换额外插入换行。

`policy.number_format` 支持 `verbatim`（默认，保留口语数词）或 `normalized`
（完整数值明确后转数字）。准备器据此选择 system 提示，**不会改写监督文本**。
未知策略、空说话人段、内嵌控制标记、错误区间均拒绝该行并继续处理其他行。

```bash
python SFT-Scripts/prepare_sft.py /data/trajectories.jsonl \
  --output /data/sft/train.jsonl
```

相对音频路径默认相对于输入文件目录；也可传 `--audio-root /data/audio`。
输出必须使用新路径，避免覆盖已有数据；没有备份行为。

该分阶段准备器生成（区别于上面的精简一体化输出）：

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

# 在原 omni_ms / ms-swift 3.12.1 训练环境检查完整模板编码长度
python SFT-Scripts/audit_encoded_lengths.py /data/sft/train.jsonl \
  --model /models/qwen3_omni_original_hf \
  --max-length 8192 --report /data/sft/encoded-audit.json

# 全量检查通过后，在现有训练容器中启动
bash SFT-Scripts/train_sft.sh \
  /models/qwen3_omni_mcore_3.12.1 \
  /data/sft/train.jsonl \
  /outputs/omnicommit_sft_001 \
  --gpu-ids 0,1,2,3,4,5,6,7
```

训练入口先抽查两个样本并导入框架插件。默认启动八个进程，使用 EP=2、LoRA r8/alpha32、
micro/global batch=2/16、lr=1e-4、3 epochs、packing、max_length=8192，
每 3000 步保存未合并适配器。训练更新 LLM，冻结音频/视觉与 aligner。显式指定四张 GPU 时，默认 micro/global batch 为 4/16。
这里 3000 步与 FLEX 虚拟音频脚本一致；FLEX 更早的原始脚本为 1000 步。默认 `--no_save_optim true --no_save_rng true`，不保存完整优化器和随机状态，不能保证精确断点续训。
保留 `modules_to_save word_embeddings output_layer`，以保存新特殊 token 的完整词表层。
不启用语音输出分支。长会话应提前在真实 tokenizer/框架里检查长度，抽样音频检查
不等于 token 长度验收；不能随意截掉后半轮次后仍声称保留完整轨迹。

token 文件默认使用 `TRAIN_JSONL.tokens.txt`，也可传 `--tokens-file FILE`。
一体化构建器在输入处理结束后才写 token 清单；如果任务异常中断，可能存在不完整 JSONL 而没有清单，不应直接当作完整训练集。检查工具不会过滤 JSONL，报告默认只保留前 100 个错误详情。
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

如果音频或视觉模块在 `flash_attn_gpu.varlen_fwd` 报过错，但希望在 A800 上开启 packing，可用
`--packed-sdpa`。音频和视觉使用分段 math SDPA，Megatron 文本侧使用 FlashAttention 处理打包序列。
复用现有的 500 ms 训练 JSONL 和 token 清单。以下示例从 MCore 基座开始新的一轮训练。

```bash
cd /data/yb/Code/OmniCommit
read -r -e -p '当前 500 ms 训练 JSONL 的绝对路径：' train_jsonl
bash SFT-Scripts/train_sft.sh \
  /data/data_process/lzy/qwen3_omni_mega/train/qwen3_omni_mcore_3.12.1 \
  "$train_jsonl" \
  /data/yb/Code/OmniCommit/outputs/sft-500ms-packed-8gpu \
  --gpu-ids 0,1,2,3,4,5,6,7 --packed-sdpa \
  --micro_batch_size 2 --global_batch_size 32 \
  --max_length 8192 --max_epochs 1 --save_interval 1000
```

启动器在第一张可见卡上检查 BF16 打包注意力的前向、梯度以及两条序列间的隔离，然后进入数据预处理。
也可在训练容器单独运行 `CUDA_VISIBLE_DEVICES=0 python3 SFT-Scripts/check_packed_attention.py`。
若该检查报 `device kernel image is invalid`，应先修复与当前 A800、PyTorch 和 CUDA 环境匹配的 FlashAttention 安装。
8192 是每个 pack 的长度上限。显存按打包后的实际 token 数评估，提速可比较相同原始样本集完成一个 epoch 的用时。

## 验证边界

CPU 测试覆盖真实区间读取、重采样前隔离未来采样、标签与说话人格式、
尾部 WAIT、坏行继续、特殊 token 清单，以及 shell 参数构造。
兼容模式另有真实小型 HF 注意力的分段等价性/梯度测试；依赖不可用时明确跳过。
模板注册测试使用 SWIFT stub。八卡训练、checkpoint 保存和加载需要在目标服务器上验证。
应先在目标服务器以少量样本完成实际模板编码、训练、保存和加载，再做全量训练。

```bash
python -m pytest SFT-Scripts -q
```

继承依据：FLEX/SFT-Scripts 的虚拟音频与 Megatron 配方；
[SWIFT 3.12.1 模板](https://github.com/modelscope/ms-swift/blob/v3.12.1/swift/llm/template/template/qwen.py)、
[Transformers 4.57.1 Omni 注意力](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py)。
