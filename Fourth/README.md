# Fourth · 多语言流式 ASR

Fourth 从带语种、说话人和逐段时间戳的原始 JSONL 构建流式 SFT 数据。默认每轮接收 1 秒音频。模型提交原语言正文时先输出 `<|DIRECT|>`，无正文时单独输出 `<|WAIT|>`。换人时在新说话人的第一段正文前输出 `<|SPK_CHANGE|>`。同一人换语种时直接续写。

## 1. 从原始标注生成训练集

### 从静音目录生成最多 3000 条全 WAIT 样本

传入已确认全程无人说话的音频根目录，脚本会递归查找各层子目录中的 WAV 等音频文件。每条录音占一行，按可解码的真实帧数构建 Fourth 的虚拟音频引用，默认每 1000 ms 一轮。音频轮和最后的 `<end_of_audio>` 均监督 `<|WAIT|>`。例如 2.35 秒录音有 3 个音频轮和 1 个结束轮。尾块覆盖最后 0.35 秒，训练读取时按既有规则补齐输入。采样假设文件夹已按无人声筛选，生成时无需运行 VAD 或 ASR。

~~~bash
cd /data/yb/Code/OmniCommit
python Fourth/Data-Scripts/build_silence_sft.py /data/silence_wavs \
  --output-dir datasets/silence_3000
python Fourth/SFT-Scripts/check_sft.py datasets/silence_3000/train.1000ms.jsonl --decode
~~~

将 `/data/silence_wavs` 换成训练容器可持续读取的真实目录。默认 `--count 3000 --seed 42 --chunk-ms 1000 --number-format normalized`。`--count` 是有效录音上限。有效录音不超过 3000 条时全部写入，超过时按固定种子抽取 3000 条。脚本跳过损坏、零时长的文件，同一源文件的多条链接只计一次。如果可用录音为 0，脚本会报错。

输出目录包含 `train.1000ms.jsonl`、`token.txt` 和 `train.1000ms.jsonl.tokens.txt`。两个 token 文件都沿用 Fourth 的完整协议，静音样本的每个 assistant 目标只有 `<|WAIT|>`。`recast.events` 中的 `silence_only` 供数据校验使用。按下例合并到现有 1 秒语音训练集，沿用原 SFT 命令。

~~~bash
cat datasets/fourth_merged_1s_v2/train.1000ms.jsonl datasets/silence_3000/train.1000ms.jsonl \
  > datasets/train.with_silence.1000ms.jsonl
cp datasets/fourth_merged_1s_v2/train.1000ms.jsonl.tokens.txt \
  datasets/train.with_silence.1000ms.jsonl.tokens.txt
~~~

### 直接用现有分段做 1 秒重铸

如果一条音频已经有合适的分段，用 `Data-Scripts/recast_1s.py`。传入原始 JSONL：每行需要 `audios`、带语种和说话人的 assistant 参考转录、`output.segments[].source_text`，以及与原段一一对应的 `timestamps[].start_ms/end_ms`。时间相对当前音频起点，以毫秒计。

~~~bash
cd /data/yb/Code/OmniCommit/Fourth
python Data-Scripts/recast_1s.py \
  /data/yb/Code/OmniCommit/datasets/timestamped.jsonl \
  --output-dir /data/yb/Code/OmniCommit/datasets/fourth_source_1s

python SFT-Scripts/check_sft.py \
  /data/yb/Code/OmniCommit/datasets/fourth_source_1s/train.1000ms.jsonl \
  --rows 100 --decode
~~~

每个原段按自己的结束时间提交。到达整秒边界时按原文顺序提交所有已完成的连续段，剩余音频不足一秒时以真实末尾为最后一个窗口。没有到期正文就输出 `<|WAIT|>`。同一秒完成的多个原段共用一个 `<|DIRECT|>`，换人在新人的正文前插入 `<|SPK_CHANGE|>`。段内即使有更细的 `spans` 也按原段末尾释放，输入的旧动作不决定输出。构建器用原 WAV 的区间引用生成训练数据。

默认 `--tail-policy keep` 保留原文末尾标点和完整音频。若这批也是之前的截取数据，沿用 `--tail-policy short-open`。训练文件、token 清单及 SFT 启动方式与下方配方一致。`summary.json` 中此入口的 `release_on` 为 `source-segment`。

### 已跑完第一步合并的训练集

`merge_timestamped_segments.py` 完成时才发布 `full_10000h_augmented.merged_v2.jsonl`。每行保留原音频路径和参考转录，`output.segments` 存新分组，`timestamps` 存对应区间。新合并组的 `timestamps[].spans` 保留细时间，`merge_source_units` 指向原段。合并阶段保留末尾标点，也没有添加录音自然结束标记。可直接把这份输出交给第二步。

~~~bash
cd /data/yb/Code/OmniCommit/Fourth
python Data-Scripts/build_sft.py \
  /data/s50062929/FLEX/SFT_data/full_10000h_augmented.merged_v2.jsonl \
  --output-dir /data/yb/Code/OmniCommit/datasets/fourth_merged_1s_v2 \
  --chunk-ms 1000 --release-on merged-unit --tail-policy short-open

python SFT-Scripts/check_sft.py \
  /data/yb/Code/OmniCommit/datasets/fourth_merged_1s_v2/train.1000ms.jsonl \
  --rows 100 --decode
~~~

`merged-unit` 直接使用已合并语义段。在每个音频窗口结束时，按参考顺序一次提交所有已说完的连续段，没有可提交内容才产生 `<|WAIT|>`。同一人换语种时直接转录新语种的正文；每次换人在新人的第一段正文前插入 `<|SPK_CHANGE|>`。例如 A 的“准备好了吗？”在 0.55 秒结束，B 的“好。”在 0.78 秒结束，B 随后的英语在 1.25 秒结束。第一秒的目标是 `<|DIRECT|>准备好了吗？<|SPK_CHANGE|>好。`，英语在下一轮交付。混合归属段有可靠子时间时可以分归属提交，只有整段时间时要等段末。重叠发言维持参考顺序。

| 尾部选项 | 适用数据与训练行为 |
| --- | --- |
| `keep`（默认） | 完整录音。处理到 WAV 末尾，保留参考标点，再训练 `<end_of_audio> → <\|WAIT\|>`。 |
| `short-open` | 本次截取数据的启发式选项。以 WAV 总长减去最晚标注结束时间，尾隙严格小于 1000 ms（含零）时，按开放尾部处理。正好 1000 ms 及以上走完整录音流程。 |
| `annotated-prefix` | 整批都只保证前缀有转录时使用。所有样本在标注结束处停止，长尾也不产生无标注音频轮。 |

开放尾部仅生成到最后有效标注的虚拟音频引用，最后一个音频回复有正文，后面不追加结束事件。尾隙不足一秒时可移除最后一个独立的句末句点，保留原词、内部标点、问号、感叹号、省略号以及规则能识别的缩写。若句号单独占一个有时间的段，移除后可回退到上一段正文的结束时间。源 WAV、合并标注及参考原文保持原样。`short-open` 也会命中快速收尾的完整句子，时间差本身无法鉴定音频是否被截断。

在 `summary.json` 的 `by_chunk_ms` 中查看 `open_ended_rows` 和 `terminal_periods_removed`。每条开放样本的 `recast.events` 以 `annotation_end` 结束，`supervised_frames` 和 `source_frames` 分别记录监督截止帧和源音频总帧数。

构建结束后，在 Fourth 的 SFT 环境中直接训练这份 `train.1000ms.jsonl`。同目录的 `train.1000ms.jsonl.tokens.txt` 由构建脚本生成，训练入口会自动读取。下面沿用 8 卡、packing 4096、全局 batch 128、1 个 epoch，每 1000 步保存一次。

~~~bash
cd /data/yb/Code/OmniCommit/Fourth
mkdir -p logs
nohup bash SFT-Scripts/train_sft.sh \
  /data/yb/models/qwen3_omni_mcore_3.12.1 \
  /data/yb/Code/OmniCommit/datasets/fourth_merged_1s_v2/train.1000ms.jsonl \
  /data/yb/Code/OmniCommit/output/fourth-merged-1s-v2 \
  --gpu-ids 0,1,2,3,4,5,6,7 --attention_backend flash \
  --packing true --padding_free true \
  --max_length 4096 --packing_length 4096 \
  --micro_batch_size 2 --global_batch_size 128 \
  --max_epochs 1 --split_dataset_ratio 0 --save_interval 1000 \
  > logs/train_fourth_merged_1s_v2.log 2>&1 < /dev/null &
~~~

### 从未合并的原始标注开始

将整个 `Fourth/` 目录放到服务器，确认容器能读取原始音频路径。在已有数据环境中执行：

~~~bash
cd /data/yb/Code/OmniCommit/Fourth
python -m pip install -r requirements-data.txt
python Data-Scripts/build_sft.py \
  /data/yb/Code/OmniCommit/datasets/timestamped.jsonl \
  --output-dir /data/yb/Code/OmniCommit/datasets/fourth_1s_v1 \
  --chunk-ms 1000 --short-ms 300
~~~

输入每行应含以下字段。时间单位为毫秒，起点是该行的音频文件起点。

| 字段 | 用途 |
| --- | --- |
| `audios` | 单个可读的源音频。脚本读取真实帧数、采样率和总时长。相对路径默认以 JSONL 所在目录为基准，也可以指定 `--audio-root`。 |
| `messages` | 单条 assistant 参考转录，包含 `<zh><spk1>` 等原始标记。重复的同人同语标记可以提供原始断句依据。支持已有的 `content_original`。 |
| `output` | 对象或 JSON 字符串，含 `segments[].source_text`。脚本利用对应的文本和归属做核验，重新计算每轮动作。 |
| `timestamps` | 与 segments 对应的 `start_ms/end_ms`。有 `spans` 时校验其正文、语种、说话人及起止时间；有 `index` 时依据从 1 开始的索引对应。 |
| `tags` | 可保留源数据已有的蒸馏标签。最终动作由新的提交规则决定。 |

**合并与重铸。** 先在同人同语范围内，利用原始段界、已对齐的短片段、数字连续部分等依据合并。300 ms 是短片段候选阈值。独立短答、跨人和跨语片段各自保留。再按实际音频样本位置逐轮处理：

- 本轮已提交得出一个分句、其后还有新起的零碎尾巴时，优先在已有标点处收住，后续从尾巴接着提交。
- 同一窗口里旧语种或旧人的末尾和新段开头都有标注时，优先发旧段。新段即使已经完整，也可以下一轮才发。下一轮新到的音频可以是静音。
- 上一轮暂存的完整文字按原序释放。一秒内 A→B→A 时，会在 B 和返回的 A 前分别标记换人。长间隔之后只有新人开口时，首个完整单元可以立即提交。
- 短间隔连续对白使用可靠标注边界。粗段缺少内部分时依据时，以有证据的整段释放。重叠发言保留参考顺序。源数据已有的错字、错标点或错误归属需要另外清洗。
- 最后一个实际音频轮释放余文，然后发送文本事件 `<end_of_audio>`，目标为 `<|WAIT|>`。脚本对每条成品比对参考正文、标点、空格与换人顺序，失败行单独记录。

默认按准备好的语义单元提交。若目标是细粒度对照，可以传 `--release-on timed-span`，使用经过校验的子段，已经合并保护的短语保持完整。同时传 `--chunk-ms 800 1000` 会对同一输入分别构建两套数据。输出目录需能容纳整套产物，已有同名文件时选择新目录。

## 2. 查看训练行与构建记录

| 输出文件 | 内容 |
| --- | --- |
| `train.1000ms.jsonl` | 每条有效原始录音一行。模型输入是 `audios` 和 `messages`，附加的 `recast` 用于核对时间和提交行为。 |
| `token.txt` | 本次构建的三行 token 清单。 |
| `train.1000ms.jsonl.tokens.txt` | 与 `token.txt` 相同的逐字节副本。训练和预检通过训练文件名自动找到它。 |
| `rejected.jsonl` | 输入行号和失败原因。单条坏数据会跳过，后续行继续处理。 |
| `summary.json` | 通过/拒绝数、每个步长的合并和延后次数、音频轮数、WAIT 数及最大未标注时间间隔。 |

`audios` 的每个虚拟引用包含原 WAV 路径及截取区间。构建阶段不生成每秒一个新 WAV。训练时按引用裁剪源音频后再做特征提取，请保留服务器上的原文件。

例如同一秒先听到 A 问完、再听到 B 完整回答「能。」，下一秒继续保留在录音中：

~~~text
第 1 轮  0.0–1.0 s     <|WAIT|>
第 2 轮  1.0–2.0 s     <|DIRECT|>能听见吗？
第 3 轮  2.0–3.0 s     <|DIRECT|><|SPK_CHANGE|>能。
结束事件               <|WAIT|>
~~~

这个三轮示例的辅助字段如下。轮次从 1 开始，`audio_end` 标出最后一轮实际音频。多枚换人符号在同一轮时，`speaker_change` 事件只记录一次，具体位置保存在该轮 assistant 正文中。

~~~json
{
  "recast": {
    "source_audio": "/data/meetings/example.wav",
    "chunk_ms": 1000,
    "merged_units": 0,
    "events": [
      {"turn": 2, "kind": "switch_deferred"},
      {"turn": 3, "kind": "speaker_change"},
      {"turn": 3, "kind": "audio_end"}
    ]
  }
}
~~~

`punctuation_deferred` 代表已经到期的尾部因标点截断而延后。单元本身还没有到结束时间时自然等待，不计入延后。用下面的命令逐轮查看第一行，或验证前两行的真实音频引用：

~~~bash
python - /data/yb/Code/OmniCommit/datasets/fourth_1s_v1/train.1000ms.jsonl <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as source:
    row = json.loads(source.readline())
print(row["recast"])
for turn, message in enumerate(row["messages"][2:-2:2], 1):
    print(turn, message["content"])
print("EOF", row["messages"][-1]["content"])
PY

python SFT-Scripts/check_sft.py \
  /data/yb/Code/OmniCommit/datasets/fourth_1s_v1/train.1000ms.jsonl \
  --rows 2 --decode
~~~

## 3. 训练

在原来的 MCore / ms-swift 镜像中使用本目录的脚本。以下是已有的八卡、4096 packing、micro batch 2、global batch 128、一轮、每千步保存配方：

~~~bash
cd /data/yb/Code/OmniCommit/Fourth
bash SFT-Scripts/train_sft.sh \
  /data/yb/models/qwen3_omni_mcore_3.12.1 \
  /data/yb/Code/OmniCommit/datasets/fourth_1s_v1/train.1000ms.jsonl \
  /data/yb/Code/OmniCommit/output/fourth-speaker-full-1s \
  --gpu-ids 0,1,2,3,4,5,6,7 --attention_backend flash \
  --packing true --padding_free true \
  --max_length 4096 --packing_length 4096 --micro_batch_size 2 \
  --global_batch_size 128 --max_epochs 1 --split_dataset_ratio 0 --save_interval 1000
~~~

启动日志会显示 `[train] vocab-update=full (word_embeddings and output_layer)`。输入词嵌入表和输出词表整表参与优化，包括普通转录用词、标点和协议 token。Thinker 继续训练 LoRA，音频编码器和对齐层冻结。assistant 的正文和正文中的换人标记一起参与监督。训练时需要为两张完整矩阵保留梯度和优化器状态。之前按本页三枚 token 构建的 Fourth 数据可以直接使用。训练集同名 token 文件按以下顺序保存：

~~~text
<|WAIT|>
<|DIRECT|>
<|SPK_CHANGE|>
~~~

如使用 ms-swift 4.1.3 的 HF 基座入口，选择对应的脚本和原始 HF 模型：

~~~bash
bash SFT-Scripts/train_sft_swift4.sh \
  /data/models/qwen3_omni_original_hf \
  /data/yb/Code/OmniCommit/datasets/fourth_1s_v1/train.1000ms.jsonl \
  /data/yb/Code/OmniCommit/output/fourth-speaker-full-swift4 \
  --gpu-ids 0,1,2,3,4,5,6,7 --save_steps 1000
~~~

如需在训练环境检查编码长度和特殊 token 的监督位置：

~~~bash
python SFT-Scripts/audit_encoded_lengths.py \
  /data/yb/Code/OmniCommit/datasets/fourth_1s_v1/train.1000ms.jsonl \
  --model /data/models/qwen3_omni_original_hf --rows 2 --max-length 4096 \
  --report /data/yb/Code/OmniCommit/datasets/fourth_1s_v1/encoded-audit.json
~~~

## 4. 合并并启动 Web

在 `Fourth/` 目录运行下面的命令。将第二个位置参数替换为实际含有 `adapter_config.json` 和适配器权重的 checkpoint 目录，传入本次训练的 token 清单。脚本在 Swift 写完模型后统一 tokenizer 的特殊标记格式，重新加载 tokenizer 和音频 processor，并将输入和输出词表逐行与 checkpoint 中保存的两张矩阵核对。看到 `[merge] SUCCESS` 后，Web、HTTP、批量推理和 GSPO 都可以使用第三个位置参数指定的同一个 HF 目录。GSPO 使用 `speaker` 协议的 Fourth 模型。

本项目当前训练镜像使用 Swift 3.14.2。Swift 3.12 和 3.14 镜像均走已有的 3.x 合并命令。合并时的实际依赖版本记录在 `omnicommit_merge_report.json` 的 `runtime_versions` 中。Web 的 vLLM 后端在启动日志输出 `[Fourth vLLM]`，GSPO 在启动计划输出自己的 `runtime_versions`。三个位置分别记录各自进程实际使用的环境。

同一个合并目录供以下两个运行环境读取。GSPO 使用训练虚拟环境中启动 `train_gspo.py` 的 Python。

| 入口 | vLLM | Transformers | tokenizers | PyTorch |
| --- | --- | --- | --- | --- |
| Web（`Flex_web`） | `0.11.1+cu129` | `4.57.1` | `0.22.1` | `2.9.0+cu129` |
| GSPO（`flex-gspo-online`） | `0.19.1` | `5.6.2` | `0.22.2` | `2.10.0+cu129` |

~~~bash
bash SFT-Scripts/merge_lora.sh \
  /data/models/qwen3_omni_original_hf \
  /data/yb/Code/OmniCommit/output/fourth-speaker-full-1s/checkpoint-1000 \
  /data/yb/Code/OmniCommit/models/fourth-speaker-full-merged \
  --special-tokens /data/yb/Code/OmniCommit/datasets/fourth_1s_v1/token.txt \
  --gpu-ids 0

python -m pip install -r requirements-web.txt
~~~

### 一条命令启动会议网页

在 `Flex_web` 容器中编辑一次 [web_config.sh](web_config.sh)。默认使用 GPU 6、checkpoint 2721、1 秒音频步长和 HTTPS 端口 18895。`ASR_MODEL` 指向合并好的 HF 目录；把 `SILERO_VAD_MODEL` 改为容器中已有的 `silero_vad.onnx` 绝对路径。首次启用需要可用的 CPU ONNX Runtime。如果容器里 `python3 -c "import onnxruntime"` 报缺少该包，执行 `python3 -m pip install 'onnxruntime>=1.20,<2'`。

服务自动识别两种流式 ONNX 接口：旧版的 `input,sr,h,c → output,hn,cn`，以及新版的 `input,sr,state → output,stateN`。启动时用本地权重连续试算两帧，确认状态能传到下一帧后，打印 `[Silero VAD] ready: layout=h/c` 或 `layout=state`，随后加载 Omni。如果已有其中一种兼容权重，直接使用现有文件。[旧版官方流式实现](https://github.com/snakers4/silero-vad/blob/v4.0/utils_vad.py) · [官方 v6.2.3 权重](https://github.com/snakers4/silero-vad/blob/v6.2.3/src/silero_vad/data/silero_vad.onnx)

~~~bash
cd /data/yb/Code/OmniCommit
bash Fourth/start_server.sh
~~~

打开 `https://10.185.1.71:18895`，状态栏会显示「静音过滤已开启（Silero）」。`/health` 中对应字段为 `"silence_gate":"silero"`。要用同一段录音做开关对照，把 `SILERO_VAD_MODEL=''` 后重启。显式传入模型路径和参数时，启动脚本使用该次命令行设置。

#### 逐轮回听和下载

展开「逐轮日志」，每个音频轮都列出新增音频的时间范围和该轮提交的文字，包括返回 `WAIT` 的轮次。「本轮」播放该轮实际收到的声音，「前一轮 + 本轮」连续播放相邻两轮，「下载本轮 WAV」导出该轮的 16 kHz 单声道音频。文件、麦克风和电脑声音均可回听。选择文件的一段转录时，日志同时标出它在原文件中的位置。

例如上一轮听到「张」并等待，这一轮听到「伟」后提交「张伟」。点「本轮」只听到「伟」，点「前一轮 + 本轮」可以听到完整姓名。文字按模型实际提交的轮次显示。

音频暂存在当前页面，会议结束后仍可回听。刷新页面或开始下一场会议会释放上一场音频，需要留存的片段可提前下载。历史文字日志仍可在本浏览器查看。

#### 安静、句尾和再次开口

| 当前 1 秒的声音 | 网页服务的处理 |
| --- | --- |
| 从开始就安静，或已进入持续静音 | 本轮回 `WAIT`，时钟照常推进，跳过主模型和补识别。默认窗口只保留最近一个连续跳过的音频块，供下次开口时回听。 |
| 刚有过人声或不确定的声音，随后判为安静 | 让主模型正常解码一个安静块，给上一轮的尾字一次提交机会。会议仍在继续。 |
| 新块里出现人声，或者有短促起音让检测结果不确定 | 当轮送入完整原始块，沿用同一场会议和原有窗口。话筒、系统声音及文件输入都使用相同逻辑。 |
| 用户结束时最后一个音频块已被静音过滤 | 结束轮回 `WAIT` 并关闭会话。逐轮日志可以区分模型返回的 `WAIT` 和静音跳过。 |

Silero 在 CPU 上计算人声概率，16 kHz 下默认每帧 32 ms。旧版 h/c 图如果声明固定的 64 或 96 ms 帧，则使用其声明的长度。模型权重由网页进程加载一次，每场会议独立维护状态。默认静音上限为 `SILERO_QUIET_THRESHOLD=0.10`，降低它会让更多可疑音频进入 Omni。检测到人声只决定是否送入 ASR。房间内其他人在说话、背景视频有人声等情况仍可能触发转录。

### 已经合并的旧模型

如果启动在 `AutoTokenizer.from_pretrained` 报 `'list' object has no attribute 'keys'`，或 GSPO 报 `Protocol token is not registered special`，在仓库根目录执行一次以下命令。`MODEL` 指向已经合并的 HF 模型，运行时请让该目录处于无写入状态。

~~~bash
cd /data/yb/Code/OmniCommit
MODEL=/data/yb/Code/OmniCommit/output/fourth-merged-1s-v2/v0-20261121-063241/sft-checkpoint-2721
python3 ASR-GSPO/repair_tokenizer.py "$MODEL" --apply
~~~

工具先在临时目录整理特殊标记，再按 `omnicommit_tokens.json` 验证训练 ID、音频占位符、tokenizer/processor 和权重行覆盖。返回 `repaired` 或 `already_valid` 即可对同一目录运行原启动命令。省略 `--apply` 可只读诊断。此操作只发布通过校验的 tokenizer 文件，权重无须重新合并。

将配置中的 `10.185.1.71` 换成访问者实际使用的内网 IP 或域名。首次启动需要服务器安装 OpenSSL 1.1.1 或更高版本。将已配置证书目录（默认 `/data06/yb/Code/OmniCommit/certs`）中的 `root-ca.crt` 复制到访问网页的 Windows 电脑，在该文件所在目录执行 `certutil -user -addstore Root .\root-ca.crt`，重新打开浏览器。私钥留在服务器。此后使用同一个证书目录即可复用信任。

如果这台电脑已信任 FLEX，并且服务器访问地址在原证书内，可把配置中的 `--https-host` 和 `--cert-dir` 两项换成 `--ssl-certfile /实际FLEX证书目录/server.crt --ssl-keyfile /实际FLEX证书目录/server.key`。服务器可用 `curl --cacert /实际证书目录/root-ca.crt https://10.185.1.71:18895/health` 验证证书和服务状态。

网页中选择「电脑声音」并点击「开始转录」，在浏览器弹窗选中正在播放的标签页或系统支持的屏幕来源，勾选共享音频。声音来自访问网页的电脑，通过同一端口的 WSS 发送。网页也支持麦克风、音视频文件、文件起止时间和会议上下文窗口设置。[Chrome 共享音频说明](https://developer.chrome.com/docs/web-platform/screen-sharing-controls#the_systemaudio_option)

新协议在健康检查中显示 `action_protocol=speaker`。网页按换人标记分段显示，同一人后续提交继续接在已有正文后。标记表示发言人变化，无法判断这一位是否曾经发过言。通过 `--window-seconds`、`--discard-seconds` 配置滚动上下文，`--no-recovery` 可关闭有声 WAIT 的局部重听。推荐和训练使用相同步长。

提供给其他程序的 HTTP 接口仍返回 `{"src":"本包新增正文"}`。服务命令、日志和调用方式见 [HTTP API](HTTP_API.md)。

## 5. 多语种测试集批量推理

在 A800 的 vLLM 0.11.1 推理环境中运行。先在 `tools/transcribe_manifests.py` 顶部设置 `ROOT`、`MODEL` 和 `FIELD`。默认测试集是 `/data/yb/Code/OmniCommit/datasets/all_audios`，模型是 `/data/yb/Code/OmniCommit/models/fourth-speaker-full-merged`，结果字段为 `OmniCommit_Fourth`。比较多个 checkpoint 时，为每个模型指定不同的字段名。

每个语种目录包含 `splited_audios/` 和一份清单。例如 `de/splited_manifest.json` 是 JSON 数组，`de/splited_manifest.jsonl` 是每行一条记录。脚本自动选择存在的文件；若两者都有，设置脚本顶部的 `MANIFEST_NAME`。脚本按每条记录 `audio` 字段的文件名，在对应语种的 `splited_audios/` 下读取 WAV。

~~~bash
cd /data/yb/Code/OmniCommit/Fourth
python tools/transcribe_manifests.py
~~~

默认 `GPU_IDS="0"`、`BATCH_SIZE=4`、`CHUNK_MS=1000`。每条 WAV 独立建会话，按 1 秒步长依次推理，多条 WAV 可以并行。将 `GPU_IDS` 改成 `"0,1"` 后，两张卡通过张量并行承载同一个模型，`BATCH_SIZE` 仍是整个引擎的并发上限。模型历史保留换人标记，写入结果字段时剥离 `<|DIRECT|>`、`<|WAIT|>` 和 `<|SPK_CHANGE|>`。评测保留模型实际生成的重复和达到 token 上限时的正文，关闭有声 WAIT 的补识别。

默认重新推理所有选中的 WAV，首轮就包含已有目标字段的记录。每个语种最多跑三遍。第一遍使用配置的并发数，第二遍将并发数减半（向下取整，最少 1），第三遍逐条识别。只有读音频或解码出错、或读完整条 WAV 和结束轮后仍无正文的记录会进入下一遍。每次从音频开头建立新会话，三遍共用已加载的模型、相同的音频步长和解码参数。一次正常的中间 WAIT 可以由后续轮次提交正文。

成功时覆盖目标字段。三遍仍失败时该字段写为空字符串，其他字段保留。每累计 20 次结果写入以及每遍结束时原子保存清单。如果显式使用 `--resume`，只跳过已有非空正文的行。只重跑指定音频时使用 TSV 名单，每行依次填写语种、从 0 开始的原记录索引和 WAV 文件名，用制表符分隔。

每次运行在 `ROOT` 下生成一个 `failed-<字段>-*.jsonl`，终端的 `[失败清单]` 打印本次路径。所有语种共用这一个文件，最终失败时立即追加一行。`attempts` 中的 `all_wait` 表示包括结束轮在内的每轮原始回复都是 WAIT，`empty_transcript` 表示去除协议标记后全程无正文，`error` 附带异常类型、消息和出错阶段。每遍记录 WAV 信息、已处理音频的峰值、完成轮数、已经产生的正文长度，以及末四轮的回复摘录和实际 vLLM 请求批量。首次音频读取失败时可直接看 `stage=open_wav` 和 `wav_path`。清单损坏或补跑名单无法匹配的条目单独标明类型，尝试列表为空。终端 `[恢复]` 显示缩小并发后恢复的音频。

~~~bash
python tools/transcribe_manifests.py --retry-list /data/yb/Code/OmniCommit/datasets/retry.tsv
~~~

### TSC 和 Byte 的双字段 JSONL

`tools/transcribe_tsc_byte.py` 按音频文件名匹配 JSONL 的 `vid`。每成功处理一条，就将两个字段一起原子写回原文件。以 `OmniCommit_Fourth` 为前缀时，`OmniCommit_Fourth_asr` 保存 Fourth 对音频的实际转录，移除三个协议标记。`OmniCommit_Fourth_trans` 逐字复制同一行的 `translation`，包括原有空格、换行和空字符串。模型只接收流式音频、协议提示和本条音频自己的推理历史。

在脚本顶部把 `DATASETS` 中的 TSC、Byte 音频目录和 JSONL 目录改为服务器实际路径，并检查 `MODEL`。运行第一条命令可预览匹配数，第二条命令处理两个数据集。

~~~bash
cd /data/yb/Code/OmniCommit/Fourth
python tools/transcribe_tsc_byte.py --check-only
python tools/transcribe_tsc_byte.py
~~~

也可以直接指定一套数据的目录。

~~~bash
python tools/transcribe_tsc_byte.py \
  --audio-dir /实际路径/TSC音频目录 \
  --jsonl-dir /实际路径/TSC原始JSONL目录 \
  --prefix OmniCommit_Fourth
~~~

默认按文件名匹配，`vid` 可以带音频扩展名。音频分布在多级目录时加 `--recursive`。若 `vid` 含相对目录，同时加 `--vid-mode relative`。JSONL 目录取第一层的 `.jsonl` 文件。WAV 等格式直接读取，读取其他容器时按需调用系统 FFmpeg。

默认重新转录所有匹配音频，包括同前缀 ASR 为空串、已有正文或由 FLEX 填过占位值的记录。成功后两字段一起写回，`_asr` 取本次结果，`_trans` 取本行 `translation`。旧命令中的 `--overwrite` 仍然有效。显式加 `--resume` 时才保留已有字符串 ASR（含空串）并同步译文。单条音频解码失败会打印 `vid` 和路径，保留该行旧值，其他音频继续处理。
