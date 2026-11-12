# 第二版流式 ASR · 数据构建、SFT 与会议服务

一条带时间戳的录音生成一条多轮训练样本。模型每轮听一块新增音频，回答本轮刚刚完整听到的文字。整轮没有可以提交的新文字时，回答独立的 `<|WAIT|>`。开始转录和每次语种或说话人变化时，输出完整标签对，例如 `<en><spk2>`。

本文命令的工作目录是 `Second/`。两个训练版本均加载本目录内的 `SFT-Scripts/check_sft.py` 和 `SFT-Scripts/omnicommit_sft_plugin.py`，使用模板 `omnicommit_second_qwen3_omni`。[训练模板](SFT-Scripts/omnicommit_sft_plugin.py) · [数据预检](SFT-Scripts/check_sft.py)

| 训练环境 | 训练入口 | 第一个位置参数 | 默认保存间隔 |
|---|---|---|---|
| ms-swift 3.12.1 / Megatron | [train_sft.sh](SFT-Scripts/train_sft.sh) | MCore 基座目录 | 3000 步，可传 `--save_interval` |
| ms-swift 4.1.3 / Megatron | [train_sft_swift4.sh](SFT-Scripts/train_sft_swift4.sh) | 原始 HF 基座目录 | 1000 步，可传 `--save_steps` |

第二版训练结束后，使用本目录的 `SFT-Scripts/merge_lora.sh` 导出 HF 模型，再用本目录的 `start_server.sh` 启动会议服务。导出和服务复用仓库顶层实现，部署时请保留完整的 OmniCommit 目录。

## 1. 准备输入

每行是一条 UTF-8 JSON 记录。沿用原有的蒸馏和对齐结果。

- `audios` 含一个原始音频路径。建议在训练服务器挂载音频后构建数据。相对路径默认从输入 JSONL 所在目录解析，也可以指定 `--audio-root`。
- `messages` 中的整段 assistant 转录提供标准文字、语种和说话人。相同标签可重复，省略的标签沿用前面的状态。说话人编号沿用原标注。
- `output.segments` 中每个 `source_text` 是一个可整体提交的语义段。`output` 可为 JSON 对象或序列化的 JSON 字符串。旧 `WAIT` 段与旧 `DIRECT` 段的正文均参与重铸，原标签不决定这一轮的输出。
- `timestamps` 为**每个语义段**提供相对当前音频起点的 `start_ms` 和 `end_ms`，单位毫秒。有索引时用 `index` 对齐，范围为 1 到段数。仅有切换点的 `time_ms` 无法判断普通语义段何时说完，此类记录会进入拒绝清单。脚本会检查全文、归属与时间范围。时间戳本身的声学准确度取决于上游对齐。

时长以原始文件的采样帧数为准。若上游将最后的结束时间向上舍入到整毫秒，导致标注超出文件末尾，该行也会进入拒绝清单。可以据原始对齐记录修正尾点后重新构建。

以下为输入字段示意，写入 JSONL 时将整条记录放在一行。

~~~json
{
  "audios": ["/data/meeting/sample.wav"],
  "messages": [
    {"role": "user", "content": "<audio>"},
    {"role": "assistant", "content": "<zh><spk1>请联系张伟。<en><spk2>Sure."}
  ],
  "output": {"segments": [
    {"source_text": "<zh><spk1>请联系", "action": "DIRECT"},
    {"source_text": "张伟。", "action": "WAIT"},
    {"source_text": "<en><spk2>Sure.", "action": "DIRECT"}
  ]},
  "timestamps": [
    {"index": 1, "start_ms": 50, "end_ms": 420},
    {"index": 2, "start_ms": 430, "end_ms": 1130},
    {"index": 3, "start_ms": 1150, "end_ms": 1470}
  ]
}
~~~

## 2. 按指定步长重铸

在训练服务器上运行。将示例中的 `timestamped.jsonl` 替换成实际的**逐语义段时间戳**文件。

~~~bash
cd /data/yb/Code/OmniCommit/Second
python -m pip install -r SFT-Scripts/requirements.txt

python Data-Scripts/build_sft.py \
  /data/yb/Code/OmniCommit/datasets/timestamped.jsonl \
  --chunk-ms 500 700 800 1000 \
  --output-dir /data/yb/Code/OmniCommit/datasets/second_run_001
~~~

`--chunk-ms` 接受一个或多个不同的正整数，默认 500。沿用虚拟音频读取器的每块 8000 ms 上限。每档独立生成 `train.<毫秒>ms.jsonl` 和同名 `.tokens.txt`。一次选择四档会得到四份训练集，每份各含相同的有效原始记录。训练单一步长时只选一份。

Python 在原始采样率下连续取块。每个边界提交已经说完、且尚未提交的连续语义段。结束时刻恰在边界的段归入本轮。某段只听到一半，就留到后续轮次。切换发生在同一块内时，前段完成就发前段，新段也完成就把两段一起发。一块内可以有多次切换。时间戳粒度决定最早可提交的时刻。例如，单独对齐的 `Sure,` 可以先发。如果整句 `Sure, I will.` 共用一个结束时间，整句一起发。

上面这条输入在 500 ms 步长下的完整输出如下，假设音频长 1.50 秒。

~~~text
0.00–0.50 s      <zh><spk1>请联系
0.50–1.00 s      <|WAIT|>
1.00–1.50 s      张伟。<en><spk2>Sure.
<end_of_audio>   <|WAIT|>
~~~

每行训练数据只含 `audios` 和 `messages`。每个真实块对应一条 user `<audio>` 和一条 assistant 回复。每个音频引用编码原始录音的绝对路径和起止时间。训练插件按区间读取，切块后再转为 16 kHz 单声道。最后不足一步长的音频也有一轮。如需补零，补零仅作用于模型输入长度。文本结束事件单独占一轮。

单行有错会记录到 `rejected.jsonl`，其他行继续处理。已通过的训练行会持续写入，可以用 `head -n 1` 查看。运行结束后，`summary.json` 列出有效行数、拒绝行数及各步长的真实音频轮数、WAIT 轮数。结束事件的 WAIT 不计入音频轮数。报告还记录最长一段没有语义段时间戳覆盖的区间和原始行号，可以据此试听这段是静音还是漏标。输出目录中如已有同名结果，请指定新的输出目录。

## 3. 检查生成的数据

~~~bash
python SFT-Scripts/check_sft.py \
  /data/yb/Code/OmniCommit/datasets/second_run_001/train.700ms.jsonl

python SFT-Scripts/check_sft.py \
  /data/yb/Code/OmniCommit/datasets/second_run_001/train.700ms.jsonl \
  --rows 2 --decode
~~~

第一条全量检查轮次、音频区间、标签继承和 token 清单。第二条实际打开前两条录音并解码所有块，用来核对服务器上的音频路径、采样数和最后一个块。`.tokens.txt` 由构建器自动生成。

在已装好 SWIFT 的训练环境中，可以用原始 HF 处理器核对实际编码长度，并逐轮检查 assistant 文本和损失掩码，包括没有归属标签的续写正文。两条启动脚本的默认上限都是 8192。下面先检查两条，去掉 `--rows 2` 可检查全量。

~~~bash
python SFT-Scripts/audit_encoded_lengths.py \
  /data/yb/Code/OmniCommit/datasets/second_run_001/train.700ms.jsonl \
  --model /data/models/qwen3_omni_original_hf \
  --rows 2 --max-length 8192 \
  --report /data/yb/Code/OmniCommit/datasets/second_run_001/encoded-audit.json
~~~

## 4. 用 FLEX 同款入口训练

下面是 **SWIFT 3.12.1 + MCore** 的四卡示例。训练输出使用容器内 `/data` 下的绝对路径。这个入口默认每 3000 步保存一次，下面显式改成每 1000 步。

~~~bash
cd /data/yb/Code/OmniCommit/Second

bash SFT-Scripts/train_sft.sh \
  /data/data_process/lzy/qwen3_omni_mega/train/qwen3_omni_mcore_3.12.1 \
  /data/yb/Code/OmniCommit/datasets/second_run_001/train.700ms.jsonl \
  /data/yb/Code/OmniCommit/outputs/second-sft-001 \
  --gpu-ids 1,2,3,4 \
  --save_interval 1000
~~~

沿用已能运行该入口的 GPU 环境。训练脚本自动读取所选 JSONL 对应的 token 清单，把 WAIT、语种和说话人标签传给 SWIFT，并将词嵌入和输出层随 LoRA 适配器保存。含正文的 assistant 回复直接从正文或归属标签开始，数据内没有 DIRECT 动作 token。

在 **SWIFT 4.1.3 + Megatron** 环境中使用另一个入口，第一项传原始 HF 模型目录。该配方沿用八卡设置。

~~~bash
bash SFT-Scripts/train_sft_swift4.sh \
  /data/models/qwen3_omni_original_hf \
  /data/yb/Code/OmniCommit/datasets/second_run_001/train.700ms.jsonl \
  /data/yb/Code/OmniCommit/outputs/second-swift4-sft-001 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --save_steps 1000
~~~

若将多档步长混训，应先按原始录音或会议划分训练集和验证集，再分别生成各步长，并在训练时传入独立验证集，以免同一录音的不同版本落入两边。混训所用的 token 清单需覆盖参与训练和验证的全部语种、说话人和 WAIT，可用 `--tokens-file` 指定。

## 5. 导出第二版 checkpoint

在 SFT 环境中运行。第一项传**原始未量化 HF 基座**，第二项传含 `adapter_config.json` 和适配器权重的实际 checkpoint 目录。`--special-tokens` 使用该次训练的原始 token 清单。示例中的目录按服务器实际路径替换。

~~~bash
cd /data/yb/Code/OmniCommit/Second

bash SFT-Scripts/merge_lora.sh \
  /data/models/qwen3_omni_original_hf \
  /data/yb/Code/OmniCommit/outputs/second-sft-001/checkpoint-1000 \
  /data/yb/Code/OmniCommit/models/second-merged \
  --special-tokens /data/yb/Code/OmniCommit/datasets/second_run_001/train.700ms.jsonl.tokens.txt \
  --gpu-ids 0
~~~

如已有第二版导出器生成的完整 HF 目录，直接用它启动服务。第二版导出器在 `omnicommit_tokens.json` 中写入 `protocol.name=omnicommit_second`。启动时会校验协议、token 清单及其真实 ID。从旧流程取得的裸 HF 目录或第一版导出目录，需要从对应的第二版训练适配器和训练清单重新导出。

## 6. 启动第二版会议网页

在 GPU 服务器的推理环境中运行。默认后端沿用仓库已适配的 vLLM 0.11.1，默认监听 `0.0.0.0:18890`，网页默认选中 500 ms，数字格式为 `normalized`。下面示例用一张可见 GPU，并将网页默认值设为 700 ms。单卡可运行的上下文长度取决于模型大小和实际显存。

~~~bash
cd /data/yb/Code/OmniCommit/Second
python -m pip install -r ../requirements-web.txt

CUDA_VISIBLE_DEVICES=0 bash start_server.sh \
  /data/yb/Code/OmniCommit/models/second-merged \
  --chunk-ms 700
~~~

网页地址为 `http://服务器内网IP:18890`，无需访问令牌。浏览器使用麦克风或共享系统声音需要受信任的 HTTPS 或 localhost。已有证书时，可在启动命令末尾加 `--ssl-certfile /path/server.crt --ssl-keyfile /path/server.key`。上传音视频还需要服务器上可执行的 `ffmpeg`。HF 推理可给同一命令追加 `--backend hf`。

~~~bash
curl http://127.0.0.1:18890/health
~~~

确认 `ready` 为 `true`、`protocol` 为 `second`。在网页的「音频来源」下找到「音频块时长」，从下拉列表选择预设，或直接输入整数毫秒数，例如 `200`，再点击「开始转录」。如果服务以 `--chunk-ms 600` 启动，页面默认显示 600 ms。选定值对本场会议的麦克风、系统声音或文件输入生效。会议结束后可为下一场输入新的步长，无需重启模型。健康检查中的 `chunk_seconds` 是启动默认值，`chunk_ms_choices` 列出建议值。每场 WebSocket 握手的 `ready.chunk_seconds` 记录该场实际步长，输入 `200` 时为 `0.2`。`--number-format verbatim` 用于保留口述数字写法的模型。

按默认配置启动，网页可输入 200 至 5000 之间的任意整数。输入框旁显示当前服务允许的范围。若要试 80 ms，可用 `--chunk-ms 80` 启动，页面默认值和下限都会变为 80 ms。

使用小于 500 ms 的步长时，文件播放可在已知的最后一块补零到训练时的最短尾块长度。实时麦克风若恰好结束在一个完整小块上，服务在收到结束事件前已处理该块，因此最后一块的补零长度可能和离线构建不同。

### 第二版模型的实际输出

~~~text
第一轮    <zh><spk1>预算是
第二轮    <|WAIT|>
第三轮    300元。<en><spk1>OK?
第四轮    <en><spk2>Sure,
第五轮     I will.
结束事件  <|WAIT|>
~~~

第三轮开头的“300元。”沿用先前提交的中文与 `spk1`，同轮的 `OK?` 切换到英文。第五轮原始字符串是 `" I will."`，词间空格也是转录的一部分。模型原始回复保存在事件的 `text` 字段。事件层沿用 `DIRECT` 代表本轮提交了正文，`segments` 给出每段的完整归属，方便网页、人工修正和日志导出。发送 `<end_of_audio>` 后模型应答独立的 `<|WAIT|>`。

网页和正文 TXT 会把同一语种、同一显示说话人的连续增量接成一段，例如 `Sure, I will.`。点击段内某次新增的文字可以单独修正其说话人。逐轮日志和 JSON 保留每轮原文。

要独立复现某场会议，回到仓库根目录运行 `tools/diagnose_stream.py`，传入相同的第二版合并模型目录、原音视频文件和网页所用步长。命令行会根据导出清单选用第二版协议。[命令与日志字段](../docs/streaming_runtime.md#音视频文件逐轮诊断)

### 长会议与说话人参照

默认使用滚动上下文。服务会携带裁剪位置之前最后一次提交的语种和说话人，使窗口从纯正文续写开始时也能恢复归属。页面允许修正一个片段、合并或重命名说话人。声音参照按会话保存，人工确认清晰且无重叠的已听音频后可以加入。长时间连续 WAIT 时会受到待提交音频及上下文预算限制。

如需自动选取参照，在同一启动命令末尾追加独立对齐环境与本地模型的路径。默认不开启自动对齐。候选只取自已收到的音频，真实会议中能否认回久未发言的人需要用模型和会议音频验证。

~~~bash
CUDA_VISIBLE_DEVICES=0 bash start_server.sh \
  /data/yb/Code/OmniCommit/models/second-merged --chunk-ms 700 \
  --auto-reference-model /data/yb/models/Qwen3-ForcedAligner-0.6B \
  --auto-reference-python /data/yb/envs/qwen-align/bin/python
~~~

麦克风、系统声音、文件播放、自动参照的语种覆盖与人工修正沿用共用网页。[网页操作和对齐环境安装](../docs/web_meeting_service_zh.md)
