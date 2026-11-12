# 第二版 SFT · WAIT / 正文

一条带时间戳的录音生成一条多轮训练样本。模型每轮听一块新增音频，回答本轮刚刚完整听到的文字。整轮没有可以提交的新文字时，回答独立的 `<|WAIT|>`。开始转录和每次语种或说话人变化时，输出完整标签对，例如 `<en><spk2>`。

本文命令的工作目录是 `Second/`。两个训练版本均加载本目录内的 `SFT-Scripts/check_sft.py` 和 `SFT-Scripts/omnicommit_sft_plugin.py`，使用模板 `omnicommit_second_qwen3_omni`。[训练模板](SFT-Scripts/omnicommit_sft_plugin.py) · [数据预检](SFT-Scripts/check_sft.py)

| 训练环境 | 训练入口 | 第一个位置参数 | 默认保存间隔 |
|---|---|---|---|
| ms-swift 3.12.1 / Megatron | [train_sft.sh](SFT-Scripts/train_sft.sh) | MCore 基座目录 | 3000 步，可传 `--save_interval` |
| ms-swift 4.1.3 / Megatron | [train_sft_swift4.sh](SFT-Scripts/train_sft_swift4.sh) | 原始 HF 基座目录 | 1000 步，可传 `--save_steps` |

本手册到第二版 LoRA checkpoint 为止。仓库顶层的合并器和 Web 服务目前接受第一版的 WAIT / DIRECT 协议。

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
