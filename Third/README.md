# Third · 多语种流式原文转录

Third 每轮接收一块新音频。默认 `explicit` 模式把 assistant 的两种回复训练为 `<|DIRECT|>新增原文` 和独立的 `<|WAIT|>`。同一轮可以提交两种语言的正文，相邻轮次也可以连续提交。语种或说话人变化时仍按文字的时间戳提交。音频结束后 user 发送 `<end_of_audio>`，assistant 回复 `<|WAIT|>`。

部署时将整个 `Third/` 目录复制到服务器，在该目录运行以下命令。数据构建、训练预检、MCore/HF 训练、LoRA 合并和 Web 服务都从本目录加载项目代码。模型、训练框架和数据按服务器原有路径使用。原始语种、说话人和时间戳保存在源数据里。最终训练 JSONL 中的 `audios` 引用原录音区间，训练容器须能读取引用中的音频绝对路径。

~~~bash
cd /data/yb/Code/OmniCommit/Third
~~~

## 1. 生成训练数据

**已有 Second 或旧 Third 纯正文 SFT 时，直接转换现成的训练文件。** 脚本根据每行的 system 指令识别输入协议，按原位置给有正文的轮次加上动作 token。转换保留每个音频引用、原有 WAIT、逐轮正文和接续空格，运行时无需读取原 WAV 或重跑 CTC。

~~~bash
python Data-Scripts/from_second_sft.py \
  /data/yb/Code/OmniCommit/datasets/second_run_001/train.800ms.jsonl \
  --protocol explicit \
  --output /data/yb/Code/OmniCommit/datasets/third_explicit_001/train.800ms.jsonl
~~~

显式模式的 `.tokens.txt` 按 `<|WAIT|>`、`<|DIRECT|>` 的顺序各写一行。若要生成纯正文对照组，对同一个输入指定 `--protocol plain` 和独立的输出路径。两条构建命令都支持该选项。`.rejected.jsonl` 记录转换时跳过的原始行号和原因。

**已有逐段时间戳时，可以从原始标注重铸。** 数据构建可以单独部署完整的 `Third/` 目录，其中 `Data-Scripts/` 放命令入口，`third_asr/` 放时间戳解析、标签重铸和音频引用代码。在服务器的 `Third/` 下运行以下命令即可生成 1 秒训练集。输入中记录的原始音频需要在该服务器可读。

~~~bash
python -m pip install -r requirements-data.txt
python Data-Scripts/build_sft.py \
  /data/yb/Code/OmniCommit/datasets/timestamped.jsonl \
  --protocol explicit --chunk-ms 1000 --release-on semantic-unit \
  --output-dir /data/yb/Code/OmniCommit/datasets/third_recast_1000ms_001
~~~

输入每行含一个原始音频路径、带归属的完整 assistant 参考转录、`output.segments[].source_text`，以及逐段对应的 `timestamps[].start_ms/end_ms`。时间相对于该录音起点，单位毫秒。也可以直接读取已有的 `combined.jsonl`。每条有效录音在 `train.1000ms.jsonl` 中生成一行。构建摘要在 `summary.json`，无效记录的行号及错误在 `rejected.jsonl`。

默认等完整语义段到期再提交。若输入确有更细的、覆盖正文的 `timestamps[].spans[]`，可用 `--release-on timed-span` 采用子段终点。只有切换时刻或另存的字符数组时，先将对齐结果整理为这个输入格式。原蒸馏段中的 `action` 不决定新一轮的 WAIT。用 `--chunk-ms 1000` 可生成 1 秒网格。与已有 800 ms Second 模型做对照时，使用相同的源录音、提交粒度及训练配方。

例如，某个 800 ms 边界前已经说完中文句尾和独立对齐的英文 `Let's`，学生目标可以是：

~~~text
用户音频一       <|DIRECT|>我把时间
用户音频二       <|DIRECT|>发给你了。Let's
用户音频三       <|DIRECT|> meet tomorrow.
<end_of_audio>   <|WAIT|>
~~~

第三轮原文是 `" meet tomorrow."`，开头的空格与前一轮的英文单词相接。如果整个 `Let's meet tomorrow.` 只有一个提交终点，整句会在后面的同一轮出现。

## 2. 检查并训练

`SFT-Scripts/train_sft.sh` 和 `SFT-Scripts/train_sft_swift4.sh` 分别启动 MCore 和 HF 配方。两者的训练脚本、预检、虚拟音频读取和 SWIFT 插件都随 Third 一起部署。预检前，日志会打印 `template=omnicommit_third_qwen3_omni` 和 Third 检查器、插件的实际路径。

MCore 示例使用 1 秒数据，其余示例展示 800 毫秒数据。按实际实验填写训练 JSONL、合并时使用的同名 `.tokens.txt` 和服务的 `--chunk-ms`。在训练服务器上可先检查前两条真实音频，去掉 `--rows 2 --decode` 可进行全量结构预检。

~~~bash
python SFT-Scripts/check_sft.py \
  /data/yb/Code/OmniCommit/datasets/third_explicit_001/train.800ms.jsonl \
  --rows 2 --decode
~~~

**MCore 训练**使用已转换的 MCore 基座。下面的 1 秒实验使用八卡、长度 4096、global batch 64 和 micro batch 4，训练一个 epoch，每 1000 步保存一次。

~~~bash
bash SFT-Scripts/train_sft.sh \
  /data/yb/models/qwen3_omni_mcore_3.12.1 \
  /data/yb/Code/OmniCommit/Third/datasets/data_10000h/train.1000ms.jsonl \
  /data/yb/Code/OmniCommit/output/sft-1117-1S \
  --gpu-ids 0,1,2,3,4,5,6,7 --attention_backend flash \
  --packing false --padding_free false --micro_batch_size 4 --global_batch_size 64 \
  --max_length 4096 --max_epochs 1 --split_dataset_ratio 0 --save_interval 1000
~~~

**ms-swift 4.1.3 / Megatron** 使用原始 Hugging Face 基座，下面沿用 Second 的八卡配方。这是另一个训练入口，同一次实验选择对应环境的一个即可。

~~~bash
bash SFT-Scripts/train_sft_swift4.sh \
  /data/models/qwen3_omni_original_hf \
  /data/yb/Code/OmniCommit/datasets/third_explicit_001/train.800ms.jsonl \
  /data/yb/Code/OmniCommit/outputs/third-explicit-swift4-sft-001 \
  --gpu-ids 0,1,2,3,4,5,6,7 --save_steps 1000
~~~

两个配方都冻结音频编码器和对齐层，对 Thinker 做 LoRA，保存词嵌入和输出层。训练模板为 `omnicommit_third_qwen3_omni`。可在 SWIFT 环境用实际处理器检查序列长度及 assistant 监督位置：

~~~bash
python SFT-Scripts/audit_encoded_lengths.py \
  /data/yb/Code/OmniCommit/datasets/third_explicit_001/train.800ms.jsonl \
  --model /data/models/qwen3_omni_original_hf --rows 2 --max-length 8192 \
  --report /data/yb/Code/OmniCommit/datasets/third_explicit_001/encoded-audit.json
~~~

## 3. 合并 checkpoint

传入原始 HF 基座、实际含 `adapter_config.json` 和适配器权重的 checkpoint、导出目录，以及此次训练的 token 清单。显式模型合并时校验 WAIT 和 DIRECT 的 ID，以及两个 token 在词嵌入和输出层的训练权重，写出 v4 协议清单。旧纯正文模型沿用 v3。要启用强制 DIRECT 解码，需要用显式数据训练并合并的新权重。

~~~bash
bash SFT-Scripts/merge_lora.sh \
  /data/models/qwen3_omni_original_hf \
  /data/yb/Code/OmniCommit/outputs/third-explicit-sft-001/checkpoint-1000 \
  /data/yb/Code/OmniCommit/models/third-explicit-merged \
  --special-tokens /data/yb/Code/OmniCommit/datasets/third_explicit_001/train.800ms.jsonl.tokens.txt \
  --gpu-ids 0
~~~

## 4. 启动内网 Web 转录

在推理环境中使用上一步导出的 HF 模型。默认后端沿用 FLEX 的 vLLM 0.11.1，`--backend hf` 可切换到 Transformers。服务读取本目录内的网页与推理代码。启动时的 `--chunk-ms` 决定当前服务的步长，按训练数据填写。

~~~bash
python -m pip install -r requirements-web.txt
CUDA_VISIBLE_DEVICES=0 bash start_server.sh \
  /data/yb/Code/OmniCommit/models/third-explicit-merged \
  --chunk-ms 800 --port 18891
~~~

在浏览器打开 `http://服务器内网IP:18891`。可在服务器执行 `curl http://127.0.0.1:18891/health` 检查 Third 服务的就绪状态。网页支持麦克风、浏览器共享的系统声音和音视频文件。浏览器远程采音通常需要 HTTPS，localhost 可以直接使用。直接提供 HTTPS 时可传 `--ssl-certfile` 和 `--ssl-keyfile`。系统声音取决于浏览器能否共享所选来源的音频轨道。服务器读取音视频文件需要 `ffmpeg`。服务面向内网，无需访问令牌。

服务启动时读取模型清单，`/health` 中的 `action_protocol` 显示 `explicit` 或 `plain`，`gap_recovery` 显示恢复开关。页面和 TXT 导出展示新增正文。JSONL 导出保留逐轮的动作、原始输出和恢复记录。

### 连续有声 WAIT 的恢复

平时由模型选择动作。对显式模型，服务在 CPU 使用 WebRTC VAD 统计尚未提交的轮次中有多少秒人声。默认累计 2.4 秒人声后，服务用同一台 GPU 上的同一个 Omni 单独重听这一段，解码从当前 assistant 的 `<|DIRECT|>` 之后开始。恢复窗口默认最多 6.4 秒，可带最近最多 0.8 秒的跨界音频。三个时长分别用 `--recovery-voiced-seconds`、`--recovery-window-seconds` 和 `--recovery-overlap-seconds` 设置。静音和结束事件均不触发补识别。

一次连续 WAIT 最多触发一次局部解码，直到再次提交正文。恢复成功时，最终提交进入下一轮的对话历史。调试时查看 `primary_text`、`recovery`、`text` 和 `display_text`，分别核对被替换的原始 WAIT、候选及处理结果、最终协议回复和页面正文。空正文、纯标点、已提交内容的重复等不能触发提交。窗口装不下待处理语音时记录 `window_exceeded` 并保留 WAIT。VAD 只提供人声证据，强制前缀本身无法验证新正文的语义正确性。

在同一份显式模型和同一段测试音频上，用默认配置和 `--no-recovery` 分别启动服务，可比较切换后的漏词、重复词和恢复耗时。旧纯正文模型自动走常规解码。对于始终无法提交的有声音频，服务保留待处理数据直至达到上下文上限，然后报告错误。
