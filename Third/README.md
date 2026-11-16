# Third · 多语种流式原文转录

Third 每轮接收一块新音频。能提交正文时，assistant 直接输出原语言的新增文字；没有新增正文时输出独立的 `<|WAIT|>`。一轮可提交多种语言，相邻轮次可以连续出正文。音频结束后，user 发送 `<end_of_audio>`，assistant 回复 `<|WAIT|>`。训练额外注册一个特殊 token：`<|WAIT|>`。

从 Third 目录运行以下命令。原始语种、说话人和时间戳继续保存在源数据里，供检查语种切换和漏词时使用。最终训练 JSONL 只含 `audios` 和 `messages`，其中 `audios` 是指向原录音区间的虚拟引用；训练容器须能访问引用中的音频绝对路径。

~~~bash
cd /data/yb/Code/OmniCommit/Third
~~~

## 1. 生成训练数据

**已有 Second 的 800 ms SFT 文件时，优先直接转换。** 这个入口逐条校验并转换 Second 消息，每个音频引用、已有 WAIT 的位置、原语言正文和词间空格都保留。无需原 WAV 在线，也无需再跑 CTC。

~~~bash
python Data-Scripts/from_second_sft.py \
  /data/yb/Code/OmniCommit/datasets/second_run_001/train.800ms.jsonl \
  --output /data/yb/Code/OmniCommit/datasets/third_run_001/train.800ms.jsonl
~~~

输出旁的 `.tokens.txt` 供训练与合并使用，`.rejected.jsonl` 记录被跳过的原始行号和原因。用于 Second/Third 对照时，应确认拒绝行数为零，或在两组都成功的相同行上比较。

**已有逐段时间戳时，可以从原始标注重铸。** 下面是上面转换命令的另一条入口。示例用独立目录，便于比较。复用 Second 数据准备环境，也可以先安装 `python -m pip install -r ../Second/SFT-Scripts/requirements.txt`。

~~~bash
python Data-Scripts/build_sft.py \
  /data/yb/Code/OmniCommit/datasets/timestamped.jsonl \
  --chunk-ms 800 --release-on semantic-unit \
  --output-dir /data/yb/Code/OmniCommit/datasets/third_recast_001
~~~

输入每行含一个原始音频路径、带归属的完整 assistant 参考转录、`output.segments[].source_text`，以及逐段对应的 `timestamps[].start_ms/end_ms`。时间相对于该录音起点，单位毫秒。可以直接使用 Second 拼接脚本生成的 `combined.jsonl`。每条有效录音在 `train.800ms.jsonl` 中生成一行。构建摘要在 `summary.json`，无效记录的行号及错误在 `rejected.jsonl`。[Second 输入示例](../Second/README.md#1-准备输入)

默认等完整语义段到期再提交。若输入确有更细的、覆盖正文的 `timestamps[].spans[]`，可用 `--release-on timed-span` 采用子段终点。只有切换时刻或另存的字符数组时，先将对齐结果整理为这个输入格式。原蒸馏段中的 `action` 不决定新一轮的 WAIT。用 `--chunk-ms 1000` 可生成 1 秒网格。与已有 800 ms Second 模型做对照时，使用相同的源录音、提交粒度及训练配方。

例如，某个 800 ms 边界前已经说完中文句尾和独立对齐的英文 `Let's`，学生目标可以是：

~~~text
用户音频一       我把时间
用户音频二       发给你了。Let's
用户音频三        meet tomorrow.
<end_of_audio>   <|WAIT|>
~~~

第三轮原文是 `" meet tomorrow."`，开头的空格与前一轮的英文单词相接。如果整个 `Let's meet tomorrow.` 只有一个提交终点，整句会在后面的同一轮出现。

## 2. 检查并训练

下面的训练和导出示例使用第一个入口生成的文件。若用第二个入口，将数据路径换成 `third_recast_001` 中的对应文件。在训练服务器上先检查前两条真实音频，去掉 `--rows 2 --decode` 可进行全量结构预检。

~~~bash
python SFT-Scripts/check_sft.py \
  /data/yb/Code/OmniCommit/datasets/third_run_001/train.800ms.jsonl \
  --rows 2 --decode
~~~

**ms-swift 3.12.1 / MCore** 使用已转换的 MCore 基座。下面按 4 卡、每 1000 步保存给出，训练结果写入容器的 `/data` 挂载目录。

~~~bash
bash SFT-Scripts/train_sft.sh \
  /data/data_process/lzy/qwen3_omni_mega/train/qwen3_omni_mcore_3.12.1 \
  /data/yb/Code/OmniCommit/datasets/third_run_001/train.800ms.jsonl \
  /data/yb/Code/OmniCommit/outputs/third-sft-001 \
  --gpu-ids 1,2,3,4 --save_interval 1000
~~~

**ms-swift 4.1.3 / Megatron** 使用原始 Hugging Face 基座，下面沿用 Second 的八卡配方。这是另一个训练入口，同一次实验选择对应环境的一个即可。

~~~bash
bash SFT-Scripts/train_sft_swift4.sh \
  /data/models/qwen3_omni_original_hf \
  /data/yb/Code/OmniCommit/datasets/third_run_001/train.800ms.jsonl \
  /data/yb/Code/OmniCommit/outputs/third-swift4-sft-001 \
  --gpu-ids 0,1,2,3,4,5,6,7 --save_steps 1000
~~~

两个配方都冻结音频编码器和对齐层，对 Thinker 做 LoRA，保存词嵌入和输出层。训练模板为 `omnicommit_third_qwen3_omni`。可在 SWIFT 环境用实际处理器检查序列长度及 assistant 监督位置：

~~~bash
python SFT-Scripts/audit_encoded_lengths.py \
  /data/yb/Code/OmniCommit/datasets/third_run_001/train.800ms.jsonl \
  --model /data/models/qwen3_omni_original_hf --rows 2 --max-length 8192 \
  --report /data/yb/Code/OmniCommit/datasets/third_run_001/encoded-audit.json
~~~

## 3. 合并 checkpoint

传入原始 HF 基座、实际含 `adapter_config.json` 和适配器权重的 checkpoint、导出目录，以及该次训练使用的 token 清单。导出器校验 WAIT 的真实 token ID、词嵌入和输出权重，并写入 Third 的协议清单。

~~~bash
bash SFT-Scripts/merge_lora.sh \
  /data/models/qwen3_omni_original_hf \
  /data/yb/Code/OmniCommit/outputs/third-sft-001/checkpoint-1000 \
  /data/yb/Code/OmniCommit/models/third-merged \
  --special-tokens /data/yb/Code/OmniCommit/datasets/third_run_001/train.800ms.jsonl.tokens.txt \
  --gpu-ids 0
~~~

## 4. 启动内网 Web 转录

在推理环境中使用上一步导出的 HF 模型。默认后端沿用 FLEX 的 vLLM 0.11.1，`--backend hf` 可切换到 Transformers。将整个 OmniCommit 仓库部署到同一台机器，训练、导出和服务都会使用仓库内的共享组件。启动时的 `--chunk-ms` 决定当前服务的步长，按训练数据填写。

~~~bash
python -m pip install -r ../requirements-web.txt
CUDA_VISIBLE_DEVICES=0 bash start_server.sh \
  /data/yb/Code/OmniCommit/models/third-merged \
  --chunk-ms 800 --port 18891
~~~

在浏览器打开 `http://服务器内网IP:18891`。可在服务器执行 `curl http://127.0.0.1:18891/health` 检查 Third 服务的就绪状态。网页支持麦克风、浏览器共享的系统声音和音视频文件。浏览器远程采音通常需要 HTTPS，localhost 可以直接使用。直接提供 HTTPS 时可传 `--ssl-certfile` 和 `--ssl-keyfile`。系统声音取决于浏览器能否共享所选来源的音频轨道。服务器读取音视频文件需要 `ffmpeg`。服务面向内网，无需访问令牌。

页面通过 TXT 导出已提交的全文，通过 JSONL 导出每轮原始回复和推理耗时。日志中 `action` 的值 `DIRECT` 是服务端对“本轮有正文”的标记。模型生成的 `text` 可直接与对应的训练目标比较。长录音采用滚动上下文，已显示的全文供浏览与导出，近期音频和对话供下一轮推理。模型连续对有声输入给出 `WAIT` 并达到待提交音频上限时，服务会报告错误，页面保留已提交的正文。
