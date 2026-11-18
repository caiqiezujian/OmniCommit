# 给 OmniCommit Third 做流式 ASR 强化学习

输入是原始录音和完整的原语言转录。模型从你们已经训练好的 **Third explicit 合并 HF 权重** 出发，每轮听到一块新音频，自己决定提交哪些正文或继续等。训练器对同一条录音在线采样一组完整转录，再比较识别质量、切换处漏词和出字时间。参考文本、语种和时间戳只交给评分器。

[看一条真实格式的记录如何从原始 JSONL 变成训练任务、逐轮输出和奖励日志](DATA_FORMAT.md)

代码位于 OmniCommit 的 `ASR-GSPO/`，复用同仓库 `Third/` 的音频读取、动作协议和部署校验。训练服务器需要这一整个 OmniCommit checkout。Python/CUDA 环境沿用 FLEX-GSPO 的 **ms-swift 提交 `f15bb91439d756d4a366b51956d72d1f088b7af6`、mcore-bridge 1.6.3 和配套的 Megatron/vLLM**。已有该环境即可使用。数据准备依赖 Python、soundfile，非 16 kHz 波形的实际读取还需要 scipy；本地评分支持 CPU。

## 1. 准备任务

以下 Linux 路径是示例。将输入 JSONL 和合并 SFT 模型目录换成服务器上的真实路径。

```bash
cd /data/yb/Code/OmniCommit/ASR-GSPO

bash 01_prepare.sh \
  /data/yb/Code/OmniCommit/datasets/timestamped.jsonl \
  --output /data/yb/Code/OmniCommit/datasets/gspo01/tasks.jsonl \
  --chunk-ms 1000 \
  --max-turns 65 \
  --rejects /data/yb/Code/OmniCommit/datasets/gspo01/rejected.jsonl

python3 split_data.py \
  /data/yb/Code/OmniCommit/datasets/gspo01/tasks.jsonl \
  --output-dir /data/yb/Code/OmniCommit/datasets/gspo01/split \
  --dev-fraction 0.02 --seed 42
```

示例中的时间相对于当前 WAV 开头。每行一个 JSON 对象，下面把一行展开显示。

```json
{
  "id": "meeting-12-clip-03",
  "meeting_id": "meeting-12",
  "audios": ["/data/asr/meeting-12-clip-03.wav"],
  "messages": [
    {"role": "user", "content": "<audio>"},
    {"role": "assistant", "content": "<zh><spk1>开始。<en><spk2>OK."}
  ],
  "alignment": [
    {"text": "开", "start_ms": 100, "end_ms": 410},
    {"text": "始", "start_ms": 420, "end_ms": 800},
    {"text": "OK", "start_ms": 1100, "end_ms": 1600}
  ]
}
```

- 原始数据中连续出现的相同语种、说话人标签可以保留。准备器提取完整正文和真实语言变化。普通静音样本应有人工或可信标注明确的空转录；缺少 assistant 字段会记作数据错误。
- **有字符或词级 CTC** 时，传 `alignment` 列表或 `chars`、`words`、`word_segments`。字段 `start_ms/end_ms` 用毫秒，`start/end` 用秒。显式 `char_start/char_end` 指去标签后原始正文里的 Python 字符偏移，左闭右开。偏移适合稀疏标注与重复词。单元可用 `trusted:false` 跳过不可信时间。已有 Third 的 `output.segments` 与起止区间、仓库的 `kind:"aligned_asr", schema_version:2, status:"ok"` 文件也可直接读取。
- **只有切换点**（例如 `timestamps[].time_ms=20440`）也能使用。该点可约束新段最早可能出现的位置。词尾缺失时，这条数据仍贡献正文和切换质量分，却无法提供精确的词级出字延迟。如果什么时间戳都没有，保留音频与完整参考即可。
- 叠加 `--rejects` 后，错误行、重复 ID、超回合上限的录音分别写入拒绝文件并继续准备其他录音。默认 65 轮含一次 EOF。1 秒步长时最多有 64 个音频轮；500、800 ms 等通过 `--chunk-ms` 配置，较长样本可同步调大准备和训练的 `--max-turns`，并核对上下文与显存。短尾按 Third 训练规则补齐，提交时间仍取真实音频末尾。
- 数据集复用原 WAV，只写 JSONL 和虚拟切块地址。新建 split 目录中有 `train.jsonl`、`dev.jsonl` 和 `split.json`。划分时将同一 `meeting_id`、原录音标识、`source_ids` 或 `synthesis.clips[].audio` 关联的样本放在一起，包括 A+B、B+C 这类跨样本关联。来自同一会议但文件名各异时，应填写共有的 `meeting_id` 或 `recording_id`。划分依据是提供的来源关系。
- 默认保留自然分布。要定量抽样，可在 split 命令中加 `--train-size 10000 --mix 0.4 0.4 0.2`，分别对应普通、语种切换、稳定性样本。稳定性样本包括有明确空转录的静音，也可在原始数据显式写 `"bucket":"stability"` 纳入长停顿后恢复说话等案例。某类数量不足时会报告可用数。切分遇到拼接数据形成的大连通组，可以先按原始会议划分，再在各自集合内拼接。

## 2. 启动在线 GSPO

```bash
cd /data/yb/Code/OmniCommit/ASR-GSPO

bash 02_train.sh \
  /data/yb/Code/OmniCommit/models/third-merged \
  /data/yb/Code/OmniCommit/datasets/gspo01/split/train.jsonl \
  /data/yb/Code/OmniCommit/output/gspo01-quality-latency \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --reward-mode quality_latency \
  --group-size 8 --micro-batch-size 2 --global-batch-size 64 \
  --train-iters 300 --save-steps 100
```

这条命令的 300 步是第一组比较实验的预算，实际迭代和显存占用需在服务器上测。先验证训练环境时，可使用独立输出目录、`--train-iters 2 --save-steps 1`，这能覆盖首次 rollout、反向更新、权重同步、第二次 rollout 和保存。仅检查最终参数可在原命令后加 `--dry-run`。已有 FLEX 项目通过本地源码加载 Swift 的话，把同一个 `--swift-source /path/to/ms-swift` 加到训练命令。

训练沿用 FLEX 的 colocated vLLM 采样、Megatron TP2/EP2、bf16 和 Thinker LoRA。冻结声学编码器与对齐层。每一组的 8 条候选看到相同音频，随各自的历史预测继续。4 卡用 `--gpu-ids 0,1,2,3 --global-batch-size 32`。8 卡示例每次生成对应 8 条不同录音 × 每条 8 条候选。GSPO 在 Swift 的 GRPO trainer 中使用 `importance_sampling_level=sequence`，按序列计算比率并裁剪，同组奖励中心化，KL 单独计入 loss。

对照实验保持同一个 SFT 起点、划分、采样和训练步数，另设一个输出目录（如 `gspo01-quality`）并改 `--reward-mode quality`。这组只优化整段与切换处的转录。保存时，原生 `checkpoint-N` 可用 `--resume-from` 续训，保持原训练配置及最终 `--train-iters`；同次导出的 `checkpoint-N-merged` 用于评测和部署。

采样返回 `<|WAIT|>` 时本轮立即结束，与 Third 部署时一致。训练保留这一动作的实际 token ID 和采样 log probability。模板为了排版插入的轮间分隔符不参与策略 loss，用户音频也不参与策略 loss。

如果这批录音以前全部参加过 SFT，这次划出的 dev 集只能检验 GSPO 阶段的变化。要判断真实会议能否泛化，请另外留一批 SFT 和 GSPO 都没有见过的录音，尤其保留人工核对过的短语种切换和长静音后重新开口的片段。

## 3. 哪些结果会得高分

假定上例最后 1 秒是静音，两个词在第 1、2 秒各自说完。三条候选按 1 秒一轮排列，末项为 EOF。

```text
按时出字    <|DIRECT|>开始。   <|DIRECT|>OK.    <|WAIT|>            <|WAIT|>
同文晚出    <|WAIT|>          <|WAIT|>         <|DIRECT|>开始。OK.  <|WAIT|>
切换漏词    <|DIRECT|>开始。   <|WAIT|>         <|WAIT|>            <|WAIT|>
```

前两条在 `quality` 模式并列。在此例的字符时间戳下，三条轨迹的 `quality_latency` 分别为 1.000、0.957、0.017（四舍五入）。一次提交可以包含前一种语言的句尾和新语言的开头，切换或静音都不会触发固定等待扣分。只有音频输入轮可提交正文，随后一次文本 `<end_of_audio>` 对应单独的 `<|WAIT|>`。

| 目标 | 当前实现 |
|---|---|
| 原语言转录正确 | 拼接各轮新增正文后全局做编辑对齐。中日韩等按字符单位，带空格的多数语种按词单位。记录插入、删除、替换；提前胡编和持续重复会影响分数。 |
| 切换短段有覆盖 | 每次真实语言变化后，给前 3 个计分单位的错误额外扣分。整段最多 3 单位且被删光时另扣一次。同语种换人不触发该项。标签只用于定位参考区间，输出语言的判断依据是原语言正文。两种语言里的同形同音词无法仅靠转录区分。 |
| 已经说完的词及时出现 | 按参考单元结束后第一个可提交块计算额外等待。举例来说，词在 1.6 秒结束、2 秒提交，额外等待为 0；拖到 5 秒为 3 秒。可用起点还可标记声音出现前的预测。整块语义区间仅使用可证实的边缘，不给内部每个词虚构时间。 |

基础分是 `1 − min(2, 全文混合单位错误率)`。再扣 `0.40 × 平均切换首段错误率` 和 `0.20 × 短切换段全删比例`。`quality_latency` 另扣 `0.15 × 时间成本` 和 `0.10 × 提前命中比例`。额外延迟 d 秒的成本为 `d/(4+d)`，有时间监督的漏词或明显抢跑按最大时间成本计。无词命中的有声参考、纯静音上的幻觉、协议错或截断有单独处理。权重是本项目可对照的初始设定。具体实现见 [reward.py](asr_gspo/reward.py)。

## 4. 评测与 Third 服务导出

先用相同的留出集分别评测 SFT、质量 GSPO、质量加时延 GSPO。评测使用 Third 后端逐块贪心解码，读取同一套音频切块，保留完整轮次。这样可以单独测模型策略，不混入生产服务里的超时重试或 VAD 补识别。

```bash
cd /data/yb/Code/OmniCommit/ASR-GSPO

CUDA_VISIBLE_DEVICES=0 python3 evaluate.py model \
  /data/yb/Code/OmniCommit/datasets/gspo01/split/dev.jsonl \
  --model /data/yb/Code/OmniCommit/models/third-merged \
  --backend vllm \
  --output /data/yb/Code/OmniCommit/output/eval/sft.jsonl

find /data/yb/Code/OmniCommit/output/gspo01-quality-latency -type d -name 'checkpoint-*-merged'
```

下面的 `<实际版本目录>` 请替换成上一条命令列出的目录名。导出检查会验证合并 HF 权重分片、processor、词表以及动作 token 的 ID，并补齐 Third 服务读取的协议清单。

```bash
python3 finalize_export.py \
  /data/yb/Code/OmniCommit/models/third-merged \
  '/data/yb/Code/OmniCommit/output/gspo01-quality-latency/<实际版本目录>/checkpoint-100-merged'

CUDA_VISIBLE_DEVICES=0 python3 evaluate.py model \
  /data/yb/Code/OmniCommit/datasets/gspo01/split/dev.jsonl \
  --model '/data/yb/Code/OmniCommit/output/gspo01-quality-latency/<实际版本目录>/checkpoint-100-merged' \
  --backend vllm \
  --output /data/yb/Code/OmniCommit/output/eval/gspo100.jsonl
```

输出每条录音的逐轮原文、终止状态、奖励及分项，同时生成 `*.summary.json`。汇总含有效轨迹的全文错误率、协议无效率、短段整段漏掉数、全体及切换开头的额外等待 P95（按匹配单元合并统计）。时延要和 `timed_units`、`timed_matches` 一起读。时延以音频时钟计，HTTP 排队、模型计算耗时与页面渲染应另外测。模型合格后，可把同一合并目录交给原有的 [Third HTTP / Web 服务](../Third/README.md)。

训练输出目录的 `rewards/rollouts-rank*-pid*.jsonl` 保存每个候选的逐轮文本、奖励和计分项，不保存 WAV 副本。取一份 rank 日志重算评分时，传入当时实际参与训练的 manifest：

```bash
python3 evaluate.py replay \
  /data/yb/Code/OmniCommit/datasets/gspo01/split/train.jsonl \
  /path/to/rollouts-rank0-pid12345.jsonl \
  --reward-mode quality_latency --limit 100 \
  --output /data/yb/Code/OmniCommit/output/eval/replay-first100.jsonl
```

## 测试与依据

```bash
cd /data/yb/Code/OmniCommit
python3 -m pytest -q ASR-GSPO/tests
# 如本机有固定版 Swift 源码，同时验证其多轮 driver 与训练样本的连接
ASR_GSPO_SWIFT_SOURCE=/path/to/ms-swift python3 -m pytest -q ASR-GSPO/tests
```

序列级比率与 clipping 取自 [GSPO 原论文](https://arxiv.org/abs/2507.18071)。用可验证的原语言转录误差优化 code-switch ASR 有 [Ye 与 Vickers 的实验](https://arxiv.org/abs/2607.02757)；这里用真实切换后的局部编辑误差，可覆盖英语、德语等共享文字系统的组合。按已经播到的音频位置检查提交文本，也可对照 [X2Streaming-ASR 的流式评测](https://github.com/X-Square-Robot/X2Streaming-ASR)。本项目的切换权重与按块延迟函数需要用自己的会议数据验证。
