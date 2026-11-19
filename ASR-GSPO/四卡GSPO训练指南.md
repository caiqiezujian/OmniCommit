# OmniCommit Third 四卡 GSPO 训练指南

在原 FLEX-GSPO 容器和虚拟环境中，用 4 张 GPU 对 Third 合并模型做流式 ASR 强化学习。**每条录音在线生成 4 条候选，global batch 16，micro batch 2**。使用整份训练集、1 秒音频块和质量奖励，训练 300 步，每 100 步保存一次，使用 TensorBoard 监视。

## 1. 更新 main 并激活原 FLEX-GSPO 环境

在训练服务器的原容器内执行。后续命令都从同一个 `ASR-GSPO` 目录运行。

```bash
cd /data/yb/Code/OmniCommit
git switch main
git pull --ff-only origin main
cd ASR-GSPO
source /data/yb/vens/flex-gspo-online/bin/activate
mkdir -p logs
```

## 2. 将整份原始 JSONL 转成训练任务

**如果已用整份原始数据按 1000 ms 生成 `datasets/gspo02/tasks.jsonl`，直接执行第 3 步。** 即使以前另建过 `split/`，第 3 步仍然读取整份 `tasks.jsonl`。

将下面的 `timestamped.jsonl` 换成你的原始训练文件。每行的 `audios` 指向容器可读取的录音，`messages` 里的 assistant 正文是完整参考转录。之前蒸馏得到的 `output` 字段可以原样保留。

```bash
bash 01_prepare.sh \
  /data/yb/Code/OmniCommit/datasets/timestamped.jsonl \
  --output /data/yb/Code/OmniCommit/datasets/gspo02/tasks.jsonl \
  --chunk-ms 1000 \
  --max-turns 65 \
  --rejects /data/yb/Code/OmniCommit/datasets/gspo02/rejected.jsonl
```

准备器会打印 `kept` 和 `rejected`。65 轮包含一轮音频结束信号（EOF），因此这组参数最多容纳 64 个一秒音频块。超过 64 秒的录音以及其他无法准备的记录会写入 `rejected.jsonl`，其中有原始行号和原因。要把更长录音纳入本轮训练，需要在第 2、3 步一起调大 `--max-turns`，再重新准备任务。

```bash
wc -l ../datasets/gspo02/tasks.jsonl ../datasets/gspo02/rejected.jsonl
```

## 3. 启动四卡 GSPO

第一个路径是之前用于 Third 服务的、带 `<|DIRECT|>` 和 `<|WAIT|>` 的合并 HF 模型。若本轮选择其他 Third 合并 checkpoint，只需替换第一个路径。第三个路径是本次训练输出目录，首次启动时使用新目录或空目录。

```bash
nohup bash 02_train.sh \
  /data/yb/Code/OmniCommit/output/sft-1S-pack4096-g128/v0-20261118-052054/sft-checkpoint-1000 \
  /data/yb/Code/OmniCommit/datasets/gspo02/tasks.jsonl \
  /data/yb/Code/OmniCommit/output/gspo02-quality-4gpu-g4-b16 \
  --gpu-ids 0,1,2,3 \
  --group-size 4 \
  --micro-batch-size 2 \
  --global-batch-size 16 \
  --max-turns 65 \
  --train-iters 300 \
  --save-steps 100 \
  --reward-mode quality \
  > logs/gspo02-quality-4gpu-g4-b16.log 2>&1 < /dev/null &
```

训练器使用 `split_dataset_ratio=0`，全部准备成功的任务都进入训练采样池。每步抽取 4 个音频任务，每个任务生成 4 条完整候选轨迹。**300 步共 1,200 次音频任务采样、4,800 条候选轨迹。** 同一任务可能被多次抽到。每条原始录音只需提供一份参考转录。

`--reward-mode quality` 已包含切换片段原词、明显异文字替换和句中多余句号的评分。本命令从 Third 合并模型开始，输出到本轮专用的 `gspo02-quality-4gpu-g4-b16`。如果这个目录已经用于其他实验，请同步更换第 3 至 5 步的输出目录和日志名。[评分说明](README.md#3-哪些结果会得高分)

## 4. 启动 TensorBoard

在同一个终端继续执行。训练器已设置 `report_to=tensorboard` 和 `logging_steps=1`。将 TensorBoard 指向本次训练的输出根目录，它会自动找到版本目录下的 `runs/`。如果第 3 步改过输出目录，这里的 `--logdir` 也要改成同一路径。

```bash
nohup python3 -m tensorboard.main \
  --logdir /data/yb/Code/OmniCommit/output/gspo02-quality-4gpu-g4-b16 \
  --host 0.0.0.0 \
  --port 6006 \
  > logs/tensorboard-gspo02-quality-4gpu-g4-b16.log 2>&1 < /dev/null &
```

在浏览器打开 `http://服务器IP:6006`。若 Docker 给 6006 映射了其他宿主机端口，使用映射后的端口。启动时训练器会先检查整份任务文件，再生成第一批候选；首个训练步完成并写入事件后，页面才会有训练曲线。

| TensorBoard 曲线 | 怎么读 |
| --- | --- |
| `train/reward` | 当前训练批次的平均奖励，可以观察候选转录的整体得分趋势。 |
| `train/frac_reward_zero_std` | 同一音频的 4 个候选得分完全相同的组所占比例。长期接近 1 时，大部分组都无法通过奖励区分候选优劣。 |

## 5. 查看训练进度

在同一目录执行下面的命令，查看当前运行到了数据检查、候选生成还是参数更新。按 `Ctrl+C` 可退出日志查看，后台训练继续运行。

```bash
tail -f logs/gspo02-quality-4gpu-g4-b16.log
```

如果 TensorBoard 页面没有曲线，可以查看服务启动日志，并确认训练是否已经生成事件文件。

```bash
tail -n 30 logs/tensorboard-gspo02-quality-4gpu-g4-b16.log
find /data/yb/Code/OmniCommit/output/gspo02-quality-4gpu-g4-b16 -type f -name 'events.out.tfevents.*'
```
