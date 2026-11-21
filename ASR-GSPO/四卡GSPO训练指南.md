# Fourth 四卡 / 八卡 GSPO：8 条轨迹、2000 步、每 500 步保存

沿用原 FLEX-GSPO 容器和虚拟环境，不自动划分验证集。使用 Fourth 合并 SFT HF 模型。正文取原始 messages 中的完整 assistant.content；语种和说话人标签留在评分侧，旧 output 的 DIRECT/WAIT 不作为新轨迹答案。下面模型路径是需替换的示例，数据使用此前提供的原始带时间戳文件，无需先做语义合并。

当前奖励版本 `fourth_boundary_v2`。本指南采用“检查末尾标点和最后 end，再以 keep + tail-only 准备任务”的策略。该批数据优化全文 CER、局部语种、换人及断句质量；最后时间戳仅用于预筛选，不伪造逐词或切换延迟奖励。旧 tasks 不当作本次 tail-only 数据，重新准备到新目录。新奖励实验从 Fourth SFT 开始，不续接旧奖励 run 的优化器。

## 1. 环境

在原容器更新代码并进入训练环境：

~~~bash
cd /data/yb/Code/OmniCommit
git pull --ff-only origin main
cd ASR-GSPO
source /data/yb/vens/flex-gspo-online/bin/activate
mkdir -p logs
~~~

不升级 Swift/Megatron/vLLM。没有 RapidFuzz 时 CER 用较慢的 Python 实现；需要加速可在该环境执行 python3 -m pip install rapidfuzz。

## 2. 按末尾标点与最后时间戳过滤

若真实末尾 DIRECT 没有标点，整行保留，即使不足 1 秒。若结尾有标点，才判断 `原 WAV 总时长 - 最后正文 end > 1000 ms`，严格大于保留，小于或等于过滤。完整原文为准，不能找到前面某个 DIRECT 就忽略其后正文；旧 WAIT 段有正文时也不删除。筛选不重新切分、不修改音频或文字。脚本顶部有默认路径，也可按以下命令显式指定：

~~~bash
python3 filter_tail.py \
  /data/s50062929/FLEX/SFT_data/full_10000h_augmented.jsonl \
  --output /data/s50062929/FLEX/SFT_data/full_10000h_augmented.gspo_tail_checked.jsonl
~~~

只读音频文件头。带标点的末尾需要相关尾段的可信 end；只有切换起点 time_ms 不能替代。无标点结尾不要求拿时间戳决定保留。需要的尾部信息冲突/缺失、音频无法读取等情况写入输出旁的 `.rejected.jsonl`，随后继续处理；`.report.json` 给出数量和原因。保留行原样复制，原 JSONL/WAV 不改动，不做备份、不另存音频。已有输出不会被覆盖。

用户提供的俄中样例末段为“和团队合作的生活方式而闻名。”，end 为 26080 ms；有句号，因此原 WAV 时长必须严格大于 27080 ms 才保留。若对应最后 DIRECT 没有末尾标点，则不因间隔短而排除；即使完整参考仍保留句号，也不改写正文，而是标记 `terminal_punctuation_trusted:false`，只关闭末尾标点的奖励要求，句内标点继续评分。样例没有提供真实 WAV 时长，不能直接判断保留或过滤。差值不是经过 VAD 验证的静音，也不能单凭它证明句号语义正确。

## 3. 准备 Fourth 任务

使用过滤后的源文件，选择 `--tail-policy keep --timing-mode tail-only`，保留完整音频和正文。不再要求旧 output 的每个切分及全部 timestamps 逐字重建原文；中间段落空白重新分配不会因此使完整转录丢失。音频仍由 GSPO 调度器按 1 秒增量读取，不另存小 WAV，也不预先写死每轮动作。

~~~bash
bash 01_prepare.sh \
  /data/s50062929/FLEX/SFT_data/full_10000h_augmented.gspo_tail_checked.jsonl \
  --output /data/yb/Code/OmniCommit/datasets/gspo04-tail-only/tasks.jsonl \
  --chunk-ms 1000 \
  --max-turns 65 \
  --tail-policy keep \
  --timing-mode tail-only \
  --rejects /data/yb/Code/OmniCommit/datasets/gspo04-tail-only/rejected.jsonl
~~~

检查 kept/rejected 和拒绝文件。此 keep 流程最多 65 个交互轮，含一个 EOF 轮。过长记录会写入 rejected；需要更长录音时，准备和训练的 max-turns 一起增加，并核对上下文预算。

相对音频路径以原始 JSONL 所在目录为基准。若过滤文件移动到了另一个目录，过滤和准备命令均显式加 `--audio-root 原始音频路径基准目录`。本指南示例把过滤文件放在原目录，因此无需追加。

tail-only 从完整原文生成无标签 reference_text、语言 spans 与真正的 speaker_changes，全部保留在评分侧。原始旧 WAIT 中的“Господи!”、“Значит,”、“150克之间，”等正文照常参与评分。Fourth 输出仍为 DIRECT / WAIT / SPK_CHANGE，不生成 `<ru>`、`<zh>` 或 `<spk1>`。

本模式 `alignment=[]`，不提供 switch 时间条目，因此 `timed_units=0`、`latency_penalty=0`。CER、局部原语种内容、换人边界和标点仍然评分。不能把最后一个 end 当作整句话每个字的时间。如果以后希望用完整可信时间标注优化延迟，可另行使用 `--timing-mode aligned`；该模式会保留严格对齐校验，不是当前样例的默认路径。

## 4. FLEX 的四卡 / 八卡 batch 对照

已对照 FLEX 当前 `06_train_qalco_v4.sh`、`07_train_qalco_v5.sh` 及训练入口：

| 参数 / 行为 | 四卡 | 八卡 |
| --- | --- | --- |
| group size | 8 | 8 |
| micro batch | 2 | 2 |
| global batch（候选轨迹数） | 32 | 64 |
| 每步原始录音抽样数 | 4 | 8 |
| 训练 TP / EP / DP | 2 / 2 / 2 | 2 / 2 / 4 |
| vLLM 采样组 | 1 组 TP4 | 2 组 TP4 |
| steps per generation | 1 | 1 |

micro batch 是每个训练 DP rank 的 micro batch；global batch 按候选轨迹计。八卡 G8/micro2/B32 不满足 DP4 的 prompt 分配，要使用 B64。入口未显式设置 global batch 时也会推导 32/64。保留 Omni 现有 vLLM 显存比例 0.40、并发上限 16，没有直接照搬 FLEX 的 0.50/32。实际八卡吞吐和显存需要服务器验证，不能凭卡数保证两倍速度。

## 5. 四卡训练

第一项替换为本次 Fourth SFT 合并模型。目录必须包含已经训练过三种动作/换人 token 的 tokenizer 和 omnicommit_tokens.json。

~~~bash
nohup bash 02_train.sh \
  /data/yb/Code/OmniCommit/models/fourth-sft-merged \
  /data/yb/Code/OmniCommit/datasets/gspo04-tail-only/tasks.jsonl \
  /data/yb/Code/OmniCommit/output/gspo04-tail-only-4gpu-g8-b32-2000 \
  --gpu-ids 0,1,2,3 \
  --group-size 8 \
  --micro-batch-size 2 \
  --global-batch-size 32 \
  --steps-per-generation 1 \
  --max-turns 65 \
  --train-iters 2000 \
  --save-steps 500 \
  --save-total-limit 6 \
  --reward-mode quality_latency \
  --switch-grace-turns 1 \
  > logs/gspo04-tail-only-4gpu-g8-b32-2000.log 2>&1 < /dev/null &
~~~

每步抽取 4 条录音，每条 8 条完整轨迹。2000 步共 8000 次录音抽取、64000 条轨迹。同一条可能重复采样，全部文件进入采样池不等于走完一个 epoch。

`--switch-grace-turns 1` 仅在 aligned 数据具有可信完成时间时提供确认宽限；本次 tail-only 没有这项时间奖励，不强迫模型 WAIT。保留 quality_latency 模式以复用已有训练入口，不代表无时间标注也能直接优化延迟。

要训 **2500 步**，启动时把 train-iters 改成 2500，并更换输出目录和日志名；仍每 500 步保存。要每 1000 步保存，用 --save-steps 1000。建议先保留 500 间隔，便于对比早期和后期切换/句号质量。

## 6. 八卡训练

与四卡命令二选一，使用独立输出目录。学习率 1e-6、KL 0.01、LoRA 和冻结声学部分等沿用当前配置。

~~~bash
nohup bash 02_train.sh \
  /data/yb/Code/OmniCommit/models/fourth-sft-merged \
  /data/yb/Code/OmniCommit/datasets/gspo04-tail-only/tasks.jsonl \
  /data/yb/Code/OmniCommit/output/gspo04-tail-only-8gpu-g8-b64-2000 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --group-size 8 \
  --micro-batch-size 2 \
  --global-batch-size 64 \
  --steps-per-generation 1 \
  --max-turns 65 \
  --train-iters 2000 \
  --save-steps 500 \
  --save-total-limit 6 \
  --reward-mode quality_latency \
  --switch-grace-turns 1 \
  > logs/gspo04-tail-only-8gpu-g8-b64-2000.log 2>&1 < /dev/null &
~~~

八卡 2000 步对应 16000 次录音抽样、128000 条候选，采样量是四卡同样步数的两倍。八卡也可启动时选择 2500 步。四卡/八卡切换改变 DP 布局，不能作为同一个 run 的精确断点续训。

## 7. TensorBoard 和日志

下面以八卡实验为例；四卡替换为对应的 4gpu-g8-b32 目录与日志。

~~~bash
nohup python3 -m tensorboard.main \
  --logdir /data/yb/Code/OmniCommit/output/gspo04-tail-only-8gpu-g8-b64-2000 \
  --host 0.0.0.0 --port 6006 \
  > logs/tensorboard-gspo04.log 2>&1 < /dev/null &

tail -f logs/gspo04-tail-only-8gpu-g8-b64-2000.log
~~~

浏览器访问 http://服务器IP:6006，Docker 映射时使用宿主机端口。

| 曲线 | 含义 |
| --- | --- |
| train/reward | 总奖励 |
| train/frac_reward_zero_std | 8 个候选奖励完全相同的组比例 |
| train/asr_cer | 当前采样字符错误率，不是测试集 CER |
| train/asr_language_penalty | 切换原词、局部 CER 和异文字罚分 |
| train/asr_speaker_penalty | 换人边界及文字归属的总罚分 |
| train/asr_speaker_attribution_penalty | 局部文字归错段及新段删除的罚分 |
| train/asr_speaker_misattributed_characters | 本日志区间识别正确但归错段的字符量 |
| train/asr_punctuation_penalty | 不合理句号及漏句末罚分 |
| train/asr_latency_penalty | tail-only 为 0；aligned 才可能有等待成本 |
| train/asr_extra_internal_periods | 日志区间额外句号数量 |
| train/asr_unsupported_short_breaks | 无根据的句子终止符造成短碎句数量 |
| train/asr_extra_internal_stops | 不受参考支持的句号、问号、感叹号总数 |
| train/asr_switch_grace_used_units | 使用了切换确认宽限的计时单元数量 |
| train/asr_attributed_timing_coverage | 可计时单元中拼写正确且未检测到归属错误的覆盖率 |

分项由各 rank 小型计数文件合并；完整候选在输出根目录 rewards/rollouts-rank*-pid*.jsonl，不额外保存音频。

无细粒度时间戳时，等待成本为零不表示真实推理零延迟。必须在真实流式测试中看响应；不能把未知时间当成零时间。

## 8. 保存及导出检查

~~~bash
find /data/yb/Code/OmniCommit/output/gspo04-tail-only-8gpu-g8-b64-2000 -type d -name 'checkpoint-*-merged'
~~~

每次同步保存完成会自动补齐/校验服务文件，成功打印 [export] Verified Fourth HF checkpoint。目标目录 omnicommit_export_status.json 中 ready:true 表示静态检查完成。原生 checkpoint-N 用于续训，Web 使用 checkpoint-N-merged。

若出现 [export ERROR]，训练继续但该合并目录未通过部署检查。修复日志中说明的问题后可重跑补全器：

~~~bash
python3 finalize_export.py \
  /data/yb/Code/OmniCommit/models/fourth-sft-merged \
  /data/yb/Code/OmniCommit/output/gspo04-tail-only-8gpu-g8-b64-2000/实际版本目录/checkpoint-500-merged
~~~

第一个模型必须与 RL 启动路径一致，不从其他模型混拷文件。

## 9. 启动 Fourth Web

把实际版本目录替换为上一步找到的目录名，使用空闲 GPU：

~~~bash
cd /data/yb/Code/OmniCommit
CUDA_VISIBLE_DEVICES=0 bash Fourth/start_server.sh \
  /data/yb/Code/OmniCommit/output/gspo04-tail-only-8gpu-g8-b64-2000/实际版本目录/checkpoint-500-merged \
  --chunk-ms 1000 \
  --window-seconds 60 \
  --discard-seconds 50 \
  --port 18899
~~~

采集电脑声音时追加既有的 --https-host IP或域名 --cert-dir 证书目录。服务环境沿用已验证的 Fourth 服务环境，和 FLEX-GSPO 训练环境分别管理。

## 10. 固定测试集逐轮评测

使用既有独立测试集准备的 Fourth tasks，不从用户提供的训练文件自动划验证集。先评测原 SFT，再对 500/1000/1500/2000 保存点执行同样命令，只替换模型和输出文件名：

~~~bash
cd /data/yb/Code/OmniCommit/ASR-GSPO
CUDA_VISIBLE_DEVICES=0 python3 evaluate.py model \
  /data/yb/Code/OmniCommit/datasets/gspo04-test/tasks.jsonl \
  --model /data/yb/Code/OmniCommit/output/gspo04-tail-only-8gpu-g8-b64-2000/实际版本目录/checkpoint-500-merged \
  --backend vllm \
  --switch-grace-turns 1 \
  --output /data/yb/Code/OmniCommit/datasets/gspo04-test/step500.jsonl
~~~

测试路径是需替换的示例。GPU/推理环境要求与已验证的 Fourth 评测一致。输出旁边的 `.summary.json` 会分别列出 `boundary/speaker_only`、`boundary/language_only`、`boundary/both` 和 `boundary/none`（无样本的类不出现）。判断依据是 CER、归错字/漏换人、短语漏转录和错误断句。只有测试集带可信详细时间戳且以 aligned 准备时，才解释正确提交延迟与计时覆盖；tail-only 的未知时间不当成零延迟。
