# OmniCommit Fourth 流式 ASR GSPO

从 Fourth 合并后的 SFT HF 模型出发，每条录音在线生成 **8 条完整轨迹**。每轮只追加已到达的音频，评分参考不会送给 actor。支持四卡 / 八卡：micro batch 2，global batch 分别为 32 / 64。默认 1 秒新音频、2000 步、每 500 步保存，开启 TensorBoard，不划分验证集。

- [四卡 / 八卡完整运行指南](四卡GSPO训练指南.md)
- [原始数据到在线轨迹](DATA_FORMAT.md)
- [奖励公式、论文依据及限制](REWARD_DESIGN.md)

**旧 Third tasks.jsonl 必须重新准备；起点必须是 Fourth 的合并 HF 模型。**

## 环境和协议

复用同仓库 Fourth/fourth_asr 的协议、音频读取和末尾投影，需要整个 OmniCommit checkout。GPU 环境沿用 FLEX-GSPO 的 ms-swift 提交 f15bb91439d756d4a366b51956d72d1f088b7af6、mcore-bridge 1.6.3 及原有 Megatron/vLLM。不要仅凭发行版本号替换镜像。数据准备需要 soundfile，非 16 kHz 音频按需使用 scipy。评分优先用 RapidFuzz，缺少时回退到同定义的 Python 编辑距离。

~~~text
<|DIRECT|>新增正文
<|WAIT|>
<|DIRECT|>前人的句尾。<|SPK_CHANGE|>新人的开头。
~~~

同人换语言不加换人标记；换人不强制 WAIT。SPK_CHANGE 表示变化，不是持久身份编号。

当前奖励版本为 `fourth_boundary_v2`：全文 CER、切换原文和换人位置共同核对，并按归错段文字量评分。Fourth 协议保持 DIRECT / WAIT / SPK_CHANGE。确认宽限需要可信详细时间戳；当前默认 tail-only 数据只用最后 end 做末尾筛选，不产生逐词延迟奖励。

## 准备与训练

完整参考只取 `messages` 中的 `assistant.content`。移除 `<ru>`、`<zh>` 等语种标签得到正文，标签信息用于评分侧语言区间；说话人标签生成真正换人边界。原 output 的 WAIT/DIRECT 不规定新的动作，WAIT 中的正文也保留。默认 `--timing-mode tail-only` 不再严格重建旧中间切分，因此旧段落重新分配空格不会误伤完整转录。

先用 `filter_tail.py` 检查末尾：真实末尾无标点则保留，即使离音频结束不足 1 秒；末尾有标点时，只有“音频总时长减最后正文 end 严格大于 1000 ms”才保留。整行复制，原音频/正文/标点不改动。只有带标点的尾部必须提供相关可信 end；必需信息有错或音频不可读会记录原因并继续。具体命令见四卡 / 八卡指南。

准备器只生成第一块虚拟音频引用及私有评分元数据，不复制每秒 WAV。既有 `--tail-policy short-open` 仍支持保留截取样本的其他实验；它会调整有效音频范围及末尾标点，和本次“整行过滤后 keep”的策略不同。

过滤后以 `--tail-policy keep --timing-mode tail-only` 准备任务。最后 end 不当作整句话所有字的时间：任务 alignment 为空、不提供 switches 时间条目，全文 CER、局部语种、换人及标点目标仍生效，延迟罚分为 0。若完整参考仍有末尾标点而对应最后 DIRECT 没有，保留原正文，仅关闭该末尾标点的可信监督，句内评分不受影响。需要详细时间监督的另一个实验可显式用 `--timing-mode aligned`，并承担全量对齐校验。末尾间隔只是筛选条件，不保证剩余音频无漏标语音或句末标点一定正确。

~~~bash
bash 02_train.sh /data/models/fourth-sft-merged /data/OmniCommit/gspo04/tasks.jsonl /data/OmniCommit/output/gspo04-g8 \
  --gpu-ids 0,1,2,3 --group-size 8 --micro-batch-size 2 --global-batch-size 32 \
  --train-iters 2000 --save-steps 500 --reward-mode quality_latency --switch-grace-turns 1
~~~

四卡使用 TP2/EP2 训练、TP4 colocated vLLM 采样，LoRA 更新 Thinker 语言模型，冻结声学编码器和对齐层，保留学习率 1e-6、KL 0.01。全局 batch 按候选轨迹计：32/8=4 条录音/步；2000 步是 8000 次录音抽取、64000 条轨迹，2500 步是 10000 次抽取、80000 条轨迹，不等于整个大数据集走完一遍。

八卡按当前 FLEX 06/07 配方使用 `--gpu-ids 0,1,2,3,4,5,6,7 --group-size 8 --micro-batch-size 2 --global-batch-size 64`。训练 TP2/EP2/DP4、两组 TP4 vLLM，每步 8 条录音各 8 个候选。八卡 2000 步共 16000 次录音抽取、128000 条候选。保持 G8/micro2 时不能沿用 B32；当前入口会检查布局。完整命令在运行指南中。

默认 save_total_limit=6，保留本计划全部保存点。实际显存与耗时取决于音频轮数和候选长度。--dry-run 只展示实际框架参数，不分配 GPU、不写文件。确切续训需要保持原来的 train-iters 学习率计划；要训 2500 步应启动时设定。

## 奖励

CER 是主质量项。旧混合词/字符编辑路径保留作局部对齐和可用时的时间统计，不再把它称为 CER。切换区间另外计算局部 CER 和原词覆盖，检查短段是否被吞、翻译或替换。默认 quality_latency 入口对有详细时间戳的 aligned 数据才计算等待成本；本次 tail-only 不伪造这部分监督。

换人位置偏移按归错文字量区分程度；删除新人的开头也计入成本。漏一个换人标记不会机械改变后续全部人的归属。对 aligned 数据，`--switch-grace-turns 1` 给新发言区间一个共享的确认截止轮，不能每个词不断顺延；tail-only 没有该时间效应，也不强制 WAIT。

句子终止符在拼接后的正文中按位置评分，不因一个 DIRECT 只有几个字就罚分；真实短句和正确句号后的新句开头允许出现。标点是受上限约束的辅助项，防止“写错字破坏对齐，比正确文字多句号反而分高”。截取末尾与句内分开处理。错误参考仍会污染奖励，不能靠加大权重修复。正确数字的小数点、正负号、百分号在全文和局部 CER 中保持一致。

本版沿用序列级 GSPO；借鉴固定角色 SA-WER 的归属原则和确认宽限，未宣称复现固定角色身份识别或 token 级局部 advantage。依据、公式、参数取舍见 [REWARD_DESIGN.md](REWARD_DESIGN.md)。

## 保存和直接部署

checkpoint-N 是可续训的原生 checkpoint；checkpoint-N-merged 是部署/评测的 HF 模型。每次同步保存后，回调从**准确的 SFT 父模型**恢复缺失 tokenizer、processor、chat template 和 omnicommit_tokens.json，并校验词表、token ID、音频处理配置及权重形状。不会用 SFT 权重覆盖 RL 权重，不覆盖冲突 tokenizer。

成功写 omnicommit_export_status.json，ready:true 表示静态检查通过。失败打印 [export ERROR] 和 ready:false，保留原生训练结果并继续训练；失败目录不要直接部署。已有缺文件的 Fourth RL 导出可执行：

~~~bash
python3 finalize_export.py /data/models/fourth-sft-merged /data/OmniCommit/output/gspo04-g8/实际版本/checkpoint-500-merged
~~~

第一个目录必须是这次 RL 的准确起点，不能混拷另一个 tokenizer 版本。

## 启动前特殊 token 注册失败

若出现 `[launch] Protocol token is not registered special: <|WAIT|>`，退出发生在训练前的 tokenizer 校验。对于报错的这个 token，词表长度和单 token ID 已通过检查，但当前环境的 `all_special_tokens` 没有包含它。其他协议 token 还需检查，不能直接删除这项校验或重新分配 ID。

在同一个 FLEX-GSPO 环境检查启动命令的第一个模型目录：

~~~bash
python3 repair_tokenizer.py /data/models/fourth-sft-merged
python3 repair_tokenizer.py /data/models/fourth-sft-merged --apply
~~~

第一条只读；第二条先核对全部协议 ID、词表和权重行覆盖，再修复已有 token 的特殊注册。发现缺 token、ID 冲突或权重不完整则拒绝修复。修复保留原有特殊 token，在临时目录重载校验后只写 tokenizer 文件，不改模型权重、模型 config、音频 processor 或训练数据。成功后重新运行原训练命令，不需要重建 tasks。不要在服务或训练进程正在使用该目录时修改 tokenizer。

这类现象可能来自导出元数据或 Transformers 环境差异，不能仅凭日志认定具体版本。上游曾记录新旧特殊 token 字段并存时加载丢失注册的问题：[Transformers #47838](https://github.com/huggingface/transformers/issues/47838)。`rope_parameters` 的警告与这里的显式校验异常是不同信息；不要为消除警告删除 Omni 的 mRoPE 配置。

## 评测与监视

~~~bash
python3 evaluate.py model /data/OmniCommit/test/tasks.jsonl \
  --model /data/OmniCommit/output/gspo04-g8/实际版本/checkpoint-500-merged \
  --backend vllm --switch-grace-turns 1 --output /data/OmniCommit/test/gspo04-step500.jsonl
~~~

用已有独立测试集比较 SFT 和各保存点。评测逐轮解码，不混入 Web 的 VAD/补识别。汇总同时看 CER、切换漏段、归错字符、换人误报/漏报与碎句，不能只看总 reward。另按只有语种切换、只有换人、两者都有、都没有四类分别汇总。只有 aligned 数据具有可信详细标注时才解释正确提交延迟，并同时看删除与计时覆盖；tail-only 的未知时间不是零延迟。

TensorBoard 包含 reward、组内奖励相同比例及 CER、语言/换人/标点/时延罚分。rewards/rollouts-*.jsonl 记录逐轮输出及评分，metrics-*.json 是小型累计快照，不保存波形。CPU 测试验证数据/评分/导出契约，不能代替 GPU 训练和真实会议评测。
