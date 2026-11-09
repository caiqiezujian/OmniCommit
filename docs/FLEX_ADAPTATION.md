# 与 FLEX 的对应关系

OmniCommit 借鉴 FLEX 的虚拟音频读取、多轮 SWIFT 模板、Megatron SFT 启动和 Web 交互，但任务改为多语种、多说话人的增量 ASR。下面的训练对照针对 FLEX 的 `SFT-Scripts/train_duplex_sft_virtual.sh`；FLEX 还有不同的原始 `Scripts/train_duplex_sft.sh`，不能把两个版本当作完全相同。

| 项目 | FLEX 虚拟音频 SFT | OmniCommit |
|---|---|---|
| 输出任务 | 同声翻译 | 保留原语言的语音转录，附语种与说话人 |
| 动作 | `<\|WAIT\|>`、`<\|DIRECT\|>`、`<\|FLEX\|>` | `<\|WAIT\|>`、`<\|DIRECT\|>`；不使用翻译重组动作 |
| 单轮输出 | 等待或新增译文 | WAIT 无文本；DIRECT 仅输出新增转录，每段带完整 `<语言><说话人>` |
| 音频输入 | 多轮虚拟音频区间 | 同样是原音频区间；当前文本标注主线按 0.5/1 秒重铸 |
| 最后处理 | 以 FLEX 数据协议为准 | 当前 schema 2 增加无音频 `<end_of_audio>`，有积压时释放；早期音频教师 schema 1 不应混同 |
| token 清单 | 项目固定的三个动作 token | 两个动作加数据使用的语种、说话人标签；训练、恢复、导出使用同一有序清单 |
| 模型训练 | Qwen3-Omni MCore、ms-swift 3.12.1 / Megatron、LoRA | 同一训练栈；OmniCommit 使用自己的虚拟音频插件和模板 |
| 默认主要配方 | 四卡、r8/alpha32、EP2、micro4/global16、lr 1e-4、3 epochs、packing、4096 | 以上主要配方相同；可选旧驱动兼容模式另有变化 |
| 保存 | 虚拟版本 3000 步，未合并 adapter | 默认 3000 步、未合并 adapter，并保存训练词表层；默认不保存 optimizer/RNG，不能保证精确续训 |
| Web 入口 | 文件/电脑声音等同传交互 | 麦克风、系统声音、音视频文件，逐轮 ASR 与说话人修正；内网无需访问令牌 |

FLEX **原始** `Scripts/train_duplex_sft.sh` 与上述虚拟脚本还有差异，例如原始脚本 `--save_interval 1000`、`--cross_entropy_loss_fusion true`；当前 OmniCommit 使用 3000 与 `false`，对应虚拟音频版本。因此不要笼统认为与所有旧 SFT 命令的每个参数完全一致。

## 数据构建与时间

当前主线使用一次文本教师调用进行细切分与初步标签构建，再由真实音频时间戳重铸。语种或说话人变化按对齐起点所在音频块 WAIT，普通源 WAIT 在完成块触发；随后等待源 DIRECT 单元完整再一起释放。普通片段可连续 DIRECT。切分、对齐、实体完整性和说话人归属都需要独立验收。[时间戳直接生成 SFT](timestamped_to_sft_zh.md)可一步生成精简 `audios/messages` 训练行。

原音频独立教师完整音频/累计前缀与稳定前缀轨迹是另一条对照路线，不是当前文本标注路线必须使用的步骤；详见 [方法与验证边界](method.md)。精简数据不带安全拆窗元数据，长录音需走保留来源信息的分阶段路线及实际模板长度检查。

## Web 与长会议

OmniCommit 的网页和推理可以部署在同一 GPU 服务器，默认无访问令牌。远程浏览器的麦克风和系统声音仍需受信任 HTTPS 或 localhost。完整命令见 [会议 Web 服务](web_meeting_service_zh.md)。

当前提供手工说话人参照、逐段修正和显示身份合并；可选 Qwen 对齐后台尝试创建实验性自动参照。Qwen 对齐不验证声音身份或发言重叠；模型历史不会因显示修正而重写。`rolling` 上下文裁剪可能遗漏未提交内容，连续 WAIT 达预算会停止；不能据此声称 30 分钟会议、跨语种说话人归属或实时性能已经验证。

实现参考：[SWIFT 3.12.1 Qwen 模板](https://github.com/modelscope/ms-swift/blob/v3.12.1/swift/llm/template/template/qwen.py)、[SWIFT 3.12.1 音频工具](https://github.com/modelscope/ms-swift/blob/v3.12.1/swift/llm/template/vision_utils.py) 和 [Transformers 4.57.1 Omni 模型](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py)。真实 GPU 训练、模型导出和推理需在目标环境验收；CPU 数据环境不会安装完整训练栈。
