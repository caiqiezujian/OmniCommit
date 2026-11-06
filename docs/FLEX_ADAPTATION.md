# 与 FLEX 的对应关系

本项目从本地 FLEX 的 `SFT-Scripts` 复用并适配独立音频区间 reader、SWIFT template plugin、Megatron launcher 及可选旧驱动注意力兼容实现。审查时 FLEX 的 HEAD 为 `78ebde69afa27872cc7b2147b7948ea8329303e0`；原工作区未被修改。

| 项目 | OmniCommit 的处理 |
|---|---|
| 基座与训练栈 | 沿用 Qwen3-Omni、ms-swift 3.12.1、Megatron；HF兼容代码针对4.57.1 |
| 音频存储 | 原始音频加区间引用，读取时裁剪；不生成海量小文件 |
| 多轮输入 | 每轮一个新增音频区间，历史保留在此前messages中 |
| 动作 | `<\|WAIT\|>`、`<\|DIRECT\|>`；删除翻译FLEX动作 |
| 监督目标 | 原语言转录 + 语言/说话人标记；不再是跨语翻译 |
| 数字 | 由数据策略选择verbatim或normalized，训练指令与监督一致 |
| 特殊tokens | 两个动作 + 数据实际使用的语言和speaker标记 |
| 最后一轮 | 允许WAIT，尾部静音不是错误 |
| 训练配方 | 四卡，LoRA r8/alpha32、EP2、micro4/global16、lr1e-4、3epochs |
| 保存 | 未合并LoRA适配器，同时保存训练的embedding/output模块 |
| 教师调用 | 新增音频base64 HTTP调用及离线replay，不复用纯文本翻译API |

实现依据：

- [SWIFT 3.12.1 Qwen模板](https://github.com/modelscope/ms-swift/blob/v3.12.1/swift/llm/template/template/qwen.py)
- [SWIFT 3.12.1音频工具](https://github.com/modelscope/ms-swift/blob/v3.12.1/swift/llm/template/vision_utils.py)
- [Transformers 4.57.1 Omni模型](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py)

GPU运行仍需要原工作环境中的匹配依赖与MCore权重。CPU安装文件只用于数据处理和测试，不自动安装整套训练依赖。
