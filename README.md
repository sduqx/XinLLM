# XinLLM - 从零构建与微调大语言模型

> 基于 PyTorch 实现的 GPT-2 架构及指令微调（SFT）全流程项目。

本项目不依赖高级封装库（如 `transformers`），而是**从零手写**了 Transformer 核心组件、数据流水线及训练循环。项目实现了从文本预处理、模型预训练、指令微调到自动化评估的完整 LLM 开发周期。

## 🎯 项目亮点

- **架构透明**：手写了包含 Causal Mask 的 `MultiHeadAttention`、LayerNorm 及 GELU 激活函数，深入理解 Transformer 内部数据流向。
- **数据管线**：实现了自定义的 `collate_fn` 处理变长序列填充（Padding）与 Label 掩码（`ignore_index=-100`），优化显存利用率。
- **指令微调 (SFT)**：在 GPT-2 Medium (355M) 基础上，使用 Alpaca 格式数据集完成监督微调，使模型具备指令遵循能力。
- **自动化评估**：构建了基于 LLM-as-a-Judge 的评估管线，支持 Ollama 本地部署及智谱 API 云端评估，实现模型表现的量化打分。

## 🛠️ 技术栈

- **核心框架**：PyTorch 2.x, Python 3.10
- **关键组件**：Tiktoken (Tokenizer), Matplotlib (可视化), Pandas
- **模型评估**：Ollama (Llama 3), Zhipu AI API (GLM-4)

## 🚀 快速开始

### 1. 环境准备
```bash
# 建议使用虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 2. 核心流程运行
本项目采用 Jupyter Notebook 形式分步教学与实现：

- **Chapter 2**: 文本数据处理 (Tokenization, BPE算法, 词嵌入向量构建)
- **Chapter 3**: 注意力机制实现 (自注意力、因果掩码 Causal Mask, 多头注意力 MHA)
- **Chapter 4**: Transformer 架构组装 (GPTModel, LayerNorm, FFN, SwiGLU 扩展)
- **Chapter 5**: 预训练阶段 (Pre-training, Loss visualization, 模型权重保存)
- **Chapter 6**: 分类任务微调 (Classification Head, Spam Detection, 垃圾邮件识别)
- **Chapter 7**: 指令微调 (SFT, Alpaca Dataset, Evaluation Pipeline, LLM-as-a-Judge)

### 3. 推理示例
加载微调后的模型并测试指令遵循效果：
```python
# 在 ch7 对应的 Notebook 中运行
from previous_chapters import GPTModel, load_weights_into_gpt

# 初始化模型并加载权重
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))

# 使用 format_input 构造 Prompt 并调用 generate 函数生成回答
```

## 📂 目录结构

```
XinLLM/
├── ch2/             # 文本数据处理 (Tokenization, BPE算法, 词嵌入向量构建)
├── ch3/             # 注意力机制实现 (Attention, MHA, Causal Mask)
├── ch4/             # Transformer 架构实现 (Attention, FFN, Block)
├── ch5/             # 预训练阶段 (Pre-training, Loss visualization)
├── ch6/             # 分类任务微调 (Classification Head, Spam Detection)
├── ch7/             # 指令微调 (SFT, Alpaca Dataset, Evaluation Pipeline)
├── notes/           # 核心概念笔记 (Tokenization, Embeddings)
└── gpt2/            # 预训练权重存储目录
```

## 📝 学习产出

1. **位置编码优化**：计划实现相对位置编码 (RoPE) 与绝对位置编码的对比。
2. **架构升级**：计划将 GELU 替换为 SwiGLU 激活函数，对齐 Llama 3 架构。

## 📄 许可证

MIT License
