# 中文参考资料 (Chinese-Language References)

英文版见 [`references.md`](references.md)。这里收录**中文**学习资源——中文博客、视频课程、开源教程与书籍，
按主题组织并映射到对应 notebook。每节先看 ⭐ 标记的入门首选。

> 说明：B 站等平台的视频链接容易变动，因此这类资源用「作者 + 标题」描述，并给出作者主页/搜索关键词，
> 而不直接贴可能失效的具体 URL。网站类资源给出稳定主页链接。

---

## 0. 最佳综述（先看这些）

- ⭐ **李沐 · 论文精读系列**（B 站「跟李沐学AI」）——逐段精读 Transformer、GPT/GPT-2/GPT-3、BERT 等经典论文，
  中文讲解 + 配套笔记，是本仓库最好的中文「论文陪读」。B 站搜索：`Transformer论文逐段精读`。
- ⭐ **李宏毅（Hung-yi Lee）· 机器学习 / 深度学习课程** —— Self-Attention、Transformer 两讲是中文世界最经典的入门。
  课程主页：<https://speech.ee.ntu.edu.tw/~hylee/>（B 站搜索：`李宏毅 self-attention`）。
- ⭐ **动手学深度学习（D2L）中文版** —— <https://zh.d2l.ai/>。注意力机制、Transformer、BERT、优化算法都有可运行代码，
  与本仓库「从零实现」的思路一致。
- **邱锡鹏 ·《神经网络与深度学习》** —— <https://nndl.github.io/>。免费中文教材，注意力机制与序列模型章节理论扎实。
- **3Blue1Brown 中文** —— 神经网络 / GPT / 注意力机制系列有官方中文字幕，B 站搜索：`3Blue1Brown GPT`。

**课程 / 教材：**
李宏毅 ML 课程 · D2L 中文版 <https://zh.d2l.ai/> · 邱锡鹏 NNDL <https://nndl.github.io/> ·
车万翔 等《自然语言处理：基于预训练模型的方法》（哈工大，社科文献/电子工业出版社）。

**开源中文教程（Datawhale 等社区）：**
- Datawhale `happy-llm` —— 从零讲解大模型原理与实践，<https://github.com/datawhalechina/happy-llm>。
- Datawhale `self-llm`（开源大模型食用指南）—— 微调/部署实操，<https://github.com/datawhalechina/self-llm>。
- Datawhale `llm-cookbook`（面向开发者的 LLM 入门，吴恩达课程中文版）—— <https://github.com/datawhalechina/llm-cookbook>。

---

## 1. 基础论文（NB 01、03–06）

- ⭐ 李沐《Transformer 论文逐段精读》（B 站「跟李沐学AI」）——配合英文原文 *Attention Is All You Need* 一起看。
- 《The Illustrated Transformer》中文翻译 —— 多个译本，搜索关键词：`图解 Transformer 中文`。
- 苏剑林 · 科学空间 —— 《Attention 是什么？》等系列，<https://kexue.fm/>。

## 2. 词嵌入与位置编码（NB 02、11）

- ⭐ **苏剑林 · 科学空间** —— RoPE（旋转位置编码）的**原作者本人**中文博客，深度第一手讲解。
  主页 <https://kexue.fm/>，搜索：`旋转式位置编码 RoPE`（《Transformer 升级之路》系列）。
- D2L 中文版 ·「位置编码」一节 —— <https://zh.d2l.ai/>。

## 3. 归一化与前馈层（NB 05、11）

- D2L 中文版 ·「批量/层归一化」章节 —— <https://zh.d2l.ai/>。
- 苏剑林 · 科学空间 —— RMSNorm、Pre-LN/Post-LN、激活函数相关分析，<https://kexue.fm/>。

## 4. 高效注意力（NB 04、09、11）

- 李沐 · 论文精读 ——《FlashAttention》等讲解（B 站「跟李沐学AI」）。
- 苏剑林 · 科学空间 —— MQA/GQA、线性注意力等中文分析，<https://kexue.fm/>。

## 5. 架构家族（NB 06、10）

- ⭐ 李沐 · 论文精读 ——《GPT、GPT-2、GPT-3 精读》《BERT 精读》（B 站「跟李沐学AI」）。
- D2L 中文版 ·「BERT」章节 —— <https://zh.d2l.ai/>。
- 张俊林 · 知乎 —— 预训练模型与 BERT 系列长文（知乎搜索作者：`张俊林 Transformer`）。

## 6. 分词 / Tokenization（NB 02 背景）

- BPE / WordPiece / SentencePiece 中文讲解 —— D2L 中文版「子词嵌入」一节 <https://zh.d2l.ai/>。
- Karpathy《Let's build the GPT Tokenizer》—— B 站有中文字幕搬运，搜索：`Karpathy 分词器`。

## 7. 训练、优化与 Scaling（NB 08、11）

- D2L 中文版 ·「优化算法」章节（SGD、Adam、学习率调度）—— <https://zh.d2l.ai/>。
- 李沐 · 论文精读 ——《Scaling Laws》《Chinchilla》相关讲解（B 站「跟李沐学AI」）。
- 苏剑林 · 科学空间 —— 学习率、Warmup、混合精度等工程实践，<https://kexue.fm/>。

## 8. 推理与解码（NB 09）

- D2L 中文版 ·「束搜索（Beam Search）」一节 —— <https://zh.d2l.ai/>。
- Top-k / Top-p（nucleus）采样中文讲解 —— 知乎搜索：`nucleus sampling 中文`。

## 9. 后训练：SFT 与高效微调（NB 12）

- ⭐ Datawhale `self-llm` —— LoRA/QLoRA 微调主流开源模型的中文实操，<https://github.com/datawhalechina/self-llm>。
- 李沐 · 论文精读 ——《InstructGPT》讲解（B 站「跟李沐学AI」）。
- 苏剑林 · 科学空间 —— LoRA 原理中文分析，<https://kexue.fm/>。

## 10. 后训练：偏好对齐（NB 13）

- ⭐ 李沐 · 论文精读 ——《InstructGPT》《DPO》相关讲解（B 站「跟李沐学AI」）。
- Datawhale `happy-llm` —— RLHF / PPO / DPO 的中文原理讲解，<https://github.com/datawhalechina/happy-llm>。
- 知乎专栏 —— 搜索：`RLHF 原理 中文`、`DPO 直接偏好优化`。

## 11. 进一步扩展：MoE 与开源模型（NB 11「下一步」）

- 李沐 · 论文精读 ——《Switch Transformer》《Mixtral》《LLaMA》相关讲解（B 站「跟李沐学AI」）。
- 苏剑林 · 科学空间 —— MoE、长上下文等前沿话题中文分析，<https://kexue.fm/>。

---

# Track A · 模型核心（NB 14–20）

## 12. 分词进阶（NB 14）

- ⭐ Karpathy《Let's build the GPT Tokenizer》—— B 站有中文字幕搬运，搜索：`Karpathy 分词器`。
- 苏剑林 · 科学空间 —— 分词、词表大小、多语言 fertility 等话题，<https://kexue.fm/>。
- DeepSeek / Qwen 等国产模型的分词器设计中文解读 —— 知乎搜索：`LLM 分词器 中文词表`。

## 13. 长上下文（NB 15）

- ⭐ **苏剑林 · 科学空间**（RoPE 原作者）——《Transformer 升级之路》系列**深度讲透** RoPE 外推、
  位置插值(PI)、NTK-aware、YaRN，是中文世界关于长上下文最权威的一手资料，<https://kexue.fm/>。
- 苏剑林 —— Attention Sink、KV Cache 压缩等话题的中文分析，<https://kexue.fm/>。

## 14. 注意力变体（NB 16）

- ⭐ 苏剑林 · 科学空间 —— MLA（多头潜在注意力）、线性注意力、DeepSeek-V2/V3 架构的中文分析，<https://kexue.fm/>。
- 李沐 · 论文精读 ——《Mamba》相关讲解（B 站「跟李沐学AI」）。
- 知乎专栏 —— 搜索：`Mamba 状态空间模型 中文`、`线性注意力 RNN`。

## 15. 混合专家 MoE（NB 17）

- 李沐 · 论文精读 ——《Switch Transformer》《Mixtral》讲解（B 站「跟李沐学AI」）。
- DeepSeek-V3 / DeepSeekMoE 技术报告中文解读 —— 知乎搜索：`DeepSeek MoE 负载均衡 无辅助损失`。
- Datawhale `happy-llm` —— MoE 原理章节，<https://github.com/datawhalechina/happy-llm>。

## 16. 推理与测试时计算（NB 18）

- ⭐ **DeepSeek-R1 技术报告中文精读** —— 知乎/公众号大量高质量解读，搜索：`DeepSeek-R1 GRPO 精读`。
- 李沐 · 论文精读 ——《Chain-of-Thought》《o1/推理模型》相关讲解（B 站「跟李沐学AI」）。
- 苏剑林 · 科学空间 —— GRPO、RLVR、思维链等强化学习对齐话题，<https://kexue.fm/>。

## 17. 效率：量化 / 蒸馏 / 剪枝（NB 19）

- Datawhale `self-llm` / `llm-cookbook` —— 量化部署（GPTQ/AWQ/GGUF）中文实操，
  <https://github.com/datawhalechina/self-llm>。
- 知乎专栏 —— 搜索：`LLM 量化 SmoothQuant AWQ 中文`、`知识蒸馏 大模型`。

## 18. 多模态（NB 20）

- ⭐ 李沐 · 论文精读 ——《ViT》《CLIP》讲解（B 站「跟李沐学AI」，中文世界最经典的多模态论文精读）。
- D2L 中文版 —— 计算机视觉相关章节，<https://zh.d2l.ai/>。
- LLaVA / Qwen-VL 中文技术解读 —— 知乎搜索：`多模态大模型 视觉编码器 中文`。

---

# Track B · 系统（NB 21–23）

## 19. 性能第一性原理（NB 21）

- 知乎专栏 —— 搜索：`LLM 推理 计算访存比 roofline 中文`、`prefill decode 显存带宽`。
- 各大厂推理优化博客（vLLM、SGLang 中文社区）—— Arithmetic Intensity、MFU/MBU 的中文讲解。

## 20. 分布式训练（NB 22）

- ⭐ **Megatron-LM / DeepSpeed(ZeRO) 中文解读** —— 知乎搜索：`ZeRO 显存优化 中文`、
  `张量并行 流水线并行 Megatron`。
- Colossal-AI / DeepSpeed 中文文档与教程 —— 数据/张量/流水线/专家并行的中文实践。
- 李沐 · 论文精读 ——《GPipe》《Megatron》相关分布式训练讲解（B 站「跟李沐学AI」）。

## 21. 推理服务（NB 23）

- ⭐ **vLLM / PagedAttention 中文解读** —— 知乎搜索：`PagedAttention 原理 中文`、
  `continuous batching 连续批处理`。
- SGLang / TensorRT-LLM 中文技术分享 —— 投机采样(speculative decoding)、前缀缓存的中文讲解。

---

# Track C · 应用（NB 24–26）

## 22. 检索增强生成 RAG（NB 24）

- ⭐ Datawhale `all-in-rag` / `tiny-universe` 等 —— RAG 全链路中文教程，
  <https://github.com/datawhalechina>。
- 知乎专栏 —— 搜索：`RAG 检索增强 中文`、`BM25 向量检索 混合检索 RRF`、`GraphRAG 中文`。

## 23. 智能体 Agent（NB 25）

- ⭐ Anthropic《Building Effective Agents》中文翻译 —— 搜索：`构建高效 Agent 中文`。
- Datawhale `hugging-multi-agent` / `tiny-agent` 等 —— Agent 与工具调用中文教程，
  <https://github.com/datawhalechina>。
- 知乎/公众号 —— 搜索：`ReAct 智能体 中文`、`提示词注入 prompt injection 中文`、`MCP 协议`。

## 24. 生产化（NB 26）

- 各大厂 LLM 工程化博客 —— 搜索：`大模型 生产部署 SLO 中文`、`S-LoRA 多租户 中文`、
  `LLM 评测 LLM-as-a-judge 中文`。

---

## 速查：notebook → 中文首选资源

| NB | 主题 | 从这里开始 |
|---|---|---|
| 01 | 演进史 | 李沐《Transformer 逐段精读》；李宏毅 Transformer 一讲 |
| 02 | 词嵌入 / 位置编码 | D2L 中文版「注意力 + 位置编码」；苏剑林 RoPE 博客 |
| 03–04 | 注意力 | 李宏毅 Self-Attention；D2L 中文版「注意力机制」 |
| 05–06 | 模块与整体 | D2L 中文版「Transformer」；李沐论文精读 |
| 07–08 | 训练 GPT | D2L 中文版「优化算法」；李沐 Scaling Laws 讲解 |
| 09 | 推理 / 解码 | D2L 中文版「束搜索」；nucleus sampling 中文讲解 |
| 10 | 编码器 & seq2seq | 李沐 BERT 精读；D2L 中文版「BERT」 |
| 11 | 现代架构 | 李沐 LLaMA 讲解；苏剑林 RoPE / RMSNorm 博客 |
| 12 | SFT & LoRA | Datawhale self-llm；李沐 InstructGPT 讲解 |
| 13 | 偏好对齐 | 李沐 DPO/InstructGPT 讲解；Datawhale happy-llm |
| 14 | 分词 | Karpathy 分词器中文字幕；苏剑林分词博客 |
| 15 | 长上下文 | ⭐ 苏剑林《Transformer 升级之路》RoPE 外推系列 |
| 16 | 注意力变体 | ⭐ 苏剑林 MLA / 线性注意力；李沐 Mamba 讲解 |
| 17 | 混合专家 | 李沐 Mixtral 讲解；DeepSeekMoE 中文解读 |
| 18 | 推理模型 | ⭐ DeepSeek-R1 GRPO 中文精读 |
| 19 | 效率 | Datawhale 量化部署；SmoothQuant/AWQ 中文 |
| 20 | 多模态 | ⭐ 李沐 ViT / CLIP 精读 |
| 21 | 性能原理 | roofline / 计算访存比 中文解读 |
| 22 | 分布式训练 | ⭐ ZeRO / Megatron 张量并行中文解读 |
| 23 | 推理服务 | ⭐ PagedAttention / vLLM 中文解读 |
| 24 | RAG | Datawhale RAG 教程；混合检索 RRF 中文 |
| 25 | Agent | 《构建高效 Agent》中文；ReAct / prompt injection 中文 |
| 26 | 生产化 | S-LoRA / LLM 评测 / SLO 中文工程博客 |

---

> 中文资源以社区与个人博客为主，链接可能随时间变动；若失效，用对应「作者 + 标题」在搜索引擎或 B 站重新检索即可。
> 想读一手原始论文（arXiv ID 已核验），请回到英文版 [`references.md`](references.md)。
