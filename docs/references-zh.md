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

---

> 中文资源以社区与个人博客为主，链接可能随时间变动；若失效，用对应「作者 + 标题」在搜索引擎或 B 站重新检索即可。
> 想读一手原始论文（arXiv ID 已核验），请回到英文版 [`references.md`](references.md)。
