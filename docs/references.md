# References

Curated, authoritative sources — original papers, official blogs, and established
courses — organized by topic. arXiv IDs verified. Start with the ⭐ items in each section.

> 中文读者：另见 [**references-zh.md**](references-zh.md) — 中文学习资料（视频课程、博客、开源教程）。

---

## 0. The best overviews (read these first)

- ⭐ Jay Alammar — [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) · [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — the canonical visual explanations.
- ⭐ Andrej Karpathy — [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) (video) and [nanoGPT](https://github.com/karpathy/nanoGPT) — the spiritual companion to this repo.
- ⭐ Harvard NLP — [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — the 2017 paper, line by line, in runnable PyTorch.
- Lilian Weng — [The Transformer Family (v2.0)](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) and [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/).
- 3Blue1Brown — [But what is a GPT? / Attention in transformers](https://www.youtube.com/watch?v=wjZofJX0v4M) (visual intuition).

**Courses:** [Stanford CS224N — NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) · [Stanford CS336 — Language Modeling from Scratch](https://stanford-cs336.github.io/) · [Hugging Face LLM Course](https://huggingface.co/learn/llm-course) · [d2l.ai — Dive into Deep Learning](https://d2l.ai/).

---

## 1. The foundational paper (NB 01, 03–06)

- ⭐ Vaswani et al., 2017 — **Attention Is All You Need** — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762). The Transformer.
- Bahdanau et al., 2014 — Neural Machine Translation by Jointly Learning to Align and Translate — [arXiv:1409.0473](https://arxiv.org/abs/1409.0473). Attention, before Transformers.

## 2. Embeddings & position (NB 02, 11)

- Vaswani et al., 2017 (above) — sinusoidal positional encoding (§3.5).
- Su et al., 2021 — **RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)** — [arXiv:2104.09864](https://arxiv.org/abs/2104.09864).
- Press et al., 2021 — Train Short, Test Long: ALiBi — [arXiv:2108.12409](https://arxiv.org/abs/2108.12409).

## 3. Normalization & feed-forward (NB 05, 11)

- Ba et al., 2016 — Layer Normalization — [arXiv:1607.06450](https://arxiv.org/abs/1607.06450).
- Zhang & Sennrich, 2019 — **Root Mean Square Layer Normalization (RMSNorm)** — [arXiv:1910.07467](https://arxiv.org/abs/1910.07467).
- Shazeer, 2020 — **GLU Variants Improve Transformer (SwiGLU)** — [arXiv:2002.05202](https://arxiv.org/abs/2002.05202).
- Xiong et al., 2020 — On Layer Normalization in the Transformer Architecture (Pre-LN vs Post-LN) — [arXiv:2002.04745](https://arxiv.org/abs/2002.04745).

## 4. Efficient attention (NB 04, 09, 11)

- Shazeer, 2019 — Fast Transformer Decoding: One Write-Head is All You Need (**MQA**) — [arXiv:1911.02150](https://arxiv.org/abs/1911.02150).
- Ainslie et al., 2023 — **GQA: Training Generalized Multi-Query Transformer Models** — [arXiv:2305.13245](https://arxiv.org/abs/2305.13245).
- Dao et al., 2022 — **FlashAttention: Fast and Memory-Efficient Exact Attention** — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135). Follow-up: [FlashAttention-2 (2307.08691)](https://arxiv.org/abs/2307.08691).

## 5. The architecture families (NB 06, 10)

- Radford et al., 2018 — Improving Language Understanding by Generative Pre-Training (**GPT-1**) — [paper (PDF)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf).
- Radford et al., 2019 — Language Models are Unsupervised Multitask Learners (**GPT-2**) — [paper (PDF)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
- Devlin et al., 2018 — **BERT** (encoder-only, MLM) — [arXiv:1810.04805](https://arxiv.org/abs/1810.04805).
- Raffel et al., 2019 — Exploring the Limits of Transfer Learning (**T5**, encoder-decoder) — [arXiv:1910.10683](https://arxiv.org/abs/1910.10683).
- Lewis et al., 2019 — **BART** (denoising seq2seq) — [arXiv:1910.13461](https://arxiv.org/abs/1910.13461).

## 6. Tokenization (NB 02 context)

- Sennrich et al., 2015 — Neural Machine Translation of Rare Words with Subword Units (**BPE**) — [arXiv:1508.07909](https://arxiv.org/abs/1508.07909).
- Kudo & Richardson, 2018 — **SentencePiece** — [arXiv:1808.06226](https://arxiv.org/abs/1808.06226).
- Karpathy — [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) (video) and [minbpe](https://github.com/karpathy/minbpe).

## 7. Training, optimization & scaling (NB 08, 11)

- Kingma & Ba, 2014 — Adam — [arXiv:1412.6980](https://arxiv.org/abs/1412.6980).
- Loshchilov & Hutter, 2017 — Decoupled Weight Decay Regularization (**AdamW**) — [arXiv:1711.05101](https://arxiv.org/abs/1711.05101).
- Loshchilov & Hutter, 2016 — SGDR: Warm Restarts (cosine schedule) — [arXiv:1608.03983](https://arxiv.org/abs/1608.03983).
- Micikevicius et al., 2017 — Mixed Precision Training — [arXiv:1710.03740](https://arxiv.org/abs/1710.03740).
- Chen et al., 2016 — Training Deep Nets with Sublinear Memory Cost (gradient checkpointing) — [arXiv:1604.06174](https://arxiv.org/abs/1604.06174).
- ⭐ Kaplan et al., 2020 — Scaling Laws for Neural Language Models — [arXiv:2001.08361](https://arxiv.org/abs/2001.08361).
- ⭐ Hoffmann et al., 2022 — Training Compute-Optimal Large Language Models (**Chinchilla**) — [arXiv:2203.15556](https://arxiv.org/abs/2203.15556).

## 8. Inference & decoding (NB 09)

- Holtzman et al., 2019 — The Curious Case of Neural Text Degeneration (**top-p / nucleus**) — [arXiv:1904.09751](https://arxiv.org/abs/1904.09751).
- Fan et al., 2018 — Hierarchical Neural Story Generation (top-k sampling) — [arXiv:1805.04833](https://arxiv.org/abs/1805.04833).
- Frantar et al., 2022 — GPTQ: Accurate Post-Training Quantization — [arXiv:2210.17323](https://arxiv.org/abs/2210.17323).

## 9. Post-training: SFT & parameter-efficient fine-tuning (NB 12)

- ⭐ Ouyang et al., 2022 — Training language models to follow instructions (**InstructGPT**) — [arXiv:2203.02155](https://arxiv.org/abs/2203.02155). The SFT→RLHF recipe.
- Wei et al., 2021 — Finetuned Language Models are Zero-Shot Learners (**FLAN**, instruction tuning) — [arXiv:2109.01652](https://arxiv.org/abs/2109.01652).
- Wang et al., 2022 — **Self-Instruct** — [arXiv:2212.10560](https://arxiv.org/abs/2212.10560).
- ⭐ Hu et al., 2021 — **LoRA: Low-Rank Adaptation of Large Language Models** — [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
- Dettmers et al., 2023 — **QLoRA: Efficient Finetuning of Quantized LLMs** — [arXiv:2305.14314](https://arxiv.org/abs/2305.14314).

## 10. Post-training: preference alignment (NB 13)

- Christiano et al., 2017 — Deep Reinforcement Learning from Human Preferences — [arXiv:1706.03741](https://arxiv.org/abs/1706.03741). The origin of RLHF.
- Stiennon et al., 2020 — Learning to Summarize from Human Feedback — [arXiv:2009.01325](https://arxiv.org/abs/2009.01325).
- Schulman et al., 2017 — Proximal Policy Optimization (**PPO**) — [arXiv:1707.06347](https://arxiv.org/abs/1707.06347).
- ⭐ Rafailov et al., 2023 — **Direct Preference Optimization (DPO)** — [arXiv:2305.18290](https://arxiv.org/abs/2305.18290).
- Bai et al., 2022 — Constitutional AI: Harmlessness from AI Feedback (**RLAIF**) — [arXiv:2212.08073](https://arxiv.org/abs/2212.08073).

## 11. Scaling out: MoE & open models (NB 11 "what's next")

- Shazeer et al., 2017 — Outrageously Large Neural Networks (the **Mixture-of-Experts** layer) — [arXiv:1701.06538](https://arxiv.org/abs/1701.06538).
- Fedus et al., 2021 — Switch Transformers — [arXiv:2101.03961](https://arxiv.org/abs/2101.03961).
- Jiang et al., 2024 — Mixtral of Experts — [arXiv:2401.04088](https://arxiv.org/abs/2401.04088).
- Touvron et al., 2023 — **LLaMA** — [arXiv:2302.13971](https://arxiv.org/abs/2302.13971) · Llama 2 — [arXiv:2307.09288](https://arxiv.org/abs/2307.09288). Modern decoder-only design (RMSNorm + RoPE + SwiGLU + GQA), exactly the stack in NB 11.

---

# Track A — Model core

## 12. Tokenization (NB 14)

- ⭐ Sennrich et al., 2015 — **BPE for NLP** — [arXiv:1508.07909](https://arxiv.org/abs/1508.07909).
- Kudo & Richardson, 2018 — **SentencePiece** — [arXiv:1808.06226](https://arxiv.org/abs/1808.06226) · Kudo, 2018 — [Subword Regularization (Unigram)](https://arxiv.org/abs/1804.10959).
- Radford et al., 2019 — **GPT-2** (byte-level BPE, §2.2) — [PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
- Land & Bartolo, 2024 — Fishing for Magikarp (under-trained tokens) — [arXiv:2405.05417](https://arxiv.org/abs/2405.05417).
- Pagnoni et al., 2024 — Byte Latent Transformer — [arXiv:2412.09871](https://arxiv.org/abs/2412.09871).

## 13. Long context (NB 15)

- ⭐ Peng et al., 2023 — **YaRN** — [arXiv:2309.00071](https://arxiv.org/abs/2309.00071).
- Chen et al., 2023 — Position Interpolation — [arXiv:2306.15595](https://arxiv.org/abs/2306.15595).
- Press et al., 2021 — **ALiBi** — [arXiv:2108.12409](https://arxiv.org/abs/2108.12409).
- Xiao et al., 2023 — **Attention sinks / StreamingLLM** — [arXiv:2309.17453](https://arxiv.org/abs/2309.17453).
- ⭐ Hsieh et al., 2024 — **RULER** (real vs advertised context) — [arXiv:2404.06654](https://arxiv.org/abs/2404.06654).
- Liu et al., 2023 — Lost in the Middle — [arXiv:2307.03172](https://arxiv.org/abs/2307.03172).

## 14. Attention variants (NB 16)

- ⭐ DeepSeek-AI, 2024 — **DeepSeek-V2 (MLA)** — [arXiv:2405.04434](https://arxiv.org/abs/2405.04434).
- Katharopoulos et al., 2020 — **Transformers are RNNs** (linear attention) — [arXiv:2006.16236](https://arxiv.org/abs/2006.16236).
- ⭐ Gu & Dao, 2023 — **Mamba** — [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) · Dao & Gu, 2024 — [Mamba-2 / SSMs are Transformers](https://arxiv.org/abs/2405.21060).
- Lieber et al., 2024 — Jamba (hybrid) — [arXiv:2403.19887](https://arxiv.org/abs/2403.19887).
- Arora et al., 2023 — Zoology (associative recall) — [arXiv:2312.04927](https://arxiv.org/abs/2312.04927).

## 15. Mixture-of-Experts (NB 17)

- Fedus et al., 2021 — **Switch Transformers** (aux loss) — [arXiv:2101.03961](https://arxiv.org/abs/2101.03961).
- Zoph et al., 2022 — ST-MoE (z-loss, stability) — [arXiv:2202.08906](https://arxiv.org/abs/2202.08906).
- Dai et al., 2024 — DeepSeekMoE (fine-grained + shared) — [arXiv:2401.06066](https://arxiv.org/abs/2401.06066).
- ⭐ Wang et al., 2024 — **Loss-Free Load Balancing** — [arXiv:2408.15664](https://arxiv.org/abs/2408.15664).
- Komatsuzaki et al., 2022 — Sparse Upcycling — [arXiv:2212.05055](https://arxiv.org/abs/2212.05055).

## 16. Reasoning & test-time compute (NB 18)

- Wei et al., 2022 — **Chain-of-Thought** — [arXiv:2201.11903](https://arxiv.org/abs/2201.11903).
- Wang et al., 2022 — Self-Consistency — [arXiv:2203.11171](https://arxiv.org/abs/2203.11171).
- Lightman et al., 2023 — Let's Verify Step by Step (PRM) — [arXiv:2305.20050](https://arxiv.org/abs/2305.20050).
- ⭐ Shao et al., 2024 — **DeepSeekMath (GRPO)** — [arXiv:2402.03300](https://arxiv.org/abs/2402.03300).
- ⭐ DeepSeek-AI, 2025 — **DeepSeek-R1** (RLVR) — [arXiv:2501.12948](https://arxiv.org/abs/2501.12948).
- Snell et al., 2024 — Scaling Test-Time Compute Optimally — [arXiv:2408.03314](https://arxiv.org/abs/2408.03314).
- Turpin et al., 2023 — CoT unfaithfulness — [arXiv:2305.04388](https://arxiv.org/abs/2305.04388).

## 17. Efficiency: quantization, distillation, pruning (NB 19)

- Dettmers et al., 2022 — LLM.int8() (the outlier discovery) — [arXiv:2208.07339](https://arxiv.org/abs/2208.07339).
- ⭐ Xiao et al., 2022 — **SmoothQuant** — [arXiv:2211.10438](https://arxiv.org/abs/2211.10438).
- Frantar et al., 2022 — GPTQ — [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) · Lin et al., 2023 — [AWQ](https://arxiv.org/abs/2306.00978).
- Hinton et al., 2015 — Distilling the Knowledge — [arXiv:1503.02531](https://arxiv.org/abs/1503.02531) · Agarwal et al., 2023 — [On-Policy Distillation](https://arxiv.org/abs/2306.13649).
- Frantar & Alistarh, 2023 — SparseGPT — [arXiv:2301.00774](https://arxiv.org/abs/2301.00774).

## 18. Multimodal (NB 20)

- ⭐ Dosovitskiy et al., 2020 — **ViT** — [arXiv:2010.11929](https://arxiv.org/abs/2010.11929).
- ⭐ Radford et al., 2021 — **CLIP** — [arXiv:2103.00020](https://arxiv.org/abs/2103.00020).
- Liu et al., 2023 — LLaVA — [arXiv:2304.08485](https://arxiv.org/abs/2304.08485) · [LLaVA-1.5](https://arxiv.org/abs/2310.03744).
- Alayrac et al., 2022 — Flamingo (cross-attention) — [arXiv:2204.14198](https://arxiv.org/abs/2204.14198).
- Radford et al., 2022 — Whisper — [arXiv:2212.04356](https://arxiv.org/abs/2212.04356) · Peebles & Xie, 2022 — [DiT](https://arxiv.org/abs/2212.09748).

See also [**variants-atlas.md**](variants-atlas.md) for Transformers beyond text.

---

# Track B — Systems

## 19. Performance first principles (NB 21)

- ⭐ Pope et al., 2022 — **Efficiently Scaling Transformer Inference** — [arXiv:2211.05102](https://arxiv.org/abs/2211.05102). The definitive treatment.
- Williams et al., 2009 — Roofline model — [ACM](https://dl.acm.org/doi/10.1145/1498765.1498785).
- Chowdhery et al., 2022 — PaLM (introduces MFU) — [arXiv:2204.02311](https://arxiv.org/abs/2204.02311).
- Databricks — [LLM Inference Performance Engineering](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) (MBU).

## 20. Distributed training (NB 22)

- ⭐ Rajbhandari et al., 2019 — **ZeRO** — [arXiv:1910.02054](https://arxiv.org/abs/1910.02054) · Zhao et al., 2023 — [PyTorch FSDP](https://arxiv.org/abs/2304.11277).
- ⭐ Shoeybi et al., 2019 — **Megatron-LM (tensor parallelism)** — [arXiv:1909.08053](https://arxiv.org/abs/1909.08053).
- Narayanan et al., 2021 — 3D parallelism, 1F1B — [arXiv:2104.04473](https://arxiv.org/abs/2104.04473) · Huang et al., 2018 — [GPipe](https://arxiv.org/abs/1811.06965).
- Korthikanti et al., 2022 — Sequence parallelism + selective recompute — [arXiv:2205.05198](https://arxiv.org/abs/2205.05198).
- Liu et al., 2023 — Ring Attention — [arXiv:2310.01889](https://arxiv.org/abs/2310.01889).
- Grattafiori et al., 2024 — The Llama 3 Herd (§3.3: real config, failure stats) — [arXiv:2407.21783](https://arxiv.org/abs/2407.21783).

## 21. Inference serving (NB 23)

- ⭐ Kwon et al., 2023 — **PagedAttention / vLLM** — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180).
- Yu et al., 2022 — Orca (continuous batching) — [OSDI](https://www.usenix.org/conference/osdi22/presentation/yu).
- Zheng et al., 2023 — SGLang / RadixAttention — [arXiv:2312.07104](https://arxiv.org/abs/2312.07104).
- ⭐ Leviathan et al., 2022 — **Speculative decoding** — [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) · Chen et al., 2023 — [Speculative sampling](https://arxiv.org/abs/2302.01318).
- Agrawal et al., 2024 — Sarathi-Serve (chunked prefill) — [arXiv:2403.02310](https://arxiv.org/abs/2403.02310) · Zhong et al., 2024 — [DistServe (disaggregation)](https://arxiv.org/abs/2401.09670).

---

# Track C — Applications

## 22. RAG (NB 24)

- ⭐ Lewis et al., 2020 — **RAG** — [arXiv:2005.11401](https://arxiv.org/abs/2005.11401).
- Robertson & Zaragoza, 2009 — BM25 and Beyond — [journal](https://dl.acm.org/doi/10.1561/1500000019) · Karpukhin et al., 2020 — [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906).
- Reimers & Gurevych, 2019 — Sentence-BERT — [arXiv:1908.10084](https://arxiv.org/abs/1908.10084) · Cormack et al., 2009 — [RRF](https://dl.acm.org/doi/10.1145/1571941.1572114).
- Gao et al., 2022 — HyDE — [arXiv:2212.10496](https://arxiv.org/abs/2212.10496) · Edge et al., 2024 — [GraphRAG](https://arxiv.org/abs/2404.16130).
- Es et al., 2023 — RAGAS (evaluation) — [arXiv:2309.15217](https://arxiv.org/abs/2309.15217).

## 23. Agents (NB 25)

- ⭐ Anthropic, 2024 — **Building Effective Agents** — [blog](https://www.anthropic.com/research/building-effective-agents).
- ⭐ Yao et al., 2022 — **ReAct** — [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) · Shinn et al., 2023 — [Reflexion](https://arxiv.org/abs/2303.11366).
- Anthropic, 2024 — [Model Context Protocol](https://modelcontextprotocol.io) · Anthropic, 2025 — [Multi-agent research system](https://www.anthropic.com/engineering/built-multi-agent-research-system).
- Greshake et al., 2023 — Indirect prompt injection — [arXiv:2302.12173](https://arxiv.org/abs/2302.12173).
- Jimenez et al., 2023 — SWE-bench — [arXiv:2310.06770](https://arxiv.org/abs/2310.06770) · Yao et al., 2024 — [τ-bench](https://arxiv.org/abs/2406.12045).

## 24. Production (NB 26)

- ⭐ Sheng et al., 2023 — **S-LoRA** — [arXiv:2311.03285](https://arxiv.org/abs/2311.03285).
- Chen et al., 2023 — FrugalGPT (cascades) — [arXiv:2305.05176](https://arxiv.org/abs/2305.05176).
- Zheng et al., 2023 — Judging LLM-as-a-Judge (MT-Bench) — [arXiv:2306.05685](https://arxiv.org/abs/2306.05685).
- Google SRE — [Service Level Objectives](https://sre.google/sre-book/service-level-objectives/).

---

## Quick map: notebook → must-read

| NB | Topic | Start here |
|---|---|---|
| 01 | Evolution | Illustrated Transformer; Attention Is All You Need |
| 02 | Embeddings | Annotated Transformer §positional encoding |
| 03–04 | Attention | Attention Is All You Need §3; Karpathy "Let's build GPT" |
| 05–06 | Blocks & full model | Annotated Transformer; nanoGPT |
| 07–08 | Train a GPT | nanoGPT; Kaplan & Chinchilla scaling laws |
| 09 | Inference | Nucleus sampling (Holtzman 2019); FlashAttention |
| 10 | Encoders & seq2seq | BERT; T5 |
| 11 | Modern architectures | LLaMA; RoPE; RMSNorm; SwiGLU; GQA; FlashAttention |
| 12 | SFT & LoRA | InstructGPT; LoRA; QLoRA |
| 13 | Alignment | InstructGPT; DPO; PPO; Constitutional AI |
| 14 | Tokenization | Sennrich (BPE); Karpathy tokenizer video |
| 15 | Long context | YaRN; ALiBi; RULER |
| 16 | Attention variants | DeepSeek-V2 (MLA); Mamba; Zoology |
| 17 | Mixture-of-Experts | Switch; DeepSeekMoE; Loss-Free Balancing |
| 18 | Reasoning | Chain-of-Thought; GRPO (DeepSeekMath); DeepSeek-R1 |
| 19 | Efficiency | SmoothQuant; GPTQ/AWQ; On-Policy Distillation |
| 20 | Multimodal | ViT; CLIP; LLaVA |
| 21 | Performance | Pope et al. (Efficiently Scaling Inference); Roofline |
| 22 | Distributed training | ZeRO; Megatron-LM; Llama 3 Herd §3.3 |
| 23 | Serving | PagedAttention (vLLM); Speculative Decoding |
| 24 | RAG | Lewis (RAG); BM25 and Beyond; RRF |
| 25 | Agents | Building Effective Agents; ReAct |
| 26 | Production | S-LoRA; FrugalGPT; LLM-as-a-Judge |
