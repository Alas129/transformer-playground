# References

Curated, authoritative sources — original papers, official blogs, and established
courses — organized by topic. arXiv IDs verified. Start with the ⭐ items in each section.

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
