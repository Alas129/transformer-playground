# Study Guide

How to get from "what is attention?" to "I can explain how ChatGPT is built." Pair this
with [references.md](references.md) (the papers) and [cheatsheet.md](cheatsheet.md) (the math).

---

## Prerequisites

You'll move fastest if you're comfortable with:

- **Python + NumPy** — array indexing, broadcasting, matrix multiply.
- **Linear algebra** — vectors, matrices, dot products, matrix multiplication. (No proofs needed.)
- **Calculus intuition** — what a gradient is and why we descend it. (You don't compute them by hand; PyTorch does.)
- **A little PyTorch** — `nn.Module`, tensors, `.backward()`. Picked up as you go from notebook 05 onward.

If a term is unfamiliar at any point, check [glossary.md](glossary.md) first.

---

## The arc

```
   FUNDAMENTALS                 BUILD A GPT              MAKE IT REAL
   01 Evolution         ┐       05 Blocks       ┐       08 Training
   02 Embeddings        ├──►    06 Full model    ├──►   09 Inference
   03 Attention         │       07 Train & gen   ┘      10 Architecture families
   04 Multi-Head        ┘                               11 Modern architectures
                                                              │
                                                        POST-TRAINING
                                                        12 SFT + LoRA
                                                        13 Preference alignment
```

- **01–07 — Build a GPT from scratch.** Attention → block → full model → train it. After 07 you have a working text generator.
- **08–11 — Make it real.** How models are *trained* at scale, how they *generate*, the other architecture families, and the modern component upgrades.
- **12–13 — Post-training.** Turn a base model into an aligned assistant: SFT + LoRA, then reward models, RLHF, and DPO.

**Suggested pace:** ~9–10 hours total. Two sittings of 01–07 and 08–13 works well. Don't
just *read* — run every cell, then change a number and predict what happens before re-running.

---

## Per-notebook objectives & self-check

You understand a notebook when you can answer its questions **without looking**.

### 01 — Evolution
- *Goal:* Why Transformers replaced RNNs.
- *Self-check:* Why can't an RNN parallelize over sequence length? What problem does attention solve that recurrence struggled with?

### 02 — Embeddings
- *Goal:* Text → vectors, and why position must be injected.
- *Self-check:* Why do we add positional information at all? What breaks if we don't? Sinusoidal vs learned positions — trade-off?

### 03 — Attention
- *Goal:* Self-attention from scratch (NumPy).
- *Self-check:* What do Q, K, V each represent? Why divide by `√d_k`? Where does the causal mask go and why `−∞`?

### 04 — Multi-Head Attention
- *Goal:* Parallel attention in subspaces.
- *Self-check:* Why multiple heads instead of one big one? What are the tensor shapes before and after splitting heads?

### 05 — Transformer Block
- *Goal:* Assemble attention + FFN + residual + norm.
- *Self-check:* What does the residual connection do for gradients? Why Pre-LN over Post-LN? What does the FFN add that attention can't?

### 06 — Full Transformer
- *Goal:* Stack blocks into a GPT; the LM head; weight tying.
- *Self-check:* Trace a tensor from `input_ids` to `logits`, naming every shape. What is weight tying and why use it?

### 07 — Text Generation
- *Goal:* Train the model and sample from it.
- *Self-check:* What loss is minimized? Why does the model improve? What does temperature change?

### 08 — Training
- *Goal:* The real training machinery.
- *Self-check:* What is perplexity intuitively? Why AdamW + warmup + cosine + clipping? How do you *prove* your training loop works? (Hint: overfit one batch.)

### 09 — Inference
- *Goal:* Decoding strategies + the KV cache.
- *Self-check:* When is greedy bad? Top-k vs top-p? Why does the KV cache make generation roughly linear instead of quadratic?

### 10 — Encoders & Seq2Seq
- *Goal:* The three architecture families.
- *Self-check:* Why can BERT attend bidirectionally but GPT cannot? What is cross-attention? When would you pick encoder-decoder over decoder-only?

### 11 — Modern Architectures
- *Goal:* From the 2017 paper to LLaMA-era models.
- *Self-check:* What does RMSNorm drop vs LayerNorm? What makes RoPE "relative"? How does GQA shrink the KV cache? Is FlashAttention a different *result* or a faster *computation*?

### 12 — Instruction Tuning & LoRA
- *Goal:* Base model → instruction-follower, cheaply.
- *Self-check:* Why mask the prompt tokens in the loss? What exactly does LoRA freeze and train? Why initialize `B = 0`? Why can a LoRA adapter be *merged* with no inference cost?

### 13 — Preference Alignment
- *Goal:* Align to human preferences (RLHF & DPO).
- *Self-check:* Why train a reward model on *comparisons* instead of scores? What does the KL penalty prevent in RLHF? How does DPO avoid needing a reward model and an RL loop?

---

## Capstone: explain the whole pipeline

If you can narrate this end-to-end, you've reached expert-level understanding:

> *Raw text* → tokenize → **pretrain** a decoder-only Transformer (RMSNorm + RoPE + SwiGLU
> + GQA) with next-token cross-entropy under [scaling laws] → **SFT** on (instruction,
> response) pairs with prompt-masked loss (optionally via **LoRA**) → **align** to human
> preferences with a reward model + **RLHF/PPO**, or directly with **DPO** → serve with
> KV-cache + nucleus sampling.

Every bolded piece is a notebook in this repo. Then read the ⭐ papers in
[references.md](references.md) to see how the frontier labs scale each step.

---

## A note on the toy scale

These notebooks use a **character-level** tokenizer and a **tiny** model so everything runs
on a CPU in minutes. That is a deliberate choice to expose *mechanics*, not to produce a
capable model. The exact same code and concepts scale to billions of parameters — what
changes is the data, the compute, and the engineering, not the ideas. The
[references](references.md) show how each idea looks at full scale.
