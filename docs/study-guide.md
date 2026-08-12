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

## Track A — Model core (14–20)

The frontier model-level knowledge the 2017 paper did not have. Runs on CPU with only
`requirements.txt`.

### 14 — Tokenization
- *Goal:* Build byte-level BPE and understand what tokenization breaks.
- *Self-check:* Why does byte-level BPE have no `UNK`? Why apply merges in training order? Why do models miscount letters and struggle with arithmetic?

### 15 — Long Context
- *Goal:* Why models break past their training length, and the four fixes.
- *Self-check:* Which RoPE frequency bands break first, and why? How does NTK differ from PI? Why does evicting position 0 collapse a sliding window? Why is the KV cache the real limit?

### 16 — Attention Variants
- *Goal:* MLA, linear attention, SSMs, and why hybrids win.
- *Self-check:* What does MLA cache, and why is it smaller than MQA's? Show that causal linear attention is a recurrence. What breaks if an SSM's `Δ,B,C` are fixed? Why does one attention layer in eight recover most of the recall?

### 17 — Mixture-of-Experts
- *Goal:* Decouple capacity from compute; make routing work.
- *Self-check:* `topk` has no gradient — how does the router learn? Why is collapse self-reinforcing? Why does loss-free balancing beat an auxiliary loss? Why is sparsity per-token but not per-batch?

### 18 — Reasoning & Test-Time Compute
- *Goal:* CoT, verifiers, RLVR, and GRPO from scratch.
- *Self-check:* Why can't a small model solve a k-step problem in one pass, regardless of width? What does GRPO replace, and what does it save? What does RLVR give up in exchange for needing no reward model? Is a CoT trace an explanation?

### 19 — Efficiency
- *Goal:* Quantization, distillation, pruning.
- *Self-check:* Why did BF16 beat FP16 despite fewer mantissa bits? Why are activations harder to quantize than weights? What does SmoothQuant migrate, and why is the matmul unchanged? Why did unstructured pruning lose to quantization on GPUs?

### 20 — Multimodal
- *Goal:* ViT, CLIP, and how vision attaches to an LLM.
- *Self-check:* Why is a strided conv identical to per-patch linear projection? What inductive bias does ViT lack? Where do CLIP's negatives come from? Which parameters train in LLaVA stage 1?

---

## Track B — Systems (21–23)

How the model is trained across thousands of GPUs and served to thousands of users. NB 21 is the
foundation — read it before 22 or 23. Track B's cells are mostly analytical and run in seconds.

### 21 — Performance First Principles
- *Goal:* Derive that prefill is compute-bound and decode is memory-bound.
- *Self-check:* Why is a matrix-vector product memory-bound but a matrix-matrix product not? At what batch size does decode become compute-bound? Why is training `6ND`? Why does the KV cache cap throughput?

### 22 — Distributed Training
- *Goal:* The five parallelism strategies and how to choose.
- *Self-check:* Why is optimizer state the memory problem, not weights? Why must the TP MLP be column-then-row? Why must TP stay intra-node while DP can cross the datacenter? Compute the pipeline bubble for 8 stages and 16 micro-batches.

### 23 — Inference Serving
- *Goal:* PagedAttention, continuous batching, speculative decoding.
- *Self-check:* Why does static batching waste so much? What is the max fragmentation per sequence under PagedAttention? Prove speculative decoding samples from the target distribution. Why prefer recompute over swap on preemption?

---

## Track C — Applications (24–26)

Turning the model into a product. Some cells use optional libraries (`sentence-transformers`,
`faiss`) and degrade gracefully without them.

### 24 — RAG
- *Goal:* Build the full retrieval pipeline from scratch.
- *Self-check:* When is fine-tuning the wrong tool for knowledge? Why does RRF fuse ranks not scores? What can a cross-encoder do that a bi-encoder cannot? How do you tell whether retrieval or generation failed?

### 25 — Agents
- *Goal:* Tool use, ReAct, memory, context engineering, safety.
- *Self-check:* What distinguishes a workflow from an agent, and why prefer a workflow? In the tool loop, what executes the tool? When does multi-agent beat single-agent? Why can't a model separate instructions from data?

### 26 — Production
- *Goal:* SLOs, multi-tenant serving, evaluation, safety, cost.
- *Self-check:* Why state SLOs at p99 not the mean? How does S-LoRA make multi-tenant serving economical? Why is a semantic cache's threshold a correctness concern? What are the three LLM-judge biases and how do you counter position bias?

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
