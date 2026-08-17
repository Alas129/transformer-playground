# Cheat Sheet

Formulas, tensor shapes, and quick-reference tables. Notation: `B` = batch, `T` = sequence
length, `d` = model dim (`d_model`), `h` = number of heads, `d_k = d/h`, `V` = vocab size.

---

## Tensor shapes through a GPT forward pass

```
input_ids            (B, T)               token indices
└─ embedding         (B, T, d)            token emb + positional emb
   └─ × N blocks:
      ├─ Q,K,V       (B, h, T, d_k)       split into heads
      ├─ scores      (B, h, T, T)         QKᵀ/√d_k, causally masked
      ├─ attn·V      (B, h, T, d_k)       weighted values
      ├─ concat+W_o  (B, T, d)            merge heads
      └─ FFN         (B, T, d)            d → 4d → d
   └─ final norm     (B, T, d)
└─ lm_head           (B, T, V)            logits over vocabulary
```

---

## Core formulas

**Scaled dot-product attention**

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$$

`M` is the mask: `0` where allowed, `−∞` where forbidden (future positions, in causal attention).

**Multi-head attention**

$$\text{MHA}(x) = \text{Concat}(\text{head}_1,\dots,\text{head}_h)\,W_O,\quad \text{head}_i=\text{Attention}(xW_Q^i, xW_K^i, xW_V^i)$$

**Transformer block (Pre-LN, as in this repo)**

$$x \leftarrow x + \text{MHA}(\text{Norm}(x)) \qquad x \leftarrow x + \text{FFN}(\text{Norm}(x))$$

**LayerNorm vs RMSNorm**

$$\text{LayerNorm}(x)=\frac{x-\mu}{\sigma}\odot\gamma+\beta \qquad\quad \text{RMSNorm}(x)=\frac{x}{\sqrt{\tfrac1d\sum x_i^2}}\odot\gamma$$

RMSNorm drops the mean (`μ`) and bias (`β`) — re-scaling only.

**RoPE (idea)** — rotate each 2-D slice of Q and K by an angle proportional to position `m`:
the dot product `q_m·k_n` then depends only on the *relative* offset `m−n`.

**Feed-forward variants**

$$\text{GELU-MLP}: W_2\,\text{GELU}(W_1 x) \qquad \text{SwiGLU}: W_2\big(\text{Swish}(W_1 x)\odot W_3 x\big)$$

---

## Training

**Cross-entropy / next-token loss** (the whole objective):

$$\mathcal{L} = -\frac{1}{T}\sum_{t} \log p_\theta(x_t \mid x_{<t}) \qquad \text{Perplexity} = e^{\mathcal{L}}$$

**Softmax with temperature** `τ`:  $p_i = \dfrac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}$  (`τ<1` sharper, `τ>1` flatter).

| Knob | Effect |
|---|---|
| AdamW | Adam + decoupled weight decay; default LR ~`3e-4` for small models |
| Warmup + cosine decay | Ramp LR up over first steps, anneal to ~0 on a cosine |
| Gradient clipping | `clip_grad_norm_(params, 1.0)` |
| Weight init | `N(0, 0.02)` for Linear/Embedding |

**Chinchilla rule of thumb:** for a fixed compute budget, scale parameters and training
tokens *equally* (≈ **20 tokens per parameter**).

---

## Decoding (notebook 09)

| Strategy | Picks | Use for |
|---|---|---|
| Greedy | argmax | deterministic, short factual answers (repetitive on long text) |
| Temperature | sample after scaling logits | dial randomness |
| Top-k | sample from top `k` tokens | open-ended generation |
| Top-p (nucleus) | sample from smallest set with cum-prob ≥ `p` | best general default (`p≈0.9`) |
| Beam search | best `b` running sequences | translation / constrained output |

**KV cache:** store past K,V → generating token `t` is O(T) work, not O(T²). Memory per
token scales with `n_layers · n_kv_heads · d_k` — which is exactly what **GQA/MQA** shrink.

---

## Post-training (notebooks 12–13)

**LoRA** — freeze `W`, learn low-rank update (`B` initialized to 0 → starts as a no-op):

$$W' = W + \frac{\alpha}{r} B A,\qquad A\in\mathbb{R}^{r\times d_{in}},\; B\in\mathbb{R}^{d_{out}\times r},\; r\ll d$$

Merge for zero inference cost: `W ← W + (α/r)·B·A`.

**Reward model (Bradley–Terry)** on a preferred `y_w` vs rejected `y_l`:

$$\mathcal{L}_{RM} = -\log \sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)$$

**RLHF objective:** maximize  $\mathbb{E}[\,r_\phi(x,y) - \beta\,\mathrm{KL}(\pi_\theta\|\pi_{ref})\,]$  (optimized with PPO).

**DPO loss** (same goal, no reward model, no RL):

$$\mathcal{L}_{DPO} = -\log\sigma\!\Big(\beta\big[(\log\pi_\theta(y_w|x)-\log\pi_{ref}(y_w|x)) - (\log\pi_\theta(y_l|x)-\log\pi_{ref}(y_l|x))\big]\Big)$$

---

## 2017 original → modern LLM (notebook 11)

| Component | 2017 "Attention Is All You Need" | LLaMA-era LLM |
|---|---|---|
| Normalization | LayerNorm, Post-LN | RMSNorm, Pre-LN |
| Position | Sinusoidal absolute | RoPE (rotary, relative) |
| FFN activation | ReLU | SwiGLU |
| Attention | Multi-head (MHA) | Grouped-Query (GQA) |
| Attention kernel | Naive `softmax(QKᵀ)V` | FlashAttention |
| Scale | Millions of params | Billions+, often Mixture-of-Experts |

---

## This repo's model configs

| | `d_model` | heads | kv_heads | layers | `d_ff` | block size |
|---|---|---|---|---|---|---|
| `create_gpt_small` (`gpt.py`) | 128 | 4 | 4 | 4 | 512 | 256 |
| `create_gpt_medium` (`gpt.py`) | 256 | 8 | 8 | 6 | 1024 | 256 |
| `create_modern_small` (`modern.py`) | 128 | 4 | 2 | 4 | ~352 | 256 |

`d_ff = 4·d_model` (2017) or `≈(8/3)·d_model` for SwiGLU.

**`train_gpt` defaults** (`src/train.py`): AdamW at LR `3e-4` with weight decay `0.1` on
matrices only (gains and biases are excluded), 2% linear warmup then cosine decay to `0.1×`
peak, grad clip `1.0`, contiguous 10% validation split, dataset stride `= seq_len`, seed `0`.

**Two initialization details** that are easy to get wrong and easy to miss:

- The `sqrt(d_model)` factor on token embeddings is **only** for the fixed sinusoidal
  encoding, whose entries are of order 1. With a *learned* position table both sides are
  learned, and the factor just starts the token signal `sqrt(d)` above the position signal.
  `TokenEmbedding(..., scale=False)` is the default; `TransformerEmbedding` turns it on for
  the sinusoidal path only.
- Projections that **write into the residual stream** (`W_o`, the FFN's second layer) are
  scaled by `1/sqrt(2·num_layers)` at init. Without it the residual stream's variance grows
  with depth.

---

# Track A — Model core (NB 14–20)

## Tokenization (NB 14)

- **BPE**: count adjacent pairs → merge most frequent → record → repeat. Apply merges in
  **training order** at encode time.
- **Byte-level** base vocab = 256 → no `UNK` possible. Contrast `CharTokenizer`, which drops
  unseen chars.
- Trade-off: compression **saturates** with vocab size; embedding cost grows **linearly**. Real
  vocabs 32k–200k, growing for multilingual fertility.

## Long context (NB 15)

RoPE angle `= pos · inv_freq[i]`, unbounded in `pos` → low-frequency bands break first.

| Method | Change | Best for |
|---|---|---|
| PI | `pos → pos/s` (all bands) | pure long-range retrieval |
| NTK | `base → base·s^(d/(d−2))` (slow bands more) | balanced |
| YaRN | per-band ramp + attn temp `1/(0.1·ln s + 1)` | language modeling |
| ALiBi | `−m_h·(i−j)` bias, no rotation | extrapolation by construction |

**Attention sink**: never evict position 0. **KV cache** `= 2·L·n_kv·d_k·T·B·bytes` — the real
long-context limit.

## Attention variants (NB 16)

Quadratic crossover: attention FLOPs overtake projections at `T = 2·d_model`.

| Scheme | Cache/token (per layer) |
|---|---|
| MHA | `2·h·d_k` |
| GQA (g:1) | `2·(h/g)·d_k` |
| MQA | `2·d_k` |
| **MLA** | `d_c + rope_dim` (≈57× < MHA at V2 scale) |

**Linear attention**: `softmax(QK)V → φ(Q)(φ(K)ᵀV)` = O(N) = RNN with matrix state. Recurrence
`S_i = S_{i-1} + φ(k_i)v_iᵀ`. **Mamba** makes `Δ,B,C` input-dependent (selective). **Hybrids** add
a few attention layers for exact recall.

## Mixture-of-Experts (NB 17)

Total params scale with `E`; active FLOPs with `top_k`. Per-block: `12d²` attn + `E·(2·d_ff·d)` FFN.

- **Aux loss**: `L_aux = E·Σ f_i·P_i` (1.0 at balance). Competes with the task.
- **Loss-free bias**: `b_i −= γ·sign(load_i − 1/E)`, on **selection only** (DeepSeek-V3).
- **Router z-loss**: `mean(logsumexp(logits)²)` keeps the softmax unsaturated.
- **Capacity** `= factor·top_k·N/E`; overflow tokens dropped to the residual.

## Reasoning & test-time compute (NB 18)

- CoT rents serial steps a fixed-depth pass cannot do internally.
- **GRPO advantage**: $A_i = (r_i − \text{mean}(r))/(\text{std}(r) + \epsilon)$ — no value network.
- **GRPO loss**: $-\mathbb{E}[\min(\rho_i A_i, \text{clip}(\rho_i, 1{-}\epsilon, 1{+}\epsilon)A_i)] + \beta\,\text{KL}(\pi_\theta \| \pi_{ref})$
- All-equal-reward group → zero advantage → zero gradient. Filter degenerate groups.
- **RLHF** (preference, reward model) vs **RLVR** (correctness, automatic checker).

## Efficiency (NB 19)

Quantize: `scale = max|x|/qmax; q = round(x/scale); x' = q·scale`. Keep `scale` small → per-group.

- **Range beat precision**: BF16 (8-bit exp) beat FP16 (5-bit, overflows at 65504).
- **SmoothQuant**: `X → X/s`, `W → sW` (exact) migrates activation outliers into weights.
- **Distillation**: `L = T²·KL(softmax(teacher/T) ‖ softmax(student/T))`. On-policy > off-policy.
- int8 ≈ free, int4 cheap with small groups, <3-bit collapses. Larger models quantize better.

## Multimodal (NB 20)

- ViT: image → patches → 1 linear layer = tokens. A 224×224 image at patch 16 = 196 tokens.
- CLIP loss (symmetric InfoNCE): $\tfrac12[\text{CE}(sZ, I) + \text{CE}(sZ^\top, I)]$, `s` learned.
- VLM: ViT patches → MLP → prepend to LLM tokens (LLaVA); or cross-attention (Flamingo); or early
  fusion (Chameleon).

---

# Track B — Systems (NB 21–23)

## Performance first principles (NB 21)

- **Machine balance** `= peak FLOP/s ÷ peak bytes/s` (~300 FLOP/byte on an H100).
- **Arithmetic intensity** `= FLOPs ÷ bytes moved`. `> balance` → compute-bound; `<` → memory-bound.
- **Prefill** intensity `≈ 2T` (compute-bound); **decode** `≈ 2B` (memory-bound until batch ~300).
- Params `≈ 12Ld²`; forward `≈ 2NP`; **training `≈ 6NP`** (2 fwd + 4 bwd).
- Training memory: **16–18 bytes/param** (weights 2 + grad 2 + AdamW 8 + master 4). Optimizer state
  dominates.

## Distributed training (NB 22)

- **all-reduce** `= 2S(N−1)/N` `=` reduce-scatter + all-gather.
- **ZeRO**: shard optimizer(1) → +grad(2) → +param(3, FSDP). Stages 1–2 nearly free.
- **TP** MLP: column-parallel then row-parallel → 1 all-reduce. Intra-node only.
- **Pipeline bubble** `= (P−1)/(M+P−1)`; need `M ≫ P`.
- Placement by frequency: TP (2×/layer) intra-node; DP (1×/step) inter-node.

## Inference serving (NB 23)

- **Continuous batching**: retire + admit per iteration.
- **PagedAttention**: fixed KV blocks + block table; fragmentation ≤ 1 block/seq.
- **Speculative** expected tokens/pass `= (1 − p^{k+1})/(1 − p)`; exact via rejection sampling.
- Preemption: **recompute** (fast) over swap (slow PCIe).

---

# Track C — Applications (NB 24–26)

## RAG (NB 24)

- **BM25**: `Σ IDF(t)·tf·(k1+1) / (tf + k1(1 − b + b·|d|/avgdl))`. Beats dense on exact strings.
- **RRF**: `Σ_r 1/(k + rank_r(d))`, `k≈60`. Fuses sparse + dense, no score normalization.
- **Cascade**: retrieve N (bi-encoder) → rerank top-N (cross-encoder) → keep few.
- **recall@k** bounds the whole system. Measure it first, separately from generation.

## Agents (NB 25)

- Ladder: prompt → chain → route → parallelize → orchestrate → **agent**. Prefer the lowest rung.
- Tool loop: model emits call → **harness** executes → result to context → repeat.
- Prompt injection: unsolved. Defend architecturally (least privilege, human-in-loop, sandbox).

## Production (NB 26)

- **SLOs at p95/p99**, never the mean.
- **S-LoRA**: shared base + stacked adapters, mixed in one batch. Base matmul dominates → throughput preserved.
- **Cascade routing** dominates any single tier on the cost/quality frontier.
- **LLM judge**: randomize position, pairwise compare, calibrate vs humans (κ > 0.6).
