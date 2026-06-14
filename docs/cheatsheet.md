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

## This repo's model configs (`src/gpt.py`)

| | `d_model` | heads | layers | `d_ff` | block size |
|---|---|---|---|---|---|
| `create_gpt_small` | 128 | 4 | 4 | 512 | 256 |
| `create_gpt_medium` | 256 | 8 | 6 | 1024 | 256 |

`d_ff = 4·d_model`. Default optimizer: AdamW, LR `3e-4`, cosine schedule, grad clip `1.0`.
