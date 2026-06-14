# Glossary

One-line definitions for every key term in the course. The **NB** column points to the
notebook where the concept is introduced. See [references.md](references.md) for the
original sources.

## Foundations

| Term | Definition | NB |
|---|---|---|
| **Token** | The atomic unit a model reads/predicts (a character here; a subword in real LLMs). | 02 |
| **Tokenizer** | Maps text ↔ integer token IDs. This repo uses a character-level tokenizer (`CharTokenizer`). | 02 |
| **Embedding** | A learned dense vector for each token ID; the model's input representation. | 02 |
| **Positional encoding** | Information about *where* a token sits in the sequence, added because attention is order-agnostic. Sinusoidal (fixed) or learned. | 02 |
| **Logits** | Raw, unnormalized scores over the vocabulary output by the final layer. | 06 |
| **Softmax** | Turns logits into a probability distribution that sums to 1. | 03 |
| **Autoregressive** | Generating one token at a time, each conditioned on all previous tokens. | 07 |

## Attention

| Term | Definition | NB |
|---|---|---|
| **Self-attention** | Each token computes a weighted sum over all tokens, weights based on relevance. The core Transformer operation. | 03 |
| **Query / Key / Value (Q/K/V)** | Three learned projections of each token: Q asks, K advertises, V carries content. | 03 |
| **Scaled dot-product attention** | `softmax(QKᵀ/√d_k)·V` — the attention formula, scaled by `√d_k` for stable gradients. | 03 |
| **Multi-head attention** | Run attention in parallel `h` times in separate subspaces, then concatenate. | 04 |
| **Causal / masked attention** | A mask preventing a token from attending to *future* tokens; required for autoregressive generation. | 03, 06 |
| **Cross-attention** | Attention where Q comes from the decoder and K/V from the encoder (used in seq2seq). | 10 |
| **Attention head** | One independent Q/K/V/output projection set within multi-head attention. | 04 |

## Architecture

| Term | Definition | NB |
|---|---|---|
| **Transformer block** | Attention + feed-forward, each wrapped with a residual connection and normalization. | 05 |
| **Feed-forward network (FFN/MLP)** | Two linear layers with a nonlinearity (GELU here); applied independently per position. | 05 |
| **Residual connection** | `x + sublayer(x)` — lets gradients and information skip layers; essential for depth. | 05 |
| **Layer normalization (LayerNorm)** | Normalizes activations per token (re-center + re-scale) to stabilize training. | 05 |
| **Pre-LN vs Post-LN** | Whether normalization comes *before* (pre, used here — more stable) or *after* the sublayer. | 05 |
| **Weight tying** | Sharing the input embedding matrix with the output projection (`lm_head`). | 06 |
| **Decoder-only / encoder-only / encoder-decoder** | The three architecture families: GPT / BERT / original Transformer (T5, BART). | 06, 10 |
| **Masked language modeling (MLM)** | BERT's objective: predict randomly masked tokens using bidirectional context. | 10 |

## Modern components

| Term | Definition | NB |
|---|---|---|
| **RMSNorm** | LayerNorm without the mean-centering step — cheaper, used in LLaMA-era models. | 11 |
| **RoPE (Rotary Position Embedding)** | Encodes position by *rotating* Q/K vectors; gives relative-position awareness and length extrapolation. | 11 |
| **SwiGLU** | A gated feed-forward variant (Swish-gated GLU) that outperforms a plain GELU MLP. | 11 |
| **Grouped-Query Attention (GQA)** | Multiple query heads share a smaller set of key/value heads — shrinks the KV cache. | 11 |
| **Multi-Query Attention (MQA)** | The extreme of GQA: all query heads share a *single* K/V head. | 11 |
| **FlashAttention** | An IO-aware GPU kernel computing exact attention without materializing the full score matrix. | 11 |
| **Mixture of Experts (MoE)** | Replace one FFN with many "experts"; a router activates only a few per token (sparse compute). | 11 |

## Training

| Term | Definition | NB |
|---|---|---|
| **Cross-entropy loss** | The next-token prediction loss; equivalent to maximum-likelihood estimation. | 08 |
| **Perplexity** | `exp(loss)` — an interpretable "average branching factor"; lower is better. | 08 |
| **AdamW** | Adam optimizer with *decoupled* weight decay; the standard for Transformers. | 08 |
| **Learning-rate warmup + cosine decay** | Ramp LR up, then anneal it down on a cosine curve — stabilizes early training. | 08 |
| **Gradient clipping** | Cap the gradient norm (e.g. 1.0) to prevent destabilizing spikes. | 08 |
| **Gradient accumulation** | Sum gradients over several mini-batches to simulate a larger batch in limited memory. | 08 |
| **Mixed precision** | Use 16-bit floats for speed/memory, keeping a 32-bit master copy for stability. | 08 |
| **Overfitting** | Memorizing training data instead of generalizing; train loss ↓ while val loss ↑. | 08 |
| **Scaling laws** | Empirical power-law relating loss to model size, data, and compute (Kaplan; Chinchilla). | 11 |

## Inference & decoding

| Term | Definition | NB |
|---|---|---|
| **Greedy decoding** | Always pick the highest-probability next token. Deterministic, often repetitive. | 09 |
| **Temperature** | Scales logits before softmax: <1 sharper/safer, >1 flatter/more random. | 07, 09 |
| **Top-k sampling** | Sample only from the `k` most likely tokens. | 09 |
| **Top-p (nucleus) sampling** | Sample from the smallest set of tokens whose cumulative probability ≥ `p`. | 09 |
| **Beam search** | Keep the `b` best partial sequences; good for translation, poor for open-ended text. | 09 |
| **Repetition penalty** | Down-weights already-generated tokens to reduce loops. | 09 |
| **KV cache** | Cache past keys/values so each new token is O(1) attention work instead of O(n). | 09 |

## Post-training & alignment

| Term | Definition | NB |
|---|---|---|
| **Base model** | A pretrained model that *continues* text but does not *follow instructions*. | 12 |
| **Supervised Fine-Tuning (SFT)** | Fine-tune on (instruction, response) pairs so the model learns to answer. | 12 |
| **Instruction tuning** | SFT on a broad, diverse set of instructions to make a general instruction-follower. | 12 |
| **Loss masking** | Setting target labels to `-1` (ignore) on prompt tokens so loss is computed on the *response* only. | 12 |
| **LoRA** | Low-Rank Adaptation: freeze `W`, learn a small `(α/r)·B·A` update. Cheap, mergeable. | 12 |
| **QLoRA** | LoRA on top of a 4-bit quantized frozen base — fine-tune huge models on one GPU. | 12 |
| **Reward model (RM)** | A model that scores responses by predicted human preference. | 13 |
| **Bradley–Terry model** | Turns pairwise "A > B" comparisons into a trainable reward via `−log σ(r_A − r_B)`. | 13 |
| **RLHF** | Reinforcement Learning from Human Feedback: optimize the policy toward RM reward under a KL penalty. | 13 |
| **PPO** | Proximal Policy Optimization — the RL algorithm typically used inside RLHF. | 13 |
| **KL penalty** | Keeps the RLHF policy close to the SFT reference, preventing reward-hacking. | 13 |
| **DPO** | Direct Preference Optimization: RLHF's objective rewritten as a simple supervised loss — no reward model, no RL. | 13 |
| **Reference model** | A frozen copy of the SFT model that DPO measures the policy's drift against. | 13 |
| **RLAIF / Constitutional AI** | Generate preference labels with an AI (guided by a written constitution) instead of humans. | 13 |
