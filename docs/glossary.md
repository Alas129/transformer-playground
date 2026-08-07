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

## Tokenization (NB 14)

| Term | Definition | NB |
|---|---|---|
| **Byte Pair Encoding (BPE)** | Subword tokenizer: repeatedly merge the most frequent adjacent pair. Merges are applied in training order at encode time. | 14 |
| **Byte-level BPE** | BPE over raw UTF-8 bytes, so the base vocabulary is 256 symbols and no input is unencodable (no `UNK`). | 14 |
| **Pre-tokenization** | Splitting text (via regex) before BPE so merges never cross word boundaries. | 14 |
| **Fertility** | Tokens per word (or per character) for a language; high fertility is a cost tax on non-English text. | 14 |
| **Glitch token** | A token in the tokenizer's vocabulary but absent from training data, so its embedding stays near-random. | 14 |
| **WordPiece / Unigram** | Alternative subword algorithms: likelihood-gain merges (BERT) / top-down pruning (T5, SentencePiece). | 14 |

## Long context (NB 15)

| Term | Definition | NB |
|---|---|---|
| **Position Interpolation (PI)** | Divide all positions by a scale factor to fit a longer context into the trained range. | 15 |
| **NTK-aware scaling** | Raise the RoPE base instead of scaling positions, stretching low-frequency bands more than high. | 15 |
| **YaRN** | Per-frequency-band interpolation plus an attention-temperature correction; strong on language modeling. | 15 |
| **ALiBi** | Attention with Linear Biases: a per-head distance penalty instead of positional embeddings; extrapolates by construction. | 15 |
| **Sliding-window attention** | Restrict each token to the last `W` keys; makes cost and the KV cache linear in length. | 15 |
| **Attention sink** | Early positions absorb disproportionate attention mass (softmax must sum to 1); evicting them collapses quality. | 15 |
| **Needle in a haystack** | A retrieval probe: hide a fact in a long document and ask for it. Perplexity does not measure this. | 15 |

## Attention variants (NB 16)

| Term | Definition | NB |
|---|---|---|
| **Multi-head Latent Attention (MLA)** | Cache a low-rank latent instead of full K/V; reconstruct K/V on the fly. Smaller cache than even MQA. | 16 |
| **Decoupled RoPE** | Splitting keys into a compressed (no-RoPE) part and a small rotated part, so MLA's up-projection stays absorbable. | 16 |
| **Linear attention** | Replace softmax with a factorizable kernel so attention reassociates into an `O(N)` recurrence — an RNN with a matrix state. | 16 |
| **State space model (SSM)** | A learned linear recurrence (`h_t = Āh_{t-1} + B̄x_t`) with a decay matrix; can forget, unlike linear attention. | 16 |
| **Mamba / selective SSM** | An SSM whose `Δ`, `B`, `C` depend on the input, so it can choose what to remember. Breaks the convolutional form. | 16 |
| **Hybrid architecture** | Mostly recurrent/SSM layers with a few full-attention layers for exact retrieval (Jamba, Samba). | 16 |
| **Associative recall** | The task that separates architectures: retrieve a value by its key from a list. Attention passes; fixed-state models degrade. | 16 |
| **Delta rule** | A recurrence that overwrites a key's stored value instead of accumulating; basis of DeltaNet, Mamba-2. | 16 |

## Mixture-of-Experts (NB 17)

| Term | Definition | NB |
|---|---|---|
| **Router / gating** | The linear layer that scores experts and selects the top-k per token. | 17 |
| **Router collapse** | The default failure: the router concentrates on a few experts, leaving the rest untrained. | 17 |
| **Load-balancing (auxiliary) loss** | `E·Σ f_i·P_i` added to the loss; 1.0 at perfect balance. Competes with the task objective. | 17 |
| **Loss-free balancing** | Per-expert bias on selection only, nudged by `sign(load − 1/E)` (DeepSeek-V3). No loss term, no gradient distortion. | 17 |
| **Capacity factor** | Per-expert token budget multiplier; tokens over it are dropped onto the residual stream. | 17 |
| **Shared expert** | An always-active expert handling the common case so routed experts can specialize (DeepSeekMoE). | 17 |
| **Upcycling** | Initializing a MoE's experts from a trained dense FFN (how Mixtral was built from Mistral). | 17 |

## Reasoning & test-time compute (NB 18)

| Term | Definition | NB |
|---|---|---|
| **Chain of Thought (CoT)** | Emitting intermediate reasoning tokens, which rents additional serial computation a fixed-depth model cannot do internally. | 18 |
| **Self-consistency** | Sample N reasoning chains and take the majority answer. | 18 |
| **Test-time scaling** | Trading inference compute (more samples / longer thinking) for accuracy; a roughly logarithmic exchange rate. | 18 |
| **ORM / PRM** | Outcome vs Process Reward Model: grading the final answer vs every reasoning step. | 18 |
| **RLVR** | RL from Verifiable Rewards: use an automatic checker (math, code) as the reward, so no reward model is needed. | 18 |
| **GRPO** | Group Relative Policy Optimization: PPO without a value network, using a sampled group's mean reward as the baseline. | 18 |
| **CoT faithfulness** | Whether the emitted reasoning reflects the actual computation. Often it does not — a trace is not an audit log. | 18 |

## Efficiency (NB 19)

| Term | Definition | NB |
|---|---|---|
| **BF16 / FP8** | Reduced-precision floats; BF16 keeps FP32's range (why it beat FP16), FP8 has E4M3 (weights) and E5M2 (gradients) variants. | 19 |
| **Per-channel / group-wise quantization** | Giving each channel or small group its own scale; the main lever against quantization error. | 19 |
| **GPTQ / AWQ / SmoothQuant** | PTQ methods: Hessian error compensation / protect salient channels / migrate activation difficulty into weights. | 19 |
| **QAT / straight-through estimator** | Quantization-aware training; quantize in the forward pass, pass the gradient through unchanged in the backward. | 19 |
| **On-policy distillation** | Student generates, teacher scores its own outputs — corrects the student where it actually errs. Beats off-policy. | 19 |
| **2:4 structured sparsity** | Exactly 2 of every 4 weights zero; the one pruning form NVIDIA hardware can actually accelerate. | 19 |

## Multimodal (NB 20)

| Term | Definition | NB |
|---|---|---|
| **Vision Transformer (ViT)** | Cut an image into patches, linearly project each into a token, run a Transformer encoder. | 20 |
| **Patch embedding** | The per-patch linear projection (implemented as a strided conv); a learned tokenizer for pixels. | 20 |
| **CLIP** | Dual image/text encoders trained contrastively into a shared space, enabling zero-shot classification. | 20 |
| **VLM (projection splice)** | LLaVA-style: run image patches through an MLP and prepend them to the LLM's tokens. | 20 |
| **Visual-token explosion** | High-resolution images cost thousands of tokens; pixel shuffle / Q-Former / tiling manage it. | 20 |

## Performance & systems (NB 21–23)

| Term | Definition | NB |
|---|---|---|
| **Arithmetic intensity** | FLOPs per byte moved. Compared to machine balance, it decides compute-bound vs memory-bound. | 21 |
| **Roofline** | Achievable performance vs intensity: a bandwidth diagonal rising to a compute ceiling. | 21 |
| **Prefill vs decode** | Prefill (`≈2T` intensity) is compute-bound; decode (`≈2B`) is memory-bound until batch ~300. The key serving fact. | 21 |
| **MFU / MBU** | Model FLOPs / Bandwidth Utilization. Report MFU for training, MBU for decode. | 21 |
| **Data / tensor / pipeline / expert parallelism** | Split by batch / matmul / layers / experts. Communication frequency dictates placement. | 22 |
| **ZeRO / FSDP** | Shard optimizer state (1), gradients (2), and parameters (3) across data-parallel ranks. | 22 |
| **Pipeline bubble** | Idle time `(P−1)/(M+P−1)` from pipeline stages waiting; needs micro-batches `M ≫ P`. | 22 |
| **Continuous batching** | Schedule at iteration granularity so finished sequences leave and new ones join every step. | 23 |
| **PagedAttention** | Manage the KV cache in fixed blocks with a block table; bounds fragmentation to one block per sequence. | 23 |
| **Prefix caching (radix)** | Share cached KV blocks across requests with a common prefix (system prompts, multi-turn). | 23 |
| **Chunked prefill** | Interleave prompt processing with decode so a long prefill does not stall streaming users. | 23 |
| **Speculative decoding** | Draft `k` tokens with a small model, verify in one target pass; rejection sampling keeps the output exact. | 23 |

## Applications (NB 24–26)

| Term | Definition | NB |
|---|---|---|
| **Chunking** | Splitting documents into retrievable units; sentence-aware is the sensible default, and size is worth tuning. | 24 |
| **BM25** | The classical sparse retriever: TF saturation + IDF + length normalization. Still beats dense on exact strings. | 24 |
| **Bi-encoder / cross-encoder** | Encode query and doc separately (fast, precomputable) vs jointly (accurate, per-pair). Used in a retrieve-then-rerank cascade. | 24 |
| **Reciprocal Rank Fusion (RRF)** | Combine rankings by `Σ 1/(k+rank)`; fuses sparse and dense without score normalization. | 24 |
| **recall@k** | Was a correct chunk in the top k? A ceiling on the whole RAG system's quality. | 24 |
| **HyDE** | Hypothetical Document Embeddings: embed a generated fake answer, which looks more like a document than a question. | 24 |
| **Workflow vs agent** | Fixed control flow vs model-decided control flow. Most "agents" are workflows, and should be. | 25 |
| **Tool use / ReAct** | The model emits a structured call the harness executes; ReAct interleaves reasoning with acting. | 25 |
| **Context engineering** | Budgeting the context window: trimming tool results, compaction, sub-agent isolation. | 25 |
| **Indirect prompt injection** | Untrusted content containing instructions the agent may follow; the central agent security risk, unsolved. | 25 |
| **S-LoRA** | Multi-tenant serving: one shared base plus stacked adapters, mixed within a batch. | 26 |
| **Cascade routing** | Try a small model first, escalate on low confidence; a Pareto-dominant cost/quality point. | 26 |
| **LLM-as-a-judge** | Using a strong model to grade open-ended output; has position/length/self-preference biases needing calibration. | 26 |
| **SLO** | A latency/availability promise at a percentile (p95/p99), not the mean — the tail is what users feel. | 26 |
