# Systems Architecture Reference

A reference companion to Track B (notebooks 21–23) and the production notebook (26). Where the
notebooks *derive* these results with runnable code, this document collects the decision tables
and formulas for quick lookup.

The whole of systems-level LLM engineering rests on one result from notebook 21, so it comes
first.

---

## The one result everything follows from

**Arithmetic intensity** = FLOPs performed ÷ bytes moved. Compare it to the hardware's **machine
balance** = peak FLOP/s ÷ peak bytes/s (a few hundred FLOP/byte on modern accelerators).

| Phase | Tokens per weight-read | Intensity | Bound by | Optimize by |
|---|---|---|---|---|
| **Prefill** | `T` (whole prompt) | `≈ 2T` — high | **Compute** | Faster math: quantized *compute*, better kernels |
| **Decode** | `B` (batch size) | `≈ 2B` — low until batch ~300 | **Memory bandwidth** | Fewer *bytes*: smaller cache, quantized weights, bigger batch |

Every technique below is a consequence:

- **Batching** raises decode throughput almost for free (moves it right along the roofline).
- **GQA / MLA / quantization** cut bytes, which is decode's actual bottleneck.
- **Speculative decoding** spends idle decode FLOPs to buy fewer serial steps.
- **FlashAttention** is an IO optimization — same FLOPs, fewer bytes.
- **Prefill/decode disaggregation** follows from the two phases wanting different hardware.

---

## Memory accounting

### Training (per parameter)

| Component | Bytes (bf16 + AdamW) |
|---|---|
| Weights | 2 |
| Gradients | 2 |
| AdamW moment 1 (fp32) | 4 |
| AdamW moment 2 (fp32) | 4 |
| fp32 master weights | 4 |
| **Total** | **16–18** |

Weights are ~11% of the total. **Optimizer state is what does not fit**, which is why sharding it
(ZeRO-1) is the highest-leverage memory optimization. Activations are on top of this and dominate
at long sequence length — see recomputation below.

### Inference (per model)

```
memory = weights + KV cache
KV cache = 2 · n_layers · n_kv_heads · head_dim · seq_len · batch · bytes_per_element
```

The cache scales with **batch × length** while weights are shared, so at serving scale the cache
dominates and **caps the batch size, which caps throughput**.

### Key numbers (Llama-3-8B, fp16, GQA 8 kv heads)

| batch × seq | Weights | KV cache | Cache share |
|---|---|---|---|
| 1 × 2048 | 16 GB | 1 GB | 6% |
| 32 × 2048 | 16 GB | 34 GB | 68% |
| 32 × 32768 | 16 GB | 537 GB | 97% |

---

## Distributed training: strategy selection

### The five strategies

| Strategy | Splits | Communicates | Frequency | Interconnect |
|---|---|---|---|---|
| **DP / ZeRO** | Batch (+ optimizer/grad/param state) | Gradients | Per step | Across nodes |
| **TP** (tensor) | Individual matmuls | Activations | 2× per layer | **Intra-node (NVLink) only** |
| **PP** (pipeline) | Layers | Boundary activations | Per stage boundary | Across nodes |
| **SP / Ring** (sequence) | The token axis | K/V blocks | Per attention | Intra-node preferred |
| **EP** (expert) | MoE experts | Tokens (all-to-all) | 2× per MoE layer | Intra-node preferred |

**Placement rule: highest-frequency communication gets the fastest link.** TP goes on NVLink
inside a node; DP can cross the datacenter. Reversing this is the most common and most expensive
misconfiguration.

### ZeRO stages

| Stage | Shards | Memory/device | Cost vs DDP |
|---|---|---|---|
| 0 (DDP) | nothing | `16Ψ` | baseline |
| 1 | optimizer state | `4Ψ + 12Ψ/N` | ~free (reduce-scatter + all-gather = one all-reduce) |
| 2 | + gradients | `2Ψ + 14Ψ/N` | ~free |
| 3 (FSDP) | + parameters | `16Ψ/N` | + one param all-gather per layer per pass |

Default to ZeRO-1 or -2; escalate to ZeRO-3/FSDP only when the model still does not fit.

### Collectives and their cost

| Collective | Per-device bytes | Identity |
|---|---|---|
| all-reduce | `2S(N−1)/N` | = reduce-scatter + all-gather |
| reduce-scatter | `S(N−1)/N` | |
| all-gather | `S(N−1)/N` | |
| all-to-all | `S(N−1)/N` | MoE dispatch/combine |

Per-device volume does not grow with `N`; latency (number of hops) does.

### Pipeline bubble

```
bubble fraction = (P − 1) / (M + P − 1)
```

`P` stages, `M` micro-batches. Need `M ≫ P` — 8 stages with 8 micro-batches wastes 47% of the
cluster; with 64, only 10%. 1F1B scheduling holds the bubble constant while capping in-flight
activations at `P` instead of `M`.

### Activation recomputation

| Policy | Activation memory | Extra compute |
|---|---|---|
| None | full | — |
| **Selective** (drop only the `O(T²)` scores) | ~1/3 of full | +5% |
| Full | one tensor per layer | +33% |

Selective recomputation captures most of the benefit for little cost, and it is what
FlashAttention does as a side effect. Not optional at long context.

### Decision tree

```
1. Fits on one device (params × 18 B)?          YES -> DDP + ZeRO-1
2. Fits with ZeRO-2 sharding?                    YES -> ZeRO-2
3. Fast intra-node interconnect (NVLink)?        YES -> add TP <= devices_per_node
4. Still too large, or TP at node width?          -> add PP, ensure M >> P
5. Very long sequences?                           -> add sequence/context parallelism
6. MoE model?                                     -> expert parallelism, consider top-1 routing
7. Always                                         -> selective activation recomputation
```

### Real configurations

| Model | Parallelism |
|---|---|
| Llama 3 405B | TP=8, PP=16, DP=128, CP=2 (16k H100s) |
| DeepSeek-V3 | TP=1, PP=16, EP=64 (MLA + EP removed the need for TP) |
| Megatron 530B | TP=8, PP=35, DP=6 |

---

## Inference serving: technique selection

| Technique | Solves | Mechanism |
|---|---|---|
| **Continuous batching** | Padding waste, head-of-line blocking | Schedule per iteration: retire and admit every step |
| **PagedAttention** | KV fragmentation | Fixed blocks + block table; fragmentation ≤ one block/seq |
| **Prefix caching** (radix) | Repeated prefixes | Share KV blocks across requests via a radix tree |
| **Chunked prefill** | TTFT vs TPOT conflict | Interleave prompt chunks with decode in mixed batches |
| **Speculative decoding** | Serial decode latency | Draft `γ`, verify in one pass; exact via rejection sampling |
| **Disaggregation** | Prefill/decode interference | Separate machines for the compute-bound and memory-bound phases |

### Serving decision points

```
Throughput too low?          -> continuous batching + paged KV (raise the batch)
Memory-limited batch?        -> quantize KV cache (int8), GQA/MLA, prefix cache
TTFT too high?               -> chunked prefill, prefix cache, disaggregated prefill nodes
TPOT too high?               -> speculative decoding, smaller model via routing, quantized weights
Repeated prompts?            -> radix prefix cache (multi-turn, few-shot, shared system prompt)
Structured output needed?    -> constrained decoding (grammar-masked logits)
Preemption needed?           -> recompute (fast, compute-bound) over swap (slow PCIe)
```

### Latency metrics

| Metric | Governed by | Scales with |
|---|---|---|
| **TTFT** (time to first token) | Prefill | Prompt length |
| **TPOT / ITL** (per output token) | Decode | Constant per token |
| **Throughput** | Batch size | Concurrency (until compute-bound) |

Set SLOs at **p95/p99**, never the mean — the tail is what users feel.

### Speculative decoding

```
expected tokens per target pass = (1 − p^(γ+1)) / (1 − p)      [p = acceptance rate]
```

There is an optimal `γ` (drafting further wastes rejected work), and it grows with acceptance.
Draft-model-free variants: Medusa (extra heads), EAGLE (feature-level), n-gram (no model, best when
output quotes input).

---

## Cost model

Per-query cost is dominated by generation input tokens (i.e. context), which is why "fewer, better
retrieved chunks" (notebook 24) and prefix caching matter for cost, not just latency.

| Lever | Notebook | Typical saving |
|---|---|---|
| int8 / int4 quantization | 19 | 2–4× memory & decode throughput |
| Distillation | 19 | Replace a large model with a small one |
| Routing / cascades | 26 | Most queries to the cheap model |
| Prefix + semantic caching | 23, 26 | Skip repeated work |
| Continuous batching | 23 | Higher GPU utilization |
| Speculative decoding | 23 | Fewer sequential steps |

---

## Efficiency targets

| Metric | Definition | Good value |
|---|---|---|
| **MFU** (training) | achieved model FLOP/s ÷ peak | 40–55% |
| **MBU** (decode) | achieved bytes/s ÷ peak bandwidth | high; MFU is near-zero here and that is correct |

Report MFU for training, MBU for decode. Using MFU to judge decode is a category error — decode is
bandwidth-bound, so its MFU is intrinsically tiny.

---

## See also

- **Notebook 21** derives the machine-balance and prefill-vs-decode results with a measured roofline.
- **Notebook 22** implements tensor parallelism and runs a real gloo DDP + ZeRO-1 job.
- **Notebook 23** builds PagedAttention, continuous batching, and speculative decoding from scratch.
- **Notebook 26** applies all of it: multi-tenant serving, routing, evaluation, cost engineering.
