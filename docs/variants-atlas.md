# Variants Atlas — Transformers Beyond Text

Notebook 20 makes the case that a Transformer is **modality-agnostic**: it consumes a sequence
of vectors and does not care where they came from. This document is the evidence, organized by
domain.

Every entry answers the same three questions, because those three answers are all that
distinguishes one of these models from a language model:

1. **How does the input become tokens?** — the only genuinely modality-specific design choice.
2. **What is the objective?** — what the model is trained to predict.
3. **What changed from a language model?** — architecture deltas, if any.

---

## Vision

### ViT — Vision Transformer (2020)
- **Tokens**: non-overlapping 16×16 pixel patches, each flattened and passed through one shared
  linear layer. A 224×224 image gives 196 tokens.
- **Objective**: supervised classification from a `[CLS]` token; later variants use masked
  patch prediction (MAE) or contrastive objectives.
- **Delta from an LM**: bidirectional attention instead of causal (an image has no reading
  order), learned position embeddings over a raster-ordered grid. Otherwise a BERT encoder.
- Notebook 20 builds this using `src/transformer.py`'s `TransformerEncoder` unmodified.

### Swin Transformer (2021)
- **Tokens**: patches, but processed in **hierarchical stages** with progressive merging, so
  resolution falls and channel count rises — like a CNN's feature pyramid.
- **Objective**: classification, detection, segmentation.
- **Delta**: attention restricted to local windows, with the windows **shifted** between
  consecutive blocks so information still crosses boundaries. Reintroduces locality as an
  explicit prior, and makes cost linear in image area rather than quadratic.

### DETR — Detection Transformer (2020)
- **Tokens**: CNN feature-map cells as encoder input; a fixed set of learned **object queries**
  as decoder input.
- **Objective**: set prediction — each query emits one box and class, matched to ground truth
  by Hungarian assignment. No anchors, no non-maximum suppression.
- **Delta**: reframes detection as a set-to-set problem, which is what removed the
  hand-engineered post-processing stack from object detection.

### SAM — Segment Anything (2023)
- **Tokens**: image patches from a ViT-H encoder, plus **prompt tokens** encoding a point, box,
  or rough mask.
- **Objective**: predict a mask matching the prompt, trained on 1.1B masks.
- **Delta**: a lightweight mask decoder cross-attends between image and prompt tokens. Notable
  for making segmentation *promptable* — the same interface shift that made language models
  general.

### DINOv2 (2023)
- **Tokens**: ViT patches.
- **Objective**: self-distillation with no labels — a student matches a teacher's output under
  different crops.
- **Delta**: none architecturally. Listed because it shows self-supervision works for vision as
  it did for text, producing features strong enough to use frozen.

---

## Speech and audio

### Whisper (2022)
- **Tokens**: the waveform becomes an 80-bin **log-Mel spectrogram** — a 2-D time-frequency
  image — then two stride-2 convolutions halve the time axis. 30 seconds becomes ~1,500 tokens.
- **Objective**: next-token prediction of text, conditioned on the audio encoder. Multitask:
  special tokens select transcription vs translation, and the language.
- **Delta**: **encoder-decoder**, not decoder-only. Speech recognition is genuinely
  sequence-to-sequence with roughly monotonic alignment, so unlike vision it kept the 2017
  architecture. Cross-attention does the alignment that HMMs used to.

### wav2vec 2.0 (2020)
- **Tokens**: raw waveform through a convolutional feature encoder; latent representations are
  **quantized** to a learned codebook.
- **Objective**: contrastive — identify the true quantized latent for a masked span among
  distractors. Masked language modelling for audio.
- **Delta**: the quantization module. Continuous audio has no natural discrete units, so it
  learns some.

---

## Generative vision

### DiT — Diffusion Transformer (2022)
- **Tokens**: patches of a **latent** image (from a VAE encoder), not raw pixels.
- **Objective**: denoising — predict the noise added at a given timestep.
- **Delta**: replaces diffusion's U-Net with a Transformer, and conditions on timestep and class
  via **adaLN-Zero** (adaptive layer norm whose scale and shift are predicted from the
  conditioning, initialized to zero so the block starts as an identity). Scaled better than the
  U-Net, which is why Stable Diffusion 3 and Sora use this family.

### Sora (2024)
- **Tokens**: **spacetime patches** — patches spanning both spatial extent and time — cut from a
  compressed video latent.
- **Objective**: diffusion denoising over spacetime patches.
- **Delta**: because token count simply varies with duration and resolution, one model handles
  images (one frame) and video of arbitrary length and aspect ratio. The clearest demonstration
  that "patchify anything" is a general strategy.

### Stable Diffusion 3 (2024)
- **Tokens**: latent image patches plus text tokens, in **one** sequence.
- **Objective**: rectified flow matching (a straighter-path variant of diffusion).
- **Delta**: **MMDiT** — separate weights for the image and text streams, but joint attention
  across both. Modality-specific parameters with a shared attention operation.

---

## Science

### AlphaFold 2 (2021)
- **Tokens**: amino-acid residues, plus a multiple-sequence-alignment representation and a
  pairwise residue-residue representation.
- **Objective**: predict 3-D atomic coordinates.
- **Delta**: substantial. **Triangle attention** operates on the pair representation and
  enforces geometric consistency (if `i–j` and `j–k` distances are known, `i–k` is
  constrained); an equivariant structure module produces coordinates. The most heavily
  domain-specialized entry here — and a reminder that strong priors still win when the geometry
  is known.

### AlphaFold 3 (2024)
- **Delta from AF2**: replaces the structure module with a **diffusion** decoder over atom
  coordinates, and generalizes beyond proteins to ligands, nucleic acids, and complexes. Fewer
  hand-built geometric components.

### ESM-2 (2022)
- **Tokens**: amino acids, exactly as text tokens.
- **Objective**: masked language modelling over protein sequences.
- **Delta**: essentially none — a BERT trained on proteins. Structure information emerges in the
  attention maps without ever being supervised, which is a striking result about what
  next-token-style objectives extract.

### Time series — PatchTST (2022), TimesFM / Chronos (2024)
- **Tokens**: **patches** of consecutive timesteps (PatchTST), or values discretized into bins
  and treated as a vocabulary (Chronos).
- **Objective**: forecast future values; masked patch reconstruction for pretraining.
- **Delta**: patching is the key trick — it shortens sequences and gives each token local
  context, both of which matter more for time series than for text. Chronos's insight is that if
  you quantize values into a vocabulary, an unmodified language model architecture works.

---

## Decision-making and control

### Decision Transformer (2021)
- **Tokens**: an interleaved sequence of `(return-to-go, state, action)` triples.
- **Objective**: next-token prediction of actions — plain supervised learning.
- **Delta**: none architecturally, and that is the point. It reframes offline reinforcement
  learning as **sequence modelling**: condition on a desired return and the model produces
  actions that achieve it. No value function, no policy gradient, no bootstrapping.

### Gato (2022)
- **Tokens**: everything, in one vocabulary — text, image patches, button presses, joint
  torques, all serialized into a single token stream.
- **Objective**: next-token prediction across 604 tasks.
- **Delta**: none. A single decoder-only Transformer playing Atari, captioning images, and
  controlling a robot arm with the same weights.

### RT-2 / robotics VLAs (2023–)
- **Tokens**: image patches and text, plus **action tokens** — robot actions discretized into
  bins that occupy slots in the text vocabulary.
- **Objective**: next-token prediction, where some tokens happen to be motor commands.
- **Delta**: none. Because actions are just tokens, web-scale vision-language pretraining
  transfers to robot control — the model's semantic knowledge carries over to manipulation.

---

## Retrieval and recommendation

### Sentence encoders — SBERT, E5, GTE
- **Tokens**: text, as usual.
- **Objective**: contrastive — matching pairs close, non-matching far. The same InfoNCE loss as
  CLIP (notebook 20), with text on both sides.
- **Delta**: pooling over token outputs to make one vector, and normalization so dot product is
  cosine similarity. Notebook 24 builds this.

### SASRec / BERT4Rec (2018–2019)
- **Tokens**: items a user interacted with, in order. Item IDs are the vocabulary.
- **Objective**: predict the next item (SASRec, causal) or masked items (BERT4Rec).
- **Delta**: none. A user's history is a sentence and items are words.

### TIGER / generative retrieval (2023)
- **Tokens**: items represented as **semantic ID** sequences — hierarchical codes from a
  quantized content embedding, so similar items share prefixes.
- **Objective**: generate the ID of the next item.
- **Delta**: replaces the retrieve-then-rank pipeline with generation. The model *emits* an
  identifier rather than scoring candidates, which removes the index entirely.

---

## Code

### FIM — Fill in the Middle (2022)
- **Tokens**: ordinary code tokens, but documents are **reordered** during training:
  `<PRE> prefix <SUF> suffix <MID> middle`.
- **Objective**: next-token prediction on the transformed sequence.
- **Delta**: none — a data transformation, not an architecture change. It gives a causal
  decoder the ability to infill, which is what code completion in an editor actually needs
  (you have code on both sides of the cursor). A good example of a capability added purely by
  reformatting data.

---

## The pattern

Read down the "delta from an LM" column and a clear picture emerges.

**Most entries have no architectural delta at all.** Decision Transformer, Gato, RT-2, SASRec,
ESM-2, Chronos, and FIM are all ordinary Transformers. What changed is how the input was
serialized into tokens and what the target was.

**Where deltas exist, they encode a genuine structural prior.** Swin adds locality because
images are local. AlphaFold adds triangle attention because distances satisfy triangle
inequalities. Whisper keeps encoder-decoder because speech-to-text is genuinely
sequence-to-sequence. These are not arbitrary — each earns its complexity from a real property
of the domain.

**The engineering difficulty concentrates in tokenization.** Patch size, quantization codebooks,
semantic IDs, discretized actions, spectrogram framing: this is where the work is, and where
most of the failure modes live. Notebook 14 makes the same point for text.

**The consistent lesson is to add fewer priors and more data** — up to the point where you
genuinely know something about the structure, at which point encoding it still wins. ViT beat
CNNs given enough data; AlphaFold's geometric constraints have not been beaten by scale.

---

## References

**Vision** — [ViT](https://arxiv.org/abs/2010.11929) · [Swin](https://arxiv.org/abs/2103.14030) · [DETR](https://arxiv.org/abs/2005.12872) · [SAM](https://arxiv.org/abs/2304.02643) · [DINOv2](https://arxiv.org/abs/2304.07193) · [MAE](https://arxiv.org/abs/2111.06377)

**Speech** — [Whisper](https://arxiv.org/abs/2212.04356) · [wav2vec 2.0](https://arxiv.org/abs/2006.11477)

**Generative** — [DiT](https://arxiv.org/abs/2212.09748) · [Stable Diffusion 3](https://arxiv.org/abs/2403.03206) · [Sora technical report](https://openai.com/research/video-generation-models-as-world-simulators)

**Science** — [AlphaFold 2](https://www.nature.com/articles/s41586-021-03819-2) · [AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w) · [ESM-2](https://www.science.org/doi/10.1126/science.ade2574) · [PatchTST](https://arxiv.org/abs/2211.14730) · [Chronos](https://arxiv.org/abs/2403.07815)

**Decision** — [Decision Transformer](https://arxiv.org/abs/2106.01345) · [Gato](https://arxiv.org/abs/2205.06175) · [RT-2](https://arxiv.org/abs/2307.15818)

**Retrieval** — [SBERT](https://arxiv.org/abs/1908.10084) · [E5](https://arxiv.org/abs/2212.03533) · [SASRec](https://arxiv.org/abs/1808.09781) · [TIGER](https://arxiv.org/abs/2305.05065)

**Code** — [Efficient Training of LMs to Fill in the Middle](https://arxiv.org/abs/2207.14255)
