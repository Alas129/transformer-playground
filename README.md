# Transformer Learning Journey

A hands-on guide to understanding Transformers from theory to practice. Build a GPT-style text generation model from scratch!

## What You'll Learn

1. **Historical Evolution**: Why Transformers replaced RNNs
2. **Core Components**: Embeddings, Attention, Multi-Head Attention
3. **Architecture**: How pieces fit together in a Transformer block
4. **Practical Application**: Train your own text generator
5. **Training Deep Dive**: Loss & perplexity, train/val splits, LR schedules, mixed precision, pretraining vs fine-tuning
6. **Inference & Decoding**: Greedy/top-k/top-p/beam search, repetition penalty, and the KV cache
7. **Architecture Families**: Encoder-only (BERT) and encoder-decoder (seq2seq) beyond decoder-only GPT
8. **Modern LLMs**: RMSNorm, RoPE, SwiGLU, Grouped-Query Attention, FlashAttention

## Project Structure

```
transformer-playground/
├── notebooks/                    # Interactive learning (start here!)
│   ├── 01_evolution.ipynb       # History: RNNs → Transformers
│   ├── 02_embeddings.ipynb      # Token + Positional Encoding
│   ├── 03_attention.ipynb       # Self-Attention (NumPy)
│   ├── 04_multihead_attention.ipynb  # Multi-Head Attention
│   ├── 05_transformer_block.ipynb    # Complete block assembly
│   ├── 06_full_transformer.ipynb     # Full architecture
│   ├── 07_text_generation.ipynb      # Train & generate text!
│   ├── 08_training.ipynb             # Training deep dive (loss, schedules, fine-tuning)
│   ├── 09_inference.ipynb            # Decoding strategies + KV cache
│   ├── 10_encoder_and_seq2seq.ipynb  # BERT (encoder) & encoder-decoder
│   └── 11_modern_architectures.ipynb # RMSNorm, RoPE, SwiGLU, GQA, FlashAttention
├── src/                         # PyTorch implementation
│   ├── embeddings.py            # Embedding layers
│   ├── attention.py             # Attention mechanisms
│   ├── transformer.py           # Transformer blocks
│   ├── gpt.py                   # GPT model
│   └── train.py                 # Training utilities
└── data/                        # Training data
    └── sample_text.txt          # Shakespeare sample
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Learning

Open the notebooks in order:

```bash
jupyter notebook notebooks/
```

Or use VS Code / Cursor with the Jupyter extension.

### 3. Train Your Model

After going through the notebooks, train your own text generator:

```python
from src.train import train_gpt, generate_text

# train_gpt returns the trained model AND the fitted tokenizer
model, tokenizer = train_gpt('data/sample_text.txt', epochs=50)

# generate_text handles encoding the prompt and decoding the output
print(generate_text(model, tokenizer, "To be or not to be", max_tokens=100))
```

Or run it straight from the command line:

```bash
python -m src.train data/sample_text.txt
```

## Learning Path

| Notebook | Topic | Time | Key Concept |
|----------|-------|------|-------------|
| 01 | Evolution | 30 min | Why Transformers exist |
| 02 | Embeddings | 30 min | Converting text to numbers |
| 03 | Attention | 45 min | The core innovation |
| 04 | Multi-Head | 30 min | Parallel attention |
| 05 | Blocks | 30 min | Assembling components |
| 06 | Architecture | 30 min | Full picture |
| 07 | Generation | 60 min | Hands-on training |
| 08 | Training | 60 min | Loss, perplexity, LR schedules, fine-tuning |
| 09 | Inference | 60 min | Decoding strategies & KV cache |
| 10 | Encoders & Seq2Seq | 45 min | BERT & encoder-decoder |
| 11 | Modern Architectures | 45 min | RMSNorm, RoPE, SwiGLU, GQA, FlashAttention |

**Total: ~7-8 hours for thorough understanding**

Notebooks 01–07 build a GPT from scratch. Notebooks 08–11 round out a complete
picture: how these models are *trained*, how they *generate* at inference time,
the other architecture families (BERT, seq2seq), and the techniques behind
today's LLMs.

## The Key Insight

Traditional sequence models (RNNs) process tokens one at a time:

```
Token1 → Token2 → Token3 → ... → TokenN
         (slow, forgets early tokens)
```

Transformers let every token look at every other token simultaneously:

```
Token1 ←→ Token2 ←→ Token3 ←→ ... ←→ TokenN
              (fast, long-range memory)
```

This is achieved through **Self-Attention** - the mechanism you'll implement from scratch!

## Requirements

- Python 3.8+
- NumPy (for understanding fundamentals)
- PyTorch (for practical implementation)
- Matplotlib (for visualizations)
- Jupyter (for notebooks)

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The original Transformer paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual guide
- [GPT Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) - Decoder-only architecture
- [BERT](https://arxiv.org/abs/1810.04805) - Encoder-only, masked language modeling
- [RoFormer / RoPE](https://arxiv.org/abs/2104.09864) - Rotary position embeddings
- [GQA](https://arxiv.org/abs/2305.13245) - Grouped-query attention
- [FlashAttention](https://arxiv.org/abs/2205.14135) - IO-aware exact attention
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Minimal, readable GPT training repo

Happy learning! 🚀

