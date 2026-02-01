# Transformer Learning Journey

A hands-on guide to understanding Transformers from theory to practice. Build a GPT-style text generation model from scratch!

## What You'll Learn

1. **Historical Evolution**: Why Transformers replaced RNNs
2. **Core Components**: Embeddings, Attention, Multi-Head Attention
3. **Architecture**: How pieces fit together in a Transformer block
4. **Practical Application**: Train your own text generator

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
│   └── 07_text_generation.ipynb      # Train & generate text!
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
from src.train import train_gpt
from src.gpt import GPT

# Train on sample data
model = train_gpt('data/sample_text.txt', epochs=100)

# Generate text
print(model.generate("To be or not to be", max_tokens=100))
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

**Total: ~4-5 hours for thorough understanding**

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

Happy learning! 🚀

