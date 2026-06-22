"""Generate the diagram PNGs used in the docs.

Every figure here plots *real math* (exact formulas / deterministic computation),
not fabricated training data, so the images stay faithful to what the notebooks teach.

Run from the repo root:
    python3 docs/images/generate.py
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def positional_encoding(seq_len=40, d_model=64):
    """Sinusoidal positional encoding (Vaswani et al., 2017, §3.5)."""
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
    pe = np.where(i % 2 == 0, np.sin(angle), np.cos(angle))

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pe, aspect="auto", cmap="RdBu")
    ax.set_xlabel("embedding dimension")
    ax.set_ylabel("position in sequence")
    ax.set_title("Sinusoidal Positional Encoding (NB 02)")
    fig.colorbar(im, ax=ax, label="value")
    save(fig, "positional_encoding.png")


def attention_weights():
    """Real causal self-attention weights for a toy 6-token sentence.

    We build small fixed Q/K so the softmax(QKᵀ/√d) pattern is reproducible,
    then apply the causal mask exactly as a decoder does.
    """
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    n = len(tokens)
    d = 8
    rng = np.random.default_rng(0)
    q = rng.standard_normal((n, d))
    k = rng.standard_normal((n, d))

    scores = q @ k.T / np.sqrt(d)
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)  # future positions
    scores = np.where(mask, -np.inf, scores)
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights = weights / weights.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(weights, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n), tokens)
    ax.set_yticks(range(n), tokens)
    ax.set_xlabel("key (attended-to token)")
    ax.set_ylabel("query (current token)")
    ax.set_title("Causal Self-Attention Weights (NB 03)\nlower-triangular: can't see the future")
    for r in range(n):
        for c in range(n):
            if not mask[r, c]:
                ax.text(c, r, f"{weights[r, c]:.2f}", ha="center", va="center",
                        color="white" if weights[r, c] < 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="attention weight")
    save(fig, "attention_weights.png")


def lr_schedule(total=1000, warmup=100, base_lr=3e-4, min_lr=3e-5):
    """Linear warmup + cosine decay — the exact schedule used to train GPTs (NB 08)."""
    steps = np.arange(total)
    lr = np.empty(total)
    for s in steps:
        if s < warmup:
            lr[s] = base_lr * (s + 1) / warmup
        else:
            prog = (s - warmup) / (total - warmup)
            lr[s] = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * prog))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, lr, color="#d6336c", lw=2)
    ax.axvline(warmup, ls="--", color="gray", alpha=0.7)
    ax.text(warmup + 10, base_lr * 0.5, "warmup ends", color="gray")
    ax.set_xlabel("training step")
    ax.set_ylabel("learning rate")
    ax.set_title("Linear Warmup + Cosine Decay LR Schedule (NB 08)")
    ax.grid(alpha=0.3)
    save(fig, "lr_schedule.png")


if __name__ == "__main__":
    positional_encoding()
    attention_weights()
    lr_schedule()
    print("done")
