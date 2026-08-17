"""
GPT Model - A decoder-only transformer for text generation.

This is a simplified GPT implementation suitable for learning and experimentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import TransformerEmbedding
from .transformer import TransformerDecoder


class GPT(nn.Module):
    """
    GPT (Generative Pre-trained Transformer) model.
    
    A decoder-only transformer for autoregressive text generation.
    """
    
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=6,
                 max_seq_len=256, d_ff=None, dropout=0.1):
        """
        Args:
            vocab_size: Size of the vocabulary
            d_model: Model dimension (default: 256)
            num_heads: Number of attention heads (default: 8)
            num_layers: Number of decoder layers (default: 6)
            max_seq_len: Maximum sequence length (default: 256)
            d_ff: Feed-forward hidden dimension (default: 4 * d_model)
            dropout: Dropout rate (default: 0.1)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout

        # Embedding layer (token + position)
        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            embed_dim=d_model,
            max_seq_len=max_seq_len,
            dropout=dropout,
            learnable_pos=True  # GPT uses learned position embeddings
        )
        
        # Transformer decoder
        self.decoder = TransformerDecoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Language model head (project to vocabulary)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying: share weights between embedding and lm_head
        # This is a common technique that improves performance
        self.lm_head.weight = self.embedding.token_embedding.embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        self._scale_residual_projections(num_layers)

    def _init_weights(self, module):
        """Initialize weights using small random values."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def _scale_residual_projections(self, num_layers):
        """
        Shrink the projections that write into the residual stream.

        Each block adds two contributions to the stream, and with N blocks all
        initialized at the same scale those contributions accumulate: the
        stream's variance grows roughly linearly in depth, so a deep stack
        starts out with much larger activations than a shallow one. Dividing
        the two output projections by sqrt(2 * num_layers) cancels that growth
        at init -- the GPT-2 trick.

        This cannot be done inside _init_weights, because nn.Module.apply sees
        a bare nn.Linear and cannot tell which of them feeds the residual add.
        """
        scale = (2 * num_layers) ** -0.5
        for name, param in self.named_parameters():
            if name.endswith("attention.W_o.weight") or name.endswith("linear2.weight"):
                with torch.no_grad():
                    param.mul_(scale)
    
    def forward(self, input_ids, targets=None):
        """
        Forward pass.

        Args:
            input_ids: Token indices (batch_size, seq_len)
            targets: Target token indices for computing loss (batch_size, seq_len)

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss (if targets provided)

        For attention maps, use attention_maps() -- this path deliberately does
        not build them.
        """
        seq_len = input_ids.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len "
                f"{self.max_seq_len}. This model learns an absolute position "
                f"embedding, so there is no position {self.max_seq_len} to look "
                f"up. Crop the input, or build the model with a larger "
                f"max_seq_len."
            )

        # Get embeddings
        x = self.embedding(input_ids)

        # Pass through decoder
        x, _ = self.decoder(x)

        # Project to vocabulary
        logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Reshape for cross entropy: (batch * seq_len, vocab_size)
            #
            # ignore_index=-1 skips positions labelled -1. Nothing in
            # TextDataset produces those; it is how notebook 12 masks prompt
            # tokens so the loss is computed on the response only. Note that
            # PyTorch's own convention is -100 -- this repo uses -1 throughout,
            # including in docs/glossary.md.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def attention_maps(self, input_ids):
        """
        Per-layer attention weights, for visualization.

        Separate from forward() so the training and inference paths never build
        these. Each is (batch, heads, seq_len, seq_len), so a stack of them is
        the largest thing in the model for any interesting sequence length.

        Args:
            input_ids: Token indices (batch_size, seq_len)

        Returns:
            List of (batch, heads, seq_len, seq_len), one per layer
        """
        x = self.embedding(input_ids)
        _, attention_weights = self.decoder(x, return_attention=True)
        return attention_weights

    def forward_cached(self, input_ids, past_kvs=None):
        """
        Forward pass reusing a KV cache (notebook 09).

        Args:
            input_ids: Token indices (batch_size, seq_len). Only the *new*
                tokens; seq_len is 1 after the first call.
            past_kvs: Optional list of per-layer (k, v) from a previous call

        Returns:
            logits: (batch_size, seq_len, vocab_size)
            presents: List of per-layer (k, v) for the next call
        """
        past_len = 0 if past_kvs is None else past_kvs[0][0].size(2)

        total = past_len + input_ids.size(1)
        assert total <= self.max_seq_len, (
            f"cached sequence length {total} exceeds max_seq_len "
            f"{self.max_seq_len}. This model uses a learned absolute position "
            f"embedding, so there is no position {self.max_seq_len} to look up. "
            f"Use generate(), which falls back to recomputation here."
        )

        # Positions are absolute here, so the embedding needs the offset. This
        # is exactly what makes the cache fragile for a model with a learned
        # absolute position table -- see the note in generate().
        x = self.embedding(input_ids, offset=past_len)

        x, presents = self.decoder.forward_cached(x, past_kvs)
        logits = self.lm_head(x)

        return logits, presents

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None,
                 use_cache=True):
        """
        Generate text autoregressively.

        Args:
            input_ids: Starting token indices (batch_size, seq_len)
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k most likely tokens
            use_cache: Reuse cached keys and values instead of recomputing the
                whole prefix each step. Numerically identical (verified in
                tests/test_kv_cache.py) and much faster. Set False to get the
                naive path for comparison or benchmarking.

        Returns:
            Generated token indices (batch_size, seq_len + max_new_tokens)
        """
        # Sampling must run with dropout off, but this method is also called
        # *during* training to print a sample (see train_gpt). Restore whatever
        # mode we found, or training silently continues without dropout.
        was_training = self.training
        self.eval()
        try:
            return self._generate(
                input_ids, max_new_tokens, temperature, top_k, use_cache
            )
        finally:
            self.train(was_training)

    def _generate(self, input_ids, max_new_tokens, temperature, top_k, use_cache):
        """The sampling loop itself. Assumes the caller has set eval mode."""
        past_kvs = None
        cache_usable = use_cache

        for _ in range(max_new_tokens):
            # This model uses a *learned absolute* position embedding, so once
            # the sequence outgrows max_seq_len the naive path crops it and
            # every remaining token is re-indexed from 0. A cache holds keys
            # built at the old positions, so it cannot survive that shift --
            # drop it and fall back to recomputing.
            #
            # Models with relative position (RoPE) have no such problem, which
            # is one more reason the field moved to it. Compare
            # ModernGPT.generate in modern.py, which caches all the way.
            if cache_usable and input_ids.size(1) > self.max_seq_len:
                cache_usable = False
                past_kvs = None

            if cache_usable:
                # First pass sees the whole prompt; later passes just one token.
                model_input = input_ids if past_kvs is None else input_ids[:, -1:]
                logits, past_kvs = self.forward_cached(model_input, past_kvs)
            else:
                logits, _ = self(input_ids[:, -self.max_seq_len:])

            # Get logits for the last position
            logits = logits[:, -1, :] / temperature

            # Optional: top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float('-inf'))

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
    
    @property
    def config(self):
        """
        Every argument needed to rebuild this model.

        Saved alongside the weights so a checkpoint is self-describing. Without
        it, loading means knowing out of band which factory produced the file,
        and guessing wrong surfaces as a wall of shape mismatches.
        """
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "max_seq_len": self.max_seq_len,
            "d_ff": self.d_ff,
            "dropout": self.dropout,
        }

    @classmethod
    def from_config(cls, config):
        """Rebuild an (untrained) model from a config dict."""
        return cls(**config)

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_gpt_small(vocab_size, max_seq_len=256):
    """Create a small GPT model suitable for CPU training."""
    return GPT(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=4,
        max_seq_len=max_seq_len,
        dropout=0.1
    )


def create_gpt_medium(vocab_size, max_seq_len=256):
    """Create a medium GPT model."""
    return GPT(
        vocab_size=vocab_size,
        d_model=256,
        num_heads=8,
        num_layers=6,
        max_seq_len=max_seq_len,
        dropout=0.1
    )

