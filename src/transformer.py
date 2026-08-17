"""
Transformer building blocks.

This module provides:
- FeedForward: Position-wise feed-forward network
- TransformerBlock: Single transformer block
- TransformerEncoder: Stack of encoder blocks
- TransformerDecoder: Stack of decoder blocks
"""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention, CausalSelfAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    FFN(x) = GELU(xW1 + b1)W2 + b2

    The 2017 paper used ReLU -- max(0, xW1 + b1)W2 + b2 -- but GPT-2 onward
    switched to GELU, which this implements. Modern LLMs go one step further to
    a gated variant; see SwiGLU in modern.py and notebook 11.
    """
    
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            d_ff: Hidden dimension (default: 4 * d_model)
            dropout: Dropout rate
        """
        super().__init__()
        
        d_ff = d_ff or 4 * d_model
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    A single Transformer block (encoder-style).
    
    Uses Pre-LN (layer norm before sublayers) for better training stability.
    """
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        # Pre-LN: Norm before sublayer
        normed = self.norm1(x)
        attn_output, attention_weights = self.attention(normed, normed, normed, mask)
        x = x + self.dropout1(attn_output)
        
        # Feed-forward with Pre-LN
        normed = self.norm2(x)
        ffn_output = self.ffn(normed)
        x = x + self.dropout2(ffn_output)
        
        return x, attention_weights


class DecoderBlock(nn.Module):
    """
    Decoder block with causal self-attention (GPT-style).
    """
    
    def __init__(self, d_model, num_heads, max_seq_len, d_ff=None, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.attention = CausalSelfAttention(d_model, num_heads, max_seq_len, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        # Pre-LN style
        normed = self.norm1(x)
        attn_output, attention_weights = self.attention(normed)
        x = x + self.dropout1(attn_output)
        
        normed = self.norm2(x)
        ffn_output = self.ffn(normed)
        x = x + self.dropout2(ffn_output)

        return x, attention_weights

    def forward_cached(self, x, past_kv=None):
        """
        Same as forward(), reusing cached keys and values.

        Only attention needs the cache. The feed-forward network is applied
        per position independently, so it has nothing to remember.

        Args:
            x: (batch, seq_len, d_model) -- the new tokens only
            past_kv: Optional (past_k, past_v) for this block

        Returns:
            output: (batch, seq_len, d_model)
            present: (k, v) for the next call
        """
        normed = self.norm1(x)
        attn_output, present = self.attention.forward_cached(normed, past_kv)
        x = x + self.dropout1(attn_output)

        normed = self.norm2(x)
        ffn_output = self.ffn(normed)
        x = x + self.dropout2(ffn_output)

        return x, present


class TransformerEncoder(nn.Module):
    """
    Stack of Transformer encoder blocks.
    """
    
    def __init__(self, d_model, num_heads, num_layers, d_ff=None, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer blocks
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None, return_attention=False):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask
            return_attention: Collect every layer's attention matrix. Off by
                default -- see TransformerDecoder.forward for why.

        Returns:
            output: (batch, seq_len, d_model)
            all_attention_weights: List of per-layer attention weights, or None
        """
        all_attention_weights = [] if return_attention else None

        for layer in self.layers:
            x, layer_weights = layer(x, mask)
            if return_attention:
                all_attention_weights.append(layer_weights)
            # Drop the reference now. Without this the name still points at the
            # previous layer's matrix for the whole of the next layer's forward
            # pass, because `x, layer_weights = layer(x)` rebinds only after
            # layer(x) returns -- so two would be alive instead of one.
            del layer_weights

        x = self.norm(x)

        return x, all_attention_weights


class TransformerDecoder(nn.Module):
    """
    Stack of Transformer decoder blocks (GPT-style, decoder-only).
    """
    
    def __init__(self, d_model, num_heads, num_layers, max_seq_len, 
                 d_ff=None, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of decoder blocks
            max_seq_len: Maximum sequence length
            d_ff: Feed-forward hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, max_seq_len, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch, seq_len, d_model)
            return_attention: Collect every layer's attention matrix. Off by
                default: each is (batch, heads, seq_len, seq_len), and holding
                all of them at once is only useful for visualization. Under
                autograd they are saved for backward regardless, so this costs
                nothing during training -- but under no_grad the list is the
                only thing keeping earlier layers' matrices alive, which made an
                inference forward pass hold L of them where one would do.

        Returns:
            output: (batch, seq_len, d_model)
            all_attention_weights: List of per-layer attention weights, or None
        """
        all_attention_weights = [] if return_attention else None

        for layer in self.layers:
            x, layer_weights = layer(x)
            if return_attention:
                all_attention_weights.append(layer_weights)
            # See TransformerEncoder.forward: rebinding happens after the call
            # returns, so without this the previous layer's matrix stays alive
            # through the next layer's forward pass.
            del layer_weights

        x = self.norm(x)

        return x, all_attention_weights

    def forward_cached(self, x, past_kvs=None):
        """
        Same as forward(), reusing a per-layer KV cache.

        Args:
            x: (batch, seq_len, d_model) -- the new tokens only
            past_kvs: Optional list of per-layer (k, v), one entry per layer

        Returns:
            output: (batch, seq_len, d_model)
            presents: List of per-layer (k, v) for the next call
        """
        presents = []

        for i, layer in enumerate(self.layers):
            past_kv = None if past_kvs is None else past_kvs[i]
            x, present = layer.forward_cached(x, past_kv)
            presents.append(present)

        x = self.norm(x)

        return x, presents

