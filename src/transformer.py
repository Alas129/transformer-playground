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
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    In modern transformers, GELU is often used instead of ReLU.
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
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            output: (batch, seq_len, d_model)
            all_attention_weights: List of attention weights from each layer
        """
        all_attention_weights = []
        
        for layer in self.layers:
            x, attention_weights = layer(x, mask)
            all_attention_weights.append(attention_weights)
        
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
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
            all_attention_weights: List of attention weights from each layer
        """
        all_attention_weights = []
        
        for layer in self.layers:
            x, attention_weights = layer(x)
            all_attention_weights.append(attention_weights)
        
        x = self.norm(x)
        
        return x, all_attention_weights

