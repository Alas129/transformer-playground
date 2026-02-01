"""
Attention mechanisms for Transformer models.

This module provides:
- ScaledDotProductAttention: Core attention operation
- MultiHeadAttention: Parallel attention heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention.
    
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, num_heads, seq_len, d_k)
            key: (batch, num_heads, seq_len, d_k)
            value: (batch, num_heads, seq_len, d_v)
            mask: Optional mask (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len)
        
        Returns:
            output: (batch, num_heads, seq_len, d_v)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        d_k = query.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention.
    
    Allows the model to jointly attend to information from different
    representation subspaces at different positions.
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.attention = ScaledDotProductAttention(dropout)
    
    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, d_k).
        
        (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        """
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def combine_heads(self, x, batch_size):
        """
        Reverse of split_heads.
        
        (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        """
        x = x.transpose(1, 2)
        return x.contiguous().view(batch_size, -1, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, seq_len, d_model)
            key: (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split into multiple heads
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        
        # Apply attention
        attn_output, attention_weights = self.attention(Q, K, V, mask)
        
        # Combine heads
        output = self.combine_heads(attn_output, batch_size)
        
        # Final linear projection
        output = self.W_o(output)
        
        return output, attention_weights


class CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention for decoder (GPT-style).
    
    Automatically applies causal mask to prevent attending to future tokens.
    """
    
    def __init__(self, d_model, num_heads, max_seq_len, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Create causal mask (lower triangular)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer('mask', mask.view(1, 1, max_seq_len, max_seq_len))
    
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        seq_len = x.size(1)
        mask = self.mask[:, :, :seq_len, :seq_len]
        return self.attention(x, x, x, mask)

