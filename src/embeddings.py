"""
Embedding layers for Transformer models.

This module provides:
- TokenEmbedding: Learnable word/token embeddings
- PositionalEncoding: Sinusoidal position encoding
- TransformerEmbedding: Combined token + position embeddings
"""

import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    Learnable token embeddings.
    
    Maps token indices to dense vectors.
    """
    
    def __init__(self, vocab_size, embed_dim):
        """
        Args:
            vocab_size: Number of unique tokens
            embed_dim: Dimension of embedding vectors
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim
    
    def forward(self, x):
        """
        Args:
            x: Token indices (batch_size, seq_len)
        Returns:
            Embeddings (batch_size, seq_len, embed_dim)
        """
        # Scale embeddings by sqrt(d_model) as in original paper
        return self.embedding(x) * math.sqrt(self.embed_dim)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (fixed, not learned).
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, embed_dim, max_seq_len=5000, dropout=0.1):
        """
        Args:
            embed_dim: Dimension of embeddings
            max_seq_len: Maximum sequence length to support
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the division term
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        
        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)  # (1, max_seq_len, embed_dim)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Embeddings (batch_size, seq_len, embed_dim)
        Returns:
            Embeddings + positional encoding
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional embeddings (like GPT uses).
    """
    
    def __init__(self, embed_dim, max_seq_len=5000, dropout=0.1):
        """
        Args:
            embed_dim: Dimension of embeddings
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
    
    def forward(self, x):
        """
        Args:
            x: Embeddings (batch_size, seq_len, embed_dim)
        Returns:
            Embeddings + positional encoding
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.position_embedding(positions)
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    """
    Complete embedding layer combining token and position embeddings.
    """
    
    def __init__(self, vocab_size, embed_dim, max_seq_len=5000, 
                 dropout=0.1, learnable_pos=True):
        """
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            learnable_pos: Use learnable (True) or sinusoidal (False) positions
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        
        if learnable_pos:
            self.position_encoding = LearnablePositionalEncoding(
                embed_dim, max_seq_len, dropout
            )
        else:
            self.position_encoding = PositionalEncoding(
                embed_dim, max_seq_len, dropout
            )
    
    def forward(self, x):
        """
        Args:
            x: Token indices (batch_size, seq_len)
        Returns:
            Embeddings (batch_size, seq_len, embed_dim)
        """
        return self.position_encoding(self.token_embedding(x))

