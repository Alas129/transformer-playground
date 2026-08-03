# Transformer Learning Journey - PyTorch Implementation
# 
# This package contains clean, modular implementations of Transformer components.
# Use alongside the notebooks for hands-on learning.

from .embeddings import TokenEmbedding, PositionalEncoding, TransformerEmbedding
from .attention import ScaledDotProductAttention, MultiHeadAttention, CausalSelfAttention
from .transformer import FeedForward, TransformerBlock, TransformerEncoder, TransformerDecoder
from .gpt import GPT, create_gpt_small, create_gpt_medium

# Modern (LLaMA-era) stack -- notebook 11 onward
from .modern import (
    RMSNorm,
    RotaryEmbedding,
    apply_rope,
    SwiGLU,
    GroupedQueryAttention,
    ModernBlock,
    ModernGPT,
    create_modern_small,
)

# Mixture-of-Experts -- notebook 17
from .moe import Router, MoEFeedForward, moe_aux_losses, moe_load_report

__all__ = [
    # 2017 stack
    'TokenEmbedding',
    'PositionalEncoding',
    'TransformerEmbedding',
    'ScaledDotProductAttention',
    'MultiHeadAttention',
    'CausalSelfAttention',
    'FeedForward',
    'TransformerBlock',
    'TransformerEncoder',
    'TransformerDecoder',
    'GPT',
    'create_gpt_small',
    'create_gpt_medium',
    # Modern stack
    'RMSNorm',
    'RotaryEmbedding',
    'apply_rope',
    'SwiGLU',
    'GroupedQueryAttention',
    'ModernBlock',
    'ModernGPT',
    'create_modern_small',
    # Mixture-of-Experts
    'Router',
    'MoEFeedForward',
    'moe_aux_losses',
    'moe_load_report',
]

