# Transformer Learning Journey - PyTorch Implementation
#
# This package contains clean, modular implementations of Transformer components.
# Use alongside the notebooks for hands-on learning.
#
# Every module in src/ surfaces its public names here, so `from src import X`
# works for anything the notebooks use. tests/test_public_api.py enforces that.

# 2017 stack -- notebooks 02-10
from .embeddings import (
    TokenEmbedding,
    PositionalEncoding,
    LearnablePositionalEncoding,
    TransformerEmbedding,
)
from .attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    CausalSelfAttention,
)
from .transformer import (
    FeedForward,
    TransformerBlock,
    DecoderBlock,
    TransformerEncoder,
    TransformerDecoder,
)
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

# Low-rank adaptation -- notebooks 12 and 26
from .lora import (
    LoRALinear,
    MultiAdapterLoRALinear,
    apply_lora,
    merge_lora,
    unmerge_lora,
    lora_parameters,
    lora_summary,
)

# Training and generation utilities -- notebooks 07-09
from .train import (
    CharTokenizer,
    TextDataset,
    split_text,
    configure_optimizer,
    lr_lambda_for,
    evaluate,
    train_gpt,
    generate_text,
    checkpoint_payload,
    load_gpt,
    tokenizer_path_for,
)

__all__ = [
    # 2017 stack
    'TokenEmbedding',
    'PositionalEncoding',
    'LearnablePositionalEncoding',
    'TransformerEmbedding',
    'ScaledDotProductAttention',
    'MultiHeadAttention',
    'CausalSelfAttention',
    'FeedForward',
    'TransformerBlock',
    'DecoderBlock',
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
    # LoRA
    'LoRALinear',
    'MultiAdapterLoRALinear',
    'apply_lora',
    'merge_lora',
    'unmerge_lora',
    'lora_parameters',
    'lora_summary',
    # Training utilities
    'CharTokenizer',
    'TextDataset',
    'split_text',
    'configure_optimizer',
    'lr_lambda_for',
    'evaluate',
    'train_gpt',
    'generate_text',
    'checkpoint_payload',
    'load_gpt',
    'tokenizer_path_for',
]
