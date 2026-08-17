"""
Training utilities for GPT model.

Provides:
- CharTokenizer: Simple character-level tokenizer
- TextDataset: Dataset for text data
- train_gpt: Main training function
"""

import math
import os
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .gpt import GPT, create_gpt_small


class CharTokenizer:
    """
    Simple character-level tokenizer.
    
    Maps each unique character to an integer index.
    """
    
    def __init__(self, text=None):
        """
        Args:
            text: Text to build vocabulary from
        """
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
        if text is not None:
            self.fit(text)
    
    def fit(self, text):
        """Build vocabulary from text."""
        chars = sorted(set(text))
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)
    
    def encode(self, text):
        """
        Convert text to list of integers.

        Characters outside the vocabulary are dropped -- a character tokenizer
        fitted on one corpus has no id to give them. Dropping them silently
        would make the returned sequence shorter than the text the caller
        passed, so warn instead.
        """
        unknown = sorted({c for c in text if c not in self.char_to_idx})
        if unknown:
            warnings.warn(
                f"{len(unknown)} character(s) not in the vocabulary were "
                f"dropped: {''.join(unknown)!r}",
                UserWarning,
                stacklevel=2,
            )
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]
    
    def decode(self, indices):
        """Convert list of integers to text."""
        return ''.join(self.idx_to_char.get(i, '?') for i in indices)
    
    def save(self, path):
        """Save tokenizer to file."""
        import json
        with open(path, 'w') as f:
            json.dump({
                'char_to_idx': self.char_to_idx,
                'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()}
            }, f)
    
    @classmethod
    def load(cls, path):
        """Load tokenizer from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls()
        tokenizer.char_to_idx = data['char_to_idx']
        tokenizer.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
        tokenizer.vocab_size = len(tokenizer.char_to_idx)
        return tokenizer


def tokenizer_path_for(save_path):
    """
    Where to write the tokenizer that belongs to a checkpoint.

    Derived from the checkpoint's stem, so it never collides with the
    checkpoint itself. A plain str.replace('.pt', ...) is not safe here: for a
    path like 'model.bin' it matches nothing and returns the checkpoint path,
    and the tokenizer JSON then overwrites the model that was just saved.

    Args:
        save_path: Path the model checkpoint was saved to

    Returns:
        Path (str) for the tokenizer JSON, alongside the checkpoint
    """
    path = Path(save_path)
    return str(path.with_name(f"{path.stem}_tokenizer.json"))


class TextDataset(Dataset):
    """
    Dataset for autoregressive language modeling.
    
    Each sample is a sequence of tokens, where the target is the same
    sequence shifted by one position.
    """
    
    def __init__(self, text, tokenizer, seq_len, stride=None):
        """
        Args:
            text: Raw text data
            tokenizer: Tokenizer to encode text
            seq_len: Sequence length for training
            stride: Distance between the starts of consecutive windows.
                Defaults to seq_len, which tiles the corpus without overlap.

                A stride of 1 -- the old behaviour -- makes consecutive samples
                share all but one token, so an epoch covers the corpus seq_len
                times over while seeing almost no new information. On this
                repo's 11.5k-character sample that is 1.46M tokens per "epoch"
                instead of 11.5k. Small strides do act as data augmentation for
                a tiny corpus, so it stays available; it is just no longer the
                silent default.
        """
        self.seq_len = seq_len
        self.stride = seq_len if stride is None else stride
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    def __len__(self):
        # A sample needs seq_len inputs plus one more token, because the target
        # is the input shifted left by one.
        max_start = len(self.data) - self.seq_len - 1
        if max_start < 0:
            return 0
        return max_start // self.stride + 1

    def __getitem__(self, idx):
        """
        Returns:
            x: Input sequence (seq_len,)
            y: Target sequence (seq_len,) - shifted by 1
        """
        start = idx * self.stride
        x = self.data[start:start + self.seq_len]
        y = self.data[start + 1:start + self.seq_len + 1]
        return x, y


def split_text(text, val_fraction=0.1):
    """
    Split a corpus into training and validation text.

    The split is *contiguous* -- validation is the tail of the corpus, not a
    random sample of windows. With a random split, a validation window sits
    beside a training window that overlaps it almost entirely, so validation
    loss tracks training loss and overfitting becomes invisible. That is the
    one thing a held-out set exists to show.

    Args:
        text: Raw corpus
        val_fraction: Fraction of the corpus to hold out, in [0, 1)

    Returns:
        (train_text, val_text)
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")

    split = int(len(text) * (1.0 - val_fraction))
    return text[:split], text[split:]


def checkpoint_payload(model, tokenizer, seq_len):
    """
    Assemble a self-describing checkpoint.

    Includes the model's full config, so load_gpt can rebuild the architecture
    without being told which factory made it.

    Args:
        model: Trained GPT
        tokenizer: Fitted tokenizer
        seq_len: Sequence length used for training

    Returns:
        dict suitable for torch.save
    """
    return {
        "model_state_dict": model.state_dict(),
        "config": model.config,
        "seq_len": seq_len,
        "vocab_size": tokenizer.vocab_size,
    }


def load_gpt(path, device=None):
    """
    Load a model and its tokenizer from a checkpoint written by train_gpt.

    Args:
        path: Path to the checkpoint (.pt)
        device: Device to place the model on (auto-detected if None)

    Returns:
        (model, tokenizer)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if "config" not in checkpoint:
        raise ValueError(
            f"{path} has no 'config' entry, so the architecture is unknown. "
            f"It predates checkpoint_payload(); rebuild the model with the "
            f"factory that trained it and load 'model_state_dict' by hand."
        )

    model = GPT.from_config(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    tokenizer = CharTokenizer.load(tokenizer_path_for(str(path)))

    return model, tokenizer


def configure_optimizer(model, lr, weight_decay=0.1, betas=(0.9, 0.95)):
    """
    Build AdamW with weight decay on matrices only.

    Weight decay expresses a prior that a weight should be small. For a
    projection matrix that is a sensible regularizer. For a normalization gain
    or a bias it is not -- those parameters set the scale and offset the layer
    needs, and decaying them just pulls against the layer's own job.
    AdamW(model.parameters()) decays every parameter indiscriminately.

    The split is by dimension, which is the standard heuristic: 2-D and higher
    are matrices, 1-D are gains and biases.

    Args:
        model: Model whose parameters to optimize
        lr: Peak learning rate
        weight_decay: Decay applied to the matrix group
        betas: AdamW betas. 0.95 for beta2 is the usual choice for Transformers

    Returns:
        A configured torch.optim.AdamW
    """
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
    )


def lr_lambda_for(warmup_steps, total_steps, min_ratio=0.1):
    """
    Linear warmup then cosine decay, as a multiplier on the peak learning rate.

    Warmup exists because Adam's second-moment estimate is meaningless for the
    first few steps: it has seen too little gradient history, so early updates
    are effectively unscaled and can move the weights far enough to spoil the
    run. Ramping the learning rate up gives the estimate time to settle.
    Cosine decay then anneals toward the end so the final steps refine rather
    than bounce.

    Args:
        warmup_steps: Steps spent ramping from ~0 to the peak
        total_steps: Total training steps; decay reaches min_ratio here
        min_ratio: Floor as a fraction of the peak learning rate

    Returns:
        f(step) -> multiplier, suitable for torch.optim.lr_scheduler.LambdaLR
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / (warmup_steps + 1)

        span = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / span)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return lr_lambda


@torch.no_grad()
def evaluate(model, dataloader, device):
    """
    Mean loss over a dataloader.

    Args:
        model: Model to evaluate
        dataloader: Batches of (x, y)
        device: Device to run on

    Returns:
        Mean loss, or float('nan') for an empty dataloader
    """
    was_training = model.training
    model.eval()
    try:
        total, batches = 0.0, 0
        for x, y in dataloader:
            _, loss = model(x.to(device), y.to(device))
            total += loss.item()
            batches += 1
        return total / batches if batches else float('nan')
    finally:
        model.train(was_training)


def train_gpt(text_path, epochs=100, batch_size=32, seq_len=128, lr=3e-4,
              device=None, save_path=None, print_every=10,
              val_fraction=0.1, weight_decay=0.1, warmup_ratio=0.02,
              stride=None, seed=0, sample_prompt=None):
    """
    Train a GPT model on text data.

    Args:
        text_path: Path to text file
        epochs: Number of training epochs
        batch_size: Batch size
        seq_len: Sequence length
        lr: Peak learning rate
        device: Device to train on (auto-detected if None)
        save_path: Path to save trained model
        print_every: Generate a sample every N *epochs*
        val_fraction: Fraction of the corpus held out for validation. 0 disables
            validation, but then overfitting is invisible
        weight_decay: Weight decay, applied to matrices only
        warmup_ratio: Fraction of total steps spent warming the learning rate up
        stride: Window stride for the dataset (defaults to seq_len)
        seed: RNG seed, so a run is reproducible
        sample_prompt: Prompt for the periodic sample. Defaults to the start of
            the corpus, which is guaranteed to be in the vocabulary -- a
            hardcoded prompt warns about dropped characters on any corpus that
            happens not to contain them

    Returns:
        model: Trained GPT model
        tokenizer: Fitted tokenizer
        history: dict with 'train_loss' and 'val_loss' per epoch
    """
    torch.manual_seed(seed)

    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")

    # Load and prepare data
    print(f"Loading data from: {text_path}")
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Text length: {len(text):,} characters")

    # Fit the tokenizer on the whole corpus, so held-out text contains no
    # characters the model has never been given an id for.
    tokenizer = CharTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    train_text, val_text = split_text(text, val_fraction)

    train_dataset = TextDataset(train_text, tokenizer, seq_len, stride)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"Train samples: {len(train_dataset):,} ({len(train_loader)} batches/epoch)")

    val_loader = None
    if val_text:
        val_dataset = TextDataset(val_text, tokenizer, seq_len, stride)
        if len(val_dataset):
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            print(f"Val samples:   {len(val_dataset):,}")
        else:
            print("Val split too short for one window -- skipping validation")

    # Create model
    model = create_gpt_small(tokenizer.vocab_size, seq_len)
    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Decay matrices, not gains and biases.
    optimizer = configure_optimizer(model, lr=lr, weight_decay=weight_decay)

    # Warmup then cosine decay. Starting at the peak learning rate is what the
    # old CosineAnnealingLR did, and it is the most common cause of a run that
    # diverges in its first few dozen steps.
    total_steps = max(1, epochs * len(train_loader))
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_for(warmup_steps, total_steps)
    )
    print(f"Schedule: {warmup_steps} warmup + {total_steps - warmup_steps} cosine steps")

    # Every character of the corpus is in the vocabulary by construction, so a
    # prompt taken from it never triggers the unknown-character warning.
    prompt = sample_prompt if sample_prompt is not None else train_text[:8]

    # Training loop
    print("\nStarting training...")
    model.train()
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)

            # Forward pass
            logits, loss = model(x, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / max(1, num_batches)
        history['train_loss'].append(avg_loss)

        # Held-out loss is the only signal that separates learning from
        # memorizing. Perplexity = exp(loss) makes the number readable.
        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            history['val_loss'].append(val_loss)
            print(
                f"Epoch {epoch+1}/{epochs} - "
                f"train {avg_loss:.4f} (ppl {math.exp(avg_loss):.1f}) | "
                f"val {val_loss:.4f} (ppl {math.exp(val_loss):.1f})"
            )
        else:
            print(f"Epoch {epoch+1}/{epochs} - train {avg_loss:.4f}")

        # Generate sample. generate() restores training mode, so dropout stays
        # on for the epochs that follow.
        if (epoch + 1) % print_every == 0:
            print("\nSample generation:")
            sample_text = generate_text(model, tokenizer, prompt, max_tokens=100)
            print(f"'{sample_text}'")
            print()

    # Save model
    if save_path:
        torch.save(checkpoint_payload(model, tokenizer, seq_len), save_path)
        tokenizer.save(tokenizer_path_for(save_path))
        print(f"Model saved to: {save_path}")

    return model, tokenizer, history


def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=0.8, 
                  top_k=40, device=None):
    """
    Generate text from a prompt.
    
    Args:
        model: Trained GPT model
        tokenizer: Tokenizer
        prompt: Starting text
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        device: Device to run on
    
    Returns:
        Generated text string
    """
    if device is None:
        device = next(model.parameters()).device

    # Encode prompt. An empty result would reach the model as a (1, 0) tensor
    # and fail deep inside the embedding lookup, so catch it here where the
    # cause is still obvious.
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        raise ValueError(
            f"prompt {prompt!r} encoded to zero tokens -- none of its "
            f"characters are in the tokenizer's vocabulary"
        )

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    # generate() handles eval mode and restores it, so the caller's training
    # state survives a mid-training sample.
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k
    )

    # Decode
    return tokenizer.decode(output_ids[0].tolist())


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        text_path = sys.argv[1]
    else:
        text_path = "data/sample_text.txt"
    
    # Check if file exists
    if not os.path.exists(text_path):
        print(f"Error: File not found: {text_path}")
        print("Please provide a text file path as argument.")
        sys.exit(1)
    
    # Train model
    model, tokenizer, history = train_gpt(
        text_path,
        epochs=50,
        batch_size=32,
        seq_len=128,
        lr=3e-4,
        save_path="gpt_model.pt"
    )
    
    # Generate text
    print("\n" + "=" * 50)
    print("Text Generation:")
    print("=" * 50)
    
    prompts = ["The ", "To be or ", "ROMEO\n"]
    for prompt in prompts:
        print(f"\nPrompt: {repr(prompt)}")
        generated = generate_text(model, tokenizer, prompt, max_tokens=200)
        print(f"Generated: {generated}")

