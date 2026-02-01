"""
Training utilities for GPT model.

Provides:
- CharTokenizer: Simple character-level tokenizer
- TextDataset: Dataset for text data
- train_gpt: Main training function
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
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
        """Convert text to list of integers."""
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


class TextDataset(Dataset):
    """
    Dataset for autoregressive language modeling.
    
    Each sample is a sequence of tokens, where the target is the same
    sequence shifted by one position.
    """
    
    def __init__(self, text, tokenizer, seq_len):
        """
        Args:
            text: Raw text data
            tokenizer: Tokenizer to encode text
            seq_len: Sequence length for training
        """
        self.seq_len = seq_len
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    def __len__(self):
        # Number of complete sequences we can extract
        return max(0, len(self.data) - self.seq_len)
    
    def __getitem__(self, idx):
        """
        Returns:
            x: Input sequence (seq_len,)
            y: Target sequence (seq_len,) - shifted by 1
        """
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y


def train_gpt(text_path, epochs=100, batch_size=32, seq_len=128, lr=3e-4,
              device=None, save_path=None, print_every=10):
    """
    Train a GPT model on text data.
    
    Args:
        text_path: Path to text file
        epochs: Number of training epochs
        batch_size: Batch size
        seq_len: Sequence length
        lr: Learning rate
        device: Device to train on (auto-detected if None)
        save_path: Path to save trained model
        print_every: Print loss every N batches
    
    Returns:
        model: Trained GPT model
        tokenizer: Fitted tokenizer
    """
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")
    
    # Load and prepare data
    print(f"Loading data from: {text_path}")
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Text length: {len(text):,} characters")
    
    # Create tokenizer
    tokenizer = CharTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create dataset and dataloader
    dataset = TextDataset(text, tokenizer, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset size: {len(dataset):,} samples")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Create model
    model = create_gpt_small(tokenizer.vocab_size, seq_len)
    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Learning rate scheduler (optional warmup)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(dataloader)
    )
    
    # Training loop
    print("\nStarting training...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (x, y) in enumerate(progress_bar):
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
        
        # Print epoch summary
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")
        
        # Generate sample
        if (epoch + 1) % print_every == 0:
            print("\nSample generation:")
            sample_text = generate_text(model, tokenizer, "The ", max_tokens=100)
            print(f"'{sample_text}'")
            print()
    
    # Save model
    if save_path:
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab_size': tokenizer.vocab_size,
            'seq_len': seq_len,
        }, save_path)
        tokenizer.save(save_path.replace('.pt', '_tokenizer.json'))
        print(f"Model saved to: {save_path}")
    
    return model, tokenizer


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
    
    model.eval()
    
    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    
    # Generate
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
    model, tokenizer = train_gpt(
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

