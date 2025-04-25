"""
Utility functions for data handling and processing.

This module contains functions for encoding/decoding text,
creating training batches, and other helper functions.
"""

import os
import torch
from typing import Tuple, List, Dict, Optional

def load_text(filepath: str) -> str:
    """
    Load text from a file.
    
    Args:
        filepath: Path to the text file
        
    Returns:
        The content of the file as a string
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def build_vocabulary(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build character-level vocabulary from text.
    
    Args:
        text: Input text to build vocabulary from
        
    Returns:
        A tuple of (char_to_idx, idx_to_char) dictionaries
    """
    vocabulary = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(vocabulary)}
    idx_to_char = {i: ch for i, ch in enumerate(vocabulary)}
    return char_to_idx, idx_to_char

def encode(text: str, char_to_idx: Dict[str, int]) -> torch.Tensor:
    """
    Encode text as a tensor of indices.
    
    Args:
        text: Input text to encode
        char_to_idx: Character to index mapping
        
    Returns:
        Tensor of encoded indices
    """
    return torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

def decode(encoded_text: torch.Tensor, idx_to_char: Dict[int, str]) -> str:
    """
    Decode a tensor of indices back to text.
    
    Args:
        encoded_text: Tensor of encoded indices
        idx_to_char: Index to character mapping
        
    Returns:
        Decoded text
    """
    if isinstance(encoded_text, torch.Tensor):
        encoded_text = encoded_text.cpu().numpy()
    return ''.join([idx_to_char[int(i)] for i in encoded_text])

def get_train_val_split(encoded_text: torch.Tensor, train_split: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split encoded text into training and validation sets.
    
    Args:
        encoded_text: Full encoded text tensor
        train_split: Ratio of data to use for training (0-1)
        
    Returns:
        Training and validation tensors
    """
    n = len(encoded_text)
    train_size = int(train_split * n)
    train_data = encoded_text[:train_size]
    val_data = encoded_text[train_size:]
    return train_data, val_data

def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a batch of data for training or evaluation.
    
    Args:
        data: Source data tensor
        batch_size: Number of sequences in batch
        block_size: Length of each sequence
        device: Device to place tensors on
        
    Returns:
        Tuple of input (x) and target (y) tensors
    """
    # Generate random indices, ensuring we don't go out of bounds
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    
    # Extract sequences and their corresponding targets (shifted by 1)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    
    return x.to(device), y.to(device)

def save_model_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                          iteration: int, save_dir: str):
    """
    Save model and optimizer state to a checkpoint file.
    
    Args:
        model: The PyTorch model to save
        optimizer: The optimizer being used
        iteration: Current training iteration
        save_dir: Directory to save the checkpoint
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    
    checkpoint_path = os.path.join(save_dir, f'checkpoint_iter_{iteration}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

def load_model_checkpoint(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], 
                         checkpoint_path: str) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], int]:
    """
    Load model and optimizer state from a checkpoint file.
    
    Args:
        model: The PyTorch model to load weights into
        optimizer: The optimizer to load state into (can be None)
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Tuple of (model, optimizer, iteration)
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    iteration = checkpoint['iteration']
    print(f"Loaded checkpoint from iteration {iteration}")
    
    return model, optimizer, iteration