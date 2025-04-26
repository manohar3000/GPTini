"""
Training script for the language model.

This module contains functionality for training the transformer model
on a text dataset and evaluating its performance.
"""

import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from config import *
from model import Transformer
from utils import (
    load_text, build_vocabulary, encode, 
    get_train_val_split, get_batch, save_model_checkpoint
)

def evaluate_model(model: torch.nn.Module, val_data: torch.Tensor, 
                  block_size: int, batch_size: int, device: torch.device) -> float:
    """
    Evaluate the model on validation data.
    
    Args:
        model: The transformer model
        val_data: Validation data tensor
        block_size: Context length
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        # Use multiple batches for more accurate evaluation
        for _ in range(10):  # Use 10 random batches
            x, y = get_batch(val_data, batch_size, block_size, device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            num_batches += 1
            
    return total_loss / num_batches

def train():
    """Train the language model using settings from config.py."""
    # Load and process the text data
    text = load_text(INPUT_FILE)
    char_to_idx, idx_to_char = build_vocabulary(text)
    vocab_size = len(char_to_idx)
    
    # Update global vocab size in config
    global VOCAB_SIZE
    VOCAB_SIZE = vocab_size
    
    # Encode the text and split into training and validation sets
    encoded_text = encode(text, char_to_idx)
    train_data, val_data = get_train_val_split(encoded_text, TRAIN_SPLIT)
    
    # Initialize the model
    model = Transformer(
        vocab_size=vocab_size,
        n_embed=N_EMBED,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        block_size=BLOCK_SIZE
    ).to(DEVICE)
    
    # Print model information
    print(f"Vocabulary size: {vocab_size}")
    print(f"Model parameters: {model.get_parameter_count():,}")
    print(f"Using device: {DEVICE}")
    
    # Initialize optimizer (outside the training loop for efficiency)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Resume from checkpoint if specified
    start_iteration = 0
    if CHECKPOINT_TO_LOAD is not None:
        from utils import load_model_checkpoint
        model, optimizer, start_iteration = load_model_checkpoint(model, optimizer, CHECKPOINT_TO_LOAD)
    
    # Training loop
    best_val_loss = float('inf')

    train_losses = []
    iterations = []
    val_losses = []

    # Early stopping parameters
    no_improve_count = 0
    early_stop_patience = 500  # Number of iterations to wait for improvement

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=50)
    
    for iteration in range(start_iteration, MAX_ITERATIONS):
        # Training step
        model.train()
        
        # Get a batch of data
        x, y = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, DEVICE)
        
        # Forward pass
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        
        # Track training loss
        iterations.append(iteration + 1)
        train_losses.append(loss.item())  
            
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Evaluation
        if (iteration + 1) % EVAL_INTERVAL == 0:
            val_loss = evaluate_model(model, val_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
            # Save validation loss
            val_losses.append(val_loss)

            # Update learning rate based on validation loss
            scheduler.step(val_loss)

            # For monitoring, print current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr}")

            # Track validation loss for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Early stopping check
            if no_improve_count >= early_stop_patience:
                print(f"Early stopping triggered after {iteration + 1} iterations")
                break

        # Regular checkpointing
        if (iteration + 1) % SAVE_CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(MODEL_SAVE_PATH, f"checkpoint_iter_{iteration + 1}.pt")
            save_model_checkpoint(model, optimizer, iteration + 1, MODEL_SAVE_PATH)
            print(f"Iteration {iteration + 1}/{MAX_ITERATIONS} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

    
    print("Training complete!")
    
    # Save final model regardless of performance
    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
        
    final_checkpoint_path = os.path.join(MODEL_SAVE_PATH, 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'args': {
            'vocab_size': vocab_size,
            'n_embed': N_EMBED,
            'n_heads': N_HEADS,
            'n_layers': N_LAYERS,
            'block_size': BLOCK_SIZE
        }
    }, final_checkpoint_path)
    print(f"Final model saved to {final_checkpoint_path}")

    # Plot training and validation loss in one graph
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, train_losses, label='Training Loss')
    
    # Calculate iterations for validation points
    val_iterations = [EVAL_INTERVAL * (i + 1) for i in range(len(val_losses))]
    plt.plot(val_iterations, val_losses, label='Validation Loss', color='orange')
    
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    
    # Save the combined plot
    combined_plot_path = os.path.join(MODEL_SAVE_PATH, 'combined_loss.png')
    plt.savefig(combined_plot_path)
    print(f"Combined loss visualization saved to {combined_plot_path}")
    plt.show()

if __name__ == "__main__":
    train()