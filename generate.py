"""
Text generation script for the language model.

This module provides functionality to generate text using a trained model.
"""

import os
import torch
import torch.nn.functional as F

from model import Transformer
from utils import decode
from config import DEVICE, BLOCK_SIZE, DEFAULT_GEN_LENGTH, INPUT_FILE, TEMPERATURE, TOP_K, TOP_P, MODEL_SAVE_PATH
from utils import load_text, build_vocabulary, decode

def encode_prompt(prompt, char_to_idx):
    """
    Encode the prompt text to tensor.
    
    Args:
        prompt: Input text prompt
        char_to_idx: Character to index mapping
        
    Returns:
        Encoded tensor
    """
    # Handle characters not in the vocabulary
    encoded = []
    for char in prompt:
        if char in char_to_idx:
            encoded.append(char_to_idx[char])
        else:
            print(f"Warning: Character '{char}' not in vocabulary, skipping")
    
    if not encoded:
        return torch.tensor([[]], dtype=torch.long)
    
    return torch.tensor([encoded], dtype=torch.long)

def generate_text(model, prompt, gen_length, char_to_idx, idx_to_char, 
                 temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P):
    """
    Generate text using the trained model.
    
    Args:
        model: Trained transformer model
        prompt: Starting text prompt
        gen_length: Number of characters to generate
        char_to_idx: Character to index mapping
        idx_to_char: Index to character mapping
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        
    Returns:
        Generated text including the prompt
    """
    model.eval()
    
    # Encode the prompt
    encoded_prompt = encode_prompt(prompt, char_to_idx)
    if encoded_prompt.numel() == 0:
        encoded_prompt = torch.zeros(1, 1, dtype=torch.long)
        prompt = idx_to_char[0]
    
    encoded_prompt = encoded_prompt.to(DEVICE)
    
    # Generated so far
    generated = encoded_prompt
    
    # Generate one character at a time
    with torch.no_grad():
        for _ in range(gen_length):
            # Take the last block_size tokens at most
            input_sequence = generated[:, -BLOCK_SIZE:]
            
            # Get logits from the model
            logits = model(input_sequence)
            
            # Focus on the last token's prediction
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k sampling if enabled
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                # Create a mask for the top-k logits
                mask = torch.zeros_like(logits).scatter_(1, top_k_indices, 1.0)
                logits = torch.where(mask > 0, logits, torch.tensor(float('-inf')).to(DEVICE))
            
            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            # Get probabilities and sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat((generated, next_token), dim=1)
    
    # Decode the generated text
    output_text = decode(generated[0], idx_to_char)
    return output_text

def generate(model_path, prompt="", length=DEFAULT_GEN_LENGTH):
    """Generate text using a trained model."""
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Check if it's a training checkpoint or final model
    if 'args' in checkpoint:
        # Final model format
        model_args = checkpoint['args']
        char_to_idx = checkpoint['char_to_idx']
        idx_to_char = checkpoint['idx_to_char']
    else:
        # Training checkpoint format
        # We need to load these from config since they're not in the checkpoint
        from config import VOCAB_SIZE, N_EMBED, N_HEADS, N_LAYERS, BLOCK_SIZE
        
        # Load the text to rebuild vocabulary
        text = load_text(INPUT_FILE)
        char_to_idx, idx_to_char = build_vocabulary(text)
        
        model_args = {
            'vocab_size': len(char_to_idx),
            'n_embed': N_EMBED,
            'n_heads': N_HEADS,
            'n_layers': N_LAYERS,
            'block_size': BLOCK_SIZE
        }
    
    # Create and load the model
    model = Transformer(
        vocab_size=model_args['vocab_size'],
        n_embed=model_args['n_embed'],
        n_heads=model_args['n_heads'],
        n_layers=model_args['n_layers'],
        block_size=model_args['block_size']
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    # Generate text
    generated_text = generate_text(
        model=model,
        prompt=prompt,
        gen_length=length,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char
    )
    
    print("\nGenerated Text:")
    print("-" * 40)
    print(generated_text)
    print("-" * 40)
    
    return generated_text

if __name__ == "__main__":
    # Configure these variables before running
    MODEL_PATH = os.path.join(MODEL_SAVE_PATH, "final_model.pt")
    PROMPT = "Once upon a time"
    LENGTH = 500
    
    generate(MODEL_PATH, PROMPT, LENGTH)