"""
Transformer language model implementation.

This module contains the implementation of a Transformer-based
language model with self-attention mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention module.
    
    This module splits the embedding dimension into multiple heads,
    computes self-attention independently for each head, and then
    recombines the results.
    """
    
    def __init__(self, n_embed: int, n_heads: int):
        """
        Initialize the multi-head attention module.
        
        Args:
            n_embed: Embedding dimension
            n_heads: Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        
        self.n_heads = n_heads
        self.head_size = n_embed // n_heads
        
        # Linear projections for query, key, and value
        self.W_query = nn.Linear(n_embed, n_embed)
        self.W_key = nn.Linear(n_embed, n_embed)
        self.W_value = nn.Linear(n_embed, n_embed)
        
        # Output projection (optional, but common in practice)
        self.W_out = nn.Linear(n_embed, n_embed)
        
        # Initialize a single causal mask to be reused
        self.register_buffer("mask", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-head self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, block_size, n_embed)
            
        Returns:
            Output tensor of shape (batch_size, block_size, n_embed)
        """
        batch_size, block_size, _ = x.size()
        
        # Create or reuse causal mask (lower triangular matrix)
        if self.mask is None or self.mask.size(0) != block_size:
            self.mask = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size).to(x.device)
        
        # Linear projections and reshape to (batch_size, n_heads, block_size, head_size)
        q = self.W_query(x).view(batch_size, block_size, self.n_heads, self.head_size).transpose(1, 2)
        k = self.W_key(x).view(batch_size, block_size, self.n_heads, self.head_size).transpose(1, 2)
        v = self.W_value(x).view(batch_size, block_size, self.n_heads, self.head_size).transpose(1, 2)

        # Compute attention scores
        # (batch_size, n_heads, block_size, head_size) @ (batch_size, n_heads, head_size, block_size)
        # -> (batch_size, n_heads, block_size, block_size)
        attn_weights = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        
        # Apply causal mask to prevent attending to future tokens
        masked_attn_weights = attn_weights.masked_fill(self.mask[:, :, :block_size, :block_size] == 0, float('-inf'))
        
        # Apply softmax to get attention probabilities
        attn_probs = F.softmax(masked_attn_weights, dim=-1)
        
        # Weight values by attention probabilities
        # (batch_size, n_heads, block_size, block_size) @ (batch_size, n_heads, block_size, head_size)
        # -> (batch_size, n_heads, block_size, head_size)
        attn_output = attn_probs @ v
        
        # Reshape and combine heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, block_size, self.n_heads * self.head_size)
        
        # Apply output projection
        return self.W_out(attn_output)

class Block(nn.Module):
    def __init__(self, n_embed: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize a transformer block with dropout.
        
        Args:
            n_embed: Embedding dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(Block, self).__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(n_embed, n_heads)
        
        # Layer normalizations
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        
        # Position-wise feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed)
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through a transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, block_size, n_embed)
            
        Returns:
            Output tensor of shape (batch_size, block_size, n_embed)
        """
        # Self-attention with residual connection and dropout
        x = x + self.dropout(self.attention(self.ln1(x)))
        
        # Feed-forward network with residual connection and dropout
        x = x + self.dropout(self.ffn(self.ln2(x)))
        
        return x

class Transformer(nn.Module):
    """
    Transformer-based language model.
    
    A decoder-only transformer that predicts the next token in a sequence.
    """
    
    def __init__(self, vocab_size: int, n_embed: int, n_heads: int, n_layers: int, block_size: int, dropout: float = 0.1):
        """
        Initialize the transformer model with dropout.
        
        Args:
            vocab_size: Size of the vocabulary
            n_embed: Embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer blocks
            block_size: Maximum sequence length
            dropout: Dropout probability
        """
        super(Transformer, self).__init__()
        
        # Token and position embeddings
        self.w_embed = nn.Embedding(vocab_size, n_embed)
        self.w_pos = nn.Embedding(block_size, n_embed)
        
        # Stack of transformer blocks with dropout passed in
        self.blocks = nn.ModuleList([Block(n_embed, n_heads, dropout=dropout) for _ in range(n_layers)])
        
        # Final layer norm and output projection
        self.ln_final = nn.LayerNorm(n_embed)
        self.fc_out = nn.Linear(n_embed, vocab_size)
        
        # Save block size and dropout
        self.block_size = block_size
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize weights for better training dynamics.
        
        Args:
            module: PyTorch module to initialize
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through the transformer model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            
        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.size()
        
        # Check sequence length doesn't exceed block size
        if seq_len > self.block_size:
            raise ValueError(f"Input sequence length {seq_len} exceeds model's block size {self.block_size}")
            
        # Get positions for this sequence
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        
        # Compute token and position embeddings and add them
        token_embeddings = self.w_embed(x)
        position_embeddings = self.w_pos(positions)
        x = token_embeddings + position_embeddings
        
        # Apply dropout after embeddings, if desired
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Apply final layer norm
        x = self.ln_final(x)
        
        # Project to vocabulary size
        logits = self.fc_out(x)
        
        return logits

    def get_parameter_count(self) -> int:
        """
        Count the total number of parameters in the model.
        
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters())