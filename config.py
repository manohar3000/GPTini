"""
Configuration settings for the language model.

This module contains all hyperparameters and settings used throughout
the training and inference processes.
"""

import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File paths
INPUT_FILE = "input.txt"
MODEL_SAVE_PATH = "model_checkpoints"

# Model architecture - Optimized for RTX 4050 GPU
VOCAB_SIZE = None  # Will be set after processing the input file
N_EMBED = 256      # Embedding dimension (size of the hidden states)
N_HEADS = 4        # Number of attention heads
N_LAYERS = 4       # Number of transformer blocks
HEAD_SIZE = N_EMBED // N_HEADS
BLOCK_SIZE = 512   # max sequence length

# Training parameters
BATCH_SIZE = 16    # Batch size for training
LEARNING_RATE = 0.001  # Learning rate for the optimizer
MAX_ITERATIONS = 10000
EVAL_INTERVAL = 1
TRAIN_SPLIT = 0.8  # 90% training, 10% validation

# Generation parameters
DEFAULT_GEN_LENGTH = 100
TEMPERATURE = 1.0
TOP_K = 0
TOP_P = 0.9

# Checkpointing
SAVE_CHECKPOINT_INTERVAL = 1000
CHECKPOINT_TO_LOAD = None  # Set to path of checkpoint if resuming training